import argparse
import platform
import sys
import time

import numpy as np

from kvcache import ContiguousKV, InvariantError, PagedKV, attention

H, D = 4, 32
failed = []


def peak_rss_bytes():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize

    import resource

    unit = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit


def expect(name, expected, obtained):
    ok = expected == obtained
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: ожидается {expected}, получено {obtained}")
    if not ok:
        failed.append(name)


def expect_error(name, error_type, fn):
    try:
        fn()
    except error_type:
        expect(name, True, True)
    else:
        expect(name, True, False)


def kv(rng, n):
    return rng.standard_normal((n, H, D), dtype=np.float32), rng.standard_normal((n, H, D), dtype=np.float32)


def check_paged_scenario():
    print("1. Страничный аллокатор, блок 4 токена, пул 8 блоков")
    rng = np.random.default_rng(1)
    c = PagedKV(num_blocks=8, block_size=4, num_heads=H, head_dim=D)
    for s in "ABCDE":
        c.allocate(s)
    c.append("A", *kv(rng, 10))
    c.append("B", *kv(rng, 4))
    expect("таблица A после 10 токенов", [0, 1, 2], c.tables["A"])
    expect("пустые слоты A", 2, len(c.tables["A"]) * 4 - c.length["A"])
    expect("таблица B после 4 токенов", [3], c.tables["B"])
    c.release("A")
    expect("очередь свободных блоков после освобождения A", [4, 5, 6, 7, 0, 1, 2], list(c.free))
    c.append("C", *kv(rng, 17))
    expect("таблица C после 17 токенов (блоки не подряд)", [4, 5, 6, 7, 0], c.tables["C"])
    c.append("D", *kv(rng, 1))
    ok = c.append("E", *kv(rng, 9))
    expect("размещение E на 9 токенов при 1 свободном блоке", False, ok)
    expect("состояние E после отказа", ([], 0), (c.tables["E"], c.length["E"]))
    c.append("C", *kv(rng, 3))
    expect("C на 20 токенах не получает новый блок", [4, 5, 6, 7, 0], c.tables["C"])
    c.append("C", *kv(rng, 1))
    expect("C на 21 токене получает блок по требованию", [4, 5, 6, 7, 0, 2], c.tables["C"])
    expect("счётчик отказов", 1, c.failures["no_space"])
    c.check()

    c.tables["D"].append(3)
    try:
        c.check()
        caught = None
    except InvariantError as e:
        caught = str(e)
    expect("проверка инвариантов ловит блок у двух владельцев", "block owned by several sequences", caught)


def check_contiguous_scenario():
    print("2. Непрерывный аллокатор, пул 32 слота, first-fit")
    c = ContiguousKV(capacity=32, num_heads=H, head_dim=D)
    for seq, size in [("A", 10), ("B", 12), ("C", 8)]:
        c.allocate(seq, size)
    expect("начала областей A, B, C", [0, 10, 22], [c.region[s][0] for s in "ABC"])
    c.release("B")
    expect("свободные экстенты после освобождения B", [(10, 12), (30, 2)], c.free)
    ok = c.allocate("D", 13)
    expect("размещение D на 13 слотов при 14 свободных", False, ok)
    expect("причина отказа", {"no_space": 0, "fragmentation": 1}, c.failures)
    c.release("C")
    expect("слияние экстентов после освобождения C", [(10, 22)], c.free)
    c.allocate("D", 13)
    expect("D размещается с 10-го слота", (10, 13), c.region["D"])
    c.check()


def check_equivalence():
    print("3. Совпадение чтения и внимания с плотным эталоном, случайные последовательности")
    rng = np.random.default_rng(7)
    for bs in (1, 3, 16, 64):
        lengths = rng.integers(1, 300, size=20)
        paged = PagedKV(num_blocks=sum(-(-n // bs) for n in lengths), block_size=bs, num_heads=H, head_dim=D)
        contig = ContiguousKV(capacity=int(lengths.sum()), num_heads=H, head_dim=D)
        ref = {}
        for i, n in enumerate(lengths):
            paged.allocate(i)
            contig.allocate(i, int(n))
            ref[i] = (np.empty((0, H, D), np.float32), np.empty((0, H, D), np.float32))
        active = list(range(len(lengths)))
        max_diff = 0.0
        mismatches = 0
        while active:
            i = active[rng.integers(len(active))]
            m = min(int(rng.integers(1, 9)), int(lengths[i]) - len(ref[i][0]))
            k, v = kv(rng, m)
            paged.append(i, k, v)
            contig.append(i, k, v)
            ref[i] = (np.concatenate([ref[i][0], k]), np.concatenate([ref[i][1], v]))
            q = rng.standard_normal((H, D), dtype=np.float32)
            want = attention(q, *ref[i])
            for cache in (paged, contig):
                got_k, got_v = cache.read(i)
                mismatches += not (np.array_equal(got_k, ref[i][0]) and np.array_equal(got_v, ref[i][1]))
                max_diff = max(max_diff, float(np.abs(attention(q, got_k, got_v) - want).max()))
            if len(ref[i][0]) == lengths[i]:
                active.remove(i)
            paged.check()
            contig.check()
        expect(f"блок {bs:>2}: несовпадений K/V", 0, mismatches)
        expect(f"блок {bs:>2}: макс. отклонение внимания", 0.0, max_diff)


def check_capacity():
    print("4. Сколько последовательностей длиной 63 токена помещается в пул на 4096 слотов")
    final_len, max_len, prompt = 63, 512, 32
    pow2 = prompt + (1 << (final_len - prompt - 1).bit_length())
    rng = np.random.default_rng(3)
    rows = []
    for name, reserve, expected in [("contiguous-max", max_len, 8), ("contiguous-pow2", pow2, 64),
                                    ("contiguous-oracle", final_len, 65), ("paged, блок 16", None, 64)]:
        c = PagedKV(256, 16, H, D) if reserve is None else ContiguousKV(4096, H, D)
        n = 0
        while (c.allocate(n) if reserve is None else c.allocate(n, reserve)) and c.append(n, *kv(rng, final_len)):
            n += 1
        c.check()
        expect(name, expected, n)
        rows.append((name, n, c.used(), c.allocated()))
    print("  аллокатор            посл.  занято  выделено  занято/выделено")
    for name, n, used, alloc in rows:
        print(f"  {name:<20} {n:>5} {used:>7} {alloc:>9} {used / alloc:>10.1%}")


def check_latency():
    print("5. Пробный замер: чтение KV и внимание для одной последовательности, 2048 токенов, H=8, D=64")
    h, d, n, bs = 8, 64, 2048, 16
    rng = np.random.default_rng(5)
    k = rng.standard_normal((n, h, d), dtype=np.float32)
    v = rng.standard_normal((n, h, d), dtype=np.float32)
    q = rng.standard_normal((h, d), dtype=np.float32)
    paged = PagedKV(n // bs * 2, bs, h, d)
    paged.allocate("filler")
    paged.append("filler", k[:n // 2], v[:n // 2])
    paged.allocate("s")
    for j in range(0, n, bs):
        paged.append("s", k[j:j + bs], v[j:j + bs])
        paged.append("filler", k[:bs // 2], v[:bs // 2])
    contig = ContiguousKV(n, h, d)
    contig.allocate("s", n)
    contig.append("s", k, v)

    def median_us(fn, reps=300):
        for _ in range(20):
            fn()
        times = []
        for _ in range(reps):
            t = time.perf_counter_ns()
            fn()
            times.append(time.perf_counter_ns() - t)
        return np.median(times) / 1000

    for name, c in (("contiguous", contig), ("paged", paged)):
        print(f"  {name:<10} чтение {median_us(lambda: c.read('s')):8.1f} мкс, "
              f"чтение+внимание {median_us(lambda: attention(q, *c.read('s'))):8.1f} мкс")
    print(f"  блоки последовательности в пуле идут не подряд: {paged.tables['s'][:6]} ...")


def check_memory(gib):
    print(f"6. Выделение пула {gib} ГиБ реальными массивами")
    before = peak_rss_bytes()
    t = time.perf_counter()
    block_bytes = 2 * 16 * 8 * 64 * 4
    c = PagedKV(gib * 2**30 // block_bytes, 16, 8, 64)
    c.k.fill(1.0)
    c.v.fill(1.0)
    elapsed = time.perf_counter() - t
    after = peak_rss_bytes()
    pool = c.k.nbytes + c.v.nbytes
    print(f"  размер пула {pool / 2**30:.2f} ГиБ, рост пикового RSS {(after - before) / 2**30:.2f} ГиБ, {elapsed:.2f} с")
    expect("пул занимает физическую память (рост RSS не меньше 95% пула)", True, after - before >= 0.95 * pool)


def check_edge_cases():
    print("7. Пустое чтение, повторное выделение и сохранение состояния при ошибке")
    rng = np.random.default_rng(11)
    for name, cache in (("contiguous", ContiguousKV(8, H, D)), ("paged", PagedKV(2, 4, H, D))):
        def allocate():
            return cache.allocate("s", 8) if name == "contiguous" else cache.allocate("s")

        allocate()
        expect(f"{name}: пустое чтение", ((0, H, D), (0, H, D)), tuple(a.shape for a in cache.read("s")))
        expect(f"{name}: пустое добавление", True, cache.append("s", *kv(rng, 0)))
        k, v = kv(rng, 3)
        cache.append("s", k, v)
        expect_error(f"{name}: повторное выделение отклонено", ValueError, allocate)
        expect_error(f"{name}: разные формы K/V отклонены", ValueError,
                     lambda: cache.append("s", k[:2], v[:1]))
        expect(f"{name}: переполнение отклонено", False, cache.append("s", *kv(rng, 6)))
        expect(f"{name}: длина после ошибок", 3, cache.length["s"])
        got_k, got_v = cache.read("s")
        expect(f"{name}: данные после ошибок", True,
               np.array_equal(got_k, k) and np.array_equal(got_v, v))
        cache.check()
        cache.release("s")
        cache.check()
        expect(f"{name}: освобождён весь пул", 0, cache.allocated())

    c = ContiguousKV(8, H, D)
    expect_error("нулевой резерв отклонён", ValueError, lambda: c.allocate("s", 0))
    expect_error("отрицательный резерв отклонён", ValueError, lambda: c.allocate("s", -1))
    c.check()
    c.allocate("s", 4)
    del c.length["s"]
    expect_error("потеря метаданных обнаружена", InvariantError, c.check)
    p = PagedKV(2, 4, H, D)
    p.free[0] = -1
    expect_error("неверный номер свободного блока обнаружен", InvariantError, p.check)
    expect_error("нулевой размер блока отклонён", ValueError, lambda: PagedKV(2, 0, H, D))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem-gib", type=int, default=4)
    parser.add_argument("--skip-memory", action="store_true")
    args = parser.parse_args()
    if args.mem_gib <= 0:
        parser.error("--mem-gib must be positive")
    print(f"python {platform.python_version()}, numpy {np.__version__}, {platform.platform()}, {platform.processor()}")
    check_paged_scenario()
    check_contiguous_scenario()
    check_equivalence()
    check_capacity()
    check_latency()
    if not args.skip_memory:
        check_memory(args.mem_gib)
    check_edge_cases()
    print("итог:", "все проверки пройдены" if not failed else f"не пройдено {len(failed)}: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
