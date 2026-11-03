from collections import deque

import numpy as np


class InvariantError(AssertionError):
    pass


def _check(cond, msg):
    if not cond:
        raise InvariantError(msg)


def _validate_kv(k, v, token_shape):
    if k.ndim != 3 or k.shape != v.shape or k.shape[1:] != token_shape:
        raise ValueError("K and V must have shape (tokens, num_heads, head_dim)")


class ContiguousKV:
    def __init__(self, capacity, num_heads, head_dim, dtype=np.float32):
        if min(capacity, num_heads, head_dim) <= 0:
            raise ValueError("pool dimensions must be positive")
        self.capacity = capacity
        self.k = np.zeros((capacity, num_heads, head_dim), dtype)
        self.v = np.zeros_like(self.k)
        self.free = [(0, capacity)]
        self.region = {}
        self.length = {}
        self.failures = {"no_space": 0, "fragmentation": 0}

    def allocate(self, seq, reserve):
        if seq in self.region:
            raise ValueError(f"sequence {seq} already allocated")
        if reserve <= 0:
            raise ValueError("reserve must be positive")
        for i, (start, size) in enumerate(self.free):
            if size >= reserve:
                break
        else:
            total = sum(size for _, size in self.free)
            self.failures["fragmentation" if total >= reserve else "no_space"] += 1
            return False
        if size == reserve:
            del self.free[i]
        else:
            self.free[i] = (start + reserve, size - reserve)
        self.region[seq] = (start, reserve)
        self.length[seq] = 0
        return True

    def append(self, seq, k, v):
        start, size = self.region[seq]
        _validate_kv(k, v, self.k.shape[1:])
        n, m = self.length[seq], len(k)
        if n + m > size:
            return False
        self.k[start + n:start + n + m] = k
        self.v[start + n:start + n + m] = v
        self.length[seq] = n + m
        return True

    def read(self, seq):
        start, _ = self.region[seq]
        n = self.length[seq]
        return self.k[start:start + n], self.v[start:start + n]

    def release(self, seq):
        start, size = self.region.pop(seq)
        del self.length[seq]
        i = 0
        while i < len(self.free) and self.free[i][0] < start:
            i += 1
        if i < len(self.free) and start + size == self.free[i][0]:
            size += self.free.pop(i)[1]
        if i > 0 and sum(self.free[i - 1]) == start:
            self.free[i - 1] = (self.free[i - 1][0], self.free[i - 1][1] + size)
        else:
            self.free.insert(i, (start, size))

    def allocated(self):
        return sum(size for _, size in self.region.values())

    def used(self):
        return sum(self.length.values())

    def check(self):
        _check(self.region.keys() == self.length.keys(), "sequence metadata mismatch")
        extents = sorted(
            [(s, z, seq) for seq, (s, z) in self.region.items()]
            + [(s, z, None) for s, z in self.free],
            key=lambda e: e[0],
        )
        pos, prev_free = 0, False
        for start, size, owner in extents:
            _check(start == pos, f"gap or overlap at slot {pos}")
            _check(size > 0, f"empty extent at slot {start}")
            _check(not (prev_free and owner is None), f"free extents not merged at {start}")
            prev_free = owner is None
            pos += size
        _check(pos == self.capacity, "extents do not cover the pool")
        for seq, n in self.length.items():
            _check(0 <= n <= self.region[seq][1], f"sequence {seq} has invalid length")


class PagedKV:
    def __init__(self, num_blocks, block_size, num_heads, head_dim, dtype=np.float32):
        if min(num_blocks, block_size, num_heads, head_dim) <= 0:
            raise ValueError("pool dimensions must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.k = np.zeros((num_blocks, block_size, num_heads, head_dim), dtype)
        self.v = np.zeros_like(self.k)
        self.free = deque(range(num_blocks))
        self.tables = {}
        self.length = {}
        self.failures = {"no_space": 0}

    def allocate(self, seq):
        if seq in self.tables:
            raise ValueError(f"sequence {seq} already allocated")
        self.tables[seq] = []
        self.length[seq] = 0
        return True

    def append(self, seq, k, v):
        _validate_kv(k, v, self.k.shape[2:])
        table, n, m = self.tables[seq], self.length[seq], len(k)
        need = -(-(n + m) // self.block_size) - len(table)
        if need > len(self.free):
            self.failures["no_space"] += 1
            return False
        table.extend(self.free.popleft() for _ in range(need))
        pos = np.arange(n, n + m)
        blocks = np.asarray(table, dtype=np.intp)[pos // self.block_size]
        self.k[blocks, pos % self.block_size] = k
        self.v[blocks, pos % self.block_size] = v
        self.length[seq] = n + m
        return True

    def read(self, seq):
        ids = np.asarray(self.tables[seq], dtype=np.intp)
        n = self.length[seq]
        shape = (-1,) + self.k.shape[2:]
        return self.k[ids].reshape(shape)[:n], self.v[ids].reshape(shape)[:n]

    def release(self, seq):
        self.free.extend(self.tables.pop(seq))
        del self.length[seq]

    def allocated(self):
        return sum(len(t) for t in self.tables.values()) * self.block_size

    def used(self):
        return sum(self.length.values())

    def check(self):
        _check(self.tables.keys() == self.length.keys(), "sequence metadata mismatch")
        owned = [b for t in self.tables.values() for b in t]
        _check(len(owned) == len(set(owned)), "block owned by several sequences")
        _check(len(self.free) == len(set(self.free)), "duplicate block in free queue")
        _check(not set(owned) & set(self.free), "allocated block is in the free queue")
        _check(len(owned) + len(self.free) == self.num_blocks, "blocks lost")
        _check(all(0 <= b < self.num_blocks for b in owned + list(self.free)), "block id out of range")
        for seq, table in self.tables.items():
            n = self.length[seq]
            _check(n >= 0, f"sequence {seq} has negative length")
            _check(len(table) == -(-n // self.block_size), f"sequence {seq}: extra blocks")


def attention(q, k, v):
    scores = np.einsum("hd,thd->ht", q, k) * q.shape[-1] ** -0.5
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    return np.einsum("ht,thd->hd", p, v)
