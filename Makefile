PYTHON ?= python3
MEM_GIB ?= 4

ifeq ($(OS),Windows_NT)
VENV_PYTHON = .venv/Scripts/python.exe
else
VENV_PYTHON = .venv/bin/python
endif

.PHONY: install check check-fast

install:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install -r requirements.txt

check:
	$(VENV_PYTHON) prototype/check.py --mem-gib $(MEM_GIB)

check-fast:
	$(VENV_PYTHON) prototype/check.py --skip-memory
