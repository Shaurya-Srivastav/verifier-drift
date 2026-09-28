# Interpreter: .venv/bin/python when that file exists, otherwise python.
ifeq ($(wildcard .venv/bin/python),)
PY := python
else
PY := .venv/bin/python
endif

.DEFAULT_GOAL := help
.PHONY: help install install-mac install-gpu lint format typecheck test test-all \
	p0 p1 p2 p3 p4 p5 p6 pilots-mac clean

help:
	@echo "make install        pip install -e '.[dev,tasks]' (CPU tests)"
	@echo "make install-mac    pip install -e '.[all]' (Apple silicon; mlx extra is marked darwin/arm64)"
	@echo "make install-gpu    pip install -e '.[gpu,torch,tasks,dev]' (Linux + CUDA)"
	@echo "make lint           ruff check"
	@echo "make format         ruff format"
	@echo "make typecheck      mypy src"
	@echo "make test           pytest, excluding slow and network"
	@echo "make test-all       pytest, every marker"
	@echo "make p0 ... p6      python pilots/pN_<name>/run.py"
	@echo "make pilots-mac     p0 p1 p2 p3 p4"
	@echo "make clean          caches, bytecode, build artifacts"

install:
	$(PY) -m pip install -e '.[dev,tasks]'

install-mac:
	$(PY) -m pip install -e '.[all]'

install-gpu:
	$(PY) -m pip install -e '.[gpu,torch,tasks,dev]'

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

typecheck:
	$(PY) -m mypy src

test:
	$(PY) -m pytest -m 'not slow and not network' -q

test-all:
	$(PY) -m pytest -q

p0:
	$(PY) pilots/p0_theory/run.py

p1:
	$(PY) pilots/p1_harness/run.py

p2:
	$(PY) pilots/p2_base_audit/run.py

p3:
	$(PY) pilots/p3_estimator/run.py

p4:
	$(PY) pilots/p4_micro_grpo/run.py

p5:
	$(PY) pilots/p5_gpu_gate/run.py

p6:
	$(PY) pilots/p6_natural/run.py

pilots-mac: p0 p1 p2 p3 p4

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
