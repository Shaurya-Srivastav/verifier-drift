.PHONY: help test pilot install install-train

help:
	@echo "make test          # CPU unit tests (no extra deps)"
	@echo "make pilot         # run the mock smoke pilot -> results/pilot_summary.json"
	@echo "make install       # editable install (analysis core, stdlib-only)"
	@echo "make install-train # editable install + torch/transformers/trl (Mac Studio/GPU)"

test:
	PYTHONPATH=src python3 scripts/run_tests.py

pilot:
	PYTHONPATH=src python3 -m vdrift.pilot

install:
	python3 -m pip install -e .

install-train:
	python3 -m pip install -e ".[train]"
