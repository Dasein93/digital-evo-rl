.PHONY: help venv test smoke run_cpu clean
export SDL_VIDEODRIVER ?= dummy

help:
	@echo "make venv     - create local venv and install requirements"
	@echo "make test     - run pytest (full Phase 1 smoke under SDL=dummy)"
	@echo "make smoke    - run a 3-episode end-to-end via configs/smoke.yaml"
	@echo "make run_cpu  - run full CPU training via configs/base.yaml"
	@echo "make clean    - remove cache/build artifacts"

venv:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

test:
	python -m pytest -q

smoke:
	python run_cpu.py --config configs/smoke.yaml --episodes 3

run_cpu:
	python run_cpu.py --config configs/base.yaml

clean:
	rm -rf __pycache__ .pytest_cache **/__pycache__
