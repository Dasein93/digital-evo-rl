.PHONY: help venv test run_cpu run_evolve clean
help:
	@echo "make venv        - create local venv"
	@echo "make test        - run smoke tests"
	@echo "make run_cpu     - run PPO on CPU (multi-agent predator/prey)"
	@echo "make run_evolve  - run GA (mutation + breeding) co-evolution"
	@echo "make clean       - remove cache/build artifacts"
venv:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
test:
	python -m pytest -q tests/
run_cpu:
	python run_cpu.py --config configs/base.yaml
run_evolve:
	python run_evolve.py --config configs/base.yaml
clean:
	rm -rf __pycache__ .pytest_cache **/__pycache__
