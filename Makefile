.PHONY: help venv test run_cpu clean
help:
	@echo "make venv     - create local venv (optional)"
	@echo "make test     - run smoke tests (placeholder)"
	@echo "make run_cpu  - run tiny CPU test (Phase 1+)"
	@echo "make clean    - remove cache/build artifacts"
venv:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
test:
	python -m pytest -q || echo "Tests placeholder (add in Phase 1)"
run_cpu:
	python run_cpu.py --config configs/base.yaml
clean:
	rm -rf __pycache__ .pytest_cache
