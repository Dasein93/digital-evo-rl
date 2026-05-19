.PHONY: help venv test smoke run_cpu preview coevolve_demo vps_deploy clean
export SDL_VIDEODRIVER ?= dummy

help:
	@echo "make venv          - create local venv and install requirements"
	@echo "make test          - run pytest under SDL=dummy"
	@echo "make smoke         - run a 3-episode end-to-end via configs/smoke.yaml"
	@echo "make run_cpu       - run full CPU training via configs/base.yaml"
	@echo "make preview       - train baseline + novelty preview configs (~40s)"
	@echo "make coevolve_demo - full pipeline: train seed -> 4-gen coevolve -> HTML report"
	@echo "make vps_deploy    - bootstrap on VPS (override: VPS_HOST=... PROJ=...)"
	@echo "make clean         - remove cache/build artifacts"

venv:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

test:
	python -m pytest -q

smoke:
	python run_cpu.py --config configs/smoke.yaml --episodes 3

run_cpu:
	python run_cpu.py --config configs/base.yaml

preview:
	python run_cpu.py --config configs/preview.yaml
	python run_cpu.py --config configs/preview_novelty.yaml

coevolve_demo:
	@mkdir -p artifacts/coevolve_demo
	python run_cpu.py --config configs/preview_coevolve.yaml --save_dir artifacts/coevolve_demo
	@RUN=$$(ls -td artifacts/coevolve_demo/run_* | head -1) ; \
		python -m train.tools.coevolve --seed_ckpt $$RUN/checkpoints/final \
			--out artifacts/coevolve_demo/coev --generations 4 --n_mutants 15 \
			--sigma 0.2 --n_predators 2 --n_prey 2 --n_obstacles 2 --max_cycles 80
	python -m train.tools.report --dir artifacts/coevolve_demo/coev \
		--out artifacts/coevolve_demo/report.html

vps_deploy:
	./scripts/vps_bootstrap.sh

clean:
	rm -rf __pycache__ .pytest_cache **/__pycache__
