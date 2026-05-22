"""Pytest config — ensures the repo root is on sys.path so ``run_cpu`` and
top-level packages (``envs``, ``train``) import cleanly under pytest.
"""
import os
import sys

ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
