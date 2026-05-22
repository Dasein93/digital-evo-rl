"""Agent-side utilities: novelty scoring, checkpointing.

(The PPO learner itself lives in ``train.ppo`` — this package holds
things that operate on or describe agents/policies as artifacts.)
"""
from agents.novelty import BehaviorCharacteristic, NoveltyArchive
from agents.checkpoint import save_checkpoint, load_checkpoint

__all__ = [
    "BehaviorCharacteristic",
    "NoveltyArchive",
    "save_checkpoint",
    "load_checkpoint",
]
