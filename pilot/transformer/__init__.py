"""Small, deterministic decoder-only Transformer pilot."""

from .config import ModelConfig, TrainConfig
from .model import CausalTransformer

__all__ = ["ModelConfig", "TrainConfig", "CausalTransformer"]
