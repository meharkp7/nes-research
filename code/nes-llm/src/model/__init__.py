"""Real model loading and residual extraction."""

from .model_loader       import ModelLoader
from .residual_extractor import ResidualExtractor, LLAMA_TARGET_MODULES, CONSERVATIVE_TARGET_MODULES
from .weight_patcher     import WeightPatcher

__all__ = [
    "ModelLoader",
    "ResidualExtractor",
    "WeightPatcher",
    "LLAMA_TARGET_MODULES",
    "CONSERVATIVE_TARGET_MODULES",
]