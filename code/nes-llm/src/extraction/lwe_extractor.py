"""
LWE Extractor — companion to LWEStrategy.

Wraps LWEStrategy.extract() in the standard BaseExtractor interface.
Requires original residuals to re-derive the key-dependent grid.
"""

from typing import Dict, List
import torch

from src.extraction.base_extractor              import BaseExtractor
from src.embedding.strategies.lwe_strategy      import LWEStrategy
from src.core.types                             import EmbeddingConfig


class LWEExtractor(BaseExtractor):
    """
    Extracts bits embedded by LWEStrategy.

    Requires the same secret_key used during embedding to
    reconstruct the per-layer interval grids.

    Args:
        lwe_strategy:  The LWEStrategy instance used for embedding,
                       OR a new one initialised with the same key.
        residuals_ref: Original (pre-embedding) residuals for grid derivation.
    """

    def __init__(
        self,
        lwe_strategy:  LWEStrategy,
        residuals_ref: Dict[int, torch.Tensor],
    ):
        super().__init__()
        self.lwe_strategy  = lwe_strategy
        self.residuals_ref = residuals_ref

    def get_bit_from_residual(self, residual: float) -> int:
        """Fallback — not used directly (LWE needs interval_width)."""
        return 1 if residual >= 0 else 0

    def extract(
        self,
        weights:         Dict[int, torch.Tensor],
        carrier_indices: Dict[int, List[int]],
    ) -> List[int]:
        """
        Extract bits using LWE interval decoding.
        """
        return self.lwe_strategy.extract(
            weights, carrier_indices, self.residuals_ref
        )