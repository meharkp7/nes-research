"""Sign-based embedding strategy — PRIMARY method."""

from src.embedding.base_embedder import BaseEmbedder
from src.core import EmbeddingConfig


class SignEmbeddingStrategy(BaseEmbedder):
    """
    Encodes bits as residual sign:
        bit=1 → positive magnitude
        bit=0 → negative magnitude

    Performance:
        σ=0.000 → BER=0.000
        σ=0.001 → BER≈0.020
        σ=0.002 → BER≈0.098
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)

    def get_bit_for_residual(self, residual: float, bit: int) -> float:
        magnitude = max(abs(residual), self.min_magnitude)
        return magnitude if bit == 1 else -magnitude


class SignExtractionStrategy:
    """Companion extraction — reads bit from residual sign."""

    @staticmethod
    def get_bit_from_residual(residual: float) -> int:
        return 1 if residual >= 0 else 0