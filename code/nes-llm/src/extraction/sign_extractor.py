"""Sign-based bit extraction — companion to SignEmbeddingStrategy."""

from src.extraction.base_extractor import BaseExtractor


class SignExtractor(BaseExtractor):
    """
    Recovers bits by reading residual sign:
        residual >= 0 → bit=1
        residual <  0 → bit=0
    """

    def __init__(self):
        super().__init__()

    def get_bit_from_residual(self, residual: float) -> int:
        return 1 if residual >= 0 else 0