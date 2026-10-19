"""
Magnitude-Aware Extractor.

The extraction rule is identical to sign-based:
    residual >= 0 → bit=1
    residual <  0 → bit=0

This is correct because magnitude-aware embedding encodes
information only in the SIGN. The magnitude is boosted to
make the sign more robust, but the decision rule is the same.

The extractor is therefore sign-based — this class exists
as a named companion to MagnitudeAwareStrategy for clarity.
"""

from src.extraction.base_extractor import BaseExtractor


class MagnitudeAwareExtractor(BaseExtractor):
    """
    Extraction companion for MagnitudeAwareStrategy.

    Decision rule: residual >= 0 → 1, residual < 0 → 0.
    Identical to SignExtractor but named for clarity in
    comparative experiments.
    """

    def __init__(self):
        super().__init__()

    def get_bit_from_residual(self, residual: float) -> int:
        return 1 if residual >= 0 else 0