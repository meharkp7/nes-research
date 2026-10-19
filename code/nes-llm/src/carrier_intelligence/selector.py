"""
QACI Carrier Selector — per-layer carrier selection orchestrator.

Pipeline:
    residual
        → CarrierFeatureExtractor  (9-column feature matrix)
        → FeatureNormalizer        (column-wise min-max)
        → QualityScore             (single score per carrier)
        → top-K selection
"""

from typing import List, Tuple

import torch

from src.carrier_intelligence.feature_extractor  import CarrierFeatureExtractor
from src.carrier_intelligence.feature_normalizer import FeatureNormalizer
from src.carrier_intelligence.quality_score      import QualityScore


class CarrierSelector:
    """QACI Carrier Intelligence Engine — selects top-K best carrier positions."""

    def __init__(self, layer_quality_factor: float = 1.0):
        self.extractor            = CarrierFeatureExtractor()
        self.normalizer           = FeatureNormalizer()
        self.scorer               = QualityScore()
        self.layer_quality_factor = layer_quality_factor

    def select(
        self,
        residual: torch.Tensor,
        fp16_weight: torch.Tensor,
        quantized_weight: torch.Tensor,
        num_carriers: int,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Full QACI selection using 9-column feature matrix.

        Returns:
            (sorted_indices, quality_scores_full_tensor)
        """
        flat_len     = residual.numel()
        num_carriers = min(num_carriers, flat_len)
        if num_carriers <= 0:
            return [], torch.zeros(flat_len)

        features       = self.extractor.extract(residual, fp16_weight, quantized_weight)
        features       = self.normalizer.normalize(features)
        quality_scores = self.scorer.compute(features, self.layer_quality_factor)

        _, top_indices = torch.topk(quality_scores, num_carriers, largest=True)
        return sorted(top_indices.cpu().tolist()), quality_scores

    def select_by_magnitude(self, residual: torch.Tensor, num_carriers: int) -> List[int]:
        """
        Magnitude-only fallback — used when fp16/nf4 weights are unavailable.
        """
        flat         = residual.flatten().abs()
        num_carriers = min(num_carriers, flat.numel())
        if num_carriers <= 0:
            return []
        _, indices = torch.topk(flat, num_carriers, largest=True)
        return sorted(indices.cpu().tolist())