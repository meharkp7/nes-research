"""
Adaptive Carrier Scorer — magnitude + entropy + variance + kurtosis.
"""

import torch
from src.carrier_selection.carrier_features import CarrierFeatures


class AdaptiveScorer:
    """
    Weighted quality score from:
      - Magnitude  60%  (larger residuals harder to flip)
      - Entropy    20%  (higher entropy = richer region)
      - Variance   15%  (local spread helps distinguish signal)
      - Kurtosis    5%  (low kurtosis = well-behaved distribution)
    """

    def __init__(
        self,
        magnitude_weight: float = 0.60,
        entropy_weight:   float = 0.20,
        variance_weight:  float = 0.15,
        kurtosis_weight:  float = 0.05,
        layer_weight:     float = 0.0,
        eps:              float = 1e-8,
    ):
        self.magnitude_weight = magnitude_weight
        self.entropy_weight   = entropy_weight
        self.variance_weight  = variance_weight
        self.kurtosis_weight  = kurtosis_weight
        self.layer_weight     = layer_weight
        self.eps              = eps

    @staticmethod
    def normalize(scores: torch.Tensor) -> torch.Tensor:
        scores  = scores.float()
        mn, mx  = scores.min(), scores.max()
        if (mx - mn).item() < 1e-8:
            return torch.full_like(scores, 0.5)
        return (scores - mn) / (mx - mn + 1e-8)

    def score(
        self,
        residual_tensor: torch.Tensor,
        layer_quality_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Returns quality scores, same shape as residual_tensor, in [0, 1].
        """
        flat  = residual_tensor.flatten()
        score = torch.zeros(flat.numel(), dtype=torch.float32)

        if self.magnitude_weight > 0:
            score += self.magnitude_weight * self.normalize(flat.abs())

        if self.entropy_weight > 0:
            try:
                score += self.entropy_weight * self.normalize(
                    CarrierFeatures.entropy(flat).flatten()
                )
            except Exception:
                pass

        if self.variance_weight > 0:
            try:
                score += self.variance_weight * self.normalize(
                    CarrierFeatures.variance(flat).flatten()
                )
            except Exception:
                pass

        if self.kurtosis_weight > 0:
            try:
                # Low kurtosis = better → invert
                score += self.kurtosis_weight * self.normalize(
                    -CarrierFeatures.kurtosis(flat).flatten()
                )
            except Exception:
                pass

        score = self.normalize(score * layer_quality_factor)
        return score.reshape(residual_tensor.shape)