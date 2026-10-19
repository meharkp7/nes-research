"""
Quality scorer for the QACI carrier selection pipeline.

Combines information, robustness, and distortion feature groups into a
single quality score per carrier position. Higher score = better carrier.
"""

import torch


class QualityScore:
    """
    Computes an overall carrier quality score from the 9-column
    carrier feature matrix produced by CarrierFeatureExtractor.

    Feature Matrix Columns:
        0  Magnitude           (higher → more signal, better carrier)
        1  Variance            (higher → more information)
        2  Std                 (higher → spread useful for sign embedding)
        3  Quantization Error  (lower  → more stable after dequant)
        4  Stability           (higher → won't flip sign under noise)
        5  Robustness          (higher → survives quantization cycles)
        6  Cost                (lower  → cheaper to embed without distortion)
        7  Z-score             (lower  → not an outlier, safer)
        8  Distortion          (lower  → less distortion after embedding)

    Final score:
        quality = (robustness_score * information_score) / (1 + distortion_score)
        then normalized to [0, 1].
    """

    MAGNITUDE          = 0
    VARIANCE           = 1
    STD                = 2
    QUANTIZATION_ERROR = 3
    STABILITY          = 4
    ROBUSTNESS         = 5
    COST               = 6
    ZSCORE             = 7
    DISTORTION         = 8

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Min-max normalize to [0, 1]."""
        mn, mx = x.min(), x.max()
        if (mx - mn).item() < self.eps:
            return torch.full_like(x, 0.5)
        return (x - mn) / (mx - mn + self.eps)

    def information_score(self, features: torch.Tensor) -> torch.Tensor:
        """High magnitude + high variance = rich carrier."""
        magnitude = features[:, self.MAGNITUDE]
        variance  = features[:, self.VARIANCE]
        std       = features[:, self.STD]
        return 0.50 * magnitude + 0.25 * variance + 0.25 * std

    def robustness_score(self, features: torch.Tensor) -> torch.Tensor:
        """High stability + high robustness + low quantization error = robust."""
        stability  = features[:, self.STABILITY]
        robustness = features[:, self.ROBUSTNESS]
        quant_inv  = 1.0 - features[:, self.QUANTIZATION_ERROR].clamp(0.0, 1.0)
        return 0.40 * robustness + 0.40 * stability + 0.20 * quant_inv

    def distortion_score(self, features: torch.Tensor) -> torch.Tensor:
        """Low cost + low z-score + low distortion = safe to embed."""
        distortion = features[:, self.DISTORTION]
        cost       = features[:, self.COST]
        zscore_abs = features[:, self.ZSCORE].abs()
        return 0.50 * distortion + 0.30 * cost + 0.20 * zscore_abs

    def compute(
        self,
        feature_matrix: torch.Tensor,
        layer_quality_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute final quality scores for all carrier positions.

        Args:
            feature_matrix:       [num_carriers, 9] normalized features.
            layer_quality_factor: Per-layer weight from LayerProfiler.

        Returns:
            quality: [num_carriers] in [0, 1].
        """
        info       = self._norm(self.information_score(feature_matrix))
        robustness = self._norm(self.robustness_score(feature_matrix))
        distortion = self._norm(self.distortion_score(feature_matrix))

        quality = (robustness * info) / (1.0 + distortion + self.eps)
        quality = self._norm(quality) * layer_quality_factor
        return quality.clamp(0.0, 1.0)

    def statistics(self, quality: torch.Tensor) -> dict:
        """Return summary statistics of quality scores."""
        return {
            "mean":   quality.mean().item(),
            "std":    quality.std().item(),
            "min":    quality.min().item(),
            "max":    quality.max().item(),
            "top10%": torch.topk(quality.flatten(),
                                 max(1, int(0.10 * quality.numel()))).values.mean().item(),
        }