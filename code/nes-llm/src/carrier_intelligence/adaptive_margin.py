"""
Adaptive Margin Controller (AMC) for magnitude-aware sign embedding.

Converts per-carrier quality scores into per-carrier embedding margins.
High-quality carriers get larger margins (more robust to noise).
Low-quality carriers get smaller margins (less distortion).
"""

import torch


class AdaptiveMarginController:
    """
    margin[i] = residual.std() * alpha * normalized_quality[i]

    For sign embedding:
        embedded = sign(bit) * (|residual| + margin)
    """

    def __init__(self, alpha: float = 0.25, min_margin: float = 1e-4, eps: float = 1e-8):
        self.alpha      = alpha
        self.min_margin = min_margin
        self.eps        = eps

    def _normalize(self, quality_scores: torch.Tensor) -> torch.Tensor:
        mn, mx = quality_scores.min(), quality_scores.max()
        span = mx - mn
        if span.item() < self.eps:
            return torch.full_like(quality_scores, 0.5)
        return (quality_scores - mn) / (span + self.eps)

    def compute(self, residual: torch.Tensor, quality_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute per-carrier adaptive margins.

        Args:
            residual:       Full residual tensor (for global scale).
            quality_scores: [num_carriers] quality scores.

        Returns:
            margins: [num_carriers], all positive.
        """
        quality      = self._normalize(quality_scores)
        global_scale = residual.float().std().item() * self.alpha
        return (global_scale * quality + self.min_margin).clamp(min=self.min_margin)

    def embed_with_margin(
        self,
        residual_values: torch.Tensor,
        bits: torch.Tensor,
        margins: torch.Tensor,
    ) -> torch.Tensor:
        """
        bit=1 → +(|residual| + margin)
        bit=0 → -(|residual| + margin)
        """
        magnitude = residual_values.abs() + margins
        signs     = bits.float() * 2 - 1   # 0→-1, 1→+1
        return magnitude * signs

    def statistics(self, margins: torch.Tensor, residual: torch.Tensor) -> dict:
        return {
            "mean_margin":  margins.mean().item(),
            "std_margin":   margins.std().item(),
            "min_margin":   margins.min().item(),
            "max_margin":   margins.max().item(),
            "residual_std": residual.float().std().item(),
            "alpha":        self.alpha,
        }