"""
Feature normalizer for the QACI pipeline.

Applies column-wise min-max normalization to the 9-column carrier
feature matrix so that every feature contributes equally to quality scoring.
"""

import torch


class FeatureNormalizer:
    """
    Column-wise min-max normalization of the carrier feature matrix.

    Input shape:  [num_carriers, 9]
    Output shape: [num_carriers, 9]  — every column in [0, 1]

    Column layout (must match CarrierFeatureExtractor):
        0  Magnitude
        1  Variance
        2  Std
        3  Quantization Error
        4  Stability
        5  Robustness
        6  Cost
        7  Z-score
        8  Distortion
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalise feature matrix column-wise to [0, 1].
        Constant columns are set to 0.5 (neutral).
        """
        features  = features.float()
        minimum   = features.min(dim=0).values
        maximum   = features.max(dim=0).values
        span      = maximum - minimum

        safe_span  = torch.where(span > self.eps, span, torch.ones_like(span))
        normalized = (features - minimum) / safe_span

        # Constant columns → neutral 0.5
        constant_cols = span <= self.eps
        normalized[:, constant_cols] = 0.5

        return normalized.clamp(0.0, 1.0)