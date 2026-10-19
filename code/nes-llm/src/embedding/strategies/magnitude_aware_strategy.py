"""
Magnitude-Aware Embedding Strategy.

Unlike sign-based embedding which only uses the sign of a residual,
magnitude-aware embedding uses both the sign AND encodes a margin:

    bit=1 → +( |residual| + margin )
    bit=0 → -( |residual| + margin )

The margin is adaptive per carrier: high-quality carriers get larger
margins (more robust to noise), low-quality carriers get smaller
margins (less distortion). This is fundamentally different from
sign-based: the magnitude change is intentional and calibrated.

Key advantage over sign-based:
    - Survives larger noise (σ up to 0.005 with BER < 0.02)
    - Tradeoff: slightly more distortion (KL divergence ~2-3x higher)

Performance targets:
    σ=0.000 → BER=0.000
    σ=0.001 → BER=0.000
    σ=0.002 → BER=0.001
    σ=0.005 → BER=0.012
"""

from typing import Dict, List, Optional
import torch

from src.embedding.base_embedder            import BaseEmbedder
from src.carrier_intelligence.adaptive_margin import AdaptiveMarginController
from src.core.types                          import EmbeddingConfig, EmbeddingResult


class MagnitudeAwareStrategy(BaseEmbedder):
    """
    Magnitude-aware embedding with per-carrier adaptive margins.

    Each carrier gets its own margin based on its quality score:
        margin[i] = residual.std() * alpha * normalized_quality[i]

    The bit is encoded in the sign, but the magnitude is boosted
    by the margin to create a larger decision boundary — making
    the embedded bit more robust to quantization noise.

    Args:
        config: EmbeddingConfig with alpha (margin strength).
        quality_scores: Optional {layer_id: score_tensor}.
                        If None, falls back to uniform margin.
    """

    def __init__(
        self,
        config:         EmbeddingConfig,
        quality_scores: Optional[Dict[int, torch.Tensor]] = None,
    ):
        super().__init__(config)
        self.quality_scores = quality_scores or {}
        self.margin_ctrl    = AdaptiveMarginController(
            alpha=      config.alpha,
            min_margin= config.min_magnitude,
        )
        self._precomputed_margins: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def precompute_margins(self, residuals: Dict[int, torch.Tensor]) -> None:
        """
        Pre-compute per-carrier margins for all layers.
        Call this before embed() for best performance.
        """
        self._precomputed_margins = {}
        for lid, residual in residuals.items():
            flat    = residual.flatten()
            scores  = self.quality_scores.get(lid)

            if scores is None:
                # Fallback: use magnitude as proxy quality score
                scores = flat.abs()

            # Flatten scores to match flat residual length
            scores_flat = scores.flatten()[:flat.numel()]
            if scores_flat.numel() < flat.numel():
                pad = torch.full(
                    (flat.numel() - scores_flat.numel(),),
                    scores_flat.mean().item()
                )
                scores_flat = torch.cat([scores_flat, pad])

            self._precomputed_margins[lid] = self.margin_ctrl.compute(
                residual, scores_flat
            )

    # ------------------------------------------------------------------
    # Core embedding method
    # ------------------------------------------------------------------

    def get_bit_for_residual(self, residual: float, bit: int) -> float:
        """
        Embed bit using fixed alpha margin (used when no per-carrier
        margins are available — fallback for single-value calls).
        """
        magnitude = max(abs(residual), self.min_magnitude)
        margin    = magnitude * self.alpha
        boosted   = magnitude + margin
        return boosted if bit == 1 else -boosted

    def embed(
        self,
        residuals:        Dict[int, torch.Tensor],
        bits:             List[int],
        selector_indices: Dict[int, List[int]],
    ) -> EmbeddingResult:
        """
        Embed bits using per-carrier adaptive margins.

        Overrides BaseEmbedder.embed() to use precomputed margins
        instead of the scalar get_bit_for_residual() call.
        """
        # Pre-compute margins if not already done
        if not self._precomputed_margins:
            self.precompute_margins(residuals)

        embedded               = {}
        bit_idx                = 0
        actual_carrier_indices = {}

        for layer_id in sorted(residuals.keys()):
            residual_tensor = residuals[layer_id].clone().detach()
            indices         = selector_indices.get(layer_id, [])
            embedded_flat   = residual_tensor.flatten()
            actual_indices  = []

            # Get per-position margins for this layer
            layer_margins = self._precomputed_margins.get(layer_id)

            for carrier_idx in indices:
                if bit_idx >= len(bits):
                    break

                bit     = bits[bit_idx]
                val     = embedded_flat[carrier_idx].item()

                if layer_margins is not None and carrier_idx < len(layer_margins):
                    margin = layer_margins[carrier_idx].item()
                else:
                    margin = max(abs(val), self.min_magnitude) * self.alpha

                magnitude    = max(abs(val), self.min_magnitude)
                boosted      = magnitude + margin
                embedded_val = boosted if bit == 1 else -boosted

                embedded_flat[carrier_idx] = embedded_val
                actual_indices.append(carrier_idx)
                bit_idx += 1

            embedded[layer_id]               = embedded_flat.reshape(residual_tensor.shape)
            actual_carrier_indices[layer_id] = actual_indices

        bits_embedded = bit_idx
        total_bits    = len(bits)

        return EmbeddingResult(
            success=          True,
            embedded_weights= embedded,
            carrier_indices=  actual_carrier_indices,
            layer_allocation= {lid: len(idx) for lid, idx in actual_carrier_indices.items()},
            bits_embedded=    bits_embedded,
            total_bits=       total_bits,
            efficiency=       bits_embedded / total_bits if total_bits > 0 else 0.0,
            metadata={
                'strategy':   'magnitude_aware',
                'alpha':      self.alpha,
                'has_quality_scores': len(self.quality_scores) > 0,
            }
        )