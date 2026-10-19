"""
Uniform Allocation Baseline.

Embeds equal bits in every layer regardless of layer quality.
Shows the benefit of QACI's quality-proportional allocation.

Expected result: Higher BER than QACI allocation because bits
land in low-quality layers (early/late) that have small residuals
and are more susceptible to quantization noise.
"""

from typing import Dict, List
import torch

from src.core.types      import EmbeddingConfig, EmbeddingResult
from src.extraction.sign_extractor import SignExtractor


class UniformAllocationEmbedder:
    """
    Sign-based embedding with equal bits allocated per layer.

    Contrasts with QACI which weights allocation by layer quality.
    This baseline shows how much QACI allocation matters.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config    = config
        self.extractor = SignExtractor()

    def embed(
        self,
        residuals:  Dict[int, torch.Tensor],
        bits:       List[int],
        total_bits: int = None,
    ) -> EmbeddingResult:
        """Embed with uniform per-layer allocation."""
        total_bits = total_bits or len(bits)
        n_layers   = len(residuals)
        per_layer  = total_bits // n_layers
        remainder  = total_bits % n_layers

        # Uniform indices: top-K by magnitude but equal K per layer
        uniform_indices = {}
        for i, lid in enumerate(sorted(residuals.keys())):
            n      = per_layer + (1 if i < remainder else 0)
            flat   = residuals[lid].flatten().abs()
            n      = min(n, flat.numel())
            _, idx = torch.topk(flat, n, largest=True)
            uniform_indices[lid] = sorted(idx.tolist())

        # Sign embed
        embedded               = {}
        bit_idx                = 0
        actual_carrier_indices = {}

        for layer_id in sorted(residuals.keys()):
            residual_tensor = residuals[layer_id].clone().detach()
            indices         = uniform_indices.get(layer_id, [])
            embedded_flat   = residual_tensor.flatten()
            actual_indices  = []

            for carrier_idx in indices:
                if bit_idx >= len(bits):
                    break
                val = embedded_flat[carrier_idx].item()
                magnitude = max(abs(val), self.config.min_magnitude)
                embedded_flat[carrier_idx] = magnitude if bits[bit_idx] == 1 else -magnitude
                actual_indices.append(carrier_idx)
                bit_idx += 1

            embedded[layer_id]               = embedded_flat.reshape(residual_tensor.shape)
            actual_carrier_indices[layer_id] = actual_indices

        return EmbeddingResult(
            success=          True,
            embedded_weights= embedded,
            carrier_indices=  actual_carrier_indices,
            layer_allocation= {lid: len(idx) for lid, idx in actual_carrier_indices.items()},
            bits_embedded=    bit_idx,
            total_bits=       len(bits),
            efficiency=       bit_idx / len(bits) if bits else 0.0,
            metadata={"strategy": "uniform_allocation"}
        )