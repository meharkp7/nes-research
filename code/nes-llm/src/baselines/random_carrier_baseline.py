"""
Random Carrier Baseline.

Uses the same sign-based embedding as NES but with RANDOM
carrier selection instead of QACI quality-aware selection.

Comparing NES (sign+QACI) vs this baseline isolates the
contribution of carrier intelligence to BER and robustness.

Expected result: BER worse by 30-50% at σ=0.001 compared to
NES with QACI selection.
"""

from typing import Dict, List
import torch

from src.core.types      import EmbeddingConfig, EmbeddingResult
from src.extraction.sign_extractor import SignExtractor


class RandomCarrierEmbedder:
    """
    Sign-based embedding with random carrier selection.

    This is the ablation baseline that shows QACI matters.
    Random selection picks any position, including low-magnitude
    positions that are fragile under quantization noise.
    """

    def __init__(self, config: EmbeddingConfig, seed: int = 42):
        self.config    = config
        self.seed      = seed
        self.extractor = SignExtractor()

    def embed(
        self,
        residuals:        Dict[int, torch.Tensor],
        bits:             List[int],
        total_bits:       int = None,
    ) -> EmbeddingResult:
        """
        Embed using random carrier selection (ignores selector_indices).
        Builds its own random indices.
        """
        torch.manual_seed(self.seed)

        total_bits = total_bits or len(bits)
        n_layers   = len(residuals)
        per_layer  = total_bits // n_layers
        remainder  = total_bits % n_layers

        # Build random indices
        random_indices = {}
        for i, lid in enumerate(sorted(residuals.keys())):
            n      = per_layer + (1 if i < remainder else 0)
            n      = min(n, residuals[lid].numel())
            idx    = torch.randperm(residuals[lid].numel())[:n]
            random_indices[lid] = sorted(idx.tolist())

        # Sign-based embedding at random positions
        embedded               = {}
        bit_idx                = 0
        actual_carrier_indices = {}

        for layer_id in sorted(residuals.keys()):
            residual_tensor = residuals[layer_id].clone().detach()
            indices         = random_indices.get(layer_id, [])
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
            metadata={"strategy": "random_carrier"}
        )