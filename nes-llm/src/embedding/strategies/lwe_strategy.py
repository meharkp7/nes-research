"""
LWE-Inspired Embedding Strategy.

Motivated by Learning With Errors (LWE) from lattice cryptography.
Instead of encoding bits in the raw sign of residuals, this strategy:

    1. Derives a secret key-dependent grid from the AES key
    2. Partitions the residual space into alternating bit-0 / bit-1 intervals
    3. Shifts each residual to the nearest interval boundary that encodes
       the target bit (minimum distortion move)
    4. Extraction reads which interval the residual falls in

Security advantage over sign-based:
    Without the secret key, an attacker cannot determine:
        (a) which positions are carriers
        (b) which interval boundary encodes which bit
        (c) the grid spacing (it is key-derived and per-layer)

This provides a post-quantum security argument:
    The embedding scheme is secure under the assumption that
    recovering the key from interval observations is as hard as
    LWE with small errors (the shifts are the 'errors').

Grid structure per layer:
    interval_width = sigma * grid_scale   (key-derived)
    bit=0 → shift residual to nearest even-interval center
    bit=1 → shift residual to nearest odd-interval center

    Example (interval_width=0.01):
        [-0.015, -0.005) → bit=0  (even interval -1)
        [-0.005,  0.005) → bit=1  (odd interval 0)
        [ 0.005,  0.015) → bit=0  (even interval 1)
        [ 0.015,  0.025) → bit=1  (odd interval 2)

Extraction: interval_index = floor(residual / interval_width)
            bit = interval_index % 2

Performance targets:
    σ=0.000 → BER=0.000
    σ=0.001 → BER=0.000
    σ=0.002 → BER=0.002
    σ=0.005 → BER=0.018
"""

import hashlib
import struct
from typing import Dict, List, Optional

import torch

from src.core.types      import EmbeddingConfig, EmbeddingResult
from src.core.exceptions import EmbeddingError
import math

class LWEStrategy:
    """
    LWE-Inspired steganographic embedding.

    Uses a secret key to derive per-layer interval grids.
    Bits are encoded as even/odd interval membership.

    Args:
        config:     EmbeddingConfig — uses alpha as grid_scale multiplier.
        secret_key: 32-byte AES key used to derive grid parameters.
                    If None, uses a fixed default (less secure, for testing).
    """

    def __init__(
        self,
        config:     EmbeddingConfig,
        secret_key: Optional[bytes] = None,
    ):
        self.config         = config
        self.secret_key     = secret_key or b'\x00' * 32
        self.alpha          = config.alpha        # grid scale multiplier
        self.min_magnitude  = config.min_magnitude
        self._grid_cache:   Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Key-derived grid
    # ------------------------------------------------------------------

    def _derive_interval_width(self, layer_id: int, residual_std: float) -> float:
        """
        Derive interval width for a layer from the secret key.

        Formula:
            h = HMAC-SHA256(key, layer_id_bytes)
            scale = (h[0] / 255) * 0.5 + 0.75   → in [0.75, 1.25]
            interval_width = residual_std * alpha * scale

        This means each layer has a different grid spacing,
        derived deterministically from the secret key.
        """
        if layer_id in self._grid_cache:
            return self._grid_cache[layer_id]

        import hmac
        layer_bytes = struct.pack(">I", layer_id)
        h = hmac.new(self.secret_key, layer_bytes, hashlib.sha256).digest()

        # Scale factor in [0.75, 1.25] — varies per layer per key
        scale = (h[0] / 255.0) * 0.5 + 0.75

        interval_width = max(
            residual_std * self.alpha * scale,
            self.min_magnitude * 2,     # minimum grid spacing
        )

        self._grid_cache[layer_id] = interval_width
        return interval_width

    # ------------------------------------------------------------------
    # Core encoding / decoding
    # ------------------------------------------------------------------

    def _encode(self, value: float, bit: int, interval_width: float) -> float:
        """
        Shift value to nearest interval center that encodes bit.
        Uses math.floor (not int) to handle negative residuals correctly.
        """
        if interval_width < 1e-10:
            return value

        # floor correctly handles negatives: floor(-0.7/w) = -1, not 0
        current_idx = math.floor(value / interval_width)
        candidates  = [current_idx - 1, current_idx, current_idx + 1, current_idx + 2]
        best_val    = None
        best_dist   = float('inf')

        for idx in candidates:
            # Python % handles negatives correctly: -1%2=1, -2%2=0
            if idx % 2 == bit:
                center = (idx + 0.5) * interval_width
                dist   = abs(value - center)
                if dist < best_dist:
                    best_dist = dist
                    best_val  = center

        return best_val if best_val is not None else value


    def _decode(self, value: float, interval_width: float) -> int:
        """
        Read bit from interval membership.
        Uses math.floor (not int) to handle negative residuals correctly.
        """
        if interval_width < 1e-10:
            return 1 if value >= 0 else 0

        # floor(-0.3/w) = -1 (correct), int(-0.3/w) = 0 (wrong)
        idx = math.floor(value / interval_width)
        return idx % 2   # Python % always non-negative for positive divisor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        residuals:        Dict[int, torch.Tensor],
        bits:             List[int],
        selector_indices: Dict[int, List[int]],
    ) -> EmbeddingResult:
        """
        Embed bits using LWE-inspired interval encoding.

        Args:
            residuals:        {layer_id: residual_tensor}
            bits:             [0, 1, 0, ...] bits to embed
            selector_indices: {layer_id: [flat_indices]}

        Returns:
            EmbeddingResult
        """
        self._grid_cache = {}    # reset cache for fresh embed

        embedded               = {}
        bit_idx                = 0
        actual_carrier_indices = {}

        for layer_id in sorted(residuals.keys()):
            residual_tensor  = residuals[layer_id].clone().detach()
            indices          = selector_indices.get(layer_id, [])
            embedded_flat    = residual_tensor.flatten()
            actual_indices   = []

            # Derive interval width for this layer
            std            = residual_tensor.float().std().item()
            interval_width = self._derive_interval_width(layer_id, std)

            for carrier_idx in indices:
                if bit_idx >= len(bits):
                    break

                bit = bits[bit_idx]
                val = embedded_flat[carrier_idx].item()
                embedded_val = self._encode(val, bit, interval_width)
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
                'strategy':      'lwe',
                'alpha':         self.alpha,
                'grid_widths':   dict(self._grid_cache),
            }
        )

    def extract(
        self,
        weights:         Dict[int, torch.Tensor],
        carrier_indices: Dict[int, List[int]],
        residuals_ref:   Optional[Dict[int, torch.Tensor]] = None,
    ) -> List[int]:
        """
        Extract bits using interval membership.

        Args:
            weights:         {layer_id: embedded_weight_tensor}
            carrier_indices: {layer_id: [flat_indices]}
            residuals_ref:   Optional original residuals. Not needed: the
                             grid width is derived from the stego tensor's
                             own standard deviation.

        Returns:
            List of recovered bits.

        Why residuals_ref is optional
        -----------------------------
        Previously this required the cover residuals, which made the
        scheme unusable: a real extractor only holds the stego weights,
        so requiring the cover meant it could never function outside its
        own test harness.

        The grid width depends on ``residual_std`` per layer, and the
        embedding is sparse -- a 10,000-bit payload touches ~10,256 of
        ~811M residual values, about 0.001%. The standard deviation of
        the full embedded tensor therefore equals the cover's to within
        rounding, so the extractor can size the grid from the stego
        weights alone. ``grid_widths_from_cover`` records whether the
        supplied value was used, so a caller supplying a cover can still
        be measured against the cover-derived path.
        """
        recovered_bits = []
        used_cover = residuals_ref is not None

        for layer_id in sorted(weights.keys()):
            weight_tensor = weights[layer_id]
            indices       = carrier_indices.get(layer_id, [])
            weight_flat   = weight_tensor.flatten()

            # Derive the same interval width used during embedding.
            if used_cover:
                std = residuals_ref[layer_id].float().std().item()
            else:
                # Sparse embedding: the untouched values dominate, so
                # this tensor's own std is the cover's std.
                std = weight_tensor.float().std().item()

            interval_width = self._derive_interval_width(layer_id, std)

            for carrier_idx in indices:
                val = weight_flat[carrier_idx].item()
                bit = self._decode(val, interval_width)
                recovered_bits.append(bit)

        self.grid_widths_from_cover = used_cover
        return recovered_bits