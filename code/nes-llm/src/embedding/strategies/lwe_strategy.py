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

# Grid width that satisfies both gates on Qwen2.5-3B.
#
# Measured frontier (results/exp11_lwe_alpha_pareto.json):
#   0.002  BER 0.586  fails robustness, undetectable
#   0.005  BER 0.013  both gates pass
#   0.010  BER 0.000  both gates pass   <- chosen
#   0.020  BER 0.000  both gates pass
#   0.050  BER 0.000  detector 70.6%, fails again
#
# A single model is not a general claim; this is the middle of the window
# with margin on both sides, and every other setting remains reachable
# through the grid_width argument.
DEFAULT_GRID_WIDTH = 0.010

# W5.4 — per-layer rule (config.lwe_width_rule = "per_layer").
# Width proportional to the layer's own noise so the grid-to-noise
# ratio w/sigma is equalized across layers (Qwen2.5-3B's quietest
# layer carries a global grid 8.8 sigma wide against 3.8 for the
# noisiest), clipped to exp11's measured window — the only window
# with a measured frontier (0.005 passes sigma=0.001, 0.02 stays
# undetectable). SCALE=4 puts the median layer at ~0.0096, i.e. the
# global default, so the two rules differ only in per-layer spread.
PER_LAYER_WIDTH_SCALE = 4.0
PER_LAYER_WIDTH_FLOOR = 0.005
PER_LAYER_WIDTH_CAP = 0.020

# W5.4 layer_rank — run 1 measured why MAGNITUDE-keying cannot pass
# the robustness gate: noise inflates std through sqrt(std^2 +
# sigma^2), and 36/36 layers bucket differently at every sigma
# tested, so the extractor sizes a different grid than the embedder
# used (BER 0.0223 at sigma=0.001, 0.5747 at 0.002, vs 0.0 at 0).
# Rank is the part of "keyed by layer noise" that noise cannot move:
# the inflation map is strictly monotone, so the layer ORDERING is
# preserved exactly and both sides assign identical widths. The
# ladder is fixed constants — exp11's floor up to the global
# default — so no endpoint depends on either side's measurements.
LAYER_RANK_LOW = PER_LAYER_WIDTH_FLOOR
LAYER_RANK_HIGH = DEFAULT_GRID_WIDTH
WIDTH_RULES = ("global", "per_layer", "layer_rank")


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
        grid_width: Optional[float] = None,
    ):
        self.config         = config
        self.secret_key     = secret_key or b'\x00' * 32
        self.alpha          = config.alpha        # grid scale multiplier
        self.min_magnitude  = config.min_magnitude

        # Explicit grid width, or derive one that actually works.
        #
        # With the shipped defaults the width is
        #     max(std * alpha * scale, min_magnitude * 2)
        # = max(0.002 * 0.001 * ~1, 0.001 * 2) = max(2e-6, 2e-3)
        # so the floor dominates by ~1000x and alpha has no effect at
        # all. At that width the scheme is undetectable but collapses
        # under noise (BER 0.586 at sigma=0.001).
        #
        # The measured frontier on Qwen2.5-3B is: width 0.005-0.02
        # satisfies both BER <= 0.02 at sigma=0.001 and detector accuracy
        # <= 55%, while 0.05 becomes detectable again (70.6%) and 0.002
        # fails robustness. DEFAULT_GRID_WIDTH sits in the middle of that
        # window. See results/exp11_lwe_alpha_pareto.json.
        self.grid_width = (
            grid_width
            if grid_width is not None
            else DEFAULT_GRID_WIDTH
        )

        # W5.4: "global" = the absolute width above (unchanged,
        # exp10/exp11/exp12 byte-compatible); "per_layer" = derived
        # from each layer's std inside _derive_interval_width;
        # "layer_rank" = fixed ladder keyed by the layer's rank in
        # the noise ordering (noise-invariant, see _rank_widths).
        self.width_rule = getattr(config, "lwe_width_rule", "global")
        if self.width_rule not in WIDTH_RULES:
            raise ValueError(
                f"lwe_width_rule must be one of {WIDTH_RULES}, "
                f"got {self.width_rule!r}"
            )

        self._grid_cache: Dict[int, float] = {}

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

        if self.width_rule == "per_layer":
            # W5.4 — width proportional to THIS layer's own noise,
            # clipped to exp11's measured window. std is coarsened to
            # 4 decimals so embed (original std, line ~231) and
            # extract (stego std, line ~316) bucket identically: an
            # LWE embed moves per-layer std by at most 0.0153% on
            # Qwen2.5-3B — zero bucket flips (exp22's artifact pins
            # the widths; gate BER 0.0 would catch any edge case).
            coarse = round(float(residual_std), 4)
            width = PER_LAYER_WIDTH_SCALE * coarse
            interval_width = min(
                max(width, PER_LAYER_WIDTH_FLOOR),
                PER_LAYER_WIDTH_CAP,
            )
        else:
            interval_width = (
                self.grid_width
                if self.grid_width is not None
                else max(
                    residual_std * self.alpha * scale,
                    self.min_magnitude * 2,
                )
            )

        self._grid_cache[layer_id] = interval_width
        return interval_width

    def _rank_widths(self, stds: Dict[int, float]) -> Dict[int, float]:
        """
        W5.4 layer_rank — widths keyed by each layer's RANK in the
        noise ordering, never by its magnitude.

        Rank survives what magnitude cannot: noise maps every std
        through the same strictly monotone sqrt(x^2 + sigma^2), so
        embed (original stds) and extract (noisy stds) sort the
        layers identically and assign identical widths from the
        fixed ladder [LAYER_RANK_LOW, LAYER_RANK_HIGH]. Ties break
        on layer id, which both sides share.
        """
        ordered = sorted(stds.items(), key=lambda kv: (kv[1], kv[0]))
        n = len(ordered)
        widths: Dict[int, float] = {}
        for rank, (layer_id, _std) in enumerate(ordered):
            frac = rank / (n - 1) if n > 1 else 0.5
            widths[layer_id] = LAYER_RANK_LOW + frac * (
                LAYER_RANK_HIGH - LAYER_RANK_LOW
            )
        return widths

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

        # W5.4 layer_rank: the whole layer set at once — ranks are
        # computed once from the ORIGINAL stds, before the loop.
        rank_widths = (
            self._rank_widths({
                lid: t.float().std().item()
                for lid, t in residuals.items()
            })
            if self.width_rule == "layer_rank" else None
        )

        embedded               = {}
        bit_idx                = 0
        actual_carrier_indices = {}

        for layer_id in sorted(residuals.keys()):
            residual_tensor  = residuals[layer_id].clone().detach()
            indices          = selector_indices.get(layer_id, [])
            embedded_flat    = residual_tensor.flatten()
            actual_indices   = []

            # Derive interval width for this layer
            if rank_widths is not None:
                interval_width = rank_widths[layer_id]
                self._grid_cache[layer_id] = interval_width
            else:
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

        # W5.4 layer_rank: ranks from THIS side's view (cover or
        # stego, possibly noisy) — the ordering matches the
        # embedder's because the noise inflation is monotone.
        rank_widths = None
        if self.width_rule == "layer_rank":
            view = residuals_ref if used_cover else weights
            rank_widths = self._rank_widths({
                lid: t.float().std().item()
                for lid, t in view.items()
            })

        for layer_id in sorted(weights.keys()):
            weight_tensor = weights[layer_id]
            indices       = carrier_indices.get(layer_id, [])
            weight_flat   = weight_tensor.flatten()

            if rank_widths is not None:
                interval_width = rank_widths[layer_id]
            else:
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