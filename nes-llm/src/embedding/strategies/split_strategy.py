"""
W5.3 — sign/parity split: a fraction of carriers carries LWE-style
parity, the rest carries production sign flips.

RESEARCH_PLAN §4 W5.3: *"Parity on a fraction of carriers, sign on the
rest. Makes the stealth-vs-robustness trade-off an explicit dial rather
than a per-scheme guess."* The dial is ``split_fraction``:

    0.0  -> pure sign     (sign's robustness, sign's detectability)
    1.0  -> pure parity   (LWE's stealth, LWE's fragility)

Mechanisms are delegated, never re-implemented (the QaeDictAdapter /
AdaptiveStrategy rule): parity goes through ``LWEStrategy.embed``, sign
through ``SignEmbeddingStrategy.embed``, on DISJOINT carrier subsets so
the two passes never touch the same position.

Partition
---------
Parity iff ``blake2b("nes-split-v1:{layer}:{position}")`` falls below
the fraction. Keyless on purpose: exp13 established that the LWE grid
width is a public constant, so a keyed partition would be a security
veneer this suite does not claim. Membership is a per-position
predicate (not rank-based), so both sides agree even though the
extractor holds only the carriers that were actually used — a rank
over the embedder's selector set would silently drift.

Bit layout
----------
Parity carriers (layer-sorted, carrier order) take the first bits;
sign carriers take the rest. Extraction decodes parity with
``LWEStrategy.extract`` (stego-std grid, no cover) and sign with
``SignExtractor``, concatenating parity then sign — the same layout.
``EmbeddingResult.carrier_indices`` reports only carriers that were
actually written, so every reported position decodes to a real bit.
"""

import hashlib
from typing import Dict, List, Optional, Tuple

from src.core.types import EmbeddingConfig, EmbeddingResult


class SplitStrategy:
    """The parity/sign dial. See module docstring."""

    SALT = "nes-split-v1"

    def __init__(
        self,
        config: EmbeddingConfig,
        fraction: Optional[float] = None,
    ):
        from src.embedding.sign_strategy_v2 import SignEmbeddingStrategy
        from src.embedding.strategies.lwe_strategy import LWEStrategy

        self.config = config
        self.fraction = float(
            config.split_fraction if fraction is None else fraction
        )
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError(
                f"split_fraction must be in [0, 1], got {self.fraction}"
            )
        # Both mechanisms, built once; each is used only on its own
        # carrier subset.
        self._sign = SignEmbeddingStrategy(config)
        self._lwe = LWEStrategy(config)

    # ------------------------------------------------------------------
    # Partition
    # ------------------------------------------------------------------
    def _is_parity(self, layer_id: int, position: int) -> bool:
        if self.fraction <= 0.0:
            return False
        if self.fraction >= 1.0:
            return True
        digest = hashlib.blake2b(
            f"{self.SALT}:{layer_id}:{position}".encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "big") / 2**64 < self.fraction

    def partition(
        self, carriers: Dict[int, List[int]]
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """Split carriers into (parity, sign), order and layers intact."""
        parity: Dict[int, List[int]] = {}
        sign: Dict[int, List[int]] = {}
        for layer_id in sorted(carriers):
            p, s = [], []
            for position in carriers[layer_id]:
                if self._is_parity(layer_id, position):
                    p.append(position)
                else:
                    s.append(position)
            if p:
                parity[layer_id] = p
            if s:
                sign[layer_id] = s
        return parity, sign

    # ------------------------------------------------------------------
    # Embed — two delegated passes over disjoint positions
    # ------------------------------------------------------------------
    def embed(
        self,
        residuals: Dict[int, "torch.Tensor"],
        bits: List[int],
        selector_indices: Dict[int, List[int]],
    ) -> EmbeddingResult:
        parity_sel, sign_sel = self.partition(selector_indices)
        n_parity = sum(len(v) for v in parity_sel.values())

        # Parity slice first, sign slice the rest (layout the extractor
        # reconstructs). Slicing past the end is harmless: Python
        # returns the whole list, which is what fraction 1.0 wants.
        parity_bits = bits[:n_parity]
        sign_bits = bits[n_parity:]

        parity_result = None
        if parity_bits:
            parity_result = self._lwe.embed(
                residuals, parity_bits, parity_sel
            )
            base = parity_result.embedded_weights
        else:
            base = {
                layer_id: residuals[layer_id].clone().detach()
                for layer_id in sorted(residuals)
            }

        sign_result = None
        if sign_bits:
            sign_result = self._sign.embed(base, sign_bits, sign_sel)
            final_weights = sign_result.embedded_weights
        else:
            final_weights = base

        # Reported carriers = written carriers only, parity then sign,
        # layers sorted — the exact sequence both decoders replay.
        used_parity = (
            parity_result.carrier_indices if parity_result else {}
        )
        used_sign = sign_result.carrier_indices if sign_result else {}
        carriers: Dict[int, List[int]] = {}
        for layer_id in sorted(set(used_parity) | set(used_sign)):
            carriers[layer_id] = list(
                used_parity.get(layer_id, [])
            ) + list(used_sign.get(layer_id, []))

        bits_embedded = (
            (parity_result.bits_embedded if parity_result else 0)
            + (sign_result.bits_embedded if sign_result else 0)
        )
        total_bits = len(bits)
        n_parity_used = sum(len(v) for v in used_parity.values())
        n_sign_used = sum(len(v) for v in used_sign.values())

        return EmbeddingResult(
            success=True,
            embedded_weights=final_weights,
            carrier_indices=carriers,
            layer_allocation={
                lid: len(idx) for lid, idx in carriers.items()
            },
            bits_embedded=bits_embedded,
            total_bits=total_bits,
            efficiency=(
                bits_embedded / total_bits if total_bits else 0.0
            ),
            metadata={
                "strategy": f"split({self.fraction})",
                "split_fraction": self.fraction,
                "parity_bits": len(parity_bits),
                "sign_bits": len(sign_bits),
                "parity_carriers_used": n_parity_used,
                "sign_carriers_used": n_sign_used,
                "partition": (
                    "keyless position predicate, blake2b "
                    f"'{self.SALT}' (public by design — exp13)"
                ),
            },
        )

    # ------------------------------------------------------------------
    # Extract — parity decode on parity carriers, sign decode on the rest
    # ------------------------------------------------------------------
    def extract(
        self,
        residuals: Dict[int, "torch.Tensor"],
        carrier_indices: Dict[int, List[int]],
        residuals_ref: Optional[Dict[int, "torch.Tensor"]] = None,
    ) -> List[int]:
        from src.extraction.sign_extractor import SignExtractor

        parity, sign = self.partition(carrier_indices)
        recovered: List[int] = []
        if parity:
            # residuals_ref=None keeps LWEStrategy on its stego-std
            # grid (what an extractor holding only stego has).
            recovered.extend(
                self._lwe.extract(residuals, parity, residuals_ref)
            )
        if sign:
            recovered.extend(
                SignExtractor().extract(residuals, sign)
            )
        return recovered
