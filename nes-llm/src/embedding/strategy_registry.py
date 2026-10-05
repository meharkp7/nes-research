"""
Embedding strategy registry.

Eight strategies exist on disk and exactly one was reachable: production
always constructed ``SignEmbeddingStrategy`` regardless of
``EmbeddingConfig.embedding_strategy``, which was accepted and ignored.

This module gives every strategy one uniform contract and records, per
strategy, the properties that actually decide whether it is viable:

``forces_sign_flip``
    Whether the scheme writes a carrier as +/-|r| chosen by the payload
    bit. This is the leak behind the 70.5% neural-detector result: it
    forces a ~50/50 sign split at carriers regardless of the clean sign
    distribution, and no choice of alpha, gamma or payload size removes
    it. Measured across a 100x alpha and 5x gamma sweep.

``extract_needs_cover``
    Whether extraction requires the original (cover) residuals. A real
    extractor only ever holds the stego weights, so a strategy that needs
    the cover is not a usable hiding scheme no matter how good its
    security claim looks.

``status``
    UNWIRED / READY / NEEDS_TRAINING / BLOCKED. Recorded honestly so a
    strategy cannot be selected and silently underperform.
"""

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from src.core.types import EmbeddingConfig, EmbeddingResult


@dataclass
class StrategySpec:
    """Everything needed to select, run and judge a strategy."""

    name: str
    factory: Callable[[EmbeddingConfig], Any]
    module: str
    class_name: str

    # Structural properties, established by measurement or by reading the
    # implementation. Not guesses.
    forces_sign_flip: bool = False
    extract_needs_cover: bool = False
    needs_trained_model: bool = False
    status: str = "UNWIRED"
    notes: str = ""

    # Populated by probe().
    probe: Dict[str, Any] = field(default_factory=dict)


def _sign_v2(config):
    from src.embedding.sign_strategy_v2 import SignEmbeddingStrategy

    return SignEmbeddingStrategy(config)


def _magnitude_aware(config):
    from src.embedding.strategies.magnitude_aware_strategy import (
        MagnitudeAwareStrategy,
    )

    return MagnitudeAwareStrategy(config)


def _lwe(config):
    from src.embedding.strategies.lwe_strategy import LWEStrategy

    return LWEStrategy(config)


def _neural(config):
    from src.embedding.strategies.neural_strategy import NeuralStrategy

    return NeuralStrategy(config)


class QaeDictAdapter:
    """Production dict contract over the per-tensor QAE ABC.

    Per layer it delegates to ``QuantizationStrategy.embed(residual_tensor,
    positions, bits)`` — the ABC's own code, margins and all — so nothing
    is re-implemented here. The adapter only does what ``BaseEmbedder``
    does for every other strategy: walk layers in sorted order, slice the
    global bit stream, record what was actually written, return an
    ``EmbeddingResult``.

    The per-tensor class's argument order is (tensor, positions, bits),
    which differs from both the ABC declaration and intuition; it is
    matched positionally below on purpose.
    """

    def __init__(self, config):
        from src.embedding.strategies.quantization_strategy import (
            QuantizationStrategy,
        )

        self.config = config
        self._inner = QuantizationStrategy()

    def embed(self, residuals, bits, selector_indices):
        embedded: Dict[int, torch.Tensor] = {}
        actual: Dict[int, List[int]] = {}
        bit_idx = 0

        for layer_id in sorted(residuals.keys()):
            indices = selector_indices.get(layer_id, [])
            n = min(len(indices), len(bits) - bit_idx)

            if n <= 0:
                embedded[layer_id] = residuals[layer_id].clone()
                actual[layer_id] = []
                continue

            embedded[layer_id] = self._inner.embed(
                residuals[layer_id],
                indices[:n],
                bits[bit_idx:bit_idx + n],
            )
            actual[layer_id] = list(indices[:n])
            bit_idx += n

        total_bits = len(bits)
        return EmbeddingResult(
            success=True,
            embedded_weights=embedded,
            carrier_indices=actual,
            layer_allocation={
                lid: len(idx) for lid, idx in actual.items()
            },
            bits_embedded=bit_idx,
            total_bits=total_bits,
            efficiency=(bit_idx / total_bits if total_bits else 0.0),
            metadata={
                "strategy": "qae",
                "adapter": "QaeDictAdapter",
                "margin_scale": getattr(self._inner, "margin_scale", None),
            },
        )


def _qae(config):
    return QaeDictAdapter(config)


def _nf4_qae(config):
    raise RuntimeError(
        "nf4_qae is BLOCKED: its reference residual comes from "
        "ReferenceBuilder.build(fp16_weight, nf4_weight), but "
        "strategy.embed only receives (residuals, bits, "
        "selector_indices), EmbeddingConfig carries no weights or "
        "model id, and no caller passes IntelligentEmbedder.embed's "
        "optional weight arguments — the reference depends on the "
        "absolute weights and cannot be rebuilt from cached "
        "residuals. See StrategySpec.notes for the full record."
    )


REGISTRY: Dict[str, StrategySpec] = {
    "sign": StrategySpec(
        name="sign",
        factory=_sign_v2,
        module="src.embedding.sign_strategy_v2",
        class_name="SignEmbeddingStrategy",
        forces_sign_flip=True,
        extract_needs_cover=False,
        status="READY",
        notes=(
            "Production default. Writes carrier as +/-|r| selected by the "
            "bit. Passes BER/PPL/robustness but is detected at 70.5% by a "
            "carrier-centered MLP; the sweep in "
            "exp7_neural_parameter_study shows retuning does not help."
        ),
    ),
    "magnitude_aware": StrategySpec(
        name="magnitude_aware",
        factory=_magnitude_aware,
        module="src.embedding.strategies.magnitude_aware_strategy",
        class_name="MagnitudeAwareStrategy",
        forces_sign_flip=True,
        extract_needs_cover=False,
        status="READY",
        notes=(
            "Per-carrier adaptive margin, but still "
            "'boosted if bit == 1 else -boosted'. Same structural sign "
            "leak as sign, with a larger perturbation, so it is expected "
            "to be at least as detectable rather than less."
        ),
    ),
    "lwe": StrategySpec(
        name="lwe",
        factory=_lwe,
        module="src.embedding.strategies.lwe_strategy",
        class_name="LWEStrategy",
        forces_sign_flip=False,
        extract_needs_cover=False,
        status="READY",
        notes=(
            "LWE-inspired parity/grid encoding, not lattice LWE: there is "
            "no matrix A and no SIS/LWE instance, so the post-quantum "
            "claim in its docstring is not supported by the "
            "implementation. Phase 3 removed the cover dependency: the "
            "grid width is derived from the stego tensor's own std, which "
            "is valid because the embedding is sparse (~0.001% of "
            "values). Requires LweParityExtractor rather than the "
            "sign-based one production uses."
        ),
    ),
    "neural": StrategySpec(
        name="neural",
        factory=_neural,
        module="src.embedding.strategies.neural_strategy",
        class_name="NeuralStrategy",
        forces_sign_flip=False,
        extract_needs_cover=False,
        needs_trained_model=True,
        status="NEEDS_TRAINING",
        notes=(
            "Learns a (residual, bit) -> value mapping, so it is not "
            "constrained to a sign flip and is the only candidate that "
            "could learn a cover-matching distribution. Requires "
            "NeuralEmbeddingTrainer training before use."
        ),
    ),
    # -----------------------------------------------------------------
    # W1.1 — the two quantization-aware strategies that implement the
    # per-tensor EmbeddingStrategy ABC rather than production's dict
    # contract. The first gets the adapter; the second's adapter is
    # blocked on what the contract does not carry (recorded, not
    # hacked around).
    # -----------------------------------------------------------------
    "qae": StrategySpec(
        name="qae",
        factory=_qae,
        module="src.embedding.strategies.quantization_strategy",
        class_name="QuantizationStrategy (via QaeDictAdapter here)",
        forces_sign_flip=True,
        extract_needs_cover=False,
        status="READY",
        notes=(
            "Quantization-aware embedding, dict-contract adapter in "
            "this module (QaeDictAdapter): per layer it delegates to "
            "the per-tensor ABC's own embed(), so the strategy's "
            "margin logic runs unmodified. Reading the class: it "
            "writes +max(|r|, 0.25*std) / -max(|r|, 0.25*std) — that "
            "is SIGN-FAMILY encoding with a margin floor, and the "
            "'stays inside the NF4 bucket' property the plan "
            "attributes to QAE lives in NF4QuantizationStrategy, not "
            "here. Extraction is sign-based, which is the correct "
            "counterpart (same polarity as sign)."
        ),
    ),
    "nf4_qae": StrategySpec(
        name="nf4_qae",
        factory=_nf4_qae,
        module="src.embedding.strategies.nf4_quantization_strategy",
        class_name="NF4QuantizationStrategy (adapter BLOCKED)",
        forces_sign_flip=False,
        extract_needs_cover=True,
        status="BLOCKED",
        notes=(
            "Encodes bit = reference_residual +/- 0.25*std, where the "
            "reference residual comes from ReferenceBuilder.build("
            "fp16_weight, nf4_weight) — one extra NF4 quantize/"
            "dequantize cycle. Blocked on the contract, not on "
            "implementation: strategy.embed receives only (residuals, "
            "bits, selector_indices), EmbeddingConfig carries no "
            "model id or weights, and no caller passes "
            "IntelligentEmbedder.embed's optional fp16_weights/"
            "quantized_weights (verified by grep). The reference "
            "depends on the ABSOLUTE weights, so it cannot be "
            "rebuilt from cached residuals alone. Extraction would "
            "also need the same reference. Wiring this honestly "
            "requires extending the shared embed contract for every "
            "strategy — an author decision, recorded here rather "
            "than bolted onto one experiment."
        ),
    ),
}

DEFAULT_STRATEGY = "sign"


def available() -> List[str]:
    return sorted(REGISTRY)


def spec(name: str) -> StrategySpec:
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown strategy {name!r}. Available: {available()}"
        )
    return REGISTRY[name]


def build(config: EmbeddingConfig, name: Optional[str] = None):
    """Instantiate a strategy by name, defaulting to the config's choice."""
    name = name or config.embedding_strategy or DEFAULT_STRATEGY
    strategy = spec(name).factory(config)

    # Tag with the registry name so extraction can be dispatched to the
    # extractor that matches this scheme's encoding. Without it a parity
    # scheme would be handed to the sign-based extractor.
    try:
        strategy._registry_name = name
    except Exception:
        pass

    return strategy


def structural_report() -> List[Dict[str, Any]]:
    """The per-strategy viability table, as data."""
    return [
        {
            "strategy": s.name,
            "status": s.status,
            "forces_sign_flip": s.forces_sign_flip,
            "extract_needs_cover": s.extract_needs_cover,
            "needs_trained_model": s.needs_trained_model,
            "class": f"{s.module}.{s.class_name}",
            "notes": s.notes,
        }
        for s in REGISTRY.values()
    ]


# ---------------------------------------------------------------------
# Uniform extraction
# ---------------------------------------------------------------------


def embed_with(
    strategy,
    residuals: Dict[int, torch.Tensor],
    bits: List[int],
    carrier_indices: Dict[int, List[int]],
) -> EmbeddingResult:
    """Embed through any registered strategy, returning EmbeddingResult."""
    result = strategy.embed(residuals, bits, carrier_indices)

    if not isinstance(result, EmbeddingResult):
        raise TypeError(
            f"{type(strategy).__name__}.embed returned "
            f"{type(result).__name__}, expected EmbeddingResult"
        )

    return result


def extract_with(
    strategy,
    embedded: Dict[int, torch.Tensor],
    carrier_indices: Dict[int, List[int]],
    residuals_ref: Optional[Dict[int, torch.Tensor]] = None,
    strategy_name: Optional[str] = None,
) -> List[int]:
    """Extract through any registered strategy.

    ``residuals_ref`` is passed only when the strategy genuinely needs
    it. Dispatch goes through ``embedding.extractors`` so a parity/grid
    scheme is decoded by a parity extractor rather than the sign-based
    one production uses, which would read the wrong quantity and return
    noise.
    """
    name = strategy_name or getattr(strategy, "_registry_name", None)

    if name:
        from src.embedding.extractors import extract_bits

        return extract_bits(
            name, strategy, embedded, carrier_indices, residuals_ref
        )

    extractor = getattr(strategy, "extract", None)

    if extractor is None:
        # BaseEmbedder subclasses provide embed() only. Production reads
        # bits with SignExtractor (DecryptPipeline uses it), so for a
        # sign-flip scheme that is the correct counterpart. Delegating
        # keeps one extraction definition instead of giving each strategy
        # a private copy that could drift from production.
        from src.extraction.sign_extractor import SignExtractor

        return SignExtractor().extract(embedded, carrier_indices)

    signature = inspect.signature(extractor)
    accepts_cover = "residuals_ref" in signature.parameters

    if accepts_cover and residuals_ref is None:
        raise CoverRequired(
            f"{type(strategy).__name__}.extract requires residuals_ref "
            "(the original cover residuals). A real extractor only holds "
            "the stego weights, so this strategy cannot be used as a "
            "hiding scheme as implemented."
        )

    if accepts_cover:
        return extractor(
            embedded, carrier_indices, residuals_ref
        )

    return extractor(embedded, carrier_indices)


def spec_for(name: str) -> Optional[StrategySpec]:
    return REGISTRY.get(name)


class CoverRequired(RuntimeError):
    """Raised when extraction cannot proceed without the cover."""


def probe_extraction(
    name: str,
    residuals: Dict[int, torch.Tensor],
    bits: List[int],
    carrier_indices: Dict[int, List[int]],
) -> Dict[str, Any]:
    """Try a full embed -> extract round trip WITHOUT the cover.

    This is the check that matters for viability: a strategy that passes
    is one that can actually extract from stego weights, which is the
    situation a real extractor is in.
    """
    spec_ = spec(name)
    report: Dict[str, Any] = {
        "strategy": name,
        "embed_ok": False,
        "extract_without_cover_ok": False,
        "recovered_bits": 0,
        "bit_errors": None,
        "ber": None,
        "error": "",
    }

    try:
        config = EmbeddingConfig(
            total_payload_bits=len(bits),
            embedding_strategy=name,
            model_family="qwen",
            num_hidden_layers=len(residuals),
        )
        strategy = build(config, name)

        result = embed_with(strategy, residuals, bits, carrier_indices)
        report["embed_ok"] = True
        report["bits_embedded"] = result.bits_embedded
        report["carriers_used"] = sum(
            len(v) for v in result.carrier_indices.values()
        )

        recovered = extract_with(
            strategy,
            result.embedded_weights,
            result.carrier_indices,
            residuals_ref=None,  # deliberately withheld
        )

        report["extract_without_cover_ok"] = True
        report["recovered_bits"] = len(recovered)

        compared = min(len(bits), len(recovered))
        errors = sum(
            1
            for a, b in zip(bits[:compared], recovered[:compared])
            if a != b
        )
        report["bit_errors"] = errors
        report["bits_compared"] = compared
        report["ber"] = errors / compared if compared else None

    except CoverRequired as exc:
        report["error"] = f"CoverRequired: {exc}"

    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    report["structurally_usable"] = (
        report["embed_ok"]
        and report["extract_without_cover_ok"]
        and (report["ber"] == 0.0)
    )

    spec_.probe = report
    return report


def probe_all(
    residuals: Dict[int, torch.Tensor],
    bits: List[int],
    carrier_indices: Dict[int, List[int]],
) -> Dict[str, Dict[str, Any]]:
    return {
        name: probe_extraction(name, residuals, bits, carrier_indices)
        for name in REGISTRY
    }