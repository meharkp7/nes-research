"""
Experiment 5 — Fidelity (real perplexity).

Three perplexities are measured, and the distinction matters:

    1. NF4 baseline PPL
    2. reconstruction-control PPL  (W_NF4 + original residual)
    3. embedded PPL                (W_NF4 + embedded residual)

Only the difference between (2) and (3) is attributable to embedding.
NF4 -> reconstructed FP16-like weights can move PPL substantially on
their own, and attributing that to the payload would be wrong (§14).

The reconstruction control is therefore mandatory: this module refuses
to report a fidelity number without it.
"""

import os
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.evaluation.exp5_model_builder import build_embedded_eval_model
from src.evaluation.fidelity_validator import FidelityValidator
from src.experiments.experiment_registry import gate_for
from src.experiments.experiments.exp3_clean_ber import embed_payload
from src.experiments.datasets import load_texts
from src.experiments.model_context import ModelContext

EXPERIMENT = "exp5"

PAYLOAD_BITS = 50_000
MESSAGE = "A" * 6_000

NUM_TEXTS = 200
MIN_TEXT_LENGTH = 50


def run(context: ModelContext) -> Dict[str, Any]:
    gate = gate_for(EXPERIMENT)

    if not context.has_models or not context.has_residuals:
        missing = []
        if not context.has_models:
            missing.append("model pair")
        if not context.has_residuals:
            missing.append("residuals")
        return {
            "experiment": EXPERIMENT,
            "title": "Fidelity (real PPL)",
            "configuration": {},
            "metrics": {},
            "thresholds": gate,
            "status": "NOT_RUN",
            "gate_status": "NOT_RUN",
            "reproducibility": context.reproducibility(),
            "notes": (
                "Requires " + " and ".join(missing)
                + "; real PPL cannot be estimated without live models."
            ),
            "source": "run",
        }

    nf4_model, fp16_model, tokenizer = context.models

    texts, dataset_error = load_texts(
        num_texts=NUM_TEXTS,
        min_length=MIN_TEXT_LENGTH,
    )

    if not texts:
        return {
            "experiment": EXPERIMENT,
            "title": "Fidelity (real PPL)",
            "configuration": {},
            "metrics": {},
            "thresholds": gate,
            "status": "NOT_RUN",
            "gate_status": "NOT_RUN",
            "reproducibility": context.reproducibility(),
            "notes": (
                "WikiText-2 validation split unavailable; PPL cannot "
                f"be measured ({dataset_error})."
            ),
            "source": "run",
        }

    validator = FidelityValidator(
        max_ppl_degradation=gate["max_ppl_degradation_pct"] / 100.0
    )

    # --- 1. NF4 baseline -------------------------------------------
    print("  [exp5] NF4 baseline PPL...")
    ppl_baseline = validator.validate_perplexity(
        nf4_model, tokenizer, texts
    )

    # --- 2. Reconstruction control ---------------------------------
    #
    # W_reconstructed = W_NF4 + R_original. No embedding. This isolates
    # the effect of reconstruction from the effect of the payload.
    print("  [exp5] reconstruction-control PPL...")
    control_model = build_embedded_eval_model(
        nf4_model,
        fp16_model,
        context.residuals,
        context.family,
    )
    ppl_control = validator.validate_perplexity(
        control_model, tokenizer, texts
    )
    del control_model

    # --- 3. Embedded model -----------------------------------------
    print("  [exp5] embedded PPL...")
    result = embed_payload(context, PAYLOAD_BITS, MESSAGE)
    embedded_model = build_embedded_eval_model(
        nf4_model,
        fp16_model,
        result.embedded_residuals,
        context.family,
    )
    ppl_embedded = validator.validate_perplexity(
        embedded_model, tokenizer, texts
    )
    del embedded_model

    # --- Attribution ------------------------------------------------
    #
    # embedding-specific delta: control -> embedded.
    embedding_delta_pct = (
        (ppl_embedded - ppl_control) / max(ppl_control, 1e-8)
    ) * 100.0

    # reconstruction effect: baseline -> control. Reported separately so
    # it is never folded into the embedding number.
    reconstruction_delta_pct = (
        (ppl_control - ppl_baseline) / max(ppl_baseline, 1e-8)
    ) * 100.0

    absolute_delta_pct = (
        (ppl_embedded - ppl_baseline) / max(ppl_baseline, 1e-8)
    ) * 100.0

    degradation = abs(embedding_delta_pct)
    passed = degradation < gate["max_ppl_degradation_pct"]

    distortion = _measure_distortion(
        context.residuals, result.embedded_residuals
    )

    metrics: Dict[str, Any] = {
        "nf4_baseline_ppl": ppl_baseline,
        "reconstruction_control_ppl": ppl_control,
        "embedded_ppl": ppl_embedded,
        "embedding_specific_delta_pct": embedding_delta_pct,
        "ppl_degradation_pct": degradation,
        "absolute_delta_vs_baseline_pct": absolute_delta_pct,
        "reconstruction_only_delta_pct": reconstruction_delta_pct,
        "threshold_pct": gate["max_ppl_degradation_pct"],
        "eval_texts": len(texts),
        "payload_bits": PAYLOAD_BITS,
        "embedding_distortion": distortion,
    }

    notes = (
        f"Embedding-specific degradation is {degradation:.6f}% "
        f"(threshold {gate['max_ppl_degradation_pct']}%). "
        f"Reconstruction alone moves PPL by "
        f"{reconstruction_delta_pct:+.3f}% relative to the NF4 "
        "baseline; that component is not attributable to the payload."
    )

    return {
        "experiment": EXPERIMENT,
        "title": "Fidelity (real PPL)",
        "configuration": {
            "dataset": "wikitext/wikitext-2-raw-v1 validation",
            "num_texts": len(texts),
            "min_text_length": MIN_TEXT_LENGTH,
            "max_length": 512,
            "batch_size": 4,
            "payload_bits": PAYLOAD_BITS,
            "protocol": (
                "three-way: NF4 baseline, reconstruction control, "
                "embedded"
            ),
        },
        "metrics": metrics,
        "thresholds": gate,
        "status": "PASS" if passed else "FAIL",
        "gate_status": "PASS" if passed else "FAIL",
        "reproducibility": context.reproducibility(),
        "notes": notes,
        "source": "run",
    }


def _measure_distortion(
    original: Dict[int, Any],
    embedded: Dict[int, Any],
) -> Dict[str, Any]:
    """How sparse and how large the residual edits actually are."""
    total_changed = 0
    total_values = 0
    max_change = 0.0
    sum_change = 0.0

    for layer_id, orig in original.items():
        emb = embedded[layer_id]
        delta = (emb - orig).abs()

        total_values += delta.numel()
        total_changed += int((delta > 0).sum().item())
        max_change = max(max_change, float(delta.max().item()))
        sum_change += float(delta.sum().item())

    return {
        "total_values": total_values,
        "changed_values": total_changed,
        "changed_ratio": (
            total_changed / total_values if total_values else 0.0
        ),
        "max_abs_delta": max_change,
        "mean_abs_delta_over_changed": (
            sum_change / total_changed if total_changed else 0.0
        ),
    }