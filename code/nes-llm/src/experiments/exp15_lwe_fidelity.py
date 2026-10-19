"""
W2 — LWE fidelity (three-way perplexity).

RESEARCH_PLAN §3 W2: "LWE perplexity is unmeasured anywhere.
Undetectability is worthless if the model is damaged." The prior is a
perturbation table (LWE moves weights 9.4x less than sign, 2.7x the
residual scale vs 25.6x) — a prior, not a result. Expect is not a
measurement.

Protocol is exp5's, structurally identical:

    1. NF4 baseline PPL
    2. reconstruction control  (W_NF4 + original residual)
    3. embedded PPL            (W_NF4 + embedded residual)

Only (2) -> (3) is attributable to the payload; reconstruction alone
moves PPL substantially on its own (exp5 measured -9.0% baseline ->
control on Qwen2.5-3B while the embedding moved it -0.0053%), so the
reconstruction delta is reported separately and never folded in.

Parity with exp5, deliberately:

    - FidelityValidator.validate_perplexity, unmodified;
    - wikitext-2 validation, 200 texts, min length 50,
      max_length 512, batch 4;
    - payload 50,000 bits, message "A" x 6000;
    - gate number 2% — THRESHOLDS['exp15'] carries exp5's own
      threshold, reused rather than invented;
    - model pair and residuals via ModelContext, as exp5 loads them.

What exp5 did not have:

    - **sign re-verification (Qwen2.5-3B only).** The plan asks to
      re-verify sign's recorded number. The recorded artifact lives in
      exp8_result_adapter.EXP5_RESULTS; this module re-runs the sign
      arm through the same path and reports both side by side rather
      than trusting the record silently.
    - **one process per model** (the suite's memory rule): run with
      --model <id>. One artifact per model:
      results/exp15_lwe_fidelity_<slug>.json.

Second model note: Phi-3-mini was the first choice (in the grid,
cache complete) but its model pair no longer constructs under
transformers 5.16.1 — the remote modeling code reads
config.rope_scaling["type"] and the checkpoint's rope_scaling has no
"type" (KeyError in _init_rope). Patching the config would change
RoPE behaviour and silently corrupt the PPL, so the loader was left
alone. gemma-2-2b (complete 26-layer cache, constructs cleanly) is
the second model instead. A record, not a workaround.

This is a fidelity measurement for the LWE strategy only; it says
nothing about LWE's (already refuted, exp13) key-gating or its
detectability.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.core.types import EmbeddingConfig  # noqa: E402
from src.embedding.intelligent_embedder import IntelligentEmbedder  # noqa: E402
from src.evaluation.exp5_model_builder import (  # noqa: E402
    build_embedded_eval_model,
)
from src.evaluation.fidelity_validator import FidelityValidator  # noqa: E402
from src.experiments.datasets import load_texts  # noqa: E402
from src.experiments.experiment_registry import gate_for  # noqa: E402
from src.experiments.experiments.exp3_clean_ber import (  # noqa: E402
    embed_payload,
)
from src.experiments.experiments.exp5_fidelity import (  # noqa: E402
    _measure_distortion,
)
from src.experiments.model_context import make_context  # noqa: E402
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import load_cached_residuals  # noqa: E402

EXPERIMENT = "exp15"

PAYLOAD_BITS = 50_000
MESSAGE = "A" * 6_000

NUM_TEXTS = 200
MIN_TEXT_LENGTH = 50

# exp5's own arm, re-run on models where a recorded exp5 number
# exists, as the plan's W2 "re-verify sign's number reproduces".
SIGN_REVERIFY_MODELS = {"Qwen/Qwen2.5-3B"}

# Models with a complete residual cache that are not in
# experiment_registry.TARGET_MODELS (the registry lists the 7-model
# grid; this experiment's second model is outside it). Only used when
# get_model raises — registry models keep make_context as the single
# source of family/layers.
LOCAL_CONTEXTS = {
    "google/gemma-2-2b": {"family": "gemma", "expected_layers": 26},
}


def _context_for(model_id: str):
    from src.experiments.experiment_registry import get_model
    from src.experiments.model_context import ModelContext

    try:
        get_model(model_id)
    except KeyError:
        spec = LOCAL_CONTEXTS.get(model_id)
        if spec is None:
            raise
        return ModelContext(
            model_id=model_id,
            family=spec["family"],
            expected_layers=int(spec["expected_layers"]),
        )
    return make_context(model_id)


def log(message: str) -> None:
    """Flush every stage — a long MPS run must never look hung (§3.7)."""
    print(message, flush=True)


def _slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace("-", "_").lower()


def _embed(context, strategy: str, payload_bits: int, message: str):
    """embed_payload plus a strategy argument.

    Otherwise identical to exp3's shared production path: same
    EmbeddingConfig fields, same IntelligentEmbedder.embed call.
    """
    config = EmbeddingConfig(
        total_payload_bits=payload_bits,
        embedding_strategy=strategy,
        model_family=context.family,
        num_hidden_layers=(
            context.actual_layers or context.expected_layers
        ),
    )
    embedder = IntelligentEmbedder(config)
    return embedder.embed(message, context.residuals)


def run_model(model_id: str) -> Dict[str, Any]:
    gate = gate_for(EXPERIMENT)
    threshold_pct = float(gate["max_ppl_degradation_pct"])

    context = _context_for(model_id)
    family = context.family

    log(f"[exp15] {model_id} ({family}) — loading models ...")
    context.ensure_models()

    log("[exp15] loading residuals from cache ...")
    context.residuals = load_cached_residuals(
        model_id, context.expected_layers
    )

    nf4_model, fp16_model, tokenizer = context.models

    texts, dataset_error = load_texts(
        num_texts=NUM_TEXTS,
        min_length=MIN_TEXT_LENGTH,
    )
    if not texts:
        return {
            "experiment": EXPERIMENT,
            "title": "W2 — LWE fidelity (three-way PPL)",
            "model_id": model_id,
            "status": "NOT_RUN",
            "gate": {"status": "NOT_RUN"},
            "notes": (
                "WikiText-2 validation split unavailable; PPL cannot "
                f"be measured ({dataset_error})."
            ),
        }

    validator = FidelityValidator(
        max_ppl_degradation=threshold_pct / 100.0
    )

    # --- 1. NF4 baseline -------------------------------------------
    log("[exp15] 1/4 NF4 baseline PPL ...")
    ppl_baseline = validator.validate_perplexity(
        nf4_model, tokenizer, texts
    )
    log(f"[exp15]   baseline = {ppl_baseline:.4f}")

    # --- 2. Reconstruction control ---------------------------------
    log("[exp15] 2/4 reconstruction control ...")
    control_model = build_embedded_eval_model(
        nf4_model, fp16_model, context.residuals, family
    )
    ppl_control = validator.validate_perplexity(
        control_model, tokenizer, texts
    )
    log(f"[exp15]   control = {ppl_control:.4f}")

    # --- 3. LWE embedded -------------------------------------------
    log("[exp15] 3/4 LWE embedding + embedded PPL ...")
    lwe_result = _embed(context, "lwe", PAYLOAD_BITS, MESSAGE)
    log(
        f"[exp15]   LWE embedded {lwe_result.total_bits} bits "
        f"over {len(lwe_result.carrier_indices)} layers"
    )
    lwe_model = build_embedded_eval_model(
        nf4_model, fp16_model, lwe_result.embedded_residuals, family
    )
    ppl_lwe = validator.validate_perplexity(lwe_model, tokenizer, texts)
    log(f"[exp15]   LWE embedded = {ppl_lwe:.4f}")

    distortion = _measure_distortion(
        context.residuals, lwe_result.embedded_residuals
    )
    del lwe_result

    # --- 4. sign re-verification (recorded models only) ------------
    sign_block: Optional[Dict[str, Any]] = None
    if model_id in SIGN_REVERIFY_MODELS:
        log("[exp15] 4/4 sign re-verification arm ...")
        sign_result = embed_payload(context, PAYLOAD_BITS, MESSAGE)
        sign_model = build_embedded_eval_model(
            nf4_model, fp16_model,
            sign_result.embedded_residuals, family,
        )
        ppl_sign = validator.validate_perplexity(
            sign_model, tokenizer, texts
        )
        sign_delta = (
            (ppl_sign - ppl_control) / max(ppl_control, 1e-8)
        ) * 100.0

        from src.model.exp8_result_adapter import get_exp5_result

        recorded = get_exp5_result(model_id)
        sign_block = {
            "rerun_embedded_ppl": ppl_sign,
            "rerun_delta_vs_control_pct": sign_delta,
            "recorded": recorded,
            "recorded_delta_pct": (
                recorded.get("ppl_delta_pct") if recorded else None
            ),
            "delta_point_difference": (
                abs(sign_delta - recorded.get("ppl_delta_pct"))
                if recorded else None
            ),
        }
        log(
            f"[exp15]   sign rerun delta = {sign_delta:+.6f}% "
            f"(recorded {recorded.get('ppl_delta_pct') if recorded else 'n/a'})"
        )
        del sign_result

    # --- Attribution (exp5's three deltas) --------------------------
    lwe_delta_pct = (
        (ppl_lwe - ppl_control) / max(ppl_control, 1e-8)
    ) * 100.0
    reconstruction_delta_pct = (
        (ppl_control - ppl_baseline) / max(ppl_baseline, 1e-8)
    ) * 100.0
    absolute_delta_pct = (
        (ppl_lwe - ppl_baseline) / max(ppl_baseline, 1e-8)
    ) * 100.0
    degradation = abs(lwe_delta_pct)
    passed = degradation < threshold_pct

    artifact = {
        "experiment": EXPERIMENT,
        "title": "W2 — LWE fidelity (three-way PPL)",
        "model_id": model_id,
        "family": family,
        "status": "PASS" if passed else "FAIL",
        "gate": {
            "max_ppl_degradation_pct": threshold_pct,
            "measured_lwe_ppl_degradation_pct": degradation,
            "status": "PASS" if passed else "FAIL",
            "gate_source": "experiment_registry.THRESHOLDS['exp15']",
        },
        "configuration": {
            "dataset": "wikitext/wikitext-2-raw-v1 validation",
            "num_texts": len(texts),
            "min_text_length": MIN_TEXT_LENGTH,
            "max_length": 512,
            "batch_size": 4,
            "payload_bits": PAYLOAD_BITS,
            "strategy": "lwe",
            "protocol": (
                "three-way: NF4 baseline, reconstruction control, "
                "embedded (exp5's protocol, unchanged)"
            ),
        },
        "metrics": {
            "nf4_baseline_ppl": ppl_baseline,
            "reconstruction_control_ppl": ppl_control,
            "lwe_embedded_ppl": ppl_lwe,
            "lwe_embedding_specific_delta_pct": lwe_delta_pct,
            "lwe_ppl_degradation_pct": degradation,
            "reconstruction_only_delta_pct": reconstruction_delta_pct,
            "absolute_delta_vs_baseline_pct": absolute_delta_pct,
            "threshold_pct": threshold_pct,
            "eval_texts": len(texts),
            "payload_bits": PAYLOAD_BITS,
            "embedding_distortion": distortion,
            "sign_reverification": sign_block,
        },
        "reproducibility": context.reproducibility(),
        "notes": (
            f"LWE embedding-specific degradation is {degradation:.6f}% "
            f"(threshold {threshold_pct}%). Reconstruction alone moves "
            f"PPL by {reconstruction_delta_pct:+.3f}% relative to the "
            "NF4 baseline; that component is not attributable to any "
            "payload. This measures fidelity of the LWE strategy only — "
            "exp13 refuted its key-gating separately."
        ),
    }

    target = RESULTS_DIR / f"{EXPERIMENT}_lwe_fidelity_{_slug(model_id)}.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    log("=" * 70)
    log(f"baseline={ppl_baseline:.4f} control={ppl_control:.4f} "
        f"LWE={ppl_lwe:.4f}")
    log(f"LWE embedding-specific degradation = {degradation:.6f}% "
        f"(gate < {threshold_pct}%)")
    log(f"status: {artifact['status']}")
    log("=" * 70)
    log(f"wrote {target}")
    return artifact


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="W2 — LWE fidelity")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B",
        help="one model id per process (memory rule)",
    )
    args = parser.parse_args(argv)

    torch.manual_seed(42)
    run_model(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
