"""
Experiment 8 — Real Cross-Model Ablation Table

Purpose
-------
Build the unified cross-model table from the REAL NF4 residual pipeline.

Important architectural rule
-----------------------------
This file intentionally does NOT modify or reuse NESBenchmark.run_all(),
because NESBenchmark uses synthetic residuals. Exp8 is an orchestration
layer around the existing real-model components.

Gate mapping used here:
    G2  QACI allocation
    G3  Clean BER / round-trip
    G4  Real WikiText-2 PPL delta
    G5  BER @ sigma=0.001 / 0.002
    G6  Statistical KL + neural detector (when a real detector result exists)

A gate is reported as NOT_RUN when its required experiment artifact is not
available. NOT_RUN is never converted into PASS.

This script is intentionally conservative: it does not alter the production
embedding, model loader, benchmark suite, or any previous experiment.
"""

import csv
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# Must be set before importing torch/model code.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

from src.core.types import EmbeddingConfig
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.extraction.decrypt_pipeline import DecryptPipeline
from src.carrier_intelligence.qaci_pipeline import QACIPipeline
from src.evaluation.fidelity_validator import FidelityValidator
from src.evaluation.robustness_validator import RobustnessValidator
from src.steganalysis.security_validator import SecurityValidator
from src.model.model_loader import load_model_pair, extract_residuals
from src.evaluation.exp5_model_builder import build_embedded_eval_model


MODELS = [
    ("meta-llama/Llama-3.1-8B", "llama", 32),
    ("mistralai/Mistral-7B-v0.3", "mistral", 32),
    ("google/gemma-2-9b", "gemma", 42),
    ("Qwen/Qwen2.5-7B", "qwen", 28),
    ("Qwen/Qwen2.5-3B", "qwen", 36),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama", 22),
    ("microsoft/Phi-3-mini-4k-instruct", "phi3", 32),
]

PAYLOAD_BITS = 50_000
ROBUSTNESS_BITS = 10_000
ROBUSTNESS_SIGMAS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.020]
MESSAGE = "A" * 6000
ROBUSTNESS_MESSAGE = "A" * 1250

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = RESULT_DIR / "cross_model_table_real.json"
CSV_PATH = RESULT_DIR / "cross_model_table_real.csv"


def _gate(status: str, **values: Any) -> Dict[str, Any]:
    return {"status": status, **values}


def run_g2_qaci(residuals: Dict[int, torch.Tensor], n_layers: int) -> Dict[str, Any]:
    pipeline = QACIPipeline(total_layers=n_layers)
    selection = pipeline.select(residuals, PAYLOAD_BITS)
    allocated = sum(selection.layer_allocation.values())
    return _gate(
        "PASS" if allocated == PAYLOAD_BITS else "FAIL",
        total_bits_allocated=allocated,
        expected_bits=PAYLOAD_BITS,
        layers_selected=len(selection.layer_allocation),
    )


def make_embedding(residuals, family: str, n_layers: int, payload_bits: int, message: str):
    config = EmbeddingConfig(
        total_payload_bits=payload_bits,
        model_family=family,
        num_hidden_layers=n_layers,
    )
    return IntelligentEmbedder(config).embed(message, residuals)


def run_g3_ber(residuals, family: str, n_layers: int) -> Dict[str, Any]:
    result = make_embedding(
        residuals, family, n_layers, PAYLOAD_BITS, MESSAGE
    )
    recovered, stats = DecryptPipeline(key=result.key).run(
        result.embedded_residuals,
        result.carrier_indices,
    )
    ok = stats.get("success", False) and recovered == MESSAGE
    return _gate(
        "PASS" if ok else "FAIL",
        ber=0.0 if ok else 1.0,
        bits_embedded=len(result.embedded_bits),
        expected_bits=PAYLOAD_BITS,
        recovered_matches=bool(recovered == MESSAGE),
    )


def run_g4_ppl(nf4, fp16, tok, residuals, family: str, n_layers: int) -> Dict[str, Any]:
    """
    Real PPL gate for Exp8.

    The original residual reconstruction control is measured first, then a
    fresh model pair is used for the embedded evaluation. This prevents the
    control/embedding comparison from being contaminated by in-place model
    mutation.
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="validation"
        )
        texts = [t for t in dataset["text"] if len(t.strip()) > 50][:200]
        validator = FidelityValidator(max_ppl_degradation=0.02)

        ppl_base = validator.validate_perplexity(nf4, tok, texts)

        # Reconstruction control: W_fp16 ≈ W_nf4 + R_original.
        control_model = build_embedded_eval_model(
            nf4, fp16, residuals, family
        )
        ppl_control = validator.validate_perplexity(
            control_model, tok, texts
        )
        del control_model

        # Build the embedding from the already extracted residuals, then
        # reconstruct a separate evaluation model with the embedded residuals.
        embed_result = make_embedding(
            residuals, family, n_layers, PAYLOAD_BITS, MESSAGE
        )
        embedded_model = build_embedded_eval_model(
            nf4, fp16, embed_result.embedded_residuals, family
        )
        ppl_embedded = validator.validate_perplexity(
            embedded_model, tok, texts
        )
        del embedded_model

        fidelity = validator.compare_perplexity(
            ppl_base, ppl_embedded
        )

        return _gate(
            "PASS" if fidelity.passed else "FAIL",
            baseline_ppl=ppl_base,
            reconstruction_control_ppl=ppl_control,
            embedded_ppl=ppl_embedded,
            ppl_degradation=fidelity.ppl_degradation,
            reconstruction_control_delta=(ppl_control - ppl_base) / max(ppl_base, 1e-8),
            evaluation_samples=len(texts),
        )
    except Exception as exc:
        return _gate(
            "NOT_RUN",
            reason=f"Real PPL gate unavailable: {type(exc).__name__}: {exc}",
        )


def run_g5_robustness(residuals, family: str, n_layers: int) -> Dict[str, Any]:
    result = make_embedding(
        residuals, family, n_layers, ROBUSTNESS_BITS, ROBUSTNESS_MESSAGE
    )

    validator = RobustnessValidator(
        max_ber_at_001=0.02,
        max_ber_at_002=0.10,
        num_trials=5,
    )
    outcome = validator.validate(
        result.embedded_residuals,
        result.carrier_indices,
        result.embedded_bits,
        ROBUSTNESS_SIGMAS,
    )

    return _gate(
        "PASS" if outcome.passed else "FAIL",
        ber_curve={str(k): v for k, v in outcome.ber_curve.items()},
        ber_at_001=outcome.ber_at_001,
        ber_at_002=outcome.ber_at_002,
        total_bits=outcome.total_bits,
        trials=outcome.num_trials,
    )


def run_g6_statistical(residuals, family: str, n_layers: int) -> Dict[str, Any]:
    result = make_embedding(
        residuals, family, n_layers, PAYLOAD_BITS, MESSAGE
    )
    validator = SecurityValidator(
        max_kl_divergence=0.05,
        max_detector_accuracy=0.55,
    )
    outcome = validator.validate(
        original_residuals=residuals,
        embedded_residuals=result.embedded_residuals,
        carrier_indices=result.carrier_indices,
    )

    # This is deliberately the statistical gate only. The neural detector
    # remains a separate sub-gate because its dataset is an experiment
    # artifact and is not regenerated implicitly here.
    return _gate(
        "PASS" if outcome.kl_divergence <= 0.05 else "FAIL",
        kl_divergence=outcome.kl_divergence,
        sign_bias=outcome.sign_bias,
        mean_shift=outcome.moment_shift["mean_shift"],
        std_shift=outcome.moment_shift["std_shift"],
        statistical_detector_accuracy=outcome.detector_accuracy,
    )


def find_neural_detector_result(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Read an already-produced real detector result if one exists.

    Exp8 never trains a detector implicitly. This avoids generating the heavy
    detector PKLs as a side effect of a cross-model table run.
    """
    candidates = [
        RESULT_DIR / "exp7_neural_detector_results.json",
        RESULT_DIR / "exp7_detector_results.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            item = data.get(model_id)
            if isinstance(item, dict) and "accuracy" in item:
                return item
    return None


def combine_gate_statuses(row: Dict[str, Any]) -> str:
    statuses = []
    for key in ("g2_qaci", "g3_ber", "g4_ppl", "g5_robustness", "g6_statistical"):
        statuses.append(row[key]["status"])

    detector = row["g6_neural_detector"]["status"]
    statuses.append(detector)

    if "FAIL" in statuses:
        return "FAIL"
    if "NOT_RUN" in statuses:
        return "INCOMPLETE"
    return "PASS"


def run_model(model_id: str, family: str, expected_layers: int) -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print(f"MODEL: {model_id}")
    print(f"FAMILY: {family} | EXPECTED LAYERS: {expected_layers}")
    print("=" * 78)

    row: Dict[str, Any] = {
        "model_id": model_id,
        "family": family,
        "expected_layers": expected_layers,
    }

    nf4 = fp16 = tok = None
    try:
        nf4, fp16, tok = load_model_pair(model_id)
        actual_layers = len(nf4.model.layers)
        row["actual_layers"] = actual_layers

        residuals = extract_residuals(
            nf4_model=nf4,
            fp16_model=fp16,
            family=family,
        )
        row["residual_layers"] = len(residuals)

        row["g2_qaci"] = run_g2_qaci(residuals, actual_layers)
        row["g3_ber"] = run_g3_ber(residuals, family, actual_layers)
        row["g4_ppl"] = run_g4_ppl(
            nf4, fp16, tok, residuals, family, actual_layers
        )
        row["g5_robustness"] = run_g5_robustness(
            residuals, family, actual_layers
        )
        row["g6_statistical"] = run_g6_statistical(
            residuals, family, actual_layers
        )

        detector = find_neural_detector_result(model_id)
        if detector is None:
            row["g6_neural_detector"] = _gate(
                "NOT_RUN",
                reason="No saved real neural-detector result supplied to Exp8.",
            )
        else:
            accuracy = float(detector["accuracy"])
            row["g6_neural_detector"] = _gate(
                "PASS" if accuracy <= 0.55 else "FAIL",
                accuracy=accuracy,
            )

        row["overall_status"] = combine_gate_statuses(row)
        return row

    except Exception as exc:
        row["overall_status"] = "FAILED_TO_RUN"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
        return row
    finally:
        # Exp8 is deliberately model-by-model so the next model does not
        # inherit large references from the previous one.
        del nf4, fp16, tok
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def write_outputs(results: Dict[str, Dict[str, Any]]) -> None:
    JSON_PATH.write_text(json.dumps(results, indent=2))

    columns = [
        "model_id", "family", "expected_layers", "actual_layers",
        "overall_status",
        "g2_qaci", "g3_ber", "g4_ppl", "g5_robustness",
        "g6_statistical", "g6_neural_detector",
    ]

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in results.values():
            writer.writerow({
                "model_id": row.get("model_id"),
                "family": row.get("family"),
                "expected_layers": row.get("expected_layers"),
                "actual_layers": row.get("actual_layers"),
                "overall_status": row.get("overall_status"),
                "g2_qaci": row.get("g2_qaci", {}).get("status"),
                "g3_ber": row.get("g3_ber", {}).get("status"),
                "g4_ppl": row.get("g4_ppl", {}).get("status"),
                "g5_robustness": row.get("g5_robustness", {}).get("status"),
                "g6_statistical": row.get("g6_statistical", {}).get("status"),
                "g6_neural_detector": row.get("g6_neural_detector", {}).get("status"),
            })


def main() -> int:
    print("\nEXPERIMENT 8 — REAL CROSS-MODEL TABLE")
    print("This does not modify NESBenchmark or the production embedding path.")

    results: Dict[str, Dict[str, Any]] = {}
    for model_id, family, n_layers in MODELS:
        results[model_id] = run_model(model_id, family, n_layers)

    write_outputs(results)

    print("\n" + "=" * 78)
    print("CROSS-MODEL SUMMARY")
    print("=" * 78)
    for model_id, row in results.items():
        print(f"{model_id}: {row.get('overall_status')}")

    print(f"\nJSON: {JSON_PATH}")
    print(f"CSV : {CSV_PATH}")

    # INCOMPLETE is intentionally a non-zero exit for CI/paper automation:
    # it means the table is not yet a complete six-gate result.
    return 0 if all(r.get("overall_status") == "PASS" for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
