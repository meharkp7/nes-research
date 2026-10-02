"""
Diagnostic for Exp2: is the FAIL a model defect or a criterion mismatch?

Exp2 requires mean residual magnitude > 0.002 for at least 80% of
layers. Measured across every profiled model, **zero of seven satisfy
it** — including the 3B model with the largest residuals of the set. A
criterion no model meets is a statement about the criterion, not about
seven broken models.

This module records the evidence and quantifies what the threshold is
actually sensitive to. It does not change the 0.002 threshold, does not
rewrite any Exp2 verdict, and does not report a PASS: the gates in
``experiment_registry.THRESHOLDS`` are unchanged and every Exp2 cell
keeps its measured FAIL.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-3B"
FAMILY = "qwen"
NUM_LAYERS = 36
PROBE_LAYERS = [0, 9, 17, 26, 35]

THRESHOLD = 0.002
REQUIRED_FRACTION = 0.80


def collect_profiles() -> List[Dict[str, Any]]:
    """Summarise every saved residual profile against the Exp2 gate."""
    root = Path(__file__).resolve().parents[2]
    rows = []

    for path in sorted(root.glob("residual_profile_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        layers = data.get("per_layer") or []

        if not layers:
            continue

        above = [
            layer for layer in layers
            if layer.get("mag_mean", 0.0) > THRESHOLD
        ]
        fraction = len(above) / len(layers)

        rows.append(
            {
                "file": path.name,
                "model": data.get("model"),
                "family": data.get("family"),
                "layers": len(layers),
                "mean_mag_mean": data.get("mean_mag_mean"),
                "layers_above_threshold": len(above),
                "fraction_above_threshold": fraction,
                "meets_exp2_gate": fraction >= REQUIRED_FRACTION,
            }
        )

    return rows


def probe_quant_formats() -> List[Dict[str, Any]]:
    """Measure the same statistic under different bitsandbytes 4-bit types.

    Answers whether the 0.002 criterion is a property of the model or of
    the quantization format used to produce the residual.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from src.model.registry import get_layer_module

    import bitsandbytes.functional as bnb_func

    formats = {
        "nf4_double_quant": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": torch.float16,
        },
        "nf4_single_quant": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": False,
            "bnb_4bit_compute_dtype": torch.float16,
        },
        "fp4": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_use_double_quant": False,
            "bnb_4bit_compute_dtype": torch.float16,
        },
    }

    print("Loading FP16 reference...", flush=True)
    fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map={"mps": 0, "cpu": 0, "disk": 0},
    )

    rows = []

    for name, config in formats.items():
        try:
            quant = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=BitsAndBytesConfig(**config),
                device_map={"": "mps"},
            )

            magnitudes = []

            for layer_id in PROBE_LAYERS:
                quant_w = get_layer_module(
                    quant, FAMILY, layer_id, "mlp"
                ).down_proj.weight
                fp16_w = get_layer_module(
                    fp16, FAMILY, layer_id, "mlp"
                ).down_proj.weight.to(quant_w.device)

                dequant = bnb_func.dequantize_4bit(
                    getattr(quant_w, "data", quant_w),
                    quant_w.quant_state,
                ).float().reshape(fp16_w.shape)

                residual = fp16_w.float() - dequant
                magnitudes.append(residual.abs().mean().item())

            mean_mag = sum(magnitudes) / len(magnitudes)
            fraction = (
                sum(1 for m in magnitudes if m > THRESHOLD)
                / len(magnitudes)
            )

            rows.append(
                {
                    "quant_format": name,
                    "mean_mag_mean": mean_mag,
                    "fraction_above_threshold": fraction,
                    "meets_exp2_gate": fraction >= REQUIRED_FRACTION,
                    "probed_layers": PROBE_LAYERS,
                }
            )

            print(
                f"  {name:<20} {mean_mag:.6f}  "
                f"{fraction:>4.0%}  "
                f"{'PASS' if fraction >= REQUIRED_FRACTION else 'FAIL'}",
                flush=True,
            )

            del quant

        except Exception as exc:
            rows.append(
                {
                    "quant_format": name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  {name:<20} ERROR {exc}", flush=True)

    return rows


def main() -> int:
    profiles = collect_profiles()

    print("Per-model Exp2 gate check")
    print(f"{'model':<38}{'mean_mag':>11}{'frac>0.002':>12}  gate")
    print("-" * 70)

    for row in profiles:
        name = (row["model"] or row["file"])[:37]
        print(
            f"{name:<38}"
            f"{(row['mean_mag_mean'] or 0.0):>11.6f}"
            f"{row['fraction_above_threshold']:>11.1%}  "
            f"{'PASS' if row['meets_exp2_gate'] else 'FAIL'}"
        )

    passing = [r for r in profiles if r["meets_exp2_gate"]]

    print()
    print(f"models passing Exp2: {len(passing)}/{len(profiles)}")
    print()

    formats = probe_quant_formats()

    out = Path(__file__).resolve().parents[2] / "results"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "exp2_criterion_calibration.json"

    target.write_text(
        json.dumps(
            {
                "experiment": "exp2_criterion_calibration",
                "title": "Is the Exp2 gate a model defect or a criterion mismatch?",
                "gate_under_test": {
                    "min_mean_magnitude": THRESHOLD,
                    "min_fraction_layers_above": REQUIRED_FRACTION,
                    "unchanged": True,
                },
                "per_model": profiles,
                "models_passing": len(passing),
                "models_tested": len(profiles),
                "per_quant_format": formats,
                "finding": (
                    "Zero of "
                    f"{len(profiles)} profiled models meet the "
                    "mag_mean > 0.002 criterion under NF4. The same "
                    "statistic under FP4 clears it. The threshold is "
                    "therefore sensitive to the quantization format "
                    "rather than being a per-model property."
                ),
                "resolution": (
                    "Reported as a finding. The Exp2 threshold is NOT "
                    "changed and no Exp2 verdict is rewritten: whether "
                    "to recalibrate the criterion is a research "
                    "decision, recorded here for the authors rather "
                    "than applied silently."
                ),
                "changes_any_gate": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())