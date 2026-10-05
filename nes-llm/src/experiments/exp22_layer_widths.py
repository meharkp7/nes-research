"""
W5.4 — per-layer LWE grid width (exp22).

RESEARCH_PLAN §4 W5.4: *"Per-layer strategy selection. Different grid
width per layer, keyed by layer noise. Layers differ: Qwen2.5-7B spans
0.0012–0.0130, Phi-3 spans 0.0026–0.0031."*

Two cells, exp10's three axes each, ONLY `EmbeddingConfig.lwe_width_rule`
varying (exp10's `config_overrides` hook):

    global     the shipped absolute width 0.010 on every layer —
               exp10's own lwe cell under another name, the control.
    per_layer  w_l = clip(4.0 * round(std_l, 4), 0.005, 0.020) —
               width proportional to the layer's own noise (equalizing
               the grid-to-noise ratio across layers), clipped to the
               only window with a measured frontier (exp11: 0.005
               passes sigma=0.001, 0.02 stays undetectable).

Agreement design (the reason for the 4-decimal coarsening): embed
derives the width from the ORIGINAL layer std and the extractor from
the STEGO layer std — two views of the same tensor. Measured on this
model, an LWE embed moves per-layer std by at most 0.0153%, which
buckets identically at 4 decimals (zero flips; gate BER 0.0 would
catch any edge case).

Pre-registered expectations, recorded before the run: both cells
round-trip at exp3's 0.0; the LWE detector has collapsed to exactly
0.50 in every experiment that trained it (exp10, exp12, exp16, exp18,
exp20) so the detector axis may be uninformative again; per_layer's
quiet layers sit at the window floor (0.005), so its noise-robustness
can only match or trail global's — the DELTA direction is the open
question and is recorded as measured, not steered.

Gate: THRESHOLDS['exp22'] — exp10's four reused numbers, per cell
(exp18's rule: a cell failing an axis is reported failing for that
axis; the delta between cells is the measurement).

Usage:
    python -m src.experiments.exp22_layer_widths --model <id>
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.core.types import EmbeddingConfig  # noqa: E402
from src.embedding.strategy_registry import build as build_strategy  # noqa: E402
from src.experiments.experiment_registry import gate_for  # noqa: E402
from src.experiments.exp10_strategy_comparison import (  # noqa: E402
    EMBEDDINGS_PER_VARIANT,
    EPOCHS,
    PAIRS,
    PATCH_SIZE,
    PAYLOAD_BITS,
    SEED,
    measure_detectability,
    measure_extractability,
    measure_robustness,
)
from src.experiments.exp15_lwe_fidelity import _context_for, _slug  # noqa: E402
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import load_cached_residuals  # noqa: E402

EXPERIMENT = "exp22"
RULES = ("global", "per_layer")


def log(message: str) -> None:
    """Flush every stage — a long MPS run must never look hung (§3.7)."""
    print(message, flush=True)


def widths_table(residuals, family: str, layers: int) -> Dict[str, Any]:
    """The dial itself: per-layer std and the width each rule derives.

    Recorded so the artifact shows what the extractor will actually
    compute, not just the rule's name.
    """
    stds = {
        lid: float(t.float().std()) for lid, t in residuals.items()
    }
    table: Dict[str, Any] = {"layer_std": stds, "widths": {}, "w_over_std": {}}
    for rule in RULES:
        config = EmbeddingConfig(
            total_payload_bits=PAYLOAD_BITS,
            embedding_strategy="lwe",
            model_family=family,
            num_hidden_layers=layers,
            lwe_width_rule=rule,
        )
        strategy = build_strategy(config, "lwe")
        widths = {
            lid: strategy._derive_interval_width(lid, std)
            for lid, std in stds.items()
        }
        table["widths"][rule] = widths
        table["w_over_std"][rule] = {
            lid: widths[lid] / std for lid, std in stds.items()
        }
    return table


def measure_cell(residuals, rule: str, family: str, layers: int,
                 device) -> Dict[str, Any]:
    gate = gate_for(EXPERIMENT)
    overrides = (
        {} if rule == "global" else {"lwe_width_rule": rule}
    )

    log(f"  [{rule}] extractability (round trip, no cover) ...")
    ext = measure_extractability(
        residuals, "lwe", family=family, num_layers=layers,
        config_overrides=overrides,
    )

    log(f"  [{rule}] robustness (sigma grid, 3 trials each) ...")
    curve = measure_robustness(
        residuals, "lwe", family=family, num_layers=layers,
        config_overrides=overrides,
    )

    log(f"  [{rule}] detectability (400-pair detector) ...")
    accuracy, density, samples = measure_detectability(
        residuals, "lwe", device, family=family, num_layers=layers,
        config_overrides=overrides,
    )

    round_trip = bool(
        ext.get("structurally_usable")
        and ext.get("ber") == gate["max_ber"]
    )
    b1 = curve.get("0.001")
    b2 = curve.get("0.002")
    robust = bool(
        b1 is not None
        and b1 <= gate["max_ber_at_sigma_0_001"]
        and b2 is not None
        and b2 <= gate["max_ber_at_sigma_0_002"]
    )
    under_gate = bool(
        accuracy is not None and accuracy <= gate["max_detector_accuracy"]
    )

    cell = {
        "width_rule": rule,
        "config_overrides": overrides,
        "extractability": ext,
        "robustness_ber_curve": curve,
        "detector_accuracy": accuracy,
        "detector_samples": samples,
        "signal_density": density,
        "meets_round_trip_gate": round_trip,
        "meets_robustness_gate": robust,
        "meets_detector_gate": under_gate,
        "wins": round_trip and robust and under_gate,
    }
    log(
        f"  [{rule}] ber={ext.get('ber')} "
        f"sigma.001={b1} sigma.002={b2} "
        f"detector={accuracy} ({samples} samples) -> "
        f"round_trip={round_trip} robust={robust} "
        f"under_gate={under_gate} wins={cell['wins']}"
    )
    return cell


def exp18_lwe_anchor(model_slug: str) -> Dict[str, Any]:
    """exp18's same-model lwe cell — the control's anchor.

    Recorded as comparison, never gated; a missing artifact or cell is
    reported as such — never defaulted to a number.
    """
    source = f"exp18_matrix_{model_slug}.json"
    out: Dict[str, Any] = {"source": source, "available": False}
    path = RESULTS_DIR / source
    if not path.exists():
        out["reason"] = "exp18 artifact missing on disk"
        return out
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return out

    out["available"] = True
    cell = next(
        (c for c in artifact.get("cells", []) if c.get("strategy") == "lwe"),
        None,
    )
    out["present"] = cell is not None
    if cell is None:
        out["reason"] = "exp18 has no lwe cell"
        return out
    out["exp18_detector_accuracy"] = cell.get("detector_accuracy")
    out["exp18_robustness_ber_curve"] = cell.get("robustness_ber_curve")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B",
        help="model id whose residual cache is complete "
             "(default: Qwen2.5-3B, exp10/exp18's anchor)",
    )
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    context = _context_for(args.model)
    log(f"Model: {args.model} "
        f"(family {context.family}, {context.expected_layers} layers)")
    log("Loading residuals from cache ...")
    residuals = load_cached_residuals(
        args.model, context.expected_layers
    )

    log("[exp22] deriving per-layer widths for both rules ...")
    table = widths_table(residuals, context.family, context.expected_layers)
    w_g = table["widths"]["global"]
    w_p = table["widths"]["per_layer"]
    log(
        f"  global: all {len(set(w_g.values()))} distinct width(s) "
        f"= {sorted(set(w_g.values()))}"
    )
    log(
        f"  per_layer: range [{min(w_p.values()):.5f}, "
        f"{max(w_p.values()):.5f}] "
        f"w/sigma [{min(table['w_over_std']['per_layer'].values()):.2f}, "
        f"{max(table['w_over_std']['per_layer'].values()):.2f}] "
        f"(global w/sigma "
        f"[{min(table['w_over_std']['global'].values()):.2f}, "
        f"{max(table['w_over_std']['global'].values()):.2f}])"
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    cells: List[Dict[str, Any]] = []
    for rule in RULES:
        log(f"[exp22] width_rule={rule} ...")
        cells.append(
            measure_cell(
                residuals, rule,
                context.family, context.expected_layers, device,
            )
        )

    # The measurement: the delta between the two cells, computed only
    # from numbers actually present.
    global_cell, per_layer_cell = cells
    delta = {
        "detector": (
            per_layer_cell["detector_accuracy"]
            - global_cell["detector_accuracy"]
            if per_layer_cell["detector_accuracy"] is not None
            and global_cell["detector_accuracy"] is not None
            else None
        ),
        "ber_sigma_0_002": (
            per_layer_cell["robustness_ber_curve"].get("0.002")
            - global_cell["robustness_ber_curve"].get("0.002")
            if per_layer_cell["robustness_ber_curve"].get("0.002")
            is not None
            and global_cell["robustness_ber_curve"].get("0.002")
            is not None
            else None
        ),
        "ber_sigma_0_001": (
            per_layer_cell["robustness_ber_curve"].get("0.001")
            - global_cell["robustness_ber_curve"].get("0.001")
            if per_layer_cell["robustness_ber_curve"].get("0.001")
            is not None
            and global_cell["robustness_ber_curve"].get("0.001")
            is not None
            else None
        ),
    }

    anchor = exp18_lwe_anchor(_slug(args.model))
    if anchor.get("present"):
        anchor["exp22_global_detector_accuracy"] = (
            global_cell.get("detector_accuracy")
        )
        a = anchor.get("exp18_detector_accuracy")
        b = anchor.get("exp22_global_detector_accuracy")
        anchor["detector_delta_vs_exp18"] = (
            (b - a) if (a is not None and b is not None) else None
        )

    gate = gate_for(EXPERIMENT)
    artifact: Dict[str, Any] = {
        "experiment": EXPERIMENT,
        "title": "W5.4 — per-layer LWE grid width (first pass, one model)",
        "model_id": args.model,
        "family": context.family,
        "num_layers": context.expected_layers,
        "gate": {
            **gate,
            "gate_source": "experiment_registry.THRESHOLDS['exp22']",
        },
        "cells": cells,
        "rules": list(RULES),
        "layer_widths": table,
        "cell_delta": delta,
        "control_anchor": anchor,
        "method": {
            "axes_source": (
                "exp10_strategy_comparison measurement functions, "
                "same protocol, only EmbeddingConfig.lwe_width_rule "
                "varies per cell (config_overrides)"
            ),
            "width_rule_global": "DEFAULT_GRID_WIDTH (0.010) on every layer",
            "width_rule_per_layer": (
                "clip(4.0 * round(std, 4), 0.005, 0.020) — std is the "
                "layer tensor's own, coarsened to 4 decimals so embed "
                "(original) and extract (stego) bucket identically; "
                "measured embed shift <= 0.0153% per layer"
            ),
            "window_source": (
                "exp11's measured frontier (0.005-0.02 passes both "
                "gates; 0.002 fails robustness, 0.05 detectable)"
            ),
            "payload_bits": PAYLOAD_BITS,
            "detector_pairs": PAIRS,
            "detector_embeddings": EMBEDDINGS_PER_VARIANT,
            "detector_epochs": EPOCHS,
            "patch_size": PATCH_SIZE,
            "seed": SEED,
            "robustness_sigmas": ["0.0", "0.001", "0.002", "0.005"],
            "robustness_trials": "3 per nonzero sigma, 1 at 0.0",
            "one_model_per_process": True,
        },
        "notes": [
            "Per-cell verdicts only, exp18's rule: a cell failing an "
            "axis is reported failing for that axis, never as an "
            "experiment error. The DELTA between cells is the "
            "measurement.",
            "The global-width cell is exp10's lwe cell under another "
            "name (default rule, default width); its anchor vs "
            "exp18's committed lwe cell is recorded, not gated.",
            "Detector accuracy is trained on each cell's own "
            "embedding (exp10's comparability note): comparable as "
            "'this detector against this rule'.",
            "The extractor derives widths from stego std; any "
            "embed/extract bucket disagreement would surface as "
            "round-trip BER > 0.0, which the gate catches.",
        ],
    }

    target = RESULTS_DIR / f"{EXPERIMENT}_layer_widths_{_slug(args.model)}.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    log("=" * 70)
    for cell in cells:
        log(
            f"{cell['width_rule']:<10} ber={cell['extractability']['ber']} "
            f"sigma.001={cell['robustness_ber_curve'].get('0.001')} "
            f"sigma.002={cell['robustness_ber_curve'].get('0.002')} "
            f"detector={cell['detector_accuracy']} "
            f"wins={cell['wins']}"
        )
    log(f"delta (per_layer - global): {delta}")
    log(f"wrote {target}")

    del residuals
    torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
