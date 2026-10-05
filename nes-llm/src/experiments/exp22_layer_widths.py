"""
W5.4 — per-layer LWE grid width (exp22).

RESEARCH_PLAN §4 W5.4: *"Per-layer strategy selection. Different grid
width per layer, keyed by layer noise. Layers differ: Qwen2.5-7B spans
0.0012–0.0130, Phi-3 spans 0.0026–0.0031."*

Three cells, exp10's three axes each, ONLY `EmbeddingConfig.lwe_width_rule`
varying (exp10's `config_overrides` hook):

    global      the shipped absolute width 0.010 on every layer —
                exp10's own lwe cell under another name, the control.
    per_layer   w_l = clip(4.0 * round(std_l, 4), 0.005, 0.020) —
                magnitude-keyed: width proportional to the layer's
                own noise, clipped to exp11's measured window.
    layer_rank  a fixed ladder [0.005, 0.010] keyed to the layer's
                RANK in the noise ordering — the same heterogeneity,
                keyed to the part of "layer noise" that measurement
                noise cannot move.

Run 1 (global + per_layer) measured the design's failure mode and
this module records it rather than smoothing it: per_layer round-trips
at 0.0 and hides like every lwe cell (detector 0.50), but its
robustness collapses — BER 0.0223 at sigma=0.001 (gate 0.02) and
0.5747 at sigma=0.002 against global's 0.0127. Cause, verified
numerically: the extractor sizes the grid from the tensor it
receives, and the robustness measurement hands it the NOISY tensor —
noise inflates std through sqrt(std^2 + sigma^2), so 36/36 layers
bucket differently at every sigma tested and the extractor's grid
drifts wider than the embedder's (the `noise_bucket_flips` field
recomputes this from the artifact's own stds). Magnitude-keying is
therefore unbuildable at extract time: the statistic moves under
exactly the perturbation the gate measures.

layer_rank isolates agreement from heterogeneity: rank is preserved
under the strictly monotone inflation map, so both sides derive
identical widths from a fixed ladder while still running different
widths per layer. What the ladder then SCORES (robustness vs the
global control) is the open question, recorded as measured.

Pre-registered expectations for the rerun: all three cells round-trip
at exp3's 0.0 (agreement is by construction for global/layer_rank
and bucket-verified for per_layer at sigma=0); the lwe detector has
been exactly 0.50 in every experiment that trained it (exp10, exp12,
exp16, exp18, exp20) so 0.50 again is expected but not assumed; the
per_layer robustness numbers are rerun confirmations of run 1's
measured values; layer_rank's robustness direction vs global is NOT
predicted — its quiet-end layers sit at exp11's floor, its bulk below
the global default, and the delta is recorded either way.

Gate: THRESHOLDS['exp22'] — exp10's four reused numbers, per cell
(exp18's rule: a cell failing an axis is reported failing for that
axis; the deltas against the global control are the measurement).

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
RULES = ("global", "per_layer", "layer_rank")
ROBUSTNESS_SIGMAS = (0.001, 0.002, 0.005)


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
        if rule == "layer_rank":
            # Ranks need the whole layer set at once.
            widths = strategy._rank_widths(stds)
        else:
            widths = {
                lid: strategy._derive_interval_width(lid, std)
                for lid, std in stds.items()
            }
        table["widths"][rule] = widths
        table["w_over_std"][rule] = {
            lid: widths[lid] / std for lid, std in stds.items()
        }
    return table


def noise_bucket_flips(stds: Dict[int, float]) -> Dict[str, Any]:
    """Why run 1's per_layer cell collapsed: the extractor sizes the
    grid from the tensor it receives, and under noise that tensor's
    std is inflated by sqrt(std^2 + sigma^2) — layer buckets move and
    the extractor's grid drifts wider than the embedder's.

    Analytic (sample-noise ignored): recomputable from the artifact's
    own recorded stds, so the claim lives in the numbers it cites.
    """
    import math

    out: Dict[str, Any] = {
        "method": "analytic: round(sqrt(std^2 + sigma^2), 4) vs "
                  "round(std, 4) per layer (sample noise ignored)",
        "layers": len(stds),
    }
    for sigma in ROBUSTNESS_SIGMAS:
        changed = sum(
            1
            for s in stds.values()
            if round(math.sqrt(s * s + sigma * sigma), 4) != round(s, 4)
        )
        out[f"{sigma}"] = {
            "layers_changed": changed,
            "of": len(stds),
        }
    return out


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

    log("[exp22] deriving per-layer widths for every rule ...")
    table = widths_table(residuals, context.family, context.expected_layers)
    w_g = table["widths"]["global"]
    w_p = table["widths"]["per_layer"]
    w_r = table["widths"]["layer_rank"]
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
    log(
        f"  layer_rank: ladder [{min(w_r.values()):.5f}, "
        f"{max(w_r.values()):.5f}] over {len(w_r)} ranks "
        f"(noise-invariant by construction)"
    )
    flips = noise_bucket_flips(table["layer_std"])
    log(
        "  magnitude-keying diagnosis: layers whose bucket noise "
        "moves = "
        + ", ".join(
            f"sigma {s}: {flips[s]['layers_changed']}/{flips[s]['of']}"
            for s in ("0.001", "0.002", "0.005")
        )
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

    # The measurement: the deltas of each non-global rule against
    # the global control, computed only from numbers actually
    # present.
    def diff(a, b):
        return b - a if (a is not None and b is not None) else None

    global_cell = cells[0]
    cell_deltas: Dict[str, Any] = {}
    for cell in cells[1:]:
        rule = cell["width_rule"]
        cell_deltas[rule] = {
            "detector": diff(
                global_cell.get("detector_accuracy"),
                cell.get("detector_accuracy"),
            ),
            "ber_sigma_0_001": diff(
                global_cell["robustness_ber_curve"].get("0.001"),
                cell["robustness_ber_curve"].get("0.001"),
            ),
            "ber_sigma_0_002": diff(
                global_cell["robustness_ber_curve"].get("0.002"),
                cell["robustness_ber_curve"].get("0.002"),
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
        "cell_deltas": cell_deltas,
        "noise_bucket_flips": noise_bucket_flips(table["layer_std"]),
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
            "width_rule_layer_rank": (
                "fixed ladder [0.005, 0.010] by the layer's RANK in "
                "the std ordering (ties on layer id) — rank is "
                "preserved under noise's monotone sqrt(std^2 + "
                "sigma^2) inflation, so embed and extract derive "
                "identical widths from their own views"
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
            "experiment error. The DELTAS against the global control "
            "are the measurement.",
            "Run 1's failure, recorded not smoothed: per_layer "
            "round-tripped at 0.0 and hid at detector 0.50, but its "
            "robustness collapsed (0.0223 at sigma=0.001, 0.5747 at "
            "0.002) because the extractor sizes the grid from the "
            "tensor it receives — and under noise that tensor's std "
            "is inflated, so its buckets (and grids) drift from the "
            "embedder's. noise_bucket_flips recomputes the drift "
            "from this artifact's own recorded stds. Magnitude-"
            "keying is unbuildable at extract time; layer_rank is "
            "the noise-invariant keying that isolates agreement "
            "from heterogeneity.",
            "The global-width cell is exp10's lwe cell under another "
            "name (default rule, default width); its anchor vs "
            "exp18's committed lwe cell is recorded, not gated.",
            "Detector accuracy is trained on each cell's own "
            "embedding (exp10's comparability note): comparable as "
            "'this detector against this rule'.",
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
    log(f"deltas vs global control: {json.dumps(cell_deltas)}")
    log(f"wrote {target}")

    del residuals
    torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
