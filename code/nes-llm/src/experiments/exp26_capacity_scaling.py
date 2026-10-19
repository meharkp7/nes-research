"""
exp26 — capacity scaling curve (W9.2).

The paper figure's data: the UNCHANGED production path at
1k / 2.5k / 5k / 10k / 20k / 50k payload bits, measuring per size —
round-trip BER (exp3's gate), full production decrypt, exp7's
detectability trio (SecurityValidator), and mean |delta| over changed
values (exp24's single distortion definition).

exp4 owns the 500k-10M band with BER only; this experiment is the
1k-50k band with detectability attached. Points that fail stay in the
artifact — a failure is the measured limit, not a reason to drop the
row.

Writes results/exp26_capacity_scaling_<slug>.json and a
dependency-free four-panel SVG figure
results/exp26_capacity_scaling_<slug>.svg (BER / detector accuracy /
KL / distortion vs payload size).

Usage
-----
    python -m src.experiments.exp26_capacity_scaling [--model <id>]
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
from src.embedding.intelligent_embedder import IntelligentEmbedder  # noqa: E402
from src.embedding.strategy_registry import (  # noqa: E402
    build as build_strategy,
    extract_with,
)
from src.experiments.exp15_lwe_fidelity import _context_for, _slug  # noqa: E402
from src.experiments.experiment_registry import THRESHOLDS  # noqa: E402
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import load_cached_residuals  # noqa: E402
from src.experiments.exp24_pareto_frontier import (  # noqa: E402
    mean_abs_delta,
)
from src.extraction.decrypt_pipeline import DecryptPipeline  # noqa: E402
from src.steganalysis.security_validator import SecurityValidator  # noqa: E402

EXPERIMENT = "exp26"
STRATEGY = "sign"            # production's own scheme
SIZES: List[int] = [1_000, 2_500, 5_000, 10_000, 20_000, 50_000]
SEED = 42


def log(message: str) -> None:
    print(f"[{EXPERIMENT}] {message}", flush=True)


def message_for(payload_bits: int) -> str:
    """exp4's construction: repeated 'A', one byte past the bit count."""
    return "A" * (payload_bits // 8 + 1)


# ---------------------------------------------------------------------
# Per-size measurement
# ---------------------------------------------------------------------

def measure_point(
    context, residuals, payload_bits: int
) -> Dict[str, Any]:
    message = message_for(payload_bits)
    config = EmbeddingConfig(
        total_payload_bits=payload_bits,
        embedding_strategy=STRATEGY,
        model_family=context.family,
        num_hidden_layers=(
            context.actual_layers or context.expected_layers
        ),
    )

    torch.manual_seed(SEED)
    result = IntelligentEmbedder(config).embed(message, residuals)
    transmitted = result.embedded_bits

    strategy = build_strategy(config, STRATEGY)
    recovered = extract_with(
        strategy, result.embedded_residuals, result.carrier_indices,
        residuals_ref=None, strategy_name=STRATEGY,
    )
    compared = min(len(transmitted), len(recovered))
    errors = sum(
        1 for a, b in zip(transmitted[:compared], recovered[:compared])
        if a != b
    )
    ber = errors / compared if compared else None

    recovered_message, stats = DecryptPipeline(key=result.key).run(
        result.embedded_residuals, result.carrier_indices
    )

    torch.manual_seed(SEED)
    sec = SecurityValidator().validate(
        original_residuals=residuals,
        embedded_residuals=result.embedded_residuals,
        carrier_indices=result.carrier_indices,
    )
    distortion = mean_abs_delta(residuals, result.embedded_residuals)

    point = {
        "payload_bits": payload_bits,
        "bits_embedded": int(result.bits_embedded),
        "bits_compared": compared,
        "bit_errors": errors,
        "ber": ber,
        "decrypt_ok": bool(stats.get("success")),
        "recovered_matches": recovered_message == message,
        "kl_divergence": float(sec.kl_divergence),
        "detector_accuracy": float(sec.detector_accuracy),
        "sign_bias": float(sec.sign_bias),
        "mean_shift": float(sec.moment_shift["mean_shift"]),
        "std_shift": float(sec.moment_shift["std_shift"]),
        "mean_abs_delta": distortion["mean_abs_delta"],
        "changed_values": distortion["changed_values"],
        "carrier_count": int(
            sum(len(v) for v in result.carrier_indices.values())
        ),
    }
    point["meets_ber_gate"] = bool(ber == 0.0)
    point["meets_detectability_gate"] = bool(
        sec.detector_accuracy <= 0.55 and sec.kl_divergence <= 0.05
    )
    return point


def evaluate_gate(points: List[Dict[str, Any]], gate: Dict[str, Any]):
    failures = []
    for p in points:
        if p["ber"] != 0.0:
            failures.append(
                f"{p['payload_bits']} bits: ber={p['ber']} "
                f"({p['bit_errors']}/{p['bits_compared']})"
            )
        if p["detector_accuracy"] > gate["max_detector_accuracy"]:
            failures.append(
                f"{p['payload_bits']} bits: detector_accuracy="
                f"{p['detector_accuracy']}"
            )
        if p["kl_divergence"] > gate["max_kl_divergence"]:
            failures.append(
                f"{p['payload_bits']} bits: kl={p['kl_divergence']}"
            )
    return ("PASS" if not failures else "FAIL"), failures


# ---------------------------------------------------------------------
# Figure — pure SVG, no dependencies, deterministic
# ---------------------------------------------------------------------

_PANELS = [
    ("Round-trip BER", "ber", 0.0, 0.6,
     [("gate 0.0", 0.0), ("chance 0.5", 0.5)], "{:.4f}"),
    ("Statistical detector accuracy", "detector_accuracy", 0.40, 0.62,
     [("gate 0.55", 0.55), ("chance 0.50", 0.50)], "{:.4f}"),
    ("KL(clean || stego)", "kl_divergence", 0.0, None,
     [("gate 0.05", 0.05)], "{:.5f}"),
    ("Mean |delta| at changed values", "mean_abs_delta", 0.0, None,
     [], "{:.3e}"),
]


def _fmt_size(bits: int) -> str:
    if bits % 1000 == 0:
        return f"{bits // 1000}k"
    return f"{bits / 1000:g}k"


def write_svg(points, model_id: str, target: Path) -> None:
    """Four-panel operating curve: BER / accuracy / KL / distortion."""
    P_W, P_H = 420, 230
    M_L, M_T, GAP_X, GAP_Y = 78, 76, 88, 96
    PAD_L, PAD_R, PAD_T, PAD_B = 58, 26, 44, 46
    width = M_L + 2 * P_W + GAP_X + 40
    height = M_T + 2 * P_H + GAP_Y + 40
    x0 = M_L + PAD_L
    x1 = M_L + P_W - PAD_R

    def x_at(i):
        return x0 + i * (x1 - x0) / max(len(points) - 1, 1)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{M_L}" y="34" font-family="Helvetica,Arial" '
        f'font-size="18" font-weight="bold" fill="#111">'
        f'exp26 — capacity operating curve (production path, '
        f'sign + QACI)</text>',
        f'<text x="{M_L}" y="54" font-family="Helvetica,Arial" '
        f'font-size="12" fill="#555">{model_id} — payload 1k→50k, '
        f'one embed per size; gates from '
        f'experiment_registry.THRESHOLDS[exp26]</text>',
    ]

    for pi, (title, key, ymin, ymax, guides, vfmt) in enumerate(_PANELS):
        px = M_L + (pi % 2) * (P_W + GAP_X)
        py = M_T + (pi // 2) * (P_H + GAP_Y)
        ytop, ybot = py + PAD_T, py + P_H - PAD_B
        vals = [p[key] for p in points]
        if ymax is None:
            hi = max(vals) if max(vals) > 0 else 1.0
            ymax = hi * 1.2
        span = (ymax - ymin) or 1.0

        def y_at(v):
            return ybot - (v - ymin) / span * (ybot - ytop)

        out.append(
            f'<text x="{px}" y="{py + 18}" font-family="Helvetica,Arial" '
            f'font-size="13" font-weight="bold" fill="#111">{title}</text>'
        )
        # axes
        out.append(
            f'<line x1="{px + PAD_L}" y1="{ytop}" '
            f'x2="{px + PAD_L}" y2="{ybot}" stroke="#333"/>'
        )
        out.append(
            f'<line x1="{px + PAD_L}" y1="{ybot}" '
            f'x2="{px + P_W - PAD_R}" y2="{ybot}" stroke="#333"/>'
        )
        # y ticks
        for t in range(5):
            v = ymin + span * t / 4
            y = y_at(v)
            out.append(
                f'<line x1="{px + PAD_L - 5}" y1="{y:.1f}" '
                f'x2="{px + PAD_L}" y2="{y:.1f}" stroke="#333"/>'
            )
            out.append(
                f'<text x="{px + PAD_L - 9}" y="{y + 4:.1f}" '
                f'text-anchor="end" font-family="Helvetica,Arial" '
                f'font-size="10" fill="#333">{v:.3g}</text>'
            )
        # guides
        for label, v in guides:
            if not (ymin <= v <= ymax):
                continue
            y = y_at(v)
            out.append(
                f'<line x1="{px + PAD_L}" y1="{y:.1f}" '
                f'x2="{px + P_W - PAD_R}" y2="{y:.1f}" '
                f'stroke="#d33" stroke-dasharray="5 4" '
                f'stroke-width="1.1"/>'
            )
            out.append(
                f'<text x="{px + P_W - PAD_R - 4}" y="{y - 5:.1f}" '
                f'text-anchor="end" font-family="Helvetica,Arial" '
                f'font-size="10" fill="#d33">{label}</text>'
            )
        # series
        coords = [
            (x_at(i), y_at(max(ymin, min(ymax, v))))
            for i, v in enumerate(vals)
        ]
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        out.append(
            f'<polyline points="{poly}" fill="none" stroke="#1864ab" '
            f'stroke-width="2"/>'
        )
        for i, ((x, y), v) in enumerate(zip(coords, vals)):
            out.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" '
                f'fill="#1864ab"/>'
            )
            out.append(
                f'<text x="{x:.1f}" y="{y - 9:.1f}" text-anchor="middle" '
                f'font-family="Helvetica,Arial" font-size="9.5" '
                f'fill="#111">{vfmt.format(v)}</text>'
            )
        # x labels (bottom row only)
        if pi >= 2:
            for i, p in enumerate(points):
                out.append(
                    f'<text x="{x_at(i):.1f}" y="{ybot + 18}" '
                    f'text-anchor="middle" font-family="Helvetica,Arial" '
                    f'font-size="11" fill="#333">'
                    f'{_fmt_size(p["payload_bits"])}</text>'
                )
            out.append(
                f'<text x="{(px + P_W) // 2}" y="{py + P_H - 6}" '
                f'text-anchor="middle" font-family="Helvetica,Arial" '
                f'font-size="11" fill="#555">payload (bits)</text>'
            )

    out.append(
        f'<text x="{M_L}" y="{height - 14}" font-family="Helvetica,Arial" '
        f'font-size="10.5" fill="#777">Every point is one full '
        f'production embed → extract → decrypt; failed points are kept. '
        f'Companion: exp4 covers 500k–10M with BER only.</text>'
    )
    out.append("</svg>")
    target.write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B",
        help="model id whose residual cache is complete "
             "(default: Qwen2.5-3B, exp3/exp10's anchor)",
    )
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    context = _context_for(args.model)
    log(f"Model: {args.model} (family {context.family}, "
        f"{context.expected_layers} layers)")

    log("loading cached residuals ...")
    residuals = load_cached_residuals(args.model, context.expected_layers)

    points = []
    for size in SIZES:
        log(f"measuring {size} bits ...")
        try:
            point = measure_point(context, residuals, size)
        except Exception as exc:  # a crash is a measured limit too
            point = {
                "payload_bits": size,
                "status": "NOT_RUN",
                "reason": f"{type(exc).__name__}: {exc}",
                "ber": None,
            }
            log(f"  NOT_RUN: {point['reason']}")
        else:
            log(
                f"  ber={point['ber']} "
                f"acc={point['detector_accuracy']:.4f} "
                f"kl={point['kl_divergence']:.5f} "
                f"mad={point['mean_abs_delta']:.3e} "
                f"decrypt_ok={point['decrypt_ok']}"
            )
        points.append(point)

    gate = dict(THRESHOLDS[EXPERIMENT])
    status, failures = evaluate_gate(points, gate)

    measured = [p for p in points if p.get("ber") is not None]
    artifact = {
        "experiment": EXPERIMENT,
        "title": "Capacity scaling curve (W9.2)",
        "model_id": args.model,
        "family": context.family,
        "status": status,
        "gate": {
            **gate,
            "gate_source": (
                f"experiment_registry.THRESHOLDS['{EXPERIMENT}']"
            ),
            "failures": failures,
        },
        "configuration": {
            "sizes": SIZES,
            "message_source": "repeated 'A', length = bits//8 + 1 (exp4)",
            "strategy": STRATEGY,
            "path": (
                "IntelligentEmbedder production path, unchanged; "
                "SecurityValidator per size (exp7's instrument), "
                "mean_abs_delta (exp24's single definition)"
            ),
        },
        "points": points,
        "metrics": {
            "sizes_tested": len(SIZES),
            "sizes_measured": len(measured),
            "sizes_at_ber_zero": [
                p["payload_bits"] for p in measured if p["ber"] == 0.0
            ],
            "max_tested_payload_at_ber_zero_bits": (
                max(
                    (p["payload_bits"] for p in measured
                     if p["ber"] == 0.0),
                    default=None,
                )
            ),
            "detector_accuracy_range": (
                [min(p["detector_accuracy"] for p in measured),
                 max(p["detector_accuracy"] for p in measured)]
                if measured else None
            ),
            "kl_range": (
                [min(p["kl_divergence"] for p in measured),
                 max(p["kl_divergence"] for p in measured)]
                if measured else None
            ),
        },
        "method": {
            "seed": SEED,
            "pre_registered": (
                "Sizes and gates fixed in this file and "
                "THRESHOLDS['exp26'] before the first run; failing "
                "points stay in the artifact."
            ),
            "relation_to_exp4": (
                "exp4 tests 500k/1M/5M/10M with BER only (guide's "
                "minimum); this is the 1k-50k band with "
                "detectability attached."
            ),
        },
        "notes": [
            "One embed per size, sequential, same model and residual "
            "cache as exp3/exp23 — the curve is a property of payload "
            "size alone.",
            "The figure (same basename, .svg) is generated from these "
            "points with no plotting dependency: committed alongside "
            "so reviewers need no environment to see it.",
        ],
        "reproducibility": context.reproducibility(),
    }

    base = f"{EXPERIMENT}_capacity_scaling_{_slug(args.model)}"
    (RESULTS_DIR / f"{base}.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    write_svg(
        [p for p in points if p.get("ber") is not None],
        args.model,
        RESULTS_DIR / f"{base}.svg",
    )
    log(f"wrote {base}.json and {base}.svg — gate: {status}"
        + (f" — {failures}" if failures else ""))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
