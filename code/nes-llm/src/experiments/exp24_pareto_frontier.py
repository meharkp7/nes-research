"""
W7 — the Pareto frontier (exp24).

RESEARCH_PLAN §4 W7: *\"Plot every strategy × every parameter: x = mean
perturbation magnitude, y = detector accuracy, marker = BER@σ0.001,
every strategy × parameter.\"* — the strongest publishable framing.

Scoping finding, decided before any code: **no committed artifact records
mean |Δ| together with detector accuracy.** exp2 and exp17 record
magnitudes under their own protocols but carry no detector axis;
every detector-bearing artifact (exp10/11/12/18/20/22, exp7's study)
records changed-value COUNTS (`signal_density`), never magnitudes.
Substituting a count for the plan's x-axis would be a different claim,
so W7 measures x here, uniformly:

    x = mean |embedded − original| over CHANGED values,
    one embed per unique (model, strategy, config) group,
    originals snapshotted before the embed (defends against any
    in-place mutation; identical numbers either way when there is none).

Sources (all committed, audit-verified):

    exp10                        results[]   sign / magnitude_aware / lwe
    exp18_matrix_<model>.json    cells[]     4 strategies × every cached model
    exp11_lwe_alpha_pareto       results[]   lwe × grid width
    exp12_lwe_cross_model        results[]   lwe × model (grid_width 0.01)
    exp20_split_dial_<model>     cells[]     split × split_fraction
    exp22_layer_widths_<model>   cells[]     lwe × width_rule
    exp7_neural_parameter_study  variants[]  sign × {min_magnitude,
                                      gamma, payload} — its "alpha" IS
                                      the perturbation floor (the
                                      study's own `min_magnitude=alpha`)

Parameter reconstruction is pinned to each source's own mechanism:
exp11/exp12 pinned the grid width by patching alpha=1.0 and
min_magnitude=w/2 (their modules' `_force_grid_width` /
`_patch_grid_width`), so their points carry exactly those overrides;
split carries split_fraction; exp22 carries lwe_width_rule; exp7's
recorded alpha maps to min_magnitude, never to EmbeddingConfig.alpha
(that would silently measure a different cell); everything else is
the shipped default config.

Exclusions are recorded, never dropped (§2 ground rule): cells with no
detector axis are excluded with the reason; sources with no detector
axis are listed as omitted with the reason; exp14/exp16 adversary and
cross-scheme readings are cited under `related` (same embedding, same
x, different question) rather than silently folded into the frontier.
Extraction or embed failures at runtime become excluded entries carrying
the actual error string.

Gate: THRESHOLDS['exp24'] — max_source_delta 0.0. Citation integrity:
every y and every marker this artifact stores must equal the value in
its source file, recomputed by claim_audit; the frontier recomputes
from the artifact's own points (nondominated on x↓, y↓). Marker
(BER@σ0.001) is an annotation, not an axis: points whose source has no
robustness curve keep marker null with the reason recorded and remain
frontier candidates.

Usage:
    python -m src.experiments.exp24_pareto_frontier
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.core.types import EmbeddingConfig  # noqa: E402
from src.embedding.intelligent_embedder import IntelligentEmbedder  # noqa: E402
from src.experiments.experiment_registry import (  # noqa: E402
    family_of,
    get_model,
)
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import load_cached_residuals  # noqa: E402

EXPERIMENT = "exp24"
SEED = 42
DEFAULT_PAYLOAD = 10_000
FRONTIER_MARK_SIGMA = "0.001"

# Fixed-name sources and glob patterns, all committed artifacts.
FIXED_SOURCES = (
    "exp10_strategy_comparison.json",
    "exp11_lwe_alpha_pareto.json",
    "exp12_lwe_cross_model.json",
    "exp7_neural_parameter_study.json",
)
GLOB_SOURCES = (
    "exp18_matrix_*.json",
    "exp20_*.json",
    "exp22_*.json",
)

# Cited under `related`, never folded into the frontier: their question
# is the adversary's, not the embedding's, so they are read for their
# values only and never passed through extract_points.
RELATED_SOURCES = (
    "exp14_blind_patch_detector.json",
    "exp16_cross_scheme_detector.json",
)

# Sources without a detector axis, named with the reason they cannot
# form a frontier point. Anything not listed here and not a frontier
# source falls under the catch-all reason below.
OMITTED_SOURCES: Dict[str, str] = {
    "exp2_*.json": "write-magnitude characterisation, no detector axis; "
                   "its mean_mag_mean is cited as a prior, not as x",
    "exp4_*.json": "capacity sweep: BER vs payload size, no detector axis",
    "exp6_*.json": "sigma-robustness curves only, no detector axis",
    "exp17_*.json": "round trip + mean_abs_delta under exp17's own 50k "
                    "protocol, no detector axis; cited as a prior",
    "exp19_*.json": "adaptive routing: round trips and branch selection, "
                    "no detector measurement",
    "everything else (exp1, exp3, exp5, exp8, exp9, exp13, exp15, "
    "exp21, exp23)": "round trip / fidelity / recovery / format / "
                     "surgery questions — none records a detector "
                     "accuracy, so none can form y",
}


def log(message: str) -> None:
    """Flush every stage — a long run must never look hung (§3.7)."""
    print(message, flush=True)


# ------------------------------------------------------------------
# Pure helpers (unit-tested without artifacts or models)
# ------------------------------------------------------------------

def mean_abs_delta(
    original: Dict[int, torch.Tensor],
    embedded: Dict[int, torch.Tensor],
) -> Dict[str, float]:
    """Mean |embedded − original| over CHANGED values.

    The x-axis, defined once so every cell is measured the same way:
    changed = exact tensor inequality (the embed writes carrier values,
    so an untouched value is bit-identical).
    """
    total = 0.0
    changed = 0
    for lid, orig in original.items():
        emb = embedded.get(lid)
        if emb is None:
            continue
        diff = (emb.detach().float() - orig.detach().float()).abs()
        mask = emb != orig
        n = int(mask.sum())
        if n:
            total += float(diff[mask].sum())
            changed += n
    return {
        "mean_abs_delta": total / changed if changed else 0.0,
        "changed_values": float(changed),
    }


def pareto_front(points: List[Dict[str, Any]]) -> List[int]:
    """Indices of points nondominated on (x minimized, y minimized).

    q dominates p iff q.x <= p.x and q.y <= p.y with at least one
    strict. Equal points do not dominate each other (both stay).
    """
    front: List[int] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if (
                q["x"] <= p["x"]
                and q["y"] <= p["y"]
                and (q["x"] < p["x"] or q["y"] < p["y"])
            ):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def config_kwargs(point: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """EmbeddingConfig kwargs for a point, from its normalized params.

    Only keys the source actually varied are passed; everything else is
    the shipped default. `spec` is experiment_registry.get_model's dict
    (family may be null → family_of, exactly as ModelContext does).
    """
    params = point["params"]
    kwargs: Dict[str, Any] = {
        "total_payload_bits": int(params.get("payload_bits", DEFAULT_PAYLOAD)),
        "embedding_strategy": point["strategy"],
        "model_family": spec.get("family") or family_of(point["model_id"]),
        "num_hidden_layers": int(spec["expected_layers"]),
    }
    for key in ("alpha", "min_magnitude", "gamma",
                "split_fraction", "lwe_width_rule"):
        if key in params:
            kwargs[key] = params[key]
    return kwargs


def _point_id(experiment: str, model_id: str, strategy: str,
              label: str = "") -> str:
    short = model_id.replace("/", "__").replace("-", "_").lower()
    base = f"{experiment}:{short}:{strategy}"
    return f"{base}:{label}" if label else base


# ------------------------------------------------------------------
# Source extraction: parsed artifact dict -> normalized points
# ------------------------------------------------------------------

def _marker(cell: Dict[str, Any]) -> Optional[float]:
    curve = cell.get("robustness_ber_curve")
    if isinstance(curve, dict):
        return curve.get(FRONTIER_MARK_SIGMA)
    return None


def _round_trip(cell: Dict[str, Any]) -> Optional[float]:
    ext = cell.get("extractability")
    if isinstance(ext, dict) and ext.get("ber") is not None:
        return ext["ber"]
    curve = cell.get("robustness_ber_curve")
    if isinstance(curve, dict):
        return curve.get("0.0")
    return None


def extract_points(parsed: Dict[str, Any]) -> Tuple[List[Dict[str, Any]],
                                                     List[Dict[str, str]]]:
    """Dispatch on the artifact's experiment id.

    Returns (points, excluded). Points are normalized: id, source,
    experiment, model_id, strategy, params, y, marker, round_trip_ber.
    Cells without a detector axis, non-measured statuses, or missing
    keys become excluded entries carrying the reason.
    """
    experiment = str(parsed.get("experiment", ""))
    model_default = parsed.get("model_id", "unknown")
    points: List[Dict[str, Any]] = []
    excluded: List[Dict[str, str]] = []

    def add(strategy: str, params: Dict[str, Any], y: Any, label: str,
            model_id: str, cell: Dict[str, Any], key: str) -> None:
        if y is None:
            excluded.append({
                "what": _point_id(experiment, model_id, strategy, label),
                "reason": "source cell has no detector accuracy "
                          "(key 'detector_accuracy'/'accuracy' absent or null)",
                "source": key,
            })
            return
        points.append({
            "id": _point_id(experiment, model_id, strategy, label),
            "source": key,
            "experiment": experiment,
            "model_id": model_id,
            "strategy": strategy,
            "params": params,
            "y": float(y),
            "marker": _marker(cell),
            "round_trip_ber": _round_trip(cell),
        })

    if experiment == "exp10_strategy_comparison":
        for i, cell in enumerate(parsed["results"]):
            key = f"{experiment}:results[{i}]"
            status = cell.get("status")
            if status and status != "READY":
                note = (cell.get("notes")
                        or "not embeddable as recorded in source")
                excluded.append({
                    "what": _point_id(experiment, model_default,
                                      cell.get("strategy", "?")),
                    "reason": f"status {status}: {note}",
                    "source": key,
                })
                continue
            add(cell["strategy"], {"payload_bits": DEFAULT_PAYLOAD},
                cell.get("detector_accuracy"), "", model_default, cell, key)

    elif experiment.startswith("exp18"):
        for i, cell in enumerate(parsed["cells"]):
            key = f"{experiment}:cells[{i}]"
            status = cell.get("status")
            if status and status != "READY":
                excluded.append({
                    "what": _point_id(experiment, model_default,
                                      cell.get("strategy", "?")),
                    "reason": f"status {status} (recorded in source)",
                    "source": key,
                })
                continue
            add(cell["strategy"], {"payload_bits": DEFAULT_PAYLOAD},
                cell.get("detector_accuracy"), "", model_default, cell, key)

    elif experiment == "exp11_lwe_alpha_pareto":
        for i, cell in enumerate(parsed["results"]):
            key = f"{experiment}:results[{i}]"
            width = float(cell["grid_width"])
            add("lwe",
                {"payload_bits": DEFAULT_PAYLOAD,
                 # The source's own patch: alpha=1.0 so the width term
                 # wins, min_magnitude = w/2 so the floor matches it.
                 "alpha": 1.0, "min_magnitude": width / 2.0},
                cell.get("detector_accuracy"), f"w{width:g}",
                model_default, cell, key)

    elif experiment == "exp12_lwe_cross_model":
        for i, cell in enumerate(parsed["results"]):
            key = f"{experiment}:results[{i}]"
            model_id = cell.get("model_id", model_default)
            status = cell.get("status")
            if status != "MEASURED":
                excluded.append({
                    "what": _point_id(experiment, model_id, "lwe"),
                    "reason": f"status {status}: "
                              f"{cell.get('reason') or 'recorded in source'}",
                    "source": key,
                })
                continue
            width = float(cell.get("grid_width",
                                   parsed.get("grid_width", 0.01)))
            add("lwe",
                {"payload_bits": DEFAULT_PAYLOAD,
                 "alpha": 1.0, "min_magnitude": width / 2.0},
                cell.get("detector_accuracy"), f"m{model_id.split('/')[-1]}",
                model_id, cell, key)

    elif experiment == "exp20":
        for i, cell in enumerate(parsed["cells"]):
            key = f"{experiment}:cells[{i}]"
            fraction = float(cell["split_fraction"])
            add("split",
                {"payload_bits": DEFAULT_PAYLOAD,
                 "split_fraction": fraction},
                cell.get("detector_accuracy"), f"sf{fraction:g}",
                model_default, cell, key)

    elif experiment == "exp22":
        for i, cell in enumerate(parsed["cells"]):
            key = f"{experiment}:cells[{i}]"
            rule = str(cell["width_rule"])
            add("lwe",
                {"payload_bits": DEFAULT_PAYLOAD,
                 "lwe_width_rule": rule},
                cell.get("detector_accuracy"), rule,
                model_default, cell, key)

    elif experiment == "exp7_neural_parameter_study":
        for i, cell in enumerate(parsed["variants"]):
            key = f"{experiment}:variants[{i}]"
            # The study's "alpha" is the perturbation floor, mapped by
            # its own module to min_magnitude (NOT EmbeddingConfig.alpha):
            # `min_magnitude=alpha` in exp7_neural_parameter_study.py.
            add("sign",
                {"payload_bits": int(cell.get("payload_bits",
                                              DEFAULT_PAYLOAD)),
                 "min_magnitude": float(cell["alpha"]),
                 "gamma": float(cell["gamma"])},
                cell.get("accuracy"), str(cell.get("variant", i)),
                model_default, cell, key)

    else:
        excluded.append({
            "what": experiment,
            "reason": "unrecognised experiment id — refusing to guess "
                      "the cell structure",
            "source": experiment,
        })

    return points, excluded


def related_values(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Cite adversary/cross readings that are NOT frontier points.

    Same embedding (same x), different question — recorded so they are
    never silently dropped from the framing.
    """
    experiment = str(parsed.get("experiment", ""))
    if experiment == "exp14":
        arms = parsed.get("arms", {})
        return [{
            "source": "exp14_blind_patch_detector.json",
            "blind_accuracy": (arms.get("blind", {}).get("metrics", {})
                               .get("accuracy")),
            "control_accuracy": (arms.get("control", {}).get("metrics", {})
                                 .get("accuracy")),
            "blind_positions_with_carrier":
                arms.get("blind", {}).get("positions_containing_carrier"),
            "reason": "adversary-placement variant of the same sign "
                      "embedding: y moves with the detector's knowledge "
                      "while x does not — a reading of one point, not "
                      "a new point",
        }]
    if experiment == "exp16":
        return [{
            "source": "exp16_cross_scheme_detector.json",
            "cross_matrix": {k: v.get("accuracy")
                             for k, v in parsed.get("results", {}).items()},
            "reason": "cross-scheme readings of existing embeddings: "
                      "same x, different question — not frontier points",
        }]
    return []


def prior_magnitude_citations(results_dir: Path) -> List[Dict[str, Any]]:
    """Independent magnitudes recorded elsewhere, cited not merged.

    Different protocols (exp2's own write-magnitude protocol; exp17's
    50k-bit payload) — they cannot be x without pretending uniformity
    this synthesis does not have.
    """
    priors: List[Dict[str, Any]] = []
    exp2 = results_dir / "exp2_qwen__qwen2.5-3b.json"
    if exp2.exists():
        data = json.loads(exp2.read_text(encoding="utf-8"))
        priors.append({
            "source": exp2.name,
            "mean_mag_mean": (data.get("mean_mag_mean")
                              or data.get("reproducibility", {})
                              .get("mean_mag_mean")),
            "protocol": "exp2's own write-magnitude protocol (no detector "
                        "axis) — cited, not used as x",
        })
    exp17 = results_dir / "exp17_qae_round_trip.json"
    if exp17.exists():
        data = json.loads(exp17.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        priors.append({
            "source": exp17.name,
            "mean_abs_delta_over_changed": metrics.get(
                "mean_abs_delta_over_changed"),
            "bits_embedded": metrics.get("bits_embedded"),
            "protocol": "qae at exp17's 50k-bit protocol — cited, not "
                        "used as x (different payload)",
        })
    return priors


# ------------------------------------------------------------------
# The x-axis: one embed per unique (model, strategy, config) group
# ------------------------------------------------------------------

def measure_magnitudes(
    points: List[Dict[str, Any]],
    excluded: List[Dict[str, str]],
    results_dir: Path,
) -> None:
    """Attach x to every point; failures become excluded entries."""
    del results_dir  # caches are located by residual_source itself
    groups: Dict[Tuple[str, str, Tuple], List[int]] = {}
    for i, point in enumerate(points):
        key = (point["model_id"], point["strategy"],
               tuple(sorted((k, json.dumps(v, sort_keys=True)
                             if isinstance(v, dict) else v)
                            for k, v in point["params"].items())))
        groups.setdefault(key, []).append(i)

    residuals_by_model: Dict[str, Dict[int, torch.Tensor]] = {}

    for (model_id, strategy, param_items), indices in groups.items():
        params = dict(param_items)
        what = (f"{_point_id('group', model_id, strategy, '')}"
                f"[{len(indices)} point(s)]")
        try:
            spec = get_model(model_id)
        except KeyError as exc:
            excluded.append({
                "what": what,
                "reason": f"model not in experiment_registry: {exc}",
                "source": "experiment_registry",
            })
            continue

        residuals = residuals_by_model.get(model_id)
        if residuals is None:
            try:
                residuals = load_cached_residuals(
                    model_id, int(spec["expected_layers"])
                )
                residuals_by_model[model_id] = residuals
            except Exception as exc:  # cache incompleteness, bad path
                excluded.append({
                    "what": what,
                    "reason": f"residual cache unavailable: {exc}",
                    "source": "residual_source.load_cached_residuals",
                })
                continue

        point = dict(points[indices[0]])
        point["params"] = params
        config = EmbeddingConfig(**config_kwargs(point, spec))

        # Snapshot before the embed: x compares against the pre-embed
        # tensors regardless of whether the strategy writes in place.
        snapshot = {lid: t.clone() for lid, t in residuals.items()}
        payload = int(params.get("payload_bits", DEFAULT_PAYLOAD))
        message = "A" * max(payload // 8, 1)
        try:
            result = IntelligentEmbedder(config).embed(message, residuals)
            stats = mean_abs_delta(snapshot, result.embedded_residuals)
        except Exception as exc:
            excluded.append({
                "what": what,
                "reason": f"embed failed: {exc}",
                "source": f"embed({model_id}, {strategy}, {params})",
            })
            del snapshot
            continue

        for i in indices:
            points[i]["x"] = stats["mean_abs_delta"]
            points[i]["changed_values"] = int(stats["changed_values"])
            points[i]["bits_embedded"] = int(
                getattr(result, "bits_embedded", 0) or 0
            )
        del snapshot
        log(f"  x({model_id}, {strategy}, {params}) = "
            f"{stats['mean_abs_delta']:.6g} "
            f"over {int(stats['changed_values'])} changed values")

    # Points whose group failed never got x: move them out of the
    # frontier pool explicitly rather than leaving a hole.
    unresolved = [p for p in points if "x" not in p]
    for p in unresolved:
        excluded.append({
            "what": p["id"],
            "reason": "magnitude not measured (group failure recorded above)",
            "source": p["source"],
        })
    if unresolved:
        keep = [p for p in points if "x" in p]
        points.clear()
        points.extend(keep)


def collect_source_files(results_dir: Path) -> List[Path]:
    files: List[Path] = []
    for name in FIXED_SOURCES:
        path = results_dir / name
        if path.exists():
            files.append(path)
    for pattern in GLOB_SOURCES:
        files.extend(sorted(results_dir.glob(pattern)))
    # De-duplicate, preserve order.
    seen = set()
    unique: List[Path] = []
    for path in files:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(
        description="W7 — Pareto frontier synthesis over committed artifacts"
    )
    parser.add_argument("--results", default=str(RESULTS_DIR),
                        help="directory holding the source artifacts")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    results_dir = Path(args.results)

    log(f"[exp24] reading sources from {results_dir}")
    points: List[Dict[str, Any]] = []
    excluded: List[Dict[str, str]] = []
    related: List[Dict[str, Any]] = []
    files_read: List[str] = []

    for path in collect_source_files(results_dir):
        parsed = json.loads(path.read_text(encoding="utf-8"))
        files_read.append(path.name)
        new_points, new_excluded = extract_points(parsed)
        points.extend(new_points)
        excluded.extend(new_excluded)
        related.extend(related_values(parsed))
        log(f"  {path.name}: {len(new_points)} point(s), "
            f"{len(new_excluded)} excluded")

    for name in RELATED_SOURCES:
        path = results_dir / name
        if not path.exists():
            continue
        parsed = json.loads(path.read_text(encoding="utf-8"))
        files_read.append(path.name)
        found = related_values(parsed)
        related.extend(found)
        log(f"  {path.name}: {len(found)} related citation(s)")

    log(f"[exp24] measuring x across "
        f"{len({(p['model_id'], p['strategy']) for p in points})} "
        f"(model, strategy) pairs ...")
    measure_magnitudes(points, excluded, results_dir)

    front_idx = pareto_front(points)
    frontier = [points[i]["id"] for i in front_idx]

    omitted = [{"source": pattern, "reason": reason}
               for pattern, reason in OMITTED_SOURCES.items()]

    artifact: Dict[str, Any] = {
        "experiment": EXPERIMENT,
        "title": "W7 — Pareto frontier (strategy × parameter, "
                 "x = mean |delta|, y = detector accuracy, "
                 "marker = BER@sigma 0.001)",
        "gate": {
            "max_source_delta": 0.0,
            "gate_source": "experiment_registry.THRESHOLDS['exp24']",
            "description": (
                "Citation integrity: every y/marker stored here must "
                "equal its source artifact exactly; the frontier "
                "recomputes from these points (nondominated x-down, "
                "y-down)."
            ),
        },
        "method": {
            "x_definition": (
                "mean |embedded - original| over changed values, one "
                "embed per unique (model, strategy, config) group, "
                "originals snapshotted before embed"
            ),
            "x_why_measured_here": (
                "no committed artifact records magnitude together with "
                "detector accuracy: exp2/exp17 carry magnitudes without "
                "detectors, detector-bearing artifacts carry "
                "changed-value COUNTS (signal_density) only — counts "
                "are never substituted for the plan's x-axis"
            ),
            "y": "detector accuracy as recorded in each source "
                 "(key 'detector_accuracy'; exp7 study key 'accuracy')",
            "marker": (
                "BER at sigma 0.001 from the source's "
                "robustness_ber_curve; annotation, not an axis — null "
                "means the source measured no robustness curve, "
                "recorded, never dropped"
            ),
            "frontier": "nondominated on (x minimized, y minimized); "
                        "equal points do not dominate each other",
            "config_reconstruction": (
                "exp11/exp12 grid-width cells carry the sources' own "
                "patch (alpha=1.0, min_magnitude=w/2); exp20 carries "
                "split_fraction; exp22 carries lwe_width_rule; exp7's "
                "recorded alpha maps to min_magnitude (its module's "
                "own `min_magnitude=alpha`), gamma as gamma, payload "
                "as payload; all others shipped defaults (payload "
                "10000)"
            ),
            "payload_bits_default": DEFAULT_PAYLOAD,
            "seed": SEED,
            "sources_read": files_read,
        },
        "points": points,
        "frontier": frontier,
        "excluded": excluded,
        "omitted_sources": omitted,
        "related": related,
        "prior_magnitude_citations": prior_magnitude_citations(results_dir),
        "counts": {
            "points": len(points),
            "frontier": len(frontier),
            "excluded": len(excluded),
            "related": len(related),
            "omitted_source_groups": len(omitted),
        },
    }

    target = results_dir / f"{EXPERIMENT}_pareto_frontier.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    log("=" * 70)
    log(f"points={len(points)} frontier={len(frontier)} "
        f"excluded={len(excluded)} related={len(related)}")
    for i in front_idx:
        p = points[i]
        log(f"  FRONT {p['id']:<60} x={p['x']:.6g} y={p['y']} "
            f"marker={p['marker']}")
    for entry in excluded:
        log(f"  excluded: {entry['what']} — {entry['reason']}")
    log(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
