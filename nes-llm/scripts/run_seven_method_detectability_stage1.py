#!/usr/bin/env python3
"""Build a provenance-checked, paired residual detectability dataset.

This first-stage evaluator reads the existing seven-method matrix artifacts.
For the five residual methods it pairs each embedded residual tensor with the
original residual from the model's residual cache, then emits aligned block
features and per-layer descriptive deltas. QSE/DCE packed-NF4 artifacts are
reported as BLOCKED until a same-source clean packed-NF4 control is supplied.

This script deliberately does not fit a classifier: blocks from a single
artifact are correlated, and the current matrix does not establish enough
independent artifact runs per cell for a leakage-resistant train/test split.
No existing outputs are overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
import re
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.residual_source import _cache_for  # noqa: E402

RESIDUAL_METHODS = {
    "sign", "magnitude_aware", "qae", "lwe_grid_parity", "split_sign_parity"
}
NF4_METHODS = {"qse", "dce"}
FEATURES = (
    "mean", "std", "min", "max", "median", "q25", "q75", "mean_square",
    "positive_fraction", "skew", "excess_kurtosis", "hist_entropy",
    "zero_fraction", "abs_mean",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def features(values: torch.Tensor, bins: int = 64) -> dict[str, float]:
    x = values.detach().float().cpu().reshape(-1)
    if x.numel() == 0:
        raise ValueError("empty block")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("non-finite values in block")
    mean = float(x.mean())
    std = float(x.std(unbiased=False))
    centered = x - mean
    if std > 0:
        z = centered / std
        skew = float(torch.mean(z ** 3))
        kurt = float(torch.mean(z ** 4)) - 3.0
    else:
        skew, kurt = 0.0, 0.0
    hist = torch.histc(x, bins=bins, min=float(x.min()), max=float(x.max())) if float(x.max()) > float(x.min()) else torch.tensor([float(x.numel())])
    probs = hist / hist.sum().clamp_min(1)
    probs = probs[probs > 0]
    entropy = float(-(probs * torch.log2(probs)).sum())
    return {
        "mean": mean, "std": std, "min": float(x.min()), "max": float(x.max()),
        "median": float(x.median()), "q25": float(torch.quantile(x, .25)),
        "q75": float(torch.quantile(x, .75)), "mean_square": float(torch.mean(x * x)),
        "positive_fraction": float(torch.mean((x > 0).float())),
        "skew": skew, "excess_kurtosis": kurt, "hist_entropy": entropy,
        "zero_fraction": float(torch.mean((x == 0).float())),
        "abs_mean": float(torch.mean(torch.abs(x))),
    }


def latest_matrix_summaries(cache_root: Path) -> list[Path]:
    paths = list(cache_root.glob("seven_method_long_matrix/**/matrix_summary.json"))
    # Every matrix summary may be incremental. Select the latest artifact per
    # model/method by artifact mtime only; never rewrite or mutate source files.
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def collect_cells(cache_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for summary_path in latest_matrix_summaries(cache_root):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in summary.get("results", []):
            model, method = row.get("model"), row.get("method")
            if not model or not method:
                continue
            key = (str(model), str(method))
            if key in cells:
                continue
            row = dict(row)
            row["_summary_path"] = str(summary_path)
            row["_run_root"] = str(summary_path.parent)
            cells[key] = row
    return cells


def resolve_artifact(row: dict[str, Any]) -> Path | None:
    raw = row.get("artifact")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_file():
        return path.resolve()
    candidate = (Path(row["_run_root"]) / str(raw)).resolve()
    return candidate if candidate.is_file() else None


def load_residual_pair(model_id: str, artifact_path: Path):
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if artifact.get("schema") != "nes.seven_method_residual_artifact.v1":
        raise ValueError(f"unsupported residual artifact schema: {artifact.get('schema')!r}")
    embedded = artifact.get("embedded_residuals")
    carrier_indices = artifact.get("carrier_indices")
    metadata = artifact.get("metadata", {})
    if not isinstance(embedded, dict) or not isinstance(carrier_indices, dict):
        raise ValueError("residual artifact missing embedded_residuals/carrier_indices")
    cache = _cache_for(model_id)
    clean_by_layer = {}
    for layer_key, stego_tensor in embedded.items():
        layer_id = int(layer_key)
        layer_path = cache._layer_path(layer_id)
        if not layer_path.is_file():
            raise FileNotFoundError(f"clean residual cache missing layer {layer_id}: {layer_path}")
        cached = torch.load(layer_path, map_location="cpu", weights_only=True)
        clean = cached.get("residual")
        if not isinstance(clean, torch.Tensor):
            raise ValueError(f"cache layer {layer_id} has no residual tensor")
        stego = stego_tensor.detach().cpu().contiguous().reshape(-1)
        clean = clean.detach().cpu().contiguous().reshape(-1)
        if clean.numel() != stego.numel():
            raise ValueError(f"layer {layer_id}: clean/stego sizes differ")
        clean_by_layer[layer_id] = (clean, stego, layer_path)
    return metadata, clean_by_layer, carrier_indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT.parent / "cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Optional exact model IDs to include")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Optional method IDs to include")
    args = parser.parse_args()
    if args.block_size < 64:
        parser.error("--block-size must be >=64")
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        parser.error(f"Refusing to overwrite existing output directory: {out}")
    cells = collect_cells(args.cache_root)
    if args.models:
        allowed = set(args.models)
        cells = {k:v for k,v in cells.items() if k[0] in allowed}
    if args.methods:
        allowed = set(args.methods)
        cells = {k:v for k,v in cells.items() if k[1] in allowed}
    out.mkdir(parents=True, exist_ok=False)
    block_path = out / "residual_paired_block_features.csv"
    layer_path = out / "residual_layer_descriptives.csv"
    status_rows = []
    block_rows = []
    layer_rows = []

    for (model, method), row in sorted(cells.items()):
        artifact_path = resolve_artifact(row)
        base = {
            "model_id": model, "method": method, "matrix_status": row.get("status"),
            "summary_path": row["_summary_path"],
            "artifact_path": str(artifact_path) if artifact_path else None,
        }
        if method in NF4_METHODS:
            status_rows.append({**base, "status": "BLOCKED",
                "reason": "Packed-NF4 method requires a same-source clean packed-NF4 control; residual-cache pairing is invalid."})
            continue
        if method not in RESIDUAL_METHODS:
            status_rows.append({**base, "status": "BLOCKED", "reason": "Unknown method family."})
            continue
        if not artifact_path:
            status_rows.append({**base, "status": "BLOCKED", "reason": "Embedded artifact path missing or not found."})
            continue
        try:
            metadata, pairs, carriers = load_residual_pair(model, artifact_path)
            if metadata.get("method") != method:
                raise ValueError(f"summary method {method} disagrees with artifact method {metadata.get('method')}")
            artifact_hash = sha256(artifact_path)
            block_count = 0
            for layer_id, (clean, stego, clean_path) in sorted(pairs.items()):
                changed = int(torch.count_nonzero(clean != stego).item())
                layer_meta = {
                    **base, "artifact_sha256": artifact_hash, "layer_id": layer_id,
                    "num_values": clean.numel(), "changed_values": changed,
                    "changed_fraction": changed / max(clean.numel(), 1),
                    "carrier_count_metadata": len(carriers.get(layer_id, carriers.get(str(layer_id), []))),
                    "clean_cache_path": str(clean_path),
                    "clean_cache_sha256": sha256(clean_path),
                }
                cfeat, efeat = features(clean), features(stego)
                layer_rows.append({
                    **layer_meta,
                    **{f"clean_{k}": v for k,v in cfeat.items()},
                    **{f"embedded_{k}": v for k,v in efeat.items()},
                    **{f"delta_{k}": efeat[k] - cfeat[k] for k in FEATURES},
                })
                for start in range(0, clean.numel(), args.block_size):
                    c = clean[start:start + args.block_size]
                    e = stego[start:start + args.block_size]
                    if c.numel() < 64:
                        continue
                    cf, ef = features(c), features(e)
                    common = {
                        "model_id": model, "method": method, "layer_id": layer_id,
                        "block_index": start // args.block_size, "block_start": start,
                        "block_size": c.numel(), "artifact_sha256": artifact_hash,
                        "clean_cache_sha256": sha256(clean_path),
                        "artifact_id": artifact_path.stem,
                        "run_group": f"{model}|{method}|{artifact_hash}",
                    }
                    block_rows.append({**common, "label": 0, **{f"f_{k}": v for k,v in cf.items()}})
                    block_rows.append({**common, "label": 1, **{f"f_{k}": v for k,v in ef.items()}})
                    block_count += 2
            status_rows.append({**base, "status": "PAIRED_DESCRIPTIVE_READY",
                "artifact_sha256": artifact_hash, "layers_with_embedding": len(pairs),
                "feature_rows": block_count,
                "warning": "Paired blocks are correlated; do not randomly split blocks or treat them as independent samples."})
        except Exception as exc:
            status_rows.append({**base, "status": "BLOCKED",
                "reason": f"{type(exc).__name__}: {exc}"})

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("status\nEMPTY\n", encoding="utf-8")
            return
        fields = list(dict.fromkeys(k for r in rows for k in r.keys()))
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(block_path, block_rows)
    write_csv(layer_path, layer_rows)
    write_csv(out / "cell_status.csv", status_rows)
    report = {
        "schema": "nes.seven_method_detectability_stage1.v1",
        "purpose": "Paired per-layer residual distribution diagnostics and block feature export; not a trained detector benchmark.",
        "cache_root": str(args.cache_root.resolve()),
        "block_size": args.block_size,
        "cells_discovered": len(cells),
        "cells_paired_descriptive_ready": sum(r.get("status") == "PAIRED_DESCRIPTIVE_READY" for r in status_rows),
        "cells_blocked": sum(r.get("status") == "BLOCKED" for r in status_rows),
        "residual_block_feature_rows": len(block_rows),
        "layer_descriptive_rows": len(layer_rows),
        "outputs": {
            "cell_status": str(out / "cell_status.csv"),
            "residual_block_features": str(block_path),
            "residual_layer_descriptives": str(layer_path),
        },
        "guardrails": [
            "No classifier was fitted because current evidence does not establish enough independent artifacts per cell for a leakage-resistant split.",
            "NF4 QSE/DCE are blocked until matched same-source clean packed-code artifacts are supplied.",
            "Feature blocks from the same tensor are correlated; block-level rows are descriptive samples, not independent inferential units.",
            "Successful pairing and distribution summaries do not establish stealth, undetectability, or cross-model generalization.",
            "Historical input artifacts and reports are read-only; outputs are written to a new directory."
        ]
    }
    report_path = out / "detectability_stage1_report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
