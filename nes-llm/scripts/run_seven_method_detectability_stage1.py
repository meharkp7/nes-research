#!/usr/bin/env python3
"""Build a provenance-checked, paired residual detectability dataset.

This evaluator pairs existing residual-method artifacts with their original
clean residual-cache tensors. QSE/DCE remain BLOCKED until a matched clean
packed-NF4 control is available. It does not train a classifier: blocks from
one tensor are correlated and are not independent evaluation units.

Performance/reproducibility safeguards:
- Quantiles are computed on a deterministic, evenly-spaced bounded sample.
- Exact mean/std/min/max/moments/histogram summaries still use all values.
- CSV rows are streamed to disk instead of accumulated in memory.
- Progress is printed and a progress JSON is atomically refreshed after each
  cell. Existing inputs are read-only; output directory must be new.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

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
QUANTILE_SAMPLE_MAX = 32768
HISTOGRAM_BINS = 64


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quantile_sample(x: torch.Tensor, max_points: int = QUANTILE_SAMPLE_MAX) -> torch.Tensor:
    """Return a deterministic, evenly-spaced sample for approximate quantiles."""
    n = x.numel()
    if n <= max_points:
        return x
    indices = torch.linspace(0, n - 1, steps=max_points, dtype=torch.float64).long()
    return x.index_select(0, indices)


def features(values: torch.Tensor, bins: int = HISTOGRAM_BINS, quantile_sample_max: int = QUANTILE_SAMPLE_MAX) -> dict[str, float]:
    """Compute full-data moments plus deterministic sampled quantiles."""
    x = values.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if x.numel() == 0:
        raise ValueError("empty block")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("non-finite values in block")

    mean_t = x.mean()
    std_t = x.std(unbiased=False)
    mean = float(mean_t)
    std = float(std_t)
    centered = x - mean_t
    if std > 0:
        z = centered / std_t
        skew = float(torch.mean(z * z * z))
        kurt = float(torch.mean(z * z * z * z)) - 3.0
    else:
        skew, kurt = 0.0, 0.0

    xmin, xmax = float(x.min()), float(x.max())
    if xmax > xmin:
        hist = torch.histc(x, bins=bins, min=xmin, max=xmax)
    else:
        hist = torch.tensor([float(x.numel())], dtype=torch.float32)
    probs = hist / hist.sum().clamp_min(1)
    probs = probs[probs > 0]
    entropy = float(-(probs * torch.log2(probs)).sum())

    qsample = quantile_sample(x, max_points=quantile_sample_max)
    # One quantile call sorts at most QUANTILE_SAMPLE_MAX values.
    qs = torch.quantile(qsample, torch.tensor([0.25, 0.50, 0.75]))
    return {
        "mean": mean, "std": std, "min": xmin, "max": xmax,
        "median": float(qs[1]), "q25": float(qs[0]), "q75": float(qs[2]),
        "mean_square": float(torch.mean(x * x)),
        "positive_fraction": float(torch.mean((x > 0).float())),
        "skew": skew, "excess_kurtosis": kurt, "hist_entropy": entropy,
        "zero_fraction": float(torch.mean((x == 0).float())),
        "abs_mean": float(torch.mean(torch.abs(x))),
    }


def latest_matrix_summaries(cache_root: Path) -> list[Path]:
    paths = list(cache_root.glob("seven_method_long_matrix/**/matrix_summary.json"))
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def collect_cells(cache_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for summary_path in latest_matrix_summaries(cache_root):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for original_row in summary.get("results", []):
            model, method = original_row.get("model"), original_row.get("method")
            if not model or not method:
                continue
            key = (str(model), str(method))
            if key in cells:
                continue
            row = dict(original_row)
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


class CsvStream:
    """Fixed-schema CSV writer that flushes each row for crash-visible progress."""
    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.stream = path.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=fields, extrasaction="ignore")
        self.writer.writeheader()
        self.stream.flush()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_status_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    else:
        fields = ["model_id", "method", "status", "reason"]
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT.parent / "cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--quantile-sample-max", type=int, default=QUANTILE_SAMPLE_MAX,
                        help="Maximum values sorted per quantile calculation (default: 32768)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Optional exact model IDs to include")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Optional method IDs to include")
    args = parser.parse_args()
    if args.block_size < 64:
        parser.error("--block-size must be >=64")
    if args.quantile_sample_max < 1024:
        parser.error("--quantile-sample-max must be >=1024")


    out = args.output_dir.expanduser().resolve()
    if out.exists():
        parser.error(f"Refusing to overwrite existing output directory: {out}")
    cells = collect_cells(args.cache_root)
    if args.models:
        allowed = set(args.models)
        cells = {k: v for k, v in cells.items() if k[0] in allowed}
    if args.methods:
        allowed = set(args.methods)
        cells = {k: v for k, v in cells.items() if k[1] in allowed}
    out.mkdir(parents=True, exist_ok=False)

    block_path = out / "residual_paired_block_features.csv"
    layer_path = out / "residual_layer_descriptives.csv"
    status_path = out / "cell_status.csv"
    report_path = out / "detectability_stage1_report.json"

    block_fields = [
        "model_id", "method", "layer_id", "block_index", "block_start",
        "block_size", "artifact_sha256", "clean_cache_sha256", "artifact_id",
        "run_group", "label",
    ] + [f"f_{name}" for name in FEATURES]
    layer_fields = [
        "model_id", "method", "matrix_status", "summary_path", "artifact_path",
        "artifact_sha256", "layer_id", "num_values", "changed_values",
        "changed_fraction", "carrier_count_metadata", "clean_cache_path",
        "clean_cache_sha256",
    ] + [f"{prefix}_{name}" for prefix in ("clean", "embedded", "delta") for name in FEATURES]

    status_rows: list[dict[str, Any]] = []
    ready = blocked = block_rows_written = layer_rows_written = 0
    started = time.monotonic()
    block_writer = CsvStream(block_path, block_fields)
    layer_writer = CsvStream(layer_path, layer_fields)

    def checkpoint(current_cell: str, completed: int) -> None:
        write_status_csv(status_path, status_rows)
        atomic_json(out / "progress.json", {
            "schema": "nes.seven_method_detectability_stage1.progress.v1",
            "cells_total": len(cells), "cells_completed": completed,
            "cells_ready": ready, "cells_blocked": blocked,
            "block_feature_rows_written": block_rows_written,
            "layer_descriptive_rows_written": layer_rows_written,
            "current_cell": current_cell,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "quantile_method": "deterministic evenly-spaced sample; exact only when tensor <= sample cap",
            "quantile_sample_max": args.quantile_sample_max,
        })

    try:
        for cell_num, ((model, method), row) in enumerate(sorted(cells.items()), start=1):
            artifact_path = resolve_artifact(row)
            base = {
                "model_id": model, "method": method, "matrix_status": row.get("status"),
                "summary_path": row["_summary_path"],
                "artifact_path": str(artifact_path) if artifact_path else "",
            }
            print(f"[{cell_num}/{len(cells)}] {model} | {method}", flush=True)

            if method in NF4_METHODS:
                status_rows.append({**base, "status": "BLOCKED",
                    "reason": "Packed-NF4 method requires a same-source clean packed-NF4 control; residual-cache pairing is invalid."})
            elif method not in RESIDUAL_METHODS:
                status_rows.append({**base, "status": "BLOCKED", "reason": "Unknown method family."})
            elif not artifact_path:
                status_rows.append({**base, "status": "BLOCKED", "reason": "Embedded artifact path missing or not found."})
            else:
                try:
                    metadata, pairs, carriers = load_residual_pair(model, artifact_path)
                    if metadata.get("method") != method:
                        raise ValueError(f"summary method {method} disagrees with artifact method {metadata.get('method')}")
                    artifact_hash = sha256(artifact_path)
                    cell_block_rows = cell_layer_rows = 0
                    for layer_num, (layer_id, (clean, stego, clean_path)) in enumerate(sorted(pairs.items()), start=1):
                        print(f"    layer {layer_num}/{len(pairs)}: {layer_id}", flush=True)
                        changed = int(torch.count_nonzero(clean != stego).item())
                        cache_hash = sha256(clean_path)
                        cfeat, efeat = features(clean, quantile_sample_max=args.quantile_sample_max), features(stego, quantile_sample_max=args.quantile_sample_max)
                        layer_meta = {
                            **base, "artifact_sha256": artifact_hash, "layer_id": layer_id,
                            "num_values": clean.numel(), "changed_values": changed,
                            "changed_fraction": changed / max(clean.numel(), 1),
                            "carrier_count_metadata": len(carriers.get(layer_id, carriers.get(str(layer_id), []))),
                            "clean_cache_path": str(clean_path), "clean_cache_sha256": cache_hash,
                        }
                        layer_writer.write({
                            **layer_meta,
                            **{f"clean_{k}": v for k, v in cfeat.items()},
                            **{f"embedded_{k}": v for k, v in efeat.items()},
                            **{f"delta_{k}": efeat[k] - cfeat[k] for k in FEATURES},
                        })
                        layer_rows_written += 1
                        cell_layer_rows += 1

                        for start in range(0, clean.numel(), args.block_size):
                            c = clean[start:start + args.block_size]
                            e = stego[start:start + args.block_size]
                            if c.numel() < 64:
                                continue
                            cf, ef = features(c, quantile_sample_max=args.quantile_sample_max), features(e, quantile_sample_max=args.quantile_sample_max)
                            common = {
                                "model_id": model, "method": method, "layer_id": layer_id,
                                "block_index": start // args.block_size, "block_start": start,
                                "block_size": c.numel(), "artifact_sha256": artifact_hash,
                                "clean_cache_sha256": cache_hash, "artifact_id": artifact_path.stem,
                                "run_group": f"{model}|{method}|{artifact_hash}",
                            }
                            block_writer.write({**common, "label": 0, **{f"f_{k}": v for k, v in cf.items()}})
                            block_writer.write({**common, "label": 1, **{f"f_{k}": v for k, v in ef.items()}})
                            block_rows_written += 2
                            cell_block_rows += 2

                    status_rows.append({**base, "status": "PAIRED_DESCRIPTIVE_READY",
                        "artifact_sha256": artifact_hash, "layers_with_embedding": len(pairs),
                        "feature_rows": cell_block_rows,
                        "layer_rows": cell_layer_rows,
                        "warning": "Paired blocks are correlated; do not randomly split blocks or treat them as independent samples."})
                except Exception as exc:
                    status_rows.append({**base, "status": "BLOCKED",
                        "reason": f"{type(exc).__name__}: {exc}"})

            if status_rows[-1].get("status") == "BLOCKED":
                blocked += 1
            else:
                ready += 1
            checkpoint(f"{model}|{method}", cell_num)

    finally:
        block_writer.close()
        layer_writer.close()

    report = {
        "schema": "nes.seven_method_detectability_stage1.v2",
        "purpose": "Paired per-layer residual distribution diagnostics and block feature export; not a trained detector benchmark.",
        "cache_root": str(args.cache_root.resolve()),
        "block_size": args.block_size,
        "quantile_method": "Deterministic evenly-spaced sample; quantiles are approximate when input exceeds the sample cap.",
        "quantile_sample_max": QUANTILE_SAMPLE_MAX,
        "cells_discovered": len(cells),
        "cells_paired_descriptive_ready": ready,
        "cells_blocked": blocked,
        "residual_block_feature_rows": block_rows_written,
        "layer_descriptive_rows": layer_rows_written,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "outputs": {
            "cell_status": str(status_path),
            "residual_block_features": str(block_path),
            "residual_layer_descriptives": str(layer_path),
            "progress": str(out / "progress.json"),
        },
        "guardrails": [
            "No classifier was fitted because current evidence does not establish enough independent artifacts per cell for a leakage-resistant split.",
            "NF4 QSE/DCE are blocked until matched same-source clean packed-code artifacts are supplied.",
            "Feature blocks from the same tensor are correlated; block-level rows are descriptive samples, not independent inferential units.",
            "Quantile features are approximate for tensors larger than the deterministic sample cap; all other listed statistics use full values.",
            "Successful pairing and distribution summaries do not establish stealth, undetectability, or cross-model generalization.",
            "Historical input artifacts and reports are read-only; outputs are written to a new directory.",
        ],
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
