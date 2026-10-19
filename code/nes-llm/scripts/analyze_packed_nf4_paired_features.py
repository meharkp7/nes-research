#!/usr/bin/env python3
"""Paired, source-stratified analysis of packed-NF4 block features.

Reads the existing grouped dataset and writes a new, immutable result directory.
Pairs are formed by (source_id, run_id, block_index), never by block_index alone.
Metadata columns are never model features.

Important inference note:
Blocks are clustered observations, not independent experimental replications.
Block-level paired permutation p-values are descriptive/exploratory only. The
report explicitly flags independent-run replication as the unit needed for
generalizable inference. This script does not train a classifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


META_COLUMNS = {
    "artifact_id", "artifact_sha256", "run_id", "source_id", "model_id",
    "tensor_key", "role", "label", "block_index", "split",
}
EXPECTED_LABELS = {"clean": 0, "embedded": 1}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, preserving input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adjusted_sorted = ranked * n / np.arange(1, n + 1)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    result = np.empty(n, dtype=float)
    result[order] = adjusted_sorted
    return result.tolist()


def sign_flip_pvalue(deltas: np.ndarray, *, draws: int, seed: int) -> float:
    """Two-sided paired sign-flip randomization p-value for mean difference.

    This operates at block-pair level and is NOT cluster-corrected for run.
    Its p-value must therefore be treated as exploratory when blocks share a run.
    """
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    observed = abs(float(d.mean()))
    # Exact enumeration for small samples; otherwise Monte Carlo sign flips.
    if len(d) <= 18:
        n = 1 << len(d)
        hits = 0
        for mask in range(n):
            signs = np.fromiter(
                (1.0 if (mask >> i) & 1 else -1.0 for i in range(len(d))),
                dtype=float, count=len(d)
            )
            hits += abs(float(np.mean(d * signs))) >= observed - 1e-15
        return hits / n
    rng = np.random.default_rng(seed)
    hits = 0
    batch = 2048
    remaining = draws
    while remaining:
        size = min(batch, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(d)))
        stats = np.abs((signs * d).mean(axis=1))
        hits += int(np.count_nonzero(stats >= observed - 1e-15))
        remaining -= size
    return (hits + 1) / (draws + 1)


def analyze_source(g: pd.DataFrame, features: list[str], draws: int, seed: int) -> dict:
    source = str(g["source_id"].iloc[0])
    # Explicit role-label consistency check.
    for role, expected in EXPECTED_LABELS.items():
        role_rows = g[g["role"] == role]
        if len(role_rows) and not role_rows["label"].eq(expected).all():
            raise ValueError(f"{source}: role {role!r} conflicts with expected label {expected}")
    if not set(g["role"].dropna().unique()).issubset(EXPECTED_LABELS):
        raise ValueError(f"{source}: unexpected role value")

    pair_keys = ["run_id", "block_index"]
    counts = g.groupby(pair_keys + ["label"], dropna=False).size().unstack(fill_value=0)
    if not {0, 1}.issubset(counts.columns):
        raise ValueError(f"{source}: at least one pair key lacks clean or embedded rows")
    bad = ~((counts[0] == 1) & (counts[1] == 1))
    if bad.any():
        examples = [str(x) for x in counts.index[bad][:5].tolist()]
        raise ValueError(
            f"{source}: {int(bad.sum())} (run_id, block_index) keys are not exactly "
            f"one clean + one embedded row; examples={examples}"
        )

    clean = g[g["label"] == 0].set_index(pair_keys)[features].sort_index()
    embedded = g[g["label"] == 1].set_index(pair_keys)[features].sort_index()
    if not clean.index.equals(embedded.index):
        raise ValueError(f"{source}: clean and embedded pair indices differ")
    delta = embedded - clean

    feature_rows = []
    for i, col in enumerate(features):
        d = delta[col].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        if not len(d):
            continue
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        mean = float(np.mean(d))
        effect = mean / sd if sd > 0 else None
        nonzero = d[d != 0]
        sign_consistency = (
            float(max(np.mean(nonzero > 0), np.mean(nonzero < 0)))
            if len(nonzero) else None
        )
        p = sign_flip_pvalue(d, draws=draws, seed=seed + i)
        feature_rows.append({
            "source_id": source,
            "feature": col,
            "n_pairs": int(len(d)),
            "mean_embedded_minus_clean": mean,
            "median_embedded_minus_clean": float(np.median(d)),
            "sd_of_paired_differences": sd,
            "paired_standardized_effect_mean_over_sd": effect,
            "fraction_pairs_nonzero": float(np.mean(d != 0)),
            "sign_consistency_among_nonzero": sign_consistency,
            "exploratory_block_level_sign_flip_p": p,
        })

    adjusted = bh_adjust([r["exploratory_block_level_sign_flip_p"] for r in feature_rows])
    for row, q in zip(feature_rows, adjusted):
        row["exploratory_block_level_bh_q"] = q

    runs = sorted(str(x) for x in g["run_id"].dropna().unique())
    return {
        "source_id": source,
        "model_id_values": sorted(str(x) for x in g["model_id"].dropna().unique()),
        "run_ids": runs,
        "independent_run_count": len(runs),
        "pair_key": ["run_id", "block_index"],
        "paired_blocks_total": int(len(delta)),
        "feature_results": feature_rows,
        "inference_warning": (
            "Block-level sign-flip p-values and BH q-values are exploratory: blocks "
            "within a run are clustered, not independent replications. Generalizable "
            "inference requires independent clean/embedded runs. With one run, "
            "source-level replication is insufficient regardless of block count."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Must not already exist; outputs are never overwritten")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    source_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if args.draws < 1000:
        raise ValueError("--draws must be >= 1000")

    df = pd.read_csv(source_path)
    if df.columns.duplicated().any():
        raise ValueError("Dataset has duplicate column names")
    required = META_COLUMNS
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")
    features = [c for c in df.columns if c not in META_COLUMNS]
    if not features:
        raise ValueError("No numeric feature columns found")
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")
    if df[list(features)].isna().any().any():
        raise ValueError("Feature matrix contains NaNs; resolve explicitly before analysis")
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary 0=clean, 1=embedded")
    if df.duplicated(["source_id", "run_id", "block_index", "label"]).any():
        raise ValueError("Duplicate source/run/block/label rows found")

    source_results = []
    for index, (source, g) in enumerate(df.groupby("source_id", sort=True)):
        source_results.append(analyze_source(g, features, args.draws, args.seed + index * 10000))

    result = {
        "schema": "nes.packed_nf4_paired_feature_analysis.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(source_path),
        "input_sha256": sha256_file(source_path),
        "rows": int(len(df)),
        "sources": int(df["source_id"].nunique()),
        "feature_count": len(features),
        "feature_columns": features,
        "pairing": {
            "keys": ["source_id", "run_id", "block_index"],
            "required_cardinality": "exactly one clean and one embedded row per key",
            "role_label_mapping": EXPECTED_LABELS,
        },
        "multiple_testing": "Benjamini-Hochberg within each source across tested features",
        "draws": args.draws,
        "seed": args.seed,
        "sources_results": source_results,
        "global_inference_warning": (
            "This is a paired descriptive analysis. Blocks are nested within artifacts/runs; "
            "block-level randomization p-values are exploratory and must not be presented as "
            "run-level evidence. Multiple independent runs/checkpoints are needed for "
            "generalizable claims. This script does not establish undetectability."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        (output_dir / "paired_feature_analysis.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        rows = [r for s in source_results for r in s["feature_results"]]
        pd.DataFrame(rows).to_csv(output_dir / "paired_feature_analysis.csv", index=False)
        summary = {
            "schema": result["schema"],
            "input_sha256": result["input_sha256"],
            "rows": result["rows"],
            "sources": result["sources"],
            "feature_count": result["feature_count"],
            "outputs": ["paired_feature_analysis.json", "paired_feature_analysis.csv"],
            "warning": result["global_inference_warning"],
        }
        (output_dir / "README.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        # Do not silently leave partial output; preserve no pre-existing files because
        # the directory was required not to exist before this run.
        for child in output_dir.iterdir():
            child.unlink()
        output_dir.rmdir()
        raise
    print(json.dumps({
        "status": "complete",
        "output_dir": str(output_dir),
        "input_sha256": result["input_sha256"],
        "sources": result["sources"],
        "features_per_source": result["feature_count"],
        "paired_blocks": {s["source_id"]: s["paired_blocks_total"] for s in source_results},
        "independent_runs": {s["source_id"]: s["independent_run_count"] for s in source_results},
        "warning": result["global_inference_warning"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
