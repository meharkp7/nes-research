#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-splits", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("schema") != "nes.packed_nf4_provenance_audit.v1":
        raise ValueError("Unexpected audit schema")
    if audit.get("status") != "passed":
        raise ValueError("Provenance audit did not pass")

    files = audit["input_files"]
    if len(files) != 28:
        raise ValueError(f"Expected 28 audited files, found {len(files)}")

    models = sorted({p["model_id"] for p in audit["pairs"]})
    if len(models) != 7:
        raise ValueError(f"Expected 7 model groups, found {len(models)}")

    rng = random.Random(args.seed)
    rng.shuffle(models)
    n = len(models)
    n_test = max(2, round(n * 0.20))
    n_val = max(1, round(n * 0.15))
    if n_test + n_val >= n:
        raise ValueError("Split leaves no training groups")

    assignment = {}
    for model in models[:n - n_test - n_val]:
        assignment[model] = "train"
    for model in models[n - n_test - n_val:n - n_test]:
        assignment[model] = "validation"
    for model in models[n - n_test:]:
        assignment[model] = "test"

    out = args.output_csv.resolve()
    manifest_path = args.output_splits.resolve()
    if out == manifest_path or out.exists() or manifest_path.exists():
        raise FileExistsError("Output path exists or paths conflict; choose new filenames")

    expected = {str(Path(x["path"]).resolve()): x for x in files}
    feature_columns = None
    metadata = [
        "artifact_id", "artifact_sha256", "run_id", "source_id",
        "model_id", "tensor_key", "role", "label", "block_index", "split"
    ]
    row_counts = {"train": 0, "validation": 0, "test": 0}
    input_info = []

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", newline="", encoding="utf-8") as dst:
            writer = None
            for item in files:
                path = Path(item["path"]).resolve()
                key = str(path)
                if key not in expected or not path.is_file():
                    raise ValueError(f"Audited input missing or unexpected: {path}")
                if sha256(path) != item["sha256"]:
                    raise ValueError(f"Input changed since audit: {path}")

                with path.open(newline="", encoding="utf-8") as src:
                    reader = csv.DictReader(src)
                    fields = reader.fieldnames or []
                    features = [f for f in fields if f not in {
                        "artifact_id", "artifact_sha256", "run_id", "source_id",
                        "model_id", "tensor_key", "role", "label", "block_index"
                    }]
                    if feature_columns is None:
                        feature_columns = features
                        writer = csv.DictWriter(dst, fieldnames=metadata + features)
                        writer.writeheader()
                    elif features != feature_columns:
                        raise ValueError(f"Feature columns/order mismatch: {path}")

                    count = 0
                    for row in reader:
                        model = row["model_id"]
                        if model not in assignment:
                            raise ValueError(f"Unexpected model ID: {model}")
                        if row["role"] not in {"clean", "embedded"}:
                            raise ValueError(f"Unexpected role: {row['role']}")
                        if row["label"] != ("0" if row["role"] == "clean" else "1"):
                            raise ValueError("Role/label mismatch")
                        row["split"] = assignment[model]
                        writer.writerow(row)
                        row_counts[row["split"]] += 1
                        count += 1

                    if count != item["rows"]:
                        raise ValueError(f"Row count changed for {path}: {count}")
                    input_info.append({
                        "path": str(path), "sha256": item["sha256"], "rows": count
                    })

        manifest = {
            "schema": "nes.packed_nf4_group_splits.v1",
            "grouping_key": "model_id",
            "seed": args.seed,
            "model_group_count": len(assignment),
            "group_to_split": dict(sorted(assignment.items())),
            "groups_per_split": {
                s: sum(v == s for v in assignment.values())
                for s in ("train", "validation", "test")
            },
            "row_counts": row_counts,
            "input_files": input_info,
            "combined_csv": str(out),
            "feature_columns": feature_columns,
            "metadata_columns_excluded_from_features": metadata,
            "guardrails": [
                "Rows were streamed; all rows for a model stay in one split.",
                "Input hashes were checked against the passed provenance audit.",
                "Pairing checks rely on the matching, hash-verified provenance audit.",
                "Historical clean-checkpoint matching remains unverified.",
                "Passing preparation is not detector evidence or proof of source independence."
            ]
        }
        with manifest_path.open("x", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        print(json.dumps({
            "status": "grouped_dataset_prepared",
            "output_csv": str(out),
            "output_splits": str(manifest_path),
            "groups_per_split": manifest["groups_per_split"],
            "row_counts": row_counts
        }, indent=2))
    except Exception:
        out.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise

if __name__ == "__main__":
    main()
