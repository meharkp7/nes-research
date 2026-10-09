#!/usr/bin/env python3
"""Run the frozen 7-method x 7-model matrix over paragraph-length JSONL corpora.

Runs one cell at a time and records PASS, BER_FAIL, EMBED_FAILED, EXTRACT_FAILED,
or BLOCKED. Residual-stream and packed-NF4 methods keep their native artifact
and receiver contracts; this runner does not claim the artifacts are full model
checkpoints or that successful recovery establishes stealth.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
CORPUS_DIR = ROOT / "data" / "seven_method_corpora"
RESIDUAL_RUNNER = ROOT / "scripts" / "seven_method_residual_pilot.py"
NF4_RUNNER = ROOT / "scripts" / "seven_method_nf4_pilot.py"

DEFAULT_MODELS = [
    "google/gemma-2-2b",
    "Qwen/Qwen2.5-3B",
    "meta-llama/Llama-3.1-8B",
    "google/gemma-2-9b",
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-v0.3",
    "Qwen/Qwen2.5-7B",
]
RESIDUAL_METHODS = [
    "sign", "magnitude_aware", "qae", "lwe_grid_parity", "split_sign_parity"
]
NF4_METHODS = ["qse", "dce"]
ALL_METHODS = RESIDUAL_METHODS + NF4_METHODS
DEFAULT_CORPORA = ["corpus_a.jsonl", "corpus_b.jsonl", "corpus_c.jsonl"]


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")


def run_command(command: list[str], log_path: Path) -> tuple[int, str, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\n[stdout]\n" + process.stdout
        + "\n[stderr]\n" + process.stderr,
        encoding="utf-8",
    )
    return process.returncode, process.stdout, process.stderr


def load_layer_count(model_id: str, local_files_only: bool) -> int:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_id, local_files_only=local_files_only, trust_remote_code=False
    )
    count = getattr(config, "num_hidden_layers", None)
    if count is None and getattr(config, "text_config", None) is not None:
        count = getattr(config.text_config, "num_hidden_layers", None)
    if not isinstance(count, int) or count < 1:
        raise ValueError(f"Could not determine transformer layer count for {model_id}")
    return count


def spread_layers(layer_count: int, requested: int = 5) -> list[int]:
    count = min(requested, layer_count)
    return sorted({round(i * (layer_count - 1) / max(count - 1, 1))
                   for i in range(count)})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--corpora", nargs="+", default=DEFAULT_CORPORA,
                        help="JSONL filenames inside data/seven_method_corpora")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--local-files-only", action="store_true",
                        help="Do not download model configs/weights; unavailable cells are recorded")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate corpus/model selection and print planned cell count only")
    args = parser.parse_args()

    corpora: list[Path] = []
    for name in args.corpora:
        path = (CORPUS_DIR / name).resolve()
        if path.parent != CORPUS_DIR.resolve() or not path.is_file():
            parser.error(f"Corpus must be an existing JSONL file directly under {CORPUS_DIR}: {name}")
        # Validate JSONL and framing before starting expensive model work.
        from src.experiments.seven_method_protocol import load_jsonl, encode_corpus
        records = load_jsonl(path)
        framed = encode_corpus(records)
        print(f"Corpus {path.name}: {len(records)} messages, "
              f"{sum(len(row.text.encode('utf-8')) for row in records)} UTF-8 bytes, "
              f"{len(framed) * 8} framed bits", flush=True)
        corpora.append(path)

    if args.dry_run:
        print(json.dumps({
            "models": args.models, "methods": args.methods,
            "corpora": [p.name for p in corpora],
            "planned_cells": len(args.models) * len(args.methods) * len(corpora),
        }, indent=2))
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (args.output_root or (REPO_ROOT / "cache" / "seven_method_long_matrix" / run_id)).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    summary_path = output_root / "matrix_summary.json"
    csv_path = output_root / "matrix_summary.csv"
    results: list[dict[str, Any]] = []

    def record(row: dict[str, Any]) -> None:
        results.append(row)
        write_json(summary_path, {
            "schema": "nes.seven_method_long_matrix.v1",
            "run_id_utc": run_id,
            "models_requested": args.models,
            "methods_requested": args.methods,
            "corpora_requested": [p.name for p in corpora],
            "completed_cells": len(results),
            "planned_cells": len(args.models) * len(args.methods) * len(corpora),
            "results": results,
        })
        fields = ["model", "corpus", "method", "status", "payload_bits", "ber",
                  "exact_match", "artifact", "extract_report", "reason"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    for model_index, model_id in enumerate(args.models, start=1):
        print(f"\n===== MODEL {model_index}/{len(args.models)}: {model_id} =====", flush=True)
        try:
            layer_count = load_layer_count(model_id, args.local_files_only)
            layers = spread_layers(layer_count)
            print(f"Detected {layer_count} layers; residual layers={layers}", flush=True)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            for corpus in corpora:
                for method in args.methods:
                    record({"model": model_id, "corpus": corpus.name, "method": method,
                            "status": "BLOCKED", "reason": reason})
            print(f"BLOCKED model config: {reason}", flush=True)
            continue

        for corpus in corpora:
            for method in args.methods:
                cell_dir = output_root / slug(model_id) / corpus.stem / method
                cell_dir.mkdir(parents=True, exist_ok=False)
                artifact = cell_dir / "embedded_artifact.pt"
                expected = cell_dir / "expected_corpus.bin"
                embed_log = cell_dir / "embed.log"
                extract_log = cell_dir / "extract.log"
                extract_report = cell_dir / "extract.json"
                print(f"--- {corpus.stem} / {method} ---", flush=True)

                if method in RESIDUAL_METHODS:
                    embed_cmd = [
                        sys.executable, str(RESIDUAL_RUNNER), "embed",
                        "--model", model_id, "--method", method,
                        "--messages-file", str(corpus), "--layers",
                        ",".join(str(x) for x in layers),
                        "--output", str(artifact), "--corpus-out", str(expected),
                    ]
                    extract_cmd = [
                        sys.executable, str(RESIDUAL_RUNNER), "extract",
                        "--artifact", str(artifact), "--expected-corpus", str(expected),
                        "--report", str(extract_report),
                    ]
                else:
                    embed_cmd = [
                        sys.executable, str(NF4_RUNNER), "embed",
                        "--model", model_id, "--method", method,
                        "--tensors", "auto", "--messages-file", str(corpus),
                        "--output", str(artifact), "--corpus-out", str(expected),
                    ]
                    if args.local_files_only:
                        embed_cmd.append("--local-files-only")
                    extract_cmd = [
                        sys.executable, str(NF4_RUNNER), "extract",
                        "--artifact", str(artifact), "--expected-corpus", str(expected),
                    ]

                embed_rc, embed_stdout, embed_stderr = run_command(embed_cmd, embed_log)
                if embed_rc != 0:
                    reason = (embed_stderr or embed_stdout)[-2500:]
                    record({"model": model_id, "corpus": corpus.name, "method": method,
                            "status": "EMBED_FAILED", "artifact": str(artifact),
                            "reason": reason})
                    print(f"EMBED_FAILED (see {embed_log})", flush=True)
                    continue

                extract_rc, extract_stdout, extract_stderr = run_command(extract_cmd, extract_log)
                try:
                    extract_data = json.loads(extract_stdout)
                    if method in NF4_METHODS:
                        write_json(extract_report, extract_data)
                except json.JSONDecodeError:
                    extract_data = {}
                if extract_rc == 0 and extract_data.get("exact_match") is True:
                    status = "PASS"
                    reason = ""
                elif extract_data.get("exact_match") is False or (
                    isinstance(extract_data.get("ber"), (int, float)) and extract_data["ber"] > 0
                ):
                    status = "BER_FAIL"
                    reason = extract_data.get("decode_error") or "payload did not match expected corpus"
                else:
                    status = "EXTRACT_FAILED"
                    reason = (extract_stderr or extract_stdout)[-2500:]
                record({
                    "model": model_id, "corpus": corpus.name, "method": method,
                    "status": status, "payload_bits": extract_data.get("expected_bits"),
                    "ber": extract_data.get("ber"), "exact_match": extract_data.get("exact_match"),
                    "artifact": str(artifact), "extract_report": str(extract_report),
                    "reason": reason,
                })
                print(f"{status}: BER={extract_data.get('ber')} exact={extract_data.get('exact_match')}", flush=True)

    print(f"\nMatrix complete. Summary: {summary_path}\nCSV: {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
