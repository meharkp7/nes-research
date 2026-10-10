#!/usr/bin/env python3
"""Prepare a framed multi-string corpus for the NES seven-method experiments.

Examples:
  python scripts/prepare_seven_method_corpus.py \
    --message "first string" --message "नमस्ते 🌍" \
    --output ../artifacts/seven_method_corpus.bin

  python scripts/prepare_seven_method_corpus.py \
    --messages-file ../tests/fixtures/messages.jsonl \
    --output ../artifacts/seven_method_corpus.bin

The output contains the UTF-8 payload corpus. The sidecar manifest intentionally
contains message IDs, byte lengths and hashes, but not plaintext message text.
Existing output files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.seven_method_protocol import (  # noqa: E402
    MessageRecord,
    corpus_summary,
    encode_corpus,
    load_jsonl,
    normalize_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--message", action="append", default=[],
        help="Message string; repeat this option to include multiple strings.",
    )
    parser.add_argument(
        "--messages-file", type=Path,
        help="Optional UTF-8 JSONL file with one {id,text} object per line.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="New binary output path for the framed corpus.",
    )
    parser.add_argument(
        "--manifest", type=Path,
        help="Sidecar JSON manifest path (default: OUTPUT.manifest.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records: list[MessageRecord] = []
    if args.messages_file is not None:
        records.extend(load_jsonl(args.messages_file))
    records.extend(
        MessageRecord(f"cli-{index + 1:04d}", text)
        for index, text in enumerate(args.message)
    )
    rows = normalize_records(records)
    if not rows:
        raise SystemExit("Provide at least one --message or --messages-file record.")

    output = args.output.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else output.with_suffix(output.suffix + ".manifest.json")
    )
    if output == manifest:
        raise SystemExit("Binary output and manifest paths must differ.")
    if output.exists() or manifest.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing output: "
            + ", ".join(str(p) for p in (output, manifest) if p.exists())
        )

    blob = encode_corpus(rows)
    summary = corpus_summary(rows)
    summary["output_file"] = output.name
    summary["output_sha256"] = hashlib.sha256(blob).hexdigest()
    summary["input_sources"] = {
        "messages_file": str(args.messages_file) if args.messages_file else None,
        "cli_message_count": len(args.message),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects against accidental clobbering between checks.
    with output.open("xb") as handle:
        handle.write(blob)
    try:
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except Exception:
        # Avoid leaving an apparently complete artifact without its manifest.
        output.unlink(missing_ok=True)
        raise
    print(json.dumps({
        "status": "prepared",
        "messages": len(rows),
        "framed_bits": summary["framed_bits"],
        "output": str(output),
        "manifest": str(manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
