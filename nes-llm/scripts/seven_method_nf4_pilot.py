#!/usr/bin/env python3
"""Multi-tensor packed-NF4 artifact pilot for QSE and DCE.

This is a research adapter, not a full quantized Hugging Face checkpoint.
DCE decodes from packed NF4 codes alone. QSE records the reference values
required by its current receiver, so it is explicitly reference-assisted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import torch
from bitsandbytes.functional import quantize_4bit, dequantize_4bit
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.nf4_artifact_codec import allocate_payload_segments, pack_codes, tensor_dequant, unpack_codes
from src.experiments.seven_method_protocol import (
    MessageRecord, bytes_to_bits, bits_to_bytes, corpus_summary, decode_corpus,
    encode_corpus, keyed_positions, load_jsonl, normalize_records,
)


def read_messages(args) -> list[MessageRecord]:
    rows = list(load_jsonl(args.messages_file)) if args.messages_file else []
    rows.extend(MessageRecord(f"cli-{i+1:04d}", value)
                for i, value in enumerate(args.message or []))
    normalized = normalize_records(rows)
    if not normalized:
        raise ValueError("Provide at least one --message or --messages-file.")
    return normalized


def run_embed(args) -> dict:
    out = args.output.expanduser().resolve()
    expected_out = args.corpus_out.expanduser().resolve()
    report_out = args.report.expanduser().resolve() if args.report else out.with_suffix(".json")
    if len({out, expected_out, report_out}) != 3:
        raise ValueError("artifact, expected corpus, and report paths must all differ")
    for path in (out, expected_out, report_out):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")
    rows = read_messages(args)
    corpus = encode_corpus(rows)
    bits = bytes_to_bits(corpus)
    key = bytes.fromhex(args.test_key_hex)
    requested_tensors = args.tensors.strip()
    if not requested_tensors:
        raise ValueError("--tensors must be a comma-separated list or 'auto'")

    print(f"Loading source model on CPU: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu",
        local_files_only=args.local_files_only, trust_remote_code=False,
    )
    state = model.state_dict()
    if requested_tensors.lower() == "auto":
        # Choose the same attention projection family across five evenly spaced
        # layers. This handles architectures with q_proj and fused-qkv variants
        # without silently inventing nonexistent tensor names.
        preferred_suffixes = (
            ".self_attn.q_proj.weight",
            ".self_attn.qkv_proj.weight",
            ".self_attn.query_key_value.weight",
            ".self_attn.Wqkv.weight",
        )
        per_layer = {}
        for key in state:
            match = re.search(r"(?:^|\\.)layers\\.(\\d+)\\.", key)
            if not match:
                continue
            layer_id = int(match.group(1))
            suffix = next((s for s in preferred_suffixes if key.endswith(s)), None)
            if suffix is None:
                continue
            previous = per_layer.get(layer_id)
            if previous is None or preferred_suffixes.index(suffix) < preferred_suffixes.index(
                next(s for s in preferred_suffixes if previous.endswith(s))
            ):
                per_layer[layer_id] = key
        layer_ids = sorted(per_layer)
        if len(layer_ids) < 1:
            raise KeyError(
                "Could not auto-select attention projection tensors; pass explicit --tensors names"
            )
        count = min(5, len(layer_ids))
        selected_positions = sorted({round(i * (len(layer_ids) - 1) / max(count - 1, 1))
                                      for i in range(count)})
        names = [per_layer[layer_ids[position]] for position in selected_positions]
        print(f"Auto-selected NF4 tensors: {names}", flush=True)
    else:
        names = [x.strip() for x in requested_tensors.split(",") if x.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("--tensors must contain unique state-dict tensor names or 'auto'")
    missing = [name for name in names if name not in state]
    if missing:
        raise KeyError(f"Tensor names not found: {missing}")
    originals = {}
    capacities = {}
    for name in names:
        value = state[name].detach().float().cpu().contiguous().flatten()
        if value.numel() % 2:
            value = value[:-1].contiguous()
        originals[name] = value
        capacities[name] = int(value.numel())
    if len(bits) > sum(capacities.values()):
        raise ValueError(f"Framed corpus needs {len(bits)} bits; selected NF4 tensors hold {sum(capacities.values())}")
    allocations = [
        (name, start, end, bits[start:end])
        for name, start, end in allocate_payload_segments(capacities, len(bits))
    ]
    del state, model

    artifact_layers = {}
    tensor_reports = {}
    for name, start, end, segment in allocations:
        original = originals[name]
        qweight, qstate = quantize_4bit(original, quant_type="nf4", blocksize=args.blocksize)
        base = dequantize_4bit(qweight, quant_state=qstate).float().cpu().flatten()
        n = len(segment)
        positions = keyed_positions(key, name, int(original.numel()), n)
        original_codes = unpack_codes(qweight, original.numel())
        codebook = qstate.code.detach().float().cpu().flatten()
        absmax = qstate.absmax.detach().float().cpu().flatten()
        if codebook.numel() != 16:
            raise RuntimeError(f"{name}: expected 16 NF4 codebook entries")
        recon = torch.tensor(tensor_dequant(original_codes, codebook, absmax, int(qstate.blocksize)), dtype=torch.float32)
        layout_rmse = float(torch.mean((recon - base)**2).sqrt().item())
        if layout_rmse > args.layout_tolerance:
            raise RuntimeError(f"{name}: packed NF4 layout mismatch, RMSE={layout_rmse}")

        if args.method == "dce":
            modified = original_codes.copy()
            for bit, pos in zip(segment, positions):
                feasible = [c for c in range(16) if (c & 1) == bit]
                scale = float(absmax[pos // int(qstate.blocksize)])
                target = float(base[pos])
                modified[pos] = min(feasible, key=lambda c: ((float(codebook[c]) * scale - target)**2, c))
            stored_codes = modified
            # DCE receiver requires only packed codes and carrier locations.
            receiver_base = torch.empty(0, dtype=torch.float32)
            receiver_reference = torch.empty(0, dtype=torch.float32)
            receiver_codebook = codebook
            receiver_absmax = absmax
            receiver_blocksize = int(qstate.blocksize)
        else:
            # QSE remains reference-assisted: preserve its current receiver
            # contract rather than claiming a blind/artifact-only decoder.
            reference = original - base
            delta = max(float(reference.std().item()) * args.qse_margin, 1e-8)
            embedded = original - base
            for bit, pos in zip(segment, positions):
                embedded[pos] = reference[pos] + delta if bit else reference[pos] - delta
            qse_weight = base + embedded
            qse_q, qse_state = quantize_4bit(qse_weight, quant_type="nf4", blocksize=args.blocksize)
            stored_codes = unpack_codes(qse_q, original.numel())
            receiver_base = base[positions].clone()
            receiver_reference = reference[positions].clone()
            receiver_codebook = qse_state.code.detach().float().cpu().flatten()
            receiver_absmax = qse_state.absmax.detach().float().cpu().flatten()
            receiver_blocksize = int(qse_state.blocksize)

        packed = torch.tensor(list(pack_codes(stored_codes)), dtype=torch.uint8)
        # Fresh-process receiver only needs packed codes plus documented side information.
        artifact_layers[name] = {
            "packed_codes": packed,
            "num_values": int(original.numel()),
            "positions": torch.tensor(positions, dtype=torch.int64),
            "codebook": receiver_codebook,
            "absmax": receiver_absmax,
            "blocksize": receiver_blocksize,
            "qse_base_at_carriers": receiver_base,
            "qse_reference_at_carriers": receiver_reference,
            "bit_start": start,
            "bit_end": end,
        }
        dequant = torch.tensor(tensor_dequant(stored_codes, receiver_codebook, receiver_absmax, receiver_blocksize), dtype=torch.float32)
        tensor_reports[name] = {
            "num_values": int(original.numel()), "payload_bits": n,
            "bit_start": start, "bit_end": end,
            "packed_code_layout_reconstruction_rmse": layout_rmse,
            "weight_rmse_vs_fp16_source": float(torch.mean((dequant-original)**2).sqrt().item()),
            "weight_rmse_vs_clean_nf4": float(torch.mean((dequant-base)**2).sqrt().item()),
            "receiver_side_information": "none beyond keyed carrier positions" if args.method == "dce"
                else "clean NF4 values and reference residual at carriers",
        }

    metadata = {
        "schema": "nes.seven_method_nf4_artifact.v1",
        "artifact_kind": "multi_tensor_packed_nf4_experiment_artifact_not_hf_checkpoint",
        "method": args.method,
        "model_id": args.model,
        "quantizer": "bitsandbytes NF4",
        "blocksize_requested": args.blocksize,
        "payload_bits": len(bits),
        "corpus": corpus_summary(rows),
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "tensor_reports": tensor_reports,
        "receiver_contract": "packed-code parity" if args.method == "dce" else "reference-assisted QSE",
        "message_texts_written_to_metadata": False,
    }
    artifact = {"schema": metadata["schema"], "method": args.method,
                "metadata": metadata, "tensors": artifact_layers}
    out.parent.mkdir(parents=True, exist_ok=True)
    expected_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with expected_out.open("xb") as f:
            f.write(corpus)
        with out.open("xb") as f:
            torch.save(artifact, f)
        with report_out.open("x", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")
    except Exception:
        out.unlink(missing_ok=True)
        expected_out.unlink(missing_ok=True)
        report_out.unlink(missing_ok=True)
        raise
    print(json.dumps({"status": "embedded", "method": args.method, "model": args.model,
                      "message_count": len(rows), "payload_bits": len(bits),
                      "tensors": list(artifact_layers), "artifact": str(out),
                      "expected_corpus": str(expected_out), "report": str(report_out)}, indent=2))
    return metadata


def run_extract(args) -> dict:
    artifact_path = args.artifact.expanduser().resolve()
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if artifact.get("schema") != "nes.seven_method_nf4_artifact.v1":
        raise ValueError("unsupported NF4 artifact schema")
    method = artifact["method"]
    if method not in {"qse", "dce"}:
        raise ValueError(f"unsupported packed-NF4 method: {method}")
    entries = sorted(artifact["tensors"].items(), key=lambda kv: kv[1]["bit_start"])
    recovered = []
    for name, entry in entries:
        codes = unpack_codes(entry["packed_codes"], int(entry["num_values"]))
        positions = entry["positions"].tolist()
        if method == "dce":
            recovered.extend(codes[pos] & 1 for pos in positions)
        else:
            values = torch.tensor(tensor_dequant(codes, entry["codebook"], entry["absmax"], int(entry["blocksize"])), dtype=torch.float32)
            for index, pos in enumerate(positions):
                observed = float(values[pos]) - float(entry["qse_base_at_carriers"][index])
                recovered.append(int(observed >= float(entry["qse_reference_at_carriers"][index])))
    recovered_corpus = bits_to_bytes(recovered)
    metadata = artifact["metadata"]
    recovered_digest = hashlib.sha256(recovered_corpus).hexdigest()
    try:
        records = decode_corpus(recovered_corpus)
        decode_error = None
        recovered_messages = [{"id": r.id, "utf8_bytes": len(r.text.encode("utf-8"))} for r in records]
    except (ValueError, UnicodeDecodeError) as exc:
        # A failed receiver is still a useful experiment result: emit BER and
        # framing diagnostics rather than crashing before reporting the errors.
        records = []
        decode_error = f"{type(exc).__name__}: {exc}"
        recovered_messages = []
    report = {
        "schema": "nes.seven_method_nf4_extract.v1",
        "method": method,
        "artifact": str(artifact_path),
        "artifact_kind": metadata["artifact_kind"],
        "message_count": len(records) if decode_error is None else None,
        "recovered_messages": recovered_messages,
        "decode_error": decode_error,
        "recovered_corpus_sha256": recovered_digest,
        "embedded_manifest_digest_match": recovered_digest == metadata["corpus_sha256"],
        "receiver_contract": metadata["receiver_contract"],
        "message_texts_written_to_report": False,
    }
    if args.expected_corpus:
        expected = args.expected_corpus.expanduser().resolve().read_bytes()
        expected_bits = bytes_to_bits(expected)
        errors = sum(a != b for a, b in zip(expected_bits, recovered)) + abs(len(expected_bits)-len(recovered))
        report.update({"expected_bits": len(expected_bits), "recovered_bits": len(recovered),
                       "bit_errors": errors, "ber": errors / max(len(expected_bits), len(recovered), 1),
                       "exact_match": expected == recovered_corpus})
    print(json.dumps(report, indent=2))
    if not report["embedded_manifest_digest_match"] or report.get("exact_match") is False:
        raise SystemExit(2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    embed = sub.add_parser("embed")
    embed.add_argument("--model", required=True)
    embed.add_argument("--method", choices=("qse", "dce"), required=True)
    embed.add_argument("--tensors", required=True, help="comma-separated state-dict tensor names, or 'auto' to select an attention projection across up to five layers")
    embed.add_argument("--message", action="append", default=[])
    embed.add_argument("--messages-file", type=Path)
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--corpus-out", type=Path, required=True)
    embed.add_argument("--report", type=Path)
    embed.add_argument("--blocksize", type=int, default=64)
    embed.add_argument("--qse-margin", type=float, default=0.25)
    embed.add_argument("--layout-tolerance", type=float, default=1e-4)
    embed.add_argument("--test-key-hex", default="00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff")
    embed.add_argument("--local-files-only", action="store_true")
    extract = sub.add_parser("extract")
    extract.add_argument("--artifact", type=Path, required=True)
    extract.add_argument("--expected-corpus", type=Path)
    args = parser.parse_args()
    if args.command == "embed":
        if args.blocksize < 1 or args.qse_margin < 0 or args.layout_tolerance < 0:
            parser.error("blocksize must be positive; qse-margin/layout-tolerance must be non-negative")
        run_embed(args)
    else:
        run_extract(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
