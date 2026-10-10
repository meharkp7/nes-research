#!/usr/bin/env python3
"""Create a clean NF4 baseline artifact for a single model tensor.

The baseline is a fresh quantization of the requested source model revision.
It is not assumed to be the exact historical source of an older embedding run
unless that run recorded matching revision provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from bitsandbytes.functional import quantize_4bit, dequantize_4bit
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.nf4_artifact_codec import unpack_codes  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_clean_control(
    *,
    model_id: str,
    revision: str,
    tensor_key: str,
    blocksize: int,
    artifact_path: Path,
    report_path: Path,
    local_files_only: bool = True,
    compress_statistics: bool = True,
) -> dict:
    artifact_path = artifact_path.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    if artifact_path == report_path:
        raise ValueError("artifact and report paths must differ")
    for path in (artifact_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")
    if blocksize < 1:
        raise ValueError("blocksize must be positive")
    if not revision.strip():
        raise ValueError("revision must be an explicit non-empty commit/ref")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    state = model.state_dict()
    if tensor_key not in state:
        raise KeyError(f"Tensor not found in model state dict: {tensor_key}")
    source = state[tensor_key].detach().float().cpu().contiguous()
    source_shape = tuple(source.shape)
    source_numel = int(source.numel())
    flattened = source.flatten()
    if source_numel % 2:
        flattened = flattened[:-1].contiguous()
    del state, model

    packed, quant_state = quantize_4bit(
        flattened, quant_type="nf4", blocksize=blocksize,
        compress_statistics=compress_statistics,
    )
    actual_blocksize = int(quant_state.blocksize)
    # Use bitsandbytes' own dequantizer so nested/compressed scale state, if
    # present in this runtime, is handled by the quantizer implementation.
    dequantized = dequantize_4bit(packed, quant_state=quant_state).float().cpu().flatten()
    source_for_rmse = flattened.detach().float().cpu()
    rmse = float(torch.mean((dequantized - source_for_rmse) ** 2).sqrt().item())

    report = {
        "schema": "nes.clean_nf4_control.v1",
        "artifact_kind": "single_tensor_clean_nf4_control_not_hf_checkpoint",
        "model_id": model_id,
        "source_revision": revision,
        "source_revision_is_historical_qse_match": "UNVERIFIED",
        "tensor_key": tensor_key,
        "tensor_shape": list(source_shape),
        "num_values_original": source_numel,
        "num_values_quantized": int(flattened.numel()),
        "quantizer": "bitsandbytes NF4",
        "quantizer_config": {
            "quant_type": "nf4",
            "blocksize_requested": blocksize,
            "blocksize_observed": actual_blocksize,
            "compress_statistics": compress_statistics,
            "note": "Runtime-observed clean control settings; historical QSE settings were only partially recorded.",
        },
        "weight_rmse_vs_fp16_source": rmse,
        "payload_bits": 0,
        "receiver_contract": "not_applicable_clean_control",
        "source_dtype_loaded": "float16",
        "flattening_order": "row-major contiguous",
        "odd_value_truncated_for_packing": bool(source_numel % 2),
        "artifact_path": str(artifact_path),
        "artifact_sha256": None,
    }
    artifact = {
        "schema": report["schema"],
        "metadata": dict(report),
        "tensor": {
            "packed_codes": packed.detach().cpu().contiguous(),
            "dequantized_values": dequantized,
            "num_values": int(flattened.numel()),
            "tensor_shape": list(source_shape),
            "blocksize": actual_blocksize,
        },
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with artifact_path.open("xb") as stream:
            torch.save(artifact, stream)
        report["artifact_sha256"] = sha256_file(artifact_path)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except Exception:
        artifact_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True, help="Explicit model commit/ref to pin")
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--blocksize", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-remote-files", action="store_true",
                        help="Allow downloading model files if absent locally")
    parser.add_argument("--compress-statistics",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use nested NF4 statistics (default: true)")
    args = parser.parse_args()
    report = create_clean_control(
        model_id=args.model,
        revision=args.revision,
        tensor_key=args.tensor,
        blocksize=args.blocksize,
        artifact_path=args.output,
        report_path=args.report,
        local_files_only=not args.allow_remote_files,
        compress_statistics=args.compress_statistics,
    )
    print(json.dumps({
        "status": "clean_control_written",
        "artifact_path": report["artifact_path"],
        "artifact_sha256": report["artifact_sha256"],
        "model_id": report["model_id"],
        "source_revision": report["source_revision"],
        "historical_qse_revision_match": report["source_revision_is_historical_qse_match"],
        "tensor_key": report["tensor_key"],
        "tensor_shape": report["tensor_shape"],
        "blocksize_observed": report["quantizer_config"]["blocksize_observed"],
        "compress_statistics": report["quantizer_config"]["compress_statistics"],
        "weight_rmse_vs_fp16_source": report["weight_rmse_vs_fp16_source"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
