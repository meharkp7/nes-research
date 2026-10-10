#!/usr/bin/env python3
"""Run a selected-tensor 10% magnitude-pruning + fresh NF4 requantization attack.

This is an expensive local lifecycle experiment, not an in-memory code mutation.
It never overwrites the input checkpoint or existing output/work directories.
Run from nes-llm/:
../.venv/bin/python scripts/contract_b_nf4_pruning_attack.py \
  --stego ../cache/contract_b_nf4_b14_10k \
  --output-dir ../cache/contract_b_b14_pruned10_nf4_20261010
Then run the normal B1.4 receiver against the output directory.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.contract_b_nf4_b14 import (
    DEFAULT_TENSOR_KEY,
    TEST_KEY_HEX,
    code_at,
    extract,
    find_shard,
    from_bits,
    positions,
    read_tensor,
)


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def bypass_bnb4bit_reverse_conversion_after_dequantization(model, report: dict) -> None:
    """Bypass only the known NF4 deserialization mapping after weights are dequantized.

    Transformers 5.16.1 attaches a Bnb4bitDeserialize mapping when loading a
    packed NF4 checkpoint. Its reverse operation is unimplemented, so
    save_pretrained() fails even after model.dequantize(). This guarded,
    version-sensitive workaround must only be called after dequantization.
    It refuses to bypass any unknown conversion operation.
    """
    conversions = getattr(model, "_weight_conversions", None)
    if not conversions:
        report["weight_conversion_reverse_bypassed_after_dequantization"] = False
        return

    operations = [
        op
        for conversion in conversions
        for op in getattr(conversion, "operations", [])
    ]
    operation_types = sorted({type(op).__name__ for op in operations})
    report["original_weight_conversion_types"] = [
        type(conversion).__name__ for conversion in conversions
    ]
    report["original_weight_conversion_operations"] = operation_types

    if not operations or any(name != "Bnb4bitDeserialize" for name in operation_types):
        raise RuntimeError(
            "Unexpected weight-conversion operations; refusing to bypass them: "
            f"{operation_types}"
        )

    model._weight_conversions = []
    report["weight_conversion_reverse_bypassed_after_dequantization"] = True


def recoverability_diagnostic(packed: bytes, expected_payload: bytes, key: bytes,
                              tensor_key: str) -> dict:
    """Measure carrier BER independently of envelope checksum acceptance."""
    n_codes = len(packed) * 2
    expected_envelope_bytes = len(expected_payload) + 16
    carrier_positions = positions(key, n_codes, expected_envelope_bytes * 8, tensor_key)
    decoded_envelope = from_bits([code_at(packed, p) & 1 for p in carrier_positions])
    header_valid = (
        decoded_envelope[:4] == b"NB14"
        and len(decoded_envelope) >= 16
        and int.from_bytes(decoded_envelope[4:8], "big") == len(expected_payload)
    )
    if not header_valid:
        return {
            "header_valid": False,
            "checksum_valid": False,
            "exact_payload_recovery": False,
            "bit_errors": None,
            "ber": None,
            "recovered_payload_sha256": None,
            "error": "Envelope header invalid after requantization",
        }
    recovered = decoded_envelope[8:8 + len(expected_payload)]
    checksum = decoded_envelope[8 + len(expected_payload):]
    checksum_valid = checksum == hashlib.sha256(recovered).digest()[:8]
    bit_errors = sum((a ^ b).bit_count() for a, b in zip(expected_payload, recovered))
    return {
        "header_valid": True,
        "checksum_valid": checksum_valid,
        "exact_payload_recovery": recovered == expected_payload,
        "bit_errors": bit_errors,
        "ber": bit_errors / (len(expected_payload) * 8),
        "recovered_payload_sha256": hashlib.sha256(recovered).hexdigest(),
        "error": None,
    }


def run(args: argparse.Namespace) -> int:
    source = Path(args.stego).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    work = Path(args.work_dir).expanduser().resolve() if args.work_dir else output.with_name(output.name + "_floating_intermediate")
    report_path = output.parent / (output.name + "_requantization_report.json")
    if not source.is_dir():
        raise FileNotFoundError(f"Stego source checkpoint not found: {source}")
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    if work.exists():
        raise FileExistsError(f"Intermediate already exists; refusing to overwrite: {work}")
    if source == output or source in output.parents:
        raise ValueError("Output must not equal or be nested inside the source checkpoint")
    if source == work or source in work.parents:
        raise ValueError("Intermediate must not equal or be nested inside the source checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    work.parent.mkdir(parents=True, exist_ok=True)
    source_bytes = directory_bytes(source)
    # Conservative reserve for a floating-point copy, quantized output, temporary serialization,
    # and the source. This is a preflight estimate, not an exact memory/disk guarantee.
    required_free = source_bytes * 5 + 1024**3
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < required_free:
        raise OSError(
            f"Insufficient free disk space: need a conservative estimate of "
            f"{required_free:,} bytes; have {free_bytes:,}. No model was loaded."
        )

    key = bytes.fromhex(args.test_key_hex)
    source_shard = find_shard(source, args.tensor_key)
    _, _, _, _, source_packed = read_tensor(source_shard, args.tensor_key)
    expected_payload, _ = extract(source_packed, key, args.tensor_key)
    source_payload_hash = hashlib.sha256(expected_payload).hexdigest()

    report = {
        "stage": "Contract-B-B1.4-real-NF4-requantization",
        "status": "RUNNING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source),
        "source_checkpoint_bytes": source_bytes,
        "intermediate_floating_checkpoint": str(work),
        "requantized_checkpoint": str(output),
        "tensor_key": args.tensor_key,
        "source_payload_bits": len(expected_payload) * 8,
        "source_payload_sha256": source_payload_hash,
        "source_packed_tensor_sha256": hashlib.sha256(source_packed).hexdigest(),
        "device": args.device,
        "transformation": [
            "load B1.4 stego checkpoint as NF4",
            "dequantize model weights to floating point",
            "zero the 10% smallest-magnitude entries of the selected q_proj tensor",
            "save an intermediate non-quantized checkpoint",
            "reload intermediate with fresh BitsAndBytes NF4 quantization",
            "save fresh NF4 checkpoint",
            "diagnose carrier BER and envelope checksum",
        ],
        "scope": (
            "Real model-level selected-tensor magnitude-pruning plus NF4 dequantize/save/requantize lifecycle test. "
            "Does not by itself establish utility, stealth, or security."
        ),
    }

    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        print(f"[B1.4 requant] Loading source NF4 model: {source}")
        model = AutoModelForCausalLM.from_pretrained(
            str(source),
            quantization_config=quant_config,
            device_map={"": args.device},
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()
        if not hasattr(model, "dequantize"):
            raise RuntimeError(
                "This Transformers model has no model.dequantize() API; "
                "stopping rather than silently substituting a different transform."
            )
        print("[B1.4 requant] Dequantizing model weights...")
        model.dequantize()

        # This workaround is deliberately guarded and tested separately.
        bypass_bnb4bit_reverse_conversion_after_dequantization(model, report)

        module_name = args.tensor_key.removesuffix(".weight")
        modules = dict(model.named_modules())
        if module_name not in modules:
            raise KeyError(f"Selected module missing after dequantization: {module_name}")
        float_weight = modules[module_name].weight
        if not float_weight.dtype.is_floating_point:
            raise TypeError(f"Selected weight did not dequantize: dtype={float_weight.dtype}")

        # Explicit unstructured magnitude pruning on the selected tensor only.
        # This is intentionally a combined pruning + fresh-NF4-requantization attack.
        prune_fraction = 0.10
        flat_abs = float_weight.detach().abs().reshape(-1)
        total_values = flat_abs.numel()
        prune_count = int(total_values * prune_fraction)
        if prune_count <= 0 or prune_count >= total_values:
            raise ValueError(f"Invalid prune count {prune_count} for {total_values} values")
        prune_indices = torch.topk(flat_abs, k=prune_count, largest=False, sorted=False).indices
        flat_weight = float_weight.view(-1)
        before_nonzero = int(torch.count_nonzero(flat_weight).item())
        flat_weight[prune_indices] = 0
        after_nonzero = int(torch.count_nonzero(flat_weight).item())
        report["pruning"] = {
            "scope": "selected tensor only",
            "tensor_key": args.tensor_key,
            "method": "unstructured magnitude pruning; zero exactly floor(10% of entries) lowest-|weight| positions before requantization",
            "requested_fraction": prune_fraction,
            "tensor_num_values": total_values,
            "pruned_positions": prune_count,
            "nonzero_before": before_nonzero,
            "nonzero_after": after_nonzero,
            "actual_newly_zeroed": before_nonzero - after_nonzero,
            "note": "This attack combines pruning with a fresh NF4 quantization pass; payload loss cannot be attributed to pruning alone."
        }
        del flat_abs, prune_indices, flat_weight

        # Ensure the intermediate config cannot imply that its floating weights are still NF4.
        if hasattr(model.config, "quantization_config"):
            model.config.quantization_config = None
        print(f"[B1.4 requant] Saving floating-point intermediate: {work}")
        model.save_pretrained(work, safe_serialization=True, max_shard_size="2GB")
        report["intermediate_selected_weight_dtype"] = str(float_weight.dtype)
        del model, modules, float_weight
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Verify the selected tensor in the intermediate is genuinely floating-point.
        intermediate_shard = find_shard(work, args.tensor_key)
        with intermediate_shard.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_size))
        intermediate_record = header.get(args.tensor_key)
        if not isinstance(intermediate_record, dict):
            raise KeyError(f"Selected tensor absent from FP16 intermediate: {args.tensor_key}")
        report["intermediate_tensor_dtype"] = intermediate_record.get("dtype")
        if intermediate_record.get("dtype") not in {"F16", "BF16", "F32", "F64"}:
            raise TypeError(
                "Intermediate tensor is not serialized as floating-point; refusing to "
                f"call this a fresh requantization (dtype={intermediate_record.get('dtype')!r})."
            )

        print("[B1.4 requant] Reloading floating checkpoint with fresh NF4 quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            str(work),
            quantization_config=quant_config,
            device_map={"": args.device},
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()
        print(f"[B1.4 requant] Saving requantized checkpoint: {output}")
        model.save_pretrained(output, safe_serialization=True, max_shard_size="2GB")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        out_shard = find_shard(output, args.tensor_key)
        _, out_record, _, _, out_packed = read_tensor(out_shard, args.tensor_key)
        report["output_tensor_dtype"] = out_record.get("dtype")
        report["output_packed_tensor_sha256"] = hashlib.sha256(out_packed).hexdigest()
        if out_record.get("dtype") != "U8":
            raise TypeError(
                "Requantized selected tensor is not serialized as packed U8 NF4 codes: "
                f"{out_record.get('dtype')!r}"
            )
        report["recovery"] = recoverability_diagnostic(
            out_packed, expected_payload, key, args.tensor_key
        )
        report["status"] = (
            "PASS_TRANSFORMATION_AND_RECOVERY"
            if report["recovery"]["exact_payload_recovery"]
            and report["recovery"]["checksum_valid"]
            else "TRANSFORMATION_COMPLETED_RECOVERY_FAILED"
        )
        report["interpretation"] = (
            "This is a combined selected-tensor magnitude-pruning plus fresh-NF4-requantization attack. "
            "PASS means the combined transformation completed and payload recovery was exact. "
            "TRANSFORMATION_COMPLETED_RECOVERY_FAILED means the combined transformation ran but the "
            "carrier payload did not survive. Because fresh requantization is included, recovery loss "
            "cannot be attributed to pruning alone. Inspect BER, header, and checksum."
        )
        return_code = 0 if report["status"] == "PASS_TRANSFORMATION_AND_RECOVERY" else 3
    except Exception as exc:
        report["status"] = "TRANSFORMATION_ERROR"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
        report["interpretation"] = (
            "The lifecycle transformation did not complete successfully. Preserve the "
            "source and intermediate/output artifacts for diagnosis; do not claim a "
            "recovery result from this run."
        )
        return_code = 2
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"[B1.4 requant] Status: {report['status']}")
        if "recovery" in report:
            print(f"[B1.4 requant] Recovery: {json.dumps(report['recovery'], sort_keys=True)}")
        if report.get("error"):
            print(f"[B1.4 requant] Error: {report['error']}")
        print(f"[B1.4 requant] Report: {report_path}")
        print("[B1.4 requant] Source checkpoint was not modified.")
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stego", required=True, help="Pristine B1.4 stego checkpoint")
    parser.add_argument("--output-dir", required=True, help="New output directory for fresh NF4 checkpoint")
    parser.add_argument("--work-dir", default="", help="New FP16 intermediate directory (must not exist)")
    parser.add_argument("--tensor-key", default=DEFAULT_TENSOR_KEY)
    parser.add_argument("--test-key-hex", default=TEST_KEY_HEX)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
