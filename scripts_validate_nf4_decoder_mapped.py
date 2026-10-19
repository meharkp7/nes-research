#!/usr/bin/env python3
"""Contract B B1.2-v3-mapped: validate the existing test artifact while explicitly
mapping the earlier low-nibble-first test indices to bitsandbytes decoder order.

Run from nes-research/nes-llm:
  ../.venv/bin/python ../scripts_validate_nf4_decoder.py \
    --original ../cache/contract_b_nf4_probe_retry \
    --patched ../cache/contract_b_nf4_pairs_b12_retry2 \
    --output ../cache/contract_b_nf4_pairs_b12_retry2/decoder_validation_report.json

This is a read-only validation. CPU is the default because the current
environment's MPS path has known NF4 dequantization limitations.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, sys
from pathlib import Path
from typing import Any

import torch
import bitsandbytes.functional as bnbf
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


SEED = 20261009
TARGET_LAYER = "model.layers.0.self_attn.q_proj"


def raw_digest(t: torch.Tensor) -> str:
    x = t.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()


def load_model(path: str, device: str, token: str | None):
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    kwargs = {
        "quantization_config": qconfig,
        "device_map": {"": device},
        "trust_remote_code": True,
    }
    if token:
        kwargs["token"] = token
    return AutoModelForCausalLM.from_pretrained(path, **kwargs)


def find_layer(model):
    modules = dict(model.named_modules())
    if TARGET_LAYER not in modules:
        raise KeyError(f"Expected layer {TARGET_LAYER!r} not found")
    module = modules[TARGET_LAYER]
    weight = module.weight
    qs = getattr(weight, "quant_state", None)
    if qs is None or getattr(qs, "quant_type", None) != "nf4":
        raise RuntimeError(f"{TARGET_LAYER}.weight does not expose NF4 quant_state")
    return weight, qs


def unpack_decoder_order(packed: torch.Tensor) -> list[int]:
    """bitsandbytes decoder order: high nibble, then low nibble per byte."""
    vals = packed.detach().cpu().contiguous().reshape(-1).tolist()
    out = []
    for byte in vals:
        v = int(byte)
        out.append((v >> 4) & 0x0F)
        out.append(v & 0x0F)
    return out


def expected_test_positions(code_count: int, count: int = 256) -> list[int]:
    rng = random.Random(SEED)
    chosen = {0, 1, 2, 3, code_count - 2, code_count - 1}
    while len(chosen) < count:
        chosen.add(rng.randrange(code_count))
    return sorted(chosen)


def dequantize_layer(weight: torch.Tensor, qs: Any) -> torch.Tensor:
    packed = weight.detach().contiguous()
    # The CPU path is intentional; avoid MPS NF4 dequantization in this environment.
    packed_cpu = packed.to("cpu")
    # QuantState.to(device) mutates the state in some bitsandbytes versions
    # and returns None; never assign its return value. The model is loaded on
    # CPU here, so its quantization-state tensors should already be CPU-resident.
    # Verify that assumption explicitly and pass the QuantState object itself.
    state_tensors = []
    for attr in ("absmax", "code", "offset"):
        value = getattr(qs, attr, None)
        if isinstance(value, torch.Tensor):
            state_tensors.append((attr, value.device.type))
    state2 = getattr(qs, "state2", None)
    if state2 is not None:
        for attr in ("absmax", "code", "offset"):
            value = getattr(state2, attr, None)
            if isinstance(value, torch.Tensor):
                state_tensors.append((f"state2.{attr}", value.device.type))
    non_cpu = [(name, dev) for name, dev in state_tensors if dev != "cpu"]
    if non_cpu:
        raise RuntimeError(f"Expected CPU quantization state, found non-CPU tensors: {non_cpu}")
    decoded = bnbf.dequantize_4bit(packed_cpu, quant_state=qs)
    return decoded.detach().cpu().contiguous().reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True, help="Successful B1.1 original checkpoint")
    ap.add_argument("--patched", required=True, help="Successful B1.2-v2 patched checkpoint")
    ap.add_argument("--output", required=True, help="JSON report output path")
    ap.add_argument("--device", choices=("cpu",), default="cpu",
                    help="CPU-only for decoder validation in this environment")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    report: dict[str, Any] = {
        "stage": "B1.2-v3-mapped-decoder-validation",
        "status": "ERROR",
        "device": args.device,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "bitsandbytes": __import__("bitsandbytes").__version__,
        "target_layer": TARGET_LAYER,
        "checks": {},
    }
    original_model = patched_model = None
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[B1.2-v3-mapped] Loading original checkpoint on CPU: {args.original}")
        original_model = load_model(args.original, args.device, args.token)
        original_weight, original_qs = find_layer(original_model)
        original_packed = original_weight.detach().cpu().contiguous().clone()
        original_state = original_qs
        original_codes = unpack_decoder_order(original_packed)
        original_decoded = dequantize_layer(original_weight, original_qs)
        report["original"] = {
            "packed_shape": list(original_packed.shape),
            "packed_sha256": raw_digest(original_packed),
            "decoded_shape_flat": list(original_decoded.shape),
            "decoded_dtype": str(original_decoded.dtype),
            "decoded_sha256": raw_digest(original_decoded),
            "logical_code_count": len(original_codes),
        }
        del original_model
        original_model = None

        print(f"[B1.2-v3-mapped] Loading patched checkpoint on CPU: {args.patched}")
        patched_model = load_model(args.patched, args.device, args.token)
        patched_weight, patched_qs = find_layer(patched_model)
        patched_packed = patched_weight.detach().cpu().contiguous().clone()
        patched_codes = unpack_decoder_order(patched_packed)
        patched_decoded = dequantize_layer(patched_weight, patched_qs)

        if len(original_codes) != len(patched_codes):
            raise ValueError("Original and patched logical code counts differ")
        # The B1.2-v2 writer numbered physical nibble positions low-then-high,
        # while bitsandbytes dequantization exposes high-then-low. Therefore the
        # same byte's paired positions swap: decoder_index = writer_index XOR 1.
        writer_positions = expected_test_positions(len(original_codes))
        writer_target_bits = {
            idx: ((idx * 1103515245 + 12345) >> 8) & 1 for idx in writer_positions
        }
        selected = {idx ^ 1 for idx in writer_positions}
        expected_target_bits = {idx ^ 1: bit for idx, bit in writer_target_bits.items()}
        changed_code_positions = [
            i for i, (a, b) in enumerate(zip(original_codes, patched_codes)) if a != b
        ]
        changed_decoded_positions = torch.nonzero(
            original_decoded != patched_decoded, as_tuple=False
        ).reshape(-1).tolist()
        changed_decoded_set = set(changed_decoded_positions)
        changed_code_set = set(changed_code_positions)

        pair_ids_preserved = all(
            patched_codes[i] // 2 == original_codes[i] // 2 for i in selected
        )
        selected_bits_match = all(
            patched_codes[i] % 2 == expected_target_bits[i] for i in selected
        )
        all_code_changes_selected = changed_code_set.issubset(selected)
        code_changes_match_decoded_positions = changed_code_set == changed_decoded_set
        no_unselected_decoded_changes = changed_decoded_set.issubset(selected)
        expected_changed = {
            i for i in selected
            if original_codes[i] != 2 * (original_codes[i] // 2) + expected_target_bits[i]
        }
        exact_expected_code_changes = changed_code_set == expected_changed
        same_packed_size = original_packed.shape == patched_packed.shape
        report.update({
            "patched": {
                "packed_shape": list(patched_packed.shape),
                "packed_sha256": raw_digest(patched_packed),
                "decoded_shape_flat": list(patched_decoded.shape),
                "decoded_dtype": str(patched_decoded.dtype),
                "decoded_sha256": raw_digest(patched_decoded),
            },
            "writer_positions_requested": len(writer_positions),
            "decoder_order_positions_selected": len(selected),
            "index_mapping": "decoder_index = writer_index XOR 1",
            "code_positions_changed": len(changed_code_positions),
            "decoded_positions_changed": len(changed_decoded_positions),
            "changed_code_positions_first_32": changed_code_positions[:32],
            "changed_decoded_positions_first_32": changed_decoded_positions[:32],
            "checks": {
                "same_packed_shape": same_packed_size,
                "packed_bytes_differ": raw_digest(original_packed) != raw_digest(patched_packed),
                "logical_code_changes_are_expected": exact_expected_code_changes,
                "all_code_changes_within_selected_positions": all_code_changes_selected,
                "selected_pair_ids_preserved": pair_ids_preserved,
                "selected_target_bits_match": selected_bits_match,
                "decoder_changes_exactly_the_changed_code_positions": code_changes_match_decoded_positions,
                "no_unselected_decoded_positions_changed": no_unselected_decoded_changes,
                "decoded_tensor_shape_unchanged": original_decoded.shape == patched_decoded.shape,
                "decoded_tensor_values_changed": not torch.equal(original_decoded, patched_decoded),
            },
        })
        report["status"] = "PASS" if all(report["checks"].values()) else "FAIL"
        report["interpretation"] = (
            "Actual bitsandbytes NF4 dequantization changed exactly the expected logical positions "
            "and preserved the within-pair identity for this test mutation. This is still not a "
            "payload-recovery, carrier-reproducibility, utility, detectability, robustness, or novelty result."
            if report["status"] == "PASS" else
            "Decoder-level semantics did not match all expected invariants. Do not proceed to payload encoding."
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[B1.2-v3-mapped] Report written: {out}")
        print(f"[B1.2-v3-mapped] Status: {report['status']}")
        if patched_model is not None:
            del patched_model
        if original_model is not None:
            del original_model


if __name__ == "__main__":
    main()
