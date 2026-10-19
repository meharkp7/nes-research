#!/usr/bin/env python3
"""NES Contract B B1.3 short-payload artifact-only recovery pilot.

Run from nes-research/nes-llm:
  ../.venv/bin/python ../scripts_probe_nf4_b13_short_payload.py \
    --original ../cache/contract_b_nf4_probe_retry \
    --output-dir ../cache/contract_b_nf4_b13_short_payload

The test payload is deliberately unencrypted. The key is used only for
reproducible keyed carrier selection. This is a mechanism pilot, NOT a
security claim, and it does not modify NES production code.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
import struct
import sys
import platform
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

TARGET_LAYER = "model.layers.0.self_attn.q_proj"
TENSOR_KEY = TARGET_LAYER + ".weight"
DOMAIN = b"NES-B1.3-carrier-v1\x00"
NONCE = b"NES-B1.3-test-nonce-v1"
MAGIC = b"NB13"
CHECKSUM_LEN = 8
DEFAULT_TEST_KEY_HEX = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
TEST_PAYLOAD = b"NES-B13 artifact-only recovery test"


def load_safetensors_header(path: Path):
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 0 or header_len > path.stat().st_size - 8:
            raise ValueError(f"Invalid safetensors header length {header_len} in {path}")
        header_bytes = f.read(header_len)
    header = json.loads(header_bytes)
    return header, 8 + header_len


def find_shard(model_dir: Path) -> Path:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_name = index.get("weight_map", {}).get(TENSOR_KEY)
        if not shard_name:
            raise KeyError(f"{TENSOR_KEY} not present in {index_path}")
        shard = model_dir / shard_name
        if not shard.exists():
            raise FileNotFoundError(shard)
        return shard
    shard = model_dir / "model.safetensors"
    if not shard.exists():
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json found in {model_dir}"
        )
    return shard


def get_tensor_record(header: dict[str, Any]) -> dict[str, Any]:
    rec = header.get(TENSOR_KEY)
    if not isinstance(rec, dict):
        raise KeyError(f"Tensor {TENSOR_KEY!r} not found in safetensors header")
    if rec.get("dtype") != "U8":
        raise TypeError(f"Expected packed tensor dtype U8, found {rec.get('dtype')}")
    shape = rec.get("shape")
    if shape != [2097152, 1]:
        raise ValueError(f"Unexpected packed tensor shape {shape}; expected [2097152, 1]")
    return rec


def logical_code(packed: bytes | bytearray, logical_index: int) -> int:
    """bitsandbytes NF4 logical order: high nibble, then low nibble per byte."""
    byte = packed[logical_index // 2]
    return ((byte >> 4) & 0x0F) if logical_index % 2 == 0 else (byte & 0x0F)


def set_logical_code(packed: bytearray, logical_index: int, code: int) -> None:
    if not 0 <= code <= 15:
        raise ValueError("NF4 code must be in [0, 15]")
    byte_idx = logical_index // 2
    old = packed[byte_idx]
    if logical_index % 2 == 0:
        packed[byte_idx] = (old & 0x0F) | ((code & 0x0F) << 4)
    else:
        packed[byte_idx] = (old & 0xF0) | (code & 0x0F)


def keyed_positions(key: bytes, code_count: int, count: int) -> list[int]:
    """Stable keyed pseudo-random positions; receiver needs only key + protocol."""
    if count > code_count:
        raise ValueError("Requested more carrier positions than available codes")
    chosen: list[int] = []
    seen: set[int] = set()
    counter = 0
    while len(chosen) < count:
        block = hmac.new(
            key,
            DOMAIN + NONCE + TENSOR_KEY.encode("utf-8") + counter.to_bytes(8, "big"),
            hashlib.sha256,
        ).digest()
        counter += 1
        for offset in range(0, len(block), 4):
            idx = int.from_bytes(block[offset:offset + 4], "big") % code_count
            if idx not in seen:
                seen.add(idx)
                chosen.append(idx)
                if len(chosen) == count:
                    break
    return chosen


def envelope(payload: bytes) -> bytes:
    return MAGIC + len(payload).to_bytes(4, "big") + payload + hashlib.sha256(payload).digest()[:CHECKSUM_LEN]


def bits_from_bytes(data: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def bytes_from_bits(bits: list[int]) -> bytes:
    if len(bits) % 8:
        raise ValueError("Bit count must be byte-aligned")
    out = bytearray()
    for start in range(0, len(bits), 8):
        value = 0
        for bit in bits[start:start + 8]:
            value = (value << 1) | int(bit)
        out.append(value)
    return bytes(out)


def read_packed_tensor(path: Path):
    header, data_start = load_safetensors_header(path)
    rec = get_tensor_record(header)
    start, end = rec["data_offsets"]
    if end - start != 2097152:
        raise ValueError(f"Unexpected packed tensor byte count {end-start}")
    with path.open("rb") as f:
        f.seek(data_start + start)
        packed = f.read(end - start)
    if len(packed) != end - start:
        raise IOError("Could not read complete packed tensor")
    return header, data_start, rec, start, end, packed


def patch_checkpoint(original_dir: Path, output_dir: Path, key: bytes, payload: bytes):
    shard_src = find_shard(original_dir)
    header, data_start, rec, start, end, packed_bytes = read_packed_tensor(shard_src)
    original_codes = [logical_code(packed_bytes, i) for i in range(len(packed_bytes) * 2)]
    wrapped = envelope(payload)
    payload_bits = bits_from_bytes(wrapped)
    positions = keyed_positions(key, len(original_codes), len(payload_bits))
    packed_mut = bytearray(packed_bytes)
    for pos, bit in zip(positions, payload_bits):
        old_code = logical_code(packed_bytes, pos)
        new_code = (old_code & 0x0E) | bit
        set_logical_code(packed_mut, pos, new_code)

    # Invariants before writing.
    changed = [i for i, (a, b) in enumerate(zip(original_codes, (
        logical_code(packed_mut, j) for j in range(len(original_codes))
    ))) if a != b]
    selected = set(positions)
    if any(i not in selected for i in changed):
        raise AssertionError("A non-carrier code changed")
    if any(
        logical_code(packed_bytes, p) // 2 != logical_code(packed_mut, p) // 2
        for p in positions
    ):
        raise AssertionError("Pair ID changed at a carrier")

    shutil.copytree(original_dir, output_dir)
    shard_out = output_dir / shard_src.name
    with shard_out.open("r+b") as f:
        f.seek(data_start + start)
        f.write(packed_mut)
        f.flush()
        os.fsync(f.fileno())

    # Verify the raw serialized target bytes and all non-target file bytes.
    _, _, _, _, _, reread = read_packed_tensor(shard_out)
    if reread != bytes(packed_mut):
        raise AssertionError("Patched tensor bytes differ after write")
    report = {
        "stage": "B1.3-short-payload",
        "status": "RUNNING",
        "original_checkpoint": str(original_dir.resolve()),
        "stego_checkpoint": str(output_dir.resolve()),
        "safetensors_shard": shard_src.name,
        "tensor_key": TENSOR_KEY,
        "packed_tensor_shape": rec["shape"],
        "logical_code_count": len(original_codes),
        "payload_length_bytes": len(payload),
        "envelope_length_bytes": len(wrapped),
        "carrier_count": len(positions),
        "changed_code_count": len(changed),
        "positions_first_16": positions[:16],
        "key_id_sha256_prefix": hashlib.sha256(key).hexdigest()[:16],
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "envelope_sha256": hashlib.sha256(wrapped).hexdigest(),
        "checks": {
            "all_changed_codes_are_selected_carriers": all(i in selected for i in changed),
            "pair_ids_preserved": all(
                logical_code(packed_bytes, p) // 2 == logical_code(packed_mut, p) // 2
                for p in positions
            ),
            "all_carrier_parities_match_envelope": all(
                logical_code(packed_mut, p) % 2 == bit
                for p, bit in zip(positions, payload_bits)
            ),
            "patched_tensor_bytes_read_back_exactly": reread == bytes(packed_mut),
        },
    }
    return report, payload_bits


def receiver_extract(stego_dir: Path, key: bytes, expected_tensor_bytes: int):
    """Receiver phase: reads only the stego checkpoint and key/protocol."""
    shard = find_shard(stego_dir)
    _, _, _, _, _, packed = read_packed_tensor(shard)
    code_count = len(packed) * 2
    # Header is MAGIC(4) + length(4) = 64 bits. Read enough positions to parse it.
    header_positions = keyed_positions(key, code_count, 64)
    header_bits = [logical_code(packed, p) % 2 for p in header_positions]
    header_bytes = bytes_from_bits(header_bits)
    if header_bytes[:4] != MAGIC:
        raise ValueError(f"Bad receiver magic: {header_bytes[:4]!r}")
    payload_len = int.from_bytes(header_bytes[4:8], "big")
    total_bytes = 8 + payload_len + CHECKSUM_LEN
    total_bits = total_bytes * 8
    positions = keyed_positions(key, code_count, total_bits)
    bits = [logical_code(packed, p) % 2 for p in positions]
    wrapped = bytes_from_bits(bits)
    if wrapped[:4] != MAGIC:
        raise ValueError("Envelope magic mismatch during full extraction")
    extracted_len = int.from_bytes(wrapped[4:8], "big")
    if extracted_len != payload_len:
        raise ValueError("Payload length header mismatch")
    payload = wrapped[8:8 + payload_len]
    checksum = wrapped[8 + payload_len:]
    if checksum != hashlib.sha256(payload).digest()[:CHECKSUM_LEN]:
        raise ValueError("Payload checksum mismatch")
    return payload, {
        "receiver_tensor_sha256": hashlib.sha256(packed).hexdigest(),
        "receiver_carrier_count": len(positions),
        "receiver_envelope_length_bytes": len(wrapped),
        "receiver_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "receiver_magic_valid": True,
        "receiver_length_valid": True,
        "receiver_checksum_valid": True,
    }


def load_and_check_patched_model(stego_dir: Path, key: bytes, payload: bytes, device: str):
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(stego_dir),
        quantization_config=qconfig,
        device_map={"": device},
        trust_remote_code=True,
    )
    modules = dict(model.named_modules())
    if TARGET_LAYER not in modules:
        raise KeyError(f"Target layer {TARGET_LAYER} missing after reload")
    weight = modules[TARGET_LAYER].weight
    qs = getattr(weight, "quant_state", None)
    if qs is None or getattr(qs, "quant_type", None) != "nf4":
        raise RuntimeError("Reloaded target weight does not expose NF4 quantization state")
    packed = weight.data.detach().cpu().contiguous().reshape(-1).tolist()
    # Model weight storage is U8 bytes. Extract from loaded artifact, not the raw file.
    code_count = len(packed) * 2
    wrapped_header_positions = keyed_positions(key, code_count, 64)
    def code_at(i):
        byte = int(packed[i // 2])
        return ((byte >> 4) & 0x0F) if i % 2 == 0 else (byte & 0x0F)
    header = bytes_from_bits([code_at(p) % 2 for p in wrapped_header_positions])
    if header[:4] != MAGIC:
        raise ValueError("Loaded model packed codes do not expose expected envelope magic")
    payload_len = int.from_bytes(header[4:8], "big")
    total_bits = (8 + payload_len + CHECKSUM_LEN) * 8
    positions = keyed_positions(key, code_count, total_bits)
    wrapped = bytes_from_bits([code_at(p) % 2 for p in positions])
    recovered = wrapped[8:8 + payload_len]
    if wrapped[:4] != MAGIC or recovered != payload:
        raise ValueError("Payload recovery from reloaded model weights failed")
    if wrapped[8 + payload_len:] != hashlib.sha256(recovered).digest()[:CHECKSUM_LEN]:
        raise ValueError("Checksum from reloaded model weights failed")
    return {
        "reloaded_model_nf4": True,
        "reloaded_model_carrier_count": len(positions),
        "reloaded_model_payload_sha256": hashlib.sha256(recovered).hexdigest(),
        "reloaded_model_payload_matches": recovered == payload,
        "reloaded_model_checksum_valid": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True, help="Successful B1.1 original checkpoint directory")
    ap.add_argument("--output-dir", required=True, help="New, non-existing output directory")
    ap.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                    help="Use CPU by default for more predictable NF4 handling")
    ap.add_argument("--test-key-hex", default=DEFAULT_TEST_KEY_HEX)
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--skip-model-reload", action="store_true",
                    help="Skip Transformers reload check; not recommended for a B1.3 pass")
    args = ap.parse_args()

    original_dir = Path(args.original).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not original_dir.is_dir():
        raise FileNotFoundError(original_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}; choose a new directory")
    try:
        key = bytes.fromhex(args.test_key_hex)
    except ValueError as exc:
        raise ValueError("--test-key-hex must be valid hex") from exc
    if len(key) < 16:
        raise ValueError("Test key must be at least 16 bytes")

    report_path = output_dir.parent / (output_dir.name + "_b13_report.json")
    report: dict[str, Any] = {
        "stage": "B1.3-short-payload",
        "status": "ERROR",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": args.device,
        "payload_note": "Unencrypted synthetic test payload; test-only keyed carrier selection, no security claim.",
        "checks": {},
    }
    try:
        payload = TEST_PAYLOAD
        print(f"[B1.3] Original checkpoint: {original_dir}")
        print(f"[B1.3] Creating test artifact: {output_dir}")
        sender_report, _ = patch_checkpoint(original_dir, output_dir, key, payload)
        print("[B1.3] Receiver phase: recover from stego artifact using test key/protocol only...")
        recovered, receiver_report = receiver_extract(output_dir, key, len(payload))
        receiver_report["receiver_payload_matches"] = recovered == payload
        receiver_report["receiver_payload_text"] = recovered.decode("utf-8", errors="replace")
        model_report = {}
        if not args.skip_model_reload:
            print(f"[B1.3] Reloading only the stego checkpoint on {args.device} for model-weight extraction...")
            # token parameter is not needed for local artifact; kept in CLI for future private models.
            model_report = load_and_check_patched_model(output_dir, key, payload, args.device)
        report.update(sender_report)
        report.update(receiver_report)
        report.update(model_report)
        report["checks"].update(sender_report["checks"])
        report["checks"].update({
            "receiver_payload_matches": recovered == payload,
            "receiver_checksum_valid": receiver_report["receiver_checksum_valid"],
        })
        if model_report:
            report["checks"].update({
                "reloaded_model_nf4": model_report["reloaded_model_nf4"],
                "reloaded_model_payload_matches": model_report["reloaded_model_payload_matches"],
                "reloaded_model_checksum_valid": model_report["reloaded_model_checksum_valid"],
            })
        report["status"] = "PASS" if all(report["checks"].values()) else "FAIL"
        report["interpretation"] = (
            "Short test payload recovered from the patched checkpoint with keyed deterministic "
            "carrier selection and envelope checksum. This is an unencrypted mechanism pilot, "
            "not evidence of confidentiality, stealth, utility preservation, robustness, "
            "generalization, or novelty."
            if report["status"] == "PASS" else
            "At least one B1.3 check failed. Do not scale payload or claim Contract B success."
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[B1.3] Report written: {report_path}")
        print(f"[B1.3] Status: {report['status']}")


if __name__ == "__main__":
    main()
