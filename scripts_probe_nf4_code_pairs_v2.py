#!/usr/bin/env python3
"""B1.2 v2: test-only NF4 nibble invariants and artifact save/reload.

Avoids Transformers save_pretrained because some Transformers 5.x weight
conversion reverse operations are unimplemented for this quantized model.
Instead, it patches only the selected U8 packed-weight bytes in a copied
safetensors file, leaving the safetensors header and every other byte intact.

Run from nes-research/nes-llm:
  ../.venv/bin/python ../scripts_probe_nf4_code_pairs_v2.py \
    --model ../cache/contract_b_nf4_probe_retry \
    --output-dir ../cache/contract_b_nf4_pairs_b12_retry2

This is a test-only representation probe, not a payload encoder or production
implementation. It writes a full copy of the input checkpoint.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, shutil, sys
from pathlib import Path
from typing import Any
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def raw_tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_digest(t: torch.Tensor | None) -> dict[str, Any] | None:
    if t is None: return None
    x = t.detach().cpu().contiguous()
    raw = raw_tensor_bytes(x)
    return {"shape": list(x.shape), "dtype": str(x.dtype), "sha256": hashlib.sha256(raw).hexdigest(), "numel": x.numel()}


def state_snapshot(qs: Any, depth: int = 0) -> dict[str, Any] | None:
    if qs is None: return None
    if depth > 2: return {"truncated": True}
    result: dict[str, Any] = {}
    for name in ("shape", "blocksize", "quant_type", "dtype", "nested"):
        if hasattr(qs, name):
            value = getattr(qs, name)
            result[name] = value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
    for name in ("code", "absmax", "offset"):
        value = getattr(qs, name, None)
        if isinstance(value, torch.Tensor): result[name] = tensor_digest(value)
        elif value is not None: result[name] = str(value)
    state2 = getattr(qs, "state2", None)
    if state2 is not None: result["state2"] = state_snapshot(state2, depth + 1)
    return result


def find_first_nf4_layer(model):
    candidates = []
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        qs = getattr(weight, "quant_state", None)
        if qs is not None and getattr(qs, "quant_type", None) == "nf4": candidates.append((name, module))
    if not candidates: raise RuntimeError("No weight with quant_state.quant_type == 'nf4' was found.")
    return candidates[0], len(candidates)


def load_nf4(model_id: str, device: str, token: str | None):
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    kwargs = {"quantization_config": config, "device_map": {"": device}, "trust_remote_code": True}
    if token: kwargs["token"] = token
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def unpack_nibbles(packed: torch.Tensor) -> list[int]:
    """bitsandbytes 4-bit packing: first logical code is low nibble, second high."""
    vals = packed.detach().cpu().contiguous().view(-1).tolist()
    out: list[int] = []
    for byte in vals:
        v = int(byte)
        out.append(v & 0x0F)
        out.append((v >> 4) & 0x0F)
    return out


def pack_nibbles(codes: list[int]) -> torch.Tensor:
    if len(codes) % 2: raise ValueError("Expected an even number of logical 4-bit codes.")
    packed = bytearray(len(codes) // 2)
    for i in range(0, len(codes), 2):
        low, high = int(codes[i]), int(codes[i + 1])
        if not (0 <= low <= 15 and 0 <= high <= 15): raise ValueError("NF4 codes must be integers in [0, 15].")
        packed[i // 2] = low | (high << 4)
    return torch.tensor(list(packed), dtype=torch.uint8).reshape(-1, 1)


def read_safetensors_header(path: Path):
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8: raise ValueError(f"Invalid safetensors header in {path}")
        header_len = int.from_bytes(prefix, "little", signed=False)
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len: raise ValueError(f"Truncated safetensors header in {path}")
    return json.loads(header_bytes.decode("utf-8")), 8 + header_len


def locate_tensor(source: Path, key: str):
    for shard in sorted(source.glob("*.safetensors")):
        header, data_start = read_safetensors_header(shard)
        if key in header:
            return shard, header[key], data_start
    # Also allow weight keys with a prefix difference, but only a unique exact suffix.
    matches = []
    for shard in sorted(source.glob("*.safetensors")):
        header, data_start = read_safetensors_header(shard)
        for k, info in header.items():
            if k.endswith("." + key) or k == key:
                matches.append((shard, k, info, data_start))
    if len(matches) == 1:
        shard, actual_key, info, data_start = matches[0]
        return shard, {"_actual_key": actual_key, **info}, data_start
    raise KeyError(f"Could not uniquely locate safetensors tensor key {key!r}; matches={len(matches)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local B1.1 saved checkpoint directory")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--positions", type=int, default=256)
    args = ap.parse_args()
    if args.positions < 16: ap.error("--positions must be at least 16")
    source = Path(args.model).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    if not source.is_dir(): raise FileNotFoundError(f"Input model directory not found: {source}")
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Output directory is not empty: {out}. Choose a new directory.")
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "contract_b_nf4_code_pairs_report.json"
    report: dict[str, Any] = {"status":"ERROR", "stage":"B1.2-v2", "model_id":str(source), "device":device,
        "python":sys.version, "platform":platform.platform(), "torch":torch.__version__, "positions_requested":args.positions, "checks":{}}
    model = reloaded = None
    try:
        print(f"[B1.2-v2] Loading NF4 artifact {source} on {device}...")
        model = load_nf4(str(source), device, args.token)
        (layer_name, module), nf4_count = find_first_nf4_layer(model)
        packed_before = module.weight.data.detach().cpu().contiguous().clone()
        if packed_before.dtype != torch.uint8: raise TypeError(f"Expected packed uint8 tensor, found {packed_before.dtype}")
        qs_before = state_snapshot(getattr(module.weight, "quant_state", None))
        codes_before = unpack_nibbles(packed_before)
        if len(codes_before) < args.positions: raise ValueError(f"Only {len(codes_before)} logical codes; requested {args.positions}")
        rng = random.Random(20261009)
        chosen = {0,1,2,3,len(codes_before)-2,len(codes_before)-1}
        while len(chosen) < args.positions: chosen.add(rng.randrange(len(codes_before)))
        positions = sorted(chosen)
        target_bits = {idx: ((idx * 1103515245 + 12345) >> 8) & 1 for idx in positions}
        codes_after = list(codes_before)
        for idx in positions: codes_after[idx] = 2 * (codes_before[idx] // 2) + target_bits[idx]
        packed_after = pack_nibbles(codes_after).reshape(packed_before.shape)
        selected_pair_ids_preserved = all(codes_after[i] // 2 == codes_before[i] // 2 for i in positions)
        unselected_unchanged = all(codes_after[i] == codes_before[i] for i in range(len(codes_before)) if i not in target_bits)
        selected_target_bits_match = all(codes_after[i] % 2 == target_bits[i] for i in positions)
        module.weight.data.copy_(packed_after.to(device=module.weight.data.device))
        metadata_unchanged_in_memory = qs_before == state_snapshot(getattr(module.weight, "quant_state", None))
        report.update({"selected_layer":layer_name, "nf4_layer_count":nf4_count, "packed_shape":list(packed_before.shape),
            "logical_code_count":len(codes_before), "packed_before":tensor_digest(packed_before), "packed_after_mutation":tensor_digest(packed_after),
            "quant_state_before":qs_before, "positions_tested":len(positions), "positions_changed":sum(codes_after[i] != codes_before[i] for i in positions),
            "position_sample_first_32":positions[:32], "checks":{
                "packed_tensor_is_uint8":packed_before.dtype == torch.uint8,
                "all_codes_in_range_0_15":all(0 <= q <= 15 for q in codes_after),
                "selected_pair_ids_preserved":selected_pair_ids_preserved,
                "unselected_codes_unchanged":unselected_unchanged,
                "selected_target_bits_match":selected_target_bits_match,
                "quant_state_unchanged_in_memory":metadata_unchanged_in_memory}})
        if not all(report["checks"].values()):
            report["status"] = "FAIL"
            report["interpretation"] = "An in-memory code/pair/metadata invariant failed. Do not continue."
        else:
            # Find the packed weight bytes in the source checkpoint and verify exact correspondence.
            key = f"{layer_name}.weight"
            source_shard, info, data_start = locate_tensor(source, key)
            actual_key = info.pop("_actual_key", key)
            if info.get("dtype") != "U8": raise TypeError(f"Expected safetensors U8 packed tensor, found {info.get('dtype')}")
            start, end = info["data_offsets"]
            with source_shard.open("rb") as f:
                f.seek(data_start + start); serialized_before = f.read(end-start)
            if serialized_before != raw_tensor_bytes(packed_before):
                raise RuntimeError("Loaded packed weight bytes do not match raw safetensors tensor bytes; refusing to patch.")
            if len(serialized_before) != len(raw_tensor_bytes(packed_after)):
                raise RuntimeError("Packed tensor byte length changed; refusing to patch.")
            # Copy checkpoint assets, then patch only the target tensor data bytes in the copied shard.
            print(f"[B1.2-v2] Copying checkpoint and patching tensor {actual_key} in {source_shard.name}...")
            for item in source.iterdir():
                dest = out / item.name
                if item.is_dir(): shutil.copytree(item, dest)
                elif item.is_file(): shutil.copy2(item, dest)
            target_shard = out / source_shard.name
            with target_shard.open("r+b") as f:
                f.seek(data_start + start)
                f.write(raw_tensor_bytes(packed_after))
                f.flush(); os.fsync(f.fileno())
            report.update({"safetensors_key":actual_key, "safetensors_shard":source_shard.name,
                "safetensors_tensor_shape":info.get("shape"), "serialized_original_bytes_sha256":hashlib.sha256(serialized_before).hexdigest(),
                "serialized_mutated_bytes_sha256":hashlib.sha256(raw_tensor_bytes(packed_after)).hexdigest(),
                "save_method":"copied checkpoint; patched only target U8 tensor data region; header and other tensor bytes unchanged"})
            del model; model = None
            if device == "mps": torch.mps.empty_cache()
            print("[B1.2-v2] Reloading patched checkpoint...")
            reloaded = load_nf4(str(out), device, args.token)
            (reload_name, reload_module), reload_count = find_first_nf4_layer(reloaded)
            packed_reloaded = reload_module.weight.data.detach().cpu().contiguous()
            qs_reloaded = state_snapshot(getattr(reload_module.weight, "quant_state", None))
            report["packed_after_reload"] = tensor_digest(packed_reloaded)
            report["quant_state_after_reload"] = qs_reloaded
            report["checks"].update({
                "same_layer_after_reload":layer_name == reload_name,
                "same_nf4_layer_count_after_reload":nf4_count == reload_count,
                "modified_packed_tensor_exact_after_reload":torch.equal(packed_after, packed_reloaded),
                "logical_codes_exact_after_reload":codes_after == unpack_nibbles(packed_reloaded),
                "quant_state_unchanged_after_reload":qs_before == qs_reloaded})
            report["status"] = "PASS" if all(report["checks"].values()) else "FAIL"
            report["interpretation"] = ("Test-only nibble mutation survived direct safetensors patch and model reload, with checked pair IDs, unselected codes, and quantization metadata unchanged. This still does not validate dequantized-value mapping, receiver carrier reproduction, payload recovery, utility, detectability, robustness, or novelty." if report["status"] == "PASS" else "At least one invariant failed; do not proceed to payload embedding until diagnosed.")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "ERROR"
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[B1.2-v2] Report written: {report_path}")
        print(f"[B1.2-v2] Status: {report['status']}")
        if reloaded is not None: del reloaded
        if model is not None: del model

if __name__ == "__main__": main()
