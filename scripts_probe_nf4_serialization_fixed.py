#!/usr/bin/env python3
"""Contract B B1.1 diagnostic: test NF4 checkpoint serialization/reload invariants.

Run from nes-research/nes-llm:
  ../.venv/bin/python ../scripts_probe_nf4_serialization.py \
      --model Qwen/Qwen2.5-3B \
      --output-dir ../cache/contract_b_nf4_probe

This intentionally writes a full temporary quantized checkpoint. Keep the output
until you have inspected the report; delete it only when you no longer need it.
It does not edit NES production code or the source checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_digest(t: torch.Tensor | None) -> dict[str, Any] | None:
    if t is None:
        return None
    x = t.detach().cpu().contiguous()
    raw = x.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "numel": x.numel(),
    }


def state_snapshot(qs: Any, depth: int = 0) -> dict[str, Any] | None:
    if qs is None:
        return None
    if depth > 2:
        return {"truncated": True}
    result: dict[str, Any] = {}
    for name in ("shape", "blocksize", "quant_type", "dtype", "nested"):
        if hasattr(qs, name):
            value = getattr(qs, name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[name] = value
            else:
                result[name] = str(value)
    for name in ("code", "absmax", "offset"):
        value = getattr(qs, name, None)
        if isinstance(value, torch.Tensor):
            result[name] = tensor_digest(value)
        elif value is not None:
            result[name] = str(value)
    state2 = getattr(qs, "state2", None)
    if state2 is not None:
        result["state2"] = state_snapshot(state2, depth + 1)
    return result


def find_first_nf4_layer(model):
    candidates = []
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        qs = getattr(weight, "quant_state", None)
        if qs is not None and getattr(qs, "quant_type", None) == "nf4":
            candidates.append((name, module))
    if not candidates:
        raise RuntimeError(
            "No loaded weight with quant_state.quant_type == 'nf4' was found."
        )
    return candidates[0], len(candidates)


def load_nf4(model_id: str, device: str, token: str | None):
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    kwargs = {
        "quantization_config": config,
        "device_map": {"": device},
        "trust_remote_code": True,
    }
    if token:
        kwargs["token"] = token
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    out = Path(args.output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out}\n"
            "Choose a new directory; this script will not overwrite artifacts."
        )
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "contract_b_nf4_serialization_report.json"

    report: dict[str, Any] = {
        "status": "ERROR",
        "model_id": args.model,
        "device": device,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "output_dir": str(out),
        "source_revision": None,
        "checks": {},
    }

    model = None
    reloaded = None
    try:
        print(f"[B1.1] Loading NF4 model {args.model} on {device}...")
        model = load_nf4(args.model, device, args.token)
        report["source_revision"] = getattr(model.config, "_commit_hash", None)
        (name, module), nf4_layer_count = find_first_nf4_layer(model)
        weight = module.weight
        packed_before = weight.data.detach().cpu().contiguous().clone()
        quant_before = state_snapshot(getattr(weight, "quant_state", None))
        report["selected_layer"] = name
        report["nf4_layer_count"] = nf4_layer_count
        report["packed_before"] = tensor_digest(packed_before)
        report["quant_state_before"] = quant_before
        print(f"[B1.1] Selected {name}; packed tensor {tuple(packed_before.shape)}")
        print(f"[B1.1] Saving quantized checkpoint to {out} ...")
        model.save_pretrained(out, safe_serialization=True)
        report["saved_files"] = [
            {
                "name": p.name,
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p) if p.is_file() else None,
            }
            for p in sorted(out.iterdir())
            if p.is_file() and p.name != report_path.name
        ]
        print("[B1.1] Reloading from the saved artifact...")
        del model
        model = None
        if device == "mps":
            torch.mps.empty_cache()
        reloaded = load_nf4(str(out), device, args.token)
        (reloaded_name, reloaded_module), reloaded_count = find_first_nf4_layer(reloaded)
        packed_after = reloaded_module.weight.data.detach().cpu().contiguous()
        quant_after = state_snapshot(
            getattr(reloaded_module.weight, "quant_state", None)
        )
        raw_equal = torch.equal(packed_before, packed_after)
        quant_equal = quant_before == quant_after
        report["reloaded_layer"] = reloaded_name
        report["reloaded_nf4_layer_count"] = reloaded_count
        report["packed_after"] = tensor_digest(packed_after)
        report["quant_state_after"] = quant_after
        report["checks"] = {
            "same_selected_layer_name": name == reloaded_name,
            "same_nf4_layer_count": nf4_layer_count == reloaded_count,
            "packed_code_tensor_exact_equal": raw_equal,
            "quant_state_snapshot_equal": quant_equal,
        }
        report["status"] = (
            "PASS"
            if all(report["checks"].values())
            else "FAIL"
        )
        report["interpretation"] = (
            "The selected NF4 packed tensor and captured quantization state "
            "survived save/reload exactly. This is only a B1.1 serialization "
            "prerequisite; it does not demonstrate payload embedding, recovery, "
            "utility, detectability, or robustness."
            if report["status"] == "PASS"
            else
            "At least one serialization invariant changed. Do not implement "
            "the payload writer on this path until the mismatch is diagnosed."
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "ERROR"
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[B1.1] Report written: {report_path}")
        print(f"[B1.1] Status: {report['status']}")
        if reloaded is not None:
            del reloaded
        if model is not None:
            del model


if __name__ == "__main__":
    main()
