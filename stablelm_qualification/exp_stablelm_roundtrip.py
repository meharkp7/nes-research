#!/usr/bin/env python3
"""StableLM-2-1.6B independent-process round-trip runner for the frozen codec.

Two modes, run as SEPARATE processes:
  --mode embed    : embed payload, requantize through cycles, save packed
                    artifacts to disk, write an embed manifest with hashes.
  --mode extract  : (fresh process) load each saved artifact from disk and
                    extract using only (artifact bytes, key, tensor, method).

Resumable: each condition is written atomically; existing outputs are skipped.
The frozen codec is used unchanged. No hidden in-memory state is used for
extraction (the extractor reconstructs carriers from the artifact alone).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = "/workspace/nes_run_20261016_anchor_ablation/code/nes-llm"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts")
from scripts.run_nf4_anchor_pilot import load_tensor, requantize, sha_file
from src.experiments.nf4_anchor_channel import (
    METHODS, carrier_positions, embed, eligible, extract, recovery_diagnostic, unpack,
)

CODEC = Path(f"{REPO}/src/experiments/nf4_anchor_channel.py")
BLOCKSIZE = 64
METHOD = "interior_nearest"
BASELINES = ("pair_parity", "nearest_parity")


def atomic_write_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def payload_for(bits, seed):
    return np.random.default_rng(seed).bytes(bits // 8)


def key_for(seed):
    return hashlib.sha256(f"NES-development-{seed}".encode()).digest()


def cond_name(method, tensor, bits, seed):
    short = tensor.replace("model.layers.", "L").replace(".weight", "").replace("_proj", "")
    return f"{method}__{short}__{bits}__{seed}"


def embed_mode(args):
    outdir = args.output
    (outdir / "artifacts").mkdir(parents=True, exist_ok=True)
    (outdir / "embed_records").mkdir(parents=True, exist_ok=True)
    rows = []
    for method in args.methods:
        for tensor in args.tensors:
            source, state = load_tensor(args.checkpoint, tensor)
            raw = source.numpy().tobytes()
            codes = unpack(raw)
            cb = state.code.float().numpy()
            clean_sha = hashlib.sha256(raw).hexdigest()
            anchor_mask = (codes == 0) | (codes == 15)
            anchor_before = int(anchor_mask.sum())
            for bits in args.sizes:
                for seed in args.seeds:
                    name = cond_name(method, tensor, bits, seed)
                    rec_path = outdir / "embed_records" / f"{name}.json"
                    if rec_path.exists():
                        rows.append(json.loads(rec_path.read_text()))
                        continue
                    t0 = time.monotonic()
                    payload = payload_for(bits, seed)
                    key = key_for(seed)
                    try:
                        stego, info = embed(raw, payload, key, tensor, method, cb)
                    except Exception as exc:
                        rec = {"condition": name, "method": method, "tensor": tensor,
                               "bits": bits, "seed": seed, "status": "EMBED_FAILED",
                               "error": f"{type(exc).__name__}: {exc}"}
                        atomic_write_json(rec_path, rec); rows.append(rec); continue
                    stego_codes = unpack(stego)
                    # carrier agreement: carrier set must be stable sender->receiver
                    framed = (bits // 8 + 16) * 8
                    pos0 = carrier_positions(codes, framed, key, tensor, method)
                    pos1 = carrier_positions(stego_codes, framed, key, tensor, method)
                    carrier_agree = float(np.mean(pos0 == pos1))
                    # requantize through cycles; save each cycle's packed bytes
                    tensor_t = torch.from_numpy(
                        np.frombuffer(stego, dtype=np.uint8).copy()).reshape(source.shape)
                    cur_state = state
                    cycle_info = []
                    for cycle in range(args.cycles + 1):
                        if cycle:
                            tensor_t, cur_state = requantize(tensor_t, cur_state)
                        cur = tensor_t.numpy().tobytes()
                        art = outdir / "artifacts" / f"{name}_c{cycle}.bin"
                        if not art.exists():
                            art.write_bytes(cur)
                        cur_codes = unpack(cur)
                        a_after = int(((cur_codes == 0) | (cur_codes == 15)).sum())
                        cycle_info.append({
                            "cycle": cycle,
                            "artifact": str(art.relative_to(outdir)),
                            "artifact_sha256": hashlib.sha256(cur).hexdigest(),
                            "changed_anchor_codes": int(np.count_nonzero(
                                codes[anchor_mask] != stego_codes[anchor_mask])),
                            "anchor_codes_after": a_after,
                            "anchor_codes_delta": a_after - anchor_before,
                            "blocks_without_endpoint": int(np.sum(~np.any(
                                (cur_codes.reshape(-1, BLOCKSIZE) == 0) |
                                (cur_codes.reshape(-1, BLOCKSIZE) == 15), axis=1))),
                            "packed_code_hamming_vs_source": int(np.count_nonzero(cur_codes != codes)),
                        })
                    rec = {"condition": name, "status": "EMBEDDED", "method": method,
                           "tensor": tensor, "bits": bits, "seed": seed,
                           "blocksize": int(state.blocksize), "nested": bool(state.nested),
                           "quant_type": str(state.quant_type),
                           "clean_tensor_sha256": clean_sha,
                           "stego_tensor_sha256": hashlib.sha256(stego).hexdigest(),
                           "payload_sha256": hashlib.sha256(payload).hexdigest(),
                           "eligible_codes": int(info["eligible_codes"]),
                           "changed_codes": int(info["changed_codes"]),
                           "changed_anchor_codes": int(np.count_nonzero(
                               codes[anchor_mask] != stego_codes[anchor_mask])),
                           "carrier_agreement_cycle0": carrier_agree,
                           "anchor_codes_before": anchor_before,
                           "elapsed_s": round(time.monotonic() - t0, 2),
                           "cycles": cycle_info}
                    atomic_write_json(rec_path, rec); rows.append(rec)
                    print(json.dumps({"embed": name, "status": "EMBEDDED",
                                      "carrier_agree": carrier_agree,
                                      "changed_anchor": rec["changed_anchor_codes"]}), flush=True)
    atomic_write_json(outdir / "embed_summary.json", {
        "mode": "embed", "checkpoint": str(args.checkpoint),
        "codec_sha256": sha_file(CODEC), "method_primary": METHOD,
        "methods": list(args.methods), "sizes": list(args.sizes), "seeds": list(args.seeds),
        "tensors": list(args.tensors), "cycles": args.cycles,
        "n_conditions": len(rows),
        "n_embedded": sum(r.get("status") == "EMBEDDED" for r in rows)})
    print(json.dumps({"mode": "embed", "done": True, "n_conditions": len(rows)}), flush=True)


def extract_mode(args):
    """Fresh process: read artifacts from disk, extract with documented inputs only."""
    outdir = args.output
    (outdir / "extract_records").mkdir(parents=True, exist_ok=True)
    recs = sorted((outdir / "embed_records").glob("*.json"))
    n_exact = n_total = 0
    for rp in recs:
        rec = json.loads(rp.read_text())
        if rec.get("status") != "EMBEDDED":
            continue
        method, tensor, bits, seed = rec["method"], rec["tensor"], rec["bits"], rec["seed"]
        payload = payload_for(bits, seed)
        key = key_for(seed)
        for ci in rec["cycles"]:
            name = f"{rec['condition']}_c{ci['cycle']}"
            ep = outdir / "extract_records" / f"{name}.json"
            if ep.exists():
                e = json.loads(ep.read_text())
            else:
                art = outdir / ci["artifact"]
                raw = art.read_bytes()
                t0 = time.monotonic()
                diag = recovery_diagnostic(raw, payload, key, tensor, method)
                try:
                    recovered = extract(raw, key, tensor, method)
                    checksum_ok = hashlib.sha256(recovered).digest()[:8] == \
                        hashlib.sha256(payload).digest()[:8]
                    length_ok = len(recovered) == len(payload)
                except Exception as exc:
                    recovered, checksum_ok, length_ok = b"", False, False
                    diag.setdefault("decode_error", f"{type(exc).__name__}: {exc}")
                e = {"condition": name, "method": method, "tensor": tensor,
                     "bits": bits, "seed": seed, "cycle": ci["cycle"],
                     "artifact_sha256": ci["artifact_sha256"],
                     "recovered_sha256": hashlib.sha256(recovered).hexdigest(),
                     "payload_sha256": hashlib.sha256(payload).hexdigest(),
                     "length_ok": bool(length_ok), "checksum_ok": bool(checksum_ok),
                     "exact_recovery": bool(diag.get("exact_recovery")),
                     "payload_ber": diag.get("payload_ber"),
                     "decode_error": diag.get("decode_error"),
                     "elapsed_s": round(time.monotonic() - t0, 2)}
                atomic_write_json(ep, e)
            n_total += 1
            n_exact += int(bool(e["exact_recovery"]))
    atomic_write_json(outdir / "extract_summary.json", {
        "mode": "extract", "n_records": n_total, "n_exact": n_exact,
        "codec_sha256": sha_file(CODEC)})
    print(json.dumps({"mode": "extract", "n_records": n_total, "n_exact": n_exact}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("embed", "extract"), required=True)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("/workspace/nes_run_20261016_anchor_ablation/ckpt/stablelm_clean"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=[METHOD, *BASELINES])
    p.add_argument("--tensors", nargs="+", default=[
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.11.self_attn.k_proj.weight",
        "model.layers.23.self_attn.v_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.11.mlp.down_proj.weight",
        "model.layers.23.mlp.gate_proj.weight"])
    p.add_argument("--sizes", nargs="+", type=int, default=[1000, 5000, 10000, 20000, 50000])
    p.add_argument("--seeds", nargs="+", type=int, default=[20261016, 20261017, 20261018])
    p.add_argument("--cycles", type=int, default=3)
    args = p.parse_args()
    if args.mode == "embed":
        embed_mode(args)
    else:
        extract_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
