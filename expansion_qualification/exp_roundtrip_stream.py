#!/usr/bin/env python3
"""Per-model independent-process round trip for the frozen interior_nearest codec.

Generalized from stablelm_qualification/exp_stablelm_roundtrip_stream.py without
changing the frozen codec or the protocol. For each (method, tensor, bits, seed):
embed -> requantize cycles 0..N -> write each cycle's artifact to a temp file ->
spawn a FRESH subprocess that loads ONLY that file and extracts with
(artifact bytes, key, tensor name, method) -> record -> delete temp artifact.

Resumable: existing condition records are skipped. New output dirs only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/nes_run_20261016_anchor_ablation")
REPO = ROOT / "code/nes-llm"
sys.path.insert(0, str(REPO))
sys.path.insert(0, f"{REPO}/scripts")
from scripts.run_nf4_anchor_pilot import load_tensor, requantize, sha_file  # noqa: E402
from src.experiments.nf4_anchor_channel import (  # noqa: E402
    carrier_positions, embed, extract, unpack,
)

CODEC = REPO / "src/experiments/nf4_anchor_channel.py"
BLOCKSIZE = 64
METHOD = "interior_nearest"
BASELINES = ("pair_parity", "nearest_parity")

EXTRACT_SNIPPET = r'''
import hashlib, json, sys
sys.path.insert(0, r"{repo}"); sys.path.insert(0, r"{repo}/scripts")
import numpy as np
from src.experiments.nf4_anchor_channel import extract
raw = open(sys.argv[1], "rb").read()
bits = int(sys.argv[2]); seed = int(sys.argv[3]); tensor = sys.argv[4]; method = sys.argv[5]
payload = np.random.default_rng(seed).bytes(bits // 8)
key = hashlib.sha256(f"NES-development-{{seed}}".encode()).digest()
out = {{"exact_recovery": False, "length_ok": False, "checksum_ok": False,
       "recovered_sha256": None, "payload_sha256": hashlib.sha256(payload).hexdigest(),
       "decode_error": None}}
try:
    rec = extract(raw, key, tensor, method)
    out["recovered_sha256"] = hashlib.sha256(rec).hexdigest()
    out["length_ok"] = len(rec) == len(payload)
    out["checksum_ok"] = hashlib.sha256(rec).digest()[:8] == hashlib.sha256(payload).digest()[:8]
    out["exact_recovery"] = rec == payload
except Exception as e:
    out["decode_error"] = f"{{type(e).__name__}}: {{e}}"
print(json.dumps(out))
'''.format(repo=str(REPO))


def payload_for(bits, seed):
    return np.random.default_rng(seed).bytes(bits // 8)


def key_for(seed):
    return hashlib.sha256(f"NES-development-{seed}".encode()).digest()


def atomic_write_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def cond_name(method, tensor, bits, seed):
    short = tensor.replace("model.layers.", "L").replace(".weight", "").replace("_proj", "")
    return f"{method}__{short}__{bits}__{seed}"


def run_condition(method, tensor, bits, seed, args, tmpdir, outdir):
    name = cond_name(method, tensor, bits, seed)
    rec_path = outdir / "records" / f"{name}.json"
    if rec_path.exists():
        return json.loads(rec_path.read_text())
    t0 = time.monotonic()
    source, state = load_tensor(args.checkpoint, tensor)
    raw = source.numpy().tobytes()
    codes = unpack(raw)
    cb = state.code.float().numpy()
    anchor = (codes == 0) | (codes == 15)
    anchor_before = int(anchor.sum())
    payload = payload_for(bits, seed)
    key = key_for(seed)
    try:
        stego, info = embed(raw, payload, key, tensor, method, cb)
    except Exception as exc:
        rec = {"condition": name, "status": "EMBED_FAILED", "method": method,
               "tensor": tensor, "bits": bits, "seed": seed,
               "error": f"{type(exc).__name__}: {exc}"}
        atomic_write_json(rec_path, rec)
        return rec
    scodes = unpack(stego)
    framed = (bits // 8 + 16) * 8
    try:
        p0 = carrier_positions(codes, framed, key, tensor, method)
        p1 = carrier_positions(scodes, framed, key, tensor, method)
        carrier_agree = float(np.mean(p0 == p1))
    except ValueError:
        carrier_agree = None
    t = torch.from_numpy(np.frombuffer(stego, dtype=np.uint8).copy()).reshape(source.shape)
    st = state
    cycles = []
    for cycle in range(args.cycles + 1):
        if cycle:
            t, st = requantize(t, st)
        cur = t.numpy().tobytes()
        cc = unpack(cur)
        art = Path(tmpdir) / f"{name}_c{cycle}.bin"
        art.write_bytes(cur)
        proc = subprocess.run(
            [sys.executable, "-c", EXTRACT_SNIPPET, str(art), str(bits), str(seed), tensor, method],
            capture_output=True, text=True)
        if proc.returncode != 0:
            ex = {"exact_recovery": False, "length_ok": False, "checksum_ok": False,
                  "decode_error": f"subprocess rc={proc.returncode}: {proc.stderr[-300:]}",
                  "recovered_sha256": None, "payload_sha256": hashlib.sha256(payload).hexdigest()}
        else:
            ex = json.loads(proc.stdout.strip().splitlines()[-1])
        art.unlink()
        cycles.append({
            "cycle": cycle, "artifact_sha256": hashlib.sha256(cur).hexdigest(),
            "fresh_process": True,
            "exact_recovery": ex["exact_recovery"], "length_ok": ex["length_ok"],
            "checksum_ok": ex["checksum_ok"],
            "recovered_sha256": ex["recovered_sha256"],
            "payload_sha256": ex["payload_sha256"],
            "decode_error": ex.get("decode_error"),
            "changed_anchor_codes_vs_source": int(np.count_nonzero(codes[anchor] != cc[anchor])),
            "anchor_codes_after": int(((cc == 0) | (cc == 15)).sum()),
            "anchor_codes_delta": int(((cc == 0) | (cc == 15)).sum()) - anchor_before,
            "blocks_without_endpoint": int(np.sum(~np.any(
                (cc.reshape(-1, BLOCKSIZE) == 0) | (cc.reshape(-1, BLOCKSIZE) == 15), axis=1))),
            "packed_code_hamming_vs_source": int(np.count_nonzero(cc != codes)),
        })
    rec = {"condition": name, "status": "DONE", "method": method, "tensor": tensor,
           "bits": bits, "seed": seed, "blocksize": int(state.blocksize),
           "nested": bool(state.nested), "quant_type": str(state.quant_type),
           "clean_tensor_sha256": hashlib.sha256(raw).hexdigest(),
           "stego_tensor_sha256": hashlib.sha256(stego).hexdigest(),
           "payload_sha256": hashlib.sha256(payload).hexdigest(),
           "eligible_codes": int(info["eligible_codes"]),
           "changed_codes": int(info["changed_codes"]),
           "carrier_agreement": carrier_agree,
           "anchor_codes_before": anchor_before,
           "elapsed_s": round(time.monotonic() - t0, 2),
           "cycles": cycles}
    atomic_write_json(rec_path, rec)
    n_ex = sum(c["exact_recovery"] for c in cycles)
    print(json.dumps({"cond": name, "status": "DONE", "exact": f"{n_ex}/{len(cycles)}",
                      "carrier_agree": carrier_agree}), flush=True)
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=[METHOD, *BASELINES])
    p.add_argument("--tensors", nargs="+", required=True)
    p.add_argument("--sizes", nargs="+", type=int, default=[1000, 5000, 10000, 20000, 50000])
    p.add_argument("--seeds", nargs="+", type=int, default=[20261016, 20261017, 20261018])
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "records").mkdir(exist_ok=True)
    recs = []
    with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
        for method in args.methods:
            for tensor in args.tensors:
                for bits in args.sizes:
                    for seed in args.seeds:
                        recs.append(run_condition(method, tensor, bits, seed, args, tmpdir, args.output))
    n_done = sum(r["status"] == "DONE" for r in recs)
    n_fail = sum(r["status"] == "EMBED_FAILED" for r in recs)
    exact = sum(1 for r in recs if r["status"] == "DONE" for c in r["cycles"] if c["exact_recovery"])
    total_cycles = sum(len(r.get("cycles", [])) for r in recs)
    atomic_write_json(args.output / "summary.json", {
        "model_id": args.model_id, "revision": args.revision,
        "codec_sha256": sha_file(CODEC), "method_primary": METHOD,
        "methods": list(args.methods), "sizes": list(args.sizes), "seeds": list(args.seeds),
        "tensors": list(args.tensors), "cycles": args.cycles,
        "n_conditions": len(recs), "n_done": n_done, "n_embed_failed": n_fail,
        "n_cycle_extractions": total_cycles, "n_exact": exact})
    print(json.dumps({"done": True, "conditions": len(recs), "done_ok": n_done,
                      "embed_failed": n_fail, "exact": f"{exact}/{total_cycles}"}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
