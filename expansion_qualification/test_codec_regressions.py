#!/usr/bin/env python3
"""Focused regression tests for the frozen NF4 anchor codec (Track 3), per model.

Generalized from stablelm_qualification/test_stablelm_codec_regressions.py. The
tests exercise the FROZEN codec on a given model's clean checkpoint; the codec is
never modified. Runs with the plain interpreter (no pytest needed):
    /workspace/nes_env/bin/python test_codec_regressions.py --checkpoint <dir>

Exit code 0 = all pass; 1 = at least one failure. Failures are printed, not hidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = "/workspace/nes_run_20261016_anchor_ablation/code/nes-llm"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts")
from scripts.run_nf4_anchor_pilot import load_tensor, requantize  # noqa: E402
from src.experiments.nf4_anchor_channel import (  # noqa: E402
    carrier_positions, embed, eligible, extract, pack, unpack,
)

BITS = 10000
SEED = 20261016
KEY = hashlib.sha256(f"NES-development-{SEED}".encode()).digest()
PAYLOAD = np.random.default_rng(SEED).bytes(BITS // 8)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append({"test": name, "pass": bool(cond), "detail": detail})
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tensor", default="model.layers.0.self_attn.q_proj.weight")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    CKPT, TENSOR = args.checkpoint, args.tensor

    source, state = load_tensor(CKPT, TENSOR)
    raw = source.numpy().tobytes()
    codes = unpack(raw)
    cb = state.code.float().numpy()

    # 1. Determinism: identical input -> identical packed output.
    a, _ = embed(raw, PAYLOAD, KEY, TENSOR, "interior_nearest", cb)
    b, _ = embed(raw, PAYLOAD, KEY, TENSOR, "interior_nearest", cb)
    check("deterministic_fixed_seed", a == b, hashlib.sha256(a).hexdigest()[:12])

    # 2. Only intended carriers change; endpoints untouched.
    sc = unpack(a)
    anchor = (codes == 0) | (codes == 15)
    check("endpoints_unchanged", int(np.count_nonzero(codes[anchor] != sc[anchor])) == 0)
    check("interior_only", bool(np.all(((sc > 0) & (sc < 15)) | anchor)))

    # 3. Carrier eligibility stable sender->receiver.
    framed = (BITS // 8 + 16) * 8
    p0 = carrier_positions(codes, framed, KEY, TENSOR, "interior_nearest")
    p1 = carrier_positions(sc, framed, KEY, TENSOR, "interior_nearest")
    check("carrier_eligibility_stable", bool(np.array_equal(p0, p1)))

    # 4. Fresh-process extraction from a saved artifact (subprocess).
    with tempfile.TemporaryDirectory() as td:
        art = Path(td) / "artifact.bin"
        art.write_bytes(a)
        script = f'''
import hashlib,sys,numpy as np
sys.path.insert(0,"{REPO}"); sys.path.insert(0,"{REPO}/scripts")
from src.experiments.nf4_anchor_channel import extract
raw=open(r"{art}","rb").read()
key=hashlib.sha256(b"NES-development-{SEED}").digest()
p=extract(raw,key,"{TENSOR}","interior_nearest")
print(len(p), hashlib.sha256(p).hexdigest())
'''
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        parts = out.stdout.split()
        ok = out.returncode == 0 and len(parts) == 2 and int(parts[0]) == len(PAYLOAD) \
            and parts[1] == hashlib.sha256(PAYLOAD).hexdigest()
        check("fresh_process_extraction", ok, out.stdout.strip() or out.stderr[-200:])

    # 5. Repeated requantization cycles 0..5 stay exact.
    t = torch.from_numpy(np.frombuffer(a, dtype=np.uint8).copy()).reshape(source.shape)
    st = state
    exact_all = True
    for c in range(6):
        if c:
            t, st = requantize(t, st)
        rec = extract(t.numpy().tobytes(), KEY, TENSOR, "interior_nearest")
        exact_all = exact_all and rec == PAYLOAD
    check("requantize_cycles_0_5_exact", exact_all)

    # 6. Corrupted CARRIER artifact -> error/mismatch, never silent exact.
    # NOTE: corrupting fixed byte offsets (as in the StableLM original) is
    # model-dependent: if no carrier position falls in the corrupted range, exact
    # recovery is the CORRECT outcome, not a failure. So this test corrupts an
    # actual carrier code (parity flip), which must always be detected.
    carrier_idx = int(carrier_positions(codes, framed, KEY, TENSOR, "interior_nearest")[0])
    cj = carrier_idx // 2
    corrupted = bytearray(a)
    corrupted[cj] ^= 0x01 if carrier_idx % 2 == 1 else 0x10
    try:
        r = extract(bytes(corrupted), KEY, TENSOR, "interior_nearest")
        check("corrupted_carrier_detected", r != PAYLOAD, "carrier corruption not detected as exact")
    except (ValueError, AssertionError) as e:
        check("corrupted_carrier_detected", True, f"{type(e).__name__}: {e}")

    # 6b. Fixed-offset corruption: report whether carriers were hit (observational).
    corrupt_codes = set()
    for i in range(0, min(4096, len(a)), 7):
        corrupt_codes.add(2 * i); corrupt_codes.add(2 * i + 1)
    carriers_hit = len(set(carrier_positions(codes, framed, KEY, TENSOR, "interior_nearest").tolist()) & corrupt_codes)
    try:
        r = extract(bytes(bytearray(
            bytes([a[i] ^ 0xFF if i % 7 == 0 and i < 4096 else a[i] for i in range(len(a))])),
            ), KEY, TENSOR, "interior_nearest")
        check("fixed_offset_corruption_consistent",
              (r == PAYLOAD) == (carriers_hit == 0),
              f"carriers_hit={carriers_hit}, exact={r == PAYLOAD}")
    except (ValueError, AssertionError) as e:
        check("fixed_offset_corruption_consistent", carriers_hit > 0,
              f"carriers_hit={carriers_hit}, raised {type(e).__name__}")

    # 7. Truncated artifact -> error, not crash-with-wrong-answer.
    try:
        r = extract(a[:len(a)//4], KEY, TENSOR, "interior_nearest")
        check("truncated_artifact_handled", r != PAYLOAD, "truncation detected")
    except (ValueError, AssertionError) as e:
        check("truncated_artifact_handled", True, f"{type(e).__name__}")

    # 8. Wrong key -> not exact (magic/checksum failure).
    wrong = hashlib.sha256(b"NES-development-999").digest()
    try:
        r = extract(a, wrong, TENSOR, "interior_nearest")
        check("wrong_key_not_exact", r != PAYLOAD)
    except ValueError as e:
        check("wrong_key_not_exact", True, f"ValueError: {e}")

    # 9. Over-capacity payload rejected.
    tiny = pack(np.zeros(64, dtype=np.uint8))
    try:
        embed(tiny, PAYLOAD, KEY, TENSOR, "interior_nearest", cb)
        check("over_capacity_rejected", False, "no error raised")
    except (ValueError, AssertionError) as e:
        check("over_capacity_rejected", True, f"{type(e).__name__}")

    # 10. Baseline pair_parity is not silently weakened: fails after requantization.
    pp, _ = embed(raw, PAYLOAD, KEY, TENSOR, "pair_parity", cb)
    tp = torch.from_numpy(np.frombuffer(pp, dtype=np.uint8).copy()).reshape(source.shape)
    tp, stp = requantize(tp, state)
    try:
        rp = extract(tp.numpy().tobytes(), KEY, TENSOR, "pair_parity")
        check("pair_parity_fails_after_requant", rp != PAYLOAD, "BER path")
    except ValueError as e:
        check("pair_parity_fails_after_requant", True, f"ValueError: {e}")

    n_pass = sum(r["pass"] for r in RESULTS)
    summary = {"model": args.model_id, "revision": args.revision,
               "checkpoint": str(CKPT), "tensor": TENSOR,
               "n_pass": n_pass, "n_total": len(RESULTS), "results": RESULTS}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\n{n_pass}/{len(RESULTS)} tests passed", flush=True)
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
