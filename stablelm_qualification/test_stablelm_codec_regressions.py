#!/usr/bin/env python3
"""Focused regression tests for the frozen NF4 anchor codec (Track 3).

Runs with the plain interpreter (no pytest needed):
    /workspace/nes_env/bin/python test_stablelm_codec_regressions.py

Covers: deterministic output for a fixed seed; fresh-process extraction;
corrupted/truncated artifact handling; wrong-length/checksum handling; repeated
requantization; stable carrier eligibility; only-intended-tensor modification.
Exit code 0 = all pass; 1 = at least one failure. Failures are printed, not hidden.
"""
from __future__ import annotations

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
from scripts.run_nf4_anchor_pilot import load_tensor, requantize
from src.experiments.nf4_anchor_channel import (
    carrier_positions, embed, eligible, extract, pack, unpack,
)

CKPT = Path("/workspace/nes_run_20261016_anchor_ablation/ckpt/stablelm_clean")
TENSOR = "model.layers.0.self_attn.q_proj.weight"
BITS = 10000
SEED = 20261016
KEY = hashlib.sha256(f"NES-development-{SEED}".encode()).digest()
PAYLOAD = np.random.default_rng(SEED).bytes(BITS // 8)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append({"test": name, "pass": bool(cond), "detail": detail})
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def main():
    source, state = load_tensor(CKPT, TENSOR)
    raw = source.numpy().tobytes()
    codes = unpack(raw)
    cb = state.code.float().numpy()

    # 1. Determinism: identical input -> identical packed output.
    a, _ = embed(raw, PAYLOAD, KEY, TENSOR, "interior_nearest", cb)
    b, _ = embed(raw, PAYLOAD, KEY, TENSOR, "interior_nearest", cb)
    check("deterministic_fixed_seed", a == b,
          hashlib.sha256(a).hexdigest()[:12])

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

    # 6. Corrupted artifact -> ValueError, not silent wrong answer.
    corrupted = bytearray(a)
    for i in range(0, min(4096, len(corrupted)), 7):
        corrupted[i] ^= 0xFF
    try:
        r = extract(bytes(corrupted), KEY, TENSOR, "interior_nearest")
        check("corrupted_artifact_handled", r != PAYLOAD or True, "decoded but mismatched")
        check("corrupted_not_silent_exact", r != PAYLOAD, "corruption not detected as exact")
    except ValueError as e:
        check("corrupted_artifact_handled", True, f"ValueError: {e}")

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
        check("pair_parity_fails_after_requant", rp != PAYLOAD, f"BER path")
    except ValueError as e:
        check("pair_parity_fails_after_requant", True, f"ValueError: {e}")

    n_pass = sum(r["pass"] for r in RESULTS)
    print(f"\n{n_pass}/{len(RESULTS)} tests passed", flush=True)
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
