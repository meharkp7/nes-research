# NF4 Anchor Preservation — Arm E (Interior Eligibility, Baseline Parity Op)

Run id: `nes_anchor_arm_e_20261016`
Status: COMPLETED — 4 measured rows (cycles 0–3), no errors.

## What Arm E is

Arm E is the missing control that separates **endpoint preservation** from the
**embedding operation**:

- Eligibility = interior NF4 codes **1–14** (no anchorless-block restriction).
  Endpoint codes 0 and 15 are **never selectable** as carriers.
- Operation = the **baseline parity op** `code & 14 | bit` (the same op as
  `pair_parity` / arm D).

Because the baseline op can map an interior carrier onto an endpoint
(1→0 when bit=0; 14→15 when bit=1), E writes endpoint codes **without ever
selecting** an endpoint as a carrier. That is exactly the confound the A–D run
could not separate.

## Provenance (verified equal to the A–D run before comparing)

| Item | Value |
|---|---|
| Source tensor sha256 | `283aa75d1f34daff33b4aae984c42fac6a8741deecd3348eaed6459852fd2161` (identical to A–D) |
| Clean NF4 checkpoint | `ckpt/tinyllama_clean` — same `checkpoint_hashes` as A–D |
| Codec revision | `nf4_anchor_channel.py` sha256 `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` (unchanged) |
| Payload | 10,000 bits, seed 20261016, sha256 identical to A–D |
| blocksize / nested | 64 / True |
| Versions | torch 2.14.1+cpu, bitsandbytes 0.50.2, numpy 2.5.4, safetensors 0.8.0 (identical to A–D) |
| Platform | Linux x86_64, CPU only |

## Exact commands

```bash
cd /workspace/nes_run_20261016_anchor_ablation/code/nes-llm

# 1. deterministic unit test (runs before the full run)
/workspace/nes_env/bin/python scripts/exp_anchor_ablation_e.py --self-test

# 2. full Arm E run (new output directory; A-D outputs untouched)
HF_HOME=/workspace/nes_run_20261016_anchor_ablation/hf_home \
/workspace/nes_env/bin/python scripts/exp_anchor_ablation_e.py \
  --checkpoint /workspace/nes_run_20261016_anchor_ablation/ckpt/tinyllama_clean \
  --output     /workspace/nes_run_20261016_anchor_ablation/out/anchor_arm_e_v1
```

Self-test result (deterministic, 40-block synthetic tensor):

```
self-test PASS: {"selected_endpoint_origin_codes": 0, "carriers_written_to_endpoint": 26,
                 "changed_codes": 189, "embed_eligibility_changed": true,
                 "same_artifact_recovered": false, "decode_error": "Envelope magic mismatch"}
```

## Arm E results (cycles 0–3)

| cycle | eligible | changed | anchor changes | carriers→endpoint | hamming | no-endpoint blks | pos-agree | exact | BER | decode_error |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 0 | 4,091,662 | 5,089 | **0** | 130 | 5,089 | 0 | 1.0000 | False | 0.49130 | Envelope checksum mismatch |
| 1 | 4,091,662 | 5,089 | 0 | 130 | 5,089 | 0 | 1.0000 | False | 0.49130 | Envelope checksum mismatch |
| 2 | 4,091,662 | 5,089 | 0 | 130 | 5,089 | 0 | 1.0000 | False | 0.49130 | Envelope checksum mismatch |
| 3 | 4,091,662 | 5,089 | 0 | 130 | 5,089 | 0 | 1.0000 | False | 0.49130 | Envelope checksum mismatch |

- `output_packed_sha256` = `10e7c45bbfbc0c6c44db6563…` at **every** cycle (byte-idempotent).
- `selected_endpoint_origin_codes = 0` at every cycle (no endpoint ever chosen as a carrier).
- `carriers_written_to_endpoint = 130` and `eligibility_changed_after_transform = 130`
  (the 130 positions the op pushed onto endpoints leave the interior set).
- `changed_anchor_codes = 0`: E never **selects** an endpoint, so no original
  endpoint code is directly rewritten.

## Five-arm comparison (steady state, cycle 3)

| Arm | eligibility | op | anchor changes | exact | BER |
|---|---|---|---:|---|---:|
| A `interior_nearest_anchored` | anchored blocks, codes 1–14 | nearest same-parity | 0 | **True** | 0.0 |
| C `interior_nearest_unanchored` | codes 1–14 | nearest same-parity | 0 | **True** | 0.0 |
| B `nearest_parity_anchored` | anchored blocks, all codes | `code&14\|bit` | 110 | False | 0.00380 |
| D `pair_parity_unrestricted` | all codes | `code&14\|bit` | 110 | False | 0.00400 |
| **E `interior_parity_baseline_op`** | **codes 1–14** | **`code&14\|bit`** | **0** | **False** | **0.49130** |

## Causal conclusion

Arm E **fails** (BER 0.49130) while sharing the *exact same interior
eligibility set* as the passing arms A and C, and sharing the *exact same
operation* as the failing arms B and D. Therefore:

1. **The operation is the operative factor, not the eligibility mask.**
   Holding eligibility fixed at codes 1–14, switching from the nearest-same-parity
   op (A/C, pass) to the baseline parity op (E, fail) breaks recovery.
2. **Restricting to interior codes is not sufficient on its own.** E restricts to
   interior codes and still fails, because the baseline op pushes carriers onto
   endpoints (130 of them here), which shifts the receiver's interior set and
   causes a checksum mismatch.
3. **The failure mode is a receiver-side eligibility shift.** The op's writes to
   endpoints change which codes are interior, so sender and receiver no longer
   agree on the carrier set. `changed_anchor_codes = 0` for E shows the failure
   is *not* from directly rewriting an original endpoint, but from the op
   *creating* new endpoints inside previously-interior carriers.

Combined with A–D, the defensible statement is: requantization robustness is
produced by the **nearest-same-parity interior mapping**, which both (i) keeps
carriers inside the interior set and (ii) leaves block endpoints untouched. The
anchorless-block exclusion is neither necessary (C) nor sufficient (E).

## Guardrails honoured

- No checkpoint downloaded (reused `ckpt/tinyllama_clean`).
- Frozen codec and historical 2026-10-16 observations untouched.
- A–D outputs untouched; Arm E wrote to a new directory `out/anchor_arm_e_v1/`.
- No branch switch, commit, or push; repo working tree remains clean.
- No residual-method or QSE/DCE data used.
