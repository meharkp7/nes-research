# StableLM-2-1.6B End-to-End Round-Trip Report (Gates A, B, C)

Run: `stablelm_qualification/roundtrip_stream_v1/`. Status: **COMPLETE**.
Protocol frozen in `QUALIFICATION_PROTOCOL_STABLELM.md` before this run.

## 1. Provenance (Gate A)

| Item | Value |
|---|---|
| Model | `stabilityai/stablelm-2-1_6b` @ `f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef` (pinned in `download_ckpt.py`) |
| Clean NF4 checkpoint | `ckpt/stablelm_clean`, `model.safetensors` sha256 `bff3b65ebb686092b7234ddef66d3a69b762299826ac3e29e2015343c432d591` |
| FP16 source | `ckpt/stablelm_fp16`, sha256 `8bdf317e2b35ab5c8009cbb6c7ce495e4e608a6b9b843d44054edf25b8c5860d` |
| Codec sha256 | `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` (unchanged) |
| NF4 config | blocksize 64, nested statistics True, quant_type nf4 |
| Runner | `exp_stablelm_roundtrip_stream.py` |
| Conditions | 270 = 3 methods × 6 tensors × 5 sizes × 3 seeds |
| Cycle extractions | 1,080 = 270 × 4 cycles (0–3) |
| Embed failures | **0** |

Every record carries: condition name, method, tensor, bits, seed, blocksize,
nested flag, quant_type, clean and stego tensor sha256, payload sha256,
eligible codes, changed codes, carrier agreement, anchor counts, per-cycle
artifact sha256, and elapsed time.

## 2. Design — why this satisfies "independent process" (Gate B)

Embedding and extraction do **not** share an in-memory object:

1. For each condition the runner embeds with the frozen codec and requantizes
   through cycles 0–3, writing each cycle's packed artifact to a temp file.
2. It then spawns a **fresh Python subprocess** that opens only that file and
   calls `extract(artifact_bytes, key, tensor, method)`. The extractor
   reconstructs carrier positions from the artifact alone; it never receives the
   embedding object or the payload.
3. The subprocess returns exact/​length/​checksum/​decode-error as JSON; the temp
   artifact is deleted.

The payload and key are regenerated in the subprocess from the seed via the
documented `payload_for` / `key_for`; no hidden state is passed.

## 3. Gate B results — exact end-to-end recovery

`interior_nearest`: **360/360 cycle extractions exact (100%)**.

| Check | Result |
|---|---|
| exact_recovery | 360/360 |
| length_ok | 360/360 |
| checksum_ok | 360/360 |
| payload_ber | 0.0 everywhere |
| carrier_agreement | 1.0 everywhere |
| decode_error | none |

`interior_nearest` exact by cycle: c0 90/90, c1 90/90, c2 90/90, c3 90/90.
By payload size: 1k 72/72, 5k 72/72, 10k 72/72, 20k 72/72, 50k 72/72.
By seed: 20261016 120/120, 20261017 120/120, 20261018 120/120.
By tensor: each of the six tensors 60/60.

**Gate B: PASS.**

## 4. Gate C results — requantization robustness and baselines

| Method | Exact cycle-extractions | Anchor codes changed (sum) | Carrier agreement |
|---|---|---|---|
| `interior_nearest` | **360/360** | **0** | 1.0 |
| `pair_parity` | 93/360 | 88,297 | 1.0 |
| `nearest_parity` | 93/360 | 88,150 | 1.0 |

Baselines by cycle: `pair_parity` c0 90/90, then 1/90 at c1, c2, c3;
`nearest_parity` identical. Both baselines fail at the first requantization with
`ValueError: Envelope checksum mismatch`. Their semantics were **not** modified —
this is the frozen baseline implementation.

Additional `interior_nearest` invariants:
- **Byte idempotence:** for all 90 conditions, the artifact sha256 is identical
  across cycles 0–3 (1 distinct hash per condition).
- **Anchor preservation:** `anchor_codes_delta = 0` and
  `changed_anchor_codes_vs_source = 0` in every condition and cycle.
- **No endpointless blocks:** `blocks_without_endpoint = 0` everywhere.
- Packed-code hamming vs source at cycle 3 ≤ 25,306 codes out of ≥4.2M (≤0.6%),
  consistent with a sparse carrier set.

**Gate C: PASS** for cycles 0–3, five payload sizes, three seeds, six tensors.
Stress extension to cycles 0–5 is covered by the regression test
(`requantize_cycles_0_5_exact`), not by the full matrix; reported separately.

## 5. Regression tests (Gate F support)

`test_stablelm_codec_regressions.py` — **11/11 PASS** (standalone, no pytest):

| Test | Result |
|---|---|
| deterministic_fixed_seed | PASS |
| endpoints_unchanged | PASS |
| interior_only | PASS |
| carrier_eligibility_stable | PASS |
| fresh_process_extraction (subprocess) | PASS |
| requantize_cycles_0_5_exact | PASS |
| corrupted_artifact_handled | PASS |
| truncated_artifact_handled | PASS |
| wrong_key_not_exact | PASS |
| over_capacity_rejected | PASS |
| pair_parity_fails_after_requant | PASS |

Command: `/workspace/nes_env/bin/python stablelm_qualification/test_stablelm_codec_regressions.py`
(exit code 0).

## 6. Failures, exclusions, deviations

- No embed failures; no exclusions.
- An earlier run (`roundtrip_v1`, non-streaming) was aborted for **disk-space**
  reasons (270 conditions × ~14 MB artifacts ≈ 3.8 GB vs ~3.7 GB free). It is
  retained as `roundtrip_v1_ABORTED_DISK/` (artifacts removed to reclaim space;
  records preserved). The streaming runner replaced it and is the reported result.
- `pair_parity` / `nearest_parity` failures are recorded per condition and
  reported above; they are not discarded.
- The full matrix used cycles 0–3; cycles 4–5 are only in the regression test.

## 7. Gate A/B/C verdict

| Gate | Verdict |
|---|---|
| A — implementation and artifact integrity | **PASS** |
| B — independent-process exact recovery | **PASS** |
| C — multi-sample and transformation robustness | **PASS** |
