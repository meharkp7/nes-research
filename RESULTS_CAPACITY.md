# NF4 Anchor Preservation — Payload-Capacity Characterization

Run ids: `nes_capacity_tinyllama_20261016`, `nes_capacity_qwen_20261016`, plus four expansion runs.
Status: COMPLETED — **0 failures**, no capacity rejections at any requested size.

## Frozen prior outputs

All existing outputs frozen in `FROZEN_ALL_HASHES.txt` (30 files) and `FROZEN_AE_HASHES.txt`. Not rerun, not modified.

## Predeclared design

- Checkpoints: `ckpt/tinyllama_clean`, `ckpt/qwen_clean` (frozen)
- Representative tensor per model: `model.layers.0.self_attn.q_proj.weight`
- Sizes (bits): 1,000 / 5,000 / 10,000 / 20,000 / 50,000
- Seeds: 20261016, 20261017, 20261018
- Cycles: 0–3; codec, blocksize 64, nested True pinned
- Baseline: `pair_parity` at 10,000 bits only (3 seeds)

## Deterministic sanity test

`exp_capacity.py --self-test` PASS — framing verified (`framed_bits = payload_bits + 128`), eligibility = capacity, embed deterministic.

## Eligible carrier capacity (max feasible payload = capacity − 128 framing bits)

| Model | Tensor | eligible codes | max feasible payload bits |
|---|---|---:|---:|
| TinyLlama | L0 q_proj | 4,091,662 | 4,091,534 |
| Qwen2.5-3B | L0 q_proj | 4,051,738 | 4,051,610 |

All five requested sizes (max framed 50,128 bits) fit with **>98% headroom**; no size was capacity-rejected.

## interior_nearest — per-seed cycle-3 outcomes

### TinyLlama L0 q_proj
| bits | seed | framed | util | exact | BER | chg_anchor | pos-agree |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1000 | …016/017/018 | 1128 | 0.00028 | True×3 | 0.0 | 0 | 1.0000 |
| 5000 | …016/017/018 | 5128 | 0.00125 | True×3 | 0.0 | 0 | 1.0000 |
| 10000 | …016/017/018 | 10128 | 0.00248 | True×3 | 0.0 | 0 | 1.0000 |
| 20000 | …016/017/018 | 20128 | 0.00492 | True×3 | 0.0 | 0 | 1.0000 |
| 50000 | …016/017/018 | 50128 | 0.01225 | True×3 | 0.0 | 0 | 1.0000 |

### Qwen2.5-3B L0 q_proj
| bits | seed | framed | util | exact | BER | chg_anchor | pos-agree |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1000 | …016/017/018 | 1128 | 0.00028 | True×3 | 0.0 | 0 | 1.0000 |
| 5000 | …016/017/018 | 5128 | 0.00127 | True×3 | 0.0 | 0 | 1.0000 |
| 10000 | …016/017/018 | 10128 | 0.00250 | True×3 | 0.0 | 0 | 1.0000 |
| 20000 | …016/017/018 | 20128 | 0.00497 | True×3 | 0.0 | 0 | 1.0000 |
| 50000 | …016/017/018 | 50128 | 0.01237 | True×3 | 0.0 | 0 | 1.0000 |

**All 15 size×seed conditions per model pass** (exact recovery, BER 0, 0 anchor changes, position agreement 1.0000, byte-idempotent across cycles 0–3). The known 10,000-bit/seed-20261016 hashes reproduce the frozen evidence (`783d43d8…` TinyLlama, `904f523a…` Qwen).

## pair_parity baseline @ 10,000 bits (per-seed, cycle 3)

| Model | seed | exact | BER | chg_anchor |
|---|---:|---|---:|---:|
| TinyLlama | 20261016 | False | 0.00400 | 110 |
| TinyLlama | 20261017 | False | 0.00470 | 111 |
| TinyLlama | 20261018 | False | 0.00520 | 128 |
| Qwen2.5-3B | 20261016 | False | 0.00380 | 204 |
| Qwen2.5-3B | 20261017 | False | 0.00350 | 155 |
| Qwen2.5-3B | 20261018 | False | 0.00280 | 179 |

Baseline fails 6/6; failure is independent of seed.

## Expansion to additional tensors at 50,000 bits (requirement 4)

Since all sizes passed on the representative tensor, the winning setting (50,000 bits) was expanded to additional tensors across both models, including the smallest-capacity ones:

| Model | Tensor | eligible codes | max feasible bits | seeds | exact | BER | chg_anchor |
|---|---|---:|---:|---:|---|---:|---:|
| TinyLlama | L21 k_proj | 506,748 | 506,620 | 3 | True×3 | 0.0 | 0 |
| TinyLlama | L11 mlp.up_proj | 11,128,339 | 11,128,211 | 3 | True×3 | 0.0 | 0 |
| Qwen2.5-3B | L35 v_proj | 506,836 | 506,708 | 3 | True×3 | 0.0 | 0 |
| Qwen2.5-3B | L17 mlp.down_proj | 21,798,448 | 21,798,320 | 3 | True×3 | 0.0 | 0 |

All 12 expanded conditions pass. The smallest-capacity tensor tested (k_proj/v_proj, ~506k eligible codes) still has ~10× headroom for 50,000 bits.

## Findings

1. **interior_nearest succeeds at every requested payload size** (1,000–50,000 bits) on both checkpoints, all three seeds, all four cycles. No capacity limit was reached.
2. **Capacity is bounded by interior-code count**, not by the method: max feasible payload ≈ eligible codes − 128 bits. TinyLlama q_proj ≈ 4.09 M bits; Qwen q_proj ≈ 4.05 M bits. The smallest tensor tested supports ≈ 0.51 M bits.
3. **Carrier utilization is tiny** at these sizes (0.03%–1.2%), so robustness is not a saturation effect.
4. **Failure is baseline-specific and seed-independent**: pair_parity fails 6/6 at 10,000 bits; interior_nearest passes 15/15 per model.
5. **No failures to retain** — 0 failures across all runs; every requested size fit.

## Limits

- Two checkpoints; one representative tensor for the size sweep plus four expansion tensors.
- Payload sizes above 50,000 bits were not tested (not requested); the reported max feasible is analytic capacity, not a measured ceiling.
- No detector, utility/PPL, or additional models.

## Guardrails honoured

- Frozen codec unmodified; all prior outputs preserved and hashed.
- New output directories only; nothing overwritten.
- No branch change, commit, or push; repo working tree clean.
- Kept separate from the five residual methods and QSE/DCE tracks.
