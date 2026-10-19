# NF4 Anchor Preservation — Tensor-Generalization Matrix

Run id: `nes_tensor_matrix_20261016`
Status: COMPLETED — 48 measured rows (6 tensors × 2 methods × 4 cycles), **0 failures**.

## Frozen A–E results

A–E outputs are frozen; hashes recorded in `FROZEN_AE_HASHES.txt`. Not rerun, not modified.

| Artifact | sha256 |
|---|---|
| `out/anchor_ablation_ad_v2/observations.jsonl` | `3032834ede86f981f43dc56b18e5968d1b821e35f72480dc52157f25ab7f0d14` |
| `out/anchor_arm_e_v1/observations.jsonl` | `01c1fd373b62be166b551ec1261db378fe9831789b9a7a16ab07a298585d3f56` |
| `out/anchor_ablation_ad_v2/manifest.json` | `3ac5fe62969d5679b5a95635e0d2273da608e94abdd84acf36483d70ecb4dc85` |
| `out/anchor_arm_e_v1/manifest.json` | `f4c49ed8fe3622950b331433c3af94fcaf3bb3268b16fe05ece2d61392bbe0bf` |

## Predeclared tensor set (verified to exist before running)

| Tensor | depth | type | packed bytes | blocks | interior-eligible | pair-parity-eligible |
|---|---|---|---:|---:|---:|---:|
| `model.layers.0.self_attn.q_proj.weight` | early | attn q | 2,097,152 | 32,768 | 4,091,662 | 4,194,304 |
| `model.layers.11.self_attn.k_proj.weight` | mid | attn k | 262,144 | 4,096 | 506,113 | 524,288 |
| `model.layers.21.self_attn.v_proj.weight` | late | attn v | 262,144 | 4,096 | 506,157 | 524,288 |
| `model.layers.0.mlp.up_proj.weight` | early | mlp up | 5,767,168 | 90,112 | 11,127,569 | 11,534,336 |
| `model.layers.11.mlp.down_proj.weight` | mid | mlp down | 5,767,168 | 90,112 | 11,129,091 | 11,534,336 |
| `model.layers.21.mlp.gate_proj.weight` | late | mlp gate | 5,767,168 | 90,112 | 11,128,690 | 11,534,336 |

All six exist; all packed U8 NF4, blocksize 64, nested True. Preflight recorded in `manifest.json → preflight`.

## Provenance

- Checkpoint: `ckpt/tinyllama_clean`, sha256 `f84d0a8ec6e011d683ae7d1aec961d4bb021b254da9d84516185df428f3b37a7` (same as A–D)
- Codec: `nf4_anchor_channel.py` sha256 `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` (frozen, unchanged)
- Payload 10,000 bits, seed 20261016, blocksize 64, nested True
- Versions: torch 2.14.1+cpu, bitsandbytes 0.50.2, numpy 2.5.4, safetensors 0.8.0
- Platform: Linux x86_64, CPU only

## Sanity test (run before the matrix)

Single-tensor run on the known tensor reproduced the frozen A–D cycle-0 hashes exactly:
- `interior_nearest` → `783d43d8932897e70aa6…` ✅ (matches A/C)
- `pair_parity` → `2ee7ba3661b12ad6d1f1…` ✅ (matches D)

## Results per tensor × method (cycle 3)

| Tensor | Method | chg_anchor | hamming | scale_chg blocks | max abs scale Δ | no-endpoint | pos-agree | exact | BER |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| L0 attn q_proj | interior_nearest | **0** | 5,089 | 31,158 | 3.91e-03 | 0 | 1.0000 | **True** | 0.00000 |
| L0 attn q_proj | pair_parity | 110 | 5,856 | 25,904 | 3.91e-03 | 0 | 1.0000 | False | 0.00400 |
| L11 attn k_proj | interior_nearest | **0** | 5,092 | 3,222 | 1.83e-03 | 0 | 1.0000 | **True** | 0.00000 |
| L11 attn k_proj | pair_parity | 184 | 6,278 | 2,416 | 2.93e-03 | 0 | 1.0000 | False | 0.00460 |
| L21 attn v_proj | interior_nearest | **0** | 5,165 | 789 | 1.47e-03 | 0 | 1.0000 | **True** | 0.00000 |
| L21 attn v_proj | pair_parity | 162 | 6,191 | 2,501 | 1.10e-03 | 0 | 1.0000 | False | 0.00510 |
| L0 mlp up_proj | interior_nearest | **0** | 4,975 | 48 | 2.44e-04 | 0 | 1.0000 | **True** | 0.00000 |
| L0 mlp up_proj | pair_parity | 176 | 6,204 | 1,959 | 7.32e-04 | 0 | 1.0000 | False | 0.00250 |
| L11 mlp down_proj | interior_nearest | **0** | 5,077 | 3,714 | 7.32e-04 | 0 | 1.0000 | **True** | 0.00000 |
| L11 mlp down_proj | pair_parity | 170 | 6,553 | 8,650 | 7.32e-04 | 0 | 1.0000 | False | 0.00380 |
| L21 mlp gate_proj | interior_nearest | **0** | 5,116 | 5,493 | 7.32e-04 | 0 | 1.0000 | **True** | 0.00000 |
| L21 mlp gate_proj | pair_parity | 177 | 6,772 | 4,259 | 9.77e-04 | 0 | 1.0000 | False | 0.00430 |

### Cycle-0 → 3 detail (interior_nearest, all six tensors)
- `changed_anchor_codes = 0` at every cycle, every tensor.
- `blocks_without_endpoint_after_transform = 0` at every cycle, every tensor.
- `carrier_position_agreement = 1.0000` at every cycle, every tensor (carrier set is stable).
- `output_packed_sha256` constant across cycles 0–3 → byte-idempotent (12/12 tensor×method? no: 6/6 interior idempotent; pair_parity is **not** idempotent, 2 distinct hashes).
- `exact_recovery = True`, `BER = 0.0` at every cycle, every tensor.

### pair_parity baseline (all six tensors)
- cycle 0: exact recovery, but `changed_anchor_codes` 110–184 and `blocks_without_endpoint` 24–42.
- cycles 1–3: **exact recovery fails** on all six tensors, BER 0.0025–0.0051, checksum mismatch.
- Not byte-idempotent (2 distinct hashes: cycle 0 vs cycles 1–3).

## Aggregate (cycle 3)

| Method | exact | BER mean | BER max | anchors changed (sum) |
|---|---|---|---|---|
| interior_nearest | **6/6** | 0.00000 | 0.00000 | **0** |
| pair_parity | **0/6** | 0.00405 | 0.00510 | 979 |

## Findings

1. **interior_nearest generalizes across all six representative tensors** — early/mid/late depths, attention q/k/v, and MLP up/down/gate. Exact recovery with BER 0 at every cycle on every tensor.
2. **The anchor mechanism holds universally in this set**: 0 endpoint codes changed and 0 anchorless blocks created, on every tensor, at every cycle.
3. **pair_parity fails on every tensor** (0/6 exact), with BER 0.0025–0.0051 and 110–184 anchor codes changed. The baseline failure is not tensor-specific.
4. **Scale changes do not explain the difference.** Both methods show block-scale changes under requantization (max |Δscale| ~1e-4 to 4e-3). interior_nearest recovers exactly *despite* those scale changes, because its carrier set is stable and its carriers stay interior. So "scale preservation" is not the operative mechanism — anchor/carrier stability is.
5. **No failures to retain** — all 48 rows MEASURED, `failures: []`. No tensor was dropped.

## Limits (not yet tested)

- Single checkpoint (TinyLlama clean NF4), single payload size (10,000 bits), single seed, 3 requantization cycles.
- No detector/steganalysis, no capacity sweep, no PPL/utility, no second model.
- These remain open and were explicitly out of scope for this phase.

## Guardrails honoured

- No checkpoint downloaded (reused `ckpt/tinyllama_clean`).
- Frozen codec unmodified; A–E and historical observations untouched.
- New output directory `out/tensor_matrix_v1/`; nothing overwritten.
- No commit, push, or branch change; repo working tree clean.
- No residual-method or QSE/DCE data used.
