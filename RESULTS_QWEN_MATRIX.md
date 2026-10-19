# NF4 Anchor Preservation — Cross-Model Replication on Qwen2.5-3B

Run id: `nes_qwen_matrix_20261016`
Status: COMPLETED — 48 measured rows (6 tensors × 2 methods × 4 cycles), **0 failures**.

## Preconditions reported before download

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-3B` |
| Pinned revision | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` (current `main`; no historical model revision was ever recorded) |
| Expected download | ~6.18 GB (shards 3.969 GB + 2.203 GB) + tokenizer |
| Disk at decision | `/workspace` 21 GB free, `/` 57 GB free |
| RAM | 15 GB total, CPU-only |
| Compatibility | NF4, blocksize 64, nested True — matches codec |

Note: the repo string `c16e1a848d0e386d4daa50872927407aa433df8c` is the *nes-research repo's own* git revision, not a Qwen model revision (HF returned `RevisionNotFoundError` for it). No historical Qwen model revision exists to pin, consistent with the earlier UNVERIFIED provenance finding.

## Frozen prior outputs

Tensor-matrix (TinyLlama) and A–E hashes appended to `FROZEN_AE_HASHES.txt`. Not rerun, not modified:
`out/tensor_matrix_v1/observations.jsonl` = `bcf35836…`, `out/anchor_ablation_ad_v2/observations.jsonl` = `3032834e…`, `out/anchor_arm_e_v1/observations.jsonl` = `01c1fd37…`.

## Provenance

- FP16 source (pinned rev): shard1 `f9558df9…`, shard2 `51410930…`
- Clean NF4 checkpoint: `ckpt/qwen_clean` — shard1 `10b8e0f8…`, shard2 `70512912…`; export report status **PASS**, stored dtype bfloat16, nested True
- Codec: `nf4_anchor_channel.py` sha256 `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` (frozen, unchanged)
- Payload 10,000 bits, seed 20261016, blocksize 64, nested True
- Versions: torch 2.14.1+cpu, bitsandbytes 0.50.2, numpy 2.5.4, safetensors 0.8.0

## Predeclared tensors (verified before execution)

| Tensor | depth | type |
|---|---|---|
| `model.layers.0.self_attn.q_proj.weight` | early | attn q |
| `model.layers.17.self_attn.k_proj.weight` | middle | attn k |
| `model.layers.35.self_attn.v_proj.weight` | late | attn v |
| `model.layers.0.mlp.up_proj.weight` | early | mlp up |
| `model.layers.17.mlp.down_proj.weight` | middle | mlp down |
| `model.layers.35.mlp.gate_proj.weight` | late | mlp gate |

All six exist; all packed U8 NF4, blocksize 64, nested True. Qwen2.5-3B has 36 layers (0–35), same tensor-type set as TinyLlama.

## Sanity test (before the matrix)

Single tensor `L0 q_proj`, cycle 0. Result reproduced the historical committed evidence **byte-for-byte**:
- source tensor sha256 `ce26ecb75d4ce505…` — matches historical
- `interior_nearest` output `904f523abcca6cfbd5f0…` — matches historical
- `pair_parity` output `971cc9b077410ba6810b…` — matches historical

## Results per tensor × method (cycle 3)

| Tensor | Method | chg_anchor | hamming | scale-chg blocks | no-endpoint | pos-agree | exact | BER |
|---|---|---:|---:|---:|---:|---:|---|---:|
| L0 attn q_proj | interior_nearest | **0** | 5,037 | 13,639 | 0 | 1.0000 | **True** | 0.00000 |
| L0 attn q_proj | pair_parity | 204 | 6,566 | 11,042 | 0 | 1.0000 | False | 0.00380 |
| L17 attn k_proj | interior_nearest | **0** | 5,110 | 4,303 | 0 | 1.0000 | **True** | 0.00000 |
| L17 attn k_proj | pair_parity | 162 | 6,533 | 3,108 | 0 | 1.0000 | False | 0.00700 |
| L35 attn v_proj | interior_nearest | **0** | 4,998 | 675 | 0 | 1.0000 | **True** | 0.00000 |
| L35 attn v_proj | pair_parity | 172 | 6,227 | 1,948 | 0 | 1.0000 | False | 0.00470 |
| L0 mlp up_proj | interior_nearest | **0** | 5,048 | 1,009 | 0 | 1.0000 | **True** | 0.00000 |
| L0 mlp up_proj | pair_parity | 184 | 6,555 | 2,868 | 0 | 1.0000 | False | 0.00360 |
| L17 mlp down_proj | interior_nearest | **0** | 5,085 | 7,254 | 0 | 1.0000 | **True** | 0.00000 |
| L17 mlp down_proj | pair_parity | 185 | 7,006 | 4,851 | 0 | 1.0000 | False | 0.00450 |
| L35 mlp gate_proj | interior_nearest | **0** | 5,057 | 6,059 | 0 | 1.0000 | **True** | 0.00000 |
| L35 mlp gate_proj | pair_parity | 170 | 6,566 | 5,421 | 0 | 1.0000 | False | 0.00350 |

## Aggregate (cycle 3)

| Method | exact | BER mean | BER max | anchors changed (sum) | idempotent cycles 0–3 |
|---|---|---|---|---|---|
| interior_nearest | **6/6** | 0.00000 | 0.00000 | **0** | yes (1 distinct hash) |
| pair_parity | **0/6** | 0.00452 | 0.00700 | 1,077 | no (2 distinct hashes) |

## Cross-model comparison vs TinyLlama

| Model | Method | exact | BER mean | anchors changed |
|---|---|---|---|---|
| TinyLlama-1.1B | interior_nearest | 6/6 | 0.00000 | 0 |
| TinyLlama-1.1B | pair_parity | 0/6 | 0.00405 | 979 |
| **Qwen2.5-3B** | **interior_nearest** | **6/6** | **0.00000** | **0** |
| **Qwen2.5-3B** | **pair_parity** | **0/6** | **0.00452** | **1,077** |

## Findings

1. **interior_nearest replicates on Qwen2.5-3B** — exact recovery, BER 0, at every cycle on all six representative tensors (early/middle/late, attention q/k/v, MLP up/down/gate).
2. **Anchor mechanism holds cross-model**: 0 endpoint codes changed, 0 anchorless blocks created, carrier positions stable (agreement 1.0000), byte-idempotent across cycles.
3. **pair_parity fails on every Qwen tensor too** (0/6), BER 0.0035–0.0070. Failure is method-specific, not model- or tensor-specific.
4. **Different architecture, same result.** Qwen2.5-3B (36 layers, bfloat16, different width) shows the same pattern as TinyLlama (22 layers, float16).
5. **No failures to retain** — all 48 rows MEASURED, `failures: []`.

## Limits

- Two checkpoints only; single payload size (10k), single seed, 3 cycles.
- No detector, capacity, utility/PPL, or additional models.
- The exact pinned revision is the current `main`; no historical Qwen revision exists to compare against.

## Guardrails honoured

- Preconditions (revision, size, disk) reported before download.
- Frozen codec unmodified; A–E and TinyLlama matrix outputs preserved.
- New output directory `out/qwen_matrix_v1/`; nothing overwritten.
- No branch change, commit, or push; repo working tree clean.
- Kept separate from the five residual methods and QSE/DCE tracks.
