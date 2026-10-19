# Full-Model Utility (Perplexity) Evaluation — `interior_nearest`

Run ids: `nes_utility_20261016` (`utility_v2`, TinyLlama + Qwen 10k),
`utility_qwen50k_v3` (Qwen 50k). Status: COMPLETED.

This measures whether embedding a payload into one weight tensor of an otherwise
unchanged model preserves language-model utility. The frozen codec was used
unchanged; no capacity or detectability experiment was rerun.

## 1. Configuration and provenance

| Item | Value |
|---|---|
| Method | `interior_nearest` (frozen) |
| Tensor | `model.layers.0.self_attn.q_proj.weight` (both models) |
| Payload sizes | 10,000 and 50,000 bits |
| Seeds | primary `20261016`; replication `20261017`, `20261018` |
| Codec revision | `src/experiments/nf4_anchor_channel.py` sha256 `07ce008d…` (unchanged) |
| Checkpoint runner | `scripts/run_nf4_anchor_pilot.py` (`load_tensor`, `sha_file`) |
| New scripts | `scripts/exp_utility.py`, `scripts/exp_utility_qwen50k_resumable.py` |
| Quantization | nf4, blocksize 64, nested statistics, uint8 storage |

### Model provenance

| Model | Pinned revision | Source tensor sha256 | Note |
|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` | `283aa75d1f34daff…` | matches frozen `tensor_matrix_v1` |
| Qwen2.5-3B | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | `ce26ecb75d4ce505…` | matches frozen `qwen_sanity` |

**Qwen provenance caveat (preserved):** the pinned Qwen revision
`3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` is the current `main`, not a
historically recorded revision. A previously assumed repo string
`c16e1a848d0e386d4daa50872927407aa433df8c` is the *nes-research repo's own* git
revision, not a Qwen model revision (HF returned `RevisionNotFoundError`). No
historical Qwen model revision exists to pin, so Qwen provenance remains
**UNVERIFIED** at the revision level. The source-tensor hash does match the
frozen `qwen_sanity` manifest, which anchors this run to the same artifact used
by prior experiments.

## 2. Corpus and tokenization

| Item | Value |
|---|---|
| Corpus | `Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `test` |
| File sha256 | recorded in `manifest.json` (`_corpus_wikitext2_test.parquet`) |
| Tokenization | each model's own tokenizer from the `*_fp16` dirs, `add_special_tokens=False` |
| Context length | 512 tokens, non-overlapping windows, no truncation within a window |
| Sequences | 30 per condition (target 1,000 not feasible on CPU — see limitations) |
| Tokens | 30 × 511 = **15,330** evaluated tokens per condition |
| Corpus tokens | TinyLlama 338,533 (661 windows available); Qwen 298,938 (583 windows) |

Perplexity is token-weighted: per-sequence NLL is summed, divided by total
tokens, then exponentiated. This was validated against the model's own scalar
loss (manual mean NLL == `model(labels=...).loss`, e.g. Qwen clean 2.31641 ==
2.31641), confirming the aggregation is correct rather than an average of batch
perplexities.

## 3. Results — clean vs embedded

Clean baselines: TinyLlama PPL **10.69483**, Qwen PPL **10.13918** (15,330 tokens each).

### Primary seed 20261016

| Model | Size (bits) | Clean PPL | Embedded PPL | Δ PPL (abs) | Δ PPL (rel) | Paired 95% CI (ΔNLL) |
|---|---|---:|---:|---:|---:|---|
| TinyLlama | 10,000 | 10.69483 | 10.69369 | −0.00114 | **−0.0107%** | [−0.274, +0.154] |
| TinyLlama | 50,000 | 10.69483 | 10.69317 | −0.00166 | **−0.0155%** | [−0.367, +0.190] |
| Qwen | 10,000 | 10.13918 | 10.13959 | +0.00041 | **+0.0040%** | [−0.415, +0.448] |
| Qwen | 50,000 | 10.13918 | 10.13321 | −0.00598 | **−0.0589%** | [−0.769, +0.206] |

### Replication seeds

| Model | Size | Seed | Embedded PPL | Δ rel | Paired 95% CI |
|---|---|---:|---:|---:|---|
| TinyLlama | 10,000 | 20261017 | 10.69436 | −0.0043% | [−0.266, +0.230] |
| TinyLlama | 10,000 | 20261018 | 10.69458 | −0.0023% | [−0.252, +0.224] |
| TinyLlama | 50,000 | 20261017 | 10.69379 | −0.0097% | [−0.370, +0.278] |
| TinyLlama | 50,000 | 20261018 | 10.69402 | −0.0075% | [−0.322, +0.248] |
| Qwen | 10,000 | 20261017 | 10.14039 | +0.0119% | [−0.329, +0.447] |
| Qwen | 10,000 | 20261018 | 10.13306 | −0.0604% | [−0.764, +0.160] |
| Qwen | 50,000 | 20261017 | 10.14232 | +0.0310% | [−0.137, +0.437] |
| Qwen | 50,000 | 20261018 | 10.13649 | −0.0266% | [−0.531, +0.286] |

### Summary across the 3 seeds

| Model | Size | Mean Δ rel | Range | All CIs cross 0? |
|---|---|---:|---|---|
| TinyLlama | 10,000 | −0.0058% | −0.0107 … −0.0023% | yes |
| TinyLlama | 50,000 | −0.0109% | −0.0155 … −0.0075% | yes |
| Qwen | 10,000 | −0.0148% | −0.0604 … +0.0119% | yes |
| Qwen | 50,000 | −0.0182% | −0.0589 … +0.0310% | yes |

Every embedded condition lies within ±0.06% of clean PPL, and **all 12 paired
95% CIs include zero**. The two payload sizes are statistically
indistinguishable from each other and from clean.

## 4. Integrity, recovery, and load checks

For all 12 embedded conditions:

| Check | Result |
|---|---|
| Checkpoint loads, finite logits and loss | PASS (all conditions) |
| Exact payload recovery from saved tensor | PASS (12/12, `exact_recovery=true`) |
| Only intended tensor differs from clean | PASS (12/12, `differing_keys == [q_proj]`) |
| Source tensor hash matches frozen manifest | PASS (both models) |
| Changed code counts | TinyLlama 10k ≈ 5,025–5,102; 50k ≈ 25,012–25,135; Qwen 10k ≈ 5,035–5,093; 50k ≈ 24,874–25,096 |

Payload recovery correctness (12/12 exact) is reported separately from utility
and does not by itself imply any utility property.

## 5. Limitations and incomplete runs

- **Reduced scope:** 30 sequences / 15,330 tokens per condition, not the 1,000
  sequences / 100,000 tokens target. The sandbox has no GPU (4 CPU threads);
  throughput is ~100 tok/s (TinyLlama) and ~36 tok/s (Qwen), so the full target
  was not feasible. The reduced scope is reported explicitly and was not
  extrapolated.
- **One tensor, one layer, one method:** only `layers.0.self_attn.q_proj` with
  `interior_nearest`. Results do not generalize to other tensors, layers, or the
  other candidate methods.
- **Two models only.**
- **Paired CIs are wide** relative to the effect; the test cannot resolve
  sub-0.5% PPL differences at this sample size.
- **Interrupted runs preserved, not overwritten:**
  - `utility_v1` — all 6 conditions computed; crashed on a `results.json`
    exclusive-write bug (fixed). Preserved via hashes and log.
  - `utility_v2` — killed by a sandbox restart after 10/14 conditions; its
    completed conditions are the TinyLlama and Qwen-10k results above.
  - `utility_qwen50k_v1`, `_v2` — killed by sandbox restarts before any
    condition completed; manifests preserved.
  - `utility_qwen50k_v3` — completed all three Qwen 50k conditions via a
    resumable driver after two restarts.
- **Peak RSS not instrumented** for the interrupted runs; observed host memory
  use stayed within the 15 GB sandbox (≈13 GB available throughout).
- **No generation-quality or semantic-equivalence claim:** a small PPL change
  does not establish unchanged generation quality.

## 6. Conclusion

**What the results do establish.** On one representative tensor
(`layers.0.self_attn.q_proj`) of TinyLlama-1.1B and Qwen2.5-3B, embedding a
10,000- or 50,000-bit payload with the frozen `interior_nearest` codec produced
a fully loadable model whose perplexity on a fixed wikitext-2 test split is
within ±0.06% of the clean baseline, with all paired 95% CIs containing zero,
while payload extraction recovered the exact bits and only the intended tensor
changed. Within this scope, the method preserves full-model utility.

**What the results do not establish.** They do not establish utility
preservation for other tensors, layers, models, or methods; they do not
establish unchanged generation quality or semantic equivalence; and the
reduced evaluation scope (15,330 tokens, 30 sequences) means small PPL
differences below ~0.5% cannot be resolved. No claim of universal robustness or
undetectability is made or supported.

## 7. Deliverables (outside the repo)

`/workspace/nes_run_20261016_anchor_ablation/`
- `code/nes-llm/scripts/exp_utility.py`, `exp_utility_qwen50k_resumable.py`
- `out/utility_v2/` (manifest, results, per-condition stego tensors + materialization)
- `out/utility_qwen50k_v3/` (clean baseline, 3 conditions, stego tensors)
- `out/_corpus_wikitext2_test.parquet` (frozen corpus)
- extended `FROZEN_ALL_HASHES.txt`

## 8. Guardrails honoured

- Frozen codec unmodified; capacity and detectability experiments not rerun.
- New output directories only; failed run dirs preserved, never overwritten.
- No branch change, commit, or push; repo working tree clean at `aa7e1a7`.
- Investigation kept separate from the five residual-method experiments and the
  QSE/DCE packed-NF4 track.
