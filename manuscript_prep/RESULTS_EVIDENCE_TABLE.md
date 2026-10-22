# RESULTS_EVIDENCE_TABLE.md — Track C Evidence Map

**Status:** Complete — every reported Track C result mapped to source file  
**Date:** 2026-10-21  
**Commit:** `5b40120` (`research/contract-b-b14-and-evaluation`)  
**Frozen codec:** `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764`

---

## 1. Roundtrip Exact Recovery (Gates A2/A3)

| Claim | Models | Tensor Count | Sizes | Seeds | Cycles | Exact | Source File |
|-------|--------|--------------|-------|-------|--------|-------|-------------|
| `interior_nearest` 360/360 exact per model | SmolLM2, Qwen2.5, Pythia, Falcon3, StableLM | 6 tensors × 5 sizes × 3 seeds = 90 conditions × 4 cycles = 360 | 1k, 5k, 10k, 20k, 50k | 20261016, 20261017, 20261018 | 0–3 | 360/360 per model | `expansion_qualification/expansion_audit_table.csv` (4320 rows) |
| Baseline `pair_parity` failures | All 4 exp. models | 6 tensors × 5 sizes × 3 seeds × 4 cycles | 1k–50k | 3 seeds | 0–3 | 90–108/360 | `expansion_audit_table.csv` |
| Baseline `nearest_parity` failures | All 4 exp. models | Same as above | Same | Same | Same | 90–108/360 | `expansion_audit_table.csv` |
| StableLM reference | StableLM-2-1.6B | 6 tensors × 5 sizes × 3 seeds × 4 cycles | 1k–50k | 3 seeds | 0–3 | 360/360 | `stablelm_qualification/roundtrip_stream_v1/summary.json` |

**Verification:** `expansion_audit_table.csv` has 4321 lines (1 header + 4320 rows = 1080 conditions × 4 cycles). 1080 conditions / 4 models = 270 conditions/model = 6 tensors × 5 sizes × 3 seeds = 90 conditions × 3 methods = 270. Wait — 6 tensors × 5 sizes × 3 seeds = 90 per method. 3 methods × 90 = 270 per model. 4 models × 270 = 1080. × 4 cycles = 4320. ✅

**Exact recovery denominator:** 360 per model = 6 tensors × 5 sizes × 3 seeds × 4 cycles = 360. ✅

**Source:** `expansion_qualification/expansion_audit_table.csv` (4320 rows, 0 missing, all VERIFIED)

---

## 2. Integrity (Gate A1)

| Model | n_tensors | Only intended differs | Shape preserved | Payload exact | Finite logits/loss | Source |
|-------|-----------|----------------------|-----------------|---------------|-------------------|--------|
| SmolLM2 | 1058 | true | true | true | true | `expansion_qualification/smollm2/integrity.json` |
| Qwen2.5 | 1318 | true | true | true | true | `expansion_qualification/qwen15/integrity.json` |
| Pythia | 772 | true | true | true | true | `expansion_qualification/pythia/integrity.json` |
| Falcon3 | 795 | true | true | true | true | `expansion_qualification/falcon/integrity.json` |
| StableLM | 1180 | true | true | true | true | `stablelm_qualification/integrity_10k.json` |

**Verification:** Each `integrity.json` shows `differing_keys == [target_tensor]`, `shape_preserved: true`, `payload_exact: true`, `finite_logits_loss: true`.

---

## 3. Roundtrip Exact Recovery (A2/A3) — Detailed

### Expansion Models (4 models × 6 tensors × 5 sizes × 3 seeds × 4 cycles)

| Model | Method | Exact | Total | Anchor Changes | Carrier Agreement |
|-------|--------|-------|-------|----------------|-------------------|
| SmolLM2 | interior_nearest | 360/360 | 360 | 0 | 1.0 |
| SmolLM2 | nearest_parity | 90/360 | 360 | — | — |
| SmolLM2 | pair_parity | 90/360 | 360 | — | — |
| Qwen2.5 | interior_nearest | 360/360 | 360 | 0 | 1.0 |
| Qwen2.5 | nearest_parity | 90/360 | 360 | — | — |
| Qwen2.5 | pair_parity | 90/360 | 360 | — | — |
| Pythia | interior_nearest | 360/360 | 360 | 0 | 1.0 |
| Pythia | nearest_parity | 90/360 | 360 | — | — |
| Pythia | pair_parity | 90/360 | 360 | — | — |
| Falcon3 | interior_nearest | 360/360 | 360 | 0 | 1.0 |
| Falcon3 | nearest_parity | 108/360 | 360 | — | — |
| Falcon3 | pair_parity | 108/360 | 360 | — | — |

**StableLM (reference):** `stablelm_qualification/roundtrip_stream_v1/summary.json` — 360/360 interior, 93/360 baselines.

**Source:** `expansion_qualification/expansion_audit_table.csv` (4320 rows, all VERIFIED); `stablelm_qualification/roundtrip_stream_v1/summary.json`

---

## 3. Integrity (A1) — Detailed

| Model | n_tensors | Only Intended Differs | Shape Preserved | Payload Exact | Finite Logits/Loss | Source |
|-------|-----------|----------------------|-----------------|---------------|-------------------|--------|
| SmolLM2 | 1058 | true | true | true | true | `expansion_qualification/smollm2/integrity.json` |
| Qwen2.5 | 1318 | true | true | true | true | `expansion_qualification/qwen15/integrity.json` |
| Pythia | 772 | true | true | true | true | `expansion_qualification/pythia/integrity.json` |
| Falcon3 | 795 | true | true | true | true | `expansion_qualification/falcon/integrity.json` |
| StableLM | 1180 | true | true | true | true | `stablelm_qualification/integrity_10k.json` |

**Verification:** Each `integrity.json` contains `differing_keys == [target_tensor]`, `shape_preserved: true`, `payload_exact: true`, `finite_logits_loss: true`.

---

## 4. Detectability (Gate A4) — Main Evaluation

### Main Evaluation (4 expansion models + StableLM reference)

| Model | Control (clean) | GB AUC | LR AUC | MLP AUC | Max TPR@1% | Groups | Source |
|-------|-----------------|--------|--------|---------|------------|--------|--------|
| SmolLM2 | 0.5031 | 0.4999 (p=0.46) | 0.5012 (p=0.35) | 0.4999 (p=0.54) | 0.0100 | 6 | `expansion_qualification/smollm2/detectability_v1/detectability_results.json` |
| Qwen2.5 | 0.5120 | 0.5025 (p=0.22) | **0.5068 (p=0.015)** | 0.5010 (p=0.38) | 0.0106 | 6 | `expansion_qualification/qwen15/detectability_v1/detectability_results.json` |
| Pythia | 0.4999 | 0.5002 (p=0.52) | 0.5007 (p=0.39) | 0.5000 (p=0.50) | 0.0101 | 6 | `expansion_qualification/pythia/detectability_v1/detectability_results.json` |
| Falcon3 | 0.5059 | 0.5008 (p=0.38) | 0.5018 (p=0.22) | 0.5003 (p=0.45) | 0.0104 | 6 | `expansion_qualification/falcon/detectability_v1/detectability_results.json` |
| StableLM | 0.4848 | 0.5005 (CI [0.4997, 0.5017]) | 0.5009 (CI [0.5006, 0.5016]) | 0.5000 (CI [0.4996, 0.5004]) | 0.0100 | 6 | `stablelm_qualification/detectability_v1/detectability_results.json` |

**Screen:** AUC ≤ 0.55, TPR@1% ≤ 0.02, clean-control ~0.5. All 12 detector×model cells PASS.

**Qwen LR discrepancy:** AUC 0.5068 (p=0.015) — below its own clean-control (0.5120). Retained as unresolved minor discrepancy.

**StableLM controls:** Clean-vs-clean AUC 0.4848 [0.4444, 0.5106]; permutation p 0.32–0.48.

**Limitations:** Only 6 independent groups (model::tensor); no seed-vs-seed control in StableLM dataset.

**Sources:** 
- `expansion_qualification/smollm2/detectability_v1/detectability_results.json`
- `expansion_qualification/qwen15/detectability_v1/detectability_results.json`
- `expansion_qualification/pythia/detectability_v1/detectability_results.json`
- `expansion_qualification/falcon/detectability_v1/detectability_results.json`
- `stablelm_qualification/detectability_v1/detectability_results.json`

---

## Small-Capacity Supplement (A4.1)

**Experiment:** `detectability_smallcap_v1` — small-capacity tensors (k_proj, v_proj) at 50k bits  
**Models:** TinyLlama, Qwen2.5-3B (2 models × 2 tensors = 4 groups)  
**Blocks touched:** ~95% (vs 0.9–32% on large tensors)  
**Groups:** 4 (leave-one-tensor-out CV)

| Detector | AUC | CI95 (group bootstrap) | TPR@1% | TPR@5% |
|----------|-----|------------------------|--------|--------|
| Logistic Regression | **0.5343** | [0.5332, 0.5363] | 0.0139 | 0.0611 |
| Gradient Boosting | 0.5197 | [0.5162, 0.5236] | 0.0101 | 0.0530 |
| MLP (exploratory) | 0.5053 | [0.5045, 0.5062] | 0.0107 | 0.0518 |

**Permutation test (LR):** p = 0.0050 (200 permutations)  
**Clean-vs-clean control:** AUC 0.4860 [0.4752, 0.4931] — at chance  
**Effect sizes:** max \|Cohen's d\| = 0.0886 (vs 0.016 in main)  
**Limitation:** Only 4 groups (4 tensors) → exploratory, underpowered for definitive claim

**Source:** `out/detectability_smallcap_v1/supplement_results.json`  
**Merged analysis (14 groups):** `out/detectability_smallcap_analysis_v2/detector_results.json` — LR AUC 0.5059 [0.5020, 0.5120], still within screening threshold.

---

## Held-Out Checkpoint Neural Detection (A4.2) — FIXED v3

**Design:** Leave-one-checkpoint-out neural detector (MLP) on 4 local checkpoints  
**Checkpoints:** SmolLM2, Qwen2.5-1.5B, Pythia-1.4b, Falcon3 (StableLM/TinyLlama/Qwen2.5-3B not available locally)  
**Protocol:** Train MLP on 3 checkpoints, test on held-out 1; train-only preprocessing; clean-vs-clean control on held-out.  
**Fixes in v3:** Class-weighted BCE loss (pos_weight), balanced test sets (equal clean/embedded per tensor), bootstrap 95% CI.

| Held-out | Neural MLP AUC | AUC 95% CI | Balanced Acc | n_train | n_test_bal |
|----------|----------------|------------|--------------|---------|------------|
| SmolLM2 | 0.5007 | [0.4932, 0.5084] | 0.5002 | 258,048 | 24,576 |
| Qwen2.5-1.5B | 0.5007 | [0.4934, 0.5087] | 0.5003 | 258,048 | 24,576 |
| Pythia | 0.5000 | [0.4926, 0.5071] | 0.5003 | 258,048 | 24,576 |
| Falcon | 0.5003 | [0.4931, 0.5073] | 0.4997 | 258,048 | 24,576 |

**Mean AUC:** 0.5004 ± 0.0003 — all at chance  
**Clean-vs-clean controls:** AUC ≈ 0.5 (all CIs include 0.5)  
**Interpretation:** No cross-checkpoint generalization; neural detector performs at chance under balanced evaluation.

**Source:** `evidence/heldout_neural_v3/heldout_neural_results.json` + `manifest.json`  
**Previous v2 (defective):** All-positive prediction collapse (TN=0, FN=0, accuracy=85.7%) due to 6:1 class imbalance.

---

## Utility (Gate A5) — Paired PPL Equivalence

### Expansion Models (4 models × 6 conditions = 24 conditions)

| Model | Clean PPL | n_seq | 6/6 PASS? | Max |Δ| | Max CI half-width |
|-------|-----------|-------|-----------|-----|-----|-------------------|
| SmolLM2-1.7B | 12.216089 | 580 | 6/6 | 0.0019% | 0.0033% |
| Qwen2.5-1.5B | 13.078533 | 580 | 6/6 | 0.014% | 0.0115% |
| Pythia-1.4b | 16.430426 | 563* | 6/6 | 0.019% | 0.0191% |
| Falcon3-1B | 12.566386 | 580 | 6/6 | 0.014% | 0.0022% |

*Pythia: 563 sequences (vs 580 target) — tokenizer vocabulary limitation. All 6 CIs within ±0.5%.

**StableLM (reference):** Clean PPL 12.977962, 296,380 tokens/condition, 6/6 PASS, max |Δ| 0.0139%, max CI half-width 0.037% (13× tighter than margin).

**Protocol:** wikitext-2-raw-v1 test, own tokenizer, ctx 512, non-overlapping windows, batch=4, paired 95% CI, ±0.5% margin.  
**Protocol-matched StableLM:** n_seq=580, sizes 10k/50k, seeds ×3, 296,380 tokens/condition.

**Sources:**
- `expansion_qualification/smollm2/utility_v1/comparisons.json`
- `expansion_qualification/qwen15/utility_v1/comparisons.json`
- `expansion_qualification/pythia/utility_v1/comparisons.json`
- `expansion_qualification/falcon/utility_v1/comparisons.json`
- `stablelm_qualification/utility_v2/gate_e_reanalysis.json` (`all_within_margin: true`, `n_within_margin: 6/6`)

**Note:** Previous "BLOCKED" label in CLAIMS_AUDIT.md was incorrect — it conflated Track A residual-method utility (which failed to load) with Track C `interior_nearest` utility (which completed successfully). Track C utility is **DIRECTLY_SUPPORTED**.

---

## Regression Tests (Gate A6) — 12/12 per Model

| Test | Description | SmolLM2 | Qwen2.5 | Pythia | Falcon3 | StableLM |
|------|-------------|---------|---------|--------|---------|----------|
| deterministic_fixed_seed | Identical input → identical output | ✅ | ✅ | ✅ | ✅ | ✅ |
| endpoints_unchanged | Endpoints 0/15 never rewritten | ✅ | ✅ | ✅ | ✅ | ✅ |
| interior_only | All codes stay in 1–14 | ✅ | ✅ | ✅ | ✅ | ✅ |
| carrier_eligibility_stable | Carrier positions stable across requant | ✅ | ✅ | ✅ | ✅ | ✅ |
| fresh_process_extraction | Fresh process extracts artifact | ✅ | ✅ | ✅ | ✅ | ✅ |
| requantize_cycles_0_5_exact | Cycles 0–5 exact recovery | ✅ | ✅ | ✅ | ✅ | ✅ |
| corrupted_carrier_detected | Carrier corruption detected | ✅ | ✅ | ✅ | ✅ | ✅ |
| fixed_offset_corruption_consistent | Fixed-offset corruption behavior | ✅ | ✅ | ✅ | ✅ | ✅ |
| truncated_artifact_handled | Truncation detected | ✅ | ✅ | ✅ | ✅ | ✅ |
| wrong_key_not_exact | Wrong key fails extraction | ✅ | ✅ | ✅ | ✅ | ✅ |
| over_capacity_rejected | Over-capacity payload rejected | ✅ | ✅ | ✅ | ✅ | ✅ |
| pair_parity_fails_after_requant | Baseline fails after requant | ✅ | ✅ | ✅ | ✅ | ✅ |

**Total:** 12/12 PASS per model (4 expansion + StableLM = 60/60)  
**Source:** `test_codec_regressions_local.py` (run on each checkpoint)

---

## Small-Capacity Supplement (A4.1) — Detailed

**Design:** Only small-capacity attention tensors (k_proj, v_proj), 50k bits, 95% blocks touched  
**Models:** TinyLlama, Qwen2.5-3B (2 models × 2 tensors = 4 groups)  
**Expansion:** +4 groups (Pythia L11 dense/down_proj, Falcon L8 k_proj, L17 v_proj) = 8 groups planned, 14 groups in merged analysis

### Original 4-group results (supplement_results.json)

| Detector | AUC | CI95 (group bootstrap) | TPR@1% | TPR@5% | Perm p |
|----------|-----|------------------------|--------|--------|--------|
| Logistic Regression | 0.5343 | [0.5332, 0.5363] | 0.0139 | 0.0611 | 0.0050 |
| Gradient Boosting | 0.5197 | [0.5162, 0.5236] | 0.0101 | 0.0530 | — |
| MLP (exploratory) | 0.5053 | [0.5045, 0.5062] | 0.0107 | 0.0518 | — |

**Permutation test (LR):** 200 permutations, p = 0.0050  
**Clean-vs-clean:** AUC 0.4860 [0.4752, 0.4931] — at chance  
**Max |Cohen's d|:** 0.0886 (vs 0.016 in main) — driven by `mean_abs_value` (negative) and `std_code` (negative)  
**Clean-vs-clean control:** AUC 0.4860 [0.4752, 0.4931] — at chance  

**Limitation:** Only 4 groups → underpowered; labeled **exploratory**

### Merged 14-group analysis (expansion + original)

| Detector | AUC | CI95 (group bootstrap) |
|----------|-----|------------------------|
| Logistic Regression | 0.5059 | [0.5020, 0.5120] |
| Gradient Boosting | 0.5014 | [0.5007, 0.5021] |
| MLP | 0.5011 | [0.5001, 0.5022] |

**Source:** `out/detectability_smallcap_analysis_v2/detector_results.json`

---

## Requantization Robustness Extension

**Current:** Cycles 0–3 in matrix (reported); 0–5 in regression stress test (`requantize_cycles_0_5_exact`)  
**Planned extension:** Cycles 0–10 on one tensor per model (low cost, high value)

**Source:** `test_codec_regressions_local.py` — `requantize_cycles_0_5_exact` test passes on all 4 expansion models + StableLM.

---

## Small-Capacity Expansion (Planned → Executed)

**Added tensors:**
- Pythia: `gpt_neox.layers.11.attention.dense.weight`, `gpt_neox.layers.11.mlp.dense_4h_to_h.weight`
- Falcon: `model.layers.8.self_attn.k_proj.weight`, `model.layers.17.self_attn.v_proj.weight`
- SmolLM2: already had L11 k_proj, L23 v_proj, L11 down_proj, L23 gate_proj (4 tensors)
- Qwen15: already had L13 k_proj, L27 v_proj, L13 down_proj, L27 gate_proj (4 tensors)

**Total groups:** 14 (4 original + 4 Pythia + 4 Falcon + 2 Qwen15 already had / 2 SmolLM2 already had = 14)

**Analysis:** `out/detectability_smallcap_analysis_v2/detector_results.json` — LR AUC 0.5059 [0.5020, 0.5120], GB 0.5014, MLP 0.5011

---

## Frozen Codec Verification

| Item | Expected SHA-256 | Actual | Status |
|------|------------------|--------|--------|
| `code/nes-llm/src/experiments/nf4_anchor_channel.py` | `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` | Verified at every pipeline stage | ✅ |
| `FROZEN_ALL_HASHES_v2.txt` checksum | `d1090a247b91510cb837b29c54eb9c1ef91c48c7f70dced27f71a346fbb4bb16` | Matches computed | ✅ |
| `FROZEN_ALL_HASHES.txt` (original) | Self-hash mismatch (6 narrative docs changed post-manifest) | Documented, not repaired | ⚠️ Documented |

---

## Bibliography Verification Status

| Citation | Status | Notes |
|----------|--------|-------|
| Chen & Wornell 2001 (QIM) | ✅ Verified | IEEE Trans. IT 47(4):1423–1443 |
| Altun et al. 2006 (Set-theoretic QIM) | ✅ Verified | ICASSP 2006 |
| Liu et al. 2016 (NID) | ✅ Verified | J. Ambient Intell. Humaniz. Comput. 7(1):139–147 |
| Xiao et al. 2008 (CNV-QIM) | ✅ Verified | GLOBECOM 2008 |
| Kuribayashi et al. 2020 (DeepWatermark) | ✅ Verified | APSIPA 2020 |
| Qin et al. 2023 (R-QIM) | ✅ Verified | arXiv:2305.17879 |
| Li et al. 2023 (Quantization watermark) | ✅ Verified | EMNLP Findings 2023 |
| Zhang & Koushanfar 2024 (EmMark) | ✅ Verified | DAC 2024, arXiv:2402.17938 |
| Guo et al. 2025 (Invariant watermark) | ✅ Verified | arXiv:2507.08288 |
| Luan et al. 2025 (Permutation + ECC) | ✅ Verified | PoPETs 2025(4) |
| Chen & Ding & Wu & Ye 2025 (FreMark) | ⚠️ Paywalled | DSPP 2025, LNCS 16177, DOI 10.1007/978-981-95-3185-1_14 |
| Li & Chang (HPQIM) | ✅ Verified | Warwick TR / literature |
| Kumaresam et al. 2026 (Defensive pub) | ✅ Verified | TDCommons dpubs_series/10523 |

---

## Frozen Artifacts Checklist

| Artifact | SHA-256 | Status |
|----------|---------|--------|
| `code/nes-llm/src/experiments/nf4_anchor_channel.py` | `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` | ✅ Verified |
| `FROZEN_ALL_HASHES.txt` | `625de8f8867b...` (current) vs `4d379eb0...` (recorded) | ⚠️ Self-hash mismatch (6 narrative docs) |
| `FROZEN_ALL_HASHES_v2.txt` | `d1090a247b91...` | ✅ Matches |
| `FROZEN_ALL_HASHES_v2.txt.sha256` | `d1090a247b91510cb837b29c54eb9c1ef91c48c7f70dced27f71a346fbb4bb16` | ✅ Verified |
| `PRIOR_ART_BIBLIOGRAPHY.bib` | — | 3 author lists corrected (P3) |

---

## Model Revision Verification

| Model | Revision | Verified | Notes |
|-------|----------|----------|-------|
| StableLM-2-1.6B | `f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef` | ✅ Pinned in `download_ckpt.py` |
| SmolLM2-1.7B | `effd688a12921b4cc83e3312b6feb579f70f9c71` | ✅ |
| Qwen2.5-1.5B | `8faed761d45a263340a0528343f099c05c9a4323` | ✅ |
| Pythia-1.4b | `fedc38a16eea3bd36a96b906d78d11d2ce18ed79` | ✅ |
| Falcon3-1B | `cb37ef3559b157b5c9d9226296ba01a5162da1f7` | ✅ |
| Qwen2.5-3B (historical) | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | ⚠️ **UNVERIFIED** — current `main` |
| TinyLlama-1.1B | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` | ✅ |

---

## Disk Usage Summary

| Directory | Size | Contents |
|-----------|------|----------|
| `expansion_qualification/` | ~20 MB | Audit tables, detectability, utility, integrity |
| `stablelm_qualification/` | ~15 MB | Roundtrip, detectability, utility, regression |
| `out/detectability_*` | ~2 GB | Raw features, detector results |
| `cache/` | ~50 GB | Model checkpoints, residual caches |
| `results/` | ~50 MB | exp1–exp27 JSON results |