# CLAIMS_AUDIT.md — Track C Claims Classification

**Status:** Complete — every proposed claim classified  **Date:** 2026-10-21  **Commit:** `5b40120` (`research/contract-b-b14-and-evaluation`)  **Frozen codec:** `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764`

---

## Classification Legend

| Label | Definition |
|-------|------------|
| **DIRECTLY_SUPPORTED** | Evidence exists in frozen artifacts; numbers match source files exactly |
| **QUALIFIED** | Supported but with explicit caveats (sample size, scope, exploratory label) |
| **UNVERIFIED** | Claimed in reports but source evidence missing, incomplete, or lost |
| **PROHIBITED** | Contradicted by evidence or prior art; must not appear in manuscript |

---

## 1. Core Technical Claims

| # | Claim | Classification | Evidence Source | Notes |
|---|-------|----------------|-----------------|-------|
| C1 | `interior_nearest` achieves exact payload recovery (BER=0) after NF4 requantization cycles 0–3 on all 5 models | **DIRECTLY_SUPPORTED** | `expansion_audit_table.csv` (4320 rows, all `interior_nearest` rows: `exact_recovery=1`, `changed_anchor_codes=0`) | 360/360 per model; 1800/1800 total |
| C2 | `interior_nearest` achieves exact payload recovery (BER=0) after NF4 requantization cycles 0–5 (stress test) | **DIRECTLY_SUPPORTED** | `test_codec_regressions.py` — `requantize_cycles_0_5_exact` test PASS on all 5 models | Regression test, not full matrix |
| C3 | Endpoint codes (0, 15) are never modified during embedding or requantization | **DIRECTLY_SUPPORTED** | `expansion_audit_table.csv` — `changed_anchor_codes=0` for all `interior_nearest` rows; integrity checks `endpoints_unchanged` PASS | Core invariant |
| C4 | Carrier positions are stable across requantization cycles | **DIRECTLY_SUPPORTED** | Regression test `carrier_eligibility_stable` PASS on all 5 models; artifact SHA-256 identical across cycles 0–3 | Verified by byte-idempotence |
| C5 | Baseline `pair_parity` fails after requantization (cycle 1+) | **DIRECTLY_SUPPORTED** | `expansion_audit_table.csv` — `pair_parity` exact_recovery drops to 90–108/360; regression test `pair_parity_fails_after_requant` PASS | Contrast baseline |
| C6 | Baseline `nearest_parity` fails after requantization | **DIRECTLY_SUPPORTED** | `expansion_audit_table.csv` — `nearest_parity` exact_recovery 90–108/360 | Contrast baseline |

---

## 2. Detectability Claims

| # | Claim | Classification | Evidence Source | Notes |
|---|-------|----------------|-----------------|-------|
| D1 | Main detectability: all 5 models PASS screening (AUC ≤ 0.55, TPR@1% ≤ 0.02, clean-control ~0.5) | **DIRECTLY_SUPPORTED** | 5 × `detectability_results.json` — max GB AUC 0.5025 (Qwen), max LR AUC 0.5068 (Qwen), max MLP AUC 0.5010 | Qwen LR nominal p=0.015 but < clean control |
| D2 | Qwen2.5-1.5B LR AUC 0.5068 (p=0.015) is an unresolved minor discrepancy | **DIRECTLY_SUPPORTED** | `expansion_qualification/qwen15/detectability_v1/detectability_results.json` | AUC < clean-control (0.5120) |
| D3 | Small-capacity supplement (4 groups): LR AUC 0.5343, permutation p=0.005 | **QUALIFIED** | `out/detectability_smallcap_v1/supplement_results.json` | Only 4 groups → exploratory, underpowered |
| D4 | Merged small-capacity (14 groups): LR AUC 0.5059 [0.5020, 0.5120], within screening | **DIRECTLY_SUPPORTED** | `out/detectability_smallcap_analysis_v2/detector_results.json` | 14 groups = 4 original + 4 Pythia + 4 Falcon + 2 Qwen15 |
| D5 | Held-out checkpoint neural detector: mean AUC 0.4998 ± 0.0002 (all at chance) | **UNVERIFIED** | Results in `/tmp/nes_local_run/out/detectability_heldout_neural/` — **lost in /tmp cleanup** | Must re-run before submission |
| D6 | Clean-vs-clean controls at chance (AUC ~0.5) | **DIRECTLY_SUPPORTED** | All `detectability_results.json` + `supplement_results.json` (0.4860) | Required for valid screening |
| D7 | Embedded seed-vs-seed control at chance | **UNVERIFIED** | Not run on StableLM; not run on expansion models | Gap in controls |

---

## 3. Utility Claims

| # | Claim | Classification | Evidence Source | Notes |
|---|-------|----------------|-----------------|-------|
| U1 | Expansion models (4): all 24 conditions (4 models × 2 sizes × 3 seeds) have paired 95% CI within ±0.5% PPL | **DIRECTLY_SUPPORTED** | 4 × `comparisons.json` — all `"within_margin_0p5pct": true` | 6/6 PASS per model |
| U2 | StableLM (protocol-matched): 6/6 conditions within ±0.5% PPL, 296,380 tokens/condition | **DIRECTLY_SUPPORTED** | `stablelm_qualification/utility_v2/gate_e_reanalysis.json` — `all_within_margin: true`, `n_within_margin: 6/6` | 13× tighter CI than margin |
| U3 | Pythia used 563 sequences (vs 580 target) due to tokenizer vocabulary | **DIRECTLY_SUPPORTED** | `expansion_qualification/pythia/utility_v1/comparisons.json` — `"n_sequences": 563` | All 6 CIs still within margin |
| U4 | Utility equivalence holds for generation quality / downstream tasks | **PROHIBITED** | Not tested | Only wikitext-2 PPL measured |
| U5 | Zero utility impact (absolute) | **PROHIBITED** | ±0.5% is non-zero; CI includes non-zero values | "Within margin" ≠ "zero impact" |

---

## 4. Integrity & Reproducibility Claims

| # | Claim | Classification | Evidence Source | Notes |
|---|-------|----------------|-----------------|-------|
| I1 | Only the intended tensor differs between clean and stego model | **DIRECTLY_SUPPORTED** | 5 × `integrity.json` — `differing_keys == [target_tensor]` | Per-model verification |
| I2 | Model shape preserved, finite logits/loss, payload exactly recoverable in fresh process | **DIRECTLY_SUPPORTED** | 5 × `integrity.json` — all four flags true | Fresh process extraction |
| I3 | All 60 regression tests (12 per model × 5 models) PASS | **DIRECTLY_SUPPORTED** | `test_codec_regressions.py` output per model | 12 tests × 5 models |
| I4 | Frozen codec SHA-256 unchanged throughout all experiments | **DIRECTLY_SUPPORTED** | Verified at every pipeline stage; `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` | Hard constraint |
| I5 | v2 manifest checksum externally verified | **DIRECTLY_SUPPORTED** | `FROZEN_ALL_HASHES_v2.txt.sha256` matches computed hash | `d1090a247b91510cb837b29c54eb9c1ef91c48c7f70dced27f71a346fbb4bb16` |
| I6 | Original manifest `FROZEN_ALL_HASHES.txt` is internally consistent | **PROHIBITED** | Self-hash mismatch; 6 narrative docs changed post-manifest | Documented, not repaired |

---

## 5. Novelty / Prior Art Claims (Gate G)

| # | Claim | Classification | Evidence Source | Notes |
|---|-------|----------------|-----------------|-------|
| N1 | Nearest-same-parity code-index embedding is NID (Liu et al. 2016) / CNV-QIM (Xiao et al. 2008) | **DIRECTLY_SUPPORTED** | `overnight_third_model_prior_art/PRIOR_ART_BIBLIOGRAPHY.bib` — verified primary sources | Confirmed overlap |
| N2 | Invariant-region framing for quantized LLM weights: Li et al. 2023 (EMNLP Findings) | **DIRECTLY_SUPPORTED** | `PRIOR_ART_BIBLIOGRAPHY.bib` — verified | Confirmed overlap |
| N3 | FreMark 2025 (LNCS 16177) proposes "free-region marking" — potential overlap | **UNVERIFIED** | Full text paywalled (DOI 10.1007/978-981-95-3185-1_14); only abstract accessible | Cannot verify extent of overlap |
| N4 | HPQIM (Li & Chang) — invariant embedding, high payload | **DIRECTLY_SUPPORTED** | Warwick TR / literature — verified | Confirmed overlap |
| N5 | This work is the first to embed payloads in quantized LLMs | **PROHIBITED** | NID 2016, CNV-QIM 2008, Li 2023, EmMark 2024, FreMark 2025 | False claim |
| N6 | This work introduces a new QIM variant / embedding principle | **PROHIBITED** | Nearest-same-parity is NID/CNV-QIM | False claim |
| N7 | Contribution: NF4 packed-code formulation + endpoint exclusion protocol + repeated-requantization exactness demonstration | **DIRECTLY_SUPPORTED** | Method (METHODS.md) + Evidence (RESULTS_EVIDENCE_TABLE.md) | Narrow, accurate claim |
| N8 | Gate G = PARTIAL (novelty risk unresolved) | **DIRECTLY_SUPPORTED** | `EXPANSION_QUALIFICATION_REPORT_V1.md` §0.1, `STABLELM_QUALIFICATION_REPORT.md` | Documented in both reports |

---

## 6. Robustness Claims

| # | Claim | Classification | Evidence Source | Notes |
|---|-------|----------------|-----------------|-------|
| R1 | Robust to requantization cycles 0–5 (exact recovery) | **DIRECTLY_SUPPORTED** | Regression test `requantize_cycles_0_5_exact` PASS on all 5 models | Stress test, not full matrix |
| R2 | Robust to carrier corruption (detected via checksum) | **DIRECTLY_SUPPORTED** | Regression test `corrupted_carrier_detected` PASS | Single-bit flip detected |
| R3 | Robust to truncation (detected) | **DIRECTLY_SUPPORTED** | Regression test `truncated_artifact_handled` PASS | Checksum fails |
| R4 | Wrong key fails extraction (not exact) | **DIRECTLY_SUPPORTED** | Regression test `wrong_key_not_exact` PASS | Key-dependent recovery |
| R5 | Over-capacity payload rejected | **DIRECTLY_SUPPORTED** | Regression test `over_capacity_rejected` PASS | Capacity bound enforced |
| R6 | Robust to fine-tuning / pruning / distillation / quantization-aware training | **PROHIBITED** | Not tested | Out of scope |
| R7 | Robust to adversarial fine-tuning / weight averaging | **PROHIBITED** | Not tested | Out of scope |

---

## 7. Scope & Generality Claims

| # | Claim | Classification | Evidence Source | Notes |
|---|-------|----------------|-----------------|-------|
| S1 | Evaluated on 5 model architectures (Llama×4, GPT-NeoX×1) | **DIRECTLY_SUPPORTED** | Model list in METHODS.md; all 5 have full gate results | 1.4B–1.7B range |
| S2 | Results generalize to 7B, 70B, 400B models | **PROHIBITED** | Not tested | Extrapolation prohibited |
| S3 | Results generalize to FP4, INT4, AWQ, GPTQ, HQQ, SpQR, QuIP# | **PROHIBITED** | Only NF4 (bitsandbytes) tested | Quantization-specific |
| S4 | Results generalize to all tensors in model | **PROHIBITED** | Only 6 tensors/model tested (depth × projection) | 99.4% untested |
| S5 | Results generalize to payload sizes >50k bits | **PROHIBITED** | Max tested 50k bits (1.2% carrier utilization) | Capacity regime unknown |
| S6 | Works with any NF4-quantized model using bitsandbytes | **QUALIFIED** | Tested on 5; protocol assumes Llama/GPT-NeoX naming | Naming convention dependency |

---

## 8. Security / Threat Model Claims

| # | Claim | Classification | Evidence Source | Notes |
|---|-------|----------------|-----------------|-------|
| T1 | Passive detector (known algorithm, unknown key) cannot detect embedding at >chance | **QUALIFIED** | Detectability results support under our threat model | Only 3 detectors, 16 features, 5 models |
| T2 | Adaptive attacker with full model access cannot detect | **PROHIBITED** | Not tested | Stronger threat model |
| T3 | Key-dependent recovery: wrong key → extraction fails | **DIRECTLY_SUPPORTED** | Regression test `wrong_key_not_exact` PASS | Verified |
| T4 | Carrier positions are secret | **PROHIBITED** | Carrier positions deterministic from (key, tensor, bit index) — known algorithm | Security by payload encryption only |
| T5 | Embedding is invisible to model owner | **PROHIBITED** | Model owner knows algorithm; can run detector | Only hidden from passive observer |

---

## 9. Summary: Claims Safe for Manuscript

### ✅ DIRECTLY_SUPPORTED (Safe to state as findings)
- C1–C6: Core roundtrip exactness, endpoint preservation, carrier stability, baseline failures
- D1, D2, D4, D6: Detectability screening PASS, Qwen discrepancy documented, merged small-cap within threshold, clean controls at chance
- U1–U3: Utility ±0.5% PASS on all conditions, StableLM tighter CI, Pythia 563 sequences noted
- I1–I5: Integrity, regression tests, frozen codec, v2 manifest
- N1, N2, N4, N7, N8: Prior art overlaps confirmed, narrow contribution stated, Gate G = PARTIAL
- R1–R5: Requantization, corruption, truncation, wrong key, over-capacity
- S1: 5 models evaluated

### ⚠️ QUALIFIED (Safe with explicit caveats)
- D3: Small-cap 4-group LR AUC 0.534 (p=0.005) — **label exploratory, underpowered**
- T1: Detectability at chance — **only under our threat model (3 detectors, 16 features, 5 models)**
- S6: NF4 bitsandbytes generality — **naming convention dependency noted**

### ❌ UNVERIFIED (Do not state as findings; re-run or remove)
- D5: Held-out neural detector results lost
- D7: Seed-vs-seed control not run

### 🚫 PROHIBITED (Must not appear)
- N5, N6: Novelty overclaims
- U4, U5: Utility overclaims
- R6, R7: Untested robustness
- S2–S5: Generalization overclaims
- T2, T4, T5: Security overclaims
- I6: Original manifest consistency

---

## 10. Required Corrections to Existing Reports

| Report | Section | Current Text | Corrected Text |
|--------|---------|--------------|----------------|
| `EXPANSION_QUALIFICATION_REPORT_V1.md` | §0.1 | "raw artifacts re-verify clean" | "raw artifacts re-verify clean; 6 narrative docs changed post-manifest (self-hash mismatch documented)" |
| `STABLELM_QUALIFICATION_REPORT.md` | Gate G | "PARTIAL / UNRESOLVED RISK" | "PARTIAL — NID 2016 / CNV-QIM 2008 overlap confirmed; FreMark 2025 paywalled" |
| `FINAL_SPRINT_REPORT.md` | Track C status | "QUALIFIED ✅" | "QUALIFIED ✅ (Gate G = PARTIAL, small-cap exploratory, held-out neural to re-run)" |
| All reports | "first / novel / universal" | Any occurrence | Remove or replace with qualified claim N7 |

---

## 11. Final Audit Checklist

- [x] Every number in RESULTS_EVIDENCE_TABLE.md traceable to source file
- [x] Every claim in CLAIMS_AUDIT.md classified
- [x] No PROHIBITED claims in MANUSCRIPT_OUTLINE.md
- [x] QUALIFIED claims have explicit caveats in LIMITATIONS_AND_THREATS.md
- [x] UNVERIFIED claims flagged for re-run (held-out neural)
- [x] Gate G = PARTIAL documented with NID/CNV-QIM overlap and FreMark paywall
- [x] Qwen LR discrepancy documented as unresolved minor discrepancy
- [x] Pythia 563 vs 580 sequences documented
- [x] Qwen2.5-3B revision UNVERIFIED documented
- [x] Frozen manifest self-hash mismatch documented
- [x] No commits, pushes, merges performed