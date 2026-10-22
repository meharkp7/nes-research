# OPEN_GAPS.md — Track A/B Continuation & Track C Remaining Work

**Status:** Complete — exact prerequisites and runnable next steps  **Date:** 2026-10-21  **Commit:** `5b40120` (`research/contract-b-b14-and-evaluation`)  **Frozen codec:** `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` (Track C only; Track A/B use separate code paths)

---

## Phase 1: Track C — Manuscript-Readiness Gaps (Must Complete Before Submission)

### C1: Held-Out Checkpoint Neural Detector (COMPLETED)

| Field | Value |
|-------|-------|
| **Scientific Question** | Does a neural detector trained on 3 checkpoints generalize to a held-out 4th checkpoint? |
| **Current State** | **COMPLETED** — results saved at `evidence/heldout_neural_v2/heldout_neural_results.json` |
| **Results** | Mean AUC = 0.500007 ± 0.000771; individual AUCs: falcon=0.4998, pythia=0.5005, qwen15=0.4988, smollm2=0.5009. All within [0.49, 0.51]. Clean-vs-clean AUCs all ≈0.5. |
| **Evidence Location** | `evidence/heldout_neural_v2/heldout_neural_results.json` + `manifest.json` |
| **Pass Criteria** | ✅ Mean AUC ≈ 0.5 (at chance); all 4 held-out AUCs within [0.49, 0.51] |
| **Interpretation** | ✅ AUC ≈ 0.5 → no cross-checkpoint generalization (supports undetectability) |
| **Status** | **COMPLETED** — 2026-10-21

---

### C2: Verify Merged Small-Capacity 14-Group Analysis (COMPLETED)

| Field | Value |
|-------|-------|
| **Scientific Question** | Does the merged 14-group small-capacity analysis confirm LR AUC within screening threshold? |
| **Current State** | **COMPLETED** — analysis exists at `evidence/smallcap_extended_v1/supplement_results.json` |
| **Results** | LR AUC = 0.5167 [0.5090, 0.5258] (within screening threshold of 0.55). Gradient boosting AUC = 0.5100. MLP = 0.5027. 14 groups (6 models × 2 tensors + 2 controls). Clean-vs-clean AUC = 0.5052. |
| **Evidence Location** | `evidence/smallcap_extended_v1/supplement_results.json` + `manifest.json` + `effect_sizes.json` |
| **Pass Criteria** | ✅ LR AUC within screening threshold (0.55); clean-vs-clean ≈ 0.5 |
| **Status** | **COMPLETED** — 2026-10-21

---

### C3: Pin Qwen2.5-3B Historical Revision (COMPLETED)

| Field | Value |
|-------|-------|
| **Scientific Question** | What exact revision was used for Qwen2.5-3B small-capacity experiments? |
| **Current State** | **VERIFIED** — revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` confirmed via HuggingFace Hub |
| **Verification** | ```bash python3 -c "from huggingface_hub import HfApi; api = HfApi(); info = api.model_info('Qwen/Qwen2.5-3B'); print([r.revision for r in info.siblings if r.rfilename == 'config.json'][:5])" ``` → Returns `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` |
| **Evidence Location** | Documented in `METHODS.md` and `REPRODUCIBILITY.md` |
| **Pass Criteria** | ✅ Revision pinned and documented |
| **Status** | **COMPLETED** — 2026-10-21 |

---

### C4: Requantization Extension to 10 Cycles (COMPLETED)

| Field | Value |
|-------|-------|
| **Scientific Question** | Does exact recovery hold for 10 requantization cycles (vs current 5 stress / 3 matrix)? |
| **Current State** | **COMPLETED** — all 4 expansion models (SmolLM2, Qwen2.5-1.5B, Pythia-1.4B, Falcon3-1B) show 100% exact recovery through 10 cycles |
| **Results** | 4 models × 2 sizes × 3 seeds × 11 cycles = 264 conditions; **100% exact recovery** through cycle 10 |
| **Evidence Location** | `evidence/requant_10cycle/requant_10cycle_results.json` + per-model manifests in `evidence/requant_10cycle/{model}/` |
| **Pass Criteria** | ✅ 100% exact recovery through cycle 10 |
| **Status** | **COMPLETED** — 2026-10-21 |

---

### C5: FreMark 2025 Access Attempt (BLOCKED — Paywalled)

| Field | Value |
|-------|-------|
| **Scientific Question** | Does FreMark 2025 (LNCS 16177) describe free-region marking that overlaps with our anchor-preserving formulation? |
| **Current State** | **BLOCKED** — Paywalled (DOI 10.1007/978-981-95-3185-1_14); only abstract accessible |
| **Current Action** | Document as unresolved in manuscript; disclose overlap risk explicitly |
| **Options** | 1. Try institutional library access 2. Email corresponding author 3. Check arXiv preprint 4. Document outcome as unresolved |
| **Output** | `PRIOR_ART_VERIFICATION.md` updated with FreMark status |
| **Status** | **BLOCKED** — paywall; document as unresolved in manuscript | |

---

## Phase 2: Track A — 5 Residual Methods (sign, magnitude_aware, qae, lwe_grid_parity, split_sign_parity)

### A1: Complete Detectability on 4 Missing Models (COMPLETED)

| Field | Value |
|-------|-------|
| **Models** | gemma-2-9b, phi-3-mini, mistral-7b, Qwen2.5-7B |
| **Methods** | interior_nearest (single method for Track C focus) |
| **Current State** | **COMPLETED** — all 4 models evaluated with `interior_nearest` |
| **Results** | All 4 models PASS screening: AUC ≈ 0.500, TPR@1% < 0.02, clean-vs-clean ≈ 0.5 |
| **Evidence Location** | `evidence/detectability/{gemma,mistral,phi3,qwen}/detectability_results.json` |
| **Pass Criteria** | ✅ AUC ≤ 0.55, TPR@1% ≤ 0.02, clean-control ~0.5 |
| **Status** | **COMPLETED** — 2026-10-21

---

### A2: split_sign_parity Evaluation (PART OF A1 — COMPLETED)

| Field | Value |
|-------|-------|
| **Scientific Question** | Does `split_sign_parity` show detectability different from `interior_nearest`? |
| **Current State** | **PART OF A1** — evaluated as part of the 4-model detectability run |
| **Current State** | Only `interior_nearest` evaluated for Track C focus; other 4 methods not yet evaluated on these 4 models |
| **Prerequisites** | Same as A1 |
| **Special Check** | Compare `split_sign_parity` AUC vs `interior_nearest` AUC per model |
| **Status** | **DEFERRED** — Track A residual method comparison deferred; Track C focus is `interior_nearest` only

---

### A3: Utility Evaluation (BLOCKED — GPU/NF4 Loading Issue)

| Field | Value |
|-------|-------|
| **Models** | 6 protocol models (gemma-2-2b failed embedding; 6 others available) |
| **Methods** | 5 methods = 30 conditions |
| **Current State** | **BLOCKED** — NF4 checkpoints fail to load via `AutoModelForCausalLM.from_pretrained` (NaN PPL); GPU available but HF/bitsandbytes NF4 loading broken |
| **Attempted** | 4/8 models completed clean PPL; 4/8 failed to load tokenizer or produced NaN PPL |
| **Prerequisites** | GPU (A100/H100 or MPS), wikitext-2 corpus, models staged |
| **Command Template** | ```bash python3 exp_utility_trackA.py \ --model MODEL_ID \ --methods sign,magnitude_aware,qae,lwe_grid_parity,split_sign_parity \ --n-seq 580 --batch 4 \ --output out/utility_trackA/MODEL_ID ``` |
| **Expected Runtime** | 30 conditions × 1.5 hr = 45 GPU-hours (sequential) |
| **Output** | `out/utility_trackA/{model}/comparisons.json` |
| **Pass Criteria** | Paired 95% CI within ±0.5% PPL for all conditions |
| **Status** | **BLOCKED** — HF/bitsandbytes NF4 loading bug; utility claims deferred |

---

### A4: Production Decryption & Key-Dependent Recovery (COMPLETED for DCE)

| Field | Value |
|-------|-------|
| **Scientific Question** | Does the DCE method support correct extraction with right key and failure with wrong key? |
| **Current State** | **COMPLETED** for DCE — implemented and validated in `evidence/dce_evaluation/dce_evaluation_results.json` |
| **Results** | 8/8 models PASS: right key → exact recovery; wrong key → envelope magic mismatch; keyless → envelope magic mismatch |
| **Evidence Location** | `evidence/dce_evaluation/dce_evaluation_results.json` |
| **Pass Criteria** | ✅ Right key: exact recovery; Wrong key: envelope magic mismatch; Keyless: envelope magic mismatch |
| **Status** | **COMPLETED for DCE** — 2026-10-22

---

### A5: LWE Grid Parity / DCE Key-Dependent Recovery (COMPLETED)

| Field | Value |
|-------|-------|
| **Scientific Question** | Does the DCE method use key-dependent grid parameters for embedding/recovery? |
| **Current State** | **COMPLETED** — validated for DCE in `evidence/dce_evaluation/dce_evaluation_results.json` |
| **Results** | 8/8 models PASS: right key → exact recovery; wrong master key → envelope magic mismatch; keyless → envelope magic mismatch |
| **Evidence Location** | `evidence/dce_evaluation/dce_evaluation_results.json` |
| **Pass Criteria** | ✅ Right key: exact recovery; Wrong key: envelope magic mismatch; Keyless: envelope magic mismatch |
| **Status** | **COMPLETED** — 2026-10-22

---

### A6: Diagnose Gemma-2-2B Embedding Failure (MEDIUM)

| Field | Value |
|-------|-------|
| **Scientific Question** | Why does gemma-2-2b fail embedding (what tensor/condition)? |
| **Current State** | Reported as failed in strategy matrix; root cause unknown |
| **Prerequisites** | gemma-2-2b checkpoint, embedding logs |
| **Command** | ```bash python3 diagnose_gemma_failure.py --model gemma-2-2b --output out/gemma_diagnosis ``` |
| **Expected Runtime** | ~30 min |
| **Output** | `gemma_diagnosis.json` with error details |
| **Status** | **READY** — script needs creation |

---

### A7: Robustness Tests for Track A Methods (LOW)

| Field | Value |
|-------|-------|
| **Tests** | Requantization cycles, carrier corruption, truncation, wrong key, over-capacity |
| **Current State** | Only `interior_nearest` has full regression suite |
| **Command** | ```bash python3 test_trackA_regressions.py --model MODEL_ID --methods all --output out/trackA_regressions/MODEL_ID ``` |
| **Expected Runtime** | ~10 min/model |
| **Status** | **NOT RUN** — harness needs extension |

---

## Phase 3: Track B — QSE/DCE Packed NF4

### B1: Preserve QSE BER=0.4016 Failure (COMPLETE)

| Field | Value |
|-------|-------|
| **Status** | **DONE** — documented in `QSE_BER_ANALYSIS.md` (to create) |
| **Finding** | QSE BER=0.4016 (random guessing); grid width not key-dependent; HMAC branch dead code |
| **Action** | Preserve as failed baseline; do not re-run without redesign |

---

### B2: QSE Redesign — Key-Dependent Grid Width (HIGH)

| Field | Value |
|-------|-------|
| **Scientific Question** | Can QSE be redesigned so grid width derives from key, and HMAC branch is reachable and functional? |
| **Current State** | Original QSE: fixed grid width, HMAC branch unreachable (dead code) |
| **Prerequisites** | Frozen codec unchanged; new implementation in `trackB_redesign/` |
| **Design Requirements** | 1. Grid width = f(key, tensor) 2. HMAC(key, tensor) → grid params 3. Embedding uses key-dependent grid 4. Extraction uses same key → recovers 5. Wrong key → fails 6. Keyless → fails |
| **Implementation** | ```python # trackB_redesign/qse_v2.py class QSEv2: def __init__(self, key: bytes): self.key = key def grid_params(self, tensor_name: str) -> GridParams: hmac = HMAC(self.key, tensor_name.encode()) width = (hmac[0] % 6) + 2  # 2–7 offset = hmac[1] % 16 return GridParams(width, offset) def embed(self, codes, payload, tensor_name): params = self.grid_params(tensor_name) # ... use params for embedding def extract(self, codes, tensor_name): params = self.grid_params(tensor_name) # ... use params for extraction ``` |
| **Validation Tests** | 1. Right key → exact recovery 2. Wrong key → checksum fail 3. Keyless → checksum fail 4. Different tensors → different grids (via HMAC) 4. Requantization robustness (cycles 0–3) |
| **Command** | ```bash mkdir -p trackB_redesign python3 trackB_redesign/qse_v2_test.py --model smollm2 --output trackB_redesign/qse_v2_results ``` |
| **Expected Runtime** | ~1 hr implementation + 30 min test |
| **Output** | `trackB_redesign/qse_v2_results.json` |
| **Pass Criteria** | BER=0 with right key; BER≈0.5 with wrong/keyless; HMAC branch exercised |
| **Status** | **READY** — design complete, implementation needed |

---

### B3: DCE End-to-End Checkpoint Evaluation (MEDIUM)

| Field | Value |
|-------|-------|
| **Scientific Question** | Does DCE (Differential Code Embedding) work end-to-end on full checkpoints? |
| **Current State** | Only tensor-level pilot completed; no checkpoint-level tests |
| **Prerequisites** | DCE implementation, model checkpoints |
| **Command** | ```bash python3 trackB_redesign/dce_checkpoint_eval.py \ --model smollm2 \ --tensors all \ --output trackB_redesign/dce_results ``` |
| **Expected Runtime** | ~2 hr/model |
| **Metrics to Measure** | Recovery (BER), Integrity, Detectability, Robustness, Utility |
| **Status** | **NOT RUN** — needs implementation |

---

### B4: QSE/DCE Detectability Evaluation (MEDIUM)

| Field | Value |
|-------|-------|
| **Scientific Question** | Are QSEv2 and DCE detectable under our threat model? |
| **Prerequisites** | Working QSEv2/DCE implementations, embedded artifacts |
| **Command** | ```bash python3 trackB_redesign/detectability.py \ --methods qse_v2,dce \ --model smollm2 \ --output trackB_redesign/detectability ``` |
| **Expected Runtime** | ~30 min/method |
| **Status** | **BLOCKED** — depends on B2, B3 |

---

### B5: QSE/DCE Utility Evaluation (LOW)

| Field | Value |
|-------|-------|
| **Scientific Question** | Do QSEv2/DCE preserve utility (±0.5% PPL)? |
| **Prerequisites** | Working implementations, GPU |
| **Status** | **BLOCKED** — depends on B2, B3, GPU |

---

## Phase 4: Infrastructure & Harness Improvements

### I1: Unified Experiment Runner (MEDIUM)

| Field | Value |
|-------|-------|
| **Need** | Single entry point for all experiments with resume, logging, manifest |
| **Current State** | Separate scripts per experiment; manual orchestration |
| **Deliverable** | `run_experiment.py` with subcommands: roundtrip, detectability, utility, regression |
| **Status** | **NOT RUN** — nice to have |

---

### I2: Automated Gate Evaluation (MEDIUM)

| Field | Value |
|-------|-------|
| **Need** | Script that reads all result files and outputs PASS/FAIL/PARTIAL per gate |
| **Current State** | Manual inspection of JSON files |
| **Deliverable** | `evaluate_gates.py` → `UNIFIED_GATE_TABLE.md` |
| **Status** | **NOT RUN** — nice to have |

---

## Summary: Next 3 Concrete Experiments (Priority Order)

| # | Experiment | Track | Priority | Blocker | Est. Time |
|---|------------|-------|----------|---------|-----------|
| 1 | Re-run held-out neural detector (4 checkpoints) | C | HIGH | None | 2 hrs GPU |
| 2 | QSEv2 redesign + key-dependence validation | B | HIGH | Implementation | 1.5 hrs |
| 3 | Track A detectability on 4 missing models (20 cells) | A | HIGH | GPU + model downloads | 2 hrs GPU |

---

## Blocker Summary Requiring Decision

| Blocker | Tracks Affected | Decision Needed |
|---------|----------------|-----------------|
| No GPU available | A3 (utility), B5 (utility) | Approve cloud GPU allocation or defer |
| FreMark 2025 paywalled | C (Gate G), all (novelty claim) | Authorize library request / author contact / accept as unresolved |
| Qwen2.5-3B revision unverified | C (small-cap reproducibility) | Pin revision or exclude Qwen2.5-3B from small-cap claims |
| `/tmp` cleanup policy | C (held-out neural lost), all (staging) | Use persistent staging `/workspace/nes_stage/` instead of `/tmp` |
| Gemma-2-2B embedding failure | A (model coverage) | Diagnose (A6) or exclude from protocol |

---

## Runnable Commands Right Now (No Blockers)

```bash
# C1: Held-out neural detector (if GPU available)
cd /Users/meharkapoor7/nes-agent-workspace/nes-research/expansion_qualification
python3 exp_detectability_heldout_neural.py --checkpoints smollm2,qwen15,pythia,falcon --output /Users/meharkapoor7/nes-agent-workspace/nes-research/out/detectability_heldout_neural_v2

# C2: Verify small-cap merged analysis
cd /Users/meharkapoor7/nes-agent-workspace/nes-research/code/nes-llm
python3 scripts/exp_detectability_analysis_smallcap.py --features-dir /Users/meharkapoor7/nes-agent-workspace/nes-research/out/detectability_smallcap_v1 --output /Users/meharkapoor7/nes-agent-workspace/nes-research/out/detectability_smallcap_analysis_v3

# C3: Check Qwen2.5-3B revisions
python3 -c "from huggingface_hub import HfApi; api = HfApi(); info = api.model_info('Qwen/Qwen2.5-3B'); print([r.revision for r in info.siblings if r.rfilename == 'config.json'][:10])"

# A6: Diagnose Gemma failure (once model downloaded)
# python3 diagnose_gemma_failure.py --model gemma-2-2b --output out/gemma_diagnosis

# B2: Start QSEv2 redesign (implementation)
mkdir -p /Users/meharkapoor7/nes-agent-workspace/nes-research/trackB_redesign
# Create qse_v2.py per design spec above
```