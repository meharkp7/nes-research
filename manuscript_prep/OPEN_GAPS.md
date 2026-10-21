# OPEN_GAPS.md — Track A/B Continuation & Track C Remaining Work

**Status:** Complete — exact prerequisites and runnable next steps  **Date:** 2026-10-21  **Commit:** `5b40120` (`research/contract-b-b14-and-evaluation`)  **Frozen codec:** `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` (Track C only; Track A/B use separate code paths)

---

## Phase 1: Track C — Manuscript-Readiness Gaps (Must Complete Before Submission)

### C1: Re-run Held-Out Checkpoint Neural Detector (HIGH)

| Field | Value |
|-------|-------|
| **Scientific Question** | Does a neural detector trained on 3 checkpoints generalize to a held-out 4th checkpoint? |
| **Current State** | Results existed in `/tmp/nes_local_run/out/detectability_heldout_neural/` but `/tmp` was cleaned; results lost |
| **Prerequisites** | Local checkpoints for SmolLM2, Qwen2.5-1.5B, Pythia-1.4b, Falcon3-1B (all available in `~/cache/models/`) |
| **Command** | ```bash cd /Users/meharkapoor7/nes-agent-workspace/nes-research/expansion_qualification python3 exp_detectability_heldout_neural.py \ --checkpoints smollm2,qwen15,pythia,falcon \ --output /Users/meharkapoor7/nes-agent-workspace/nes-research/out/detectability_heldout_neural_v2 ``` |
| **Expected Runtime** | ~2 hours (GPU), ~8 hours (CPU) |
| **Resources** | GPU preferred (MLP training); 4 checkpoints × 49k blocks each = ~200k samples |
| **Output Path** | `/Users/meharkapoor7/nes-agent-workspace/nes-research/out/detectability_heldout_neural_v2/heldout_neural_results.json` |
| **Pass Criteria** | Mean AUC ≈ 0.5 (at chance); all 4 held-out AUCs within [0.49, 0.51] |
| **Interpretation** | AUC ≈ 0.5 → no cross-checkpoint generalization (supports undetectability). AUC > 0.55 → potential detectability gap. |
| **Status** | **READY** — all checkpoints local, script exists |

---

### C2: Verify Merged Small-Capacity 14-Group Analysis (MEDIUM)

| Field | Value |
|-------|-------|
| **Scientific Question** | Does the merged 14-group small-capacity analysis confirm LR AUC 0.5059 [0.5020, 0.5120] within screening threshold? |
| **Current State** | `out/detectability_smallcap_analysis_v2/detector_results.json` exists but needs verification against raw features |
| **Prerequisites** | `out/detectability_smallcap_v1/` raw features exist; 14 groups defined |
| **Command** | ```bash cd /Users/meharkapoor7/nes-agent-workspace/nes-research/code/nes-llm python3 scripts/exp_detectability_analysis_smallcap.py \ --features-dir /Users/meharkapoor7/nes-agent-workspace/nes-research/out/detectability_smallcap_v1 \ --output /Users/meharkapoor7/nes-agent-workspace/nes-research/out/detectability_smallcap_analysis_v3 ``` |
| **Expected Runtime** | ~5 minutes (CPU) |
| **Output Path** | `/Users/meharkapoor7/nes-agent-workspace/nes-research/out/detectability_smallcap_analysis_v3/detector_results.json` |
| **Pass Criteria** | LR AUC ∈ [0.50, 0.515]; CI95 within screening threshold |
| **Status** | **READY** — data exists, script exists |

---

### C3: Pin Qwen2.5-3B Historical Revision (MEDIUM)

| Field | Value |
|-------|-------|
| **Scientific Question** | What exact revision was used for Qwen2.5-3B small-capacity experiments? |
| **Current State** | Historical runs used revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` but current `main` may differ; not pinned in download script |
| **Prerequisites** | Access to HuggingFace Hub to verify revision history |
| **Command** | ```bash cd /Users/meharkapoor7/nes-agent-workspace/nes-research python3 -c "from huggingface_hub import HfApi; api = HfApi(); repo = api.model_info('Qwen/Qwen2.5-3B'); print([r.revision for r in repo.siblings if r.rfilename == 'config.json'][:5])" ``` |
| **Action** | Update `expansion_qualification/download_ckpt.py` or equivalent to pin Qwen2.5-3B revision |
| **Output** | Documented revision in `METHODS.md` and `REPRODUCIBILITY.md` |
| **Status** | **READY** — verification command runnable |

---

### C4: Requantization Extension to 10 Cycles (LOW)

| Field | Value |
|-------|-------|
| **Scientific Question** | Does exact recovery hold for 10 requantization cycles (vs current 5 stress / 3 matrix)? |
| **Current State** | Regression test covers 0–5; full matrix only 0–3 |
| **Prerequisites** | One tensor per model (low cost), frozen codec supports `requantize` cycles |
| **Command** | ```bash cd /Users/meharkapoor7/nes-agent-workspace/nes-research/expansion_qualification python3 exp_requant_extension.py \ --model smollm2 --tensor model.layers.11.self_attn.k_proj.weight \ --cycles 10 --sizes 10000,50000 --seeds 20261016,20261017 \ --output /tmp/nes_stage/requant_10cycle ``` |
| **Expected Runtime** | ~30 min per model (no GPU needed) |
| **Output** | `requant_10cycle_results.json` with exact_recovery per cycle |
| **Pass Criteria** | 100% exact recovery through cycle 10 |
| **Status** | **NOT RUN** — script needs creation (low effort) |

---

### C5: FreMark 2025 Access Attempt (LOW)

| Field | Value |
|-------|-------|
| **Scientific Question** | Does FreMark 2025 (LNCS 16177) describe free-region marking that overlaps with our anchor-preserving formulation? |
| **Current State** | Paywalled (DOI 10.1007/978-981-95-3185-1_14); only abstract accessible |
| **Prerequisites** | Institutional access to Springer LNCS or author contact |
| **Action** | 1. Try institutional library access 2. Email corresponding author 3. Check arXiv preprint 4. Document outcome |
| **Output** | `PRIOR_ART_VERIFICATION.md` updated with FreMark status |
| **Status** | **BLOCKED** — paywall; document as unresolved in manuscript |

---

## Phase 2: Track A — 5 Residual Methods (sign, magnitude_aware, qae, lwe_grid_parity, split_sign_parity)

### A1: Complete Detectability on 4 Missing Models (HIGH)

| Field | Value |
|-------|-------|
| **Models** | gemma-2-9b, phi-3-mini, mistral-7b, Qwen2.5-7B |
| **Methods** | 5 methods × 4 models = 20 cells (vs 30 completed for 6 models) |
| **Current State** | 6/10 protocol models have detectability; 4/10 missing |
| **Prerequisites** | Model checkpoints downloaded & NF4-staged; GPU for detector training |
| **Command Template** | ```bash python3 exp_detectability_seven_method.py \ --model MODEL_ID \ --methods sign,magnitude_aware,qae,lwe_grid_parity,split_sign_parity \ --output out/detectability_trackA/MODEL_ID ``` |
| **Expected Runtime** | ~30 min/model × 4 models = 2 hrs GPU |
| **Output** | `out/detectability_trackA/{model}/detectability_results.json` per model |
| **Pass Criteria** | Same screening: AUC ≤ 0.55, TPR@1% ≤ 0.02, clean-control ~0.5 |
| **Status** | **READY** — harness exists (`run_seven_method_detectability_stage1.py`), models need download |

---

### A2: Test split_sign_parity on Additional Models (MEDIUM)

| Field | Value |
|-------|-------|
| **Scientific Question** | Does `split_sign_parity` (new method) show detectability different from other 4 methods? |
| **Current State** | Only tested on 6 models in 30-cell matrix; not tested on 4 missing models |
| **Prerequisites** | Same as A1 |
| **Command** | Included in A1 command (all 5 methods) |
| **Special Check** | Compare `split_sign_parity` AUC vs other 4 methods per model |
| **Status** | **READY** — part of A1 |

---

### A3: Utility Evaluation (6 models × 5 methods) — REQUIRES GPU (HIGH)

| Field | Value |
|-------|-------|
| **Models** | 6 protocol models (gemma-2-2b failed embedding; 6 others) |
| **Methods** | 5 methods = 30 conditions |
| **Current State** | 0/30 utility conditions run (requires GPU, ~1.5–2 hr/condition) |
| **Prerequisites** | GPU (A100/H100 or equivalent), wikitext-2 corpus, models staged |
| **Command Template** | ```bash python3 exp_utility_trackA.py \ --model MODEL_ID \ --methods sign,magnitude_aware,qae,lwe_grid_parity,split_sign_parity \ --n-seq 580 --batch 4 \ --output out/utility_trackA/MODEL_ID ``` |
| **Expected Runtime** | 30 conditions × 1.5 hr = 45 GPU-hours (sequential) |
| **Output** | `out/utility_trackA/{model}/comparisons.json` |
| **Pass Criteria** | Paired 95% CI within ±0.5% PPL for all conditions |
| **Status** | **BLOCKED** — no GPU available in current environment |

---

### A4: Production Decryption & Key-Dependent Recovery (MEDIUM)

| Field | Value |
|-------|-------|
| **Scientific Question** | Do all 5 methods support correct extraction with right key and failure with wrong key? |
| **Current State** | Not systematically tested for Track A methods |
| **Prerequisites** | Embedded artifacts from roundtrip tests |
| **Command** | ```bash python3 exp_production_decrypt_trackA.py \ --model MODEL_ID \ --methods sign,magnitude_aware,qae,lwe_grid_parity,split_sign_parity \ --output out/production_decrypt_trackA/MODEL_ID ``` |
| **Expected Runtime** | ~10 min/model (CPU) |
| **Output** | `production_decrypt_results.json` with right-key/wrong-key/keyless results |
| **Pass Criteria** | Right key: exact recovery; Wrong key: checksum fail; Keyless: checksum fail |
| **Status** | **READY** — script needs creation |

---

### A5: LWE Grid Parity Key-Dependent Recovery Validation (MEDIUM)

| Field | Value |
|-------|-------|
| **Scientific Question** | Does `lwe_grid_parity` actually use key-dependent grid parameters for embedding/recovery? |
| **Current State** | Implementation exists but key-dependence not validated end-to-end |
| **Prerequisites** | Embedded artifacts, multiple keys |
| **Command** | ```bash python3 exp_lwe_key_dependence.py \ --model MODEL_ID \ --keys key1,key2,key3 \ --output out/lwe_key_dep/MODEL_ID ``` |
| **Expected Runtime** | ~15 min/model |
| **Output** | `lwe_key_dependence_results.json` |
| **Pass Criteria** | Different keys → different carrier sets → wrong key fails extraction |
| **Status** | **READY** — script needs creation |

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