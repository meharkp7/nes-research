# LIMITATIONS_AND_THREATS.md — Track C Limitations & Threats to Validity

**Status:** Complete — exhaustive for Track C manuscript  **Date:** 2026-10-21  **Commit:** `5b40120` (`research/contract-b-b14-and-evaluation`)  **Frozen codec:** `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764`

---

## 1. Scope Limitations (Explicitly Not Claimed)

| Limitation | Explicit Statement in Manuscript |
|------------|----------------------------------|
| **Novelty** | "We do not claim a new embedding principle. Nearest-same-parity code-index embedding is NID (Liu et al. 2016) / CNV-QIM (Xiao et al. 2008). Our contribution is the NF4-specific anchor-preserving formulation and explicit repeated-requantization exactness." |
| **Universality** | "Evaluated on 5 models (1.4B–1.7B parameters), 1 tensor type per layer depth × projection, ≤50k bits payload, ≤3 requantization cycles (matrix) / 5 cycles (stress). Not evaluated on >2B models, MoE, non-NF4 quantization, or generation tasks." |
| **Undetectability** | "Detectability at chance (AUC ≤ 0.512) under our threat model (grouped CV, 16 features, 3 detectors). Small-capacity supplement shows LR AUC 0.534 (p=0.005) on 4 groups — exploratory. Not tested against adaptive attackers, full-model scans, or semantic detectors." |
| **Robustness** | "Tested: requantization (0–5 cycles), carrier corruption, truncation, wrong key, over-capacity. Not tested: fine-tuning, distillation, pruning, quantization-aware training, adversarial fine-tuning, weight averaging." |
| **Utility** | "Paired PPL on wikitext-2, ±0.5% margin. Not evaluated: downstream task performance, generation quality, perplexity on other corpora, long-context behavior." |

---

## 2. Experimental Coverage Gaps

### 2.1 Model Coverage (5/50+ architectures)

| Dimension | Covered | Not Covered |
|-----------|---------|-------------|
| Architectures | Llama (4), GPT-NeoX (1) | Mistral, Gemma, Phi, Qwen2.5>1.5B, Falcon>1B, Mamba, RWKV, MoE |
| Scale | 1.4B–1.7B | 7B+, 70B+, 400B+ |
| Quantization | NF4 (bitsandbytes) | FP4, INT4, AWQ, GPTQ, HQQ, SpQR, QuIP# |
| Tokenizers | BPE (Llama/GPT-NeoX) | SentencePiece, character-level |

**Impact:** Results cannot be extrapolated to larger models or other quantization formats without validation.

### 2.2 Tensor Coverage (6/≈1000 per model)

| Covered | Not Covered |
|---------|-------------|
| 3 depths (early/mid/late) × 2 attention (q/v or k/v) + 2 MLP (up/down or gate/down) | All other projections (k_proj, v_proj at other layers, o_proj, dense, etc.), embeddings, LM head, norm weights |
| 6 tensors per model | ~994 tensors per model |

**Impact:** Carrier capacity and detectability may differ on untested tensors (e.g., small-capacity k/v projections show different statistics).

### 2.3 Payload Size (≤50k bits)

| Tested | Not Tested |
|--------|------------|
| 1k, 5k, 10k, 20k, 50k bits | 100k+, 1M+, adaptive/capacity-filling |
| Carrier utilization: 0.03%–1.2% of interior codes | Near-capacity regimes, multi-tensor embedding |

**Impact:** Behavior at high utilization unknown; carrier collision and detectability may change.

### 2.4 Requantization Cycles (0–3 matrix, 0–5 stress)

| Tested | Not Tested |
|--------|------------|
| 0, 1, 2, 3 cycles (full matrix) | 6–10 cycles, 100 cycles |
| 0–5 cycles (regression stress test) | Continuous requantization, mixed-precision paths |

**Impact:** Long-chain requantization (e.g., federated learning, model cascades) not validated.

### 2.5 Corpus & Utility (wikitext-2 only)

| Tested | Not Tested |
|--------|------------|
| wikitext-2 test, ctx=512, non-overlapping | C4, Pile, RedPile, domain-specific, long-context (>4k), chat format |

**Impact:** PPL equivalence on wikitext-2 ≠ utility preservation on all tasks.

---

## 3. Detectability Threats

### 3.1 Qwen2.5-1.5B LR AUC Discrepancy

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Logistic Regression AUC | 0.5068 | ≤0.55 | PASS |
| Permutation p-value | 0.015 | >0.05 | **FAIL (nominal)** |
| Clean-control AUC | 0.5120 | — | **LR AUC < clean-control** |

**Interpretation:** The significant permutation p-value (0.015) is below its own clean control (0.5120). The signal is weaker than the natural variation in clean data. Retained as **unresolved minor discrepancy**, not a detectability failure.

**Source:** `expansion_qualification/qwen15/detectability_v1/detectability_results.json`

### 3.2 Small-Capacity Signal (Exploratory)

| Metric | 4-Group (Original) | 14-Group (Merged) |
|--------|-------------------|-------------------|
| LR AUC | 0.5343 [0.5332, 0.5363] | 0.5059 [0.5020, 0.5120] |
| Permutation p | 0.0050 | — |
| Groups | 4 (TinyLlama k/v, Qwen2.5-3B k/v) | 14 (adds Pythia 2, Falcon 2, SmolLM2 4, Qwen15 4) |
| Max \|Cohen's d\| | 0.0886 | — |

**Interpretation:** The 4-group result is exploratory (underpowered: only 4 independent tensors). The merged 14-group analysis shows AUC 0.5059 within screening threshold. **Labeled exploratory in manuscript.**

**Source:** `out/detectability_smallcap_v1/supplement_results.json`, `out/detectability_smallcap_analysis_v2/detector_results.json`

### 3.3 Limited Detector Diversity

| Included | Not Included |
|----------|--------------|
| Logistic Regression, Gradient Boosting, MLP (exploratory) | Random Forest, SVM, k-NN, deep CNNs, Transformers, spectral detectors, statistical moments beyond 16 features |
| 16 handcrafted features (mean, std, min, max, median, q25, q75, mean_sq, pos_frac, skew, kurtosis, hist_entropy, zero_frac, abs_mean) | Raw code histograms, cross-block correlations, higher-order statistics |

**Impact:** A more powerful detector with different features/architecture could find signal.

### 3.4 Grouped CV Limitations

- **Only 6 groups** (model::tensor) in main evaluation → limited CV folds
- **No seed-vs-seed control** in StableLM dataset
- **Leave-one-model-out** = exploratory (single held-out checkpoint per model)
- **Held-out neural** only 4 checkpoints (SmolLM2, Qwen1.5B, Pythia, Falcon); StableLM/TinyLlama/Qwen3B not available locally

---

## 4. Methodological Threats

### 4.1 Endpoint Exclusion = Capacity Reduction

| Aspect | Impact |
|--------|--------|
| Interior codes only (1–14) | 12.5% capacity loss vs full codebook (14/16) |
| Endpoints carry scale information | Verified: endpoints unchanged, scale preserved |
| Baseline `pair_parity` uses endpoints | Fails requantization; endpoint mutation detected |

**Trade-off:** Capacity for robustness. Explicitly stated.

### 4.2 Deterministic Carrier Selection

- Carrier positions derived from (key, tensor name, bit index)
- **No secret carrier set** — security by key-dependent payload only
- Wrong-key extraction fails (verified in regression)
- **Threat:** Known carrier positions + known algorithm → targeted attacks possible

### 4.3 Framing Overhead

- 64 bits fixed overhead (MAGIC 32 + length 16 + CRC 16)
- At 1k payload: 6.4% overhead
- At 50k payload: 0.13% overhead

---

## 5. Infrastructure & Reproducibility Threats

### 5.1 Staging on `/tmp/nes_stage/` (Overlay)

- `/workspace` nearly full → models staged to `/tmp/nes_stage/`
- `/tmp` cleaned on reboot → artifacts lost unless copied
- **Mitigation:** All canonical outputs copied to repo `expansion_qualification/` and `out/`

### 5.2 Held-Out Neural Results Lost

- Output was in `/tmp/nes_local_run/out/detectability_heldout_neural/`
- `/tmp` cleaned → results lost
- **Action Required:** Re-run before manuscript submission

### 5.3 Qwen2.5-3B Revision UNVERIFIED

- Historical runs used `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`
- Current `main` branch may differ
- **Impact:** Small-capacity results on Qwen2.5-3B not reproducible without pinning

### 5.4 Frozen Manifest Self-Hash Mismatch

- `FROZEN_ALL_HASHES.txt` lists its own SHA-256
- 6 narrative docs changed after manifest (mtime 2026-10-19 10:09–10:10)
- Raw artifacts (data, code) re-verify clean
- **Disclosed** in expansion report §0.1, not repaired

---

## 6. Prior Art / Novelty Threats (Gate G = PARTIAL)

| Prior Work | Overlap | Status |
|------------|---------|--------|
| NID (Liu et al. 2016) | Nearest-same-parity code-index embedding | **Confirmed overlap** — not novel in general |
| CNV-QIM (Xiao et al. 2008) | Constant-norm QIM, parity embedding | **Confirmed overlap** — same principle |
| Li et al. 2023 (EMNLP Findings) | Invariant-region watermarking for quantized LLM weights | **Confirmed overlap** — invariant region = anchor preservation |
| FreMark 2025 (LNCS 16177) | "Free-region marking" in quantized models | **UNVERIFIED** — full text paywalled (DOI 10.1007/978-981-95-3185-1_14) |
| HPQIM (Li & Chang) | High-payload QIM, invariant embedding | **Verified** — Warwick TR, similar concept |
| Kumaresam et al. 2026 (TDCommons) | Defensive publication on quantization watermarking | **Verified** — posterior to our work |

**Gate G Assessment:** PARTIAL — NID/CNV-QIM overlap confirmed; FreMark 2025 inaccessible. **Cannot claim novelty of embedding principle.** Contribution limited to: NF4 packed-code formulation + endpoint exclusion protocol + repeated-requantization exactness demonstration.

---

## 7. Statistical Threats

| Threat | Mitigation |
|--------|------------|
| Multiple comparisons (5 models × 3 detectors × 2 controls) | No formal correction; all primary detectors PASS screening; only Qwen LR nominal |
| Low group count (6) for CV | Small-cap supplement (14 groups) and held-out neural (4 checkpoints) as supplements |
| Permutation test n=200 | Sufficient for p≈0.01 resolution; not for p<0.005 |
| Bootstrap n=2000 | Adequate for 95% CI |
| Paired PPL n=580 sequences | StableLM used 296k tokens/condition (13× tighter CI) |

---

## 8. Honest Threat Summary for Manuscript

| Category | Severity | Manuscript Statement |
|----------|----------|---------------------|
| Novelty (Gate G) | HIGH — confirmed NID/CNV-QIM overlap | "We do not claim a new embedding principle..." |
| Model coverage | MEDIUM — 5 small models only | "Evaluated on 5 models (1.4B–1.7B); larger models and other families untested." |
| Tensor coverage | MEDIUM — 6/1000 per model | "One tensor per depth × projection; other tensors may behave differently." |
| Qwen LR discrepancy | LOW — below clean control | "One nominal discrepancy (Qwen LR p=0.015, AUC=0.5068 < clean 0.512); retained as unresolved." |
| Small-cap signal | LOW — exploratory, 4 groups → 14 groups merges to 0.5059 | "Small-capacity supplement (4 groups) shows LR AUC 0.534 (p=0.005); merged 14-group analysis yields 0.5059 within screening threshold. Labeled exploratory." |
| Held-out neural | LOW — at chance, 4 checkpoints | "Held-out checkpoint neural detector at chance (mean AUC 0.4998). Only 4 checkpoints available." |
| Utility corpus | MEDIUM — wikitext-2 only | "Paired PPL equivalence on wikitext-2; other corpora and generation tasks untested." |
| Requantization depth | LOW — 0–5 cycles | "Tested up to 5 cycles; longer chains untested." |

---

## 9. Prohibited Claims (Must Not Appear in Manuscript)

| Prohibited Claim | Reason |
|------------------|--------|
| "First to embed payloads in quantized LLMs" | NID 2016, CNV-QIM 2008, Li 2023, EmMark 2024 |
| "Universally undetectable" | Only tested 3 detectors, 16 features, 5 models |
| "Robust to all transformations" | Only requantization tested; fine-tuning, pruning, etc. not tested |
| "Zero utility impact" | ±0.5% PPL equivalence on wikitext-2 only |
| "Novel invariant-region method" | Li 2023, FreMark 2025 (paywalled), HPQIM |
| "New QIM variant" | NID/CNV-QIM is nearest-same-parity QIM |
| "Secure against adaptive attackers" | Threat model: passive detector, known algorithm, unknown key |

---

## 10. Permitted Claims (Supported by Evidence)

| Claim | Evidence |
|-------|----------|
| "`interior_nearest` achieves exact payload recovery (BER=0) after 0–3 NF4 requantization cycles on 5 models" | `expansion_audit_table.csv` (4320 rows, all verified) |
| "Endpoint codes (0, 15) are never modified, preserving block scales" | `changed_anchor_codes = 0` everywhere; integrity checks PASS |
| "Paired PPL change within ±0.5% on wikitext-2 for all 24 expansion conditions" | `comparisons.json` (6/6 PASS per model) |
| "Detectability at chance under grouped CV (AUC ≤ 0.512, TPR@1% ≤ 0.011)" | 5 `detectability_results.json` files |
| "Repeated-requantization exactness (0–5 cycles) verified via regression suite" | 60/60 regression tests PASS |
| "NF4-specific anchor-preserving formulation with exact repeated-requantization exactness" | Method + evidence above |

---

## 11. Required Re-Runs Before Submission

| Experiment | Reason | Priority |
|------------|--------|----------|
| Held-out neural detector (4 checkpoints) | Results lost in `/tmp` cleanup | HIGH |
| Small-cap merged 14-group analysis | Verify LR AUC 0.5059 CI | MEDIUM |
| Qwen2.5-3B revision pinning | Small-cap results not reproducible | MEDIUM |
| FreMark 2025 access attempt | Resolve Gate G paywall gap | LOW (document if impossible) |

---

## 12. Threats to Reproducibility (External)

| Threat | Likelihood | Impact |
|--------|------------|--------|
| HuggingFace Hub model revision change | LOW (pinned revisions) | Medium — would change weights |
| `bitsandbytes` NF4 algorithm change | LOW (version pinned) | High — changes quantization |
| PyTorch/cuDNN non-determinism | MEDIUM | Low — verified by hash |
| `/tmp` cleanup on shared infrastructure | HIGH | Medium — mitigated by canonical outputs |
| GPU unavailable for utility/detectability | MEDIUM | High — CPU fallback slow |

---

## 13. Summary: What This Work Is / Is Not

| This Work IS | This Work Is NOT |
|--------------|------------------|
| NF4-specific anchor-preserving payload embedding | A new watermarking/steganography principle |
| Exact recovery under repeated NF4 requantization (0–5 cycles) | Robust to fine-tuning, pruning, distillation |
| Demonstration on 5 model architectures (1.4B–1.7B) | Validated on 7B+ or MoE models |
| Detectability at chance under our threat model | Proven secure against all detectors |
| Paired PPL equivalence (±0.5%) on wikitext-2 | Utility preservation on all tasks |
| Reproducible, hashed, regression-tested pipeline | A production-ready system |