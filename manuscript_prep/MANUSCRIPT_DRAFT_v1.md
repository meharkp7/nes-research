# Anchor-Preserving Payload Embedding in NF4-Quantized Language Model Weights Survives Requantization

**Authors:** [Anonymized for submission]  
**Date:** 2026-10-21  
**Code:** `code/nes-llm/src/experiments/nf4_anchor_channel.py` (SHA-256: `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764`)  
**Data:** `evidence/ckpt_nf4/`, `evidence/heldout_neural_v2/`, `evidence/smallcap_extended_v1/`  
**Protocol:** `stablelm_qualification/QUALIFICATION_PROTOCOL_V2.md` (Track A gates)

---

## Abstract

NF4 quantization with double quantization is the de facto standard for compressing large language models. However, requantization — the dequantize/requantize cycle — perturbs block scales and code assignments, breaking most payload embedding schemes. We present `interior_nearest`, an NF4-specific anchor-preserving formulation that restricts carriers to interior NF4 codes (1–14), never rewriting endpoints (0, 15) that carry double-quantized block scales. The embedding moves each carrier to the nearest interior code with the target parity bit. Across five model architectures (SmolLM2-1.7B, Qwen2.5-1.5B, Pythia-1.4b, Falcon3-1B, StableLM-2-1.6B), six tensors per model, five payload sizes (1k–50k bits), three seeds, and four requantization cycles (0–3), all 1,800 roundtrip conditions achieve exact payload recovery (BER = 0). Baseline methods (`pair_parity`, `nearest_parity`) fail from cycle 1 (90–108/360 exact). Paired perplexity on wikitext-2 (580 sequences, ±0.5% margin) shows all 24 expansion conditions within margin (max |Δ| 0.019%). Detectability under grouped cross-validation (logistic regression, gradient boosting, exploratory MLP; 16 features) is at chance: max AUC 0.512, TPR@1% ≤ 0.011, clean-control AUC ~0.5. A small-capacity supplement (12 groups, k_proj/v_proj at 50k bits) shows logistic regression AUC 0.517 [0.509, 0.526] — above the main evaluation but within screening threshold (AUC < 0.55). A leave-one-checkpoint-out neural detector on four checkpoints (balanced test, class-weighted loss) yields mean AUC 0.5004 ± 0.0003. Novelty is limited: nearest-same-parity code-index embedding originates in NID (Liu et al. 2016) / CNV-QIM (Xiao et al. 2008); invariant-region preservation for quantized LLM weights is proposed by Li et al. (2023) and Guo et al. (2025); FreMark 2025 (LNCS 16177) describes free-region marking but is paywalled and overlap cannot be assessed (Gate G = PARTIAL). Our contribution is the NF4 packed-code formulation with endpoint exclusion and empirical demonstration of exact repeated-requantization exactness.

---

## 1. Introduction

### 1.1 Problem Statement

Quantization is essential for deploying large language models (LLMs). NF4 (NormalFloat-4) with double quantization (Dettmers et al., 2023) is the standard 4-bit format. However, NF4 weights are not static: they undergo **requantization** — dequantization to float16 followed by fresh NF4 quantization — during model conversion, federated learning, cascade pipelines, and quantization-aware training. Requantization changes block scales and reassigns codes, destroying payloads embedded by naive parity or LSB methods.

### 1.2 Threat Model

We consider a **passive detector** who:
- Knows the embedding algorithm and tensor locations
- Has white-box access to the model weights
- Does not know the secret key used for payload generation
- Attempts to distinguish clean vs. embedded weights using statistical features

We do **not** claim robustness against adaptive attackers, fine-tuning, pruning, distillation, or generation-quality degradation.

### 1.3 Contributions

1. **NF4 anchor-preserving formulation:** Carriers restricted to interior codes (1–14); endpoints (0, 15) never modified, preserving double-quantized block scales.
2. **Nearest-same-parity embedding:** Each carrier moved to nearest interior code with target parity (minimal |Δ| = 1).
3. **Exact repeated-requantization exactness:** BER = 0 across 0–5 requantization cycles on 5 model checkpoints, 1,800 conditions (cycles 0–3 matrix) + separate stress tests.
4. **Utility preservation:** Paired PPL on wikitext-2 within ±0.5% on all 24 expansion conditions.
5. **Detectability at chance:** Main evaluation AUC ≤ 0.512; small-capacity supplement AUC 0.517 (exploratory); neural cross-checkpoint AUC 0.5000.
6. **Reproducible pipeline:** Frozen codec (SHA-256 verified), pinned model revisions, regression suite (60/60 tests PASS).

### 1.4 Non-Claims (Explicit Limitations)

| We Do NOT Claim | Reason |
|-----------------|--------|
| Novel embedding principle | Nearest-same-parity is NID (Liu 2016) / CNV-QIM (Xiao 2008) |
| First invariant-region LLM watermarking | Li (2023), Guo (2025) propose invariant anchors |
| Universal undetectability | Only 3 detectors, 16 features, 5 models tested |
| Robustness to fine-tuning/pruning | Only requantization tested |
| Utility on generation tasks | Only wikitext-2 PPL measured |
| Novelty over FreMark 2025 | Full text paywalled (LNCS 16177); Gate G = PARTIAL |

---

## 2. Related Work

### 2.1 Quantization Index Modulation (QIM) and Variants

**Chen & Wornell (2001)** introduced QIM: embed by quantizing to nearest point in codebook subset indexed by message. **Altun et al. (2006)** generalized via set-theoretic formulation.

**Liu et al. (2016, NID)** and **Xiao et al. (2008, CNV-QIM)** describe **nearest-same-parity code-index embedding**: map each coefficient to the nearest code with the target parity bit. This is the core embedding principle used in `interior_nearest`. We do not claim novelty of this principle.

### 2.2 LLM Weight Watermarking

**Kuribayashi et al. (2020, DeepWatermark):** Watermark DNN weights via quantization; robust to fine-tuning but not requantization-specific.

**Li et al. (2023, EMNLP Findings):** **Invariant-region watermarking for quantized LLM weights** — preserve "anchor" weights during embedding. Directly anticipates our endpoint preservation concept.

**Zhang & Koushanfar (2024, EmMark):** Embedding via error-correcting codes; robust to quantization but different technique.

**Guo et al. (2025, arXiv:2507.08288):** **Invariant watermarking for quantized LLMs** preserving quantization anchors. Very recent, similar invariant-anchor concept in NF4 context.

**Luan et al. (2025, PoPETs):** Permutation-based embedding + ECC; different principle.

**FreMark (Chen et al., 2025, DSPP, LNCS 16177):** "Free-region marking for quantized LLMs" with claimed requantization robustness. **Full text paywalled** (DOI: 10.1007/978-981-95-3185-1_14); abstract suggests potential overlap with our anchor-preserving formulation. Overlap cannot be assessed. **Gate G = PARTIAL.**

### 2.3 High-Payload QIM Variants

**Li & Chang (HPQIM, Warwick TR):** High-payload QIM using invariant embedding regions. Technical report, not peer-reviewed.

**Kumaresam et al. (2026, TDCommons dpubs_series/10523):** Defensive publication on quantization watermarking; posterior to our work.

---

## 3. Method

### 3.1 NF4 Quantization Layout

| Parameter | Value |
|-----------|-------|
| Quantization type | NF4 (NormalFloat-4) |
| Block size | 64 weights |
| Nested (double) quantization | True |
| Storage | uint8 (4-bit packed, 2 codes/byte) |
| Compute dtype | float16 |

Each block of 64 weights shares one scale (and double-quantized scale). The 16 codebook levels map to indices 0–15. **Endpoints 0 and 15 carry scale information and are never rewritten.**

### 3.2 Carrier Eligibility

For each 64-weight block, a code at position `i` is eligible iff:
```
eligible[i] = (1 ≤ code[i] ≤ 14) AND (the 64-code block containing i has at least one code equal to 0 or 15)
```
Only interior codes (1–14) in blocks containing at least one endpoint code (0 or 15) participate. Blocks without an endpoint anchor are excluded; endpoint codes are never modified.

### 3.3 Payload Framing

Payload `P` (L bits) → framed bitstream:
```
MAGIC (32 bits) || LENGTH (16 bits) || P || CRC16 (16 bits)
```
Total framed bits: `L + 64`. Seeds: 20261016, 20261017, 20261018. Key: `SHA-256("NES-development-{seed}")[:32]`.

### 3.4 Deterministic Carrier Selection

Carrier positions derived from `(key, tensor_name, bit_index)` via deterministic PRNG. Only eligible positions used. Carrier set is **stable across requantization cycles** (verified).

### 3.5 Embedding Operation: `interior_nearest`

For carrier code `c ∈ [1, 14]` and payload bit `b ∈ {0,1}`:
```
target_parity = b
if (c % 2) == target_parity:
    c' = c
else:
    if c == 1:      c' = 2   # only even neighbor
    elif c == 14:   c' = 13  # only odd neighbor
    else:
        # Two interior neighbors: c-1 and c+1
        c' = c-1 if abs((c-1) - c) ≤ abs((c+1) - c) else c+1
```
**Invariants:** `c' ∈ [1,14]`, `(c' % 2) == b`, `|c' - c| = 1`, endpoints unchanged.

### 3.6 Baseline: `pair_parity` (Fails Under Requantization)

```
c' = (c & 14) | b   # Forces parity by masking LSB; can write 0 or 15
```
Mutates endpoints → breaks double-quantized scales → fails from cycle 1.

### 3.7 Extraction

1. Load stego tensor, unpack NF4 codes
2. Re-derive carrier positions from `(key, tensor_name, bit_index)`
3. Read parity bits → recover framed bitstream
4. Verify MAGIC, length, CRC16 → return payload
6. No clean cover needed; wrong key → checksum failure (verified).

### 3.8 Requantization Cycle

One cycle: `dequantize_4bit(NF4) → float16 → quantize_4bit(NF4, same blocksize/nesting)`
Implemented via `requantize(tensor, state)` in frozen codec.

---

## 4. Experimental Setup

### 4.1 Models (Pinned Revisions)

| Model | Revision | Architecture | Layers |
|-------|----------|--------------|--------|
| StableLM-2-1.6B | `f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef` | Llama | 24 |
| SmolLM2-1.7B | `effd688a12921b4cc83e3312b6feb579f70f9c71` | Llama | 24 |
| Qwen2.5-1.5B | `8faed761d45a263340a0528343f099c05c9a4323` | Llama | 28 |
| Pythia-1.4b | `fedc38a16eea3bd36a96b906d78d11d2ce18ed79` | GPT-NeoX | 24 |
| Falcon3-1B | `cb37ef3559b157b5c9d9226296ba01a5162da1f7` | Llama | 18 |

### 4.2 Tensors (6 per model)

Depth × projection: early/mid/late × {attention (q/k/v), MLP (up/down/gate)}. Exact names per model in manifests.

### 4.3 Payload Parameters

| Size (bits) | Seeds | Framed bits |
|-------------|-------|-------------|
| 1,000 | 3 | 1,064 |
| 5,000 | 3 | 5,064 |
| 10,000 | 3 | 10,064 |
| 20,000 | 3 | 20,064 |
| 50,000 | 3 | 50,064 |

Carrier utilization: 0.03%–1.2% of interior codes.

### 4.4 Corpus & Utility

wikitext-2-raw-v1 test split, model's own tokenizer, context 512, non-overlapping windows, batch 4, 580 sequences (296k tokens for StableLM protocol-matched), paired 95% CI, ±0.5% relative PPL margin.

### 4.5 Detectability

**Main:** 5 models × 6 tensors = 30 groups. Features: 16 block statistics (endpoint_frac, zero_count, fifteen_count, interior_frac, mean_code, std_code, odd_frac, parity_imbalance, odd_interior_frac, distinct_codes, code_entropy, mean_abs_value, absmax_value, adjacent_equal_frac, adjacent_parity_change, max_run). Detectors: Logistic Regression (LR), Gradient Boosting (GB), MLP (exploratory). Grouped CV: GroupKFold by `model::tensor`. Metrics: AUC, bootstrap CI, TPR@1%/5% FPR, permutation test (n=200), clean-vs-clean control.

**Small-capacity supplement:** k_proj/v_proj tensors at 50k bits (~95% blocks touched). 12 groups (6 models × 2 tensors). Same features/detectors.

**Held-out neural:** Leave-one-checkpoint-out MLP on 4 checkpoints (SmolLM2, Qwen1.5B, Pythia, Falcon). Train on 3, test on 1.

### 4.6 Regression Suite (12 tests per model)

1. `deterministic_fixed_seed`  2. `endpoints_unchanged`  3. `interior_only`
4. `carrier_eligibility_stable`  5. `fresh_process_extraction`  6. `requantize_cycles_0_5_exact`
7. `corrupted_carrier_detected`  8. `fixed_offset_corruption_consistent`  9. `truncated_artifact_handled`
10. `wrong_key_not_exact`  11. `over_capacity_rejected`  12. `pair_parity_fails_after_requant`

---

## 5. Results

### 5.1 Roundtrip Exact Recovery (Gates A2/A3)

| Model | Method | Exact / Total | Anchor Changes | Cycles |
|-------|--------|---------------|----------------|--------|
| SmolLM2 | interior_nearest | 360/360 | 0 | 0–3 |
| SmolLM2 | nearest_parity | 90/360 | — | 0–3 |
| SmolLM2 | pair_parity | 90/360 | — | 0–3 |
| Qwen2.5-1.5B | interior_nearest | 360/360 | 0 | 0–3 |
| Qwen2.5-1.5B | nearest_parity | 90/360 | — | 0–3 |
| Qwen2.5-1.5B | pair_parity | 90/360 | — | 0–3 |
| Pythia-1.4b | interior_nearest | 360/360 | 0 | 0–3 |
| Pythia-1.4b | nearest_parity | 90/360 | — | 0–3 |
| Pythia-1.4b | pair_parity | 90/360 | — | 0–3 |
| Falcon3-1B | interior_nearest | 360/360 | 0 | 0–3 |
| Falcon3-1B | nearest_parity | 108/360 | — | 0–3 |
| Falcon3-1B | pair_parity | 108/360 | — | 0–3 |
| StableLM-2-1.6B | interior_nearest | 360/360 | 0 | 0–3 |

**Total:** 1,800/1,800 exact for `interior_nearest`; 0 anchor code mutations.

**Stress test (regression):** `requantize_cycles_0_5_exact` PASS on all 5 models (cycles 0–5). **Extended stress test:** Single-tensor (k_proj) on 4 expansion models shows 100% exact recovery through 10 requantization cycles (264 conditions: 4 models × 2 sizes × 3 seeds × 11 cycles).

**Source:** `expansion_qualification/expansion_audit_table.csv` (4,320 rows), `stablelm_qualification/roundtrip_stream_v1/summary.json`.

### 5.2 Integrity (Gate A1)

All 5 models + StableLM: `differing_keys == [target_tensor]`, `shape_preserved: true`, `payload_exact: true`, `finite_logits_loss: true`.

**Source:** `expansion_qualification/{model}/integrity.json`, `stablelm_qualification/integrity_10k.json`.

### 5.3 Detectability — Main Evaluation (Gate A4)

| Model | Clean Control AUC | GB AUC (p) | LR AUC (p) | MLP AUC (p) | Max TPR@1% |
|-------|-------------------|------------|------------|-------------|------------|
| SmolLM2 | 0.5031 | 0.4999 (0.46) | 0.5012 (0.35) | 0.4999 (0.54) | 0.0100 |
| Qwen2.5-1.5B | 0.5120 | 0.5025 (0.22) | **0.5068 (0.015)** | 0.5010 (0.38) | 0.0106 |
| Pythia-1.4b | 0.4999 | 0.5002 (0.52) | 0.5007 (0.39) | 0.5000 (0.50) | 0.0101 |
| Falcon3-1B | 0.5059 | 0.5008 (0.38) | 0.5018 (0.22) | 0.5003 (0.45) | 0.0104 |
| StableLM | 0.4848 | 0.5005 [0.4997,0.5017] | 0.5009 [0.5006,0.5016] | 0.5000 [0.4996,0.5004] | 0.0100 |

**Screening:** AUC ≤ 0.55, TPR@1% ≤ 0.02, clean-control ~0.5 → **All PASS**.

**Qwen LR discrepancy:** AUC 0.5068, permutation p=0.015 — but **below its own clean control (0.5120)**. Retained as unresolved minor discrepancy, not a detectability failure.

**Source:** `expansion_qualification/{model}/detectability_v1/detectability_results.json`, `stablelm_qualification/detectability_v1/detectability_results.json`.

### 5.4 Detectability — Small-Capacity Supplement (Exploratory)

| Detector | AUC | 95% CI (group bootstrap) | TPR@1% | TPR@5% |
|----------|-----|--------------------------|--------|--------|
| Logistic Regression | **0.51669** | [0.50903, 0.52585] | 0.0128 | 0.0589 |
| Gradient Boosting | 0.51002 | [0.50452, 0.51690] | 0.0103 | 0.0528 |
| MLP (exploratory) | 0.50265 | [0.50112, 0.50459] | 0.0106 | 0.0522 |

**Clean-vs-clean:** AUC 0.5052 [0.4963, 0.5142]

**Groups:** 12 model–tensor groups (6 checkpoints × 2 tensors: k_proj, v_proj at 50k bits). This is the primary durable supplement. Earlier 4-group experiment (TinyLlama + Qwen2.5-3B) yielded LR AUC 0.5343; merged 14-group analysis yielded LR AUC 0.5059. All detectors PASS screening (AUC < 0.55). Labeled **exploratory** — 12 groups underpowered for definitive conclusion.

**Source:** `evidence/smallcap_extended_v1/supplement_results.json` (primary); `out/detectability_smallcap_v1/supplement_results.json` (original 4-group, archived); `out/detectability_smallcap_analysis_v2/detector_results.json` (merged 14-group, archived).

### 5.5 Detectability — Held-Out Checkpoint Neural (Fixed v3)

| Held-out | Neural MLP AUC | AUC 95% CI | Clean-vs-Clean AUC | n_train | n_test_bal |
|----------|----------------|------------|-------------------|---------|------------|
| SmolLM2 | 0.5007 | [0.4932, 0.5084] | 0.5027 | 258,048 | 24,576 |
| Qwen2.5-1.5B | 0.5007 | [0.4934, 0.5087] | 0.4899 | 258,048 | 24,576 |
| Pythia-1.4b | 0.5000 | [0.4926, 0.5071] | 0.4994 | 258,048 | 24,576 |
| Falcon3-1B | 0.5003 | [0.4931, 0.5073] | 0.5098 | 258,048 | 24,576 |

**Mean AUC:** 0.5004 ± 0.0003 — all at chance. Balanced accuracy ≈ 0.500.  
**Fixes applied:** Class-weighted BCE loss, balanced test sets (equal clean/embedded per tensor), bootstrap CI.  
**Previous v2 issue:** Severe class imbalance (6:1) caused all-positive prediction collapse (TN=0, FN=0, accuracy=85.7%).  
**Interpretation:** No cross-checkpoint generalization; neural detector performs at chance under balanced evaluation.

**Source:** `evidence/heldout_neural_v3/heldout_neural_results.json`.

### 5.6 Utility — Paired PPL Equivalence (Gate A5)

| Model | Clean PPL | n_seq | Conditions PASS | Max |Δ| | Max CI half-width |
|-------|-----------|-------|-----------------|-----|-----|-------------------|
| SmolLM2-1.7B | 12.21609 | 580 | 6/6 | 0.0019% | 0.0033% |
| Qwen2.5-1.5B | 13.07853 | 580 | 6/6 | 0.014% | 0.0115% |
| Pythia-1.4b | 16.43043 | 563* | 6/6 | 0.019% | 0.0191% |
| Falcon3-1B | 12.56639 | 580 | 6/6 | 0.014% | 0.0022% |
| StableLM-2-1.6B | 12.97796 | 580 | 6/6 | 0.0139% | 0.037% (13× tighter) |

*Pythia: 563 sequences (vs 580 target) — tokenizer vocabulary limitation. All CIs within ±0.5% margin.

**Source:** `expansion_qualification/{model}/utility_v1/comparisons.json`, `stablelm_qualification/utility_v2/gate_e_reanalysis.json`.

### 5.7 Regression Tests (Gate A6)

| Test | SmolLM2 | Qwen2.5 | Pythia | Falcon3 | StableLM |
|------|---------|---------|--------|---------|----------|
| deterministic_fixed_seed | ✅ | ✅ | ✅ | ✅ | ✅ |
| endpoints_unchanged | ✅ | ✅ | ✅ | ✅ | ✅ |
| interior_only | ✅ | ✅ | ✅ | ✅ | ✅ |
| carrier_eligibility_stable | ✅ | ✅ | ✅ | ✅ | ✅ |
| fresh_process_extraction | ✅ | ✅ | ✅ | ✅ | ✅ |
| requantize_cycles_0_5_exact | ✅ | ✅ | ✅ | ✅ | ✅ |
| corrupted_carrier_detected | ✅ | ✅ | ✅ | ✅ | ✅ |
| fixed_offset_corruption_consistent | ✅ | ✅ | ✅ | ✅ | ✅ |
| truncated_artifact_handled | ✅ | ✅ | ✅ | ✅ | ✅ |
| wrong_key_not_exact | ✅ | ✅ | ✅ | ✅ | ✅ |
| over_capacity_rejected | ✅ | ✅ | ✅ | ✅ | ✅ |
| pair_parity_fails_after_requant | ✅ | ✅ | ✅ | ✅ | ✅ |

**Regression results:** 48/48 tests passed in the reported run across four expansion checkpoints (12 per checkpoint). StableLM's 12 tests were run separately; 60/60 is an aggregate across separate runs, not one jointly executed run.

**Source:** `expansion_qualification/{model}/regressions.json`, `stablelm_qualification/regressions.json`.

---

## 6. Discussion

### 6.1 Scope of Contribution

Our contribution is **not** a new embedding principle. The nearest-same-parity code-index mapping is established in NID (Liu et al., 2016) and CNV-QIM (Xiao et al., 2008). Invariant-region preservation for quantized LLM weights is proposed by Li et al. (2023) and Guo et al. (2025). FreMark 2025 (LNCS 16177) describes free-region marking with requantization robustness but is **paywalled**; overlap cannot be verified. **Gate G (novelty) = PARTIAL.**

Our **specific contribution** is the **NF4 packed-code formulation with endpoint exclusion** and the **empirical demonstration of exact repeated-requantization exactness** (BER=0 across 0–5 cycles) on five architectures with utility preservation.

### 6.2 Detectability Caveats

1. **Qwen2.5-1.5B LR discrepancy:** Nominal p=0.015 but AUC < clean control (0.5068 < 0.5120). Signal weaker than natural variation. Unresolved minor discrepancy.
2. **Small-capacity supplement:** LR AUC 0.517 [0.509, 0.526] on 12 groups — above main evaluation but within screening threshold. Qwen15/TinyLlama groups drive elevation (LR AUC 0.53–0.55). MLP/GB near chance. **Exploratory label retained.**
3. **Limited detector diversity:** Only 16 handcrafted features, 3 detectors. Adaptive or deep detectors untested.
4. **Group count:** 6 (main) / 12 (small-cap) / 4 (held-out neural) — limited statistical power.

### 6.3 Utility Limitations

- wikitext-2 PPL only; no generation quality, downstream tasks, or long-context evaluation.
- Pythia 563 vs 580 sequences due to tokenizer; all CIs still within margin.
- ±0.5% margin is a protocol choice, not a theoretical bound.

### 6.4 Robustness Limitations

Tested: requantization (0–5 cycles), carrier corruption, truncation, wrong key, over-capacity.  
Not tested: fine-tuning, distillation, pruning, QAT, adversarial fine-tuning, weight averaging.

### 6.5 Model Coverage

5 models, 1.4B–1.7B parameters, Llama (4) + GPT-NeoX (1). No Mistral, Gemma, Phi, Qwen>1.5B, Falcon>1B, MoE, Mamba, or >7B models.

---

## 7. Conclusion

We present `interior_nearest`, an NF4-specific anchor-preserving payload embedding formulation that achieves **exact repeated-requantization exactness** (BER=0 across 0–5 cycles in regression stress test; 10 cycles on single-tensor k_proj stress test) by restricting carriers to interior NF4 codes (1–14) and never rewriting endpoints (0, 15) that carry double-quantized block scales. Across five model architectures, 1,800 roundtrip conditions (cycles 0–3 matrix), all achieve exact recovery. Paired perplexity on wikitext-2 is within ±0.5% on all 24 expansion conditions. Detectability under grouped CV is at chance (max AUC 0.512); a small-capacity supplement (12 groups) shows LR AUC 0.517 (exploratory); a leave-one-checkpoint-out neural detector on four checkpoints (balanced test, class-weighted loss) yields mean AUC 0.5004 ± 0.0003. The embedding principle (nearest-same-parity) is not novel (NID 2016, CNV-QIM 2008); invariant-region preservation is proposed by Li 2023 and Guo 2025; FreMark 2025 overlap is unresolved due to paywall (Gate G = PARTIAL). We do not claim universal undetectability, universal robustness, or unqualified novelty. The contribution is the NF4-specific formulation and its empirical validation.

---

## 8. Reproducibility

| Item | Location |
|------|----------|
| Frozen codec | `code/nes-llm/src/experiments/nf4_anchor_channel.py` |
| Codec SHA-256 | `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` |
| Model revisions | Pinned in manifests (Table 1) |
| Checkpoints | `evidence/ckpt_nf4/` (durable) |
| Roundtrip audit | `expansion_qualification/expansion_audit_table.csv` |
| Detectability (main) | `expansion_qualification/*/detectability_v1/` |
| Detectability (small-cap) | `evidence/smallcap_extended_v1/`, `out/detectability_smallcap_v1/` |
| Detectability (held-out neural) | `evidence/heldout_neural_v3/` (fixed: balanced test, class-weighted loss) |
| Utility | `expansion_qualification/*/utility_v1/`, `stablelm_qualification/utility_v2/` |
| Regression | `expansion_qualification/*/regressions.json` |
| Manifest v2 | `FROZEN_ALL_HASHES_v2.txt` (+ `.sha256`) |

All experiments use fixed seeds (20261016/017/018 for payloads, 20261016 for CV/bootstrap). Grouped CV prevents tensor leakage. Run ledger in `evidence/EVIDENCE_VALIDATION_REPORT.md`.

---

## 9. References

1. Chen & Wornell, "Quantization Index Modulation," IEEE Trans. IT 47(4):1423–1443, 2001.
2. Altun et al., "Set-Theoretic QIM," ICASSP 2006.
3. **Liu et al., "Nearest Interpolation Decoding," J. Ambient Intell. Humaniz. Comput. 7(1):139–147, 2016.** (NID — core principle)
4. **Xiao et al., "Constant-Norm QIM," GLOBECOM 2008.** (CNV-QIM — core principle)
5. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," ICML 2023 (NF4).
6. Kuribayashi et al., "DeepWatermark," APSIPA 2020.
7. **Li et al., "Watermarking Quantized LLMs," EMNLP Findings 2023.** (Invariant-region overlap)
8. Zhang & Koushanfar, "EmMark," DAC 2024, arXiv:2402.17938.
9. **Guo et al., "Invariant Watermarking for Quantized LLMs," arXiv:2507.08288, 2025.** (Invariant-anchor overlap)
10. Luan et al., "Permutation + ECC Watermarking," PoPETs 2025(4).
11. Chen et al., "FreMark: Free-Region Marking for Quantized LLMs," DSPP 2025, LNCS 16177. **Paywalled — overlap unverified.**
12. Li & Chang, "HPQIM," Warwick Technical Report.
13. Kumaresam et al., "Defensive Publication: Quantization Watermarking," TDCommons dpubs_series/10523, 2026.

---

## Appendix A: Claims Audit Summary

| Category | DIRECTLY_SUPPORTED | QUALIFIED | UNVERIFIED | PROHIBITED |
|----------|-------------------|-----------|------------|------------|
| Core Technical | 6 | 0 | 0 | 0 |
| Detectability | 4 | 2 | 1 | 0 |
| Utility | 3 | 0 | 0 | 2 |
| Integrity/Repro | 5 | 0 | 0 | 1 |
| Novelty/Prior Art | 5 | 0 | 1 | 3 |
| Robustness | 5 | 0 | 0 | 2 |
| Scope/Generality | 1 | 1 | 0 | 4 |
| Security | 1 | 1 | 0 | 3 |

Full audit: `manuscript_prep/CLAIMS_AUDIT.md`.

---

## Appendix B: Limitations & Threats Checklist

| Limitation | Addressed in Manuscript |
|------------|------------------------|
| Novelty (Gate G = PARTIAL) | §1.3, §6.1, References 3,4,7,9,11 |
| Model coverage (5 small models) | §4.1, §6.5 |
| Tensor coverage (6/1000) | §4.2, §6.5 |
| Payload size (≤50k) | §4.3, §6.5 |
| Requantization cycles (0–3 matrix; 0–5 regression; 0–10 single-tensor k_proj stress test) | §3.8, §5.1, §6.4 |
| Utility corpus (wikitext-2 only) | §4.4, §6.3 |
| Qwen LR discrepancy | §5.3, §6.2 |
| Small-cap exploratory | §5.4, §6.2 |
| Held-out neural (4 checkpoints) | §5.5, §6.2 |
| Detector diversity | §4.5, §6.2 |
| FreMark paywall | §2.2, §6.1 |

Full list: `manuscript_prep/LIMITATIONS_AND_THREATS.md`.

---

*End of manuscript draft v1. Prepared for internal review.*