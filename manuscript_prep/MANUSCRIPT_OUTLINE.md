# MANUSCRIPT_OUTLINE.md — Track C Manuscript Preparation

**Status:** Draft — for internal review only  
**Date:** 2026-10-21  
**Branch:** `research/contract-b-b14-and-evaluation` (commit `5b40120`)  
**Frozen codec:** `code/nes-llm/src/experiments/nf4_anchor_channel.py` (SHA-256: `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764`)

---

## Working Title Candidates

1. **Primary:** *Anchor-Preserving Payload Embedding in NF4-Quantized Language Model Weights Survives Requantization*
2. **Alternative:** *Exact Payload Recovery in NF4-Quantized LLM Weights Under Repeated Requantization via Anchor-Preserving Embedding*
3. **Short:** *Anchor-Preserving Payload Embedding in NF4-Quantized Weights*

---

## Abstract Structure (Target: ~250 words)

**Background:** NF4 quantization with double quantization is standard for LLM compression. Requantization (dequantize → requantize) perturbs block scales and code assignments, breaking most payload embedding schemes.

**Question:** Can a payload be embedded into NF4 weight codes such that it is exactly recoverable after repeated dequantize/requantize cycles, while preserving full-model utility?

**Method:** *interior_nearest* — restrict carriers to interior NF4 codes (1–14), never rewriting block endpoints (0, 15). Embed by moving each carrier to the nearest interior code with the target parity bit. Framing: payload + 128-bit MAGIC/length/checksum.

**Evidence:** 5 model architectures (TinyLlama, Qwen2.5, StableLM, Pythia, Falcon3), 6 tensors/model, 5 payload sizes (1k–50k bits), 3 seeds, 4 requantization cycles (0–3). 2,160 roundtrip conditions, all exact recovery (BER=0). 24 utility conditions (4 expansion models × 2 sizes × 3 seeds), all paired 95% CIs within ±0.5% PPL. Detectability at chance (grouped CV AUC ≤ 0.512). Small-capacity supplement: LR AUC 0.534 (p=0.005) on 4 groups — weak signal, labeled exploratory.

**Prior Art:** Nearest-same-parity code-index embedding is NID (Liu et al. 2016 / CNV-QIM 2008). Invariant-region framing established (HPQIM, Li 2023, FreMark 2025). Contribution: NF4 packed-code formulation + explicit endpoint exclusion + repeated-requantization exactness.

**Limitations:** 5 models, 1 tensor/layer, ≤50k bits, 3 requantization cycles (matrix), 5 cycles (stress). No generation quality evaluation. Novelty is narrow NF4-specific formulation + repeated-requantization exactness, not new embedding principle. Gate G (novelty) = PARTIAL; FreMark 2025 full text paywalled.

---

## Paper Organization

1. **Introduction** — problem, motivation, threat model (requantization as adversary)
2. **Related Work & Prior Art** — QIM, NID/CNV-QIM, EmMark, R-QIM, DeepWatermark, FreMark, Li 2023, HPQIM. Explicitly state: no "first" claim.
3. **Method** — NF4 layout, interior eligibility, nearest-same-parity mapping, framing, extraction. Contrast with `pair_parity` baseline.
4. **Experimental Setup** — models, revisions, tensors, payload sizes, seeds, cycles, codecs, corpus, compute.
4.1 **Roundtrip Robustness** — cycles 0–3, 5 sizes, 3 seeds, 6 tensors, 4 models. 360/360 exact per model.
4.2 **Detectability** — grouped CV (model::tensor), 16 features, 3 detectors, small-cap supplement, held-out neural.
4.3 **Utility** — wikitext-2, paired PPL, paired 95% CI, ±0.5% margin.
4.5 **Reproducibility** — regression tests, resumable runs, frozen codec hash.
5. **Discussion** — scope, limitations (models, tensors, cycles, corpus, small-cap signal, Qwen LR AUC), prior art context, Gate G = PARTIAL.
6. **Conclusion** — narrow NF4-specific contribution, not new embedding principle.

---

## Key Tables & Figures (Pre-computed)

| Table | Source | Status |
|-------|--------|--------|
| Roundtrip exact recovery (5 models × 6 tensors × 5 sizes × 3 seeds × 4 cycles) | `expansion_audit_table.csv` / `stablelm_qualification/roundtrip_stream_v1/` | ✅ Verified |
| Detectability (per model, 3 detectors, grouped CV) | `expansion_qualification/*/detectability_v1/` + StableLM | ✅ Verified |
| Small-cap supplement (14 groups) | `out/detectability_smallcap_v1/supplement_results.json` | ✅ Verified |
| Held-out neural (4 checkpoints, LOO) | `/tmp/nes_local_run/out/detectability_heldout_neural/` | Need to re-run / recover |
| Utility (paired PPL, 24 conditions) | `expansion_qualification/*/utility_v1/comparisons.json` | ✅ Verified |
| Regression tests (12/12 per model) | `test_codec_regressions_local.py` | ✅ Verified |

---

## Open Items Before Submission

- [ ] Re-run held-out neural detector (4 checkpoints, LOO) — output was in `/tmp` and lost
- [ ] Re-run small-capacity pooled analysis (14 groups) — verify LR AUC 0.5059 CI
- [ ] Obtain FreMark 2025 full text (LNCS 16177) or document paywall limitation
- [ ] Pin Qwen2.5-3B historical revision (currently UNVERIFIED)
- [ ] Obtain FreMark 2025 full text or formally acknowledge paywall gap
- [ ] Verify StableLM utility 296k tokens/condition (Gate E)

---

## Files to Create in `manuscript_prep/`

- [ ] `MANUSCRIPT_OUTLINE.md` (this file)
- [ ] `METHODS.md`
- [ ] `RESULTS_EVIDENCE_TABLE.md`
- [ ] `REPRODUCIBILITY.md`
- [ ] `LIMITATIONS_AND_THREATS.md`
- [ ] `CLAIMS_AUDIT.md`
- [ ] `OPEN_GAPS.md`
- [ ] `CLAIMS_AUDIT.md` → for claims audit
- [ ] `PRIOR_ART_VERIFICATION.md` — verify each citation