# Paper Outline — NF4 Anchor-Preserving Weight Payload Embedding

Working title: *Anchor-Preserving Payload Embedding in NF4-Quantized Language
Model Weights Survives Requantization.*

Scope note: this outline covers **only** the `interior_nearest` method and its
comparison baselines (`pair_parity`, `nearest_parity`, plus the A–E ablation
arms). It excludes the five residual methods and the QSE/DCE packed-NF4 track.

---

## 1. Problem and motivation
- Deployed LLMs are increasingly stored in 4-bit formats (NF4 + double quant);
  weights are redistributed and requantized in practice.
- Weight watermarking / payload embedding could support provenance, licensing,
  or integrity signaling.
- Core difficulty: a payload written into packed 4-bit codes must survive
  **requantization** (dequantize → requantize), which perturbs block scales and
  code assignments.
- Question: can a payload be embedded such that it is exactly recoverable after
  requantization, while leaving full-model behavior essentially unchanged?

## 2. Prior work and novelty question
- **Status: NOT REVIEWED in this workspace.** No literature review or prior-art
  search exists in the repository or run root.
- Must be completed before any novelty claim (see `MANUSCRIPT_READINESS.md`,
  blocker P1). Candidate areas to survey: LLM watermarking, weight
  steganography, quantization-robust watermarking, NF4/QLoRA quantization
  internals, error-correcting codes for quantized weights.
- Novelty question to answer: is "anchor preservation" (never rewriting block
  endpoint codes) an already-described technique under other terminology?

## 3. Method and encoding/decoding procedure
- NF4 layout: blocks of 64 codes, nested (double) quantization, block endpoints
  0 and 15.
- Carrier eligibility: interior codes 1–14 only.
- Embedding operation: **nearest same-parity interior mapping** — each carrier
  is moved to the nearest interior code with the target parity bit; endpoints
  are never selected and never written.
- Framing: payload + 128 framing bits (magic, length, checksum).
- Decoding: recompute carrier set, read parity bits, verify checksum.
- Contrast with the baseline `code & 14 | bit` parity operation, which can push
  carriers onto endpoints.
- Note the precise mechanism claim: robustness comes from the **operation**,
  not from the interior eligibility mask or the anchorless-block exclusion
  (evidence: Arm E).

## 4. Experimental setup
- Models: TinyLlama-1.1B-Chat-v1.0 (revision pinned), Qwen2.5-3B (revision
  UNVERIFIED — stated as a caveat).
- Frozen codec hash; blocksize 64; nested True; CPU-only.
- Tensors: six per model (attention q/k/v, MLP up/down/gate; early/mid/late).
- Payloads: 1,000–50,000 bits; seeds 20261016/017/018; cycles 0–3.
- Baselines: `pair_parity`, `nearest_parity`; A–E causal ablation.

## 5. Recovery and requantization robustness
- 240 capacity rows, 0 interior failures; byte-idempotent across cycles.
- 6/6 exact per model on the tensor matrix; anchors changed 0.
- A–E ablation: operation vs eligibility decomposition; Arm E failure (BER
  0.4913) isolates the operation.
- Baseline failure: pair_parity 0/6 per model.

## 6. Capacity
- Analytic bound = eligible interior codes − 128 framing bits.
- TinyLlama q_proj ≈ 4.09 M bits; Qwen q_proj ≈ 4.05 M bits; smallest tested
  ≈ 0.51 M bits.
- Carrier utilization 0.03%–1.2% at tested sizes.

## 7. Detectability and limitations
- Grouped-CV detectors at chance on the large-tensor setting; ungrouped 0.697
  shown to be leakage (retracted); grouped oracle 0.416.
- Small-capacity supplement: LR AUC 0.534, p = 0.005, but only 4 groups and
  weak practical power — presented as exploratory, **not** as a detector.
- Explicit non-claim: undetectability is not established.

## 8. Utility / perplexity
- Full-model PPL on wikitext-2-raw test; 15,330 tokens/condition.
- All Δ within ±0.06%; all paired 95% CIs include 0.
- Reduced scope stated; no generation-quality claim.

## 9. Threat model, limitations, and conclusion
- Threat model: adversary who can inspect weights and requantize; the payload is
  a covert/provenance channel, not a cryptographic primitive.
- Limitations: 2 models, 2 tensor sets, ≤50k bits, 3 cycles, no literature
  review, reduced PPL scope, weak small-capacity signal.
- Conclusion: within the tested scope, anchor-preserving interior embedding is
  exactly recoverable after requantization, generalizes across the tested
  tensors and two models, is at chance for grouped steganalysis on large
  tensors, and preserves perplexity within ±0.06%; universal robustness,
  undetectability, semantic equivalence, and novelty are **not** claimed.

---

## Addendum (2026-10-18) — third model and prior-art status

Sections 2 and 4–8 are updated as follows. Original text preserved above.

- **§2 Prior work:** now **reviewed** (was "NOT REVIEWED"). See
  `overnight_third_model_prior_art/PRIOR_ART_REVIEW.md` + Addenda A/B. Closest
  matches: QIM (Chen & Wornell 2001), R-QIM (arXiv 2305.17879, full text),
  EmMark (DAC 2024), Li et al. 2023, Luan et al. 2025, FreMark (DSPP 2025,
  abstract-level). Novelty verdict: **unresolved, leaning narrow-and-potentially
  -novel**; the parity step is QIM, and FreMark's interval-constrained embedding
  is a conceptual cousin of endpoint preservation. No "first" claim.
- **§4 Models:** now **three** architectures. Add StableLM-2-1.6B
  (`stabilityai/stablelm-2-1_6b` @ `f499ead7...`, revision pinned).
- **§5 Robustness:** StableLM adds 360/360 exact, anchors changed 0, byte
  -idempotent, baselines 93/360. Total across three models: interior 0 failures
  under the tested matrices.
- **§7 Detectability:** StableLM screening pass — GB AUC 0.5005, LR 0.5009,
  MLP 0.5000, 6 groups, permutation p 0.32–0.48, clean-vs-clean 0.4848. The
  small-capacity signal (AUC 0.534, p 0.005, 4 groups) is preserved unchanged.
- **§8 Utility:** StableLM run at 296,380 tokens/condition (3 seeds × 2 sizes),
  vs the prior 15,330. Gate E result reported in
  `stablelm_qualification/STABLELM_QUALIFICATION_REPORT.md`.
- **§9 Limitations:** now three models (not two); StableLM detectability has only
  6 independent groups; FreMark full text still unread; capacity ceiling still
  unmeasured; cycles still ≤3 in the matrix.

## Addendum 2 (2026-10-19) — Gate E closed; novelty narrowed

- **§2 Prior work / novelty:** further narrowed. Addendum C of the prior-art
  review identifies **NID (Neighbor-Index-Division, Liu et al. 2016)** — code-index
  parity partition plus "move to the neighbor index of the opposite sub-codebook"
  — as the *same operation* as `interior_nearest`, and HPQIM / Li-2023 / FreMark
  as established invariant-region framing. Verdict: the contribution is a
  **narrow NF4-specific formulation + a repeated-requantization exactness result**,
  not a new embedding principle. The manuscript must say so explicitly; no
  "first"/"unprecedented" sentence. Evidence:
  `overnight_third_model_prior_art/PRIOR_ART_EVIDENCE_GATE_G_20261019.md`.
- **§8 Utility:** Gate E for StableLM is now **PASS** — 6/6 embedded conditions
  with paired 95% CIs inside the predeclared ±0.5% margin (max effect 0.0139%).
  Table: `stablelm_qualification/tables/table4_utility_stablelm.csv`.
- **Expansion:** still **locked** per protocol §3 (Gate G not PASS). Expansion is
  a user decision, not an automatic next step.

