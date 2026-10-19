# Contribution-Scope Assessment — NF4 anchor-preserving embedding

Created 2026-10-19. Read-only assessment. No claim of completeness of the
literature; conclusions are scoped to the sources actually read, listed in
`overnight_third_model_prior_art/PRIOR_ART_REVIEW.md` and
`PRIOR_ART_EVIDENCE_GATE_G_20261019.md`.

Method under assessment: frozen `interior_nearest` NF4 codec
(`nf4_anchor_channel.py`, sha256 `07ce008d…`). Track 3 only; not pooled with the
five residual methods or QSE/DCE.

**Standing constraint (stated up front).** The nearest-neighbour same-parity
operation on quantized code indices is **not** novel. It is a named, published
technique (NID — Liu et al. 2016; CNV-QIM — Xiao et al. 2008) and an instance of
QIM (Chen & Wornell 2001) applied to DNN weights (DeepWatermark 2020; R-QIM
2023). This assessment does not dispute that and does not claim otherwise.

The question is whether the *remaining* elements — endpoint (0/15) exclusion,
stable carrier eligibility, exact recovery over repeated requantization, plus the
detectability and utility results — collectively support a *useful* contribution.

---

## Framing 1 — "A new embedding principle"

### Supporting evidence
- None that survives the prior-art review. The carrier domain (code indices),
  the partition rule (parity/label), and the move rule (nearest neighbour of the
  opposite sub-codebook) each have direct antecedents.

### Prior-art overlap
- **Fatal.** NID (2016) partitions codebook indices by label and moves the
  carrier to the neighbour index of the opposite sub-codebook. QIM (2001) is the
  general principle. This framing would be contradicted by the project's own
  prior-art evidence.

### Unsupported claims to avoid
- "New embedding principle," "first," "unprecedented," "novel method."
- Any framing that presents the parity move as the contribution.

### Remaining experiments
- None would rescue this framing. More experiments cannot create novelty.

### Likely reviewer objections
- "This is NID/QIM on a different carrier." (Decisive.)
- "The authors' own related-work section shows the mechanism predates the work."

### Verdict
**Reject.** Not defensible. Recommending it would be inventing novelty.

---

## Framing 2 — "An NF4-specific engineering formulation of an established quantization-index embedding family"

### Supporting evidence
- The method operates on **NF4 packed code indices** (blocksize 64, nested
  statistics), a specific and widely deployed representation (QLoRA/bitsandbytes).
  No source read targets NF4 packed codes specifically.
- **Explicit endpoint (0/15) exclusion**: NID/CNV/EmMark do not state endpoint
  exclusion as a mechanism. In NF4, codes 0/15 are the per-block absmax extremes;
  excluding them is a concrete, testable design choice.
- **Causal ablation (Arm E)** isolates the operation from the eligibility mask:
  interior eligibility with the baseline op fails (BER 0.4913), so the
  nearest-same-parity *operation* is operative, not the mask alone. This is a
  measured engineering finding, not a novelty claim.
- **Stable carrier eligibility** under requantization is measured (carrier
  agreement 1.0 across cycles).

### Prior-art overlap
- The *family* (quantization-index embedding) is fully established. This framing
  does not contest that. The claim is a **formulation**, not a principle.

### Unsupported claims to avoid
- Presenting the formulation as a new method family.
- Claiming endpoint exclusion is *necessary* for robustness: Arm C shows the
  anchorless-block exclusion is neither necessary nor sufficient (ledger claim 10).
- Claiming general NF4 superiority without testing other 4-bit formats (FP4, any4).

### Remaining experiments
- Head-to-head against a NID-style baseline *on NF4 codes* (parity partition
  without endpoint exclusion) at matched capacity — to quantify what endpoint
  exclusion buys, if anything.
- Comparison across 4-bit formats (NF4 vs FP4) to show the formulation is
  NF4-specific in a meaningful way.
- Causal ablation extended to a second model/tensor (Arm E currently one cell).

### Likely reviewer objections
- "The 'engineering formulation' is a thin delta over NID; endpoint exclusion may
  be incidental."
- "Why NF4 specifically? Show it fails or degrades on FP4."
- "The ablation is one model/tensor."

### What would make it stronger / weaker
- **Stronger:** a matched NID-on-NF4 baseline showing endpoint exclusion changes
  a measured outcome; replication of Arm E across models.
- **Weaker:** if a NID-on-NF4 baseline matches results without endpoint
  exclusion, the delta collapses and only Framing 3 remains.

### Verdict
**Defensible if and only if** the endpoint-exclusion delta is measured against a
matched baseline. Currently supported by a single-cell ablation. Promising but
incomplete.

---

## Framing 3 — "An empirical robustness and evaluation study of interior-code NF4 embedding under repeated requantization"

### Supporting evidence (strongest of the three)
- **Exact recovery across repeated requantization:** cycles 0–3 (0–5 stress),
  across 6 tensors × 5 sizes × 3 seeds on StableLM (360/360 exact, BER 0),
  plus TinyLlama and Qwen matrices. Byte-idempotent (one distinct artifact hash
  per condition across cycles).
- **Baselines fail:** `pair_parity`/`nearest_parity` fail from cycle 1
  (checksum mismatch) — the study has a working negative control.
- **Detectability:** grouped detectors at chance on large tensors (AUC ≈ 0.500);
  a weak, honestly-reported small-capacity signal (AUC 0.534, p = 0.005, 4
  groups). Reporting the weak signal rather than hiding it is a study strength.
- **Utility:** StableLM Gate E PASS — 6/6 embedded conditions with paired 95%
  CIs inside the predeclared ±0.5% margin (max effect 0.0139%).
- **Three architectures** qualified with independent-process extraction.
- **Negative and inconclusive results retained** (Qwen provenance UNVERIFIED;
  small-capacity signal not suppressed).

### Prior-art overlap
- The *mechanism* overlaps NID/QIM. This framing **concedes** that and positions
  the contribution as the **evaluation**: repeated-requantization exactness,
  detectability screening, and paired-PPL equivalence on NF4 packed codes. The
  review found no prior source reporting repeated-requantization exactness or a
  paired-PPL equivalence test for this family.

### Unsupported claims to avoid
- **Undetectability** (only a screening pass; 6 groups; the weak signal is real).
- **Universal robustness / generalization** (three models, ≤50k bits, ≤3 cycles
  in the matrix).
- **Semantic / generation equivalence** (PPL only).
- **Absolute novelty** of the method.
- Any claim that the evaluation is exhaustive of prior work.

### Remaining experiments
- Extend the matrix to more cycles (beyond 3) and larger payloads to test where
  exactness breaks — an honest limit-finding study.
- Add a fourth+ architecture (see `ADDITIONAL_MODEL_EXPANSION_PLAN.md`) to widen
  the empirical base.
- Report the small-capacity detectability signal with more independent groups to
  firm up or bound it.
- Utility on additional tensors/layers and a second corpus.

### Likely reviewer objections
- "The evaluation is on one tensor per model; is it representative?"
- "Detectability is a screening pass, not a security claim — state that."
- "Three models is not 'universal'; scope the title and claims accordingly."

### What would make it stronger / weaker
- **Stronger:** per-model (not pooled) results across more architectures; a
  documented exactness limit; more detectability groups; per-model utility CIs.
- **Weaker:** if another NF4 codec trivially matches the exactness result, the
  empirical novelty shrinks (the study becomes a reproduction).

### Verdict
**Recommended.** This is the framing the current evidence actually supports. It
concedes the mechanism is established and claims a bounded empirical
contribution. It does not require inventing novelty and survives the NID overlap.

---

## Collective assessment: do the remaining elements support a useful contribution?

Yes, **conditionally**, under Framing 3 (with Framing 2 as a supporting angle):

1. **Endpoint exclusion** — a concrete NF4-specific design choice, currently
   supported by a single-cell causal ablation; useful *if* measured against a
   matched baseline.
2. **Stable carrier eligibility** — measured (agreement 1.0), a supporting
   engineering result.
3. **Exact recovery over repeated requantization** — the strongest element;
   demonstrated with working negative controls across three models.
4. **Detectability** — a screening pass with an honestly-reported weak signal;
   bounds the claim rather than inflating it.
5. **Utility** — a properly predeclared equivalence test (Gate E PASS), which
   most prior weight-watermarking papers do not report in this form.

**Collectively** these support a **narrow empirical contribution**: a
characterization of when an established quantization-index embedding survives
repeated NF4 requantization exactly, with detectability and utility bounds. They
do **not** support a new-principle claim, and no element individually is novel.

**Recommendation:** adopt Framing 3 as the manuscript's primary claim; present
Framing 2 (endpoint exclusion) as a scoped sub-result pending the matched-baseline
experiment; state the NID/QIM overlap explicitly in related work; make no
"first"/"unprecedented" statement.
