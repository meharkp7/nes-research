# DECISION REQUIRED

**RESOLVED 2026-10-19 — Option A approved by the user.** The V2 amendment was
created as `QUALIFICATION_PROTOCOL_V2.md` (active) and the V1 protocol
`stablelm_qualification/QUALIFICATION_PROTOCOL_STABLELM.md` was preserved
unchanged (sha256 `581bd57e28c0768829386e0a0230a527d212eba2612e8197cf75b09dba68de87`).
Expansion now proceeds under V2 with Track B (novelty) disclosed as PARTIAL.

Created 2026-10-19. This document requests one explicit decision. Nothing is
committed, pushed, deleted, or overwritten. The frozen method and protocol are
untouched. No additional-model experiment has been started.

---

## 1. Current status

| Gate | Meaning | Status |
|---|---|---|
| A — Implementation/artifact integrity | codec hash, hashes, NF4 config, only intended tensor | **PASS** |
| B — Exact end-to-end recovery (fresh process) | BER 0, checksum ok, carrier agreement 1 | **PASS** (360/360) |
| C — Multi-sample/transformation robustness | cycles 0–3 × 5 sizes × 3 seeds × 6 tensors | **PASS** (360/360) |
| D — Detectability screening | AUC ≤ 0.55, TPR@1% ≤ 0.02, clean null | **PASS** (AUC ≈ 0.500) |
| E — Utility ±0.5% equivalence | paired 95% CI inside ±0.5% rel. PPL | **PASS** (6/6) |
| F — Reproducibility/reporting | rerunnable, hashed, failures recorded | **PASS** |
| G — Novelty/prior-art integrity | closest prior art compared | **PARTIAL** |

Verified this session: frozen codec sha256 `07ce008d…b764` (unchanged); protocol
file untouched (mtime 2026-10-17); repo `/workspace/project` clean at `aa7e1a7`.
No document describes expansion as authorized under the current protocol.

## 2. Why the original protocol currently locks expansion

`QUALIFICATION_PROTOCOL_STABLELM.md` §3: "Additional-model expansion is **locked**
unless every gate A–G is PASS." Gate G is PARTIAL, so expansion is locked. That
is the protocol working as written, not a defect.

The lock exists because V1 bundles two different questions — *does it work*
(technical) and *is it novel* (literature) — behind a single all-gates-pass rule.
An unresolved literature question therefore blocks empirical work.

## 3. Strongest and weakest parts of the novelty argument

**Strongest (what actually supports a contribution):**
- Exact, byte-idempotent recovery across **repeated dequantize/requantize cycles**,
  with working negative controls (`pair_parity`/`nearest_parity` fail from cycle 1),
  demonstrated across three architectures.
- A **causal ablation (Arm E)** showing the nearest-same-parity *operation* is
  operative, not the eligibility mask (interior mask + baseline op → BER 0.4913).
- **Detectability** screening at chance on large tensors, with an honestly-reported
  weak small-capacity signal (not suppressed).
- **Utility** as a predeclared ±0.5% equivalence test (Gate E PASS), a form most
  prior weight-watermarking papers do not report.

**Weakest (what cannot be claimed):**
- The core operation is a **named published technique**: NID (Liu et al. 2016) /
  CNV-QIM (Xiao et al. 2008) — code-index parity partition + move to the neighbour
  of the opposite sub-codebook. QIM (2001) on DNN weights (DeepWatermark 2020;
  R-QIM 2023) is also prior.
- The **invariant-region** framing predates the work (Li 2023 int8; FreMark 2025
  DCT; HPQIM image).
- **FreMark full text is inaccessible** (paywalled), so that comparison is
  incomplete.
- Endpoint exclusion's benefit is supported by **one ablation cell** on one
  model/tensor; not yet shown against a matched NID-on-NF4 baseline.

Net: the contribution is **narrow** — an NF4-specific formulation plus an
empirical repeated-requantization evaluation — not a new embedding principle.

## 4. Proposed protocol amendment

`QUALIFICATION_PROTOCOL_V2_PROPOSAL.md` (proposal only; V1 preserved). It splits
gates into **Track A (technical generalization)** and **Track B (novelty/prior-art
qualification)**, states that passing one track does not prove the other, keeps
every V1 criterion and threshold unchanged, and lists five labelled proposals
(P-1…P-5) each requiring explicit approval. It does not edit the frozen protocol.

## 5. Recommended narrow research framing

From `CONTRIBUTION_SCOPE_ASSESSMENT.md`: adopt **Framing 3** —
"An empirical robustness and evaluation study of interior-code NF4 embedding
under repeated requantization" — with **Framing 2** (NF4-specific engineering
formulation, endpoint exclusion) as a scoped sub-result pending a matched
baseline. Reject Framing 1 (new principle). State the NID/QIM overlap in related
work. Make no "first"/"unprecedented" claim.

## 6. Proposed additional-model plan

`ADDITIONAL_MODEL_EXPANSION_PLAN.md`: two recommended architectures
(SmolLM2-1.7B, Qwen2.5-1.5B) and two options (pythia-1.4b, Falcon3-1B-Base),
metadata verified via the HF API on 2026-10-19. Binding constraint: **5.9 GB free
disk**, so models are processed one at a time or the set is reduced to 2. Per-model
results are preserved; failures are reported as counterexamples, never pooled away.
The plan states plainly that more models expand evidence and do **not** establish
universal generalization or improve Track B.

## 7. The exact decision that requires approval

Please choose one:

- **(A) Approve the V2 amendment** (`QUALIFICATION_PROTOCOL_V2_PROPOSAL.md`),
  which would unlock expansion on Track A PASS with the novelty limitation
  disclosed. If approved, I will create a new frozen `QUALIFICATION_PROTOCOL_V2.md`
  (not an in-place edit) and then proceed to expansion.
- **(B) Keep V1 and expand anyway**, explicitly accepting that this contradicts
  V1 §3 — only if you want the empirical work to proceed without amending the
  protocol; I will record the exception in the report.
- **(C) Keep V1 and do not expand.** Continue only non-experimental work
  (manuscript drafting, matched-baseline experiment design, prior-art follow-up
  on FreMark).
- **(D) Something else** you specify.

I will not activate the amendment, download models, or start experiments until
you choose. Recommended: **(A)**, because it preserves the scientific separation
of "works" vs "novel" while disclosing the novelty limitation honestly.
