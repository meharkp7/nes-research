# Qualification Protocol V2 — PROPOSAL (NOT ACTIVE)

**Status: PROPOSAL. Not activated. Requires explicit approval.**
Created 2026-10-19. This document does **not** modify, supersede, or reinterpret
`QUALIFICATION_PROTOCOL_STABLELM.md` (frozen 2026-10-17). The frozen protocol
remains in force and in place until the user explicitly approves a change.

Nothing here relaxes, removes, or reinterprets an existing acceptance criterion.
Every change relative to V1 is listed in §5 and labelled a proposal.

---

## 1. Why a V2 is proposed

The frozen V1 protocol answers one question well: *does the method technically
work, robustly, with acceptable detectability and utility?* It bundles that
question with a second, different question: *is the method novel?* V1 §3 then
gates expansion on **all** gates A–G passing, so an unresolved literature
question (Gate G) blocks empirical work that has nothing to do with literature.

The current state exposes the conflation:

- Gates A–F are **PASS** (technical generalization is demonstrated within scope).
- Gate G is **PARTIAL** (novelty is narrow; a named prior method, NID, covers the
  core operation).

Under V1 §3, expansion is locked. That is correct *as V1 is written*. V2 proposes
to separate the two questions into two tracks so each can be judged on its own
evidence, without weakening either.

## 2. Track A — Technical generalization (must stand on its own)

A method may be technically qualified independent of whether it is novel. Track A
keeps every V1 criterion, unchanged:

- **A1 Independent-process embedding and recovery.** Embed in one process; recover
  in a separate process from the on-disk artifact using only
  (artifact, key, tensor name, method). No hidden in-memory state.
- **A2 Exact recovery verification.** BER == 0 and payload checksum == original;
  recovered length == original; framing/truncation errors absent.
- **A3 Repeated dequantize/requantize robustness.** Cycles 0–3 in the primary
  matrix; 0–5 as a separately reported stress test. Failures retained.
- **A4 Multiple tensors, payload sizes, seeds.** The predeclared six tensors
  (early/mid/late × q/k/v or up/down/gate), the five payload sizes
  (1k/5k/10k/20k/50k), three seeds.
- **A5 Detectability.** The 16 predeclared block features; grouped splits
  (`model::tensor`, GroupKFold / leave-one-tensor-out); normalization fit on
  train only; LR + GB + exploratory MLP; AUC with grouped-bootstrap CI; TPR@1%/5%
  FPR; Cohen's d; clean-vs-clean control; permutation test; group counts. Practical
  thresholds retained: AUC ≤ 0.55, TPR@1% ≤ 0.02, clean control ~0.5. These are
  screening thresholds, **not** a definition of undetectability.
- **A6 Utility.** wikitext-2-raw-v1 test, each model's own tokenizer, context 512,
  non-overlapping windows, paired sequence-level evaluation, token-weighted PPL.
  Predeclared equivalence margin **±0.5% relative PPL**, PASS only if the paired
  95% CI fits entirely inside the margin. CIs must be *adequately informative*:
  if the achievable CI cannot fit the margin, report **INCONCLUSIVE**, never
  widen the margin.
- **A7 Integrity, provenance, reproducibility, reporting.** Codec hash verified;
  model revision pinned; immutable manifests; all failures/exclusions/deviations
  recorded; measured vs derived vs speculation distinguished; results per-model
  and never hidden in pooled averages.

Track A is judged PASS only when A1–A7 hold for the declared scope, and the scope
is stated with the result.

## 3. Track B — Novelty and prior-art qualification

Track B is a **separate** qualification, judged on literature evidence:

- **B1 NID / CNV-QIM similarity.** The code-index parity-partition +
  neighbor-of-opposite-subcodebook operation is a named published method
  (Liu et al. 2016; Xiao et al. 2008). Record the exact overlap.
- **B2 QIM-on-DNN-weights similarity.** QIM applied to sampled NN weights
  (DeepWatermark 2020; R-QIM 2023). Record the exact overlap.
- **B3 Quantization-invariant-region methods.** Li et al. 2023 (int8), FreMark
  2025 (DCT), HPQIM (image). Record the overlap.
- **B4 Inaccessible primary sources.** FreMark (DSPP 2025, LNCS 16177) full text
  is paywalled; record this as a known gap, not a resolved question.
- **B5 Precise scope of the potential NF4-specific contribution.** State exactly
  what remains unmatched (NF4 packed code indices; explicit endpoint 0/15
  exclusion; exact recovery over repeated requantization; detectability + paired
  PPL reporting) and what does not (the embedding principle).

Track B PASS means: the overlap is documented honestly, the residual scope is
stated narrowly, and no "first"/"unprecedented" claim is made. Track B **cannot**
be made PASS by asserting novelty; it can only be *qualified*.

## 4. The independence of the two tracks

This is the central clarification V2 makes:

- **Passing Track A does not prove Track B.** A method can be technically
  flawless and still not novel. Technical qualification is not a novelty claim.
- **Passing Track B does not replace technical validation.** A genuinely novel
  idea that fails recovery, robustness, detectability, or utility is not a
  publishable result. Literature qualification is not a substitute for evidence.
- **Neither track's thresholds may be weakened after seeing results.**

A paper may report Track A results with Track B disclosed as a bounded limitation.
It may **not** convert a Track A pass into a novelty claim.

## 5. Proposed changes vs V1 (each requires explicit approval)

| # | Change | V1 | V2 proposal | Justification | Approval needed |
|---|---|---|---|---|---|
| P-1 | Split gates into two tracks | Gates A–G in one list | Track A (technical) + Track B (novelty) | Separates two different questions so each is judged on its own evidence | **YES** |
| P-2 | Expansion gating | §3: expansion locked unless **all** A–G PASS | Expansion unlocked when **Track A** PASS, with **Track B disclosed** as a bounded limitation | Empirical work should not be blocked by an unresolved literature question; the novelty limitation is disclosed, not hidden | **YES** |
| P-3 | Gate G status semantics | PASS / PARTIAL / FAIL | Track B is *qualified* or *unqualified*; "PARTIAL" is expected when primary sources are inaccessible | An inaccessible paywalled paper is a documentation gap, not a failed method | **YES** |
| P-4 | New-model novelty gating | Implicit | Each new model inherits the same Track B disclosure; expansion does not strengthen Track B | Prevents "more models" from being misread as "more novel" | **YES** |
| P-5 | Reporting requirement | — | Every result must state Track A scope and Track B disclosure together | Keeps the two tracks visibly separate in every artifact | **YES** |

**No V1 acceptance criterion is changed.** P-1…P-5 add structure and a disclosure
requirement; they do not lower any threshold, margin, or pass rule.

## 6. What V2 explicitly does NOT do

- It does not delete or edit `QUALIFICATION_PROTOCOL_STABLELM.md`.
- It does not relabel Gate G from PARTIAL to PASS.
- It does not authorize expansion by itself. It is a proposal.
- It does not weaken the ±0.5% utility margin, the detectability thresholds, or
  the exact-recovery requirement.
- It does not permit pooling results across models to hide a failure.

## 7. Activation

V2 takes effect **only** on explicit user approval. Until then, V1 §3 governs and
expansion remains locked. If approved, a new frozen file
`QUALIFICATION_PROTOCOL_V2.md` will be created (not an in-place edit), with the
V1 file preserved as historical evidence.
