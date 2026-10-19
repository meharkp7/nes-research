# Qualification Protocol V2 — Multi-Model (`interior_nearest` NF4)

**Status: ACTIVE.** Approved by the user on 2026-10-19 (Option A of
`DECISION_REQUIRED.md`). This is a **new versioned file**; it does not edit
`stablelm_qualification/QUALIFICATION_PROTOCOL_STABLELM.md` (V1), which is
preserved unchanged as historical evidence.

Frozen: 2026-10-19, before any additional-model outcome was inspected. The
predeclared payload sizes, seeds, cycles, tensors, thresholds, and margins below
are fixed for all models and are not tuned after seeing results.

Method: frozen codec `code/nes-llm/src/experiments/nf4_anchor_channel.py`, sha256
`07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` (must match at
every step; never modified). NF4, blocksize 64, nested statistics True. Track 3
only (`interior_nearest`; baselines `pair_parity`, `nearest_parity`). The five
residual methods and QSE/DCE tracks remain out of scope and are never pooled with
these results.

## Amendment record

| Field | Value |
|---|---|
| Supersedes (for new work) | `stablelm_qualification/QUALIFICATION_PROTOCOL_STABLELM.md` (V1, frozen 2026-10-17) |
| V1 status | **Preserved, unchanged, still the governing record for the StableLM qualification already reported** |
| Amendment proposal | `QUALIFICATION_PROTOCOL_V2_PROPOSAL.md` |
| Approved | 2026-10-19, user, Option A (`DECISION_REQUIRED.md` §7) |
| Nature of change | Structural: split gates into Track A (technical) and Track B (novelty); expansion gating tied to Track A with Track B disclosed. **No acceptance threshold, margin, or pass rule is changed.** |

## 0. What a gate status means (unchanged from V1)

Four distinct questions are tracked separately and never conflated:

1. **Executed correctly** — the experiment ran and produced inspectable output.
2. **Technically passed** — the method met the numeric criterion.
3. **Hypothesis supported** — the scientific claim is supported by the result.
4. **Sufficient to generalize** — the evidence supports extending the claim.

A script finishing is *executed correctly*, not *passed*. Every gate is reported
as PASS / FAIL / INCONCLUSIVE / NOT RUN with a reason.

## 1. Frozen constants (unchanged from V1)

| Item | Value |
|---|---|
| Codec sha256 | `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` (frozen) |
| Blocksize / nested | 64 / True |
| Payload sizes | 1,000 / 5,000 / 10,000 / 20,000 / 50,000 bits |
| Seeds | 20261016, 20261017, 20261018 |
| Requantization cycles | 0–3 primary; 0–5 stress (separately reported) |
| Primary tensor | layer-0 `self_attn.q_proj.weight` |
| Matrix tensors | six predeclared tensors (early/mid/late × q/k/v or up/down/gate) |
| Framing | B1.4: MAGIC + 4-byte length + payload + 8-byte SHA-256 prefix (16 bytes) |
| Utility margin | ±0.5% relative PPL (paired 95% CI must fit inside) |
| Detectability thresholds | grouped AUC ≤ 0.55; TPR@1% FPR ≤ 0.02; clean control ~0.5 |

Model-specific constants (model ID, revision, clean NF4 checkpoint path, exact
tensor names) are recorded per model in that model's manifest and provenance table.

---

# TRACK A — Technical generalization

Track A is judged on technical evidence only, independent of novelty. It is
declared PASS only for the **precise tested scope**, which must be stated with the
result. Track A passing is **not** a novelty claim.

### A1 — Implementation and artifact integrity
Codec hash matches at every step; source/output hashes recorded; intended NF4
config; **only the intended tensor differs** (`differing_keys == [<target>]`);
logs complete; all settings recorded; no hidden in-memory state needed for
extraction. Evidence: manifest + integrity JSON per run.

### A2 — Exact end-to-end recovery (independent process)
An embedded artifact saved to disk is loaded in a **fresh process**; extraction
reads the saved artifact, not an in-memory object; recovered length == original;
**BER == 0**; recovered checksum == original; carrier agreement == 1 where stable
carriers are required; no truncation/padding/framing failure.
Method: two-process protocol — process 1 embeds and saves; process 2 (separate
invocation) loads from disk and extracts with only (artifact, key, tensor name,
method). Key derived from seed via the documented `key_for`.

### A3 — Multi-sample and transformation robustness
Cycles 0–3 pass across the five payload sizes, three seeds, and the six tensors;
failures recorded not discarded; baseline (`pair_parity`/`nearest_parity`) run
under the same settings; idempotence and endpoint-change counts recorded.
Extension: cycles 0–5 stress test, reported separately.

### A4 — Detectability
Screening thresholds (practical, **not** a definition of undetectability):
- held-out grouped AUC ≤ 0.55
- TPR at 1% FPR ≤ 0.02
- clean-vs-clean control consistent with null (~0.5)
- no leakage/split-design flaw
- adequate independent groups

Protocol: the 16 predeclared block features; grouped splits by `model::tensor`
(GroupKFold, leave-one-tensor-out where groups allow); normalization fit on
train only; detectors = Logistic Regression, Gradient Boosting, exploratory MLP
(labeled exploratory); report AUC + grouped-bootstrap CI, TPR@1%/5% FPR, Cohen's
d effect sizes, clean-vs-clean AUC, permutation test, group counts, runtime.
Models are evaluated **separately** and never pooled across models.

### A5 — Utility (perplexity equivalence)
Corpus: `wikitext-2-raw-v1` test, each model's own tokenizer, context 512,
non-overlapping windows. Paired sequence-level evaluation; token-weighted PPL.
- **Predeclared practical equivalence margin: ±0.5% relative PPL.**
- PASS requires the **paired 95% CI to fit entirely inside ±0.5%**.
- A CI that merely contains 0 is NOT a pass.
- If the CPU budget cannot produce a CI inside ±0.5%, report **INCONCLUSIVE**.
- PPL alone cannot establish semantic/generation equivalence.

### A6 — Reproducibility and reporting
Rerunnable from recorded commands + frozen inputs; results linked to immutable
manifests/hashes; all failures/exclusions/deviations recorded; measured vs derived
vs speculation distinguished; **per-model results preserved** and never hidden in
a pooled average.

### Track A verdict rule
Track A PASS requires A1–A6 to hold for the declared scope. The scope (models,
tensors, sizes, seeds, cycles) is stated with the verdict. Track A is a
technical-generalization statement, not a novelty statement.

---

# TRACK B — Novelty and prior-art qualification

Track B is a **separate** qualification judged on literature evidence. It is
**qualified** or **unqualified**, not PASS/FAIL in the technical sense.

- **B1 NID / CNV-QIM similarity.** The code-index parity-partition +
  neighbour-of-opposite-subcodebook operation is a named published method
  (Liu et al. 2016; Xiao et al. 2008). Record the exact overlap.
- **B2 QIM-on-DNN-weights similarity.** QIM on sampled NN weights (DeepWatermark
  2020; R-QIM 2023). Record the exact overlap.
- **B3 Quantization-invariant-region methods.** Li et al. 2023 (int8), FreMark
  2025 (DCT), HPQIM (image). Record the overlap.
- **B4 Inaccessible primary sources.** FreMark (DSPP 2025, LNCS 16177) full text
  is paywalled. Record as a known gap, not a resolved question.
- **B5 Precise scope of the potential NF4-specific contribution.** State exactly
  what remains unmatched (NF4 packed code indices; explicit endpoint 0/15
  exclusion; exact recovery over repeated requantization; detectability + paired
  PPL reporting) and what does not (the embedding principle).

### Track B current status: PARTIAL (unchanged)

Gate G of the StableLM qualification remains **PARTIAL**. The core embedding
operation is a named prior technique; the remaining NF4-specific scope is narrow.
Track B stays PARTIAL until its evidence genuinely supports a different
conclusion. **No "first"/"unprecedented" claim; no new-embedding-principle
claim** is made anywhere in this project.

---

# Independence of the tracks

- **Passing Track A does not prove Track B.** A method can be technically
  flawless and not novel.
- **Passing Track B does not replace technical validation.** A novel idea that
  fails recovery, robustness, detectability, or utility is not a publishable
  result.
- Neither track's thresholds may be weakened after seeing results.
- Additional models expand **Track A** evidence only. Model count does **not**
  improve Track B.

---

# Expansion rule (amended)

Additional-model expansion is authorized when **Track A is PASS for the declared
scope**, with **Track B disclosed as PARTIAL** (a bounded limitation) in every
artifact. This replaces V1 §3's "all gates A–G PASS" rule. The novelty limitation
is disclosed, never hidden; expansion does not convert a Track A pass into a
novelty claim.

A model that fails any Track A gate is a **counterexample**: it is reported with
its evidence, not deleted, not averaged away, and not silently replaced. On a
gate failure, stop and investigate before proceeding to the next model.

---

# Seed and reporting policy (unchanged from V1)

- Three seeds where computationally feasible; the primary robustness claim uses
  BER = 0 across all completed conditions.
- Failures and inconclusive conditions are retained and reported, never relabeled.
- New runs use unique output directories; completed runs are never overwritten.
- The small-capacity detectability result (LR AUC **0.5343**, grouped CI95
  [0.5332, 0.5363], TPR@1% 0.0139, permutation **p = 0.0050**, 4 groups) is
  **preserved and reported** for every relevant model. It is not hidden and the
  detectability criteria are not weakened.

# Known resource limits (recorded before results)

CPU-only, 4 cores, ~15 GB RAM, **~5.9 GB free disk** at 2026-10-19. Models are
qualified **one at a time**; existing evidence and frozen artifacts are **never
deleted** to make space. If disk cannot accommodate a candidate, the candidate is
deferred or the count is reduced (minimum three additional architectures), and
that is reported.
