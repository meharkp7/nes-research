# Qualification Protocol — StableLM-2-1.6B (`interior_nearest`)

Protocol frozen: 2026-10-17, **before** any new StableLM evaluation outcome was
inspected. Track 3 only (NF4 anchor-preserving: `interior_nearest`; baselines
`pair_parity`, `nearest_parity`). The five residual methods and QSE/DCE tracks
are out of scope and are not pooled with these results.

Method: frozen codec `src/experiments/nf4_anchor_channel.py`, sha256
`07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764`. NF4,
blocksize 64, nested statistics True. These are fixed and are not tuned.

## 0. What a gate status means

Four distinct questions are tracked separately and never conflated:

1. **Executed correctly** — the experiment ran and produced inspectable output.
2. **Technically passed** — the method met the numeric criterion.
3. **Hypothesis supported** — the scientific claim is supported by the result.
4. **Sufficient to generalize** — the evidence supports extending the claim.

A script finishing is *executed correctly*, not *passed*. Every gate below is
reported as PASS / FAIL / INCONCLUSIVE / NOT RUN with a reason.

## 1. Frozen constants

| Item | Value |
|---|---|
| Model | `stabilityai/stablelm-2-1_6b` @ `f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef` |
| Clean NF4 checkpoint | `ckpt/stablelm_clean` |
| Codec sha256 | `07ce008d…` (must match at every step) |
| Blocksize / nested | 64 / True |
| Payload sizes | 1,000 / 5,000 / 10,000 / 20,000 / 50,000 bits |
| Seeds | 20261016, 20261017, 20261018 |
| Requantization cycles | 0–3 primary; 0–5 stress (separately reported) |
| Primary tensor | `model.layers.0.self_attn.q_proj.weight` |
| Matrix tensors | the six predeclared StableLM tensors (L0/L11/L23 × q/k/v, up/down/gate) |
| Framing | B1.4: MAGIC + 4-byte length + payload + 8-byte SHA-256 prefix (16 bytes) |

## 2. Gates and acceptance criteria

### Gate A — Implementation and artifact integrity
Criterion: codec hash matches; source/output hashes recorded; intended NF4
config; only the intended tensor differs; logs complete; all settings recorded;
no hidden in-memory state needed for extraction.
Evidence: manifest + integrity JSON per run.

### Gate B — Exact end-to-end recovery (independent process)
Criterion: an embedded artifact saved to disk is loaded in a **fresh process**;
extraction reads the saved artifact, not an in-memory object; recovered length
== original; **BER == 0**; recovered checksum == original; carrier agreement == 1
where the protocol requires stable carriers; no truncation/padding/framing
failure.
Method: two-process protocol — process 1 embeds and saves; process 2 (separate
invocation) loads from disk and extracts with only (artifact, key, tensor name,
method). Key derived from seed via the documented `key_for`.

### Gate C — Multi-sample and transformation robustness
Criterion: cycles 0–3 pass across the five payload sizes, three seeds, and the
six tensors; failures recorded not discarded; baseline run under the same
settings; idempotence and endpoint-change counts recorded.
Extension: cycles 0–5 stress test, reported separately.

### Gate D — Detectability
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
StableLM is evaluated **separately** and never pooled with TinyLlama/Qwen.

### Gate E — Utility (perplexity equivalence)
Corpus: `wikitext-2-raw-v1` test, each model's own tokenizer, context 512,
non-overlapping windows. Paired sequence-level evaluation; token-weighted PPL.
- **Predeclared practical equivalence margin: ±0.5% relative PPL.**
- PASS requires the **paired 95% CI to fit entirely inside ±0.5%**.
- A CI that merely contains 0 is NOT a pass.
- If the CPU budget cannot produce a CI inside ±0.5%, report **INCONCLUSIVE**.
- PPL alone cannot establish semantic/generation equivalence.

### Gate F — Reproducibility and reporting
Criterion: rerunnable from recorded commands + frozen inputs; results linked to
immutable manifests/hashes; all failures/exclusions/deviations recorded; measured
vs derived vs speculation distinguished.

### Gate G — Novelty and prior-art integrity
Criterion: closest prior-art comparisons completed; no "first"/"unprecedented"
claim; unresolved high-overlap papers recorded as novelty risk.

## 3. Expansion rule (Phase 7)

Additional-model expansion is **locked** unless every gate A–G is PASS. If any
gate is FAIL or INCONCLUSIVE, produce a blocker report with the smallest targeted
unblocking experiment and do not expand. Thresholds are **not** weakened after
seeing results.

## 4. Seed and reporting policy

- Three seeds where computationally feasible; the primary robustness claim uses
  BER = 0 across all completed conditions.
- Failures and inconclusive conditions are retained and reported, never relabeled.
- New runs use unique output directories; completed runs are never overwritten.

## 5. Known resource limits (recorded before results)

CPU-only, 4 cores, 15 GB RAM. StableLM PPL throughput is measured before sizing
Gate E; if the achievable CI cannot fit ±0.5%, Gate E is INCONCLUSIVE by
construction, and that is reported rather than worked around.
