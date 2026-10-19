# NF4 Anchor Preservation — Detectability Evaluation of interior_nearest

Run ids: `nes_detectability_20261016` (features), `nes_detectability_analysis_20261016`,
plus supplement and grouped-oracle runs.
Status: COMPLETED.

## Frozen prior results
All earlier outputs frozen in `FROZEN_ALL_HASHES.txt` / `FROZEN_AE_HASHES.txt`. Not rerun, not modified.

## Design (as executed)

- Checkpoints: `ckpt/tinyllama_clean`, `ckpt/qwen_clean`; source tensor hashes verified before use
  (`out/detectability_v1/source_hashes.json`).
- Method: frozen `interior_nearest`. Payloads 10,000 and 50,000 bits × seeds 20261016/017/018.
- Matched clean controls from the exact same source tensors.
- Examples: 64-code blocks, 1,024 sampled per artifact (fixed `SAMPLE_SEED=424242`), 84 artifacts
  → 86,016 block examples.
- **16 predeclared features** (fixed before fitting): endpoint fraction/counts, interior fraction,
  code mean/std, odd fraction, parity imbalance, odd-interior fraction, distinct codes, code entropy,
  dequantized mean |value| and absmax, adjacent-equal fraction, adjacent-parity-change, max run.
  Rationale per feature is in `manifest.json → feature_rationale`.
- **Leakage control:** all CV is grouped by `model::tensor` (GroupKFold); blocks from one tensor never
  split across train and test.
- Detectors: logistic regression, gradient boosting, and a small MLP (exploratory).
- Controls: clean-vs-clean, embedded seed-vs-seed; plus leave-one-model-out transfer.
- Library versions pinned: numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1, torch 2.14.1+cpu, bnb 0.50.2.
- Deterministic sanity test (`--self-test`) PASS before the run.

## Measured findings

### Detector performance (grouped CV by model::tensor, out-of-fold)

| Detector | ROC-AUC | TPR@FPR 1% | TPR@FPR 5% | AUC @10k | AUC @50k |
|---|---:|---:|---:|---:|---:|
| gradient boosting | 0.5016 | 0.0096 | 0.0492 | 0.4999 | 0.5034 |
| logistic regression | 0.5056 | 0.0099 | 0.0500 | 0.5018 | 0.5093 |
| MLP (exploratory) | 0.4992 | 0.0099 | 0.0500 | 0.4996 | 0.4987 |

All three sit at **chance** (AUC ≈ 0.50; TPR equals FPR at both operating points).
Per model: TinyLlama 0.499–0.505, Qwen 0.499–0.506.

### Effect sizes (embedded vs clean, pooled per model)

Maximum absolute Cohen's d across all 16 features is **0.016** (Qwen `mean_abs_value`) and
**0.013** (TinyLlama). No per-tensor effect exceeds |d| = 0.15 (zero such effects). The two
distributions are effectively identical on every predeclared feature.

### Leakage controls

| Control | ROC-AUC | Expected |
|---|---:|---|
| clean vs clean (block halves) | 0.4900 | ~0.5 |
| embedded seed vs seed | 0.4994 | ~0.5 |
| leave-one-model-out (held-out Qwen) | 0.5016 | exploratory |
| leave-one-model-out (held-out TinyLlama) | 0.5030 | exploratory |

Controls behave as expected, confirming the pipeline is not trivially separable by
sampling order or by seed identity.

## Exploratory results (labelled, not confirmed)

### Why block-level AUC is at chance: sparsity

interior_nearest changes only as many codes as the framed payload needs, so most blocks are
byte-identical to clean. Blocks actually touched (per artifact, of 1,024 sampled):

| Tensor type | 10k bits | 50k bits |
|---|---|---|
| large (q_proj, up/down/gate) | 0.9–7.5% touched | 6.9–32% touched |
| small (k_proj, v_proj) | ~46% touched | ~95% touched |

At 10,000 bits on large tensors, ~93–99% of sampled blocks are untouched, which alone drives the
block-level result toward chance.

### The ungrouped "oracle" number was leakage, and is retracted

- **Ungrouped 5-fold** on modified-vs-clean blocks gave AUC **0.697** — but that split mixes blocks
  from the same tensor across folds, so it leaks tensor identity.
- **Grouped leave-one-tensor-out** on the same task gives AUC **0.416** (TPR@FPR1% = 0.014,
  TPR@FPR5% = 0.053), i.e. **at or below chance**, on 10.5 M examples / 12 groups.

The 0.697 figure is therefore an artifact of the ungrouped split and is **not** evidence of
detectability. The grouped number is the one to trust.

### Artifact-level detection

Leave-one-tensor-out at the artifact level (mean+std of block features per artifact, 84 artifacts):
AUC **0.530** overall (0.510 at 10k bits, 0.550 at 50k bits). This is near chance with only
84 artifacts and 12 groups; it is underpowered and is reported as exploratory only.

## Claims that remain unsupported

- **"interior_nearest is undetectable" is NOT supported.** Chance-level AUC here reflects (a) sparse
  embedding on large tensors and (b) an underpowered design. Absence of detection under these
  features is not proof of undetectability.
- **No held-out neural-detector result is defensible.** Only two checkpoints exist; all neural
  numbers are exploratory, and the leave-one-model-out rows have a single test checkpoint each.
- **No calibration analysis** is supported: no detector exceeded chance, so reliability/calibration
  metrics would be meaningless.
- The artifact-level AUC of 0.53 is not significant at this sample size and is not claimed as a
  positive detection.

## Blockers / limitations to a definitive detectability claim

1. **Few checkpoints (2).** A defensible held-out neural evaluation needs more independent
   checkpoints (e.g. 5+ models), which are not present and were not downloaded.
2. **Sparse embedding.** On large tensors at 10k bits, few blocks are modified. A matched
   evaluation that concentrates on touched blocks (or on small tensors where ~95% of blocks are
   touched at 50k bits) is the smallest change that would give a higher-powered test.
3. Smallest required action to strengthen this: run the grouped detector restricted to the
   small-capacity tensors at 50,000 bits, where nearly every block is modified — that removes the
   sparsity confound without downloading anything new.

## Deliverables (outside the repo)

`/workspace/nes_run_20261016_anchor_ablation/`
- `code/nes-llm/scripts/exp_detectability.py` — feature extraction
- `code/nes-llm/scripts/exp_detectability_analysis.py` — detectors + controls
- `code/nes-llm/scripts/exp_detectability_supplement.py` — sparsity + oracle
- `code/nes-llm/scripts/grouped_oracle.py` — grouped oracle (leakage check)
- `out/detectability_v1/` (features, source hashes, manifest)
- `out/detectability_analysis_v1/` (detector results, effect sizes)
- `out/detectability_supplement_v1/` (sparsity, oracle, grouped oracle)
- extended `FROZEN_ALL_HASHES.txt`

## Guardrails honoured

- Frozen codec unmodified; all prior outputs preserved and hashed.
- New output directories only; nothing overwritten.
- No branch change, commit, or push; repo working tree clean.
- Kept separate from the five residual methods and QSE/DCE tracks.
