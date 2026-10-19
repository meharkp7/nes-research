# Targeted Detectability Supplement — Small-Capacity Tensors at 50,000 Bits

Run id: `nes_detectability_smallcap_20261016`.
Status: COMPLETED.

This executes the supplement proposed in `RESULTS_DETECTABILITY.md`: it removes the
sparsity confound by evaluating only the small-capacity attention tensors
(`k_proj`, `v_proj`), where ~95% of blocks are modified at 50,000 bits.

## Frozen prior results
All earlier outputs frozen in `FROZEN_ALL_HASHES.txt` (60 entries). The full detector
evaluation and prior experiments were not rerun.

## Design

- Checkpoints: `tinyllama_clean`, `qwen_clean`; source tensor hashes verified
  (`out/detectability_smallcap_v1/source_hashes.json`).
- Tensors (4 groups): `tinyllama L21 k_proj`, `tinyllama L21 v_proj`,
  `qwen L17 k_proj`, `qwen L35 v_proj`.
- Method: frozen `interior_nearest`, payload 50,000 bits, seeds 20261016/017/018.
- Matched clean controls from the same source tensors.
- Reused the **16 predeclared features** and the existing detector pipeline; no feature
  tuning against test results.
- Examples: 64-code blocks, 2,048 sampled per artifact; 32,768 examples, 16 artifacts, **4 groups**.
- All splits grouped by `model::tensor`; 4 groups = leave-one-tensor-out.
- Bootstrap CIs resample groups (not blocks), so they reflect group-level uncertainty.

## Measured findings

### Sparsity confound removed

| Tensor | blocks touched @50k |
|---|---|
| tinyllama L21 k_proj | 95.9% |
| tinyllama L21 v_proj | 95.9% |
| qwen L17 k_proj | 95.3% |
| qwen L35 v_proj | 95.5% |

Nearly every block is modified, versus 0.9–32% on the large tensors in the original
evaluation.

### Detectors (grouped leave-one-tensor-out)

| Detector | ROC-AUC | CI95 (group bootstrap) | TPR@FPR1% | TPR@FPR5% |
|---|---:|---|---:|---:|
| logistic regression | **0.5343** | [0.5332, 0.5363] | 0.0139 | 0.0611 |
| gradient boosting | 0.5197 | [0.5162, 0.5236] | 0.0101 | 0.0530 |
| MLP (exploratory) | 0.5053 | [0.5045, 0.5062] | 0.0107 | 0.0518 |

Per model: TinyLlama 0.522–0.536, Qwen 0.518–0.533. Per tensor (LR): k_proj 0.534/0.536,
v_proj 0.534/0.537.

### Effect sizes (embedded vs clean)

Maximum |Cohen's d| = **0.0886** (vs 0.016 in the original block-level evaluation — a
~5× larger separation). The driving features are consistent across all four tensors:

| Feature | Sign | Interpretation |
|---|---|---|
| `mean_abs_value` | negative | interior remap lowers dequantized magnitude |
| `std_code` | negative | interior remap narrows the code spread |

### Within-group label-permutation test (valid null)

Leave-one-tensor-out LR, 200 within-group label permutations:

- observed AUC = 0.5343; null mean = 0.4998, null sd = 0.0052, null p95 = 0.5074
- **p = 0.0050**

The observed AUC exceeds every one of 200 null draws except one.

### Clean-vs-clean control

AUC 0.4860 (CI95 [0.4752, 0.4931]) — at chance, as expected.

## Comparison with the original evaluation (not rerun)

| Setting | GB AUC | LR AUC | Blocks touched | Effect max \|d\| |
|---|---:|---:|---|---:|
| Original, large tensors, 10k | 0.4999 | 0.5018 | 0.9–7.5% | ~0.01 |
| Original, large tensors, 50k | 0.5034 | 0.5093 | 6.9–32% | ~0.01 |
| Original, pooled all tensors | 0.5016 | 0.5056 | mixed | 0.016 |
| **This supplement, small tensors, 50k** | **0.5197** | **0.5343** | **~95%** | **0.0886** |

Removing the sparsity confound moves AUC from chance to a small but measurable signal,
and roughly quintuples the largest feature effect size.

## Interpretation

- **Measured:** on small-capacity tensors at 50,000 bits, `interior_nearest` leaves a
  small, statistically significant distributional signature (LR AUC 0.534, p = 0.005;
  clean-vs-clean control at chance).
- **Exploratory:** with only **4 groups**, the CIs are narrow in block terms but the
  effective sample size for a generalisation claim is 4 tensors. A p-value from 4 groups
  should not be treated as a definitive result.
- **Unsupported / not claimed:** this does **not** establish that the method is detectable
  in general, and it does **not** establish that it is undetectable. A ~0.53 AUC with
  TPR@FPR1% ≈ 0.014 is a weak, near-threshold signal, not a reliable detector. The signal
  is consistent in sign across all four tensors, which argues it is real, but its practical
  detection power at low false-positive rates is minimal.
- The MLP detector stayed at chance (0.505); the signal is carried by the linear
  magnitude/spread features, not by higher-order structure the MLP could exploit at this
  sample size.

## Blocker for a definitive claim

**Single concrete blocker: too few independent groups (4 tensors, 2 checkpoints).**
The smallest required action is to add more independent checkpoints/tensors (e.g. 5+
models, multiple layers per tensor type), which requires downloading additional models —
this was not done because it was not requested and would need a prerequisite/blocker report
first.

## Deliverables (outside the repo)

`/workspace/nes_run_20261016_anchor_ablation/`
- `code/nes-llm/scripts/exp_detectability_smallcap.py`
- `code/nes-llm/scripts/smallcap_permutation.py`
- `out/detectability_smallcap_v1/` (features, sparsity, effect sizes, results, permutation test, manifest)
- extended `FROZEN_ALL_HASHES.txt`

## Guardrails honoured

- Frozen codec unmodified; all previous artifacts preserved and hashed.
- New output directory only; nothing overwritten; no experiment rerun.
- No branch change, commit, or push; repo working tree clean.
- Kept separate from the five residual methods and QSE/DCE tracks.
