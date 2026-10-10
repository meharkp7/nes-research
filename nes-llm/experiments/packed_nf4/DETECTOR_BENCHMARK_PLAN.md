# Packed-NF4 detector benchmark — experiment contract

## Goal
Evaluate whether the existing 22 artifact-only packed-NF4 block features discriminate clean from embedded blocks in the current dataset, with the user's intended mixed-model random split as the primary result.

## Single consolidated run
Entry point: `scripts/run_packed_nf4_detector_benchmark.py`

The runner:
1. Validates role/label agreement and exactly one clean + one embedded row per matched `(source_id, run_id, block_index)` pair.
2. Creates a seeded 60/20/20 split at matched-pair level, stratified within model. Every partition must contain every model family. Both rows of a matched pair always stay together.
3. Trains logistic regression, random forest, histogram gradient boosting and a compact MLP on the same feature set. Preprocessing is fitted only on training data.
4. Selects the classification threshold on validation using Youden's J, then evaluates the test set once.
5. Reports pooled and per-model ROC-AUC, PR-AUC, balanced accuracy, precision, recall, F1, FPR and confusion matrix.
6. Runs source/run-grouped cross-validation (each group held out once) and five default label-randomization sanity controls.
7. Emits one JSON report and a separate reproducible split manifest.

## Interpretation guardrails
- **Primary mixed-block split:** matched pairs are held together, but source/run groups can cross partitions. The primary estimate can be optimistic due to correlated blocks and is not a confirmatory independent-sample estimate.
- **Grouped sensitivity:** uses GroupKFold out-of-fold predictions so each complete source/run group is held out once. Per-model metrics and fold membership are recorded. The available number of independent groups may still be too small for stable inference.
- **Label randomization:** a pipeline sanity check only, not evidence about stealth.
- **Clean-vs-clean:** marked unavailable because this dataset does not provide an independent, provenance-qualified clean-vs-clean class. Repeated copies of the same packed codes are not independent negatives.
- **Uncertainty:** no naive row-level confidence intervals. A credible cluster-level interval requires enough independent source/run groups.
- **Claims:** this benchmark cannot establish universal undetectability, generalization to unseen model families, or independent-run significance.

## Run
From `nes-llm/`:

```bash
../.venv/bin/python -m pytest -q tests/test_run_packed_nf4_detector_benchmark.py
../.venv/bin/python scripts/run_packed_nf4_detector_benchmark.py \
  --dataset ../cache/packed_nf4_grouped_dataset_20261010.csv \
  --output-dir ../cache/packed_nf4_detector_benchmark_$(date +%Y%m%d_%H%M%S)
```

The output directory must not already exist. Existing dataset files, the frozen split, and the pilot report are not modified.
