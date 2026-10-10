# Packed-NF4 Detector Evaluation Design Audit — 2026-10-10

## Status

**Gate: do not train or report a headline detector score yet.**

The local file `cache/packed_nf4_grouped_dataset_20261010.csv` was checked by
`scripts/validate_packed_nf4_split_manifest.py` from the audit branch.

Observed validation report:
- Rows: 606,208
- Unique artifact hashes: 16
- Unique runs: 8
- Pair groups: 8
- Status: PASS
- Errors: none

This is a bookkeeping/pairing pass only. It does not establish statistical
independence, valid detector evaluation, or absence of all leakage.

## Observed split structure

The CSV has 303,104 clean rows (label 0) and 303,104 embedded rows (label 1).
Model identity is perfectly aligned with the split:

| Split | Model | Rows |
|---|---|---:|
| train | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 131,072 |
| validation | Qwen/Qwen2.5-3B | 327,680 |
| test | google/gemma-2-2b | 147,456 |

Each model occurs in one split only. Artifact counts per role are:
- train: 1 clean and 1 embedded artifact
- validation: 6 clean and 6 embedded artifacts
- test: 1 clean and 1 embedded artifact

The Qwen validation set includes Q, K, and V projection sources and nested/plain
runs. The test score would be based on a single clean and a single embedded
Gemma artifact, with many block-level rows per artifact.

## Interpretation

1. A detector could exploit model-specific or tensor-specific distributions
   rather than an embedding effect.
2. The large block count must not be interpreted as a large count of independent
   artifact-level replications.
3. Balanced labels are necessary but do not rule out source, artifact, model,
   or preprocessing leakage.
4. Random block-level splitting is not acceptable when related blocks from the
   same source/artifact/run can land on both sides of evaluation.
5. A leave-one-model-out result can be a legitimate cross-model-transfer
   experiment, but this particular three-model layout has only one model in each
   split and one held-out test model. Model identity and split are inseparable,
   so a single result cannot support broad generalization claims.

## Evaluation questions must be separated

### A. Within-model detectability
Train and evaluate on different independent source artifacts/runs from the
same model, keeping all blocks from a source artifact/run together. This
requires enough independent clean and embedded artifacts; current artifact
counts are insufficient for a strong artifact-level estimate.

### B. Cross-model transfer
Train on multiple models, tune on separate groups, and test on a held-out
model. A stronger design needs more than one model in training and enough
held-out models/artifacts to avoid treating a single model as general evidence.
Report per-model results as well as aggregated results.

### C. Cross-configuration transfer
Explicitly hold out nested/plain configurations or Q/K/V tensor families as
the transfer target. Do not mix these claims into the cross-model result.

## Required work before detector training

1. Trace how `artifact_id`, `artifact_sha256`, `run_id`, `source_id`,
   `tensor_key`, `role`, `label`, and `block_index` are generated.
2. Confirm whether clean and embedded artifacts are paired from the same base
   artifact, and whether one clean artifact is reused across multiple embedded
   configurations.
3. Audit feature construction for source/label leakage and repeated patch/block
   overlap across any planned splits.
4. Define a group key that prevents dependent blocks or reused source artifacts
   from crossing train/validation/test boundaries.
5. Create clean-vs-clean controls and a known-positive control, then evaluate
   using a held-out group split with ROC-AUC, balanced accuracy, FPR at a stated
   operating point, and uncertainty estimated at the group/artifact level.
6. Freeze the split manifest and report its hashes before training.

## Claim boundary

Current evidence supports only: **the CSV passed the validator's specified
bookkeeping and clean/embedded block-index pairing checks.**

It does not yet support claims about detector performance, statistical
independence, detectability/undetectability, or cross-model generalization.
