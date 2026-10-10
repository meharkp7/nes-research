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

## Verification performed

The user ran the builder and split-validator test suites in the isolated audit
worktree after fetching the audit branch head:

```text
python -m pytest tests/test_prepare_packed_nf4_grouped_dataset.py \
  tests/test_validate_packed_nf4_split_manifest.py -q

11 passed in 0.27s
```

The 11 passing tests include the three original builder tests, two new builder
regression tests, and six split-validator tests. An earlier `unittest`
invocation ran zero tests because the builder tests are pytest-style; that
zero-test result is not a pass.

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

## Pairing and provenance audit

A local read-only grouping report showed that every displayed group has equal
clean/embedded row counts and the same continuous block-index range beginning
at zero. The ranges are:
- Gemma Q: 0–73,727
- TinyLlama Q: 0–65,535
- Qwen Q: 0–65,535 for each nested/plain run
- Qwen K and V: 0–8,191 for each nested/plain run

The dataset has 16 distinct artifact hashes: eight clean and eight embedded.
No hash is shared across roles. This is consistent with distinct serialized
artifacts but does **not** prove clean-to-embedded parentage.

The `artifact_id` label is not a unique artifact key: `clean_control` maps
to multiple hashes, as does `clean_control_rebuilt`. Use `artifact_sha256`
for byte-level identity, while retaining run/source/model/tensor/configuration
as provenance metadata.

The CSV has no explicit parent-artifact or paired-control identifier. Therefore,
matching block indices and source/tensor labels establish structural pairing
only; they do not prove that the clean artifact is the correct baseline for its
embedded counterpart.

## Local dataset-builder code review and patch

The builder has been updated on this audit branch to:
- include model ID and tensor key in duplicate-block identity;
- require each source/run to map to exactly one model/tensor combination;
- compare clean and embedded block-index sets for each source/run/model/tensor
  pair and reject any mismatch;
- require exactly one distinct artifact hash per role in each pair group;
- preserve the existing role/label checks, model-group split, input CSV hashes,
  split manifest, and no-overwrite behavior.

Two regression tests were added:
- reject clean/embedded groups with different block-index sets;
- reject multiple artifact hashes within one pair-group role.

**Verification:** the combined builder and split-validator suites passed, 11/11,
in the isolated worktree at branch head `6fe803ea7a1ddbb24a28b9bdf200515b8eaccfab`.

These checks strengthen structural consistency but do not establish artifact
parentage. A clean and embedded artifact can have compatible metadata and block
indices while still being the wrong scientific pair. The builder does not yet
consume an authoritative clean-control-to-embedded-artifact mapping.

The local `README.md` does not document the specific command that produced the
current 606,208-row CSV. A search found no other CSV feature inputs under the
searched `../cache` path and no dataset-generation invocation outside the
builder/tests. No split/manifest JSON was found in the bounded cache search.
Thus, the exact original generation invocation, original input feature CSVs,
and their recorded hashes have not been recovered. This is a reproducibility
gap, not proof that the assembled CSV is invalid.

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
6. The same `artifact_id` text can refer to multiple hashes, so `artifact_id`
   must not be treated as a globally unique artifact key.

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

1. Recover the exact dataset-generation command and inventory the source feature
   CSVs if available elsewhere; verify their hashes against a split manifest.
2. Add an authoritative pairing manifest linking each embedded artifact to its
   intended clean control and recording base/source artifact hashes, model
   revision, tensor key, run/configuration, quantization settings, and payload
   mode where applicable.
3. Confirm whether one clean control is reused across multiple embedded
   configurations. Preserve this dependency in grouping and uncertainty analysis.
4. Audit feature construction for source/label leakage and repeated patch/block
   overlap across any planned splits.
5. Define the evaluation target and group split before training. With the current
   three-model layout, model-disjoint evaluation has one model per split and is
   not sufficient for broad generalization claims.
6. Create clean-vs-clean controls and a known-positive control, then evaluate
   using a held-out group split with ROC-AUC, balanced accuracy, FPR at a stated
   operating point, and uncertainty estimated at the group/artifact level.
7. Freeze the split manifest and report its hashes before training.

## Claim boundary

Current evidence supports only: **the CSV passed the validator's specified
bookkeeping and clean/embedded block-index pairing checks, the displayed groups
have matching block-index coverage, the builder now enforces stricter pair
consistency checks, and all 11 builder/validator tests pass.**

It does not yet support claims about detector performance, statistical
independence, verified parentage, detectability/undetectability, or cross-model
generalization.
