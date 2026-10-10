# NES Research — frozen execution order

**Status date:** 10 October 2026  
**Branch:** `research/contract-b-b14-and-evaluation`  
**Status:** Execution order frozen; pull request remains unmerged.

## Purpose
Finish the existing NES evidence, assess the intended seven-method/seven-model coverage, and prepare a defensible manuscript without scope drift. Existing results and failed runs are immutable evidence. Do not overwrite reports, relabel datasets to imply independence, or start a new embedding family.

## Fixed order and exit criteria

### Stage 1 — Evidence and coverage inventory (START HERE)
Create one master matrix for the intended 7 embedding methods × 7 models (49 cells). For each cell, record embedding status, artifact path/hash, payload bits, recovery status/BER/hash verification, detector coverage, utility/attack coverage where applicable, source revision, run provenance, and evidence path. Each field must be `PASS`, `FAIL`, `NOT RUN`, `NOT COMPARABLE`, or `UNKNOWN`, with a report path when evidence exists. Inventory existing outputs before running experiments. Do not infer missing cells are successful. Exit: matrix exists and every cell has an evidence-backed status or explicit unknown.

### Stage 2 — Embedding and recovery gap closure
Use Stage 1 to identify missing or incomparable method–model cells. Run only missing, required combinations, using fixed payload/corpus and receiver contract where method semantics permit. Preserve every run under a new unique output path. Record exact recovery, BER, checksum/hash, payload size, runtime, model/checkpoint revision, quantizer config, seeds, and command. Exit: all 49 intended cells are measured or explicitly blocked/not comparable with reason.

### Stage 3 — Detection evaluation across the matrix
Inventory existing detectors and feature definitions, then evaluate available method–model pairs using a fixed protocol. Separate per-cell detection from pooled analysis. Record sample counts, independent grouping unit, ROC-AUC and uncertainty, fixed-threshold metrics, and baselines. Fit preprocessing only on training data. Do not call near-chance results stealth proof. Exit: every comparable cell has results or an explicit blocked reason; no leakage or false independence claims.

### Stage 4 — Generalization evaluation
Run model-held-out evaluation with the entire model family/checkpoint group excluded from training, only where source provenance and pair construction support it. Rotate held-out models. If independence or matched construction cannot be established, mark the result exploratory/blocked instead of manufacturing splits. Exit: transfer evidence and limitations are recorded for all supported folds.

### Stage 5 — Transformation, spillover, and utility interpretation
Consolidate existing lifecycle results first. Keep pristine recovery, requantization, pruning-plus-requantization, and older residual-domain experiments separate. Investigate carrier/non-carrier code changes only from existing artifacts and implementation; do not assume changed non-carrier codes contain payload bits. Run a new transformation only if a specific paper-critical question remains. Utility results stay scoped to their exact prompts/model/protocol. Exit: each robustness and utility claim is linked to evidence and scoped correctly.

### Stage 6 — Reproducibility and final test gate
Run the focused test suites once after code/results freeze; record commands, versions, seeds, and exit status. Generate an evidence index with relative paths and SHA-256 hashes. Preserve failed outputs. Exit: report bundle is reproducible and internally consistent.

### Stage 7 — Manuscript claim audit and freeze
Compare every claim in abstract, introduction, methods, results, figures, captions, discussion, conclusion, and README against the frozen evidence. Do literature/novelty comparison separately; empirical results alone do not establish novelty or cryptographic security. Exit: all claims supported, qualified, or removed; PR remains unmerged until user reviews.

## Scope lock
- No new embedding family or unrelated side project during this sequence.
- No further detector feature ablations unless Stage 1–4 identifies a specific claim-critical gap.
- No reruns merely to improve a metric or make a table look complete.
- No counting hardlinks, nested variants, tensor projections, or repeated paths as independent model replications.
- No claims of universal undetectability, universal robustness, cryptographic authentication/confidentiality, or LWE security without the corresponding evidence.
- The 7×7 matrix is the target coverage plan, not an assumption that all seven methods are compatible with all seven models. Incompatible cells must be marked `NOT COMPARABLE` with reasons.
- If required checkpoints, compute, or provenance are unavailable, record `BLOCKED`/`UNKNOWN` with reason and move forward; do not loop indefinitely.

## Current starting point
Existing seven-method recovery work and detector work are substantial, but complete 49-cell embedding/recovery/detection coverage has not yet been demonstrated by the current audit. Detector provenance is limited: the grouped dataset has eight bookkeeping source/run groups across three model IDs; run IDs are caller-supplied, some source revisions are unrecorded, and hardlinked duplicate artifacts exist. Cross-model AUCs are near chance but exploratory and group-limited. A clean-control repeat was deterministic (0 changed packed codes), while the matched QSE comparison had 76,336 changed codes, including 40,955 non-carrier changes; this is consistent with possible block-local spillover but does not establish the cause or imply that non-carrier codes carry payload bits. The latest QSE artifact recovered 36,128/36,128 bits exactly under reference-assisted decoding.

## Stage 1 investigation note — historical matrix status discrepancy (10 October 2026)

The local Git history shows commit `53cb25ffc043c241a31db439845b473448d896b2` (`Parse extraction JSON despite device logs and preserve failure reports`) dated 2026-10-10 00:47:19+05:30. It adds stdout JSON scanning after device/progress text and preserves structured extraction failures. The inspected local working tree is clean at `6463630da77af55d702a391e00f14fa59e8b5b85`.

In the historical matrix run `20261009T185340Z`, the `sign` row is marked `EXTRACT_FAILED` and has null top-level `ber`/`exact_match`, while the row's `reason` contains a full extraction JSON report with `exact_match=true`, `ber=0.0`, `bit_errors=0`, `expected_bits=recovered_bits=36128`, and `embedded_manifest_digest_match=true`. The per-method `extract.json` reports for `sign`, `magnitude_aware`, `qae`, `lwe_grid_parity`, and `split_sign_parity` in that run show the same exact-recovery indicators.

This is consistent with the older runner failing to parse stdout prefixed by `Using device: mps`; it is not proof of the exact subprocess exit code or of every historical row's cause. Run IDs are UTC: the early runs ending in `185340Z`, `185703Z`, `185921Z`, `190017Z`, `190043Z`, and `191611Z` predate the parser-fix commit timestamp; `191727Z` and later runs postdate it. Keep historical matrix statuses unchanged. In the evidence inventory, classify rows with corroborating exact-recovery reports as **recovery verified / matrix status inconsistent**, retain their original status, and do not count repeated/hardlinked artifacts as independent replications.

Stage 1 remains open until the 7×7 evidence-backed coverage matrix is created; this note does not establish complete model coverage.

## Operating rule
At the end of each stage, update this file and the matrix with evidence and exit decision before moving to the next stage. No stage changes order without a documented paper-critical blocker and explicit user agreement.

## Stage 3 implementation note — residual descriptive batch (11 October 2026)

A first residual-domain descriptive batch completed locally at
`../cache/detectability_tinyllama_residual_20261011/`. The final report schema is
`nes.seven_method_detectability_stage1.v2`; it records five discovered cells,
five paired-descriptive-ready cells, zero blocked cells, 309,760 block-feature
rows, 55 layer-descriptive rows, a 79.08-second runtime, block size 4096, and a
deterministic evenly-spaced quantile sample capped at 32,768 values. Quantiles
are approximate when the tensor exceeds that cap; other listed statistics use
full values. The five methods are `sign`, `magnitude_aware`, `qae`,
`lwe_grid_parity`, and `split_sign_parity`, all on
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Each cell reports recovery matrix status
`PASS`, 61,952 feature rows, and 11 layer summaries.

This batch is **paired descriptive feature export only**, not a trained detector
benchmark. No classifier was fitted because available evidence does not support
a leakage-resistant independent split. Blocks from the same tensor are
correlated and must not be treated as independent inferential units. The batch
does not establish stealth, undetectability, or cross-model generalization.
Quantile approximation and all source/artifact identities should be preserved
with the generated report and CSV outputs.

QSE/DCE remain part of the seven-method plan; their comparative packed-NF4
detectability remains **BLOCKED / NOT COMPARABLE** until a historically matched
same-source clean packed-code control and its provenance are verified. Do not
substitute a newly generated control and label it historically matched without
evidence. Their existing recovery evidence remains intact and separate from
this residual-domain batch.

**Exit decision:** this batch is complete and may be frozen as descriptive
evidence. Stage 3 overall remains open: this single-model, five-method batch
does not satisfy matrix-wide detector coverage. Proceed with evidence indexing,
focused tests, and claim audit; do not launch a classifier or repeat the batch
without a specific, documented paper-critical reason.
