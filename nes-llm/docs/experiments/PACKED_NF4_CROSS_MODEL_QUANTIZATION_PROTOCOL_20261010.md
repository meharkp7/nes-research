# Packed-NF4 Cross-Model and Quantization-State Detectability Protocol

**Status:** protocol draft; no new cross-model result is claimed  
**Branch:** `research/contract-b-b14-and-evaluation`  
**Frozen dependency:** QSE pilot dated 2026-10-09 and packed-NF4 audit dated 2026-10-10 remain unchanged.

## Research questions

1. Does a packed-NF4 detector distinguish clean controls from embedded artifacts within a model family?
2. Does it transfer to a model family not used for training or threshold selection?
3. Does detection change when NF4 nested/double quantization is enabled versus disabled?
4. Does an embedding survive when quantization is applied after embedding, or does requantization destroy the payload?
5. Are apparent signals caused by the embedding, model identity, tensor identity, or quantization configuration?

These questions must be reported separately. A result on packed NF4 codes is not automatically transferable to GPTQ, AWQ, or other packed formats.

## Frozen evidence and guardrails

- Preserve the existing QSE pilot, audit report, run outputs, and their hashes.
- Do not treat blocks from one tensor as independent experimental sources.
- Do not assign new source IDs to repeated exports, runs, or quantization variants of the same underlying tensor to inflate source counts.
- Each clean/embedded pair must share a source tensor identity, model revision, tensor key, compatible quantization configuration, and block coverage.
- The clean and embedded artifacts must be distinct and have recorded SHA-256 digests.
- Source IDs must be supported by a provenance record; the validator cannot prove independence from strings alone.
- Keep all tensors, repeated runs, and quantization variants belonging to a model in the same partition for the **unseen-model** evaluation.
- Keep all variants of one source tensor together for tensor-level splits.
- No detector, threshold, feature selection, or hyperparameter choice may use the held-out test groups.

## Stage 0 — verify the current adapter before expanding

Use the current Qwen2.5-3B QSE pilot as a frozen regression fixture, not as an independent test group. Verify that the adapter contract can load artifacts, extract codes/features, pair clean and embedded controls, and report payload BER without overwriting any existing output.

For new runs, choose and record one compatible tensor per model (same semantic layer/role where architecture permits, but record the exact key and shape). Do not assume layer 16 or a Qwen-specific tensor key exists unchanged in other families.

## Stage 1 — small cross-model smoke matrix

Start with three model families:
- Qwen family (existing model is the implementation/regression anchor).
- Llama family.
- Mistral family.

Before any large run, pin exact model IDs and immutable revisions, check licenses/access, record tensor keys/shapes, and confirm that the chosen embedding/extraction method works on each. These are candidate families, not a claim that every checkpoint is currently compatible.

For each model, compare two NF4 configurations that differ only in nested/double quantization:
- NF4 with `compress_statistics=True`.
- NF4 with `compress_statistics=False`.

Keep block size, tensor, embedding method, payload/corpus, and all other settings fixed within a matched comparison wherever technically possible. Record any unavoidable deviations. This is a smoke study to expose compatibility issues and estimate variability; three model families are not enough for a strong generalization claim.

For every model × quantization configuration, produce:
1. A clean quantized control.
2. An embedded artifact produced under the same configuration.
3. A machine-readable metadata/provenance record.
4. An extraction/BER report.
5. Packed-code features and a paired descriptive comparison.

If a method does not support a particular model/configuration, record an explicit unsupported/failed status and reason. Do not silently omit it or change methods without recording that change.

## Stage 2 — broader seven-model and seven-method evaluation

After Stage 1 passes, expand to the previously planned seven-model roster and all seven NES embedding methods. First audit the repository's existing model/method definitions and method-specific contracts; reuse them rather than inventing a second roster. Run one model end-to-end before launching the full matrix.

Do not assume every method operates on packed NF4 codes. Methods that embed in floating-point residuals and methods that alter packed quantization codes need separate, format-appropriate adapters and shared reporting fields.

## Experimental factors and run identity

Record at least:
- `study_id`, `run_id`, `source_id`, `pair_id`, `method_id`
- model ID, immutable model revision, architecture/family
- tensor key, shape, dtype and source tensor digest when available
- quantizer/library and version, quantization type, block size, nested/double-quant flag, compute dtype, device/backend
- operation order: quantize→embed or embed→requantize
- clean artifact SHA-256 and embedded artifact SHA-256
- payload/corpus digest (never store secret payloads in the registry), expected and recovered lengths, BER, exact extraction flag
- feature schema/version, feature CSV digest, extraction code commit
- explicit status: planned, completed, failed, unsupported, or excluded; reason for any non-completion
- provenance note and operator/date

Suggested stable identifiers:
- `source_id`: model revision + tensor key + source tensor digest (or another documented immutable identity).
- `pair_id`: one clean/embedded matched pair for one source, quantization configuration, and method.
- `run_id`: one execution attempt; retries get new run IDs and must not be counted as independent sources.

## Evaluation plan

### Descriptive stage
Report clean-vs-clean repeatability first, then paired clean-vs-embedded feature shifts. These analyses are descriptive; blocks are correlated and are not the inferential sample size.

### Detector stage
Use a simple baseline (e.g., regularized logistic regression) before a more complex detector. Fit preprocessing and feature selection on training groups only. Compare against:
- majority-class / chance baseline;
- a quantizer-only and model-only diagnostic to reveal confounding;
- clean-vs-clean negative controls, where feasible.

Report ROC-AUC and precision/recall at a threshold chosen without test labels, plus false-positive rate and group-aware confidence intervals. Report BER/payload survival separately from detectability.

### Generalization splits
- **Within-model/tensor transfer:** group by complete source tensor and all its repeated runs/variants.
- **Unseen-model transfer:** leave one entire model family out for final evaluation; all quantization states and tensors from that family remain held out.
- **Unseen-quantizer transfer:** hold out an entire quantizer/configuration family, not random blocks.
- **Cross-model × cross-quantizer:** only attempt once there are enough model families and compatible artifacts to populate all required cells.

Do not describe three source groups as statistically sufficient. The minimum-source check is a software safety guard, not a sample-size justification. A credible held-out-model estimate needs multiple independent training families and multiple held-out families, with uncertainty reported at the model/group level.

## Stop/go gates

1. **Contract gate:** exact artifact schema and quantization metadata validated.
2. **Pair gate:** clean/embedded identity, model/tensor/configuration, and block coverage match.
3. **Recovery gate:** BER and extraction outcomes recorded; failures remain in the ledger.
4. **Confounding gate:** clean-vs-clean and model/quantizer-only diagnostics included.
5. **Generalization gate:** independent group counts support the exact claim being made.
6. **Reproducibility gate:** command, environment, code commit, input/output hashes, and all exclusions recorded.

No detector-performance claim is allowed until all applicable gates pass.
