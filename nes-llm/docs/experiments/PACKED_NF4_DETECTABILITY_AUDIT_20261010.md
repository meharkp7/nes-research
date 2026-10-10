# Packed-NF4 Detectability Harness — Audit and Implementation Contract

**Status:** design/audit committed; no detector run claimed  
**Branch:** `research/contract-b-b14-and-evaluation`  
**Frozen record:** `nes-llm/docs/experiments/QSE_PILOT_20261009.md` (leave unchanged)

## 1. Audit verdict

Reuse components, not prior verdicts.

| Existing component | Reuse decision | Why / required adaptation |
|---|---|---|
| `src/steganalysis/exp7_neural_detector.py` | Reuse the MLP skeleton and metric implementation only after adding threshold-free score outputs | It expects dense 4096-element residual patches, not discrete packed NF4 codes or full tensor summaries. Its default thresholded accuracy is insufficient by itself. |
| `src/steganalysis/build_exp7_detector_dataset.py` | Treat as a pairing/split design reference; do not use as the current NF4 data loader | It embeds on FP16-derived NF4 residuals. Current seven-method methods operate on packed NF4 artifacts and different sender/receiver contracts. |
| `src/experiments/exp14_blind_patch_detector.py` | Reuse the blind-vs-informed threat-model distinction and positive-control requirement | The sampling and labels must be defined at the packed-code/artifact level. Blind means no carrier map, no reference weights, and no method identifier. |
| `src/experiments/exp16_cross_scheme_detector.py` | Reuse the two-way transfer matrix and embedding-level split principle | Historical artifact records `controls_valid: false`; its 0.50 cross readings are uninformative and must not be cited as stealth evidence. It uses residuals and the sign/LWE schemes, not the current seven packed-NF4 methods. |
| `scripts/contract_b_detectability_diagnostic.py` | Reuse histogram/TV/KL/LSB/change-count ideas selectively | It is hard-wired to the B1.4 fixture/key/carrier count and a particular tensor reader. Generalize it to artifact adapters; never expose private key or carrier map to blind features. |

## 2. Target artifact contract

Build a small adapter interface around actual run directories. Each adapter should return, when available:

- method identifier and run/replicate identifier;
- model ID and exact tensor keys/shapes;
- original clean NF4 packed codes, embedded packed codes, and clean NF4 re-quantization controls;
- packed-code format and quantizer metadata;
- payload bit count and sender/receiver contract class (artifact-only vs reference-assisted);
- carrier map/reference availability **for evaluation bookkeeping only**, never as blind-detector input;
- SHA-256 hashes for inputs/outputs and paths relative to the run root.

No adapter should silently substitute a model checkpoint for a packed artifact. Missing clean/re-quantized controls should produce an explicit `NOT_AVAILABLE`, not an implicit clean baseline.

## 3. Minimum experimental arms

1. **Clean-vs-clean control:** independent clean NF4 artifacts / re-quantizations, matched to the same model, tensor, quantizer and preprocessing.
2. **Embedded-vs-clean:** embedded artifact against its matched source and control; never pair mismatched model/tensor shapes.
3. **Artifact-only blind detector:** only observable released packed codes/statistics; no clean reference, carrier map, payload/key, method label, or run identifier as feature.
4. **Method-aware diagnostic:** a separately labelled upper-bound/stress test allowed to know the method class, but no test-run IDs or payload-derived metadata.
5. **Cross-method transfer:** train on method A and test on unseen embedding runs from method B; include within-method positive controls for each direction.
6. **Optional reference-assisted audit:** report separately because QSE's receiver requires clean NF4/reference residuals. Do not blend it with artifact-only adversaries.

## 4. Leakage-resistant split policy

The indivisible group is an independent embedding run (including all tensors/patches derived from that run). Split groups before patch extraction or feature expansion. Never split patches from one embedded artifact across train/test. Keep clean and embedded versions of the same source in the same split, while pairing them within a split.

Freeze train/validation/test run IDs in a manifest. Test runs are opened once after detector/hyperparameter selection. Add a negative test that intentionally duplicates a run ID across splits and ensure the validator rejects it.

Match non-semantic nuisance variables across classes: model, tensor key, shape, quantizer config, source checkpoint, preprocessing, and run generation path. Include a detector-control check that predicts labels after class labels are randomized and a control distinguishing two clean NF4 generation paths. A failed positive control or unexpectedly strong clean-vs-clean separation invalidates the affected detector result; do not mark it PASS.

## 5. Features and detector families

**Statistical, descriptive first:** 16-bin code frequencies; total variation and smoothed Jensen-Shannon divergence; entropy; LSB/bit-plane frequencies; adjacent-code/transition counts; blockwise versions of these values; changed-code fraction and code-pair transitions when clean reference is available. Clearly distinguish blind features from paired/reference-only diagnostics.

**Learned detectors:** begin with a small regularized logistic regression and a gradient-boosted/tree baseline on aggregate/blockwise features; then adapt the existing MLP to packed-code windows only if sample volume and held-out controls justify it. Scale the detector family gradually so a coding bug or confound is caught cheaply before neural training.

Feature extraction must be identical for train/test, deterministic where possible, fitted only on train data, and versioned in the run manifest. No labels may be inferred from filenames, paths, method-specific metadata, or availability of optional files.

## 6. Metrics and reporting

Report ROC-AUC, balanced accuracy, sensitivity/recall, specificity, false-positive rate at a pre-registered operating point, confusion counts, calibration/score distributions, and 95% uncertainty intervals. Use bootstrap resampling clustered by independent embedding run—not by patch—to avoid pseudo-replication. With few independent runs, show the run-level points and explicitly say the interval is underpowered.

Accuracy near 0.50 alone is not evidence of stealth. The report must first pass control validity, then report effect sizes and intervals against an explicit attack model. Thresholds are preregistered evaluation criteria, not proofs of invisibility.

## 7. Shared report schema

Each method/run should preserve the five evaluation dimensions:

- **Capacity:** requested and recovered bits, BER, exact-match/checksum.
- **Distortion:** changed codes/fraction, tensor/weight RMSE where meaningful, TV/JS and blockwise divergence.
- **Utility:** controlled evaluation against matched clean NF4 and re-quantization baselines; not inferred from detector success.
- **Detectability:** detector family, information visible to attacker, data split manifest, ROC-AUC/balanced accuracy/FPR, run-clustered uncertainty, controls-valid flag.
- **Robustness:** explicit transformation and receiver contract, pre/post BER and distortion. Reference-assisted and artifact-only receivers must not be conflated.

Every field should carry one of: measured value, `NOT_RUN`, `NOT_AVAILABLE`, or `NOT_APPLICABLE`, plus a reason. Do not drop failed methods/cells from aggregate reports.

## 8. Execution order

1. Implement and test artifact discovery/adapter contract with tiny fixtures.
2. Validate pairing, hashes, provenance, and run-group split checks.
3. Implement deterministic descriptive statistics and clean-vs-clean validity controls.
4. Run small logistic/tree baselines on one model/corpus and all seven methods.
5. Add held-out blind MLP and transfer matrix only after the controls pass.
6. Repeat independent runs and compute cluster-bootstrap uncertainty.
7. Expand models/corpora only after the single-model matrix is auditable.
8. Keep all heavy results in new timestamped output folders; do not overwrite the frozen QSE pilot.

## 9. Explicit non-claims

This audit does not run a detector and does not demonstrate that any method is stealthy. Existing residual-based detector scores do not automatically transfer to packed NF4. A single run, a near-chance accuracy number, or a failed/invalid positive control cannot establish undetectability.
