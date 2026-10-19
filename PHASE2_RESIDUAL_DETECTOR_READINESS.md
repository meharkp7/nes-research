# Phase 2 — Residual Detector Readiness and Protocol Boundaries

**Reviewed:** 2026-10-16 (Asia/Kolkata). **Status:** evidence reconciliation complete; scientific limitations remain.

The five-method 20261011 feature batch supports a **TinyLlama-only exploratory LOLO diagnostic**. Its extension to a model-held-out five-method detector comparison is **BLOCKED by coverage**. This does not erase exp18: exp18 already measured four methods on seven models using a different, separately fitted per-model neural detector protocol. No new experiments are required to close this documentation gate.

## Evidence scopes

| Evidence | Coverage | Measurement | Limit |
|---|---|---|---|
| exp18 strategy matrix | 4 methods × 7 model IDs; TinyLlama excluded for incomplete cache and Gemma-2-2b included | 28/28 bit round trips at BER 0.0, 28/28 noise gates pass; LWE detector accuracy 0.50 on 7/7 | Separate detector trained per method/model; not a detector trained on some models and evaluated on unseen models. Split_sign_parity is not one of the four methods. |
| 20261010 corpus artifacts | Five residual methods × 7 model IDs; TinyLlama included, Gemma-2-2b absent | 35 saved extraction reports: 36,128 bits each, BER 0.0 and exact corpus match | Selected-layer residual tensor bundles, not deployable model checkpoints; these recovery reports do not expand detector coverage. See Phase 3 source list. |
| 20261011 Stage 1 | TinyLlama only; 5 methods, 11 layers | 309,760 paired feature rows; 61,952 per method; 55 layer descriptive rows | Descriptive generator did not fit a classifier or establish independent-artifact inference. |
| 20261011 Stage 2 | Same TinyLlama batch; 11 layer folds | LR and HGB independently evaluated per method using LeaveOneGroupOut | Within-model diagnostic only; no unseen-model or independent-run claim. |

Exp18 source files: [google/gemma-2-2b](results/exp18_matrix_google__gemma_2_2b.json), [google/gemma-2-9b](results/exp18_matrix_google__gemma_2_9b.json), [meta-llama/Llama-3.1-8B](results/exp18_matrix_meta_llama__llama_3.1_8b.json), [microsoft/Phi-3-mini-4k-instruct](results/exp18_matrix_microsoft__phi_3_mini_4k_instruct.json), [mistralai/Mistral-7B-v0.3](results/exp18_matrix_mistralai__mistral_7b_v0.3.json), [Qwen/Qwen2.5-3B](results/exp18_matrix_qwen__qwen2.5_3b.json), [Qwen/Qwen2.5-7B](results/exp18_matrix_qwen__qwen2.5_7b.json). Protocol implementation: [exp18](nes-llm/src/experiments/exp18_strategy_model_matrix.py) and [exp10 measurement functions](nes-llm/src/experiments/exp10_strategy_comparison.py). The carrier-centered neural detector uses 400 paired patches, 20 embeddings, 30 epochs, patch size 4096 and a split grouped by sample ID. These accuracy measurements cannot be compared directly with LOLO ROC-AUC.

## Existing LOLO integrity and protocol

Sources: [Stage 1 report](cache/detectability_tinyllama_residual_20261011/detectability_stage1_report.json), [Stage 2 report](cache/detectability_tinyllama_detector_20261011/detectability_stage2_report.json), [detector summary](cache/detectability_tinyllama_detector_20261011/detector_summary.csv) and [fold metrics](cache/detectability_tinyllama_detector_20261011/detector_fold_metrics.csv). Copies and original hashes are in the [frozen evidence record](artifacts/closeout_20261016/evidence.json).

| Check | Status | Recorded evidence |
|---|---|---|
| Five named methods present | PASS | sign, magnitude_aware, qae, lwe_grid_parity, split_sign_parity |
| Paired construction | PASS | 154,880 groups keyed by method/layer/block; exactly one clean and one embedded row each; 0 invalid pairs |
| Features | PASS for this input | 14 residual summary features; 0 missing values; runner rejects nonfinite values |
| Layer separation | PASS | All clean and embedded blocks from a held-out layer remain together; 11 folds |
| Decision threshold | Fixed at 0.5 | Specified in advance; never selected on held-out layers |
| Model-held-out comparison | BLOCKED | One model/run only; no model-level holdout possible from this input |
| Independent clean controls / replications | NOT MEASURED | Correlated blocks and related layers; no independent clean-vs-clean artifact experiment |
| Sign/QAE independence | Not supported for this configuration | Stage 2 records a prior audit finding equal tensors/carriers; summary metrics coincide. Equality explains the two methods' duplicate outcomes, not why clean-vs-embedded scores are near chance. |

Input feature SHA-256: `e550f412f80787441a21af5a508c13644fa133683426e7ac20f715508bb42bc3`; recomputed and matched to Stage 2. Fourteen features: mean, std, min, max, median, q25, q75, mean-square, positive fraction, skew, excess kurtosis, histogram entropy, zero fraction and absolute mean (`f_` column prefixes).

## Measured diagnostic results

| Method | Detector | Pooled OOF ROC-AUC | Mean layer ROC-AUC | SD across layers |
|---|---|---:|---:|---:|
| sign | Logistic Regression | 0.5034 | 0.5055 | 0.0158 |
| sign | HistGradientBoosting | 0.5045 | 0.5059 | 0.0123 |
| magnitude_aware | Logistic Regression | 0.6214 | 0.6213 | 0.0280 |
| magnitude_aware | HistGradientBoosting | 0.8258 | 0.8284 | 0.0599 |
| qae | Logistic Regression | 0.5034 | 0.5055 | 0.0158 |
| qae | HistGradientBoosting | 0.5045 | 0.5059 | 0.0123 |
| lwe_grid_parity | Logistic Regression | 0.7842 | 0.7883 | 0.0432 |
| lwe_grid_parity | HistGradientBoosting | 0.8273 | 0.8328 | 0.0445 |
| split_sign_parity | Logistic Regression | 0.6420 | 0.6441 | 0.0207 |
| split_sign_parity | HistGradientBoosting | 0.6822 | 0.6839 | 0.0224 |


Sign and QAE are near chance for these detectors on this input. Magnitude-aware, LWE grid/parity and split sign/parity yield higher AUC. These are measured within-model outcomes, not invalid merely because they are exploratory. The three higher-scoring methods cannot be described as generally hidden, and their scores cannot be extrapolated to the seven-model exp18 roster. Fold SD describes layer variation; it is not a confidence interval over independent models.

No security certification follows from the LWE name. [exp13](results/exp13_keyless_recovery.json) records a public width of 0.01, inactive keyed branch and a keyless BER-0 attack (security gate FAIL). The revised Phase 7B tests document this limitation; they do not repair it. exp18's 0.50 detector accuracy remains a result for that detector, even though a different diagnostic detects structure.

**Exit gate met:** the existing diagnostic is retained with its exact scope, and the blocker for a new model-held-out five-method comparison is explicit. Broadening model coverage, collecting independent controls or changing the embedding would be separate research work, outside this closeout. [Phase 3](PHASE3_CONSOLIDATED_SEVEN_METHOD.md) presents both protocols without merging their conclusions.
