# NES Multi-Model Research — Final Summary

Generated: 2026-10-02T08:52:37.801175+00:00

This summary is generated from `results/experiment_manifest.json`. Every number below corresponds to a saved artifact.

## 1. Coverage

| Status | Cells |
| --- | --- |
| PASS | 6 |
| FAIL | 3 |
| NOT_RUN | 2 |

Total cells: 11. PASS and FAIL both represent completed experiments; only FAIL means the experiment ran and its gate was not met.

## 2. Model x experiment matrix

| Model | exp1 | exp2 | exp3 | exp4 | exp5 | exp6 | exp7 | exp7_neural | exp8 | exp9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `meta-llama/Llama-3.1-8B` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `mistralai/Mistral-7B-v0.3` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `google/gemma-2-9b` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `Qwen/Qwen2.5-7B` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `Qwen/Qwen2.5-3B` | PASS | **FAIL** | PASS | PASS | PASS | PASS | PASS | **FAIL** | **FAIL** | NOT_RUN |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `microsoft/Phi-3-mini-4k-instruct` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

### Missing cells (7)

| Model | Experiment |
| --- | --- |
| `meta-llama/Llama-3.1-8B` | all |
| `mistralai/Mistral-7B-v0.3` | all |
| `google/gemma-2-9b` | all |
| `Qwen/Qwen2.5-7B` | all |
| `Qwen/Qwen2.5-3B` | exp9 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | all |
| `microsoft/Phi-3-mini-4k-instruct` | all |

## 3. Cross-model gate table

| Model | Layers (exp/act) | g2_qaci | g3_ber | g4_ppl | g5_robustness | g6_statistical | g6_neural_detector | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-3B` | 36/36 | PASS | PASS | PASS | PASS | PASS | **FAIL** | **FAIL** |

## 4. Key findings

### 4.1 Neural steganalysis (Exp7 neural)

- Status: **FAIL**
- Detector accuracy: 70.5% (gate <= 55%)
- A carrier-centered MLP detects the embedding above threshold even though the statistical gate passes. This is a real FAIL and is preserved as such.
- Scope: paired patches from a single model. This is a stress test of the embedding, not a claim that a detector trained this way generalizes to other models or carriers.

### 4.2 Statistical security (Exp7 statistical)

- Status: **PASS**
- KL divergence: 4.17361e-05 (gate <= 0.05)
- Statistical detector accuracy: 50.28% (gate <= 55%)
- This gate and the neural gate are reported separately and are not combined into a single security score.

### 4.3 Capacity

- Status: **PASS**
- **Maximum tested** payload at BER=0: 10,000,000 bits
- This is the largest payload tested, not an absolute capacity bound. No stopping criterion for the true maximum was established.

### 4.4 Fidelity

- Status: **PASS**
- NF4 baseline PPL: 12.4707
- Reconstruction control PPL: 11.3494
- Embedded PPL: 11.3488
- Embedding-specific degradation: 0.00528662% (gate < 2%)

### 4.5 Non-NF4 formats (Exp9)

- `Qwen/Qwen2.5-3B-AWQ` (AWQ): **NOT_RUN** — Checkpoint Qwen/Qwen2.5-3B-AWQ is not in the local model cache. Exp9 needs a real AWQ checkpoint; downloading it is an explicit step, not something a suite run does implicitly.
- `Qwen/Qwen2.5-3B-GPTQ-Int4` (GPTQ): **NOT_RUN** — Checkpoint Qwen/Qwen2.5-3B-GPTQ-Int4 is not in the local model cache. Exp9 needs a real GPTQ checkpoint; downloading it is an explicit step, not something a suite run does implicitly.
- GPTQ and AWQ dequantization adapters are implemented and verified bit-exact against the AutoGPTQ reference unpack. No GPTQ or AWQ checkpoint has been run yet, so these cells carry no experimental evidence.

## 5. Architecture support

Registry entries and experimental validation are different things. Only the families below have been exercised.

| Family | Status |
| --- | --- |
| falcon | NOT_VALIDATED |
| gemma | VALIDATED |
| llama | VALIDATED |
| mistral | VALIDATED |
| mixtral | NOT_VALIDATED |
| moe | NOT_VALIDATED |
| phi3 | VALIDATED |
| qwen | VALIDATED |

Falcon and MoE families are not validated. Their presence in the registry does not imply support.

## 6. Methodological caveats

- The LWE component is **LWE-inspired** (an HMAC-SHA256 keyed interval/grid mechanism), not a mathematically validated LWE cryptosystem.
- QRNG infrastructure exists but no working production QRNG provider has been established.
- Exp6 and Exp7 are stochastic. Seeds and trial counts are recorded in each artifact; small drift in non-gate values between runs is expected.
- A FAIL in this table means the experiment ran and did not meet its gate. It is never converted to PASS, and a missing cell is never reported as one.

## 7. Artifacts

- `results/cross_model_table_real.json`
- `results/exp1_qwen__qwen2.5-3b.json`
- `results/exp2_qwen__qwen2.5-3b.json`
- `results/exp3_qwen__qwen2.5-3b.json`
- `results/exp4_qwen__qwen2.5-3b.json`
- `results/exp5_qwen__qwen2.5-3b.json`
- `results/exp6_qwen__qwen2.5-3b.json`
- `results/exp7_neural_detector_results.json`
- `results/exp7_neural_qwen__qwen2.5-3b.json`
- `results/exp7_qwen__qwen2.5-3b.json`
- `results/exp8_cross_model.json`
- `results/exp9_formats.json`
- `results/experiment_manifest.json`
- `results/experiment_matrix.json`
- `results/table3_ppl_task_accuracy.json`

Superseded artifact versions are preserved under `results/_archive/` rather than overwritten in place.

