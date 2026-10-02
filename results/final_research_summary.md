# NES Multi-Model Research — Final Summary

Generated: 2026-10-02T11:36:56.977270+00:00

This summary is generated from `results/experiment_manifest.json`. Every number below corresponds to a saved artifact.

## 1. Coverage

| Status | Cells |
| --- | --- |
| PASS | 19 |
| FAIL | 5 |
| NOT_RUN | 2 |

Total cells: 26. PASS and FAIL both represent completed experiments; only FAIL means the experiment ran and its gate was not met.

## 2. Model x experiment matrix

| Model | exp1 | exp2 | exp3 | exp4 | exp5 | exp6 | exp7 | exp7_neural | exp8 | exp9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `meta-llama/Llama-3.1-8B` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `mistralai/Mistral-7B-v0.3` | PASS | PASS | PASS | NOT_RUN | NOT_RUN | PASS | PASS | NOT_RUN | NOT_RUN | NOT_RUN |
| `google/gemma-2-9b` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `Qwen/Qwen2.5-7B` | PASS | **FAIL** | PASS | NOT_RUN | NOT_RUN | PASS | PASS | NOT_RUN | NOT_RUN | NOT_RUN |
| `Qwen/Qwen2.5-3B` | PASS | **FAIL** | PASS | PASS | PASS | PASS | PASS | **FAIL** | **FAIL** | NOT_RUN |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | PASS | **FAIL** | PASS | NOT_RUN | NOT_RUN | PASS | PASS | NOT_RUN | NOT_RUN | NOT_RUN |
| `microsoft/Phi-3-mini-4k-instruct` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

### Missing cells (19)

| Model | Experiment |
| --- | --- |
| `meta-llama/Llama-3.1-8B` | all |
| `mistralai/Mistral-7B-v0.3` | exp4 |
| `mistralai/Mistral-7B-v0.3` | exp5 |
| `mistralai/Mistral-7B-v0.3` | exp7_neural |
| `mistralai/Mistral-7B-v0.3` | exp8 |
| `mistralai/Mistral-7B-v0.3` | exp9 |
| `google/gemma-2-9b` | all |
| `Qwen/Qwen2.5-7B` | exp4 |
| `Qwen/Qwen2.5-7B` | exp5 |
| `Qwen/Qwen2.5-7B` | exp7_neural |
| `Qwen/Qwen2.5-7B` | exp8 |
| `Qwen/Qwen2.5-7B` | exp9 |
| `Qwen/Qwen2.5-3B` | exp9 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | exp4 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | exp5 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | exp7_neural |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | exp8 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | exp9 |
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

## 5. Why the FAILs fail

A FAIL is a measurement, not a bug. Each was investigated to determine whether it is fixable; the answer is recorded rather than assumed.

### 5.1 Exp2 residual-magnitude gate

- 1/7 profiled models meet the `mag_mean > 0.002` criterion.
- `nf4_double_quant`: mean magnitude 0.001879, 0% of probed layers above threshold
- `nf4_single_quant`: mean magnitude 0.001877, 0% of probed layers above threshold
- `fp4`: mean magnitude 0.002526, 100% of probed layers above threshold
- Zero of 7 profiled models meet the mag_mean > 0.002 criterion under NF4. The same statistic under FP4 clears it. The threshold is therefore sensitive to the quantization format rather than being a per-model property.
- The threshold was **not** changed and no Exp2 verdict was rewritten. Recalibrating it is a research decision for the authors, not something to apply silently.

### 5.2 Exp7 neural-detector gate

- Swept 6 configurations (alpha x100, gamma x5, payload x10): detector accuracy stayed within 69.38%-74.38%.
- No configuration came near the 55% gate, so this is **not fixable by retuning**.
- Mechanism: sign embedding rewrites a carrier to +/-|r|, which changes nothing when the payload bit already agrees with the carrier's sign. Only ~25 of 4096 values per patch actually differ and 6% of pairs are byte-identical.
- No variant reaches the 55% gate: accuracy stays in 69.38%-74.38% (spread 5.00%), i.e. at least 14.4% above the threshold across a 100x change in alpha, a 5x change in gamma and a 10x change in payload size. Detectability is structural to sign-based embedding at carrier positions, not a tuning problem, and the recorded 70.5% FAIL is not fixable by retuning these parameters.
- Getting under the gate would need a different embedding scheme, one that does not force a sign flip at carriers. That is future work, not a retuning.

## 6. Architecture support

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

## 7. Methodological caveats

- The LWE component is **LWE-inspired** (an HMAC-SHA256 keyed interval/grid mechanism), not a mathematically validated LWE cryptosystem.
- QRNG infrastructure exists but no working production QRNG provider has been established.
- Exp6 and Exp7 are stochastic. Seeds and trial counts are recorded in each artifact; small drift in non-gate values between runs is expected.
- A FAIL in this table means the experiment ran and did not meet its gate. It is never converted to PASS, and a missing cell is never reported as one.

## 8. Artifacts

- `results/cross_model_table_real.json`
- `results/exp1_meta-llama__llama-3.1-8b.json`
- `results/exp1_mistralai__mistral-7b-v0.3.json`
- `results/exp1_qwen__qwen2.5-3b.json`
- `results/exp1_qwen__qwen2.5-7b.json`
- `results/exp1_tinyllama__tinyllama-1.1b-chat-v1.0.json`
- `results/exp2_criterion_calibration.json`
- `results/exp2_meta-llama__llama-3.1-8b.json`
- `results/exp2_mistralai__mistral-7b-v0.3.json`
- `results/exp2_qwen__qwen2.5-3b.json`
- `results/exp2_qwen__qwen2.5-7b.json`
- `results/exp2_tinyllama__tinyllama-1.1b-chat-v1.0.json`
- `results/exp3_meta-llama__llama-3.1-8b.json`
- `results/exp3_mistralai__mistral-7b-v0.3.json`
- `results/exp3_qwen__qwen2.5-3b.json`
- `results/exp3_qwen__qwen2.5-7b.json`
- `results/exp3_tinyllama__tinyllama-1.1b-chat-v1.0.json`
- `results/exp4_qwen__qwen2.5-3b.json`
- `results/exp5_qwen__qwen2.5-3b.json`
- `results/exp6_mistralai__mistral-7b-v0.3.json`
- `results/exp6_qwen__qwen2.5-3b.json`
- `results/exp6_qwen__qwen2.5-7b.json`
- `results/exp6_tinyllama__tinyllama-1.1b-chat-v1.0.json`
- `results/exp7_mistralai__mistral-7b-v0.3.json`
- `results/exp7_neural_detector_results.json`
- `results/exp7_neural_parameter_study.json`
- `results/exp7_neural_qwen__qwen2.5-3b.json`
- `results/exp7_qwen__qwen2.5-3b.json`
- `results/exp7_qwen__qwen2.5-7b.json`
- `results/exp7_tinyllama__tinyllama-1.1b-chat-v1.0.json`
- `results/exp8_cross_model.json`
- `results/exp9_formats.json`
- `results/experiment_manifest.json`
- `results/experiment_matrix.json`
- `results/table3_ppl_task_accuracy.json`

Superseded artifact versions are preserved under `results/_archive/` rather than overwritten in place.

