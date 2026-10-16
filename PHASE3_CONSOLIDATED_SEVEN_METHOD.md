# Phase 3 — Consolidated Seven-Method Evidence and Final Claims

**Reviewed:** 2026-10-16 (Asia/Kolkata). **Status:** evidence reconciliation complete; scientific limitations remain.

This is the current claim table. Track A contains five residual methods; Track B contains QSE and DCE packed-NF4 methods. A PASS applies only to the stated measurement or gate. “Diagnostic completed” describes execution, not a stealth pass. NOT MEASURED, BLOCKED, SKIPPED and UNVERIFIED are retained where applicable. Historical source files are unchanged; compact cache reports are copied into the [frozen evidence record](artifacts/closeout_20261016/evidence.json) so report evidence is reviewable without adding multi-gigabyte datasets to Git.

## Track A — Residual evidence by method and protocol

| Method | Experiment / model coverage | Recovery | Gaussian-noise robustness | Detectability | Status and limitation |
|---|---|---|---|---|---|
| sign | exp18; 7 model IDs (sources below) | BER 0.0; 10,256 bits per cell | BER at σ=0.001: 0.000000–0.000780; at σ=0.002: 0.000000–0.010693 | Carrier-centered neural accuracy 0.556250–0.781250; 0/7 within 0.55 gate | Recovery PASS; robustness PASS; detector FAIL 7/7. Exact message recovery recorded; separate detector fitted per model and method. |
| magnitude_aware | exp18; 7 model IDs (sources below) | BER 0.0; 10,256 bits per cell | BER at σ=0.001: 0.000000–0.000065; at σ=0.002: 0.000000–0.004160 | Carrier-centered neural accuracy 0.525000–0.781250; 1/7 within 0.55 gate | Recovery PASS; robustness PASS; detector FAIL 6/7. Exact message recovery recorded; separate detector fitted per model and method. |
| qae | exp18; 7 model IDs (sources below) | BER 0.0; 10,256 bits per cell | BER at σ=0.001: 0.000000–0.001073; at σ=0.002: 0.000000–0.009588 | Carrier-centered neural accuracy 0.556250–0.837500; 0/7 within 0.55 gate | Recovery PASS; robustness PASS; detector FAIL 7/7. Exact message recovery recorded; separate detector fitted per model and method. |
| lwe_grid_parity | exp18; 7 model IDs (sources below) | BER 0.0; 10,256 bits per cell | BER at σ=0.001: 0.000000; at σ=0.002: 0.011505–0.012806 | Carrier-centered neural accuracy 0.500000; 7/7 within 0.55 gate | Recovery PASS; robustness PASS; detector PASS 7/7. Bits recovered by own extractor; historical production decrypt failed. |
| split_sign_parity | [exp20](results/exp20_split_dial_qwen__qwen2.5_3b.json); Qwen2.5-3B only, fractions 0, 0.25, 0.5, 0.75, 1 | BER 0.0; 10,256 bits at each fraction | σ=0.001: 0.0 throughout; σ=0.002: 0.0–0.012676 | Same detector protocol; accuracies 0.7875, 0.69375, 0.7000, 0.5875, 0.5000 | Recovery/noise gates PASS 5/5; detector FAIL 4/5. Non-monotonic dial; only pure parity passes detector gate. Historical production decrypt fails for nonzero fractions. |
| sign | 20261011 TinyLlama-only LOLO; 11 layers | NOT MEASURED by detector; separate corpus recovery below | NOT MEASURED by detector | LR / HGB pooled OOF AUC 0.5034 / 0.5045; fixed threshold 0.5 | Diagnostic completed; cross-model extension BLOCKED by one-model feature coverage. No detector pass/fail gate was registered for this diagnostic. |
| magnitude_aware | 20261011 TinyLlama-only LOLO; 11 layers | NOT MEASURED by detector; separate corpus recovery below | NOT MEASURED by detector | LR / HGB pooled OOF AUC 0.6214 / 0.8258; fixed threshold 0.5 | Diagnostic completed; cross-model extension BLOCKED by one-model feature coverage. No detector pass/fail gate was registered for this diagnostic. |
| qae | 20261011 TinyLlama-only LOLO; 11 layers | NOT MEASURED by detector; separate corpus recovery below | NOT MEASURED by detector | LR / HGB pooled OOF AUC 0.5034 / 0.5045; fixed threshold 0.5 | Diagnostic completed; cross-model extension BLOCKED by one-model feature coverage. No detector pass/fail gate was registered for this diagnostic. |
| lwe_grid_parity | 20261011 TinyLlama-only LOLO; 11 layers | NOT MEASURED by detector; separate corpus recovery below | NOT MEASURED by detector | LR / HGB pooled OOF AUC 0.7842 / 0.8273; fixed threshold 0.5 | Diagnostic completed; cross-model extension BLOCKED by one-model feature coverage. No detector pass/fail gate was registered for this diagnostic. |
| split_sign_parity | 20261011 TinyLlama-only LOLO; 11 layers | NOT MEASURED by detector; separate corpus recovery below | NOT MEASURED by detector | LR / HGB pooled OOF AUC 0.6420 / 0.6822; fixed threshold 0.5 | Diagnostic completed; cross-model extension BLOCKED by one-model feature coverage. No detector pass/fail gate was registered for this diagnostic. |


**exp18 scope:** the 28 successful bit round trips are four methods × seven models, not 28 successes per method. The files record LWE bit recovery through its own extractor while `decrypt_ok` and `recovered_matches` remain false in the historical sign-only production path. Preserve both facts. For sign, magnitude_aware and qae, exact message recovery is recorded in exp18. “Robustness PASS” means BER ≤0.02 at σ=0.001 and ≤0.10 at σ=0.002; it does not mean every measured BER is zero.

Exp18 sources: [google/gemma-2-2b](results/exp18_matrix_google__gemma_2_2b.json), [google/gemma-2-9b](results/exp18_matrix_google__gemma_2_9b.json), [meta-llama/Llama-3.1-8B](results/exp18_matrix_meta_llama__llama_3.1_8b.json), [microsoft/Phi-3-mini-4k-instruct](results/exp18_matrix_microsoft__phi_3_mini_4k_instruct.json), [mistralai/Mistral-7B-v0.3](results/exp18_matrix_mistralai__mistral_7b_v0.3.json), [Qwen/Qwen2.5-3B](results/exp18_matrix_qwen__qwen2.5_3b.json), [Qwen/Qwen2.5-7B](results/exp18_matrix_qwen__qwen2.5_7b.json). Protocol: [exp10](nes-llm/src/experiments/exp10_strategy_comparison.py) / [exp18](nes-llm/src/experiments/exp18_strategy_model_matrix.py), with per-model, per-method carrier-centered neural fits and sample-ID grouping. Full LOLO metrics, fold variability and provenance appear in [Phase 2](PHASE2_RESIDUAL_DETECTOR_READINESS.md).

### Separate saved corpus round trips

| Track / method | Artifact protocol and model coverage | Recovery | Status / limit |
|---|---|---|---|
| A: sign | 20261010 residual bundle, 7 model IDs | 36,128/36,128 bits, BER 0.0; exact 7-message corpus | PASS in saved extract reports; selected tensors only |
| A: magnitude_aware | Same residual corpus protocol, 7 IDs | 36,128/36,128 bits, BER 0.0; exact corpus | PASS; no utility or detector inference |
| A: qae | Same residual corpus protocol, 7 IDs | 36,128/36,128 bits, BER 0.0; exact corpus | PASS; distinct from blocked `nf4_qae` |
| A: lwe_grid_parity | Same residual corpus protocol, 7 IDs | 36,128/36,128 bits, BER 0.0; exact corpus | PASS in artifact receiver; historical exp18 decrypt failure remains unchanged |
| A: split_sign_parity | Same residual corpus protocol, 7 IDs | 36,128/36,128 bits, BER 0.0; exact corpus | PASS for chosen configurations; not a multi-model detector result |
| B: QSE | 20261009–20261010 packed artifact reports, 7 IDs | 36,128/36,128 bits, BER 0.0; exact corpus with reference-assisted receiver | PASS for saved artifact protocol; requires receiver reference; not a deployable HF checkpoint |
| B: DCE | 20261009–20261010 packed artifact reports, 7 IDs | 36,128/36,128 bits, BER 0.0; exact corpus with packed-code parity receiver | PASS for saved artifact protocol; no full-model utility conclusion |

Corpus roster: Qwen2.5-3B, Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3, Gemma-2-9b, Phi-3-mini-4k-instruct and TinyLlama-1.1B-Chat-v1.0. This differs from exp18's roster (Gemma-2-2b instead of TinyLlama).

Track A sources: [cache/seven_method_long_matrix/20261010T184242Z/matrix_summary.csv](cache/seven_method_long_matrix/20261010T184242Z/matrix_summary.csv), [cache/seven_method_long_matrix/20261010T190121Z/matrix_summary.csv](cache/seven_method_long_matrix/20261010T190121Z/matrix_summary.csv) (30 + 5 cells). Track B sources: [cache/seven_method_long_matrix/20261009T192218Z/matrix_summary.csv](cache/seven_method_long_matrix/20261009T192218Z/matrix_summary.csv), [cache/seven_method_long_matrix/20261009T192702Z/matrix_summary.csv](cache/seven_method_long_matrix/20261009T192702Z/matrix_summary.csv), [cache/seven_method_long_matrix/20261010T190345Z/matrix_summary.csv](cache/seven_method_long_matrix/20261010T190345Z/matrix_summary.csv), [cache/seven_method_long_matrix/20261010T190634Z/matrix_summary.csv](cache/seven_method_long_matrix/20261010T190634Z/matrix_summary.csv), [cache/seven_method_long_matrix/20261010T190941Z/matrix_summary.csv](cache/seven_method_long_matrix/20261010T190941Z/matrix_summary.csv), [cache/seven_method_long_matrix/20261010T191208Z/matrix_summary.csv](cache/seven_method_long_matrix/20261010T191208Z/matrix_summary.csv), [cache/seven_method_long_matrix/20261010T191437Z/matrix_summary.csv](cache/seven_method_long_matrix/20261010T191437Z/matrix_summary.csv), [cache/seven_method_long_matrix/20261010T191745Z/matrix_summary.csv](cache/seven_method_long_matrix/20261010T191745Z/matrix_summary.csv). Every successful summary row was checked against its `extract.json`: bit counts, BER, exact match and receiver contract. These are existing measurements, not fresh extractions. Original artifacts remain local; source-revision verification is not implied.

**Failures retained:** the Qwen QSE row at `20261009T192218Z` is `BER_FAIL`, BER 0.3982783437, invalid corpus magic; its successful reference-assisted report is a later, separate record at `20261009T192702Z`. Earlier matrix reports retain `EMBED_FAILED` / `EXTRACT_FAILED` statuses in the evidence record. The initial [single-tensor candidate pilot](cache/real_nf4_candidate_eval_qwen3b_20261009_233632.json) reports QSE BER 0.4016 (4,016/10,000 bit errors; FAIL exact recovery) and DCE BER 0.0 (0/10,000; PASS parity recovery). Those values describe that pilot, not the later corpus artifact protocol.

## Track B — Packed-NF4 detector evidence

The **20261011** dataset has 6,111,232 rows, 21 unique artifacts and 7 model groups, split 4 train / 1 validation / 2 test. Each method was fitted separately. The test IDs are Qwen2.5-3B and Phi-3-mini; Qwen2.5-7B appears in training, so the result is not a complete unseen-family test. The exact source is [cache/packed_nf4_method_eval_20261011/method_specific_detector_report.json](cache/packed_nf4_method_eval_20261011/method_specific_detector_report.json); hashes, thresholds, full group allocation and historical comparison are in [Phase 1](PHASE1_PACKED_NF4_EVIDENCE_AUDIT.md).

| Method | Detector | Test ROC-AUC | Test accuracy | Validation-selected threshold |
|---|---|---:|---:|---:|
| QSE | Logistic Regression | 0.5005 | 0.5003 | 0.5003620386 |
| QSE | HistGradientBoosting | 0.5004 | 0.5003 | 0.5001840161 |
| DCE | Logistic Regression | 0.5011 | 0.5007 | 0.4995805025 |
| DCE | HistGradientBoosting | 0.5010 | 0.5008 | 0.4997909209 |


**Allowed conclusion:** in this evaluated model-grouped split, LR and HGB performed near chance on the packed-NF4 feature dataset. Source-revision matching remains **UNVERIFIED**; blocks within artifacts may be correlated. There is one validation model and two test model IDs. No universal undetectability, verified checkpoint independence or full reproducibility claim follows.

**20261010 remains separate:** [historical mixed-block benchmark](cache/packed_nf4_detector_benchmark_diagnostics_20261010_194641/packed_nf4_detector_benchmark.json) used 606,208 rows / 3 model IDs / 8 source-run groups; pooled LR/HGB AUCs were 0.5125/0.5124, while RF was 0.6641. Source/run GroupKFold OOF AUCs were 0.5109/0.5103 for LR/HGB and 0.7501 for RF. These are not method-specific QSE/DCE arms. Do not silently transfer the new QSE/DCE values into this older report or discard the RF result.

## Final evidence-backed claim audit

| Claim area | Source and measured value | Actual status | Permitted scope / limitation |
|---|---|---|---|
| Production sign recovery | [exp3 Qwen](results/exp3_qwen__qwen2.5-3b.json) and six other `results/exp3_*.json`: 48,256 compared bits, BER 0.0 | PASS 7/7 | Exact recovery under the recorded production protocol; roster differs from exp18 |
| QAE adapter | [exp17](results/exp17_qae_round_trip.json): 48,256 bits, BER 0.0; `nf4_qae` probe cannot embed | qae PASS; nf4_qae BLOCKED | Dict-adapter residual method; blocked reference contract is not evidence against every NF4 approach |
| Carrier-centered neural detection | [exp7_neural](results/exp7_neural_qwen__qwen2.5-3b.json): accuracy 0.705 against 0.55 gate | FAIL | Qwen sign embedding; failure preserved even though other detectors scored near chance |
| Blind detector | [exp14](results/exp14_blind_patch_detector.json): accuracy 0.500, tp=fp=0 | PASS screening gate; degenerate prediction caveat | Blind patches without carrier positions; chance alone is not proof of stealth |
| LWE key dependence | [exp13](results/exp13_keyless_recovery.json): width 0.01, inactive keyed branch; phase-stream attacker BER 0.0, precision/recall 1.0 | FAIL security gate | LWE-inspired grid/parity method, no formal LWE guarantee. This is a channel-read failure, distinct from AES-GCM message authentication |
| Public re-run hypothesis | [exp25](results/exp25_selection_ablation_qwen__qwen2.5_3b.json): H3 false; public-rule BER 0.4713–0.5084 | H3 FAIL / unsupported; experiment's separate gate PASS | Fixed allocation/policy ablation, Qwen sign; not the same attacker or embed as exp13 |
| Message confidentiality boundary | [exp27](results/exp27_threat_model_qwen__qwen2.5_3b.json): 0/10 wrong-key messages, 0/6 partial-access messages; raw channel BER with map/no key 0.0 | PASS stated gates | Tested AES-GCM path and access conditions only; does not establish embedding key secrecy |
| Sign utility | [exp5](results/exp5_qwen__qwen2.5-3b.json): embedding-specific signed PPL Δ **−0.0052866%**, absolute gate metric 0.0052866% | PASS | Qwen, three-way reconstruction-control protocol; not +0.0053% signed degradation and not all downstream tasks |
| LWE utility | [exp15 Qwen](results/exp15_lwe_fidelity_qwen__qwen2.5_3b.json): +0.0076998%; [Gemma-2-2b](results/exp15_lwe_fidelity_google__gemma_2_2b.json): +0.0501199% | PASS ≤2% gates | Two models, specified PPL protocol. Qwen exp15 sign re-verification gives +0.0529227%; retain its difference from the earlier exp5 −0.0052866% |
| Maximum tested sign capacity | [exp4](results/exp4_qwen__qwen2.5-3b.json): 10,000,000 requested payload bits, 10,000,264 embedded, BER 0.0 | PASS | Largest tested Qwen point; not an absolute capacity limit and no detector score at that size |
| Small-capacity detector curve | [exp26](results/exp26_capacity_scaling_qwen__qwen2.5_3b.json): 1,000–50,000 requested bits, six points at BER 0.0, statistical accuracy 0.5028 | PASS | SecurityValidator protocol only; cannot stand in for the neural detector or exp4's 10M point |
| Sign surgery | [exp23](results/exp23_model_surgery_qwen__qwen2.5_3b.json): 8/12 cells BER 0.0 including control; 4 nonzero BER cells | PASS 8; FAIL 4 | Qwen down_proj, 10,256 bits, specified perturbations only; details below |
| Mismatched QAE/LWE reader | [exp21](results/exp21_qae_lwe_qwen__qwen2.5_3b.json): BER 0.5432917; public correction 0.0 | Raw interop FAIL | Corrected structural identity is not a rescue of the original gate |
| Per-layer LWE-width variant | [exp22](results/exp22_layer_widths_qwen__qwen2.5_3b.json): magnitude-dependent width BER 0.0225884 / 0.5762806 at σ=0.001/0.002 | Robustness FAIL | Separate exploratory variants; no repair of the shipped key-independence weakness |
| Packed-NF4 detection and provenance | 20261011 report and audits in Phase 1; four method/detector AUCs 0.5004–0.5011 | Detector run completed; readiness PASS; revision match UNVERIFIED | Specific detectors/split/features; no universal stealth or independent checkpoint claim |
| Packed-NF4 full-model utility / robustness | No such measurement in the reviewed packed artifact/detector reports | NOT MEASURED | Corpus recovery, tensor RMSE and classifier scores cannot fill these cells |

Exp23 zero-BER cells: control; LoRA-shaped perturbations 0.001 and 0.01; magnitude pruning 10% and 30%; task-vector merge 0.01 and 0.05; specified 1,000-step down_proj LoRA fine-tune. Failures: NF4 re-quantization BER 0.3831903; merge 0.5 BER 0.2475624; GPTQ re-quantization BER 0.4956123; AWQ re-quantization BER 0.4107839. Each fails the exact-recovery gate. These scores do not prove useful message survival or that compression “never erases” the payload. Surgery for the other methods is NOT MEASURED in exp23.

The [152-claim audit log](artifacts/closeout_20261016/claims.txt) verifies the repository's existing claim set; it is not an automatic semantic audit of this Markdown or a provenance certificate. This closeout additionally compares the dated packed-NF4 reports, CSV summaries, per-cell extraction reports and fingerprints. Statistical, neural, LOLO and packed-block metrics retain their separate evaluation conditions. Cross-track rankings are not supported.

**Exit gate met:** all stated conclusions have sources and bounded wording; missing evidence and scientific failures remain visible. Package/review status and commit selection are recorded in [Phase 0](PHASE0_STATUS_LEDGER.md).
