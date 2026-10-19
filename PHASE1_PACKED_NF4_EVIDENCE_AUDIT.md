# Phase 1 — Packed-NF4 Evidence Audit

**Reviewed:** 2026-10-16 (Asia/Kolkata). **Status:** evidence reconciliation complete; scientific limitations remain.

In the **20261011 model-grouped split**, the two tested detectors performed near chance for QSE and DCE. This is a descriptive result on the evaluated block features. Historical source-revision matching remains **UNVERIFIED**. The 20261010 evaluation is a different dataset and protocol and remains separately reported below. No classifier or dataset was rerun during closeout.

## Dated comparison

| Evaluation | Dataset / allocation | QSE metrics | DCE metrics | Detector and threshold protocol | Scope / provenance |
|---|---|---|---|---|---|
| **20261010 historical** | 606,208 rows; 8 source/run pairs; 3 model IDs. Preparation manifest: TinyLlama train, Qwen2.5-3B validation, Gemma-2-2b test (1/1/1). The benchmark instead uses a mixed pair split: 363,726 / 121,242 / 121,240 rows. | NOT SEPARATELY REPORTED as a method-specific arm; historical pooled metrics below | NOT SEPARATELY REPORTED; cannot assign the pooled historical result to DCE | LR, HGB, RF, MLP; mixed-pair test with validation-only Youden J; separate 4-fold source/run GroupKFold diagnostic with fixed 0.5 threshold | All 3 model IDs occur in each mixed partition. Model revisions and checkpoint independence UNVERIFIED. |
| **20261011 grouped** | 6,111,232 rows; 21 unique artifact hashes; 14 method/source/run pairs across 7 model IDs. Train/validation/test: 4/1/2 model groups, 3,031,040 / 1,048,576 / 2,031,616 rows | LR AUC/accuracy 0.5005/0.5003; HGB 0.5004/0.5003 | LR AUC/accuracy 0.5011/0.5007; HGB 0.5010/0.5008 | LR and HGB fitted separately per method; train-only preprocessing; external validation selects Youden J; fixed test evaluated once | Two held-out model IDs, one validation model. Correlated blocks; historical QSE source-revision match UNVERIFIED. |

The historical preparation split and the historical benchmark's mixed split are **different manifests**. One model per preparation partition limits coverage; it does not mean model IDs crossed that preparation split. The mixed benchmark explicitly permits source/run groups across partitions.

## Dataset and report fingerprints

SHA-256 values were recomputed from file bytes on 2026-10-16. All hashes stored by the detector reports, and all 44 input feature-file hashes in the two preparation manifests, match. The [frozen evidence record](artifacts/closeout_20261016/evidence.json) contains these hashes and copies of the small reports, keyed by original repository path.

| Date / object | Source | SHA-256 |
|---|---|---|
| 20261010 / dataset | [cache/packed_nf4_grouped_dataset_20261010.csv](cache/packed_nf4_grouped_dataset_20261010.csv) | `1b815887d73e40463bb7694d4e921108fb6bb2f29c1e8765ab4cb7240a60c58c` |
| 20261010 / preparation split | [cache/packed_nf4_grouped_splits_20261010.json](cache/packed_nf4_grouped_splits_20261010.json) | `72d63936a16b11b75fd1b3089434826608939adae1f371aa71e45325ab399565` |
| 20261010 / benchmark mixed split | [cache/packed_nf4_detector_benchmark_diagnostics_20261010_194641/mixed_model_split_manifest.json](cache/packed_nf4_detector_benchmark_diagnostics_20261010_194641/mixed_model_split_manifest.json) | `42a376fe39f2776a4eee72e1878e17042387f83bf7be27dbbba5acc27d68d6a5` |
| 20261010 / detector report | [cache/packed_nf4_detector_benchmark_diagnostics_20261010_194641/packed_nf4_detector_benchmark.json](cache/packed_nf4_detector_benchmark_diagnostics_20261010_194641/packed_nf4_detector_benchmark.json) | `ca2563b764309b5dd21492934d313e116f87e1c92d498684637215a2731d52d1` |
| 20261011 / dataset | [cache/packed_nf4_grouped_dataset_20261011.csv](cache/packed_nf4_grouped_dataset_20261011.csv) | `2ad0cf1c4a9c845d01b6250a46913216e12cb55df8da740543f478317f9348a7` |
| 20261011 / split | [cache/packed_nf4_grouped_splits_20261011.json](cache/packed_nf4_grouped_splits_20261011.json) | `974e39e7f78ed5a3dfdafd45edcbabe66ccc7f52acac4752052740056504be4f` |
| 20261011 / detector report | [cache/packed_nf4_method_eval_20261011/method_specific_detector_report.json](cache/packed_nf4_method_eval_20261011/method_specific_detector_report.json) | `0fd716983942a222f0501eb45758b06ae9f7b2d9bdccc2aef62f80e00179f9dc` |

## 20261011 method-specific results

Source: [cache/packed_nf4_method_eval_20261011/method_specific_detector_report.json](cache/packed_nf4_method_eval_20261011/method_specific_detector_report.json) → `methods.<method>.models.<detector>.test_pooled`. Values below are rounded from that report, not new measurements. Each method has 1,015,808 test rows (507,904 per label).

| Method | Detector | Test ROC-AUC | Test accuracy | Validation-selected threshold |
|---|---|---:|---:|---:|
| QSE | Logistic Regression | 0.5005 | 0.5003 | 0.5003620386 |
| QSE | HistGradientBoosting | 0.5004 | 0.5003 | 0.5001840161 |
| DCE | Logistic Regression | 0.5011 | 0.5007 | 0.4995805025 |
| DCE | HistGradientBoosting | 0.5010 | 0.5008 | 0.4997909209 |


| Partition | Model IDs |
|---|---|
| Train | Qwen/Qwen2.5-7B; TinyLlama/TinyLlama-1.1B-Chat-v1.0; google/gemma-2-9b; meta-llama/Llama-3.1-8B |
| Validation | mistralai/Mistral-7B-v0.3 |
| Test | Qwen/Qwen2.5-3B; microsoft/Phi-3-mini-4k-instruct |

The saved report says “two model families are held out.” The precise claim supported by its manifest is **two held-out model IDs**: Qwen2.5-7B is in training while Qwen2.5-3B is in test. Thus this is not a complete unseen-family holdout. The historical report is preserved verbatim; this audit bounds its wording.

## 20261010 historical results

Source: [cache/packed_nf4_detector_benchmark_diagnostics_20261010_194641/packed_nf4_detector_benchmark.json](cache/packed_nf4_detector_benchmark_diagnostics_20261010_194641/packed_nf4_detector_benchmark.json) → `primary_mixed_block_evaluation` and `grouped_leakage_sensitivity`. These numbers must not be replaced by the 20261011 method-specific values.

| Detector | Mixed-block test AUC | Mixed-block accuracy | GroupKFold OOF AUC | GroupKFold OOF accuracy |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.5125 | 0.5092 | 0.5109 | 0.5085 |
| HistGradientBoosting | 0.5124 | 0.5084 | 0.5103 | 0.5089 |
| Random Forest | 0.6641 | 0.5919 | 0.7501 | 0.6485 |
| MLP | 0.5122 | 0.5084 | 0.5120 | 0.5084 |


GroupKFold uses 8 source/run groups and 4 folds, each holding out 2 groups. It does not hold out model identity consistently. Random Forest's above-chance results remain visible: the historical dataset does not justify saying all tested detectors performed near chance. The separate [Qwen-only ablation](cache/packed_nf4_detector_ablation_20261010_205523/packed_nf4_detector_ablation.json) reports full-feature RF AUC 0.7699 versus entropy-only 0.5422, under a random matched-pair split with possible source/run overlap. It is not a QSE-versus-DCE comparison.

## Readiness, provenance and limitations

| Check | Status | Evidence / interpretation |
|---|---|---|
| Current dataset readiness | PASS | [cache/packed_nf4_readiness_20261011.json](cache/packed_nf4_readiness_20261011.json): 6,111,232 rows, 21 artifacts, 7 model groups, 4/1/2 allocation, no errors or warnings. Report names the evaluated dataset and split. |
| Earlier readiness attempt | BLOCKED, preserved | `cache/packed_nf4_readiness_7models_20261011.json` refers to another filename and records duplicate clean blocks. It is not the current PASS report. Its fingerprint is preserved in the evidence record. |
| Matched blocks and split assignment | PASS for recorded preflight | Readiness checks roles/labels, paired block-index sets, artifact identity and model-to-split agreement. Adding source/run to duplicate identity allows shared clean controls across method runs; artifact hash still prevents conflicting role/model/tensor/split identities. |
| Provenance audit execution | PASS (`passed`) | [cache/packed_nf4_provenance_audit_20261010.json](cache/packed_nf4_provenance_audit_20261010.json): 8 pairs / 3 IDs. [cache/packed_nf4_provenance_audit_20261011.json](cache/packed_nf4_provenance_audit_20261011.json): 14 pairs / 7 IDs. This establishes recorded pairing, not checkpoint independence. |
| Historical source-revision matching | UNVERIFIED | The method-specific report explicitly retains `historical_qse_revision_match UNVERIFIED`; identifiers and CSV hashes do not verify original checkpoints. |
| Independent source checkpoints | UNVERIFIED | Distinct model IDs and artifact hashes alone cannot establish independence. Seven clean artifact hashes are reused between QSE and DCE; 14 method pairs are not 14 independent models. |
| Direct metadata features | Excluded | The 22 numeric block features exclude label, role, model, artifact, run, source and split metadata. This does not prove absence of indirect source or quantization effects. |
| Independent clean-vs-clean calibration | NOT MEASURED in the newer report | Historical benchmark explicitly reports `NOT_AVAILABLE_IN_CURRENT_DATASET`; shared clean controls are not independent replications. |
| Broad detectability or utility claims | NOT ESTABLISHED | One validation ID, two test IDs, family overlap, correlated blocks and no independent-block confidence intervals. No full-model PPL or robustness result follows from these detector metrics. |

The saved 20261011 preparation manifest still contains a historical “three distinct model IDs” guardrail. The current preparation policy requires at least five (2 train + 1 validation + 2 test); its actual seven-group allocation satisfies that policy. We preserve the manifest bytes and explain the stale prose here.

**Exit gate met:** dates, metrics, split definitions, fingerprints and provenance limits have been reconciled. Recovery evidence is reported separately in [Phase 3](PHASE3_CONSOLIDATED_SEVEN_METHOD.md); a detector result is not a recovery or utility result.
