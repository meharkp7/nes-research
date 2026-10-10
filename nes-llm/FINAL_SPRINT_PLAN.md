# NES Research — fast completion sprint

**Status date:** 10 October 2026  
**Branch:** `research/contract-b-b14-and-evaluation`  
**Policy:** keep the PR unmerged; preserve historical failures and frozen reports. Do not launch more isolated feature ablations unless a result changes the paper's main claim.

## Finish criterion

Close the project with a defensible answer to the research question: how reliably does the payload recover, what transformations break it, how detectable is it under independent and cross-model evaluation, and what is the measured utility/distortion trade-off? A limitation or negative result is a valid endpoint; universal stealth or universal robustness is not required and must not be claimed.

## Current evidence — freeze these conclusions

- Established Exp1–Exp27 line: broad capacity, recovery, utility, detector, and transformation-specific evidence exists; retain the recorded failures and caveats in `FINAL_RESEARCH_FLOW.md`.
- Artifact-only Contract B B1.4: pristine packed-NF4 recovery succeeded for 10,000 bits (BER 0); the observed fresh NF4 requantization lifecycle produced BER 0.003 and checksum failure.
- DCE: synthetic multi-seed batch matching improves histogram TV/KL at roughly 8.66x baseline distortion; not an overall win. Do not wire it into production or spend more time tuning the current proxy.
- Packed-NF4 detector benchmark: mixed-model RF test AUC 0.6641, driven by Qwen (AUC 0.7689); TinyLlama and Gemma are near chance (~0.52). Grouped Qwen-only OOF AUC 0.8913 varies strongly by held-out projection/variant group. This is not model-generalized evidence.
- Qwen ablation: full features AUC 0.7699; without entropy 0.7668; entropy-only 0.5422. Entropy alone does not explain the signal. Treat as diagnostic only.
- Existing reports, datasets, and frozen pilot remain unchanged.

## Parallel execution plan

### Batch A — one local integrity gate (fast; run once)
Run the relevant test suites together after pulling this branch. Include the detector benchmark and ablation, Contract B sender/receiver and robustness-matrix tests, and DCE optimizer/comparison tests. Fix failures at source; do not relax readiness gates or overwrite any report.

### Batch B — model generalization + provenance (same priority)
1. Verify what `source_id` and `run_id` mean, and whether clean/embedded pairs differ only by the embedding operation.
2. Use `scripts/run_packed_nf4_cross_model_detector.py` to train on two model families and test on the third, rotating the held-out model. Preprocessing is fitted on training models only; the primary transfer metric is ROC-AUC, with fixed-threshold secondary metrics and group counts. This is a new evaluation, not yet locally run.
3. Keep the existing mixed-model and Qwen-only results as exploratory comparisons, not the headline generalization result.

Run from `nes-llm/` after pulling the branch:

```bash
../.venv/bin/python -m pytest -q tests/test_run_packed_nf4_cross_model_detector.py
../.venv/bin/python scripts/run_packed_nf4_cross_model_detector.py \\
  --dataset ../cache/packed_nf4_grouped_dataset_20261010.csv \\
  --output ../cache/packed_nf4_cross_model_detector_$(date +%Y%m%d_%H%M%S).json
```
4. If there are too few independent runs or matched provenance to support the split, mark the generalization claim blocked; do not manufacture independence by splitting rows.

### Batch C — packed-NF4 lifecycle attacks (parallel to Batch B)
Run independent B1.4 transformations on separate copies of the pristine artifact: reload/save control, pruning, fine-tuning or LoRA merge, task/weight merge if a valid controlled setup exists, and fresh requantization. Run the receiver after every transformation and record BER, checksum/recovery, changed-code count, and utility where feasible. Preserve each input/output/report; never mutate the pristine artifact. Do not treat the older residual-domain Exp23 results as B1.4 evidence.

### Batch D — utility + final trade-off
Expand utility evaluation only enough to support a defensible comparison: held-out prompts, original-vs-embedded baseline, fixed decoding/scoring protocol, and a declared acceptance threshold. Consolidate payload capacity/recovery, utility/distortion, detector results, and attack outcomes into one evidence table. Clearly separate model families, embedding protocols, and transformation types.

### Batch E — evidence freeze and paper
Run the claim audit, consistency checks, and reproducibility checks once after code/results stop changing. Freeze the report bundle, figures/tables, exact commands, seeds, versions, limitations, and failed gates. Draft the paper from measured evidence; do not start an SDK/dashboard or new embedding family before submission.

## Stop rules

- No more single-feature ablations unless they resolve a paper-critical uncertainty.
- No more DCE weight tuning for the current independent per-carrier objective.
- No claims of universal undetectability, cryptographic lattice security, or universal robustness.
- A failed attack, failed gate, or blocked independent split is a reportable result, not a reason to keep experimenting indefinitely.
- Only run expensive model transformations after confirming the required checkpoint, disk space, and output paths; all output paths must be new.

## Completion checklist

- [x] Existing core experiment record and historical failures documented.
- [x] B1.4 pristine recovery and fresh NF4-requantization failure documented.
- [x] Synthetic DCE trade-off evaluated across five seeds; current objective stopped.
- [x] Consolidated packed-NF4 detector benchmark implemented and tested.
- [x] Qwen entropy ablation implemented and tested.
- [ ] Provenance/source-run meaning checked against actual artifact generation.
- [x] Cross-model holdout benchmark implementation and synthetic tests added; local dataset run pending.
- [ ] B1.4 lifecycle attack matrix completed for feasible transformations.
- [ ] Utility comparison expanded to held-out prompts and fixed acceptance criteria.
- [ ] Final claim audit, consistency checks, and reproducibility bundle frozen.
- [ ] Paper figures/tables and final limitations aligned to the frozen evidence.
