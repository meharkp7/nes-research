# NES Research — fast completion sprint

**Status date:** 10 October 2026  
**Branch:** `research/contract-b-b14-and-evaluation`  
**Policy:** keep the PR unmerged; preserve historical failures and frozen reports. Do not launch more isolated feature ablations unless a result changes the paper's main claim.

## Finish criterion

Close the project with a defensible answer to the research question: how reliably does the payload recover, what transformations break it, how detectable is it under independent and cross-model evaluation, and what is the measured utility/distortion trade-off? A limitation or negative result is a valid endpoint; universal stealth or universal robustness is not required and must not be claimed.

## Current evidence — freeze these conclusions

- Established Exp1–Exp27 line: broad capacity, recovery, utility, detector, and transformation-specific evidence exists; retain the recorded failures and caveats in `FINAL_RESEARCH_FLOW.md`.
- Artifact-only Contract B B1.4: pristine packed-NF4 recovery succeeded for 10,000 bits (BER 0); fresh NF4 requantization produced BER 0.003 and checksum failure. A separate 10% selected-tensor magnitude-pruning + fresh NF4 requantization attack completed twice and both runs produced an invalid envelope header and checksum failure; BER unavailable. This repeated qualitative result is for the combined transformation, not isolated evidence about pruning.
- DCE: synthetic multi-seed batch matching improves histogram TV/KL at roughly 8.66x baseline distortion; not an overall win. Do not wire it into production or spend more time tuning the current proxy.
- Packed-NF4 detector benchmark: mixed-model RF test AUC 0.6641, driven by Qwen (AUC 0.7689); TinyLlama and Gemma are near chance (~0.52). Grouped Qwen-only OOF AUC 0.8913 varies strongly by held-out projection/variant group. This is not model-generalized evidence.
- Cross-model leave-one-family-out benchmark (dataset SHA-256 `1b815887d73e40463bb7694d4e921108fb6bb2f29c1e8765ab4cb7240a60c58c`): all three estimators are near chance on every held-out family. Qwen AUC 0.5010–0.5014; TinyLlama 0.5141–0.5289; Gemma 0.5118–0.5223. This does not support transferable detector signal. TinyLlama and Gemma each have only one held-out source/run group, and the Qwen holdout has only two training groups, so the estimates are exploratory and group-limited—not proof of universal undetectability.
- Qwen ablation: full features AUC 0.7699; without entropy 0.7668; entropy-only 0.5422. Entropy alone does not explain the signal. Treat as diagnostic only.
- Provenance audit of the tracked manifest adapter: `source_id` can default to `revision-unrecorded`; `run_id` is caller-supplied and not independently validated. Therefore group IDs do not prove independent replications. The original local generation logs still need checking; see `B1_4_LIFECYCLE_MATRIX.md`.
- Existing reports, datasets, and frozen pilot remain unchanged.

## Parallel execution plan

### Batch A — one local integrity gate (fast; run once)
Run the relevant test suites together after pulling this branch. Include the detector benchmark and ablation, Contract B sender/receiver and robustness-matrix tests, and DCE optimizer/comparison tests. Fix failures at source; do not relax readiness gates or overwrite any report.

### Batch B — model generalization + provenance (same priority)
1. Verify what `source_id` and `run_id` mean, and whether clean/embedded pairs differ only by the embedding operation.
2. Use `scripts/run_packed_nf4_cross_model_detector.py` to train on two model families and test on the third, rotating the held-out model. Preprocessing is fitted on training models only; the primary transfer metric is ROC-AUC, with fixed-threshold secondary metrics and group counts. **Completed locally**; results are summarized in Current evidence. No held-out family shows meaningful cross-model discrimination.
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
- [x] Tracked manifest-adapter semantics audited; independent provenance is explicitly **not established** by the adapter alone. Original local generation logs/artifact hashes still require a bounded local check.
- [x] Cross-model holdout benchmark implementation, synthetic tests, and local dataset run completed; all held-out-family AUCs near chance, with source/run group-count limitations recorded.
- [x] B1.4 lifecycle matrix and per-run evidence requirements documented in `B1_4_LIFECYCLE_MATRIX.md`; requantization negative result and direct NF4 reload/save serializer blocker frozen.
- [x] Controlled selected-tensor 10% magnitude-pruning + fresh NF4 requantization completed twice from fresh copies. Both recoveries failed at envelope-header validation; BER unavailable. Combined attack only, not isolated pruning. Initial autograd error, fix, and both completed runs are documented in `B1_4_LIFECYCLE_MATRIX.md`; stop repeating this attack. Fine-tuning and task merge remain optional.
- [ ] Utility evaluation script added: `scripts/contract_b_utility_eval_heldout.py` uses 32 fixed project-curated prompts distinct from the earlier 8-text diagnostic, paired per-prompt NLL differences, deterministic 10,000-replicate bootstrap CI, and a predeclared project-defined 2% upper-CI threshold. Local run and report review still pending; this is not an external benchmark or task-utility claim.
- [ ] Final claim audit, consistency checks, and reproducibility bundle frozen.
- [ ] Paper figures/tables and final limitations aligned to the frozen evidence.
