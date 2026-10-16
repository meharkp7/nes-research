# NES Publication Plan — NF4 Anchor-Preserving Method

**Status:** Research plan, not a report of completed new experiments.  
**Primary repository:** https://github.com/meharkp7/nes-research  
**Research branch:** research/contract-b-b14-and-evaluation  
**Objective:** Convert existing NES work into a defensible journal submission by preserving prior evidence, validating a candidate method, and adding only experiments needed to establish a substantive contribution.

## 1. Strategic decision

Primary direction: anchor-preserving payload embedding for packed NF4 artifacts, conditional on validation. Do not name it the final winner until matched confirmation. The core method candidate is interior_nearest; interior_pair is an ablation/alternative. pair_parity and nearest_parity are required baselines.

Fallback: if the method does not establish a meaningful practical advantage, pivot to an artifact-level measurement/steganalysis/robustness paper only if a distinct, consequential and reproducible finding is supported.

## 2. Repository evidence map

### Keep as established prior work
- Exp1–Exp27 results and original manifests, with historical FAIL/BLOCKED/SKIPPED/NOT_MEASURED labels retained.
- Contract B1.4 pristine packed-NF4 artifact recovery and B1.5 requantization failure as separate evidence.
- Five residual methods: sign, magnitude_aware, qae, lwe_grid_parity, split_sign_parity.
- Two packed-NF4 methods: QSE and DCE.
- Packed-NF4 and residual track datasets, splits, detector results, and claims remain separate.

### New candidate code inspected
- nes-llm/src/experiments/nf4_anchor_channel.py
- nes-llm/scripts/run_nf4_anchor_pilot.py
- nes-llm/scripts/nf4_anchor_checkpoint.py
- nes-llm/tests/test_nf4_anchor_channel.py
- Contract B1.4/requantization runners and tests.

The anchor implementation has methods pair_parity, nearest_parity, interior_nearest, and interior_pair. Interior modes leave NF4 codes 0 and 15 untouched, restrict eligibility to blocks containing endpoint anchors, and carry no explicit carrier map/reference weights in the method metadata. The test suite includes exact legacy-control equivalence, a simplified arithmetic invariant, wrong-key rejection, and anchorless-block exclusion.

Prior pilot logs summarized in project context report encouraging tensor-level recovery for interior methods across TinyLlama and Qwen2.5-3B, while parity baselines fail after requantization. Treat these as preliminary user-recorded outcomes until raw committed artifacts and complete result manifests are verified. The public GitHub main branch is behind the October 16 research branch; never imply local, unpushed pilot results are in main.

## 3. Claim discipline

Do not claim:
- Novelty from NF4 or neural-weight payload embedding alone.
- LWE or post-quantum security from the historical LWE-inspired method.
- Universal undetectability from accuracy/AUC near chance for a limited detector.
- Universal robustness from selected transformations.
- Full checkpoint deployment from selected-tensor or tensor-bundle recovery.
- Cryptographic confidentiality from a checksum or unencrypted corpus pilot.
- Scale preservation merely because codes 0 and 15 remain present.

Every final claim should be linked to immutable result files, source revision, checkpoint hash, environment, configuration, and a named pass/fail gate.

## 4. Publication workstreams and gates

### Workstream A — Freeze and provenance
1. Create a dedicated method-paper branch from the current research branch; never amend frozen historic result records.
2. Hash candidate source files, source checkpoints/shards, payloads, configs and outputs.
3. Inventory current artifacts with a manifest that identifies committed versus local/unpushed results.
4. Re-run only minimal code/test/consistency checks required to prove the current frozen state, not the entire historical suite by default.
Exit: reproducible manifest and clean evidence ledger.

### Workstream B — Validate the mechanism
Hypothesis: endpoint-anchor-preserving carrier selection reduces scale/carrier drift under a fresh NF4 quantization pipeline.
1. Compare interior_nearest, interior_pair, pair_parity, and nearest_parity on identical source tensors, payload bytes, net payload sizes, keys, seeds, and transformations.
2. Measure per block: original/post-embedding extrema, endpoint-anchor presence, actual quantization scales/statistics, eligible carrier count, carrier-position agreement, packed-code Hamming distance, recovery BER, exact payload/hash validity.
3. Distinguish (i) scale remains stable, (ii) scale changes but carrier map remains stable, and (iii) scale/carrier map changes. Build explicit failure fingerprints.
4. Test sequential requantization cycles and use identical clean/stego transformations. Report exact per-cycle outcomes, not only final BER.
5. Ablate anchor preservation and each candidate selection rule. Include anchorless blocks and capacity loss from excluding them.
Exit: reproducible matched baseline advantage and observed mechanism explaining why. If anchors do not mediate the advantage, revise or reject the explanation.

### Workstream C — Capacity and distortion
1. Use net useful payloads of 10k, 50k, and 250k bits where capacity allows; use repeated independent payload seeds on identical tensors.
2. Report encoded/framed bits separately from net useful bits.
3. Report eligible carriers, utilization of eligible set, number/fraction of changed codes, codebook-weight distortion/MSE, block-scale changes, and complete artifact file growth.
4. Test saturation and failure behavior on low-anchor-density and anchorless blocks.
5. Avoid comparing results at different effective payload or side-information budgets.
Exit: transparent payload-distortion-recovery curve and a declared operating point.

### Workstream D — Full checkpoint and receiver contract
1. Run nf4_anchor_checkpoint.py for a clean and stego checkpoint with a fixed, pinned environment.
2. Require extraction from delivered stego checkpoint in a fresh process without original checkpoint, expected payload, carrier map or reference weights.
3. Verify standard serializer/model load, NF4 state and inference execution.
4. Test both raw artifact extraction and extraction through reloaded model state; compare recovered digests.
5. Compare clean/stego model utility on a sufficiently varied held-out evaluation set; include paired perplexity/loss and an appropriate task-quality metric where practical.
6. Count every delivered byte: packed weights, scales/statistics, config, framing, encryption/ECC, map, key material, sidecars and any other reference information. State separately incremental cost when the base model was already being distributed.
Exit: no undeclared receiver dependency; exact recovery; valid load/inference; quantified utility and byte budget. If fails, stop scale-up for that method until diagnosed.

### Workstream E — Independent model generalization
1. Start with at least three independently sourced model families if feasible, distinguishing family, parameter size, checkpoint revision, and ancestry.
2. Make paired clean/stego comparisons; same-source derivatives remain in the same split.
3. Do not count tensors/blocks/repeated seeds as independent models.
4. Preserve source model IDs and hashes in results. One held-out family is a diagnostic; multiple independent held-out families are better evidence.
5. Include at least one model where eligible-anchor density differs substantially from the development model.
Exit: candidate advantage is reproduced outside the development checkpoint family, or its limits are explicitly narrowed.

### Workstream F — Steganalysis/detectability
1. Treat complete artifact as the primary inference unit. Block/patch detectors are secondary mechanistic diagnostics.
2. Include clean-versus-clean controls, sham save/reload, benign re-quantized controls, and deliberately detectable positive controls.
3. Compare simple NF4 code/anchor/scale statistics, LR/HGB/RF, an appropriate raw-byte/code baseline, and Model X-Ray if reproduction is feasible.
4. Freeze train/validation/test splits by checkpoint ancestry; no source-derived variants cross splits.
5. Fix the validation FPR/threshold before a single final test; report ROC-AUC, TPR at low FPR, realized FPR and uncertainty.
6. Keep artifact-only detector, method-aware detector, carrier-map-known oracle, and trusted-reference integrity comparison as separate threat models.
Exit: valid and accurately scoped detectability characterization. Near-chance under one detector is not a claim of universal undetectability.

### Workstream G — Related work and novelty
Prioritize direct studies:
- MEASER (arXiv:2510.10486 v2)
- Model X-Ray (JISA 2026; arXiv:2409.19310)
- MaleficNet 2.0 / Do You Trust Your Model? (arXiv:2403.03593)
- Yang et al. (Information Sciences 2023, DOI 10.1016/j.ins.2023.119250)
- Hao et al. (IEEE IoT Journal 2025, DOI 10.1109/JIOT.2024.3520994)
- Watermarking LLMs with Weight Quantization (EMNLP Findings 2023)
- StegoNet (ACSAC 2020)
- QIM and syndrome-trellis coding foundations.

Complete backward/forward citation screening, version matching, code availability and feasibility. Distinguish native reproduction from a new packed-NF4 adaptation. Compare methods only under matched payload, receiver knowledge, format, redundancy, compute and artifact budget. Do not claim firstness based only on not seeing NF4 in one prior work.
Exit: citation matrix with inclusion/exclusion and exact overlap statements.

### Workstream H — Statistics and claim audit
- Pre-register primary endpoint, experimental unit, comparison, test, uncertainty and practical effect threshold before confirmatory results.
- Use paired comparisons on same checkpoints/payload seeds where appropriate.
- Use confidence intervals over independent sources/models rather than treating block count as sample size.
- Run label-permutation and positive-control checks for detectors.
- Record all seeds, exact software/hardware versions, source revision, model hashes, configuration, stdout/stderr and result hashes.
- Reconcile all paper figures/tables with source artifacts; no manual copying of final numbers without an auditable script.
Exit: every reported number traceable; failed gates and missing measurements visible.

### Workstream I — Manuscript and release
Draft after the first confirmatory comparison, but freeze claims only after results settle. Expected sections:
1. Abstract
2. Introduction and problem formulation
3. Threat model and artifact/receiver contract
4. NF4 background and failure mechanism
5. Anchor-preserving method and theoretical/invariant conditions
6. Experimental protocol and baselines
7. Results: mechanism, capacity, robustness, utility, artifact cost, detection
8. Ablations and failure cases
9. Related work
10. Limitations, ethics/misuse, reproducibility
11. Conclusion.

Expected figures: (1) sender/transform/receiver lifecycle, (2) block-code examples and anchor behavior, (3) per-cycle recovery/BER, (4) anchor/scale/carrier drift mechanism, (5) payload vs distortion/capacity, (6) utility paired plot, (7) artifact-level detector ROC/TPR-FPR.

Expected tables: methods and receiver assumptions; checkpoint/model matrix; main matched recovery results; utility/overhead; detector metrics; failure cases and provenance.

Prepare a reproducibility release with protocol version, configs, scripts, small non-sensitive fixtures, result manifests and exact reproduction instructions. Do not publish hidden test payload/key secrets or present test keys as secure.

## 5. Execution order (priority)
1. Freeze/provenance audit.
2. Confirm current pilot numbers directly from raw reports.
3. Mechanism/failure-fingerprint table on paired source tensors.
4. Matched repeated comparison across multiple payload sizes and seeds.
5. Full-checkpoint fresh-process lifecycle and utility.
6. Extend to independent model families.
7. Detection evaluation with sound artifact-level splits and controls.
8. Prior-art reproduction feasibility and any high-value baseline runs.
9. Statistical and claims audit.
10. Write final manuscript and target-journal fit memo.

Each stage has a stop/go gate. Do not launch large-scale detector training or a broad multilingual sweep until full-checkpoint recovery and the anchor mechanism have survived the first gates.

## 6. Outcomes and pivots
- Method-paper GO: reproducible practical recovery advantage over matched strong baselines; causal ablation supports the anchor mechanism; end-to-end artifact contract works; utility and byte overhead acceptable; independent-model evidence exists; claims accurately bounded.
- Narrowed-method GO: recovery advantage only in a restricted class of blocks/models/formats. State the conditional scope and characterize failure boundaries.
- Measurement/defense pivot: candidate fails to outperform, but a distinct artifact-level robustness/detection result is reproducible and consequential.
- Technical report/no submission yet: neither method advantage nor a sufficiently novel measurement/defense finding emerges.
- Do not guarantee acceptance or label work journal-ready merely because experiments/tests pass.

## 7. Immediate next session
First inspect actual pilot output manifest and observations for source hashes, measured row counts, cycle-specific BER, anchor loss, scale statistics and model identity. Build one audit table with a row per method × model × tensor × payload × seed × cycle. Verify reported 108/108 TinyLlama and 432/432 Qwen candidate outcomes against these artifacts, while preserving paired baseline failures and any incomplete lifecycle runs. Don't start a new large experiment until this table exists.
