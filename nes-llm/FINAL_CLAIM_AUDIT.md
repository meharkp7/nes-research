# NES Research — final claim audit

**Status date:** 10 October 2026  
**Branch:** `research/contract-b-b14-and-evaluation`  
**Purpose:** constrain paper language to the evidence currently recorded. This is an audit checklist, not a substitute for inspecting every raw report and draft before submission.

## Claim-by-claim disposition

| Candidate claim | Evidence currently recorded | Disposition / safe wording |
|---|---|---|
| A 10,000-bit payload can be recovered from the pristine B1.4 packed-NF4 artifact. | Pristine artifact receiver recovered 10,000 bits with BER 0 using the fixed test key and protocol. | **Supported for this artifact and protocol.** Say exact recovery was demonstrated on the tested artifact. Do not imply generality across models/checkpoints. |
| The payload survives NF4 requantization. | Fresh NF4 requantization completed; 30/10,000 bit errors (BER 0.003), checksum failure, independent receiver rejected payload. | **Contradicted for the tested transformation.** State the specific negative result; do not generalize to every quantizer/configuration. |
| The payload survives pruning. | Two completed runs combined selected-tensor 10% magnitude pruning with fresh NF4 requantization; both invalidated envelope header/checksum and BER was unavailable. | **Not isolated.** Say the combined pruning-plus-requantization pipeline failed twice. Do not attribute failure to pruning alone and do not report BER as zero or 1.0. |
| A clean/embedded detector is near chance across model families. | Cross-model leave-one-family-out AUCs are near chance (Qwen 0.5010–0.5014; TinyLlama 0.5141–0.5289; Gemma 0.5118–0.5223). Only a small number of source/run groups; independence and matched-pair construction not established. | **Exploratory, group-limited evidence only.** Say the evaluated transfer splits were near chance. Do not claim universal undetectability or transferable stealth. |
| A pooled detector works well. | Mixed-model RF AUC 0.6641 is driven by Qwen (0.7689); TinyLlama/Gemma are near chance. Qwen-only grouped OOF AUC 0.8913 varies with held-out projection/variant group. | **Exploratory and heterogeneous.** Report per-family metrics and group limitations; do not use pooled AUC as the headline model-generalization result. |
| The detector benchmark uses independent replications. | Manifest adapter can default `source_id` to `revision-unrecorded`; `run_id` is caller-supplied and not independently validated; only eight bookkeeping groups. | **Not established.** Must check original local generation logs, checkpoint revisions/hashes, run invocations, and clean/embedded pair construction. Never relabel the frozen dataset to imply independence. |
| Embedding preserves model utility. | 32 project-curated prompts, local Qwen2.5-3B, max 256 tokens: original PPL 51.539661; stego PPL 51.532198; token-weighted delta −0.014479%; equal-prompt-weighted delta +0.046468%; paired prompt bootstrap 95% CI [−0.345350%, +0.438795%], below project-defined +2% upper-CI threshold. | **Limited prompt-set-specific support only.** State the protocol and threshold. Not an external benchmark, formal equivalence test, broad task-utility guarantee, or evidence for other models/tasks. |
| The checksum provides cryptographic authentication or confidentiality. | B1.4 uses a fixed test key and a truncated SHA-256 envelope checksum; documentation explicitly treats it as accidental-corruption detection, not cryptographic authentication. | **Unsupported; do not claim.** Do not imply confidentiality, authenticated encryption, or cryptographic security from the checksum or pilot. |
| The method is novel or secure because it uses lattice-based carrier selection. | Current experiment summaries alone do not establish priority over prior literature or formal security. | **Requires literature review and formal argument.** Do not assert novelty, LWE security, or cryptographic guarantees solely from empirical results. |
| The method is robust. | Pristine exact recovery; specific requantization failure; combined pruning+requantization failure; older Exp23 surgery results use a different residual-domain protocol. | **Must be scoped by protocol and transformation.** Keep Exp23 residual-domain findings separate from B1.4 packed-NF4 evidence. Avoid universal robustness claims. |
| DCE improves the overall method. | Synthetic multi-seed batch matching improves histogram TV/KL at roughly 8.66× baseline distortion. | **Not an overall win.** Report the trade-off and do not describe it as an improvement without qualification. |

### Dataset group-count observation — local inspection, 10 October 2026

The user inspected the frozen CSV `cache/packed_nf4_grouped_dataset_20261010.csv` without modifying it: 606,208 rows; 303,104 `clean` and 303,104 `embedded`; tensor-key counts are Q projection 540,672 rows, K projection 32,768, and V projection 32,768. The six Qwen source/run combinations are Q/K/V crossed with `nested`/`plain`; TinyLlama and Gemma each have one Q-projection group. These are dataset bookkeeping groups, not demonstrated independent replications. The summary still does not establish matched clean/embedded pairing, source checkpoint hashes/revisions, or unique run invocations. Original generation artifacts/logs remain to be checked locally.

## Required final gates

- [ ] Inspect original local dataset-generation logs/manifests and hashes to determine whether detector source/run groups are genuinely independent and clean/embedded examples are matched.
- [ ] Run the agreed focused test suite once after the code/results freeze; record exact command and exit status. Do not relax tests or overwrite reports to obtain a pass.
- [ ] Reconcile this table against every claim in the actual paper draft, abstract, conclusion, figures, captions, and README.
- [ ] Verify raw JSON reports, source checkpoint IDs/revisions, package versions, seeds, and command lines for every headline number.
- [ ] Freeze an evidence index with relative paths and SHA-256 hashes for the reports/datasets used in the paper. Do not assume local cache artifacts are tracked in Git.
- [ ] Keep the pull request unmerged until the claim audit and paper consistency checks are complete.

## Language to avoid

Do not write: “undetectable,” “universally robust,” “cryptographically secure,” “LWE-secure” (without a formal reduction), “pruning-robust” based on the combined attack, “utility preserved” without the evaluation scope, or “independent replications” until provenance is established.

## Current overall assessment

The evidence supports a scoped empirical study of payload embedding/recovery in the tested packed-NF4 artifact, a measured utility diagnostic on a fixed project-curated prompt set, and negative results for specific lifecycle transformations. Detector transfer results are near chance but too group-limited to establish general stealth. The paper's contribution and novelty claims still require comparison against prior work and alignment with the complete literature review.
