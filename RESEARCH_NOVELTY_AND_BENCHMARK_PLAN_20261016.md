# NES: research direction and benchmark specification

Assessment date: 2026-10-16. Status: **research proposal, not new experimental evidence**.

The recommended next phase is an artifact-level audit of hidden payloads in quantized LLMs. A superior embedding method is a possible outcome, not the premise. Existing NES results establish useful feasibility and failure cases, but do not establish superiority over published methods or readiness for a high-impact journal.

This document adds a new research direction alongside the completed closeout. The historical evidence, failures and provenance limitations remain as recorded in [Phase 3](/Users/meharkapoor7/nes-research/PHASE3_CONSOLIDATED_SEVEN_METHOD.md). No embedding run, detector training, dataset regeneration or baseline reproduction was performed for this assessment.

## 1. Why study this carrier?

A defensible motivation is that useful model files are already distributed between organizations and developers. The research question is whether those files can also carry recoverable information that ordinary model evaluation overlooks, and what integrity checks or transformations reveal or remove it. That matters to model supply-chain auditing independently of whether NES wins a capacity leaderboard.

This motivation has explicit boundaries:

| Intended purpose | Sensible starting comparison | Implication for NES |
| --- | --- | --- |
| Attach provenance or ordinary metadata | Signed manifest or sidecar | Hidden weights need a demonstrated benefit over simpler, auditable storage. |
| Verify ownership | Model watermarking | Evaluate ownership verification and attack resistance; arbitrary-message BER is a different objective. |
| Communicate arbitrary data | Conventional encrypted communication or established image/text steganography | A multi-gigabyte carrier is not automatically efficient. Define the permitted communication channel and count all transmitted bytes. |
| Audit an already-distributed model | Weight steganography, weight steganalysis, integrity verification and model sanitization | Best present fit for the NES evidence and a substantive security question. |

Report both total distribution cost and incremental cost when the model would already be transferred. An apparent advantage that depends on excluding the model's size must be stated as conditional. Do not compare image bits/pixel, generated-token capacity and model bits/parameter as interchangeable units.

An authenticated hash of the exact expected checkpoint detects any byte modification, assuming the trusted reference is available. It does not identify the meaning of a modification. A statistical detector is relevant when legitimate model variations are allowed or no exact reference is available; those conditions need to be specified, not assumed away.

Research starts from an important, falsifiable question; it does not require proof of superiority before the first experiment. Publication subsequently requires a contribution supported by evidence. New mechanisms, explanations, measurements and defenses can qualify; a larger experiment count alone does not.

## 2. Prior art that changes the novelty claim

The following is a targeted primary-source review, not an exhaustive systematic review. Dates refer to the identified version or publication. None of these methods was reproduced during this assessment. Numerical claims from unlike experiments must not be pasted into a shared NES leaderboard.

| Study | Relevance and evaluation distinction | Access and next comparison |
| --- | --- | --- |
| [MEASER: Malware embedding attacks on open-source LLMs](https://arxiv.org/html/2510.10486v2), preprint, April 2026 revision | Direct LLM-weight overlap: QIM-based embedding, error correction, spread spectrum and quantization/PEFT experiments. Its `SR = 1 - d_PAI` measures performance preservation; scanner detection rates are separate. Neither is clean/stego weight ROC-AUC. | Full HTML inspected. The linked [repository](https://github.com/uuutmtm/MEASER) returned 404 during this review; reproducibility remains unresolved. Compare inert-payload carrier behavior if reproducible. |
| [Model X-Ray](https://doi.org/10.1016/j.jisa.2026.104517), Journal of Information Security and Applications, 2026; [2024 preprint](https://arxiv.org/html/2409.19310v1) | Direct steganalysis baseline. Published materials include few-shot detection, float32 weight representations and CNN collections; architecture generalization is already studied. Its embedding-rate definition must be aligned before comparison with NES. | Preprint and [author repository](https://github.com/danigil/ModelXRay) inspected. The current reproduction guide includes statistical and raw-byte baselines. A packed-NF4 adaptation must be labeled separately from native reproduction. |
| [Do You Trust Your Model? Emerging Malware Threats in the Deep Learning Ecosystem](https://arxiv.org/html/2403.03593v3), MaleficNet 2.0, accepted IEEE TDSC 2025 per arXiv | Direct parameter-payload comparator using spread spectrum and error correction. Section VII-B3 distinguishes embedding into an already reduced-precision model from quantizing after embedding, which can mitigate the payload. | Full HTML inspected. Reproduce only the inert carrier experiment relevant to the comparison; verify exact code/version and receiver inputs before scheduling. |
| [A general steganographic framework for neural network models](https://doi.org/10.1016/j.ins.2023.119250), Yang, Wang and Zhang, Information Sciences, 2023 | Direct neural-network steganography; training-time CNN parameter embedding with a different extraction mechanism and compute budget. | Indexed primary abstract/metadata inspected; publisher full text unavailable through the research tool. Capacity figures in the pasted note were not independently verified and are excluded here. |
| [General Steganography for Neural Network Models Based on Graph Convolutional Network](https://doi.org/10.1109/JIOT.2024.3520994), Hao et al., IEEE IoT Journal, 2025 | Broader layer support and language-generation applications challenge a generic “first beyond CNNs” claim. Training requirements and receiver assumptions need full-text extraction. | Indexed primary abstract/metadata and [application repository](https://github.com/timedeadline/ApplicationInstance) inspected; complete training reproduction not established. |
| [Watermarking LLMs with Weight Quantization](https://aclanthology.org/2023.findings-emnlp.220/), Li et al., EMNLP Findings, 2023 | Adjacent prior art: watermark behavior differs between full precision and INT8. Relevant to quantization-based mechanisms, but ownership protection is not arbitrary-message transmission. | Primary abstract and [repository](https://github.com/Twilight92z/Quantize-Watermark) inspected. Use for mechanism/context; do not equate watermark success with payload BER. |
| [StegoNet](https://doi.org/10.1145/3427228.3427268), ACSAC 2020 | Earlier parameter-based payload hiding establishes that the broad carrier idea predates NES. | Primary conference material identified. Include in the historical background and baseline screening. |
| [Look Who's Talking Now: Covert Channels From Biased LLMs](https://aclanthology.org/2024.findings-emnlp.971/), Silva, Sala and Gabrys, EMNLP Findings, 2024 | Output-token communication, rather than data stored in model weights. | Primary abstract inspected. A contextual comparison, not a directly interchangeable payload baseline. |

The immediate novelty exclusions are: hiding bits in neural weights; doing so in LLMs; exploiting quantization differences; adding QIM, ECC or spread spectrum; and evaluating a detector or a model collection. These are established directions. Classical [QIM](https://doi.org/10.1109/18.923725) and [syndrome-trellis coding](https://doi.org/10.1109/TIFS.2011.2134094) also predate this project. Combining known ingredients needs its own substantive justification.

Before a firstness claim, complete backward/forward citation screening around the direct papers and search the terms model steganography, parameter embedding, model steganalysis, quantization watermarking, packed quantization and payload robustness. Record inclusion/exclusion reasons and update the search before submission. Absence of NF4 in one inspected paper is not proof that an NF4 contribution is novel.

## 3. What the present NES evidence actually establishes

| Evidence | Supported interpretation | Remaining gap |
| --- | --- | --- |
| Seven-method corpus: 49 successful saved extraction reports, 36,128 framed bits each, seven multilingual messages | Repeated artifact-level round trips under recorded receiver contracts | Tensor bundles are not full deployable checkpoints; earlier QSE failures remain part of the record. |
| Packed evaluation: 6,111,232 rows, seven model IDs, 4/1/2 train/validation/test groups | LR/HGB block-feature AUC approximately 0.5004–0.5011 in the evaluated split | Two test IDs; Qwen appears in train and test; correlated blocks; historical revision matching unverified. |
| TinyLlama residual LOLO: LWE HGB AUC 0.8273 | A detector identifies structure in that within-model diagnostic | Near-chance results under another detector do not imply general concealment. |
| Largest sign test: 10 million requested bits, BER zero | Largest tested Qwen payload point in that protocol | No matched detector result at that size; not a measured packed-NF4 capacity result. |
| Sign surgery: NF4/GPTQ/AWQ requantization failures | Transformation order and format can destroy recovery | No universal quantization-survival claim. |
| LWE keyless recovery failure; separate AES-GCM experiments | Message encryption and carrier/key secrecy are distinct properties | No cryptographic guarantee follows from “LWE-inspired.” |

Sources and exact scopes: [packed audit](/Users/meharkapoor7/nes-research/PHASE1_PACKED_NF4_EVIDENCE_AUDIT.md), [residual audit](/Users/meharkapoor7/nes-research/PHASE2_RESIDUAL_DETECTOR_READINESS.md), [consolidated evidence](/Users/meharkapoor7/nes-research/PHASE3_CONSOLIDATED_SEVEN_METHOD.md).

### Receiver and payload findings from the current source

The current [corpus encoder](/Users/meharkapoor7/nes-research/nes-llm/src/experiments/seven_method_protocol.py:114) writes UTF-8 lengths, IDs, text and SHA-256 integrity digests. Both corpus pilots call this encoder and convert its output directly to bits. **This path does not encrypt the messages.** A hash is not encryption or keyed authentication. Separate production AES-GCM results must not be attributed to these pilots.

The [packed pilot](/Users/meharkapoor7/nes-research/nes-llm/scripts/seven_method_nf4_pilot.py:269) stores an int64 carrier position per framed payload bit. QSE additionally stores two float32 values per carrier: a clean quantized value and a reference residual. Its metadata explicitly labels the result an experiment artifact rather than an HF checkpoint. The [receiver](/Users/meharkapoor7/nes-research/nes-llm/scripts/seven_method_nf4_pilot.py:350) uses the stored positions; DCE does not require an extraction key in this path. Keyed selection alone does not make the stored map secret.

A useful accounting calculation follows directly from these dtypes: for 36,128 framed bits (4,516 bytes), the position tensor contains 289,024 bytes, or 64 times the framed payload byte count. QSE's two reference tensors together add another 289,024 bytes. These are raw tensor-storage calculations, not measured complete file sizes, minimum necessary overhead, or information-theoretic bounds. They show why current recovery success cannot yet establish an efficient deployment contract.

The residual-carrier premise also needs a precise boundary. If all payload choices produce exactly the same delivered quantized artifact, a receiver seeing only that artifact and a fixed independent key cannot distinguish those payloads. Information discarded by quantization is unavailable to that receiver. If recovery depends on altered codes, scales or a residual/reference attachment, identify that actual carrier and account for it. This observation is a basic consistency constraint, not a novelty claim.

## 4. Recommended scientific question and possible contribution

**Question:** Under realistic model distribution, how do serialization, receiver side information, transformation order and detector evaluation unit change the measured payload–utility–detectability trade-off of quantized LLMs?

Use DCE as the first packed-artifact feasibility candidate because its current decoding rule does not require original weight values. That is an engineering prioritization, not a claim that code parity is new or superior. Keep QSE in an explicitly reference-assisted comparison. Use the residual methods as historical/mechanistic controls; avoid expanding all seven equally before the deployment gate succeeds.

| Hypothesis to test | Evidence that could support it | Evidence against it |
| --- | --- | --- |
| H1: a practical channel remains in a complete quantized checkpoint with a declared receiver contract | Fresh-process exact recovery and normal inference with all required bytes accounted for | Recovery needs undeclared original data, payload-dependent metadata, or a tensor-only representation incompatible with the deployed model. |
| H2: block-level evaluation materially misstates whole-artifact detectability in this setting | Reproducible, held-out differences across evaluation units, with positive controls and a mechanistic explanation | Calibrated whole-artifact evaluation agrees with block conclusions across independent hosts. |
| H3: transformation order explains important robustness differences | Matched before/after experiments with utility controls, replicated across formats and families | Differences disappear after matching net payload, receiver information and representation. |
| H4: an NES candidate improves a practically relevant trade-off against strong baselines | A predeclared, practically meaningful paired improvement with uncertainty across held-out hosts | Baseline matches it, improvement depends on extra information/compute, or it occurs only in development models. |

H2 and H3 are candidate research gaps, not established discoveries or claims that prior work ignored all such issues. A strong paper needs a finding beyond the protocol itself. An algorithm paper additionally needs a substantive mechanism and ablations explaining the gain. An audit/defense paper needs a consequential new measurement or mitigation result compared with existing steganalysis.

## 5. Benchmark specification

### A. Deployment and receiver gate — first priority

For each candidate, save a complete checkpoint in its claimed supported standard format, reload it in a fresh process, run inference, and extract the payload. The receiver can access only the declared delivered artifact, fixed decoder, agreed key and explicitly budgeted side information. Expected payload files are evaluation-only and inaccessible to decoding. Pin serializer, quantizer, dependencies, model revision and code revision.

Inventory every delivered file and tensor. Measure ordinary checkpoint bytes, additional bytes, framing/encryption/ECC bytes, carrier-map bytes and reference bytes separately. Check whether a distinctive manifest already identifies the experiment to a file-level observer. Distinguish a weights-only detector from a detector allowed to inspect the whole distributed package. A controlled weights-only experiment remains useful, but it cannot support a claim about the whole package.

A deployment failure is a recorded result and stops scale-up of that arm. It does not become a success by changing the receiver contract without reporting a new variant.

### B. Baselines and comparison rules

Use a native-setting reproduction table and a separate matched deployment table. Start with simple LSB/sign or code-parity controls, a standard QIM control where representable, the strongest reproducible published parameter-payload comparators, and NES candidates. MEASER and MaleficNet are priority literature comparators; training-time Yang/GCN methods belong in a separate compute-budget stratum unless a fair common setup is available.

Use inert text or random bytes as payloads. Payload execution is not part of the scientific endpoint. A paper-faithful reimplementation is not an author-code reproduction. An unsupported-format port is a new adaptation and must be named as such. Report code/version availability and reproduction failures; do not score an unavailable baseline as defeated.

Match exact source checkpoint, target format, **net useful payload**, receiver information, redundancy, calibration data and compute budget. Report time and peak memory for embedding, extraction and detection. Keep native float comparisons separate from quantized endpoints when methods cannot support the same receiver representation.

### C. Controls, splits and statistical unit

The primary detector decision is one complete artifact. Blocks, patches and layers are correlated observations used to construct that decision. Train any pooling rule using training/validation data only. Record architecture/family/base-checkpoint ancestry so base/instruct variants, revisions and shared ancestors are not quietly spread across an intended family holdout. Treat uncertain ancestry conservatively.

All clean/stego pairs and derived transformations of a source stay in the same split. Payloads, keys and experimental runs are grouped as required by the declared generalization claim. Repeated seeds from one checkpoint are conditional replicates, not new independent models. Reusing a clean control across methods is efficient but does not increase the independent sample count.

Use untouched controls, sham save/reload controls, and justified benign quantization/fine-tuning variations. Specify which benign variations belong to the deployment population. Apply balanced processing to clean and stego files so format, filenames and metadata cannot accidentally encode labels. Include label-permutation checks, deliberately detectable positive controls and clean-vs-clean calibration.

Report touched-region detection as an oracle diagnostic and whole-artifact detection as the primary deployment outcome. This checks whether sparse modification dilutes a block result. Do not give a carrier map to a blind detector and retain the blind label.

### D. Detector suite and threat assumptions

Retain LR/HGB, add the existing RF comparator, and include a published Model X-Ray baseline plus a learned raw-byte/code baseline appropriate to the format. Include simple method-aware code/weight statistics. Use a common development budget and preserve all predeclared detector outcomes. Selection of the primary detector/ensemble and its threshold happens before confirmatory testing, not by picking the least successful test detector.

Separate artifact-only detection, algorithm-aware detection, carrier-map-known diagnostics, trusted-original comparison and active sanitization. No measured detector suite proves universal indistinguishability. Safe deserialization and statistical payload detection address different questions.

### E. Outcomes and uncertainty

| Outcome | Required measurement |
| --- | --- |
| Capacity | Net useful bits; framed/coded bits; bits per parameter; bits per complete delivered byte; fraction of weights/codes changed; side-information overhead. |
| Recovery | Raw BER; post-ECC BER if applicable; exact message/hash recovery; authenticated recovery for encrypted arms; extraction failures. |
| Detectability | Artifact-level ROC-AUC and score distributions; TPR at validation-calibrated FPR; realized test FPR; confusion counts and uncertainty. Block metrics remain auxiliary. |
| Utility | Paired loss/perplexity and appropriate downstream task changes versus the same clean quantized control; full precision is a separate reference. |
| Robustness | Recovery and utility after each declared transformation, including failures and output validity. |
| Cost | Time, peak RAM/VRAM, disk and transferred bytes, calibration/fine-tuning requirements. |

The principal comparative endpoint should be **maximum tested net payload meeting predeclared recovery, utility and detector limits**, or a Pareto curve if no single operating point is justified. Report matched-payload comparisons as well. A lower measured AUC against one detector is not a general “security score.”

Use paired uncertainty across source checkpoints/families for method comparisons, and paired task-item uncertainty for utility. Account for the hierarchy; do not bootstrap millions of blocks as independent models. Very few families require descriptive reporting and explicit limits rather than artificially precise intervals. Preregister a primary comparison and handle secondary comparisons transparently.

Low false-positive claims require many independent negatives. With zero false positives among `n` independent draws from a specified negative population at a fixed threshold, the one-sided 95% binomial upper bound is `1 - 0.05^(1/n)`. It takes 299 such negatives to bound FPR below 1%, and 2,995 below 0.1%. Blocks from one model do not supply those draws. If this scale is infeasible, report uncertainty at the available scale instead of promising repository-wide low-FPR performance.

Failure to reject chance performance is not evidence of equivalence to chance. A concealment claim requires a predeclared equivalence margin, adequate power and a bounded detector/threat scope. Proposed utility tolerances, such as at most 1% relative perplexity increase or one percentage-point task loss, are starting points for development calibration—not universal standards or existing NES achievements. Freeze justified margins before confirmatory runs.

### F. Transformation order and utility

Keep serialization-only, embed-then-quantize, quantize-then-embed, and subsequent requantization as separate experiments. Record format, bit width, group/block size, scale handling and calibration data. NF4 feasibility comes first; add a second practically relevant representation such as GPTQ or AWQ only when both full-checkpoint pipelines are verified.

An unmerged adapter leaving the base weights unchanged is not evidence that base-weight changes are survived. Distinguish adapter use, adapter merging and full weight updates. For pruning/fine-tuning/merging, measure the transformed clean control too: a transformation that erases the payload by destroying the model is not a useful mitigation.

## 6. Multilingual dataset: useful, with the right experimental role

Do build a structured corpus when the runner contract is settled. Current raw UTF-8 pilots make language/byte distribution a relevant variable, but byte length, bit balance and redundancy must be controlled. Start with English, Hindi, Arabic, Chinese, Spanish, French, Bengali and Tamil, plus combining marks, emoji, mixed scripts and right-to-left text. Preserve exact source bytes; test normalization explicitly rather than silently normalizing.

Keep three separate suites:

1. **Payload correctness:** exact UTF-8 round trips, empty/boundary-length records, framing overhead, malformed-input rejection and binary payloads through an explicitly versioned binary-capable protocol.
2. **Payload distribution:** equal-byte-length natural text, compressed data, random data and properly encrypted data as distinct arms. For encrypted arms, record algorithm, nonce policy, padding/length and framing. At matched ciphertext lengths, language diversity generally does not provide distinct bit-level security conditions; visible headers or lengths can still differ.
3. **Multilingual model utility:** evaluate what the host model can do after modification, using supported-language tasks and held-out data. Candidate benchmarks include XNLI for inference, MGSM for mathematical reasoning and FLORES where translation is in scope. Verify task versions, licenses and model suitability before selection. Avoid pooling token perplexities across different tokenizers as though they were directly comparable.

More messages improve correctness coverage. They do not create independent cover models, fix split leakage or establish stronger detectability evidence.

Use linked CSV/JSONL manifests with immutable artifact hashes rather than one enormous flattened CSV:

| Table | Unit and minimum fields |
| --- | --- |
| `models` | One source checkpoint: ID, revision, hash, family/ancestry group, format, tokenizer, supported-language basis, split. |
| `payloads` | One byte string: ID, path/hash, script/language, encoding, byte count, distribution arm, seed, normalization/encryption metadata. No secret keys in public manifests. |
| `runs` | One method execution: run ID, model/payload IDs, method/version, seed/key ID, configuration hash, net/gross bits, receiver contract, source revision, status. |
| `artifacts` | One delivered artifact: hash/path, source/run ID, role, parent/transformation ID, size and side-information accounting. |
| `recovery` | One extraction: artifact ID, decoder version, raw/post-code BER, exact/authenticated success, duration and errors. |
| `detector_scores` | One detector prediction per artifact: detector/version, split, score, fixed threshold, predicted label and evaluation run ID. |
| `utility` | One model/task/language evaluation: artifact/control IDs, dataset revision, item count, metric, paired difference and uncertainty. |
| `summaries` | One evaluation comparison: included groups, metric, interval, aggregation method and evidence references. ROC-AUC belongs here, not copied into every message row. |

## 7. Execution order and stop/go decisions

| Stage | Concrete output | Advance only if |
| --- | --- | --- |
| 0. Current assessment | This prior-art triage, evidence gap analysis and protocol | Completed; this is the current deliverable. |
| 1. Close the direct-comparator review | Full receiver/format/metric/code matrix for the closest papers, with reproduction feasibility and unresolved access recorded | The proposed claim has a meaningful remaining gap. If a comparator already resolves it, revise the question before scaling. |
| 2. Prove deployment feasibility | Full-checkpoint save/load, inference, exact extraction, declared receiver and measured storage accounting | At least one useful receiver contract works without undeclared data. Otherwise publishable work must address that failure, not scale tensor-only successes. |
| 3. Bounded development pilot | Two development checkpoints from different families; three payload sizes; three repeats per feasible method/format; clean/sham controls; runtime and variance estimates | A meaningful channel, evaluation effect or mitigation result merits further study and has an affordable confirmatory design. This is 18 stego outputs per method/format before transformations, not 18 independent models. |
| 4. Freeze confirmatory protocol | Held-out family/ancestry manifest, power/precision calculation, baseline settings, primary endpoint, utility margins, detector budget and compute estimate | Independent model coverage supports the actual claim. The two development families and all inspected historical runs remain development evidence. |
| 5. Confirmatory evaluation | Artifact-level comparisons, multilingual utility, transformation tests, confidence intervals, ablations and retained failures | Results survive the prespecified analysis and meaningful controls. If underpowered, narrow the claim; do not substitute more block rows. |
| 6. Manuscript and artifact review | Claim-to-evidence table, independently replayable results, limitations, responsible inert-payload release and updated literature search | The paper contains a substantive supported contribution beyond applying an existing technique to a new model label. |

For stage 3, use 10,000, 50,000 and 250,000 **net** payload bits as provisional engineering probes, subject to explicit capacity failure reporting. These are proposed future points, not existing packed-NF4 achievements. Use existing Qwen/Phi models as development candidates only; final host selection follows resource and license checks. Estimate GPU-hours and storage from this pilot before allocating a larger matrix. A format change, encryption change or receiver-contract change creates a new versioned run.

## 8. Publication decision

**Route A — method paper:** proceed only if a substantively new mechanism produces a practically relevant advantage over strong reproducible baselines under matched budgets, with cross-family evidence and explanatory ablations. A new name for code parity or an added ECC layer is insufficient.

**Route B — measurement/defense paper:** currently the more defensible direction. Establish a consequential, generalizable finding about deployed quantized artifacts, detectability evaluation or a useful mitigation, and compare it with published steganalysis. A careful negative result can contribute, but an inventory of NES failures alone is insufficient.

**Route C — technical report:** if deployment fails, apparent advantages vanish, or the remaining observations duplicate existing work, preserve the experiments as a transparent report and choose a different research question. Journal ambition is not a reason to relabel that outcome.

[IEEE TIFS](https://signalprocessingsociety.org/publications-resources/ieee-transactions-information-forensics-and-security) is a plausible scope match for a strong information-hiding/forensics contribution. IEEE TDSC is a plausible direction for a substantial model supply-chain security study; final scope and author requirements should be checked against the finished contribution. Venue fit is not an acceptance prediction, and impact factors are not evidence of research quality.

Recommended working title: **“Hidden Payloads in Quantized LLMs: An Artifact-Level Audit of Recovery, Detectability, and Utility.”** Keep this as a question-led study until the confirmatory evidence determines the claims.
