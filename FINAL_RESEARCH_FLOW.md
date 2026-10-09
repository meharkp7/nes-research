# NES Research — Final Project Record

**Purpose:** one concise record of the research flow, the main measured results, what they mean, and what remains before submission.

**Repository:** `meharkp7/nes-research`  
**Scope:** the established NES experiment line (Exp1–Exp27), the later Contract B artifact-only NF4 pilot, and the DCE optimization investigation.  
**Status date:** 9 October 2026

## 1. Executive summary

The project studies whether payloads can be embedded in quantized LLM weight representations and recovered, while measuring distortion, utility, detectability and robustness.

The established experiment line is substantially characterized. Its strongest result is that the LWE-inspired grid/parity strategy achieved BER 0 across the seven-model strategy matrix and a detector score of 0.50 on all seven models under the project's measured detector. The method is **LWE-inspired**, not a cryptographic lattice construction, and the result does **not** establish universal undetectability. A keyless attacker can read the encoded channel under the public rule, even though the separate AES-GCM message remains confidential without the key in the tested threat model.

Robustness is transformation-specific: eight of twelve Exp23 cells recovered exactly, including a real 1,000-step fine-tune, while NF4/GPTQ/AWQ re-quantization and one task-vector merge failed with BER below chance. The later Contract B1.4 test recovered its 10,000-bit payload from the pristine packed-NF4 artifact, but a fresh NF4 re-quantization caused 30 bit errors (BER 0.003) and checksum failure.

The DCE work is an **experimental extension**, not a validated replacement. Its synthetic benchmarks show a real trade-off between aggregate histogram matching and perturbation. No DCE result currently justifies a model-level stealth, utility or robustness claim.

## 2. Research flow and results

The first table follows the established experiment sequence. Values are from the repository's recorded experiment summaries; failed gates are retained as findings, not relabelled as successes.

| Step | Question / method | Main result | Meaning |
|---|---|---|---|
| Exp1 | Does residual extraction and the QACI path line up? | PASS on 7 models. | Established the residual extraction and round-trip foundation. |
| Exp2 | Are residual fingerprints informative across models? | 3 PASS / 4 FAIL. | The chosen fingerprint threshold does not generalize to every tested model. |
| Exp3 | Can a clean payload be embedded, extracted and decrypted? | BER 0 across 48,256 compared bits per tested model; PASS on 7 models. | The production path has a clean-recovery baseline. |
| Exp4 | How much data can the sign baseline carry? | At least 10 million bits at BER 0 in the recorded capacity sweep. | Capacity is high in the tested clean setting; this alone says nothing about stealth. |
| Exp5 | Does embedding affect language-model utility? | Recorded perplexity change +0.0053% under the original three-way protocol. | Small measured change for that specific model and evaluation; not a universal utility guarantee. |
| Exp6 | Does sign embedding survive Gaussian noise? | All 7 models met the configured BER gates; BER was 0 on 6/7 at σ=0.001, with Mistral-7B at 0.00115. | The baseline passed the declared gates, but robustness is not identical across models. |
| Exp7 | Do statistical detectors distinguish clean and embedded residuals? | Statistical gates passed in the recorded 7-model run. | These tests did not reveal the same signal as the neural detector. |
| Exp7-neural | Can a carrier-centred neural detector identify sign embedding? | 70.5% accuracy against a 55% gate; FAIL. | This is a real detectability limitation. Do not claim the sign method is undetectable. |
| Exp8 | Does the aggregate cross-model suite meet every gate? | FAIL, inherited from the neural detectability result. | The overall gate must preserve the failed sub-result. |
| Exp9 | Does the payload survive clean reads of different 4-bit formats? | BER 0 for NF4, GPTQ and AWQ in the tested clean setup; GPTQ correlation 0.9903, AWQ 0.9941, with 35/36 AWQ layers verified. | Supports clean format-specific extraction only; robustness and detectability were not established for GPTQ/AWQ. |
| Exp10 | Can the LWE-inspired grid be read without the original cover? | BER improved from 0.5036 to 0.0000; `extract_needs_cover: false`. | The receiver can read the channel from the embedded weights alone. |
| Exp11 | Sweep LWE-inspired grid width against robustness and detector gates. | Widths 0.005, 0.010 and 0.020 passed both gates; 0.002 failed robustness (BER 0.586 at σ=0.001), while 0.050 failed detectability (70.625%). | Width 0.010 was later retained as the global default; the measured frontier informed that choice. |
| Exp12 | Does the LWE-inspired strategy generalize across measured models? | 5/5 measured models passed the recorded BER/detector gates; TinyLlama was skipped because its cache was incomplete. | Strong cross-model evidence within the tested scope, not a universal claim. |
| Exp13 | Is the LWE-inspired channel key-gated? | Keyless phase attacker recovered all 10,256 channel bits at BER 0; position precision/recall 1.0; shipped grid width was a public constant. | **Gate FAIL.** The channel is readable without the key under the tested public rule. Do not call the grid itself key-secure. |
| Exp14 | Can a blind detector identify carrier locations it is not given? | Blind accuracy 50.0%; carrier-centred control 68.75%. | The earlier neural result is placement-conditioned; this blind test does not erase Exp7-neural's failure. |
| Exp15 | Does the LWE-inspired strategy preserve utility? | Embedding-specific perplexity change +0.0077% on Qwen2.5-3B and +0.0501% on gemma-2-2b. | Both passed the project's 2% gate under the recorded protocol. |
| Exp16 | Does a detector transfer between sign and LWE-inspired schemes? | Sign-trained detector: 62.85% on sign, 50.00% on LWE. Reverse direction was uninformative because its control collapsed to chance. | Supports a directional no-transfer observation only; not a validated two-way conclusion. |
| Exp17 | Can QAE use the production round-trip interface? | `qae` recovered 48,256 bits at BER 0; `nf4_qae` remained BLOCKED because its reference requires weights not carried by the interface. | QAE round-trip plumbing works in the tested path; the blocked residual variant is not a success. |
| Exp18 | Which strategy works across the strategy × model matrix? | 28/28 cells had BER 0 and passed the recorded robustness gates; LWE-inspired strategy had detector 0.50 on 7/7 models. | Detectability separated the methods in this matrix. TinyLlama was skipped for incomplete cache. |
| Exp19 | Does adaptive routing select and run different strategy branches? | Three models selected three branches; available round trips recovered at BER 0. Qwen's selected neural route failed because the trained model was unavailable. | The routing experiment exposed a design limitation; the unavailable branch was recorded, not hidden. |
| Exp20 | Is there a sign/parity mixture trade-off? | All five settings round-tripped at BER 0; as parity share rose, BER at σ=0.002 rose from 0 to 0.0127 and detector accuracy fell toward 0.50. | The stealth/robustness trade-off is tunable; pure parity was the only setting meeting all recorded gates. |
| Exp21 | Does QAE encoding work with LWE read-out? | Raw interoperation BER 0.5433 (5,572/10,256 errors); public correction recovered BER 0. | **FAIL as proposed.** The apparent hybrid reduces to sign read-out plus a public relabelling rule. |
| Exp22 | Can per-layer grid width improve the global width setting? | Global width remained best; magnitude-keyed per-layer width failed robustness (0.0226 / 0.5763), while rank-keyed width passed but was less robust than global. Detector was 0.50 for all three. | More adaptive width did not improve the overall trade-off; keep the global 0.010 setting as the measured reference. |
| Exp23 | Which model operations preserve an embedded payload? | 8/12 cells recovered at BER 0, including a real 1,000-step fine-tune, LoRA tests, pruning and one merge setting. NF4 BER 0.3832; half-merge 0.2476; GPTQ 0.4956; AWQ 0.4108. | Light model operations sometimes preserve the payload; re-quantization can substantially damage it. State results per transformation. |
| Exp24 | What configuration lies on the measured distortion/detectability Pareto frontier? | The recorded frontier selected Exp22's rank-keyed point: mean absolute change 0.00372693, detector 0.50, BER at σ=0.001 of 0.00146256; it dominated 34/34 compared points in the recorded dataset. | A dataset-specific Pareto result, not proof of a global optimum or security guarantee. |
| Exp25 | Does carrier-position selection affect robustness? | Across 7 replicates, BER 0 and detector accuracy 0.5028 for all arms. At σ=0.001, production magnitude selection had BER 0, while random/keyed baselines had 0.0724–0.0757. | Selection improved measured noise robustness without a measured detector change in this test. The public-rule re-run still recovered positions nearly exactly, so channel readability and message confidentiality must remain separate. |
| Exp26 | How does capacity scale while tracking detectability? | 1k–50k payloads all had BER 0; detector accuracy stayed at 0.5028. Mean absolute change at changed values fell from 0.0662 to 0.0351. | Strong capacity evidence for the tested configuration and band; Exp4 separately covers the larger 500k–10M range. |
| Exp27 | What happens with wrong keys, partial model access and public-rule access? | Wrong keys: 0/10 recoveries and 10/10 GCM authentication failures. Partial access: 0/6 recoveries. Public-rule attacker was near chance on the tested detector/readout metric. | AES-GCM message confidentiality held in these tests, while the channel itself remained readable when the carrier map was available. This is not a claim that every attack is defeated. |

### Overall state recorded for the core suite

| Item | Recorded result | Interpretation |
|---|---|---|
| Experiment manifest | 35 PASS / 6 FAIL / 0 NOT_RUN / 0 ERROR | Includes the core suite's cell-level outcomes. The six FAILs are real measured gate failures: Exp2 on four models, Exp7-neural, and Exp8 inheriting the detector failure. |
| Claim audit | The methodology summary records 152/152 claims passing. | Re-run the audit before submission and align documentation counts if they differ. |
| Consistency checks | 9/9 in the methodology summary. | Re-run after any change to experiment artifacts or reporting. |
| Terminology | The grid method is LWE-inspired; no LWE/SIS lattice instance was constructed. | Do not claim post-quantum or lattice-based cryptographic security from the name. |

## 3. Later Contract B artifact-only NF4 pilot

This is a separate, fixed-contract pilot. It tests recovery from a serialized packed-NF4 artifact without relying on the original FP16 weights, a residual cache or a delta sidecar.

| Step | Result | Meaning |
|---|---|---|
| B1.4 pristine artifact recovery | 10,000-bit payload recovered exactly; BER 0; checksum valid. The envelope adds 16 bytes and the embedding changed 5,011 packed codes in the recorded run. | Demonstrates the artifact-only sender/receiver contract for the pristine tested artifact. It does not prove stealth, confidentiality, or robustness. |
| B1.4 utility diagnostic | On 8 texts / 217 scored tokens, reported perplexity changed from 36.808438 to 36.555752 (−0.686%). | A small diagnostic, not a sufficiently broad utility benchmark. |
| B1.5 NF4 re-quantization | 30 errors in 10,000 bits; BER 0.003; checksum failed. Maximum contiguous error run was 1 in the recorded diagnostic. | Exact recovery did not survive this transformation. This is one measured channel outcome, not a universal error rate. |

The payload and key used by this pilot are test fixtures. The checksum detects accidental corruption; it is not a replacement for cryptographic authentication.

## 4. Candidate B / Distribution-Constrained Embedding (DCE)

DCE was investigated as an extension to improve the aggregate weight distribution while retaining payload recovery. It remains isolated from the production strategy registry.

| DCE stage | Main result | Decision |
|---|---|---|
| Independent per-carrier objective; 2,000 synthetic carriers | On seed 20261009, BER was 0 for both methods by construction. DCE had MSE 0.064852 vs 0.021024 baseline, TV 0.111773 vs 0.036264, and cover-to-embedded KL 0.068248 vs 0.006574. | Negative result for this objective; it worsened perturbation and distribution metrics. |
| Repeated-seed weight sweep | Across the five visible non-zero weights (0.5, 1, 2, 5, 10), mean paired changes were positive for MSE, TV and KL; increasing the distribution weight made the observed losses larger. The uploaded excerpt did not include the summaries for weights 0 and 0.1. | Stop tuning the independent per-carrier proxy. |
| Greedy batch DCE | Across five synthetic seeds, mean MSE was about 0.182154 vs 0.021028 baseline; mean TV 0.006250 vs 0.037735; mean KL 0.001368 vs 0.007226. | Better histogram metrics, but roughly 8.7× the baseline MSE. This is a trade-off, not an overall win. |
| Distortion-budgeted batch DCE | Five-seed sweeps at 0%, 5%, 10%, and 25% extra distortion stayed within the declared budgets. Against nearest-feasible, mean TV change was −0.000695 (5%), −0.001192 (10%), and −0.000794 (25%). Mean KL changes were −0.000109, −0.000158 and −0.000186 respectively, with mixed per-seed wins. | Gains over baseline are small. At 10% budget, TV improves by about 0.00119 while MSE increases by about 10%. |
| Matched comparisons against other DCE variants | Budgeted DCE improves TV/KL relative to independent DCE, but greedy batch DCE has substantially better histogram metrics at much higher distortion. | No method is an established winner across the full trade-off. |
| Synthetic recovery | BER 0 across the tested methods and runs. | Expected from the toy candidate set and quantizer; it is not NF4 robustness evidence. |

**DCE conclusion:** keep the implementation and negative results as an auditable research branch, but do not claim that DCE is stealthier or superior. Do not add it to the production method unless a matched real-model evaluation demonstrates a meaningful improvement in recovery, utility and detectability.

## 5. Candidate status and final research position

- **Candidate A (QSE / quantization-state or residual-channel direction): NOT CLOSED.** Earlier work measured the QAE adapter round trip, but `nf4_qae` was blocked at the production strategy interface. That is an integration gap, not a valid negative result for QSE. A new real-NF4 tensor pilot is now implemented in `nes-llm/scripts/real_nf4_candidate_eval.py`; its model-dependent run has not yet been executed in the user's local model environment.
- **Candidate B (DCE): NOT CLOSED.** The existing optimizer evidence remains synthetic-only and does not count as a real-NF4 test. The same pilot now compares a nearest-feasible NF4-code baseline against a histogram-aware DCE selector on actual BitsAndBytes NF4 codes. Its model-dependent run has not yet been executed. Do not claim either candidate succeeds or fails until the real run and required follow-up validation are recorded.
- **Established Exp1–Exp27 line: frozen as the main measured research record.** The final paper should be based on those artifact-backed results, plus Contract B1.4/B1.5 as a separate packed-artifact case study, with all limitations retained.

A real-NF4 DCE pilot is not an automatic next step: the current DCE candidate objective is not integrated with the packed-code carrier contract, and the synthetic comparison did not establish a clear overall win. Reopen Candidate B only if a specific NF4-aware, receiver-reproducible protocol has a clear hypothesis worth testing. This avoids turning a negative result into another open-ended implementation cycle.

The best-supported research story is not “the method is undetectable under all conditions.” It is:

1. A payload can be embedded and recovered in the tested LLM weight settings.
2. Capacity, utility, detection and robustness depend on the embedding scheme and evaluation conditions.
3. The LWE-inspired grid/parity strategy performs well on the recorded detector and model matrix, but public-rule channel recovery means the channel is not key-gated.
4. Some model operations preserve the payload, while specific 4-bit re-quantization operations substantially damage recovery.
5. DCE's current synthetic results reveal a distribution-versus-distortion trade-off but do not yet justify a model-level claim.

## 6. Final work remaining

| Status | Work | Result / completion condition |
|---|---|---|
| **DONE** | Fix CI standalone-script imports. | Added `PYTHONPATH=.` to the workflow; the standalone DCE scripts now run in CI. |
| **DONE** | Re-run branch CI and inspect the failure. | Workflow run **147 passed**: syntax checks, Contract B synthetic tests, core artifact audits, DCE unit tests and smoke commands. |
| **DONE** | Audit the recorded core results. | `claim_audit.py`: **152/152 claims verified**. `check_consistency.py`: **9/9 checks passed**. The historical absolute-path issue was fixed in the checker without changing results or gates. |
| **DONE** | Add a real-NF4 candidate pilot harness. | `scripts/real_nf4_candidate_eval.py` evaluates Candidate A/QSE after a real NF4 quantize/dequantize cycle and Candidate B/DCE over actual BitsAndBytes NF4 code indices, with a matched nearest-feasible baseline. Syntax is CI-checked. |
| **NOT RUN — LOCAL** | Execute Candidate A and Candidate B on the actual cached FP16 model tensor. | Run the command in §7. The JSON output is the first real NF4 result for these candidates; GitHub Actions does not have the user's model cache, so this must run locally. |
| **NOT RUN — FOLLOW-UP** | Validate full artifact save/reload, utility and detectability for any promising candidate. | Tensor-level BER alone is not an end-to-end result. Preserve every report and compare both candidates under the same payload, tensor, carrier positions, and NF4 settings. |
| **PAPER** | Final manuscript and submission package. | Use this ledger and the committed artifacts to write the manuscript; every central claim must map to an artifact, and failures/limitations must remain explicit. This is paper-writing work, not a reason to add more experiments. |

No new optimizer, embedding architecture, ECC design or broad benchmark sweep is planned. The current evidence is ready to be consolidated into the paper, with model-cache-dependent work clearly separated from completed CI validation.

## 7. Reproducibility commands

Run from `nes-llm/` using the repository environment:

```bash
# Core suite audits
../.venv/bin/python run_nes_experiments.py --audit
../.venv/bin/python check_consistency.py
../.venv/bin/python claim_audit.py
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v

# Real NF4 Candidate A/B pilot (use a local FP16/BF16 model directory; report path must be new)
../.venv/bin/python scripts/real_nf4_candidate_eval.py \\
  --model Qwen/Qwen2.5-3B \\
  --tensor model.layers.0.self_attn.q_proj.weight \\
  --payload-bits 10000 \\
  --output ../cache/real_nf4_candidate_eval_qwen3b_$(date +%Y%m%d_%H%M%S).json \\
  --local-files-only

# DCE prototype tests
../.venv/bin/python -m unittest discover -s tests -p 'test_distribution_constrained_optimizer.py' -v
../.venv/bin/python -m unittest discover -s tests -p 'test_batch_distribution_optimizer.py' -v
../.venv/bin/python -m unittest discover -s tests -p 'test_budgeted_batch_optimizer.py' -v
../.venv/bin/python -m unittest discover -s tests -p 'test_dce_distortion_budget_sweep.py' -v
../.venv/bin/python -m unittest discover -s tests -p 'test_dce_candidate_benchmark.py' -v
../.venv/bin/python -m unittest discover -s tests -p 'test_dce_three_way_comparison.py' -v
../.venv/bin/python -m unittest discover -s tests -p 'test_dce_three_way_multiseed.py' -v
../.venv/bin/python -m unittest discover -s tests -p 'test_dce_weight_sweep.py' -v
```

The synthetic DCE commands do not require the local Qwen cache. The real-NF4 pilot requires a local FP16/BF16 model directory and the installed PyTorch, Transformers, and BitsAndBytes environment. It writes a JSON report and refuses to overwrite an existing report. This first stage is tensor-level; it does not itself establish artifact-only recovery, utility, detectability, or transformation robustness.

## 8. Reporting rules

- Keep every measured FAIL; never change a threshold to make it pass.
- Label synthetic results as synthetic.
- Do not infer stealth from histogram TV/KL alone.
- Do not infer re-quantization robustness from pristine-artifact recovery.
- Distinguish channel readability from cryptographic message confidentiality.
- Treat the saved JSON artifacts and audit tools as the source of truth for final numerical claims.
