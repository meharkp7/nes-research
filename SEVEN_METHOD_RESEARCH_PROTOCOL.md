# NES Seven-Method Research Protocol
**Status:** Proposed unified protocol; audit baseline recorded 2026-10-09
**Repository:** `meharkp7/nes-research`
**Working branch:** `research/contract-b-b14-and-evaluation`
**Purpose:** Freeze the research flow before implementing the shared multi-message, multi-layer evaluation harness.

## 1. Repository audit: what already exists

This plan is based on the current working branch and the repository's recorded artifacts. It distinguishes source-code inspection from measured results.

### Existing infrastructure
- `nes-llm/src/embedding/strategy_registry.py`: strategy metadata and receiver requirements.
- `nes-llm/src/embedding/strategies/`: LWE-inspired grid/parity, magnitude-aware, QAE, split sign/parity and other existing strategy implementations.
- `nes-llm/src/experiments/exp10_strategy_comparison.py`: shared measurement functions for extractability, robustness and detector performance.
- `nes-llm/src/experiments/exp18_strategy_model_matrix.py`: existing four-strategy cross-model comparison.
- `nes-llm/src/experiments/exp20_split_dial.py` and `src/embedding/strategies/split_strategy.py`: hybrid split method with parity and sign on disjoint carriers.
- `nes-llm/src/experiments/exp21_qae_lwe_readout.py`: QAE encoder + LWE parity readout interoperability test.
- `nes-llm/scripts/real_nf4_candidate_eval.py`: new tensor-level BitsAndBytes NF4 pilot for QSE and DCE. It is not a multi-layer string or checkpoint protocol.
- `nes-llm/src/experiments/manifest.py`, `artifact_manager.py`, `claim_audit.py`, `check_consistency.py`: status, artifact, and claim-audit infrastructure to preserve and reuse.
- `results/experiment_matrix.json`, `results/exp17_qae_round_trip.json`, `results/exp18_matrix_*.json`, `results/exp20_split_dial_*.json`, `results/exp21_qae_lwe_*.json`: historical measurement records.

### Audit findings and boundaries
1. Historical tests are not yet a unified seven-method, multi-string, saved-artifact experiment. Existing results remain valuable historical evidence but must not be relabelled as directly comparable to new results.
2. The existing exp18 matrix evaluated `sign`, `magnitude_aware`, `lwe`, and `qae`; its results cannot be treated as measurements of QSE, DCE or the split hybrid.
3. Exp20's `SplitStrategy` is the parity/sign hybrid: a configured fraction of carriers uses LWE-inspired parity/grid embedding and the rest use sign embedding on disjoint carrier sets. Existing fractions are 0, 0.25, 0.5, 0.75, and 1.0. The endpoints are pure sign and pure parity, not additional independent methods.
4. Exp21's QAE-encode/LWE-readout pairing is a separate failed interoperability hypothesis (raw BER 0.5433). The public correction reduces to sign reading plus a public relabelling identity; it must not be counted as the seventh independent method.
5. The implementation calls the grid method LWE-inspired, but the code does not construct a formal LWE matrix instance or establish lattice-based cryptographic security. In papers and tables use **LWE-inspired grid/parity**, not a claim of a proven lattice/LWE cryptosystem.
6. The current QAE adapter `qae` is distinct from the blocked `nf4_qae` residual-reference path. Do not silently substitute one for the other.
7. QSE and DCE currently have tensor-level real-NF4 pilot evidence only. QSE's current formulation produced BER 0.4016 in the recorded single-tensor run. DCE recovered 10,000/10,000 bits in its parity construction, with carrier-code histogram TV 0.0049 versus 0.0899 for the nearest-feasible baseline. Those are preliminary, method-specific tensor observations, not end-to-end results.
8. Contract B1.4/B1.5 is a separate artifact-only NF4 pilot (10,000-bit pristine recovery, 30 errors after re-quantization; BER 0.003). Keep its evidence separate from the QSE/DCE pilot and from the established residual-stream experiments.
9. Historical records, FAILs, blocked cells and missing-cache reasons are immutable. New runs get new IDs and output files. A passing syntax/CI workflow is not evidence that a model experiment ran.

## 2. The seven methods in scope

The primary comparison contains exactly these seven method families:

| ID | Method | Existing implementation / provenance | Main question |
|---|---|---|---|
| M1 | **Sign** | Existing production sign embedding and sign extractor | Baseline capacity, perturbation, utility, detectability and robustness |
| M2 | **Magnitude-aware** | `MagnitudeAwareStrategy` / corresponding extractor | Whether adaptive magnitude margins improve the trade-off over sign |
| M3 | **QAE** | Existing `qae` adapter over `QuantizationStrategy`; not `nf4_qae` | What the existing quantization-aware path actually contributes under the unified protocol |
| M4 | **LWE-inspired grid/parity** | `LWEStrategy` with its matching parity extractor | Recovery, robustness, detectability, distribution changes and key/threat assumptions |
| M5 | **Sign/parity hybrid (SplitStrategy)** | `SplitStrategy`; disjoint sign and parity carrier subsets | Whether intermediate parity fractions offer a useful trade-off |
| M6 | **QSE** | Current quantization-state/residual candidate | Can the residual-based candidate meet the receiver contract and recover from an artifact? |
| M7 | **DCE** | Distribution-Constrained Embedding candidate, using real NF4 codes where applicable | Can distribution-aware code selection improve the baseline without unacceptable distortion or utility loss? |

### Hybrid settings
- Primary M5 setting: parity fraction = 0.50, fixed before the main comparison.
- Secondary sensitivity sweep: 0.25 and 0.75.
- Anchor checks: 0.0 must reproduce the pure-sign endpoint; 1.0 must reproduce the pure LWE-inspired parity endpoint. These are validation controls, not extra methods.
- The QAE + LWE readout pairing remains a recorded failed interoperability test, not an eighth method and not a substitute for SplitStrategy.

If a method cannot operate on a particular representation without changing its defining algorithm, mark that cell `NOT_APPLICABLE` or `BLOCKED` with a reason. Do not silently drop it or force an unfair adapter.

## 3. Model evaluation order

### Stage A — one-model pilot
Use **Qwen/Qwen2.5-3B** first. It is the established reference model for the existing exp10/exp18/exp20 experiments and has relevant cached residuals and NF4 measurements.

Run all seven methods through the same agreed input protocol and as similar an artifact/receiver contract as each method legitimately supports. The first pilot must establish correctness and protocol compatibility before spending compute on detectors or broad robustness sweeps.

### Stage B — seven-model comparison
Only after Stage A's correctness gate passes, evaluate all seven methods across the following seven base-model IDs, matching the existing exp18 matrix roster:

1. `google/gemma-2-2b`
2. `Qwen/Qwen2.5-3B`
3. `meta-llama/Llama-3.1-8B`
4. `google/gemma-2-9b`
5. `microsoft/Phi-3-mini-4k-instruct`
6. `mistralai/Mistral-7B-v0.3`
7. `Qwen/Qwen2.5-7B`

These are the seven model IDs in the existing exp18 result files. `TinyLlama/TinyLlama-1.1B-Chat-v1.0` was previously skipped in that matrix because its cache was incomplete; it is not silently substituted into this fixed seven-model roster. The AWQ/GPTQ Qwen variants are quantization-format test cases, not additional base models in this roster.

Run one model per process and record model revision, source tensor/checkpoint hashes, hardware, library versions, quantization settings, seed and output artifact paths. If a model is unavailable or a cache is incomplete, report that cell as blocked/not-run with the exact reason; do not call the matrix complete.

## 4. Multi-string input contract

The harness must support **one or many strings in a single run**, without editing code for each new test.

### Accepted input forms
- Repeatable CLI argument: `--message "first string" --message "second string"`.
- JSONL corpus: `--messages-file path/to/messages.jsonl`, one JSON object per line, at minimum `{"id":"msg-001","text":"Hello NES"}`.
- Optional generated test corpus: fixed-seed generator with declared character/byte length, Unicode policy and corpus ID. Generated strings supplement, never replace, real user-provided test strings.

### Framing and payload
Each record is encoded as UTF-8 bytes and framed with protocol version, record ID, byte length and integrity digest. A corpus envelope records the ordered record IDs, offsets/lengths and aggregate digest. The receiver reconstructs each record independently and verifies both per-record and corpus-level integrity. Do not rely on null delimiters or assume strings contain only ASCII.

Test at least:
- one short ASCII string;
- multiple distinct strings in one run;
- Unicode / non-ASCII text;
- empty string if supported by the framing contract;
- repeated strings and duplicate IDs (duplicate-ID policy must be explicit);
- variable-length messages and a larger corpus;
- payload sizes near the measured capacity limit.

The harness reports per-message exact match, byte errors, bit errors/BER, framed overhead and net useful payload capacity. Do not aggregate away a failed message because other messages pass.

### Multi-layer allocation
Convert the ordered framed corpus into a bitstream. Allocate it deterministically across a predeclared set of eligible tensors spanning early, middle and late transformer layers. Record the exact tensor names, shapes, carrier counts, payload offsets and method-specific parameters in a manifest. The key determines carrier locations where applicable; the model/tensor identity is included in domain separation to prevent collisions across tensors. Payload order must be reconstructed without relying on the original FP16 cover.

The receiver must operate from the declared receiver inputs and the saved artifact. If a method requires original FP16 weights or a clean residual to decode, that is a receiver-contract limitation, not an artifact-only success.

## 5. Representation lanes and fairness

There are two implementation lineages, and they must not be conflated:

- **Residual-stream lane:** existing sign, magnitude-aware, QAE, LWE-inspired grid/parity and split hybrid implementations were historically built around FP16-to-quantized residuals and their existing carrier/receiver interfaces.
- **Packed-NF4 lane:** QSE/DCE pilots manipulate a real BitsAndBytes NF4 tensor/code representation.

The shared experimental framework should normalize payloads, message corpora, layer allocations where feasible, reports and evaluation metrics. It must not pretend the internal carrier substrate is identical when it is not. First run each method in its faithful native representation on the same source model and message corpus; then run matched cross-method comparisons only in representation cells where the methods can be implemented without changing their defining mechanism. Every representation and adapter is named in the result.

For model utility, compare against the corresponding clean representation/model, with identical inference prompts, decoding settings and evaluation data. Do not compare an embedded quantized model against an unrelated unquantized baseline and attribute all differences to embedding.

## 6. Experiment sequence and gates

### Gate 0 — audit and freeze
- Inventory implementations, extractors, strategy registry status, experiment artifacts, known failures, dependencies and current model availability.
- Freeze protocol version, threat model, methods, model roster, message corpus format, carrier/layer policy, baselines, metrics and success thresholds before looking at new outcomes.
- Preserve historical artifacts and current PRs. Do not merge or rewrite historical claims as part of this planning step.

### Gate 1 — adapter and correctness tests
For all seven methods, verify encoder/decoder pairing, deterministic carrier allocation, tensor ordering, capacity bounds, key handling and message framing. Record `PASS`, `FAIL`, `BLOCKED`, `NOT_APPLICABLE` and `NOT_RUN` distinctly.

### Gate 2 — one-model, one-message baseline
On Qwen2.5-3B, test one short known string on one tensor. This is only a debug baseline. Save the representation, start a fresh receiver process, reload it, and recover the string without access to the original FP16 cover unless that cover is explicitly part of that method's receiver contract.

### Gate 3 — one-model, multi-layer and multi-string
On Qwen2.5-3B, run the frozen message corpus over early/middle/late layers. Start a fresh receiver process and verify each record and corpus digest. Sweep predeclared payload sizes and carrier densities; keep all failed reports.

### Gate 4 — one-model quality evaluation
For all seven methods, with matched controls where valid, measure:
- exact recovery, BER and net capacity;
- changed-value/carrier ratio, RMSE/MAE and quantization-code changes;
- per-tensor and aggregate distribution statistics (including histogram TV/KL where appropriate);
- perplexity and predeclared task metrics;
- statistical and learned detector performance on held-out examples, with controls and uncertainty estimates;
- runtime, memory and reproducibility across seeds.

For M5, run the fixed 0.50 primary setting and the planned 0.25/0.75 sensitivity checks, with endpoint anchors at 0/1.

### Gate 5 — seven-model expansion
Only after the one-model correctness and protocol gates pass, repeat the fixed protocol over all seven listed base models. Keep model-major execution to control memory. Any model/method incompatibility is recorded explicitly, with no silent substitutions.

### Gate 6 — robustness and ablations
Measure separately after:
1. clean save/reload;
2. fresh NF4 re-quantization;
3. other supported quantization formats (GPTQ/AWQ only through their correct native loaders/dequantizers);
4. pruning;
5. fine-tuning;
6. LoRA merge and selected task-vector/model merges;
7. partial tensor/layer loss or corruption;
8. wrong key and missing metadata where relevant.

Report recovery, utility and artifact validity per transformation. Do not generalize a pass on one operation into a blanket robustness claim.

### Gate 7 — analysis and paper
Publish a seven-method × seven-model result matrix, per-message recovery table, capacity/distortion/utility plots, detector results, robustness matrix, ablations, failed/blocked cells and reproducibility instructions. Prefer Pareto comparisons over an arbitrary weighted score; define any composite score before measuring.

## 7. Metrics and decision rules

Every run records:
- message count, IDs, UTF-8 byte lengths, framed bits, net useful bits;
- payload bits attempted/transmitted/extracted, bit errors, byte errors, per-message exact match and BER;
- model ID/revision, tensor/layer allocation, representation/quantizer, method and parameters;
- weight/code distortion, histogram metrics, utility, detector results and transformation history;
- seed, software/hardware versions, elapsed time, peak memory, artifact hashes and status.

Freeze numeric thresholds before the new comparison. Existing suite gates (BER 0, BER <= 0.02 at sigma 0.001, BER <= 0.10 at sigma 0.002, detector accuracy <= 55%, PPL degradation <= 2%) may be used as historical reference gates, but must be reviewed for applicability to the new artifact protocol before being adopted. Do not change thresholds after seeing outcomes. Detector accuracy near chance is not proof of universal undetectability; BER 0 is not proof of security; low histogram distance is not proof of stealth.

## 8. Repository workflow

- Keep `main` and existing historical result files unchanged during protocol development.
- Keep PR #2 draft/open until the research specification is reviewed and the code is brought into alignment. Do not merge without explicit authorization.
- New experiment runs use new, uniquely named, append-only artifacts; no overwriting historical reports.
- Reuse `manifest.py`, `artifact_manager.py`, `claim_audit.py`, and `check_consistency.py` wherever possible.
- CI validates code and deterministic tests; actual model results require a separately recorded local/model-run artifact.
- Every experiment has: a hypothesis, baseline/control, fixed parameters, command, expected artifact schema, pass/fail gate and interpretation boundary.

## 9. Immediate next actions (in order)

1. Review this protocol against the audited files and correct any disputed method naming before implementation.
2. Create a machine-readable seven-method registry and a model/message corpus config.
3. Design a common multi-layer allocator and framed multi-string payload interface with method-specific encode/decode adapters.
4. Add unit tests for framing, UTF-8, multiple messages, ordering, duplicate IDs, capacity overflow, deterministic allocation and integrity failure.
5. Implement the Qwen2.5-3B end-to-end artifact round-trip for all seven methods, preserving unsupported/failed methods as explicit results.
6. Run the one-model matrix, review it, then expand to the seven-model matrix.
7. Only after that, execute full utility, detection and robustness studies.
