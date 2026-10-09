# Seven-Method Implementation Status

Updated: 2026-10-10
Working branch: `research/contract-b-b14-and-evaluation`
Protocol: [SEVEN_METHOD_RESEARCH_PROTOCOL.md](SEVEN_METHOD_RESEARCH_PROTOCOL.md)

This is an implementation ledger, not a result claim. **The complete seven-method model-level experiment is not implemented yet.**

## Milestone 1 — shared multi-message protocol

**Status: IMPLEMENTED; CI-tested.**

- `nes-llm/src/experiments/seven_method_protocol.py`
  - Framed corpus format with protocol version, message IDs, UTF-8 payload bytes and per-record SHA-256 integrity digest.
  - Strict decode checks for truncation, invalid magic/version, corrupted content, invalid UTF-8, duplicate IDs and trailing bytes.
  - Supports empty messages, Unicode, duplicate text under distinct IDs, and ordered multiple messages.
  - Converts bytes to MSB-first bits and back.
  - Deterministically allocates a bitstream over an ordered mapping of tensor capacities and refuses overflow.
  - Generates unique, deterministic carrier positions domain-separated by tensor name.
  - Produces a metadata-only corpus summary (message IDs, byte lengths and hashes; no plaintext).
  - Declares the seven-method roster and separates residual-stream methods from packed-NF4 candidates.

- `nes-llm/scripts/prepare_seven_method_corpus.py`
  - Repeat `--message "..." ` to pass multiple strings in one run.
  - Or pass `--messages-file corpus.jsonl` with one JSON object per line.
  - Can combine file and CLI messages, rejecting duplicate IDs.
  - Writes a framed binary corpus and JSON sidecar manifest; refuses to overwrite existing outputs.

- `nes-llm/tests/test_seven_method_protocol.py`
  - Covers multi-string and Unicode round trips, empty strings, duplicate IDs, digest corruption, JSONL defaults, deterministic allocations, capacity overflow, carrier position stability and metadata privacy.

The first CI run caught a real zero-capacity allocation bug; it was fixed and the test was rerun successfully. CI passed for commit `9b53f898` and for the subsequent runner/workflow changes; it syntax-checks the protocol and runner, runs these unit tests, and executes a Unicode multi-message CLI smoke test.

## Milestone 2 — residual-stream artifact runner

**Status: IMPLEMENTED; syntax/CI-tested; real-model run NOT_RUN.**

- `nes-llm/scripts/seven_method_residual_pilot.py` implements `embed` and `extract` commands for the five existing residual-stream methods: sign, magnitude-aware, QAE, LWE-inspired grid/parity, and SplitStrategy.
- It uses the established model residual cache, accepts a list of selected transformer layers, builds one framed multi-message bitstream, uses the existing QACI carrier selector and the method's registered encoder/decoder, then writes a residual-tensor bundle.
- The extract command reloads the artifact and can compare against a separate expected-corpus file to report bit errors, BER and exact match. It can run as a separate process from the embed command.
- **This is not a Hugging Face model checkpoint** and does not yet establish model utility, detectability or transformation robustness. It validates the shared corpus and method-specific residual receiver contract only.
- The packed-NF4 QSE and DCE paths are intentionally not routed through this residual runner.

## Milestone 3 — method adapter audit

| Method | Existing source/interface | Status for the unified artifact harness |
|---|---|---|
| Sign | `src/embedding/sign_strategy_v2.py`; `strategy_registry.py` | Existing residual-stream strategy; needs multi-tensor artifact adapter and independent reload/extract test |
| Magnitude-aware | `src/embedding/strategies/magnitude_aware_strategy.py`; registry/extractor dispatch | Existing residual-stream strategy; needs multi-tensor artifact adapter and independent reload/extract test |
| QAE | `src/embedding/strategies/quantization_strategy.py` via the `qae` registry adapter | Existing QAE path; do not substitute the separate blocked `nf4_qae` path |
| LWE-inspired grid/parity | `src/embedding/strategies/lwe_strategy.py` and parity extractor | Existing residual-stream method; label LWE-inspired, not a proven lattice cryptosystem |
| Split sign/parity hybrid | `src/embedding/strategies/split_strategy.py`; `exp20_split_dial.py` | Existing hybrid; parity/sign use disjoint carrier subsets; primary fraction 0.50 |
| QSE | `scripts/real_nf4_candidate_eval.py` | Only a single-tensor real-NF4 pilot exists; current formulation BER 0.4016 in the recorded run; needs a new artifact-only receiver and multi-layer protocol |
| DCE | `scripts/real_nf4_candidate_eval.py` plus optimization modules | Single-tensor packed-NF4 pilot exists; parity recovery is a pilot result, not end-to-end proof; needs multi-layer serialization, model utility and detectability evaluation |

## Milestone 3 — packed-NF4 candidate artifact adapter

**Status: IMPLEMENTED; syntax/CI tests PASS; real-model run NOT_RUN.**

- `nes-llm/scripts/seven_method_nf4_pilot.py` adds a separate multi-tensor packed-NF4 `embed`/`extract` path for QSE and DCE. It consumes the same framed multi-message corpus and allocates its bitstream across explicitly named source-model tensors.
- DCE writes modified packed NF4 code indices and recovers payload bits from code-index parity. Its decoder does not require the original FP16 cover.
- QSE retains the current reference-assisted receiver contract: it records the clean NF4 carrier values and reference residual values needed by the receiver. This is **not** described as blind or artifact-only recovery.
- The artifact is a research bundle of packed codes and receiver metadata, **not** a quantized Hugging Face checkpoint. The adapter records per-tensor reconstruction RMSE and refuses a clean-codebook layout mismatch.
- `src/experiments/nf4_artifact_codec.py` centralizes nibble packing/unpacking and codebook reconstruction; `tests/test_nf4_artifact_codec.py` covers these operations without loading a model.
- The expected framed corpus is stored separately. Extraction reloads the artifact, validates framing/integrity, and can report bit errors, BER and exact match in a fresh process.
- The CI workflow syntax-checks this runner and runs the pure-Python NF4 codec tests. The workflow passed for runner commit `673d90f419f7bc28d568205fc9350d3eb24c208c`; these tests do not substitute for a real model/cache run.

## Not yet implemented / not yet measured

- The shared protocol exists, with separate residual-stream and packed-NF4 runners; a single orchestrator/report aggregator for all seven methods is still missing.
- The packed-NF4 adapter has not yet been run against a real local model or independently reloaded on the user's machine; no seven-method combined artifact exists.
- The tensor-capacity allocator is implemented and tested, but it is not yet wired to live model tensors or method-specific carrier selection.
- No new seven-method × seven-model result matrix exists.
- No new utility, detector, multi-seed or transformation-robustness results have been produced by this milestone.

## Next implementation milestone

Run `scripts/run_seven_method_long_corpus_matrix.py --dry-run`, then execute the seven-cell multilingual matrix with `--local-files-only`. Fix runtime/receiver issues without changing historical experiment paths. For each method, require fresh-process artifact reload and per-message exact recovery checks before calling the round trip successful. Keep QSE reference-assisted results clearly separate from DCE artifact-only parity recovery. After reviewing all seven recovery outcomes, add/run the unified statistical and learned detectability harness on this same model; expand to other models only after the one-model gate passes.


## First local validation commands

Run from `nes-llm/` after pulling the branch. Use new timestamped output paths because the runners refuse to overwrite artifacts.

Residual-stream smoke test (one method, two strings):

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
../.venv/bin/python scripts/seven_method_residual_pilot.py embed \
  --model Qwen/Qwen2.5-3B --method lwe_grid_parity \
  --message "NES smoke test" --message "नमस्ते 🌍" \
  --output "../cache/seven_method_lwe_${RUN_ID}.pt" \
  --corpus-out "../cache/seven_method_lwe_${RUN_ID}.expected.bin"

../.venv/bin/python scripts/seven_method_residual_pilot.py extract \
  --artifact "../cache/seven_method_lwe_${RUN_ID}.pt" \
  --expected-corpus "../cache/seven_method_lwe_${RUN_ID}.expected.bin"
```

Packed-NF4 DCE smoke test across five Qwen2.5-3B projection tensors:

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
../.venv/bin/python scripts/seven_method_nf4_pilot.py embed \
  --model Qwen/Qwen2.5-3B --local-files-only --method dce \
  --tensors "model.layers.0.self_attn.q_proj.weight,model.layers.8.self_attn.q_proj.weight,model.layers.15.self_attn.q_proj.weight,model.layers.23.self_attn.q_proj.weight,model.layers.35.self_attn.q_proj.weight" \
  --message "first string" --message "second string" --message "नमस्ते 🌍" \
  --output "../cache/seven_method_dce_${RUN_ID}.pt" \
  --corpus-out "../cache/seven_method_dce_${RUN_ID}.expected.bin"

../.venv/bin/python scripts/seven_method_nf4_pilot.py extract \
  --artifact "../cache/seven_method_dce_${RUN_ID}.pt" \
  --expected-corpus "../cache/seven_method_dce_${RUN_ID}.expected.bin"
```

Run the same NF4 command with `--method qse` to characterize the reference-assisted QSE receiver separately. Do not combine its BER with DCE as if their receiver contracts were identical.

## Milestone 4 — one-model multilingual seven-method stress test

**Status: IMPLEMENTED; local real-model run pending.**

- Existing corpora `corpus_a.jsonl`, `corpus_b.jsonl`, and `corpus_c.jsonl` are retained unchanged for historical/repeated-corpus work.
- Added `nes-llm/data/seven_method_corpora/corpus_multilingual_7.jsonl`: seven paragraph-length strings in English, Hindi, Spanish, French, Chinese, Arabic and Japanese. The runner validates 5–7 records and at least 100 Unicode characters per record before model work.
- The default runner is now deliberately restricted to one cached model (`Qwen/Qwen2.5-3B`) and one multilingual corpus, giving **7 methods × 1 model × 1 corpus = 7 cells**. This avoids repeating the prior Gemma config/model failure across seven methods and focuses on the requested experiment.
- The driver determines Qwen2.5-3B layer count from config and spreads residual carriers over up to five early/middle/late layers. It runs embed and fresh-process extract, retains logs and per-cell reports, and incrementally writes JSON/CSV summaries. Use `--local-files-only` to avoid downloading anything.
- Added `--tensors auto` to the NF4 pilot to select an attention projection across up to five evenly spaced layers, preferring `q_proj` and falling back to fused-QKV naming where available.
- Per-cell outcomes are kept distinct: `PASS`, `BER_FAIL`, `EMBED_FAILED`, `EXTRACT_FAILED`, and `BLOCKED`. No failed cell is silently discarded.
- The matrix runner does **not** perform steganalysis yet. This stage expands and stress-tests payload length/corpus diversity across the full model-method roster; statistical and learned detectability comes next.
- The real seven-cell run has not been executed by the repository-editing environment; it must run against the user's local Qwen2.5-3B cache. Prior embedding failures from a multi-model run are not evidence that all methods fail on Qwen2.5-3B.
