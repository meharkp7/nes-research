# Seven-Method Implementation Status

Updated: 2026-10-09
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

The first CI run caught a real zero-capacity allocation bug; it was fixed and the test was rerun successfully. The CI workflow now syntax-checks the protocol/CLI and runs these unit tests plus a Unicode multi-message CLI smoke test.

## Milestone 2 — method adapter audit

| Method | Existing source/interface | Status for the unified artifact harness |
|---|---|---|
| Sign | `src/embedding/sign_strategy_v2.py`; `strategy_registry.py` | Existing residual-stream strategy; needs multi-tensor artifact adapter and independent reload/extract test |
| Magnitude-aware | `src/embedding/strategies/magnitude_aware_strategy.py`; registry/extractor dispatch | Existing residual-stream strategy; needs multi-tensor artifact adapter and independent reload/extract test |
| QAE | `src/embedding/strategies/quantization_strategy.py` via the `qae` registry adapter | Existing QAE path; do not substitute the separate blocked `nf4_qae` path |
| LWE-inspired grid/parity | `src/embedding/strategies/lwe_strategy.py` and parity extractor | Existing residual-stream method; label LWE-inspired, not a proven lattice cryptosystem |
| Split sign/parity hybrid | `src/embedding/strategies/split_strategy.py`; `exp20_split_dial.py` | Existing hybrid; parity/sign use disjoint carrier subsets; primary fraction 0.50 |
| QSE | `scripts/real_nf4_candidate_eval.py` | Only a single-tensor real-NF4 pilot exists; current formulation BER 0.4016 in the recorded run; needs a new artifact-only receiver and multi-layer protocol |
| DCE | `scripts/real_nf4_candidate_eval.py` plus optimization modules | Single-tensor packed-NF4 pilot exists; parity recovery is a pilot result, not end-to-end proof; needs multi-layer serialization, model utility and detectability evaluation |

## Not yet implemented / not yet measured

- No shared runner yet embeds the framed corpus into all seven methods.
- No model checkpoint or multi-tensor artifact is yet saved and independently reloaded for the full seven-method roster.
- The tensor-capacity allocator is implemented and tested, but it is not yet wired to live model tensors or method-specific carrier selection.
- No new seven-method × seven-model result matrix exists.
- No new utility, detector, multi-seed or transformation-robustness results have been produced by this milestone.

## Next implementation milestone

Build the method adapter contract and Qwen2.5-3B runner. First implement the five existing residual-stream methods through their matching extractors; separately implement packed-NF4 adapters for QSE and DCE. Both lanes must consume the same framed corpus and emit the same report schema while preserving their true representation differences. Require save/reload and a fresh-process receiver test before declaring a method's artifact round trip successful.
