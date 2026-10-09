# Contract B1.5 plan — payload error-pattern diagnosis before ECC

## Purpose

B1.4 remains the frozen baseline. The observed real Qwen2.5-3B lifecycle transformation (NF4 -> BF16 floating intermediate -> fresh NF4) completed but did not recover the 10,000-bit application payload exactly: 30 bit errors, BER 0.003, invalid payload checksum. This is one measured transformation, not an estimated universal channel error rate.

B1.5 begins with diagnosis, not an assumed error-correcting code. The first deliverable is read-only: compare the pristine B1.4 stego artifact with a transformed artifact and record exact payload error positions, counts in fixed-size blocks, and contiguous error runs.

## Run synthetic tests

From nes-llm/:

\`\`\`bash
../.venv/bin/python -m unittest discover -s tests -p 'test_contract_b_error_pattern.py' -v
\`\`\`

The tests are synthetic and verify the analysis implementation only. They do not claim that the real checkpoint transformation was repeated.

## Diagnose the existing local NF4 re-quantization result

Use the existing artifacts if they are still present. The tool never modifies either artifact and refuses to overwrite the report.

\`\`\`bash
../.venv/bin/python scripts/contract_b_error_pattern.py \\
  --reference-stego ../cache/contract_b_nf4_b14_10k \\
  --transformed-stego ../cache/contract_b_b14_nf4_requantized_retry3 \\
  --output ../cache/contract_b_b14_nf4_requantized_retry3_error_pattern.json
\`\`\`

The report contains:
- Header validity and payload checksum validity.
- Total application-payload bit errors and BER.
- Zero-based payload error positions.
- Error counts and BER in 128-bit blocks by default (configurable with \`--block-bits\`).
- Contiguous error runs and the longest run.
- Hashes for the payload and packed tensors used in the comparison.

If the local artifact paths differ, substitute the exact paths in the existing requantization report. Do not rerun the expensive model transformation solely because a path name differs. If the transformed artifact is missing, first determine whether it can be recovered from the local run; do not overwrite the pristine baseline.

## B1.5 decision gate

After the error-position report is inspected:

1. Run additional independent lifecycle trials/configurations if practical; preserve every report and failure.
2. Determine whether observed errors are scattered, clustered, or configuration-sensitive. One run is insufficient to infer an independent bit-flip channel.
3. Compare ECC candidates using a reproducible simulation informed by the measured error patterns: a simple baseline, BCH-style bit correction, and an interleaved symbol-code candidate if burst behavior warrants it.
4. Evaluate complete framing, not just application data. The current B1.4 envelope contains magic, length, payload and an 8-byte truncated SHA-256 checksum (16 bytes total framing overhead). A B1.5 format must protect or safely validate framing as well as payload.
5. Track net application capacity after redundancy, code rate, correction success, residual BER, checksum success, carrier changes, and descriptive detectability.
6. Only after the simulation and error-model assumptions are documented should a separately versioned B1.5 encoder/decoder be implemented. Do not alter B1.4 code or relabel its failed lifecycle result.

A candidate must be tested against both synthetic random errors and realistic error patterns. Correcting the 30 observed errors in one sample is not enough to claim robustness.

## Broader research roadmap after B1.5

The goal remains one integrated paper, not separate disconnected papers.

### A. Contract B protocol and lifecycle evidence
- Preserve B1.4 pristine recovery, capacity sweep, utility diagnostic, descriptive detectability diagnostic, and controlled packed-code robustness results as distinct evidence types.
- Finish error-pattern diagnosis and assess ECC/interleaving in B1.5.
- Run multiple independently declared real NF4 lifecycle trials where disk and compute permit; keep failures.
- Add separate B1.4/B1.5 artifact tests for save/reload, and then consider pruning, fine-tuning/LoRA merge, and task-vector merge one axis at a time. Do not confuse old residual-domain experiment 23 results with this packed-NF4 protocol.

### B. Evaluation quality
- Capacity: distinguish application bits from envelope/ECC overhead and repeat across payload sizes and seeds.
- Utility: replace the small 8-text perplexity diagnostic with a larger held-out corpus, paired controls, uncertainty intervals, and task-level checks where feasible.
- Detectability: go beyond histograms/TV distance to predeclared statistical and trained-detector baselines, held-out evaluation, class balance, uncertainty, and clear threat-model assumptions.
- Model breadth: once the single-model pipeline is stable, replicate on multiple model/checkpoint configurations; report skipped or incompatible models rather than silently excluding them.
- Robustness: separate serialization, requantization, fine-tuning, pruning, adapter merge, task-vector merge, and other transformations; each gets its own artifact and report.

### C. Research and paper
- Literature review: distinguish existing weight steganography, quantization-aware embedding, error correction for covert channels, keying, and cryptographic wrappers. Audit novelty claims, especially the LWE-inspired parity/grid method: it is not a demonstrated LWE/SIS cryptographic construction.
- Ablations: carrier-selection strategy, payload size, redundancy/code rate, interleaving, tensor/layer selection, quantization configuration, and key dependence.
- Threat model and terminology: keep confidentiality/integrity (e.g. AES-GCM), channel key-gating, detectability, utility, capacity, and robustness as separate properties.
- Reproducibility: freeze manifests, versions, seeds, commands, source hashes, and failed-run reports; distinguish synthetic tests from real model-level trials.
- Claim audit: retain negative results including exp7 detector accuracy 70.5%, exp13 keyless extraction, exp23 requantization failures, and the current B1.4 lifecycle failure.
- Write one integrated paper with methods, baselines, negative results, confidence/uncertainty, limitations, and a claim-to-evidence table.

## Execution boundary

The GitHub repository can hold scripts, tests, protocol docs, and tracked reports. It does not provide access to the local Mac model cache or execute the full Qwen lifecycle on that machine. Real artifact analysis and model-level trials must be run locally; this diagnostic can be tested synthetically in CI.
