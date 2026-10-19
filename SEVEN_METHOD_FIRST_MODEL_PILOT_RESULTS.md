# Seven-method study — first Qwen2.5-3B pilot results

Date: 2026-10-10 (user-run local experiments)

## Summary

Two local artifact-level round trips completed successfully with exact corpus recovery. These results validate the initial serialization/receiver paths for one residual-stream method and DCE's packed-code parity receiver. They do not establish full model checkpoint preservation, utility, detectability, robustness, or the seven-method comparison.

| Lane | Method | Corpus | Payload | Bit errors | BER | Exact match | Scope |
|---|---|---:|---:|---:|---:|---|---|
| Residual stream | LWE-inspired grid/parity (`lwe_grid_parity`, registry `lwe`) | 2 messages, including Unicode | 1,088 bits | 0 | 0.0 | Yes | Residual-tensor bundle, selected layers 0, 8, 15, 23, 35 |
| Packed NF4 | DCE | 3 messages, including Unicode | 1,544 bits | 0 | 0.0 | Yes | Packed-NF4 experiment artifact; only `model.layers.0.self_attn.q_proj.weight` actually received payload bits in this run |

Both extraction reports showed a matching corpus SHA-256 digest and correct message count/UTF-8 byte lengths. The expected corpus was written separately from each artifact, and extraction was run as a separate CLI invocation.

## Important correction: NF4 payload allocation

The DCE invocation listed five projection tensors, but the first version of the runner allocated bits sequentially until the payload was exhausted. Because the first tensor had ample capacity, the saved artifact listed only layer 0. Therefore this result is a **single-tensor DCE artifact round trip**, not a five-tensor result.

A follow-up code change adds fair payload allocation across the explicitly selected tensors, plus unit tests for even distribution, small-capacity tensors, and capacity overflow. Re-run DCE after that change passes CI; the new artifact must list all five requested tensor names before it can be reported as a multi-tensor result.

## Interpretation and limitations

- Exact artifact-level recovery is demonstrated for this test payload and these two receiver paths.
- The residual artifact is not a saved/reloaded Hugging Face model checkpoint.
- The packed-NF4 artifact is an experiment bundle, not a quantized Hugging Face checkpoint.
- DCE uses code-index parity; this result does not establish stealth, utility preservation, robustness, or cryptographic security.
- QSE has not yet been run through the new multi-tensor adapter. Its current receiver uses reference side information and must be reported separately from DCE.
- Sign, magnitude-aware, QAE, split sign/parity, and the remaining model roster are not yet validated by this run.
- The next gate is: (1) CI green for distributed allocation, (2) re-run DCE and confirm five tensors in output, (3) run QSE, (4) run the remaining three residual methods on Qwen2.5-3B, (5) add utility/detectability evaluation before the seven-model expansion.

## Reproducibility artifacts

Local artifacts (not committed; paths are machine-specific):
- Residual bundle: `cache/seven_method_lwe_20261010_000342.pt`
- Residual expected corpus: `cache/seven_method_lwe_20261010_000342.expected.bin`
- DCE bundle: `cache/seven_method_dce_20261010_000350.pt`
- DCE expected corpus: `cache/seven_method_dce_20261010_000350.expected.bin`
- DCE report: `cache/seven_method_dce_20261010_000350.json`

No plaintext message contents are included in this report.
