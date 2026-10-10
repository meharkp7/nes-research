# QSE NF4 Statistics Comparison — K Projection

**Date:** 2026-10-10  
**Status:** Completed matched clean-control comparison  
**Scope:** One tensor, one model revision, one multilingual payload

## Setup

- Model: `Qwen/Qwen2.5-3B`
- Pinned source revision: `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`
- Tensor: `model.layers.16.self_attn.k_proj.weight`
- Tensor shape: `256 × 2048` (524,288 values)
- Quantizer: bitsandbytes NF4
- Block size: 64
- Payload: 36,128 bits across seven multilingual messages
- Corpus SHA-256: `62c30141c4f73551d19ff4ab0e3d37b87eb581b2cb5f08179a4401ce4d622235`
- Receiver: reference-assisted QSE

## Recovery

Both configurations recovered all seven messages exactly.

| Metric | Nested statistics ON | Nested statistics OFF |
|---|---:|---:|
| `compress_statistics` | `true` | `false` |
| Payload bits | 36,128 | 36,128 |
| Bit errors | 0 | 0 |
| BER | 0.0 | 0.0 |
| Exact corpus match | Yes | Yes |
| Manifest digest match | Yes | Yes |
| Used delta | 0.02324502356350422 | 0.023231709375977516 |
| Weight RMSE vs FP16 | 0.006757681258022785 | 0.006753277964890003 |
| Weight RMSE vs clean NF4 | 0.0063860174268484116 | 0.006380127277225256 |

## Matched clean-control packed-code comparison

Each embedded artifact was compared against a freshly generated clean NF4 control for the same tensor and matching statistics setting.

| Metric | Nested statistics ON | Nested statistics OFF |
|---|---:|---:|
| Total values/codes | 524,288 | 524,288 |
| Carrier positions | 36,128 | 36,128 |
| Changed codes overall | 70,798 | 70,763 |
| Overall changed fraction | 13.5036469% | 13.4969711% |
| Changed carrier codes | 35,266 | 35,266 |
| Carrier change fraction | 97.6140390% | 97.6140390% |
| Changed non-carrier codes | 35,532 | 35,497 |
| Non-carrier change fraction | 7.2787611% | 7.2715913% |

The nested-ON comparison has 35 more changed codes overall, all in the non-carrier category. Carrier-change counts and carrier-change fractions are identical in this comparison.

## Interpretation

- Exact recovery was observed in both configurations for this tensor and payload.
- The matched clean-control packed-code change rates are very close.
- These measurements are descriptive for this single tensor and experiment; they do not establish statistical or neural detectability, robustness under transformation, or superiority of either configuration.
- Clean controls were freshly quantized from the pinned source revision. Their identity with the historical clean source used by the embedding procedure is **unverified**.
- The carrier/non-carrier split uses the carrier positions recorded in each embedded artifact.

## Artifact locations

### Nested statistics ON

- `../cache/qse_nf4_kproj_stats_nested_20261010/qse_embedded.pt`
- `../cache/qse_nf4_kproj_stats_nested_20261010/qse_metadata.json`
- `../cache/qse_nf4_kproj_stats_nested_20261010/extract.log`
- `../cache/qse_nf4_kproj_stats_nested_20261010/clean_control.pt`
- `../cache/qse_nf4_kproj_stats_nested_20261010/clean_control.json`

### Nested statistics OFF

- `../cache/qse_nf4_kproj_stats_plain_20261010/qse_embedded.pt`
- `../cache/qse_nf4_kproj_stats_plain_20261010/qse_metadata.json`
- `../cache/qse_nf4_kproj_stats_plain_20261010/extract.log`
- `../cache/qse_nf4_kproj_stats_plain_20261010/clean_control.pt`
- `../cache/qse_nf4_kproj_stats_plain_20261010/clean_control.json`

### Comparison output

- `../cache/qse_nf4_kproj_stats_comparison_20261010.json`

## Next experiments

1. Compare the two K-projection embedded packed-code arrays directly.
2. Compare dequantized embedded weights and matched clean controls.
3. Repeat on another eligible tensor before making broader claims.
4. Evaluate detectability separately; packed-code change rates alone are not a detectability test.
