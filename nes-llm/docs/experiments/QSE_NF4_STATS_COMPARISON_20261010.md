# QSE NF4 Statistics Configuration Comparison

**Date:** 2026-10-10  
**Status:** Initial matched comparison completed; further replication required  
**Scope:** One model, one tensor, one payload, two NF4 statistics configurations

## 1. Research question

Does enabling nested quantization of NF4 block statistics
(`compress_statistics=True`) change QSE payload recovery, packed codes,
or reconstructed-weight error compared with disabling it?

## 2. Matched experimental setup

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-3B` |
| Source revision | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` |
| Source dtype | FP16 |
| Tensor | `model.layers.16.self_attn.q_proj.weight` |
| Tensor values | 4,194,304 |
| Quantizer | bitsandbytes NF4 |
| Block size | 64 |
| Method | QSE, reference-assisted |
| Payload | 7 multilingual messages |
| Payload bits | 36,128 |
| Original framed-corpus SHA-256 | `62c30141c4f73551d19ff4ab0e3d37b87eb581b2cb5f08179a4401ce4d622235` |
| QSE initial margin | 0.25 |
| Maximum QSE attempts | 8 |

The two runs use the same recovered JSONL, pinned source revision, tensor,
payload, method, block size, key and QSE settings. The intended variable is
`compress_statistics`.

## 3. End-to-end recovery

| Metric | Nested statistics ON | Nested statistics OFF |
|---|---:|---:|
| `compress_statistics` | `true` | `false` |
| Messages recovered | 7 | 7 |
| Expected bits | 36,128 | 36,128 |
| Recovered bits | 36,128 | 36,128 |
| Bit errors | 0 | 0 |
| BER | 0.0 | 0.0 |
| Exact corpus match | Yes | Yes |
| Manifest digest match | Yes | Yes |
| Decode error | None | None |

Both receivers recovered the original corpus SHA-256 exactly.

## 4. Embedding and weight distortion

| Metric | Nested ON | Nested OFF |
|---|---:|---:|
| Packed-code layout reconstruction RMSE | 0.0 | 0.0 |
| QSE delta selected | 0.021012431010603905 | 0.02100096270442009 |
| QSE attempts | 6 | 6 |
| Carrier bit errors after quantization | 0 | 0 |
| Weight RMSE vs FP16 source | 0.0032707545906305313 | 0.00326901744119823 |
| Weight RMSE vs clean NF4 | 0.0020506789442151785 | 0.0020462991669774055 |

These distortion metrics are close. This single pair does not establish a
meaningful quality advantage for either configuration.

## 5. Matched clean controls

Clean NF4 controls were generated from the same pinned FP16 source tensor,
with block size 64 and no payload embedding.

| Metric | Nested ON clean | Nested OFF clean |
|---|---:|---:|
| Number of code values | 4,194,304 | 4,194,304 |
| Code-array SHA-256 | `e032ba7a13944ef03c93f8abb946e84ab7dea589d2ad69a9b163f55a01aa96bc` | `e032ba7a13944ef03c93f8abb946e84ab7dea589d2ad69a9b163f55a01aa96bc` |
| RMSE vs FP16 source | 0.0026266288477927446 | 0.0026251955423504114 |
| Nested state present | Yes | No |

The clean packed NF4 codes are identical in this run. However, their
dequantized tensors are not identical:

| Cross-configuration clean comparison | Result |
|---|---:|
| Packed tensors equal | Yes |
| Dequantized RMSE | 0.000089446017220126 |
| Maximum absolute dequantized difference | 0.0009980052709579468 |
| Dequantized tensors exactly equal | No |

Therefore, identical packed code indices do not imply identical reconstructed
weights when the scale representation differs.

## 6. Embedded versus matching clean codes

| Metric | Nested ON | Nested OFF |
|---|---:|---:|
| Changed code values | 76,336 | 76,295 |
| Fraction of all code values changed | 1.819992% | 1.819015% |
| Carrier positions whose codes changed | 35,381 | 35,381 |
| Non-carrier code changes | 40,955 | 40,914 |
| Carrier change fraction | 97.9324% | 97.9324% |

Additional cross-configuration comparison:

| Metric | Result |
|---|---:|
| Clean code-value differences, nested vs plain | 0 |
| Embedded code-value differences, nested vs plain | 89 |
| Carrier positions with different codes across embedded artifacts | 47 |
| Non-carrier positions with different codes across embedded artifacts | 42 |

The 89 cross-configuration differences compare two embedded artifacts; they
are not themselves an estimate of embedding-induced changes. The matching
clean controls show 41 more changed code values for nested ON in this run,
but this is descriptive and should not be treated as a general effect.

## 7. Artifact and log locations

Baseline:
- `../cache/matched_qse_nf4_20261010_r2/qse_embedded.pt`
- `../cache/matched_qse_nf4_20261010_r2/qse_metadata.json`
- `../cache/matched_qse_nf4_20261010_r2/expected_corpus.bin`
- `../cache/matched_qse_nf4_20261010_r2/messages_recovered.jsonl`

Nested statistics ON:
- `../cache/qse_nf4_stats_nested_20261010/qse_embedded.pt`
- `../cache/qse_nf4_stats_nested_20261010/qse_metadata.json`
- `../cache/qse_nf4_stats_nested_20261010/extract.log`
- `../cache/qse_nf4_stats_nested_20261010/clean_control.json`
- `../cache/qse_nf4_stats_nested_20261010/clean_control_codes.pt`

Nested statistics OFF:
- `../cache/qse_nf4_stats_plain_20261010/qse_embedded.pt`
- `../cache/qse_nf4_stats_plain_20261010/qse_metadata.json`
- `../cache/qse_nf4_stats_plain_20261010/extract.log`
- `../cache/qse_nf4_stats_plain_20261010/clean_control.json`
- `../cache/qse_nf4_stats_plain_20261010/clean_control_codes.pt`

Comparison JSON:
- `../cache/qse_nf4_stats_config_comparison_20261010.json`
- `../cache/qse_nf4_matched_clean_control_comparison_20261010.json`

## 8. Interpretation and limitations

Supported findings:
1. Both configurations achieved exact recovery for this payload and tensor.
2. The clean packed NF4 code arrays were identical across the two settings.
3. Their dequantized clean weights differed slightly.
4. The embedded code arrays differed at 89 positions.
5. Embedded-versus-clean code changes were similar in magnitude in this run.

Not established:
- A statistically significant advantage for either configuration.
- Detector resistance, stealth, or general detectability.
- Robustness to fine-tuning, pruning, or other transformations.
- Generalization to other tensors, model revisions, architectures, or payloads.
- Independence across repetitions.

## 9. Next steps

1. Preserve these artifacts and this report as the initial matched comparison.
2. Replicate on another tensor or model with the same two configurations and
   matching clean controls.
3. Record extraction results, distortion metrics and packed-code comparisons
   in a separate run entry; do not overwrite this initial result.
4. Only make broader claims after multi-run and cross-model evidence exists.
