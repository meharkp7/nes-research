# QSE NF4 Statistics Ablation: Q/K/V Synthesis

**Date:** 2026-10-10  
**Model:** `Qwen/Qwen2.5-3B`  
**Pinned source revision:** `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`  
**Method:** QSE, reference-assisted receiver  
**Payload:** 36,128 bits across seven multilingual messages  
**NF4 block size:** 64  
**Variable:** `compress_statistics=True` (nested statistics ON) versus `False` (OFF)

## 1. Scope and experimental controls

This report consolidates the Q-, K-, and V-projection statistics ablations conducted on 2026-10-10. Within each projection, the intended experimental variable was the NF4 `compress_statistics` setting. The model revision, tensor, payload corpus, block size, QSE margin, and maximum-attempt settings were held constant.

The payload corpus SHA-256 was:

`62c30141c4f73551d19ff4ab0e3d37b87eb581b2cb5f08179a4401ce4d622235`

All six embedding runs (three projections × two statistics settings) passed extraction with:
- 7 recovered messages;
- 36,128 expected and recovered bits;
- 0 bit errors and BER = 0.0;
- exact corpus match and matching embedded manifest digest.

The receiver is **reference-assisted QSE**. These results do not establish blind extraction.

## 2. Projection-level packed-code results

### 2.1 Embedded ON-versus-OFF comparison

| Projection | Tensor values | Differing embedded codes | Difference fraction |
|---|---:|---:|---:|
| Q | 4,194,304 | 89 | 0.002122% |
| K | 524,288 | 98 | 0.018692% |
| V | 524,288 | 31 | 0.005913% |

Q: 47 differing positions were carriers and 42 were non-carriers.

K: 59 differing positions were carriers and 39 were non-carriers. Carrier positions matched across runs.

V: 31 packed codes differed between runs. Carrier positions were not reported as unequal in the direct comparison; the direct-comparison report records only total differing codes.

For all three projections, the matching clean controls had identical packed codes between statistics settings. The embedded artifacts did not.

### 2.2 Embedded versus matching clean controls

| Projection | Setting | Codes changed | Change fraction | Carrier codes changed | Non-carrier codes changed |
|---|---|---:|---:|---:|---:|
| Q | ON | 76,336 | 1.819992% | 35,381 | 40,955 |
| Q | OFF | 76,295 | 1.819015% | 35,381 | 40,914 |
| K | ON | 70,798 | 13.503647% | 35,266 | 35,532 |
| K | OFF | 70,763 | 13.496971% | 35,266 | 35,497 |
| V | ON | 64,748 | 12.349701% | 35,309 | 29,439 |
| V | OFF | 64,735 | 12.347221% | 35,309 | 29,426 |

The carrier-change fractions were approximately:
- Q: 97.9324% in both settings;
- K: 97.6140% in both settings;
- V: 97.7331% in both settings.

These are packed-code comparisons against configuration-matched clean controls. A code differing from the clean baseline is not automatically attributable only to the payload: the quantization path and reconstruction settings also matter.

## 3. Dequantized-weight comparisons

### 3.1 Embedded artifact versus matching clean control

| Projection | Setting | Weight RMSE | Maximum absolute difference |
|---|---|---:|---:|
| Q | ON | 0.002050679 | 0.041592475 |
| Q | OFF | 0.002046299 | 0.041645352 |
| K | ON | 0.006386017 | 0.045533404 |
| K | OFF | 0.006380127 | 0.045451991 |
| V | ON | 0.005156734 | 0.039479390 |
| V | OFF | 0.005155111 | 0.039513499 |

The Q values are reported as `weight_rmse_vs_clean_nf4` in the saved Q comparison JSON. K and V values come from their respective direct distortion comparison outputs.

### 3.2 Direct embedded ON-versus-OFF weights

| Projection | Embedded-weight RMSE | Embedded-weight maximum absolute difference | Clean-weight RMSE ON vs OFF |
|---|---:|---:|---:|
| Q | 0.000097857 | 0.020888079 | 0.000089446 |
| K | 0.000166040 | 0.024037525 | 0.000105951 |
| V | 0.000080982 | 0.010878917 | 0.000058611 |

For K and V, clean packed codes are identical across settings, but the reconstructed clean weights are not bitwise/numerically identical. This is why packed-code equality and dequantized-weight equality are reported separately.

All tested K and V embedded and clean weight vectors were finite.

## 4. Clean-control quantization error versus FP16 source

| Projection | Nested statistics ON | Nested statistics OFF |
|---|---:|---:|
| Q | 0.002626629 | 0.002625196 |
| K | 0.002905695 | 0.002904031 |
| V | 0.002343701 | 0.002343288 |

These are the measured clean-control RMSE values versus the loaded FP16 source weights.

**Provenance limitation:** The clean controls were generated from the pinned revision shown above. Their metadata explicitly records the historical QSE revision match as `UNVERIFIED`. They should therefore be described as matched controls for the current pinned revision, not as proven reproductions of the exact historical embedding-time quantization state.

## 5. Interpretation

1. **Recovery:** Exact recovery was observed for all six QSE runs with the specified reference-assisted receiver and corpus.
2. **Representation sensitivity:** Toggling nested statistics changed the embedded packed representation in all three projections. The clean packed codes were identical across settings for each projection.
3. **Projection dependence:** The number and fraction of codes changed relative to clean controls differed across Q, K, and V. These counts are not directly interchangeable as a ranking of detectability because the tensors have different sizes and different baseline distributions.
4. **Weight-space effects:** Direct ON/OFF dequantized-weight comparisons are available for K and V. The corresponding Q measurement is missing from the inventoried results and is not inferred here.
5. **Detectability:** Code-change counts and weight RMSE do not establish detector accuracy, stealth, or indistinguishability. A separate, properly controlled detector evaluation is needed.
6. **Robustness:** Exact extraction from these artifacts does not demonstrate survival under re-quantization, fine-tuning, pruning, serialization changes, or other transformations.
7. **Model behavior:** No task-quality, perplexity, or behavioral equivalence conclusion is drawn from these measurements alone.

## 6. Source artifacts

Results were consolidated from these local JSON artifacts:

- `../cache/qse_nf4_stats_config_comparison_20261010.json`
- `../cache/qse_nf4_matched_clean_control_comparison_20261010.json`
- `../cache/qse_nf4_kproj_stats_comparison_20261010.json`
- `../cache/qse_nf4_kproj_embedded_stats_direct_comparison_20261010.json`
- `../cache/qse_nf4_kproj_distortion_comparison_20261010.json`
- `../cache/qse_nf4_vproj_stats_comparison_20261010.json`
- `../cache/qse_nf4_vproj_distortion_comparison_20261010.json`

Projection-specific reports remain separate:
- `QSE_NF4_STATS_COMPARISON_20261010.md`
- `QSE_NF4_STATS_COMPARISON_KPROJ_20261010.md`

The frozen pilot and packed-NF4 detectability audit were not modified by this synthesis.

## 7. Follow-up work

1. Direct Q embedded-weight and clean-weight ON/OFF distortion has now been measured; see the saved Q-projection distortion JSON.
2. Verify historical embedding-time quantizer settings and provenance where original metadata permits.
3. Evaluate detector performance against matched clean controls, with explicit train/test separation and controls for tensor identity, model revision, and statistics configuration.
4. Keep extraction, packed-code changes, dequantized-weight distortion, model utility, robustness, and detectability as separate evaluation dimensions.
5. Extend the study to additional models and methods only with recorded artifact hashes, pinned revisions, and predeclared acceptance criteria.
