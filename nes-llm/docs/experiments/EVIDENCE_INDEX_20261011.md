# NES Evidence Index — 11 October 2026

- Branch: `research/contract-b-b14-and-evaluation`
- Generated (UTC): `2026-10-10T21:30:30.789602+00:00`
- Hash algorithm: SHA-256

## Focused test gate

**30 passed, 4 warnings in 4.54s.** Ten focused test files; this is not
a full-repository test result. The warnings concern scikit-learn MLP
`batch_size` clipping in the grouped-sensitivity test.

## Residual descriptive batch

- Cells: 5
- Methods: lwe_grid_parity, magnitude_aware, qae, sign, split_sign_parity
- Recovery statuses all PASS: True
- Paired descriptive outputs all ready: True
- Feature rows: 309760
- Layer summaries: 55

This is descriptive feature export, not a trained detector benchmark.
Correlated blocks must not be treated as independent inferential samples.
The batch does not establish stealth, undetectability, or cross-model
generalization. QSE/DCE comparative detectability remains blocked until
a historically matched same-source clean packed-NF4 control is verified.

## Artifact manifest

| Repository-relative path | Bytes | SHA-256 |
|---|---:|---|
| `cache/detectability_tinyllama_residual_20261011/cell_status.csv` | 2758 | `d15619565a042075e8af32798c9c9554e179c9b4d79c4617ce36c70afc526f24` |
| `cache/detectability_tinyllama_residual_20261011/detectability_stage1_report.json` | 1929 | `6a126ce1968dba0fb8e3d97a9933d764129db6a9b564eeb979555ba59b74d3db` |
| `cache/detectability_tinyllama_residual_20261011/progress.json` | 465 | `bdec0bb70ff7126c82afb59723f7b18703c6a291c4322cc4cc54cfd93b421e74` |
| `cache/detectability_tinyllama_residual_20261011/residual_layer_descriptives.csv` | 73495 | `694316693fc9003e705822256df789b6186f4fdb4b79fcdab828d0c97bd4eaa1` |
| `cache/detectability_tinyllama_residual_20261011/residual_paired_block_features.csv` | 178362048 | `e550f412f80787441a21af5a508c13644fa133683426e7ac20f715508bb42bc3` |
| `cache/seven_method_evidence_inventory.json` | 71351 | `bbca8f2ad41e2e3c5d0492b6a9e9d4dae1af1c8eb782fc0ee9299f59cb5786b6` |
| `cache/seven_method_evidence_inventory.csv` | 35690 | `eb19abb77f5ff7f75fca03e677e1f14f5d4e936d7175b12bf186f5a13731fd0b` |
| `nes-llm/tests/test_seven_method_protocol.py` | 3972 | `ae11dcedc0d9a2cd4e35c241535b84b4952590e88f8db21e1d3837098ddf2abb` |
| `nes-llm/tests/test_embedding.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `nes-llm/tests/test_extraction.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `nes-llm/tests/test_residuals.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `nes-llm/tests/test_matrix_extraction_report_parser.py` | 1441 | `0d7b2905124e4a5dd28f33a5b1e559cbf4e7dc091974d197b58213fa21449f38` |
| `nes-llm/tests/test_contract_b_robustness_matrix.py` | 2444 | `d5389fbaa40507ee15044e46e696e1219cf38843914f59c15339256adc0c1e70` |
| `nes-llm/tests/test_validate_packed_nf4_detector_readiness.py` | 4204 | `0c628df489ed9f2c70923706e773e5ddefda9151555211e1a6cd736075350ebb` |
| `nes-llm/tests/test_run_packed_nf4_cross_model_detector.py` | 2389 | `95948f3aa1949f3b87fa868391c7ae3da1c7f9e57470069ce72c6e93757acc45` |
| `nes-llm/tests/test_run_packed_nf4_detector_ablation.py` | 3869 | `4d9c3fd44e8cf6fdaa367950f4df9b7d7dbdf3504eb8ad8dfe400977029f91e5` |
| `nes-llm/tests/test_run_packed_nf4_detector_benchmark.py` | 3582 | `cefed3ab1af842dfe26550012831318f178f91395817fbb8bf337913bb909bf1` |

## Missing expected artifacts

None.

## Interpretation guardrails

- Recovery success and detector performance are separate claims.
- Do not infer independent replications from blocks, projections, or duplicate artifacts.
- Preserve failed and blocked results; do not rewrite historical statuses.
- This index hashes files available at generation time; rerun it if files change.
