# Packed-NF4 paired analysis patch

Adds a standalone analysis script and unit tests without modifying existing files.

From `nes-llm`, copy the script to `scripts/` and test to `tests/`, then run:

```bash
../.venv/bin/python -m pytest -q tests/test_analyze_packed_nf4_paired_features.py
../.venv/bin/python scripts/analyze_packed_nf4_paired_features.py \
  --input ../cache/packed_nf4_grouped_dataset_20261010.csv \
  --output-dir ../cache/packed_nf4_paired_analysis_20261010_r1
```

The output directory must not exist. The script validates exact pairing by `(source_id, run_id, block_index)`, enforces clean/embedded role-label consistency, records input SHA-256, computes paired descriptive effects and exploratory block-level sign-flip p-values with Benjamini-Hochberg adjustment per source, and prominently warns that block-level p-values are not run-level inference. It does not modify the dataset or existing reports.

Note: the current dataset has only one run for Gemma and TinyLlama and two Qwen Q-projection runs. More independent runs are required for generalizable inferential claims. These block-level p-values must remain exploratory.
