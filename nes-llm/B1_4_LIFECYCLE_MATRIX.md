# B1.4 lifecycle matrix and provenance audit

**Status date:** 10 October 2026  
**Branch:** `research/contract-b-b14-and-evaluation`  
**Rule:** never mutate the pristine `cache/contract_b_nf4_b14_10k` artifact or reuse an existing output path. Each transformation gets a fresh output, independent receiver invocation, and immutable report.

## Bounded provenance audit — findings

The current repository evidence does not establish that `source_id` and `run_id` in the packed-NF4 detector dataset identify independent source checkpoints or independent embedding runs:

- `scripts/build_packed_nf4_manifest.py` constructs `source_id` as `{model_id}@{source_revision}`, but defaults to `{model_id}@revision-unrecorded`. Therefore a model ID is not revision-level source provenance.
- The manifest builder accepts `run_id` as a caller-supplied command-line string. It validates exact recovery and metadata consistency, but does not verify that the identifier maps to a separately executed embedding run.
- The adapter records only a partial quantizer configuration (requested block size; other configuration fields are unknown).
- The current grouped detector dataset has only eight source/run groups across three model families, and Q/K/V projections from the same Qwen checkpoint are not independent model sources.
- Cross-model AUCs are near chance, but small group counts limit uncertainty and transfer claims.

**Audit verdict: provenance independence is not established from the tracked adapter contract or current CSV summary.** The local CSV contains 606,208 rows balanced across clean/embedded labels and eight source/run combinations. Six Qwen groups are three layer-16 projection types (Q/K/V) crossed with `nested`/`plain` variants; TinyLlama and Gemma each have one Q-projection group. These are not eight independent model sources. The output does not establish that each clean/embedded pair differs only by embedding. Treat these as bookkeeping groups, not proven independent replications, until original dataset-generation logs/manifests and artifact hashes are checked locally. Do not regenerate or relabel the frozen dataset to make it appear more independent.

### Local CSV summary — completed

The user inspected the existing CSV without modifying it. Counts: Qwen 327,680 rows, Gemma 147,456, TinyLlama 131,072; 303,104 clean and 303,104 embedded rows. Eight source/run combinations: six Qwen projection/variant groups, one TinyLlama Q-projection group, and one Gemma Q-projection group. The CSV does not by itself prove source independence or matched-pair construction.

### Remaining minimal evidence capture

From `nes-llm/`, inspect the existing CSV's group counts and metadata without changing it:

```bash
../.venv/bin/python - <<'PY'
import pandas as pd
p = "../cache/packed_nf4_grouped_dataset_20261010.csv"
df = pd.read_csv(p, usecols=["source_id", "run_id", "model_id", "tensor_key", "role", "label"])
print("rows:", len(df))
for col in ["model_id", "source_id", "run_id", "tensor_key", "role", "label"]:
    print(f"\n[{col}]")
    print(df[col].value_counts(dropna=False).to_string())
print("\nsource/run/model combinations:")
print(df[["source_id", "run_id", "model_id"]].drop_duplicates().sort_values(
    ["model_id", "source_id", "run_id"]
).to_string(index=False))
PY
```

This is descriptive only. Independence requires the artifact-generation records that connect each ID to a source checkpoint revision/hash and a unique run invocation.

## Lifecycle matrix

Each row is a separate experiment from the pristine B1.4 stego checkpoint. Always run the standard receiver after a transformation; never infer success from the transform completing.

| Transformation | Status | Evidence / next action |
|---|---|---|
| Pristine artifact baseline | PASS | 10,000-bit payload recovered exactly, BER 0, as documented in `CONTRACT_B_B14.md`. |
| Fresh NF4 requantization | COMPLETED — RECOVERY FAILED | Existing frozen run: 30/10,000 bit errors, BER 0.003, checksum mismatch. Keep this negative result unchanged. |
| NF4 reload/save control without intentional weight change | ATTEMPTED — TRANSFORMATION ERROR | Local Transformers 5.16.1 loaded the NF4 checkpoint, but `save_pretrained()` raised `NotImplementedError` while reversing `Bnb4bitDeserialize`. No valid output checkpoint was produced; the receiver's `FileNotFoundError` on that incomplete directory is a consequence, not a payload-recovery result. Preserve the failed directory/report. This control is blocked by the serializer path; do not bypass the reverse conversion while weights remain quantized. |
| Structured pruning | NOT RUN | Apply one declared sparsity/configuration to a fresh copy; record selected tensor changes and recovery. Do not conflate pruning with arbitrary code mutation. |
| Fine-tuning / LoRA merge | NOT RUN | Optional only if a controlled, reproducible local training/update path already exists. No improvised training run is required to close the pilot. |
| Task/weight merge | NOT RUN | Optional only if a compatible, documented second checkpoint and merge configuration are already available. Otherwise mark not available. |

### Reload/save control attempt — local result

On 2026-10-10, the user attempted to load `cache/contract_b_nf4_b14_10k` as NF4 and save to `cache/contract_b_b14_reload_save_control_20261010`. Loading completed, but `save_pretrained()` raised `NotImplementedError` in Transformers 5.16.1 when `revert_weight_conversion()` requested the reverse operation for `Bnb4bitDeserialize`. The follow-up receiver failed because the output directory did not contain a model safetensors file or index. This is a transformation/serialization error, not evidence of payload loss or survival. Keep the failed output path occupied and preserve it; don't retry there. The workaround used by the existing requantization script is guarded to run only after dequantization, so it must not be copied into this control while the model remains quantized.

### Per-run record requirements

- Input artifact path and SHA-256; verify it is the pristine baseline before transformation.
- Unique, previously unused output/intermediate/report paths.
- Exact transformation, config, seed, package versions, timestamps, and exit status.
- Output artifact SHA-256 and selected packed tensor SHA-256.
- Standard B1.4 receiver status, recovered payload SHA-256, checksum result, and BER if the envelope can be decoded.
- Utility is a separate measurement; exact recovery does not imply preserved model behavior.

### Existing NF4 requantization run — do not repeat or overwrite

The recorded run in `CONTRACT_B_B14.md` used `cache/contract_b_b14_nf4_requantized_retry3_fp16` (saved selected weights were BF16 despite the directory name) and `cache/contract_b_b14_nf4_requantized_retry3`. Its report is `cache/contract_b_b14_nf4_requantized_retry3_requantization_report.json`. Those paths are occupied and frozen.

## Stop condition

The lifecycle pilot is complete when the pristine baseline and at least the already-observed requantization failure are reported alongside any feasible controlled reload/save or pruning result, and all unavailable expensive attacks are explicitly marked unavailable. Do not delay the paper indefinitely to obtain fine-tuning or task-merge runs.
