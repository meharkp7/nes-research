# Overnight Progress — Third-Model Validation + Prior-Art Investigation

Run started: 2026-10-16. Directory: `overnight_third_model_prior_art/`.
Both workstreams executed in the same run and were interleaved (literature while
the model exported; experiments while literature synthesis continued).

## Status summary

| Workstream | Status | Key result |
|---|---|---|
| A — Third-model validation | **COMPLETE** | StableLM-2-1.6B; 48/48 matrix conditions; 6/6 interior exact; capacity 1k–50k all exact |
| B — Prior-art investigation | **COMPLETE (scoped)** | 12 sources reviewed; decision: *potentially novel, important comparisons unresolved* |

## Workstream A — what was done

1. Inspected workspace: codec hash confirmed `07ce008d…`, repo clean at `aa7e1a7`.
2. Evaluated third-model candidates. **Gemma-2-2b rejected: gated repo, no
   `HF_TOKEN` in environment.** Selected `stabilityai/stablelm-2-1_6b`
   (`StableLmForCausalLM`, 24 layers, vocab 100,352), revision
   `f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef`.
3. Downloaded FP16 checkpoint (3.1 GB, 23 s).
4. Exported clean NF4 checkpoint (blocksize 64, nested True) — status PASS.
5. Smoke test on L0 q_proj — PASS.
6. Full six-tensor matrix, cycles 0–3 — 48/48 MEASURED, 0 failures.
7. Capacity extension 1k/5k/20k/50k bits — all interior exact.

Results (details in `THIRD_MODEL_VALIDATION.md`):
- `interior_nearest`: 24/24 exact, BER 0, 0 anchor changes, byte-idempotent.
- `pair_parity`: 6/24 exact (cycle 0 only), BER up to 0.0040 at cycles 1–3.
- Third architecture family reproduces the TinyLlama/Qwen pattern.

## Workstream B — what was done

Sources read (primary text / method sections): Li et al. 2023; Dubin 2023;
Guo et al. 2025; Luan et al. 2025; EmMark 2024; Chen & Wornell 2001; Altun et
al. 2006; R-QIM 2305.17879; OSM survey 2502.10525; TransMark 2026 (abstract);
Kumaresam et al. 2026 (abstract); Snyk/HiddenLayer (implementation).

Decision: **Potentially novel, but important comparisons remain unresolved.**
Narrow differentiator candidate = NF4 packed code-index carriers + interior-only/
endpoint-preserving nearest-same-parity mapping + exact recovery over repeated
requantization. Not stated as "first."

## Errors and recovery actions

| Event | Action |
|---|---|
| Gemma-2-2b gated, no HF token | Recorded reason; switched to ungated StableLM with architectural diversity |
| `exp_third_model_matrix.py` first run: `ModuleNotFoundError: scripts` | Script lives outside `code/nes-llm`; fixed `sys.path` to absolute repo paths; reran |
| (No VM restart during this run) | Resumable runner built regardless; verified skip-on-restart behaviour |

## Artifacts

Workstream A (`workstream_a/`): `download_ckpt.py`, `exp_third_model_matrix.py`,
`export_stablelm.json`, `export.log`, `smoke.log`, `smoke/`, `matrix_v1/`
(manifest + 48 conditions), `capacity_{1000,5000,20000,50000}/`,
`FROZEN_WORKSTREAM_A_HASHES.txt` (60 entries).
Checkpoints: `ckpt/stablelm_fp16`, `ckpt/stablelm_clean`.

Workstream B: `PRIOR_ART_REVIEW.md`, `PRIOR_ART_BIBLIOGRAPHY.bib`.

Top level: `THIRD_MODEL_VALIDATION.md`, `PRIOR_ART_REVIEW.md`,
`PRIOR_ART_BIBLIOGRAPHY.bib`, `OVERNIGHT_PROGRESS.md`.

## Remaining / incomplete

- Workstream A: no detector or PPL evaluation for StableLM (single/two-model
  results still stand). One payload size for the main matrix (10k); one seed.
- Workstream B: full-text read of EmMark/TransMark not completed beyond
  method/abstract; terminology sweep for equivalent names not exhaustive;
  IEEE/ACM full texts partly inaccessible.

## Exact resume instructions

Nothing in Workstream A is mid-flight; all launched processes have exited
(verified). To re-verify or extend, the runner skips finished conditions:

```bash
cd /workspace/nes_run_20261016_anchor_ablation/code/nes-llm
/workspace/nes_env/bin/python \
  /workspace/nes_run_20261016_anchor_ablation/overnight_third_model_prior_art/workstream_a/exp_third_model_matrix.py \
  --checkpoint /workspace/nes_run_20261016_anchor_ablation/ckpt/stablelm_clean \
  --output /workspace/nes_run_20261016_anchor_ablation/overnight_third_model_prior_art/workstream_a/matrix_v1
```

To add a new model, rerun steps 2–4 of Workstream A with a new `--checkpoint`
and a fresh `--output` directory (never reuse an existing one).
