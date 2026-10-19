# Third-Model Validation — StableLM-2-1.6B

Run id: `nes_third_model_stablelm_20261016`. Status: **COMPLETE**.
Method: frozen `interior_nearest` codec, sha256
`07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` (unchanged).

This is a third independent architecture, disjoint from the five residual-method
track and the QSE/DCE packed-NF4 track.

## 1. Model selection

**Gemma 2 2B was not usable.** `google/gemma-2-2b` is a *gated* repository
(`gated=manual`, requires license acceptance), and this environment has no
`HF_TOKEN` and no stored HF token file. Selecting it would have required
account-gated access that is unavailable here. Recorded as the reason, not
silently skipped.

**Selected:** `stabilityai/stablelm-2-1_6b`.

| Item | Value |
|---|---|
| Repository | `stabilityai/stablelm-2-1_6b` |
| Immutable revision | `f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef` |
| Architecture | `StableLmForCausalLM` (distinct family from Llama-style TinyLlama/Qwen) |
| Layers / hidden / vocab | 24 / 2048 / 100,352 (distinct vocab from both prior models) |
| License / access | Ungated, public download (no token) |
| FP16 checkpoint size | 3.1 GB, single shard |
| Why this model | Adds a third architecture family and a third tokenizer/vocab; single-shard keeps it restart-robust and cheap |
| Loader compatibility | Llama-style tensor names (`model.layers.N.self_attn.{q,k,v}_proj.weight`, `mlp.{up,down,gate}_proj.weight`); NF4 blocksize 64, nested True — supported by the existing pipeline without codec changes |

Storage/peak memory: `ckpt/stablelm_fp16` 3.1 GB; NF4 clean export 1.46 GB;
export peak host memory stayed within the 15 GB sandbox (≈7 GB free during
export). No GPU.

## 2. Provenance

| Item | Value |
|---|---|
| FP16 `model.safetensors` sha256 | `8bdf317e2b35ab5c8009cbb6c7ce495e4e608a6b9b843d44054edf25b8c5860d` |
| Clean NF4 checkpoint | `ckpt/stablelm_clean` — `model.safetensors` sha256 `bff3b65ebb686092b7234ddef66d3a69b762299826ac3e29e2015343c432d591` |
| Export report | `workstream_a/export_stablelm.json` (status **PASS**, stored dtype bfloat16, nested True) |
| Codec sha256 | `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` |
| Checkpoint runner sha256 | `18dbe6fb9da621d1aba6148806a585962fb16f80d7be79cc65ca9c8a676c871b` |
| Quantization | nf4, blocksize 64, nested statistics, U8 storage (504 packed tensors) |
| Payload | 10,000 bits, seed 20261016; `payload_sha256` `580fde145167ba17…` (same deterministic payload as prior matrices) |
| Versions | torch 2.14.1+cpu, bitsandbytes 0.50.2, numpy 2.5.4, safetensors 0.8.0 |
| Platform | Linux x86_64, CPU only |

Source tensor sha256 (preflight, `matrix_v1/manifest.json`):

| Tensor | source tensor sha256 (prefix) | eligible interior codes |
|---|---|---:|
| `model.layers.0.self_attn.q_proj.weight` | `216ebcf4c9eb7a6f…` | 4,085,393 |
| `model.layers.11.self_attn.k_proj.weight` | `7ec45b35a7d0e8e3…` | 4,047,359 |
| `model.layers.23.self_attn.v_proj.weight` | `334a16f07050fa9f…` | 4,047,241 |
| `model.layers.0.mlp.up_proj.weight` | `08b5bb08fbbda877…` | 11,134,060 |
| `model.layers.11.mlp.down_proj.weight` | `fb6295dcf271cb84…` | 11,143,484 |
| `model.layers.23.mlp.gate_proj.weight` | `29a90b84a7b3e236…` | 11,115,818 |

Architecture-specific adaptation: **none to the algorithm.** Only tensor
selection was adapted (layer indices 0/11/23 for a 24-layer model; roles
q/k/v and up/down/gate). No change was made to make the model pass.

## 3. Smoke test (before the matrix)

Single tensor `model.layers.0.self_attn.q_proj.weight`, cycles 0–1.
Output: `workstream_a/smoke/`.

| Condition | exact | BER |
|---|---|---|
| interior_nearest c0 | True | 0.0 |
| interior_nearest c1 | True | 0.0 |
| pair_parity c0 | True | 0.0 |
| pair_parity c1 | False | 0.0035 |

Smoke test PASS: source tensor loaded, packed NF4 valid, frozen codec embedded
and extracted, artifact reloaded, and a requantize cycle completed correctly.
The baseline failing at cycle 1 already matches the TinyLlama/Qwen pattern.

## 4. Six-tensor matrix (cycles 0–3)

`workstream_a/matrix_v1/` — **48/48 conditions MEASURED, 0 failures.**

Aggregate:

| Method | exact | BER range | anchors changed (sum) | idempotent (cycles 0–3) |
|---|---|---|---|---|
| `interior_nearest` | **24/24** | 0.00000 | **0** | yes (1 distinct hash per tensor) |
| `pair_parity` | **6/24** | 0.00000–0.00400 | 3,884 | no (2 distinct hashes per tensor) |

`interior_nearest`, cycle 3, per tensor:

| Tensor | exact | BER | chg_anchor | no-endpoint blocks | pos-agree | output sha (prefix) |
|---|---|---|---|---|---|---|
| L0 attn q_proj | True | 0.0 | 0 | 0 | 1.0000 | `658877c285bc…` |
| L11 attn k_proj | True | 0.0 | 0 | 0 | 1.0000 | `6e076afbce7f…` |
| L23 attn v_proj | True | 0.0 | 0 | 0 | 1.0000 | `420a04ce83c3…` |
| L0 mlp up_proj | True | 0.0 | 0 | 0 | 1.0000 | `8473ac337829…` |
| L11 mlp down_proj | True | 0.0 | 0 | 0 | 1.0000 | `649ae34a0153…` |
| L23 mlp gate_proj | True | 0.0 | 0 | 0 | 1.0000 | `dfb51ebe406a…` |

`pair_parity`: exact at cycle 0 only, fails cycles 1–3 on all six tensors
(BER 0.0030–0.0040), exactly the baseline behaviour seen on TinyLlama and Qwen.

## 5. Capacity extension (1k–50k bits)

`workstream_a/capacity_{1000,5000,20000,50000}/` — representative tensor
L0 q_proj, seed 20261016, cycles 0–3.

| Bits | interior_nearest exact | BER | chg_anchor | pair_parity exact | BER |
|---:|---|---:|---:|---|---:|
| 1,000 | True | 0.0 | 0 | False | 0.0030 |
| 5,000 | True | 0.0 | 0 | False | 0.0034 |
| 20,000 | True | 0.0 | 0 | False | 0.0050 |
| 50,000 | True | 0.0 | 0 | False | 0.0056 |

## 6. Cross-model comparison

| Model | Method | exact (6 tensors) | BER mean | anchors changed | idempotent |
|---|---|---|---|---|---|
| TinyLlama-1.1B | interior_nearest | 6/6 | 0.0 | 0 | yes |
| TinyLlama-1.1B | pair_parity | 0/6 | 0.00405 | 979 | no |
| Qwen2.5-3B | interior_nearest | 6/6 | 0.0 | 0 | yes |
| Qwen2.5-3B | pair_parity | 0/6 | 0.00452 | 1,077 | no |
| **StableLM-2-1.6B** | **interior_nearest** | **6/6** | **0.0** | **0** | **yes** |
| **StableLM-2-1.6B** | **pair_parity** | **0/6** | **0.00355** | **3,884** | **no** |

Three architecture families, three tokenizers, the same pattern: the
nearest-same-parity interior operation survives requantization; the baseline
parity operation does not.

## 7. Resource usage and limitations

- Runtime: export 33.6 s; smoke ~25 s; full matrix ~4 min; capacity ~3 min.
- Memory: within 15 GB sandbox; no GPU.
- **Limitations.** One payload size for the main matrix (10,000 bits), one seed
  (20261016), 3 requantization cycles. StableLM is a third model, not a
  distribution over models. The frozen revision is `f499ead7…` (a resolvable
  immutable commit, unlike the Qwen `main` caveat). No detector or PPL
  evaluation was run for this model — those remain single/double-model results.
- The reduced-scope caveats of the existing evidence package still apply; this
  run does not extend them.

## 8. Resumability

Runner: `workstream_a/exp_third_model_matrix.py`. One durable JSON per
(tensor, method, cycle), written to a temp file, validated, then atomically
renamed. Completed conditions are skipped on restart. Verified: re-running a
finished output directory skips all existing conditions.

Resume command:

```bash
cd /workspace/nes_run_20261016_anchor_ablation/code/nes-llm
/workspace/nes_env/bin/python \
  /workspace/nes_run_20261016_anchor_ablation/overnight_third_model_prior_art/workstream_a/exp_third_model_matrix.py \
  --checkpoint /workspace/nes_run_20261016_anchor_ablation/ckpt/stablelm_clean \
  --output /workspace/nes_run_20261016_anchor_ablation/overnight_third_model_prior_art/workstream_a/matrix_v1
```

## 9. Guardrails honoured

- Frozen codec unchanged (`07ce008d…`); no algorithm change to fit the model.
- New output directories only; no historical artifact read, written, or overwritten.
- No branch change, commit, or push; repo working tree clean at `aa7e1a7`.
- Kept separate from the five residual methods and QSE/DCE track.
