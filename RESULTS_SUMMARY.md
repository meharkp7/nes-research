# NF4 Anchor-Preservation — Minimal Matched Causal A–D Experiment

Run id: `nes_anchor_ablation_ad_20261016`
Status: COMPLETED — 16 measured rows (4 arms × 4 cycles), no errors.

## Provenance / pinning

| Item | Value |
|---|---|
| Source model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Source revision | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` (pinned) |
| Source FP16 `model.safetensors` sha256 | `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933` |
| Clean NF4 checkpoint (built here) | `ckpt/tinyllama_clean` (single `model.safetensors`, 762 MB) |
| Clean NF4 file sha256 | `f84d0a8ec6e011d683ae7d1aec961d4bb021b254da9d84516185df428f3b37a7` |
| Tensor | `model.layers.0.self_attn.q_proj.weight` (2048×2048, packed uint8 2,097,152 B) |
| **Source tensor sha256** | `283aa75d1f34daff33b4aae984c42fac6a8741deecd3348eaed6459852fd2161` |
| Codec revision | `src/experiments/nf4_anchor_channel.py` sha256 `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` |
| Checkpoint runner revision | `scripts/nf4_anchor_checkpoint.py` sha256 `18dbe6fb9da621d1aba6148806a585962fb16f80d7be79cc65ca9c8a676c871b` |
| blocksize | 64 |
| nested statistics | True (double quant) |
| Payload | 10,000 bits, seed 20261016 |
| Versions | torch 2.14.1+cpu, bitsandbytes 0.50.2, numpy 2.5.4, safetensors 0.8.0, transformers 5.20.1, accelerate 1.15.0 |
| Platform | Linux (CPU only; no CUDA) |

Recorded historical versions were torch 2.13.0 / numpy 2.5.2; torch and numpy differ here.
bitsandbytes 0.50.2 and safetensors 0.8.0 match exactly.

## Exact commands

```bash
# 1. deps (isolated venv, outside repo)
python3 -m venv /workspace/nes_env
/workspace/nes_env/bin/pip install numpy safetensors bitsandbytes transformers accelerate
/workspace/nes_env/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. fetch pinned snapshot (2.20 GB)
#    huggingface_hub.snapshot_download(repo, revision=fe8a4ea1..., local_dir=ckpt/tinyllama_fp16)

# 3. build clean NF4 checkpoint (existing script, export mode)
cd .../code/nes-llm
python scripts/nf4_anchor_checkpoint.py export \
  --checkpoint .../ckpt/tinyllama_fp16 \
  --output     .../ckpt/tinyllama_clean \
  --report     .../out/export_clean_v2.json \
  --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0

# 4. four-arm matched experiment (new script, reuses frozen codec helpers)
python scripts/exp_anchor_ablation_ad.py \
  --checkpoint .../ckpt/tinyllama_clean \
  --output     .../out/anchor_ablation_ad_v2
```

## Arms

| Arm | Eligibility | Op | Endpoints eligible? |
|---|---|---|---|
| A `interior_nearest_anchored` | anchored blocks, codes 1–14 | nearest same-parity | no |
| B `nearest_parity_anchored` | anchored blocks, all codes | `code&14\|bit` | **yes** |
| C `interior_nearest_unanchored` | all interior codes 1–14, no block restriction | nearest same-parity | no |
| D `pair_parity_unrestricted` | all codes | `code&14\|bit` | yes |

A and D are byte-identical to the frozen codec's `interior_nearest` and `pair_parity`
(validated: identical packed output and `changed_codes`).

## Results

| Arm | cycle | eligible | changed | anchor changes | no-endpoint blocks | pos-agree | exact | BER |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| A interior_nearest_anchored | 0 | 4,091,662 | 5,089 | 0 | 0 | 1.0000 | True | 0.0 |
| A | 1 | 4,091,662 | 5,089 | 0 | 0 | 1.0000 | True | 0.0 |
| A | 2 | 4,091,662 | 5,089 | 0 | 0 | 1.0000 | True | 0.0 |
| A | 3 | 4,091,662 | 5,089 | 0 | 0 | 1.0000 | True | 0.0 |
| B nearest_parity_anchored | 0 | 4,194,304 | 5,153 | 110 | 37 | 1.0000 | False | 0.48760 |
| B | 1 | 4,194,304 | 5,153 | 110 | 0 | 0.0271 | False | 0.00380 |
| B | 2 | 4,194,304 | 5,153 | 110 | 0 | 0.0271 | False | 0.00380 |
| B | 3 | 4,194,304 | 5,153 | 110 | 0 | 0.0271 | False | 0.00380 |
| C interior_nearest_unanchored | 0 | 4,091,662 | 5,089 | 0 | 0 | 1.0000 | True | 0.0 |
| C | 1 | 4,091,662 | 5,089 | 0 | 0 | 1.0000 | True | 0.0 |
| C | 2 | 4,091,662 | 5,089 | 0 | 0 | 1.0000 | True | 0.0 |
| C | 3 | 4,091,662 | 5,089 | 0 | 0 | 1.0000 | True | 0.0 |
| D pair_parity_unrestricted | 0 | 4,194,304 | 5,153 | 110 | 37 | 1.0000 | True | 0.0 |
| D | 1 | 4,194,304 | 5,153 | 110 | 0 | 1.0000 | False | 0.00400 |
| D | 2 | 4,194,304 | 5,153 | 110 | 0 | 1.0000 | False | 0.00400 |
| D | 3 | 4,194,304 | 5,153 | 110 | 0 | 1.0000 | False | 0.00400 |

Cycle-0 `output_packed_sha256` (arm prefix):
A/C `783d43d8932897e70aa6…` (identical), B `ede06c212953ce55b1fc…`, D `2ee7ba3661b12ad6d1f1…`.

### Causal reading

1. **Touching endpoint anchors causes the failure.** B vs A differ only by
   (i) endpoints eligible and (ii) parity vs nearest op. B fails; A passes.
2. **Excluding anchorless blocks is NOT the operative factor.** C is the
   decisive control: it drops the anchorless-block exclusion and still passes
   with 0 anchor changes. So the anchorless-block exclusion is not what
   produces robustness.
3. **The nearest-same-parity op is more robust than the baseline parity op.**
   D (unrestricted, baseline op) also fails, with BER 0.0040. B (restricted,
   baseline op) fails harder (BER 0.4876 at cycle 0 — a real decode failure).
   So arm C isolates the *interior/nearest op*, not the eligibility mask, as
   the source of robustness.

**Important limit on the claim:** this experiment does **not** cleanly isolate
endpoint preservation from the embedding *op*, because A/C and B/D differ in
both. The evidence establishes: the nearest-same-parity interior op survives
requantization; the `code&14|bit` parity op does not; and the anchorless-block
exclusion is not necessary for the survival observed here. A clean
"endpoints-only" control would require the baseline op run with endpoints
excluded, which the frozen codec does not provide.

## Cross-check against frozen historical evidence

The freshly built clean tensor hash **exactly matches** the historical
`source_tensor_sha256` for this tensor in
`artifacts/method_paper_20261016/tinyllama_pilot_v2/observations.jsonl`
(`283aa75d…`), and the source FP16 file hashes match `tinyllama_export.json`.

My arm A/D `output_packed_sha256` reproduce the historical committed hashes
byte-for-byte at every cycle:

| method | cycle | historical | mine | match |
|---|---:|---|---|---|
| interior_nearest | 0–3 | `783d43d8932897e70a…` | same | ✅ |
| pair_parity | 0 | `2ee7ba3661b12ad6d1…` | same | ✅ |
| pair_parity | 1–3 | `4090675f4b6db1812a…` | same | ✅ |

This independently reproduces the previously UNVERIFIED TinyLlama evidence
(tensor 0, 10,000 bits, seed 20261016) from a clean rebuild.

## Guardrails honoured

- Repo working tree clean; HEAD `aa7e1a7` unchanged; no branch switch/commit/push.
- Historical 2026-10-16 observations untouched (still only on the research branch).
- All new outputs under `/workspace/nes_run_20261016_anchor_ablation/` (outside repo).
- No residual-method or QSE/DCE track data used.
