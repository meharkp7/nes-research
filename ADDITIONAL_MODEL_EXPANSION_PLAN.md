# Additional-Model Expansion Plan (V2 — APPROVED FOR EXECUTION)

Revised 2026-10-19 under `QUALIFICATION_PROTOCOL_V2.md` (active; user-approved
Option A). Supersedes the earlier proposal version. **Target: four additional
architectures** beyond TinyLlama-1.1B, Qwen2.5-3B, and StableLM-2-1.6B;
**minimum three** if resources force it.

**Framing caveat (binding).** Additional models expand the *empirical evidence
base* for Track A. They do **not** establish universal generalization and do
**not** strengthen Track B (novelty). Every model is reported individually; a
failing model is a **counterexample**, reported not discarded.

Existing qualified models: TinyLlama-1.1B, Qwen2.5-3B, StableLM-2-1.6B.

---

## Resource envelope (measured 2026-10-19)

| Resource | Value | Consequence |
|---|---|---|
| `/workspace` free | **5.9 GB** (25 GB volume, 76% used) | Too small for multiple fp16 checkpoints |
| `/` overlay free | **61 GB** (`/tmp`, `/home/openhands`, `/var/tmp`) | Staging area for fp16 downloads |
| RAM | 15 GB (≈13 GB available) | Fine for ≤2B models |
| CPU | 4 cores, CPU-only | Utility ≈ 63 min/condition at 1.6B; roundtrip ≈ 2.3 s/condition |

### Disk strategy (no existing evidence deleted)

- **fp16 downloads → `/tmp/nes_stage/<model>/`** (overlay, 61 GB free). Cleared
  after the clean NF4 checkpoint is derived; the fp16 is a re-downloadable
  intermediate, **not** evidence. Nothing under `/workspace` is ever deleted.
- **Clean NF4 checkpoint → `/workspace/.../ckpt/<model>_clean/`** (≈1.5–1.8 GB),
  kept as provenance with recorded hashes.
- **One model at a time.** Model N's fp16 staging is cleared before model N+1
  downloads. If `/workspace` would be exceeded, the plan reports the shortfall
  rather than deleting evidence.
- If `/tmp` is cleared by a runtime restart, the fp16 is re-downloaded; clean NF4
  in `/workspace` persists.

## Selected four models (metadata verified via HF API 2026-10-19)

Diversity goal: four **distinct** architecture lineages, all ≤2B.

### M4 — HuggingFaceTB/SmolLM2-1.7B  (primary)

| Field | Value | Verified? |
|---|---|---|
| Model ID | `HuggingFaceTB/SmolLM2-1.7B` | yes |
| Revision (main) | `effd688a12921b4cc83e3312b6feb579f70f9c71` | yes (immutable commit; re-check at download) |
| Family | Llama-style dense decoder, independent pretraining lineage | yes |
| License | apache-2.0 | yes |
| Gating | none | yes |
| Weights | 3.42 GB (single safetensors) | yes |
| Tokenizer | `tokenizer.json` present | yes |

### M5 — Qwen/Qwen2.5-1.5B  (primary)

| Field | Value | Verified? |
|---|---|---|
| Model ID | `Qwen/Qwen2.5-1.5B` | yes |
| Revision (main) | `8faed761d45a263340a0528343f099c05c9a4323` | yes |
| Family | Qwen2 dense decoder (scale-within-family vs Qwen2.5-3B) | yes |
| License | apache-2.0 | yes |
| Gating | none | yes |
| Weights | 3.09 GB | yes |
| Tokenizer | present | yes |

### M6 — EleutherAI/pythia-1.4b  (primary)

| Field | Value | Verified? |
|---|---|---|
| Model ID | `EleutherAI/pythia-1.4b` | yes |
| Revision (main) | `fedc38a16eea3bd36a96b906d78d11d2ce18ed79` | yes |
| Family | GPT-NeoX-style dense decoder (distinct lineage, different layer names) | yes |
| License | apache-2.0 | yes |
| Gating | none | yes |
| Weights | 2.93 GB safetensors (repo also ships duplicate `.bin`; fetch safetensors only) | yes |
| Tokenizer | present | yes |

### M7 — tiiuae/Falcon3-1B-Base  (primary)

| Field | Value | Verified? |
|---|---|---|
| Model ID | `tiiuae/Falcon3-1B-Base` | yes |
| Revision (main) | `cb37ef3559b157b5c9d9226296ba01a5162da1f7` | yes |
| Family | Falcon3 dense decoder (distinct lineage) | yes |
| License | `other` (TII Falcon license) — terms not fully reviewed; research use assumed, redistribution NOT assumed | flag |
| Gating | none | yes |
| Weights | 3.34 GB | yes |
| Tokenizer | present | yes |

**Alternates** (if one of M4–M7 is blocked): `bigscience/bloom-1b1`
(`eb3dd739…`), `facebook/opt-1.3b` (`3f5c25d0…`, no repo tokenizer file — needs
`GPT2Tokenizer` fallback). **Excluded:** `meta-llama/Llama-3.2-1B` and
`google/gemma-2-2b` (gated/manual approval).

## Per-model execution (identical predeclared protocol; no per-model weakening)

For each model M, in order:

1. **Download** fp16 to `/tmp/nes_stage/<M>/` at the pinned revision; record the
   resolved revision SHA and file hashes.
2. **Derive clean NF4** checkpoint (`load_in_4bit`, nf4, blocksize 64, double
   quant) → `/workspace/.../ckpt/<M>_clean/`; record sha256.
3. **Pin tensors**: layer-0 `self_attn.q_proj.weight` primary + six-tensor matrix
   by the same depth × projection rule, using M's actual naming.
4. **Track A run**:
   - A2/A3: independent-process extraction, cycles 0–3, sizes 1k/5k/10k/20k/50k,
     seeds 20261016/017/018, six tensors, baselines `pair_parity`/`nearest_parity`.
   - A1: integrity JSON (only intended tensor differs).
   - A4: detectability — 16 features, grouped splits, clean control, CI, AUC/TPR
     thresholds.
   - A5: utility — wikitext-2, own tokenizer, 512 ctx, paired PPL, ±0.5% margin.
5. **Record** manifest, logs, resource usage, regression tests; per-model verdict.
6. **Gate check**: if any Track A gate fails → **stop**, investigate, report the
   counterexample, do not proceed to the next model until resolved.

Scripts will be **new per-model files** (e.g. `smollm2_qualification/`), reusing
the frozen codec unchanged. The StableLM scripts are not modified.

## Failure criteria (per model)

Fail if: BER ≠ 0 at cycle 0; checksum mismatch; non-idempotent artifacts;
endpoint codes rewritten; detectability above threshold beyond control noise; or
utility CI outside ±0.5% and informative enough to be a real difference (else
INCONCLUSIVE, reported as such). Failures retained and reported.

## Explicit non-claims

- Four more models expand evidence; they do **not** prove universal generalization.
- Expansion does **not** change Track B / Gate G (PARTIAL).
- Passing on more models is **not** a novelty result.
