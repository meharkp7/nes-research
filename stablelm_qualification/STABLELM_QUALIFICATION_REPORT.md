# StableLM-2-1.6B Qualification Report (Phase 6 — Go/No-Go)

Method: frozen `interior_nearest` NF4 anchor-preserving codec, sha256
`07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` (verified unchanged).
Track 3 only. Not pooled with the five residual methods or the QSE/DCE track.
Protocol frozen before outcomes in `QUALIFICATION_PROTOCOL_STABLELM.md`.

Run root: `/workspace/nes_run_20261016_anchor_ablation/stablelm_qualification/`

## Executive summary

| Gate | Criterion | Status |
|---|---|---|
| A — Implementation/artifact integrity | codec hash, hashes recorded, NF4 config, only intended tensor changes | **PASS** |
| B — Exact end-to-end recovery (fresh process) | BER 0, checksum ok, carrier agreement 1 | **PASS** (360/360) |
| C — Multi-sample/transformation robustness | cycles 0–3 × 5 sizes × 3 seeds × 6 tensors | **PASS** (360/360) |
| D — Detectability screening | AUC ≤ 0.55, TPR@1% ≤ 0.02, clean-control null | **PASS** (AUC ≈ 0.500) |
| E — Utility ±0.5% equivalence | paired 95% CI inside ±0.5% rel. PPL | **PASS** (6/6) |
| F — Reproducibility/reporting | rerunnable, hashed, failures recorded | **PASS** (see §F) |
| G — Novelty/prior-art integrity | closest prior art compared | **PARTIAL / UNRESOLVED RISK** (see §G) |

**Expansion decision:** **NOT authorized.** Per the frozen protocol (§3), expansion
requires **every** gate A–G to pass. Gate E is now **PASS**; Gate G remains
**PARTIAL / UNRESOLVED RISK**. Expansion therefore stays locked until G is closed
(or the protocol is amended with the novelty risk disclosed as a limitation).
This is a blocked state, not a pass.

## Gate A — Implementation and artifact integrity — PASS

| Item | Value |
|---|---|
| Codec sha256 | `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` (matches) |
| Model / revision | `stabilityai/stablelm-2-1_6b` @ `f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef` (pinned in `download_ckpt.py`) |
| FP16 source sha256 | `8bdf317e2b35ab5c8009cbb6c7ce495e4e608a6b9b843d44054edf25b8c5860d` |
| Clean NF4 sha256 | `bff3b65ebb686092b7234ddef66d3a69b762299826ac3e29e2015343c432d591` |
| NF4 config | blocksize 64, nested statistics True, quant_type nf4 |
| Only intended tensor differs | **Yes** — `differing_keys == [model.layers.0.self_attn.q_proj.weight]` (1 of 1180) |
| Finite logits+loss | Yes |
| Evidence | `integrity_10k.json` |

## Gate B — Exact end-to-end recovery — PASS

Fresh-subprocess extraction per condition (`exp_stablelm_roundtrip_stream.py`):
process 1 embeds + requantizes + writes the artifact to disk; process 2 (separate
invocation) loads only the file and extracts with `(artifact, key, tensor, method)`.

| Metric | Result |
|---|---|
| Exact recoveries (`interior_nearest`) | **360/360** |
| length_ok / checksum_ok | 360/360 / 360/360 |
| BER | 0.0 everywhere |
| carrier_agreement | 1.0 everywhere |
| decode_error | none |

## Gate C — Multi-sample/transformation robustness — PASS

| Method | Exact cycle-extractions | Anchor codes changed | Carrier agreement |
|---|---|---|---|
| `interior_nearest` | **360/360** | **0** | 1.0 |
| `pair_parity` | 93/360 | 88,297 | 1.0 |
| `nearest_parity` | 93/360 | 88,150 | 1.0 |

- By cycle (interior): c0 90/90, c1 90/90, c2 90/90, c3 90/90.
- By size (interior): 1k/5k/10k/20k/50k each 72/72.
- By seed (interior): each of 3 seeds 120/120.
- By tensor (interior): each of 6 tensors 60/60.
- Byte-idempotent: 1 distinct artifact sha256 per condition across cycles 0–3.
- `blocks_without_endpoint = 0` everywhere; `anchor_codes_delta = 0` everywhere.
- Baselines fail from cycle 1 with `ValueError: Envelope checksum mismatch`
  (semantics unmodified).
- Cycles 0–5 stress covered by regression test `requantize_cycles_0_5_exact`.
- Evidence: `roundtrip_stream_v1/` (270 records, `summary.json`).

## Gate D — Detectability screening — PASS

`detectability_v1/`, 86,016 block rows, **6 independent tensor groups**.

| Detector | AUC | CI95 (grouped bootstrap) | TPR@1% | TPR@5% | Perm. p |
|---|---|---|---|---|---|
| Gradient Boosting | 0.5005 | [0.4997, 0.5017] | 0.0100 | 0.0500 | 0.463 |
| Logistic Regression | 0.5009 | [0.5006, 0.5016] | 0.0100 | 0.0502 | 0.318 |
| MLP (exploratory) | 0.5000 | [0.4996, 0.5004] | 0.0097 | 0.0500 | 0.483 |

- Clean-vs-clean control: AUC 0.4848 [0.4444, 0.5106] — null-consistent.
- Per-tensor GB AUC 0.4991–0.5032.
- Max |Cohen's d| per tensor 0.002–0.009 (negligible).
- Blocks touched per artifact: 2.7%–32.1% (`block_sparsity.json`).
- Thresholds met (AUC ≤ 0.55, TPR@1% ≤ 0.02).
- **Limitations:** only 6 independent groups; no seed-vs-seed control in this
  dataset; verdict is a screening pass, not proof of undetectability.
- Evidence: `DETECTABILITY_STABLELM.md`, `detectability_v1/`.

## Gate E — Utility ±0.5% equivalence — PASS

Run: `utility_v2/` (resumable, per-batch checkpointing; supervisor-recovered across
4 runtime restarts with no completed condition lost). Clean + 6 embedded conditions
(10k/50k × seeds 20261016/017/018), `interior_nearest`, q_proj,
wikitext-2-raw-v1 test, own tokenizer, context 512, 580 windows
(296,380 evaluated tokens per condition).
Predeclared margin ±0.5% relative PPL; PASS requires the paired 95% CI inside it.
Independent re-analysis: `reanalyze_utility.py` → `utility_v2/gate_e_reanalysis.json`
(`all_within_margin: true`, `n_within_margin: 6/6`).
Manifest: `utility_v2/manifest.json` — codec sha256
`07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` (matches frozen),
model revision `f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef`.

| Condition | Embedded PPL | Δ rel PPL | Paired 95% CI (rel %) | Within ±0.5% |
|---|---|---|---|---|
| clean (baseline) | 12.977962 | — | — | — |
| emb_10000_20261016 | 12.977800 | −0.0012% | [−0.0237, +0.0206] | **yes** |
| emb_10000_20261017 | 12.979483 | +0.0117% | [−0.0100, +0.0328] | **yes** |
| emb_10000_20261018 | 12.979700 | +0.0134% | [−0.0089, +0.0355] | **yes** |
| emb_50000_20261016 | 12.979503 | +0.0119% | [−0.0076, +0.0316] | **yes** |
| emb_50000_20261017 | 12.979481 | +0.0117% | [−0.0107, +0.0336] | **yes** |
| emb_50000_20261018 | 12.979769 | +0.0139% | [−0.0093, +0.0366] | **yes** |

Verdict: **PASS.** All six embedded conditions have paired 95% CIs entirely inside
±0.5%; the largest CI half-width is +0.037%, ~13× tighter than the margin. Effect
sizes are ≤0.014% and stable across payload size and seed. Embedding `interior_nearest`
into `q_proj` at 10k–50k anchors leaves wikitext-2 test PPL statistically
indistinguishable from clean within the predeclared equivalence margin.
Per-condition stego tensor sha256 values are recorded in `gate_e_reanalysis.json`.

## Gate F — Reproducibility and reporting — PASS

- Regression tests: `test_stablelm_codec_regressions.py` **11/11 PASS** (exit 0):
  deterministic_fixed_seed, endpoints_unchanged, interior_only,
  carrier_eligibility_stable, fresh_process_extraction, requantize_cycles_0_5_exact,
  corrupted_artifact_handled, truncated_artifact_handled, wrong_key_not_exact,
  over_capacity_rejected, pair_parity_fails_after_requant.
- Every run is resumable and writes unique run IDs; no completed run overwritten.
- Deviations recorded: an earlier non-streaming run was aborted for disk space
  (`roundtrip_v1_ABORTED_DISK/`, records preserved); a utility attempt was killed
  by an environment restart before writing results (no evidence lost) and
  relaunched as `utility_v2` with checkpointing.

## Gate G — Novelty/prior-art integrity — PARTIAL (unresolved risk, narrowed)

Full method sections read for the closest matches; see `PRIOR_ART_REVIEW.md`
Addenda A–C. The parity-index embedding step is QIM (Chen & Wornell 2001), already
applied to DNN weights by R-QIM (arXiv 2305.17879) and DeepWatermark (2020).

**Addendum C (2026-10-19) narrows the novelty further.** The specific operation
of partitioning *codebook indices* into parity/label sub-codebooks and moving a
carrier to the **neighbor index of the opposite sub-codebook** is a named,
published technique: **NID (Neighbor-Index-Division)**, Liu et al., *J. Ambient
Intell. Humaniz. Comput.* 7(1):139–147, 2016 (and CNV-QIM, Xiao et al., GLOBECOM
2008). This is the same operation as `interior_nearest` on code-index carriers.
Separately, the "keep the value inside the quantization-invariant region"
framing is established by Li et al. 2023 (int8 weights), FreMark 2025 (DCT), and
HPQIM (histogram-preserving QIM, image domain). Evidence log with verbatim quotes:
`overnight_third_model_prior_art/PRIOR_ART_EVIDENCE_GATE_G_20261019.md`.

What remains unmatched in the reviewed scope: (a) NF4 **packed** code indices
(blocksize 64, nested) as the carrier domain; (b) explicit endpoint (0/15)
exclusion as the robustness mechanism; (c) exact byte-idempotent recovery over
repeated dequantize/requantize cycles as the stated target; (d) detectability
screening + paired-PPL equivalence reporting. These are a **narrow NF4-specific
formulation + empirical robustness result**, not a new embedding principle. A
reviewer can reasonably call the method "NID/QIM on NF4 packed codes with
endpoint exclusion." **No "first"/"unprecedented" claim is made.**

FreMark (DSPP 2025) full text remains paywalled (abstract-level only). Gate G is
**PARTIAL / not PASS**.

## Expansion authorization

Per protocol, expansion to 3–4 additional architectures is authorized only if
**all** gates pass. Current status: A, B, C, D, E, F **pass**; **G is an
unresolved novelty risk** (documented, not silently favorable). **Expansion is
NOT authorized.** Gate E PASS closes the utility blocker; Gate G remains the sole
blocker. The smallest targeted experiment to close G is a full-text read of the
closest prior art (FreMark, DSPP 2025) to determine whether the
combination differentiator is genuinely unclaimed, or whether it is subsumed by
QIM-over-quantized-weights prior art.

## Provenance

| Item | Value |
|---|---|
| Run root | `/workspace/nes_run_20261016_anchor_ablation/stablelm_qualification/` |
| Repo | `meharkp7/nes-research`, branch `main`, commit `aa7e1a7`, clean |
| Codec | `07ce008d…` unchanged |
