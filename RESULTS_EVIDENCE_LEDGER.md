# Results Evidence Ledger — `interior_nearest` NF4 Anchor-Preservation

Run root: `/workspace/nes_run_20261016_anchor_ablation/`
Compiled: 2026-10-16. Method: frozen `interior_nearest` codec, sha256
`07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` (unchanged).

This ledger unifies the evidence for **one** method. It does **not** include the
five residual methods (sign, magnitude_aware, qae, lwe_grid_parity,
split_sign_parity) or the QSE/DCE packed-NF4 methods. Those are separate tracks
and their results are not merged here.

## Verification status of this ledger

Every numerical value below was re-read from the frozen result files and
manifests during compilation (task 4). No experiment was rerun. Verification
script output is recorded in `LEDGER_VERIFICATION.json`.

- Codec hash: **unchanged** (`07ce008d…`).
- A–D: `out/anchor_ablation_ad_v2/observations.jsonl`, 16 rows — values match report.
- Arm E: `out/anchor_arm_e_v1/observations.jsonl`, 4 rows — values match report.
- TinyLlama matrix: `out/tensor_matrix_v1/observations.jsonl`, 48 rows — 6/6 interior exact, 0/6 pair_parity.
- Qwen matrix: `out/qwen_matrix_v1/observations.jsonl`, 48 rows — 6/6 interior exact, 0/6 pair_parity.
- Capacity: `out/capacity_*/observations.jsonl`, 240 rows — 0 interior failures, 54 pair_parity failures.
- Detectability: `out/detectability_analysis_v1/detector_results.json` — values match report.
- Utility: `out/utility_v2/results.json`, `out/utility_qwen50k_v3/condition_*.json` — values match report.

## Evidence ledger

| # | Claim | Supporting experiment | Models / tensors / seeds | Measured result | Limitation |
|---|---|---|---|---|---|
| 1 | Embedding is **invertible after requantization** for `interior_nearest` | Capacity sweep (`nes_capacity_*`) | 2 models; 5 tensors; 3 seeds; 4 cycles | 240 rows; **0 interior_nearest failures**; BER 0; exact recovery at every cycle | Payload ≤ 50,000 bits; 2 checkpoints |
| 2 | The method is **byte-idempotent** under repeated requantization | Capacity + tensor matrices | 2 models; 11 tensor-instances; cycles 0–3 | One distinct `output_packed_sha256` across cycles 0–3 for every interior condition | 3 cycles tested; not an unbounded stability proof |
| 3 | Robustness is **not** a saturation artifact | Capacity sweep | 2 models; 5 sizes 1k–50k | Carrier utilization 0.03%–1.2% | Larger payloads untested |
| 4 | The **anchor mechanism** is stable: no block endpoint is ever rewritten | Tensor + Qwen matrices, cycle 3 | 2 models; 12 tensor-instances | `changed_anchor_codes = 0` (sum 0/0); `blocks_without_endpoint = 0` | 10k bits, seed 20261016 |
| 5 | **Carrier positions are stable** across requantization | Matrices + capacity | 2 models; 12 tensor-instances | `carrier_position_agreement = 1.0000` at every cycle | — |
| 6 | `interior_nearest` **generalizes across tensor types/depths** | TinyLlama 6-tensor matrix | 1 model; 6 tensors (q/k/v, up/down/gate; early/mid/late) | **6/6 exact**, BER 0, anchors changed 0 | One model in this cell; payload 10k |
| 7 | Result **replicates on a second model** | Qwen 6-tensor matrix | Qwen2.5-3B; 6 tensors | **6/6 exact**, BER 0, anchors changed 0 | Qwen revision UNVERIFIED (see #13) |
| 8 | The **baseline parity op fails** (method-specific, not tensor/model-specific) | Matrices + capacity | 2 models; 12 tensor-instances; 3 seeds | pair_parity **0/6** per model; BER 0.0025–0.0070; 54/54 capacity failures | Baseline only; not a claim about all possible parity schemes |
| 9 | The **operation**, not the eligibility mask, causes robustness | A–E causal ablation (5 arms) | 1 model; L0 q_proj; 10k; 4 cycles | A/C pass (BER 0); B/D fail (BER 0.0038/0.0040); **E fails (BER 0.4913)** with interior eligibility but baseline op | A–E and B–D confound endpoints+op; Arm E resolves op vs mask, but a pure "endpoints-only" control is absent |
| 10 | Anchorless-block exclusion is **neither necessary nor sufficient** | A–E ablation (arms C and E) | 1 model | C passes without the exclusion; E fails despite interior eligibility | — |
| 11 | **Payload capacity** is bounded by interior-code count | Capacity sweep | 2 models; 5 tensors | Max feasible ≈ eligible codes − 128 bits; TinyLlama q_proj ≈ 4.09 M bits; Qwen q_proj ≈ 4.05 M bits; smallest tested ≈ 0.51 M bits | Analytic capacity, not a measured ceiling; >50k untested |
| 12 | **Grouped** steganalysis detectors are at chance on the large-tensor setting | Detectability (`detectability_analysis_v1`) | 2 models; 12 groups; 84 artifacts; 86,016 examples | Grouped-CV AUC ≈ 0.50; grouped leave-one-tensor-out oracle AUC 0.416; ungrouped 0.697 shown to be leakage | Small-cap tensors excluded here |
| 13 | On **small-capacity tensors** (confound removed) a small signature exists | Small-cap supplement | 2 models; 4 groups; 32,768 examples | LR AUC **0.534** (CI 0.533–0.536); permutation **p = 0.005**; max \|d\| = 0.089; clean-vs-clean control AUC 0.486 | Only **4 groups**; ~0.53 AUC, TPR@FPR1% ≈ 0.014 — weak, near-threshold |
| 14 | Embedding **preserves full-model perplexity** within measured scope | Utility/PPL | 2 models; L0 q_proj; 2 sizes; 3 seeds | All Δ within ±0.06%; **all 12 paired 95% CIs include 0** | 15,330 tokens/condition (target 100k not feasible); one tensor/layer; CIs cannot resolve <0.5% |
| 15 | Embedded checkpoints **load and run**; only the intended tensor differs | Utility/PPL | 2 models; 12 embedded checkpoints | 12/12 finite logits+loss; 12/12 exact recovery; 12/12 `differing_keys == [q_proj]` | — |
| 16 | Historical TinyLlama evidence is **reproduced byte-for-byte** | A–D cross-check | 1 model; L0 q_proj; 10k | `783d43d8…` (interior), `2ee7ba36…`/`4090675f…` (pair_parity) reproduced from a clean rebuild | Confirms one tensor only |

## Provenance anchors (shared across experiments)

| Item | Value |
|---|---|
| TinyLlama revision | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` |
| TinyLlama L0 q_proj source tensor sha256 | `283aa75d1f34daff…` |
| TinyLlama clean NF4 checkpoint sha256 | `f84d0a8ec6e011d683ae7d1aec961d4bb021b254da9d84516185df428f3b37a7` |
| Qwen revision | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` (current `main`; **UNVERIFIED** as historical) |
| Qwen L0 q_proj source tensor sha256 | `ce26ecb75d4ce505…` |
| Codec revision | `nf4_anchor_channel.py` sha256 `07ce008d…` (frozen, unchanged) |

## Conclusions

### A. Directly demonstrated
- Invertibility of `interior_nearest` after requantization, at payloads
  1,000–50,000 bits, on the tested tensors of two models, across 3 seeds and
  3 requantization cycles (claims 1–5).
- Cross-tensor and cross-model generalization within the tested 6-tensor sets of
  TinyLlama and Qwen2.5-3B (claims 6–7).
- The nearest-same-parity interior operation is the operative factor for
  robustness; the baseline parity operation fails (claims 8–10).
- Capacity is bounded by interior-code count, with analytic headroom ≥ 0.51 M
  bits on the smallest tested tensor (claim 11).
- Grouped steganalysis is at chance on the large-tensor setting (claim 12).
- Full-model perplexity is preserved within ±0.06% with all paired CIs
  including zero, on one tensor of two models (claims 14–15).

### B. Supported but exploratory
- A **small** detectability signature on small-capacity tensors (claim 13):
  AUC ≈ 0.53, p = 0.005, but only 4 groups and negligible practical detection
  power. Treat as a hypothesis-generating signal, not an established detector.
- Cross-architecture generalization: demonstrated on 2 models only; Qwen
  revision provenance is UNVERIFIED (claim 7).

### C. Not established — must NOT be claimed
- **Universal robustness.** Only 2 checkpoints, 3 cycles, payloads ≤ 50,000
  bits, and a specific NF4 blocksize-64/nested configuration were tested.
- **Undetectability.** The large-tensor chance result reflects sparsity, not
  provable undetectability; the small-capacity supplement shows a real (if weak)
  signal. Detector families beyond the 16-feature, 3-detector set were not
  tested.
- **Semantic equivalence / unchanged generation quality.** Perplexity on one
  tensor does not establish semantic or generative equivalence.
- **General utility preservation across all models or tensors.** Utility was
  measured on one tensor per model, 15,330 tokens, two models.
- **Absolute novelty.** No prior-art / literature review has been performed in
  this workspace (see `MANUSCRIPT_READINESS.md`); the novelty claim cannot be
  defended from the current evidence.
- **A measured capacity ceiling.** Only the analytic bound and sizes ≤ 50,000
  bits were tested.

---

## Addendum - Third-model (StableLM-2-1.6B) qualification evidence

Compiled 2026-10-18 from `stablelm_qualification/`. Codec sha256 `07ce008d...`
(verified unchanged). Track 3 only. These rows extend the ledger; they do not
replace or reinterpret prior rows. The small-capacity signal (row 13) is
preserved and unchanged.

| # | Claim | Supporting experiment | Models / tensors / seeds | Measured result | Limitation |
|---|---|---|---|---|---|
| 17 | `interior_nearest` is invertible on a third architecture with independent-process extraction | `stablelm_qualification/roundtrip_stream_v1/` | StableLM-2-1.6B; 6 tensors; 5 sizes (1k-50k); 3 seeds; cycles 0-3 | 360/360 exact, BER 0, checksum ok, carrier agreement 1.0; 0 embed failures | One model; <=50k bits; <=3 cycles in matrix |
| 18 | Byte-idempotence holds on StableLM | same | 90 interior conditions | 1 distinct artifact sha256 per condition across cycles 0-3 | 3 cycles tested |
| 19 | The anchor mechanism holds on StableLM | same | 6 tensors x 5 sizes x 3 seeds | `changed_anchor_codes = 0`; `blocks_without_endpoint = 0`; `anchor_codes_delta = 0` | - |
| 20 | The baseline parity op fails on StableLM (as on 2 prior models) | same | 6 tensors; 3 seeds | `pair_parity` 93/360, `nearest_parity` 93/360; fail from cycle 1 (`Envelope checksum mismatch`); anchors changed 88,297 / 88,150 | Baseline only |
| 21 | Detectability screening passes on StableLM | `stablelm_qualification/detectability_v1/` | StableLM; 6 tensor groups; 86,016 block rows | GB AUC 0.5005, LR 0.5009, MLP 0.5000; TPR@1% ~ 0.010; permutation p 0.32-0.48; clean-vs-clean 0.4848; max abs d 0.009 | Only 6 independent groups; screening pass != undetectability |
| 22 | Embedded checkpoints load and run; only the intended tensor differs | `stablelm_qualification/integrity_10k.json` | StableLM; L0 q_proj; 10k; seed 20261016 | 1180 tensors; `differing_keys == [q_proj]`; finite logits+loss; exact recovery | One condition |
| 23 | Historical StableLM results are reproduced byte-for-byte | this addendum vs `THIRD_MODEL_VALIDATION.md` | StableLM; L0 q_proj; 10k; seed 20261016 | stego sha256 `658877c285bc2d8f...` identical to the historical report | One tensor |

Row 22's stego tensor sha256 `658877c285bc2d8f03f4178fa8f51753b146edcbe7e7267507b9947d53fa47e7`
matches `THIRD_MODEL_VALIDATION.md` section 4 (`658877c285bc...`), independently
reproducing the historical claim.

### Addendum 2 - Gate E utility equivalence (StableLM)

Compiled 2026-10-19. Source: `stablelm_qualification/utility_v2/`
(`nll/*.json`, `manifest.json`, `gate_e_reanalysis.json`). Codec sha256
`07ce008d...` re-verified unchanged. Model revision
`f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef`; q_proj; wikitext-2-raw-v1 test;
context 512; 580 windows (296,380 tokens/condition); clean PPL 12.977962.

| # | Claim | Evidence | Scope | Result | Limitations |
|---|---|---|---|---|---|
| 24 | Embedding `interior_nearest` preserves wikitext-2 PPL within ±0.5% | `stablelm_qualification/utility_v2/gate_e_reanalysis.json` | StableLM; L0 q_proj; sizes 10k/50k; 3 seeds; 6 conditions | 6/6 paired 95% CIs entirely inside ±0.5%; max \|Δ\| 0.0139%; max CI half-width 0.037% | One model; one tensor; single corpus; screening-equivalence, not a no-effect proof |
| 25 | Gate E result is stable across payload size and seed | same | 10k vs 50k; seeds 20261016/017/018 | Δ rel PPL 0.0117-0.0139% for all embedded conditions | Seed count 3 |
| 26 | Gate E survived repeated runtime restarts without evidence loss | `stablelm_qualification/utility_supervisor.log`; `utility_v2/nll/*.partial.json` | 4 restarts during the run | All 7 conditions completed; 0 completed conditions lost; resume from per-batch checkpoints | Process-level resilience only |
| 27 | The embedding operation is a **named published technique** (novelty narrowed) | `overnight_third_model_prior_art/PRIOR_ART_EVIDENCE_GATE_G_20261019.md`; `PRIOR_ART_REVIEW.md` Addendum C | Literature (NID Liu 2016; CNV-QIM 2008; DeepWatermark 2020; HPQIM; Li 2023; FreMark 2025) | Code-index parity partition + neighbor-of-opposite-subcodebook move = NID; invariant-region framing established in image/speech/NN domains | Gate G remains PARTIAL; not a novelty claim; FreMark full text still paywalled |

Gate E verdict: **PASS** (`all_within_margin: true`). Remaining open item is Gate G
(novelty/prior-art), an unresolved risk documented in §G of the qualification report.
Per protocol §3, expansion remains **locked** while G is not PASS.

