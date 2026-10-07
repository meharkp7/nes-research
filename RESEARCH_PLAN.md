# NES Research & Productization Plan

Continuation document. Every claim below carries one of five statuses, defined in
§2. Nothing is written as a finding unless it has one.

Read `RESEARCH_LOG.md` first for how we got here and which mistakes to not repeat.

**Status vocabulary** (§2 gives the full ledger):

| status | means |
|---|---|
| **MEASURED** | a number produced by this repo, sitting in a `results/*.json` artifact |
| **IMPLEMENTED** | code exists and is exercised by the test suite, but no experimental result depends on it |
| **READ** | established by reading source, not by running anything |
| **NOT_RUN** | deliberately not attempted; carries no evidence either way |
| **FAIL** | measured, and the gate did not pass. Never rewritten as PASS |

---

## 0. Where we actually are

### Measured

| claim | evidence | file |
|---|---|---|
| Sign embedding round-trips at BER 0 | 48,256 bits compared, 0 errors | `results/exp3_*.json` |
| Sign capacity ≥ 10M bits at BER 0 | 4 sizes, all BER 0 | `results/exp4_*.json` |
| Sign robustness under both noise gates | σ=0.001 ≤ 0.0012, σ=0.002 ≤ 0.0108; BER 0.0 on **6 of 7** models | `results/exp6_*.json` |
| Sign fidelity 0.0053% PPL delta | 3-way protocol, Qwen3B only | `results/exp5_*.json` |
| Sign is detected at 70.5% | carrier-centered MLP | `results/exp7_neural_*.json` |
| Sign detection is structural | α×100, γ×5, payload×10 → 69.4–74.4% | `results/exp7_neural_parameter_study.json` |
| LWE grid encoding is extractable without the cover | BER 0.5036 → 0.0000 (`extract_needs_cover: false`) | `results/exp10_*.json` |
| LWE passes both gates on 5/5 measured models | BER 0.0000, detector 50.00%; TinyLlama SKIPPED (incomplete cache) | `results/exp12_lwe_cross_model.json` |
| **LWE channel is not key-gated (W4.2)** | keyless read of all 10,256 bits at **BER 0.0**, positions at precision/recall 1.0; grid width is one public constant → gate **FAIL** | `results/exp13_keyless_recovery.json` |
| **Neural detectability is carrier-conditioned (W3.2)** | blind adversary **50.0%** (constant predictor, 4/576 positions held a carrier) vs carrier-centered control **68.75%** → gate **PASS**; exp7's 70.5% is the handed-locations number | `results/exp14_blind_patch_detector.json` |
| **LWE fidelity measured on 2 models (W2)** | embedding-specific PPL Δ **+0.0077%** (Qwen2.5-3B) and **+0.0501%** (gemma-2-2b) vs 2% gate → both **PASS**; exp5's own baseline/control reproduce exactly (12.4707 / 11.3494) | `results/exp15_lwe_fidelity_*.json` |
| **Cross-scheme transfer measured (W3.1)** | sign-trained detector: **62.85%** within-scheme → **50.00%** on LWE (control-validated: no transfer); reverse direction uninformative — LWE-trained control collapsed to 50.00%, reproducing exp12's detector 50.00% → gate **PASS**, `controls_valid: false`, conclusion scoped directional | `results/exp16_cross_scheme_detector.json` |
| **QAE adapter wired, round trip run (W1.1)** | `qae` **READY** via `QaeDictAdapter`: **BER 0.0 over 48,256 bits** through exp3's production path, no-cover probe usable → gate **PASS**; `nf4_qae` registered **BLOCKED** (reference needs weight tensors the embed contract doesn't carry — diagnosed, probe error recorded) | `results/exp17_qae_round_trip.json` |
| **Strategy × model matrix (W1.3), full 7-model grid** | 4 READY strategies × 7 models: **28/28 round trips BER 0.0, 28/28 robustness gates pass — only detectability separates**; LWE detector **0.50 on all seven models** (7/7 wins — every family, every size class), magnitude_aware takes the matrix's only other win (Mistral-7B 0.525), sign 0.56–0.78 / magnitude_aware 0.53–0.78 / qae 0.56–0.84; `neural` + `nf4_qae` excluded by name; TinyLlama SKIPPED (incomplete cache) | `results/exp18_matrix_*.json` |
| **Adaptive routing as designed (W5.1)** | three models → **three different branches** (gemma σ=0.000448→lwe, Qwen σ=0.001554→neural, Llama σ=0.007736→sign); every round trip that ran holds BER 0.0; Qwen's neural route fails design-as-written (no trained model) and is **recorded as the design's own**, both available branches round-trip 0.0 as its fallback | `results/exp19_adaptive_*.json` |
| **Sign/parity split dial measured (W5.3)** | five parity shares × exp10's three axes: all round trips **BER 0.0**; BER@σ0.002 rises **0 → 0.0127** and detector falls **0.7875 → 0.50** with parity share — the stealth/robustness trade-off is a dial; pure parity **reproduces exp18's lwe cell exactly**, only it wins all gates | `results/exp20_split_dial_*.json` |
| **QAE encode + LWE read-out (W5.2)** | plan called it plausible; measured **0.5433 vs the 0.0 gate → FAIL (the finding)**, matched control **0.0**; the public correction returns **0.0** — `parity(v) = sign(v) ⊕ cell-parity(|v|)` measured: the "hybrid" is sign reading plus a public relabeling | `results/exp21_qae_lwe_*.json` |
| **Per-layer LWE grid width (W5.4)** | three width rules × exp10's axes: magnitude-keyed `per_layer` round-trips **0.0** but **fails robustness (0.0226 / 0.5763 vs gates 0.02 / 0.10)** — cause measured: the extractor sizes its grid from the *noisy* tensor, `√(std²+σ²)` moves **36/36 layer buckets** at every σ; rank-keyed `layer_rank` passes every gate (**0.0015 / 0.0736**); detector **0.50 on all three** (width-blind) and the ladder's sub-default widths cost robustness (σ0.002 0.0127→0.0736) → **global 0.010 stays the best point**; control = exp18's lwe cell bit-for-bit | `results/exp22_layer_widths_*.json` |
| **NES round-trips at BER 0.0 through three 4-bit formats: NF4, GPTQ, AWQ** | GPTQ corr 0.9903 / AWQ corr 0.9941 (raw 0.9890) | `results/exp9_formats.json` |
| 9 model ids covered, 0 ERROR | 35 PASS / 6 FAIL / **0 NOT_RUN** | `results/experiment_manifest.json` |

**How those rows are kept honest.** `nes-llm/claim_audit.py` re-derives every
MEASURED row above from disk: **127/127 pass**, with `check_consistency.py` at
9/9 and a 66-test suite running green. Four claims in this document failed
that audit while it was being built — a coverage figure with no artifact
behind it, an average reported as a maximum, a count lifted from a different
measurement, and a grid count wearing a suite's name. All four are closed,
none by changing a result: exp7_neural still fails at 70.5%, exp2 still fails
on 4 models, and every threshold is asserted unchanged by the audit itself.
Full reasoning in `RESEARCH_LOG.md` §17–§18; the corrections sit in §1 below.

**Two of those rows need the scope stated, or they read as more than they are.**

**"Three 4-bit formats" (Exp9).** This is the defensible form of *"NES works
beyond NF4"*. What it says: a payload round-trips at BER 0.0 through three
4-bit checkpoint formats, each read by a format-specific dequantizer and each
verified against its own FP16/bf16 reference *before* any residual is
computed. What it does **not** say: one model (Qwen2.5-3B-Instruct), one
payload size (10,256 bits), one module type (`mlp.down_proj`), and a **clean**
channel — no noise, no patch, no adversary. Robustness and detectability were
measured for NF4 sign embedding only, never for GPTQ or AWQ. AWQ used 35 of 36
layers; layer 2 failed verification and was excluded by name.

**"9 model ids covered, 0 ERROR".** The 7-model grid (`exp1`–`exp7_neural`)
is **NF4 only**: `bitsandbytes`, `nf4`, group 64. GPTQ and AWQ are the two
remaining cells, on separate checkpoints, with separate dequantizers — they
are not NF4 results and no NF4 cell was reused to produce them. The older
phrasing, *"7 models covered, 0 errors"*, conflated the two and is retired.

### Not established — and load-bearing

1. **LWE perplexity is unmeasured on any model.** Only sign's was measured.
   *(Closed by exp15: LWE embedding-specific PPL Δ is +0.0077% on
   Qwen2.5-3B and +0.0501% on gemma-2-2b against the 2% gate — both
   PASS, with exp5's baseline/control reproducing exactly.)*
2. **Every detectability number uses a detector trained against the same scheme it
   tests.** Cross-scheme is untested. *(Measured by exp16, one direction: a
   sign-trained detector scores 62.85% within-scheme and **50.00% on LWE** —
   no transfer, control validated. The reverse is **not** established: the
   LWE-trained control collapsed to 50.00%, reproducing exp12's measured
   50.00% — LWE is not learnable by this detector even in its own scheme.
   `controls_valid: false` in the artifact forbids the both-directions
   reading.)*
3. **Clean and stego patches are both cut at carrier positions.** The detector is
   handed the carrier locations. *(Partly resolved by exp14: a blind adversary
   given neither class at carrier positions scores **50.0%** — a constant
   predictor — while the carrier-centered control on the same embeddings scores
   68.75%. The 70.5% FAIL stands, and is now known to be placement-conditioned.)*
4. **The 55% detectability gate is self-chosen.** It is not a security property.
5. **LWE is LWE-*inspired*.** Key-derived grid and parity encoding; no lattice, no
   matrix A, no SIS/LWE instance. The post-quantum claim in `lwe_strategy.py` is
   unsupported.
6. **exp2's threshold fails on 4 of 7 models** (Qwen2.5-3B, Qwen2.5-7B,
   TinyLlama, Llama-3.1-8B; gemma-2-9b, Phi-3 and Mistral-7B pass). The calibration
   shows the threshold tracks the quantization format — NF4 puts 0% of probed layers
   above it on Qwen2.5-3B where FP4 puts 100% — but three models clear it under NF4,
   so format is not the whole story. Unresolved and an author decision.
7. **GPTQ and AWQ have no robustness or detectability numbers.** Exp9 measured a
   clean channel only.

---

## 1. Final claim audit

Every claim the programme currently makes, and what backs it. Re-run this after
any session that touches a result.

```bash
cd nes-llm
../.venv/bin/python run_nes_experiments.py --audit    # manifest matrix
../.venv/bin/python check_consistency.py              # 9/9 must pass
../.venv/bin/python claim_audit.py                    # 127/127 must pass
```

`claim_audit.py` is this section made executable: each row is re-derived from
the artifact it names — counts included, because every claim that failed this
audit failed on a count while the values underneath stayed correct — and it
exits non-zero on anything it cannot verify. It passes **127/127** as written.

Four claims in this document failed it. Three are corrected in the audit notes
below (exp12 coverage, exp6 robustness, exp2's count) and the fourth — the
retired *"7 models covered, 0 errors"* — in §0. Each is recorded rather than
silently edited: `RESEARCH_LOG.md` §17 holds the reasoning, §18 the session
record.

### MEASURED — number in `results/*.json`

| claim | number | artifact |
|---|---|---|
| Sign round trip | BER 0.0, 48,256 bits | `exp3_*.json` |
| Sign capacity | ≥ 10M bits at BER 0 | `exp4_*.json` |
| Sign robustness | every model under the 0.02 / 0.10 gates; BER 0.0 on 6 of 7 at σ=0.001 (Mistral-7B 0.00115) | `exp6_*.json` |
| Sign fidelity | +0.0053% PPL | `exp5_*.json` |
| Sign detectability (neural) | **70.5%** vs 55% gate → FAIL | `exp7_neural_*.json` |
| Sign detection structural | 69.4–74.4% across α×100, γ×5, payload×10 | `exp7_neural_parameter_study.json` |
| Statistical detection | below 55% gate | `exp7_*.json` |
| LWE extractable without cover | 0.5036 → 0.0000, `extract_needs_cover: false` | `exp10_*.json` |
| LWE both gates | BER 0.0, detector 50.00%, **5 of 5** measured models pass (TinyLlama skipped) | `exp12_lwe_cross_model.json` |
| **LWE keyless recovery (W4.2)** | phase attacker: precision 1.0, recall 1.0, **stream BER 0.0 over 10,256 bits**, no key/cover/params; clean control 0 candidates; shipped width = 0.010 on 36 layers × 6 keys, `keyed_branch_active: false` | `exp13_keyless_recovery.json` |
| **Blind-patch adversary (W3.2)** | blind accuracy **0.5000** (tp=0, fp=0 — learned nothing) vs carrier-centered control **0.6875**; 4/576 blind positions contained a carrier; 576 pairs/arm, split by embedding | `exp14_blind_patch_detector.json` |
| **LWE fidelity (W2)** | three-way PPL, exp5's protocol: LWE Δ vs control **+0.0077%** (Qwen2.5-3B), **+0.0501%** (gemma-2-2b); reconstruction alone −8.99% / −5.27%; sign re-run +0.0529% vs recorded −0.0053% (same verdict, fresh-AES-key digit) | `exp15_lwe_fidelity_*.json` |
| **Cross-scheme detector (W3.1)** | 432 pairs/scheme, split by embedding, carrier-centred both: sign→sign **0.6285** (control clears), **sign→LWE 0.5000** (no transfer), lwe→lwe **0.5000** (collapsed = exp12's 50.00%), lwe→sign **0.5000** (uninformative); gate PASS, `controls_valid: false` | `exp16_cross_scheme_detector.json` |
| **QAE round trip (W1.1)** | `qae` READY (dict adapter over the per-tensor ABC): **BER 0.0** / 48,256 bits, decrypt + match, no-cover probe usable; distortion 24,076 values changed, mean \|Δ\| 0.0355; `nf4_qae` BLOCKED with recorded diagnosis | `exp17_qae_round_trip.json` |
| **Strategy × model matrix (W1.3, full 7-model grid)** | exp10's three axes × 4 strategies × 7 models: round trip **28/28 BER 0.0**, robustness **28/28 pass** (σ0.001 all 0.0); detector — LWE **0.5000 ×7** (wins 7/7), sign 0.56–0.78, magnitude_aware 0.53–0.78 (**1/21 non-LWE win: Mistral-7B 0.525**), qae 0.56–0.84 (1/21 outside LWE overall); pipeline flag false for LWE (sign-only DecryptPipeline, known wiring gap) | `exp18_matrix_*.json` |
| **Adaptive routing as designed (W5.1)** | three models → **three different branches**: gemma σ **0.000448**→lwe, Qwen σ **0.001554**→neural (route fails design-as-written — no trained model — `EmbeddingError` recorded, forced sign/lwe fallbacks both **0.0**), Llama σ **0.007736**→sign; all four round trips that ran **BER 0.0**; each branch recomputes exactly from recorded σ + thresholds | `exp19_adaptive_*.json` |
| **Sign/parity split dial (W5.3)** | 5 fractions × Qwen2.5-3B: round trips **5/5 BER 0.0**, σ0.001 **0.0 everywhere**; detector **[0.7875, 0.6938, 0.70, 0.5875, 0.50]**, BER@σ0.002 **[0, 0.0034, 0.0067, 0.0095, 0.0127]** across parity share 0→1 (stealth↑ robustness↓); only pure parity wins; its detector **= exp18's lwe cell (delta 0.0)**, pure-sign endpoint +0.0375 from exp18's sign cell (fresh-key variance, mechanism verified bit-identical) | `exp20_split_dial_*.json` |
| **QAE encode + LWE read-out (W5.2)** | one embed, same stego, three readings: matched **0.0** (control), raw LWE parity **0.5433** (5,572/10,256 — gate 0.0, **verdict FAIL**), public cell-parity correction **0.0**; `parity(v)=sign(v)⊕cell-parity(|v|)` measured, not asserted; complement prediction missed (carriers all ≥ one grid width: sampled min 0.013 > 0.010) — miss recorded | `exp21_qae_lwe_*.json` |
| **Per-layer LWE grid width (W5.4)** | 3 rules × Qwen2.5-3B: round trips **3/3 BER 0.0**; **global** 0.0/0.0127, detector 0.50, wins; **per_layer** (magnitude-keyed) 0.0226/0.5763 → **FAILs both robustness gates** with the verified cause in-artifact (`noise_bucket_flips`: extractor's grid drifts under noise, 36/36 buckets move per σ); **layer_rank** (rank-keyed, agreement by construction) 0.0015/0.0736, **wins**; detector **0.50 ×3** (width-blind); deltas +0.0015/+0.0609 (rank) and +0.0226/+0.5636 (magnitude) vs global; control = exp18's lwe curve bit-for-bit, anchor delta 0.0 | `exp22_layer_widths_*.json` |
| **GPTQ round trip** | BER **0.0**, 10,256/10,256 bits, corr 0.9903, 36/36 layers | `exp9_formats.json` |
| **AWQ round trip** | BER **0.0**, 10,256/10,256 bits, corr 0.9941, **35/36 layers** | `exp9_formats.json` |
| Suite coverage | 35 PASS / 6 FAIL / 0 NOT_RUN / 0 ERROR | `experiment_manifest.json` |
| AWQ reconstruction perplexity | 33.59 → 35.25 (+5%); layer-2 `up_proj` zeroed → 40.65; reference `up_proj` → **inf** | `RESEARCH_LOG.md` §16 (log entry, not a `results/` artifact) |

**Audit note on the LWE cross-model row (raised by this audit, closed).**
The row was written up as *"5 of 5 measured models pass both gates"*,
with a six-row table in `RESEARCH_LOG.md` §7 and a commit message to
match. At audit time `results/exp12_lwe_cross_model.json` contained
**four entries, three measured**: TinyLlama SKIPPED, then Qwen2.5-7B,
Llama-3.1-8B and gemma-2-9b. Qwen2.5-3B and Mistral-7B were absent
entirely — not skipped, simply not in the file — and no archived version
of this artifact had ever contained them. So the 5-model figure had no
artifact behind it, while the file that existed said 3 of 3.

The claim was never contradicted: every model that was in the file
passed at BER 0.0000 / detector 50.00%, which is also what the 5-row
table reported. It was a coverage gap, not a wrong number. exp12 was
re-run across all six cached models with no `--models` filter; the
artifact now records **five measured, one skipped** with those same
numbers, and the row above says 5/5 because the file says so.

**Audit note on robustness (raised, closed).** This row and
two others read *"BER 0 at σ=0.001"*. It is 0.0 on six of seven models;
**Mistral-7B is 0.00115 at σ=0.001 and 0.01074 at σ=0.002**, both
inside the 0.02 / 0.10 gates, so every cell still reads PASS and no
threshold moved. Only the claim was wrong — an average presented as a
maximum — and the row above now says what the seven numbers actually
are.

**Audit note on exp2's count (raised, closed).** Both documents said
exp2 fails on **6 of 7** models, and `RESEARCH_LOG` §4.1 said *"1/7
pass, six fail"*. The manifest says **4 of 7 fail**: Qwen2.5-3B,
Qwen2.5-7B, TinyLlama and Llama-3.1-8B fail; gemma-2-9b, Phi-3-mini and
Mistral-7B pass. Neither older figure came from the cells.

Two figures existed because two measurements exist.
`exp2_criterion_calibration.json` is a *separate* run over the legacy
`residual_profile_*.json` files — the suite's models plus `gemma-2-2b`,
eight profiles — and the copy on disk predated the re-runs of gemma-2-9b
and Llama-3.1-8B, still carrying their pre-rerun values (0.000382 /
0.000879 against 0.003605 / 0.022642). Its own `finding` string also read
*"Zero of 7"* while its own `models_passing` read `1`, because "Zero" was
hard-coded rather than counted. The generator now counts and labels the
FP4 comparison as one model with five probed layers; the artifact was
regenerated and reports **3 of 8**, agreeing with exp2 on every shared
model — including the pair that exposed the staleness.

`python nes-llm/claim_audit.py` re-derives every MEASURED claim in this
section from `results/*.json` and exits non-zero on any it cannot
verify. All 127 checks pass at the time of writing. Run it before citing
any number here.

### FAIL — measured, gate did not pass, not rewritten

| cell | number | why it stands |
|---|---|---|
| `exp2` on 4 of 7 models | ≥ 80% of layers with mag_mean > 0.002 | threshold tracks quantization format; changing it is an author decision |
| `exp7_neural`, sign | 70.5% vs 55% | proved structural across a 100× α and 5× γ sweep |
| `exp8` cross-model | FAIL | correctly inherits the neural FAIL rather than averaging it away |
| AWQ `model.layers.2.mlp.down_proj` | corr 0.9337 | excluded from Exp9's residual set and named in `metrics.layers_excluded` |
| AWQ `model.layers.32` / `33` `self_attn.v_proj` | 0.9211 / 0.9147 | same; outside Exp9's module type entirely |
| `exp13` LWE keyless recovery | attacker BER **0.0** vs `min_keyless_ber` 0.5 | the strategy's own docstring claims were tested with falsification criteria attached; the message itself stays AES ciphertext |

### IMPLEMENTED — exists, tested, no result depends on it

| item | where | note |
|---|---|---|
| GPTQ + AWQ dequantizers | `src/quantization/adapters.py` | 20 tests in `tests/test_quantization_adapters.py` |
| `verify_dequantization` gate | `adapters.py` | includes the absorbed-scale correction and the NaN-safe comparison |
| `claim_audit.py` | `nes-llm/claim_audit.py` | 127 checks re-deriving every MEASURED claim — counts included — and exiting non-zero on any it cannot verify |
| Delta distribution W8.1/W8.2 (`src/delta/`, `nes delta-export/inspect/extract`) | `src/delta/format.py`, `recipient.py`, `src/cli.py` | integrity-verified delta file + recipient CLI; 14 tests in `tests/test_delta_distribution.py`; measured end-to-end on Qwen2.5-3B (`RESEARCH_LOG.md` §32); no result depends on it |
| `QuantizationStrategy`, `NF4QuantizationStrategy` | strategy registry | per-tensor ABC, needs an adapter; **never run** |
| Neural strategy `train_sampled()` | adaptive strategy | **never run** |
| `adaptive_strategy` noise routing | strategy registry | **never run** |

### READ — source, not behaviour

| claim | where it was read |
|---|---|
| AWQ nibble order is an even/odd interleave, unreachable by rotation | `awq/utils/packing_utils.py` (`AWQ_ORDER`, `AWQ_REVERSE_ORDER`) |
| AWQ folds a per-channel scale into the preceding LayerNorm | AutoAWQ source + measured agreement to ~1% (`RESEARCH_LOG.md` §16) |
| QAE is quantization-*aware* (knows NF4; sign and LWE do not) | `quantization_strategy.py` |
| Neural objective rewards mean/std match and penalises no sign leak | objective function |
| Clean and stego patches are both cut at carrier positions | `src/steganalysis/*` |
| Four embedders, two loaders, three `real_residual_embedder*` | `src/model/*` |
| The pre-fix Exp9 path applied an NF4 config to GPTQ ids | `src/model/exp9_alternative_quant.py` (removed) |

### NOT_RUN — no evidence either way

| item | why it matters |
|---|---|
| Strategy × model matrix — **measured, full 7-model grid (exp18)** | breadth deliverable; TinyLlama SKIPPED (incomplete cache), gemma-2-2b holds the small slot |
| Model surgery — **done, exp23 all 12 cells measured (legs run 2026-10-07): 8 survive at exp3's 0.0 — incl. the real 1,000-step fine-tune; NF4 0.3832 / half-merge 0.2476 / GPTQ 0.4956 / AWQ 0.4108 — every failure graceful (short of chance), gate untouched; no legs NOT_RUN** | determines viability — answered: light-touch surgery preserves the payload, int4 re-quantization does not |
| Pareto frontier — **measured (exp24): one point dominates the whole set** | the strongest publishable framing |
| Paper hardening (①②③) — **measured (exp25/26/27, all PASS): selection ablation over one digest-pinned allocation (H1/H2/H4 supported, H3 missed-and-kept), capacity curve 1k–50k flat 0.0 with detector 0.5028 at every size, threat boundary (wrong keys 0/10, partial access 0/6, public-rule attacker ≈ chance)** | the reviewer-proofing asked for before paper writing — all three answered with pre-registered gates; **NEXT: paper writing, not more implementation** |
| Robustness / detectability for GPTQ and AWQ | Exp9's channel was clean |

### NOT_RUN by design, and the gate that keeps it that way

The specific failure this programme guards against is a **plausible number
measuring nothing**. Two gates exist because of it:

1. **Dequantization verification** — a dequantizer must reach corr > 0.95 and
   residual ratio < 0.5 against the FP16 reference before any residual derived
   from it is used. Thresholds are in `experiment_registry.THRESHOLDS` and are
   never relaxed to make a cell pass. Control: the *wrong* nibble order fails
   **0/252** modules through the same corrected comparison, so the correction
   cannot paper over a layout bug.
2. **Detector sample floor** — ≥ 400 pairs. At 120, every variant scores
   exactly 50% and the study is void.

---

## 2. Phase A — finish the original NES experiment suite

### Workstream 0 — the original suite

The original guide's Exp1–Exp9, plus the manifest, the cross-model report and
this audit. **This phase is complete.**

| cell | status | note |
|---|---|---|
| exp1 residual extraction / QACI | PASS ×7 | NF4 grid |
| exp2 residual fingerprint | FAIL ×4, PASS ×3 | threshold tracks quantization format; unchanged (§1 audit note) |
| exp3 round trip | PASS ×7 | BER 0.0 |
| exp4 capacity | PASS | ≥ 10M bits at BER 0 |
| exp5 fidelity | PASS | +0.0053% PPL |
| exp6 robustness | PASS ×7 | under both gates; BER 0.0 on 6 of 7 models (Mistral-7B 0.00115) |
| exp7 statistical | PASS ×7 | below the 55% gate |
| exp7 neural | **FAIL** ×1 (Qwen2.5-3B) | 70.5% vs 55%; structural, not a fluke (100× α, 5× γ, 10× payload sweep) |
| exp8 cross-model | **FAIL** | correctly inherits the neural FAIL |
| exp9 GPTQ | **PASS** | BER 0.0, corr 0.9903, 36/36 layers |
| exp9 AWQ | **PASS** | BER 0.0, corr 0.9941, **35/36 layers** (layer 2 excluded) |

Nothing is left at NOT_RUN. The two remaining failures are recorded failures —
`check_consistency.py` asserts that `exp7_neural` stays FAIL and that no PASS
exists without an artifact behind it.

**What closing this phase cost**, in case it reads as a small item: the AWQ
dequantizer could not be verified for three sessions. The naive next step was
to install `gptqmodel`/`autoawq` and diff against their unpack. Reading the
unpack was enough, and the real obstacle turned out not to be the layout at
all — AWQ does not quantize the weight you think it does. Full reasoning in
`RESEARCH_LOG.md` §16.

---

## 3. Phase B — scientific gaps

Four workstreams, all of them holes in claims we have already made. Ordered by
how cheaply each could invalidate the rest.

### W1. Complete the strategy set

#### W1.1 Quantization-aware strategies (never run) — **done, exp17: `qae` PASS / `nf4_qae` BLOCKED**

`QuantizationStrategy` and `NF4QuantizationStrategy` encode bits **relative to the
NF4 centroid** — move the value but keep it inside its quantization bucket. That is
a *third* mechanism, distinct from both sign-flip and parity, and structurally it
should have near-zero PPL impact because the dequantized value barely moves.

Both implement the per-tensor `EmbeddingStrategy` ABC
(`embed(residual_tensor, bits, positions)`), not the dict interface production uses.
Needs an adapter in `strategy_registry.py`.

**Trap to avoid:** QAE is quantization-*aware*. It uses knowledge of NF4 that LWE
and sign do not. If QAE wins, the honest statement is "QAE is well-matched to NF4",
not "LWE is stealthy". Any comparison must state this.

**Done — exp17.** `qae` wired via `QaeDictAdapter` (per-layer delegation to the
ABC's own `embed()`, `BaseEmbedder`-shaped result — nothing re-implemented):
**BER 0.0 over 48,256 bits** through exp3's exact production path, decrypt + match,
no-cover probe `structurally_usable` → gate PASS (exp3's 0.0). The trap note above
stands: the class writes `±max(|r|, 0.25·std)` — sign-family with a margin floor —
so exp17 establishes *wiring and round trip*, not a stealth number (no PPL /
robustness / detectability measured yet). `nf4_qae` registered **BLOCKED**: its
`ReferenceBuilder` reference depends on the absolute fp16/nf4 weights, which
neither `strategy.embed` nor `EmbeddingConfig` carries and no caller supplies —
residuals alone cannot rebuild it; the probe's recorded `RuntimeError` is the
measurement of that status. Full record: `RESEARCH_LOG.md` §23.

#### W1.2 Neural strategy (written, never run)

`train_sampled()` is implemented but untested. Blocked on `AdaptiveStrategy`, which
delegates to it.

**Known design concern [read]:** the objective rewards matching the value
distribution's mean and std. A sign flip satisfies both trivially, and nothing
penalises the sign leak. The encoder may converge to the same sign-based solution.
Running it answers this either way.

#### W1.3 Full strategy × model matrix — **done, exp18 (full 7-model grid)**

All viable strategies × 7 models, one table, four axes: extractability, BER,
robustness (BER @ σ=0.001/0.002), detectability.

**Cost control:** detectability is ~10 min/model/strategy. Full 7×5 ≈ 6 h of pure
compute that will thrash 26 GB RAM. **First pass on 3 models** (one per size class:
TinyLlama 1.1B, Qwen2.5-3B, Llama-3.1-8B), then widen. Run one model per process.

**First pass done — exp18, 12 cells** (TinyLlama skipped under the incomplete-cache
rule; gemma-2-2b took the small slot). The four axes came from exp10's own
measurement functions, parametrised by model — protocol unchanged, so the table
extends exp10/exp12 instead of starting a parallel dialect. Result: **round trip
12/12 at BER 0.0 and robustness 12/12 pass — only detectability separates
strategies.** LWE scores **exactly 0.50 on all three models** (3/3 wins; third
reproduction after exp10/exp12); the sign family fails the gate everywhere
(0.59–0.84), with **qae worst** (0.84/0.73/0.59) — exp17's margin floor amplifies
the sign signal rather than hiding it, which also makes W5.2's *"both mechanisms
avoid sign flips"* premise half-false as written. Excluded by name: `neural`
(untrained, W1.2), `nf4_qae` (blocked, exp17). Full record of the first
pass: `RESEARCH_LOG.md` §24.

**Widened — 7 models, 28 cells (2026-10-06).** The four remaining
TARGET_MODELS with complete caches ran under the identical protocol, one
model per process: Phi-3-mini (10 min), Qwen2.5-7B (70 min), Mistral-7B
(75 min), gemma-2-9b (64 min) — all exit 0. TinyLlama stays SKIPPED
(incomplete cache, the standing rule) and gemma-2-2b keeps the small slot,
so the grid is 7 artifacts covering 6 of TARGET_MODELS' 7. Widened result:
**28/28 round trips at BER 0.0, 28/28 robustness gates pass — only
detectability still separates strategies.** LWE scores **exactly 0.50 on
all seven models (7/7 wins)** — the invariant now measured across all five
families and every size class. One new cell result: **magnitude_aware wins
its first cell anywhere — Mistral-7B at 0.525**, the matrix's only
non-LWE win; Mistral is the least detectable model in the grid (sign
0.5563 and qae 0.5563 both land just over the 0.55 gate). Ranges widen
downward: sign 0.56–0.78 (0/7), magnitude_aware 0.53–0.78 (1/7), qae
0.56–0.84 (0/7). **No misuse re-run trigger holds** — every model was
already in the measured set and no strategy changed. Audit re-pinned to
the widened evidence: 7/7 artifacts, 28/28 cells, `lwe_dets == [0.5]×7`,
8 wins with the single non-LWE win named to Mistral. Record:
`RESEARCH_LOG.md` §35.

#### W1.4 Dead or duplicate code

`real_residual_embedder.py` / `_v2.py` have no embed/extract. `residual_embedder.py`,
`_v2`, `_qcae`, `embedder.py` each reimplement `ResidualEmbedder` differently.
`loader.py` (3-value return) and `model_loader.py` (dict return) remain two
divergent residual implementations, and legacy `scripts/exp*.py` import the former.
**Decide: consolidate or delete.** Do not leave four embedders and two loaders.

**Worked decision — executed and verified.** The import graph was
re-grepped live before each deletion (no importer of any deleted file
lay outside the deleted set; no committed `results/*.json` came from
the seven evaluation leaves). The cache build was ported onto
`model_loader.extract_residuals` (optional `cache=` keyword; default
behavior unchanged for every existing caller) and **rebuild-compared
against the committed Qwen2.5-3B cache: 36 layers × 3 tensors, all 108
identical at delta 0.0** — so `loader.py` could not take the only
evidence-rebuilding path with it. Then `scripts/exp1–4` and
`src/model/loader.py` were deleted. The console entry point is
`nes=src.cli:main`, which reaches neither cluster. Post-deletion: 66
tests OK, `claim_audit` 127/127, `check_consistency` 9/9.

**Delete — zero importers anywhere (2 files).** `real_residual_embedder.py`
and `_v2.py` contain only `build_residual()` + `main()` demos: no class, no
`embed`/`extract`, nothing imports them, no doc cites them (the inventory
line above says "three"; there are two on disk).

**Delete as one cluster — the four embedders and their only consumers
(12 files).** `residual_embedder.py` (← `src/main.py`, `qcae_noise_robustness`,
`qcae_embedding_benchmark`), `_v2` (← `multi_cycle_requantization_v2`,
`noise_robustness_v2`, `nf4_embedding_benchmark`,
`nf4_requantization_study_v2`), `_qcae` (← the two qcae modules), `embedder.py`
(← `selector_benchmark`). Those eight consumers are imported by nothing — no
experiment, no test, no other module — and the seven `src/evaluation/*` leaves
produce no committed `results/*.json` and are cited nowhere in README/LOG/PLAN
(only in this section, plus pre-history `attempt2/docs`). Untouched:
`keyed_residual_embedder` (two passing tests) and the live stack
(`intelligent_embedder` + strategies).

**Consolidate on `model_loader.py`, but port the cache builder first
(7 files).** Divergence verified: `load_model_pair` is a 3-tuple in both;
`extract_residuals` differs — `model_loader` returns a `dict` (every modern
caller: runner, `model_context`, `residual_source`, exp8/9, steganalysis)
while `loader` returns `(residuals, fp16_weights, quantized_weights)` **and
writes the residual cache** (`cache.save_layer`, loader.py:553) — the only
cache-*build* path in the repo; `scripts/cache_model.py` is its sole caller
and `model_loader` has no save. So: (a) port the build (save via
`cache_manager` from the `model_loader` path) and verify a rebuild matches an
existing cache for one model; (b) delete `scripts/exp1_probe`,
`exp2_residual_fingerprint`, `exp3_clean_ber`, `exp4_capacity_curve` —
superseded: the manifest's exp1–4 are `src.experiments.experiments.exp1..4`
(registry `"module"` entries) and `run_nes_experiments.py` never touches
`scripts/`; (c) only then delete `src/model/loader.py` — never delete the
only path that can rebuild the evidence. `scripts/exp5a/5b/6` already import
`model_loader`: no divergence, left alone.

**Verification order (executed):** delete the 14 zero-risk files →
suite (66 tests) + `claim_audit` (127/127) + `check_consistency` (9/9); port the
cache build → rebuild-compare one cache → delete `loader.py` + the four
superseded scripts → all three checks again → own commit batch.
**All steps executed and green at each stage:** the rebuild-compare
rebuilt Qwen2.5-3B's cache from scratch (36 layers × 3 tensors) and
found **108/108 tensors identical at delta 0.0** against the committed
cache before `loader.py` was deleted. Commits `5635ecc` (14 files) and
`50ee50b` (port + 5 deletions); full record: `RESEARCH_LOG.md` §31.

---

### W2. Fidelity is the biggest hole

**LWE perplexity is unmeasured anywhere.** Undetectability is worthless if the model
is damaged.

Encouraging prior, measured:

| scheme | mean perturbation | vs residual scale |
|---|---|---|
| sign | 0.0471 | 25.6× |
| LWE | 0.0050 | **2.7×** |

LWE perturbs **9.4× less** than sign. Expect excellent PPL. Expect is not a result.

**Blocker:** the 3-way PPL protocol stalled 20+ min on MPS
(`_local_scalar_dense_mps` behind 3.2 GB of weight uploads). Needs:

- FP16 reference on CPU, NF4 on MPS, or
- chunked layer-by-layer reconstruction

Then: PPL for LWE on ≥2 models, and re-verify sign's number reproduces.

**Done — exp15, both models PASS.** The blocker did not recur (flushed
per-stage logging; the stall's worst symptom was also block-buffered
stdout). LWE embedding-specific PPL Δ: **+0.0077%** (Qwen2.5-3B) and
**+0.0501%** (gemma-2-2b), both orders of magnitude under the 2% gate —
the prior was right. Sign re-verification: exp5's baseline and control
reproduce to every printed digit (12.4707 / 11.3494); the sign arm's
*verdict* reproduces (+0.053% vs recorded −0.005%, both « 2%) while
the digit varies with the fresh-AES-key payload (§21.3). One environment
finding on the way: Phi-3-mini's model pair no longer constructs under
transformers 5.16.1 (`rope_scaling["type"]` KeyError) — not patched,
because patching config would corrupt the measurement; gemma-2-2b took
the second slot. Full record: `RESEARCH_LOG.md` §21.

---

### W3. Adversaries

The detectability claim is only as good as the adversary.

| test | what it answers | cost |
|---|---|---|
| **W3.1 cross-scheme** | train on sign, test on LWE | moderate — **done, exp16** |
| **W3.2 blind patches** | adversary that doesn't know carrier positions | low — **done, exp14** |
| **W3.3 stronger detector** | wider/deeper, more epochs, more pairs | low |
| **W3.4 multi-model detector** | trained on model A, tested on B | high |
| **W3.5 active cleaner** | can an attacker remove the embedding? | high |

**W3.2 was the cheapest high-value item.** Both clean and stego patches were cut
*at carrier positions* — the detector is handed the answer. Carriers are ~0.001%
of positions, so a blind detector sees little per-patch signal, but could still
catch an aggregate distribution shift. **Done — `results/exp14_blind_patch_detector.json`,
gate PASS.** It catches nothing: 50.0% (a constant all-clean predictor, tp=fp=0),
against 68.75% for the carrier-centered control cut from the *same* embeddings
with the *same* detector. Only 4 of 576 blind positions contained a carrier at
all — QACI's magnitude selection clusters carriers into dense aligned blocks,
which the blind adversary almost never lands in. Side finding, and the
verdict's one caveat: this measures placement sensitivity for the production
sign strategy on Qwen2.5-3B; exp7_neural's 70.5% FAIL still stands for an
adversary that knows placement. Full record: `RESEARCH_LOG.md` §20.

**W3.1 result — one direction established, one not.** A sign-trained
detector that scores 62.85% on its own scheme scores **50.00% on LWE**:
transfer fails, with a valid control behind the number. The reverse
direction is **not** established — the LWE-trained detector collapsed to
50.00% on *its own* scheme too, which reproduces exp12's measured LWE
detector accuracy of 50.00% rather than indicating a broken pipeline
(LWE's modifications are 9.4× smaller than sign's). The artifact carries
`controls_valid: false`, so the both-directions reading is forbidden at
the artifact level. Gate PASS as pre-registered (both cross directions
at chance). Full record: `RESEARCH_LOG.md` §22.

**W3.5 matters for the threat model.** If a defender can strip the payload on sight,
the mechanism is not durable even if it is undetectable.

---

### W4. Security engineering

Things a research prototype has no answer for:

- **W4.1 key entropy & extraction.** AES-256 key from `os.urandom`. What happens if
  the key is weak, reused, or partially leaked? Is there a KDF?
- **W4.2 keyless detection.** Could an attacker find the key by searching grid widths?
  LWE's grid width is derived from `HMAC(key, layer_id)` — is the *scale* recoverable
  from the weights alone? If an attacker recovers the scale they may not need the key
  to read bits, which breaks confidentiality. **Done — `results/exp13_keyless_recovery.json`,
  gate FAIL.** The scale was never keyed in the shipped path (one public constant,
  `keyed_branch_active: false`, 0/36 layers key-dependent even in the designed
  formula), and an attacker with the weights and that constant reads all 10,256 bits
  at BER 0.0 with positions at precision/recall 1.0. Scope: the payload is AES-GCM
  ciphertext, so *message* confidentiality is untouched — what fails is the channel's
  key-gating claim. Full record: `RESEARCH_LOG.md` §19.
- **W4.3 multiple payloads.** What if two payloads share a model? Collision and
  crosstalk behaviour is unknown.
- **W4.4 capacity limits.** What is the true maximum, and what breaks first —
  QACI allocation, PER, or detectability?
- **W4.5 malformed input.** Recovery from a partially-written payload.

**W4.2 is the one I'd do first.** It is a potential break of the security property and
it is cheap to test. *(Done — it is a break: exp13, gate FAIL, §0 and
`RESEARCH_LOG.md` §19.)*

---

## 4. Phase C — optional extensions

Useful, publishable, but nothing currently claimed depends on them. Do not let
these displace Phase B.

### W5. Hybrids

Increasing ambition. Each is a separate artifact; none replaces an existing result.

- **W5.1 — `adaptive_strategy` as designed — done, exp19: three models, three
  branches.** Noise-threshold routing to LWE/neural/sign: gemma→lwe,
  Qwen→neural (fails as written without a trained model — recorded, its
  available branches round-trip 0.0), Llama→sign; every round trip 0.0.
  Baseline for anything better. Cheap, as promised.
- **W5.2 — QAE encode + LWE read-out — done, exp21: FAIL, the finding.**
  The premise *"both mechanisms avoid sign flips"* was half-false
  (exp17: qae flips signs); the combination measures **0.5433** vs the
  0.0 gate (matched control 0.0 — attributable to the pairing), and the
  public correction `raw ⊕ cell-parity(|stego|) ⊕ 1` returns **0.0**:
  `parity(v) = sign(v) ⊕ cell-parity(|v|)` measured — a "hybrid" would
  be sign reading plus a public relabeling, not a second channel.
- **W5.3 — sign/parity split — done, exp20: it is a dial.** Parity on a
  fraction of carriers, sign on the rest: robustness falls and
  detectability falls monotonically with parity share (σ0.002 0→0.0127,
  detector 0.7875→0.50); pure parity reproduces exp18's lwe cell
  exactly and is the only cell winning all gates. First pass, 1 model.
- **W5.4 — per-layer strategy selection — done, exp22: buildable, and it does not
  help.** Magnitude-keyed widths (`clip(4.0·round(std,4), 0.005, 0.020)`)
  round-trip 0.0 but **fail both robustness gates** (0.0226/0.5763): the
  extractor sizes its grid from the tensor it receives, and under noise that
  std is inflated — 36/36 layer buckets move at every σ (recorded, recomputed
  from the artifact). Rank-keyed widths (noise cannot move order) **pass
  every gate** — but the detector is width-blind (0.50 ×3) and the ladder's
  sub-default median costs robustness (σ0.002 0.0127→0.0736): **the shipped
  global 0.010 stays the best point of the three.**

### W6. Model surgery

Does the embedding survive contact with the rest of the ML lifecycle? A carrier that
survives one cycle but dies on first fine-tune is not viable.

- **W6.1** survive LoRA merge
- **W6.2** survive 1–10k fine-tuning steps
- **W6.3** survive re-quantization (NF4 → GPTQ → back)
- **W6.4** survive pruning / distillation
- **W6.5** survive model merging

Each is: embed → surgery → extract → BER. Expect degradation; the question is
whether it is total or graceful.

### W7. The Pareto frontier

Stop treating stealth and robustness as separate gates. Map the frontier:

- x-axis: mean perturbation magnitude
- y-axis: detector accuracy
- marker: BER @ σ=0.001

Plot every strategy × every parameter. The publishable claim is the *shape* of this
frontier and where sign and LWE sit on it — not two isolated PASS/FAIL cells. The
grid-width sweep already hints at it: 0.002 undetectable but fragile, 0.05 robust but
detected, 0.005–0.020 both.

---

## 5. Phase D — productization

Gated on Phase B. Nothing here ships before W3.1, W3.2 and W6.1 hold.

### W8. Delta-only distribution — the key reframing

The embedding changes 10,256 of ~811M values (0.0013%). So the product is **not** a
modified base model. It is a delta, `W_stego − W_clean`, shipped like a LoRA adapter:

- ~0.2% of model size instead of 100%
- auditable: the recipient can inspect exactly what changed
- no modified base model enters any registry
- recipient reconstructs locally; nothing is uploaded

This also makes the security story much cleaner — you distribute a *patch*, not a
weaponised checkpoint.

### W8.1 Recipient-side tool

A CLI/library that takes base model + delta + key → recovers the payload. Needs to be
usable by someone who did not build it.

**Implemented** (`src/delta/` + `nes delta-export | delta-inspect |
delta-extract`, commit `6fa6f54`): recovery runs one arithmetic path
(`R_embed = R_clean + delta`) through production's `DecryptPipeline`, so
the recipient reads with the same extractor production wrote with.
**Measured end-to-end on Qwen2.5-3B**: a 30,253-byte delta (824
carriers, 36 layers, 387 values changed, 792-bit payload) recovered the
message byte-identical from base model + delta + key. The reader
decodes the production `sign` scheme and refuses any other by name.
Suite: 14 delta tests (`tests/test_delta_distribution.py`).

### W8.2 Statistical guards on the delta itself

If the delta is the deliverable, it should carry its own integrity metadata: hash,
carrier count, payload length. Prevents silent corruption.

**Implemented**: sha256 over canonical metadata JSON + canonical
tensor bytes, carrier count, payload length (+ changed count, layer
shapes, strategy, model id). `verify_delta` re-derives every count
from the arrays on load, and extraction cross-checks the decoded
length header against metadata — so a tampered value, a forged count
or a re-signed payload length fails loudly instead of extracting as
noise. Pinned by tests, including the re-signed-forgery case.

### Risk register

| risk | severity | note |
|---|---|---|
| **Smuggling past safety scanning** | **high** | A hidden payload extracted by the model at inference is a direct route around output filters. Any product must be able to answer "can this be abused?" before shipping. |
| **Evading model fingerprinting / provenance** | **high** | Weights-distribution changes are used for provenance in some regimes. Hidden data in weights conflicts with that. |
| **IP / licensing** | **high** | Redistributing derived weights may violate base-model licences. Unreviewed. |
| Regulatory (EU AI Act-style obligations) | medium–high | Needs a compliance answer before distribution. |
| Dual-use by others | medium | The capability exists regardless of what we ship. |

**This is a research programme, and legitimate as such.** It becomes a product only
after an explicit misuse assessment. Flagging once, here, so it is a decision rather
than a surprise.

### Misuse assessment — §7's gate, worked through

The paragraph above is the trigger; this is the assessment. Every evidence cell is
audit-pinned (127/127 at writing) and carries a §2 status — nothing is argued from a
number that is not in `results/`.

| register risk | evidence | assessment | gate condition |
|---|---|---|---|
| **Smuggling past safety scanning** (high) | **MEASURED**: capacity ≥ 10M payload bits at BER 0 (exp4) touching 0.0013% of values; payload = AES-GCM ciphertext under a `os.urandom` key; the *bits* need no key (exp13: all 10,256 read at BER 0.0, precision/recall 1.0, clean control 0 candidates); a parity reader recovers a sign stream up to a public relabeling (exp21: 0.5433 ≡ sign ⊕ cell-parity, matched control 0.0); the payload-bearing model behaves normally (LWE Δ vs control +0.0077% / +0.0501%, both PASS vs the 2% gate, exp15) | The abuse path never needs to break AES — the deployer holds the key. What is measured: a full-size payload fits a passing model and is readable offline by anyone with weights + key. **This repo extracts offline over weights (READ: `DecryptPipeline`'s pipeline); the register's "at inference" route is neither implemented nor measured here — recorded as hypothesis, not finding.** | Distribution only with W8's audit story attached (inspectable delta + integrity metadata) and the detection evidence below |
| **Evading fingerprinting / provenance** (high) | **MEASURED**: a *knowing* adversary detects `sign` at 70.5% (exp7_neural FAIL vs the 0.55 gate; structural 69.4–74.4% across α×100, γ×5, payload×10) and 68.75% carrier-centered (exp14's control); a *blind* adversary learns nothing — 50.0%, tp=fp=0 (exp14 PASS; only 4/576 blind positions held a carrier); the **LWE** channel evades every detector trained in this programme — 0.50 invariantly across six experiments (exp10/12/16/18/20/22) | Evasion is strategy-conditional, and both halves are measured: `sign` is detectable by an adversary who knows where to look, LWE defeats every adversary we could build — our own negative result is the provenance conflict. | Any provenance or fingerprinting claim published alongside this work must state the LWE 0.50 invariant as its own limit |
| **IP / licensing** (high) | **NOT_RUN** — no base-model licence review exists | Unreviewed, exactly as the register says | No external distribution of code, weights or deltas until reviewed |
| **Regulatory** (medium–high) | **NOT_RUN** — no compliance analysis exists | Owed before any product claim | Same gate as IP |
| **Dual-use by others** (medium) | **READ** — the capability is a property of the method, not of anything held back | Withholding artifacts reduces spread at the margin, not existence | Decision recorded at publication time, not discovered later |

**Permitted today:** the programme as research — artifacts, gates and these
documents inside the repo, results citable as measurements. **Forbidden today:**
any external distribution of code, weights or deltas, until (a) the IP and
regulatory rows are no longer NOT_RUN and (b) W8's recipient-side audit exists,
so whatever leaves the repo is inspectable by whoever receives it.

**Re-run triggers — the assessment is redone as a new revision, not edited in
place, if any holds:** W8 ships a recipient tool; extraction is demonstrated at
inference time; a detector defeats the LWE 0.50 invariant; or a model/strategy
outside the measured set is added.

### Misuse assessment — revision 2 (post-W8), §7's gate re-run

Revision 1 stands as written above; this revision is *added*, not edited in,
because its own rule requires it: **"W8 ships a recipient tool"** now holds
(`nes delta-export | delta-inspect | delta-extract`, commits `6fa6f54` +
`12a8c4a`, measured end-to-end in `RESEARCH_LOG.md` §32). The evidence base
also advanced past rev 1's exp13–22: exp23 (model surgery) and exp24 (Pareto
frontier) are measured; `claim_audit` still pins 127/127.

| register risk | evidence delta since rev 1 | assessment after W8 | gate condition |
|---|---|---|---|
| **Smuggling past safety scanning** (high) | **W8 ships the artifact class now**: the payload moved from a ~6 GB checkpoint to a **30,253-byte delta** that verifies itself; the audit path exists and ran end-to-end (inspect checks hash / carrier count / payload length with no models loaded; extract recovered the message byte-identical; a forged-and-re-signed payload length still fails at the decoded header). The register's "at inference" extraction route remains neither implemented nor measured here — unchanged: hypothesis, not finding. | **Rev 1's W8 half is met**: whatever leaves the repo is inspectable — the delta carries its integrity metadata, the key travels out of band in a separate file, and non-sign schemes are refused by name rather than misread. The smaller artifact also moves more easily than a checkpoint; that cuts both ways, and the audit story is the counter, not the size. | Unchanged, and its W8 half is now **satisfied** — audit story attached. Detection evidence below still required. |
| **Evading fingerprinting / provenance** (high) | exp22: rank-keying also 0.50; exp24: the entire measured frontier is dominated by one y = 0.50 point, 34/34 | Direction unchanged, now frontier-wide: **no measured configuration beats chance**. On the honest side, a shipped delta carries its own sha256, strategy and key id — the patch itself is fingerprintable and verifiable, which is what provenance tooling needs. | Unchanged: any provenance claim must state the LWE 0.50 invariant — now, the whole frontier — as its own limit. |
| **IP / licensing** (high) | nothing — still **NOT_RUN** | Unreviewed; W8 changed nothing here | Unchanged: no external distribution of code, weights or deltas until reviewed |
| **Regulatory** (medium–high) | nothing — still **NOT_RUN** | Owed before any product claim | Same as IP |
| **Dual-use by others** (medium) | the workflow is now three commands inside the repo | Withholding matters even less once the capability is a CLI invocation; it remains a property of the method either way | Decision recorded at publication time, not discovered later |

**Verdict after W8 — unchanged in effect, narrowed in reason.** Permitted
today: the programme as research — artifacts, gates and these documents
inside the repo, results citable as measurements. Forbidden today: any
external distribution of code, weights or deltas, **solely because (a) the IP
and regulatory rows are still NOT_RUN** — condition (b), W8's recipient-side
audit, is now built, tested and measured. If (a) closes, this register has no
other outstanding distribution condition.

**Re-run triggers — rev 1's consumed, remainder re-armed.** "W8 ships a
recipient tool" is consumed: it is what produced this revision. Still armed:
extraction demonstrated at inference time; a detector defeats the LWE 0.50
invariant; a model/strategy outside the measured set is added.

### IP sign-off — status record (not a revision)

Recorded 2026-10-06 on the author's instruction: **the base-model licence
review is done and signed off — the author confirms the licences of the base
models in the measured set permit redistribution of derived artifacts**
(code, checkpoints, deltas). §5's **IP row moves NOT_RUN → REVIEWED (author
sign-off)**. No re-run trigger holds — no inference-time extraction, no
detector beats the 0.50 invariant, no model or strategy outside the measured
set — so this is appended as a status record rather than a revision: rev 1
and rev 2 stand as written, and their "still NOT_RUN" rows are historically
true at their revision dates. **Regulatory remains NOT_RUN**, so rev 2's
verdict is unchanged in effect with its condition (a) now halved:
distribution of code, weights or deltas outside the repo stays **forbidden
solely on the regulatory row** (condition (b), W8's recipient-side audit,
was cleared in rev 2).

### What I would not do

- **Add more models.** 5 families, 22–42 layers. Low marginal value versus W3/W6/W4.
- **Build a product before W3.1/W3.2/W6.1.** Publishing a stealth claim that a
  cross-scheme or blind detector defeats would be the worst outcome available.
- **Rename `LWE-Inspired` unilaterally.** It is a claimed contribution; renaming it
  is the authors' call, not a code fix.
- **Touch the 0.002 threshold.** An author decision, recorded not applied.
- **Lower a gate to close a cell.** See §2; `check_consistency.py` fails if a FAIL
  is quietly rewritten.

---

## 6. How to resume

`nes-llm/README.md` is the entry point: package layout, one line per experiment
with its gate and current state, and these same commands. Then:

```bash
cd nes-llm

# one command, any subset
../.venv/bin/python run_nes_experiments.py --models Qwen/Qwen2.5-3B
../.venv/bin/python run_nes_experiments.py --models Qwen/Qwen2.5-3B --exp exp3 exp6

# audit, verify, report
../.venv/bin/python run_nes_experiments.py --audit
../.venv/bin/python check_consistency.py
../.venv/bin/python claim_audit.py            # every MEASURED claim vs disk
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'

# diagnostics
../.venv/bin/python -m src.model.verify_residual_cache <model> <family> <n_layers>
../.venv/bin/python -m src.experiments.exp12_lwe_cross_model --models <id>
../.venv/bin/python -m src.experiments.exp2_criterion_calibration
../.venv/bin/python src/steganalysis/exp7_neural_parameter_study.py
```

Machine: 26 GB RAM, ~30 GB MPS ceiling, 208 GB free, 7 models + GPTQ/AWQ cached.
Residuals 1.0–8.6 GB; run **one model per process**.

### Ground rules that must survive

1. A completed cell (PASS **or** FAIL) is skipped unless `--force`. Never rerun a
   FAIL until it passes.
2. Thresholds live in `experiment_registry.THRESHOLDS`. No experiment edits its own.
3. A missing artifact is reported missing. Never defaulted to PASS.
4. `EXP` results are written to `RESULTS_DIR`. Two modules got this wrong
   (`nes-llm/results` vs repo-root `results/`); use `paths.RESULTS_DIR`.
5. Any experiment that trains a detector needs ≥400 pairs. At 120 every variant
   scores exactly 50% and the study is void.
6. Verify a dequantizer against the FP16 reference before using it (corr > 0.95,
   residual ratio < 0.5). AWQ is compared *after* removing the per-channel scale it
   folds into the LayerNorm — the correction, not the threshold, is what moved.
7. Never reshape a mismatched matrix — transpose or raise.
8. A layer that fails verification is excluded and **named** in the artifact
   (`metrics.layers_excluded`), never embedded into and never silently dropped.
9. NaN compares False against every threshold. Write `not (x >= limit)`, not
   `x < limit`, or an all-zero dequantization passes as verified.
10. A number in these documents comes from `results/*.json`, never from console
    output, and `python claim_audit.py` must exit 0 for it to be cited. Two claims
    in this repo failed that rule before it existed.

---

## 7. Suggested order

Interchangeable, but this sequence front-loads the cheapest results that could
invalidate later work.

| # | Item | Why here |
|---|---|---|
| 1 | **W4.2** key/scale recoverability — **done, exp13: FAIL** | cheap; potential break of the security property (it was) |
| 2 | **W3.2** blind-patch adversary — **done, exp14: PASS** | cheapest test of the central claim (it was: 50.0% blind vs 68.75% control) |
| 3 | **W2** LWE perplexity — **done, exp15: PASS ×2** | biggest hole; invalidates the strategy choice if it fails (it did not: 0.008% / 0.050%) |
| 4 | **W3.1** cross-scheme detector — **done, exp16: PASS (scoped)** | the claim's main weakness (sign→LWE: no transfer, control-validated; reverse uninformative) |
| 5 | **W1.1** QAE adapter + round trip — **done, exp17: `qae` PASS, `nf4_qae` BLOCKED** | adds two strategies cheaply (one wired, one's blocker diagnosed) |
| 6 | **W1.3** strategy × model matrix — **done, exp18: full 7-model grid — one axis decides (28/28 round trips + robustness, LWE 0.50 ×7)** | the breadth deliverable — closed; TinyLlama SKIPPED (incomplete cache), gemma-2-2b holds the small slot |
| 7 | **W5** hybrids — **all done: 7a/exp19 (adaptive), W5.3/exp20 (parity-share dial), W5.2/exp21 (FAIL = finding: interop 0.5433 vs 0.0, parity ≡ sign ⊕ public relabeling), W5.4/exp22 (buildable but does not help: magnitude-keying fails the noise gate with cause measured, rank-keying passes, global width stays best)** | most interesting science — closed; the last item it named (exp18's 7-model widening) is now measured too |
| 8 | **W6** model surgery | **measured, all legs run (exp23, 12 cells): W6.1 LoRA ✓ (both ratios), W6.4 prune ✓ (both fractions, zero carriers displaced), W6.5 merge ✓ to t=0.05, W6.2 real 1,000-step fine-tune ✓ at 0.0 (loss 2.271→2.169, deterministic); NF4 re-quant fails at 0.3832, half-merge at 0.2476, GPTQ at 0.4956 (near chance), AWQ at 0.4108 — all graceful (short of chance), gate untouched; not_run empty — determines viability |
| 9 | **W7** Pareto frontier | **measured (exp24): the frontier is ONE point — exp22's layer_rank (x 0.00372693, y 0.50, marker 0.00146256) dominates all 34/34 others, minimal on both axes at once, so no trade-off exists among committed results; x measured here (no artifact pairs magnitude with a detector), y/marker cited at delta 0.0, 10 exclusions + 6 omissions recorded with reasons — the strongest publishable framing** |
| 10 | **W1.4** consolidation | **executed + verified: 14 zero-risk files (4 duplicate embedders + 8 consumers + 2 demos) deleted, cache-build ported onto `model_loader` and rebuild-compared against the committed Qwen2.5-3B cache (36 layers × 3 tensors, 108/108 identical at delta 0.0), `scripts/exp1–4` + `src/model/loader.py` deleted — suite 66 OK, audit 127/127, consistency 9/9 after every batch** |
| 11 | **W8** delta productization | **implemented + measured end-to-end (commit `6fa6f54`): `src/delta` format with W8.2 integrity metadata (sha256 + carrier count + payload length, all re-derived on load), `nes delta-export/inspect/extract` recipient CLI reading through production's `DecryptPipeline`; Qwen2.5-3B round trip: 30,253-byte delta (824 carriers, 387 changed, 792-bit payload) → message byte-identical — suite 80 OK, audit 127/127, consistency 9/9** |
| 12 | **W9** paper hardening (exp25–27) | **measured, all three PASS (gates pre-registered in THRESHOLDS, +25 audit claims): exp25 selection ablation — one shared Hamilton allocation digest-pinned `8660969e1ddd`, magnitude buys σ=0.001 robustness (0.0 vs 0.0724–0.0757) at equal detectability (0.5028 / KL ≤ 3.7e-05), no policy publicly readable (keyless 0.4713–0.5084 across all arms), H1/H2/H4 supported, H3 (public re-run ⇒ BER < 0.1) missed and kept as written — mechanism attributed: allocation drift, never reproduces in full (7/36); exp26 capacity curve — 1k→50k all BER 0.0 + decrypt, acc flat 0.5028, KL 0–5.53e-05, mean\|Δ\| 0.0662→0.0351, four-panel SVG committed; exp27 boundary — control 0.0, wrong keys 0/10 (10/10 GCM auth, 0 plaintext), partial access 0/6 (available bits all 0.0, coverage 0.037–0.521 monotone), public-rule attacker 0.4875 exact / 0.4937 nominal at precision 0.9899 (alloc 2/36)** | the three experiments demanded before paper writing — closed; **the next workstream is writing, not building** |
| — | misuse assessment — **rev 1 worked through in §5 against exp13–22; rev 2 (post-W8) re-run after the recipient tool shipped — verdict unchanged in effect, narrowed in reason: research permitted, distribution forbidden solely on IP/regulatory NOT_RUN (W8 audit half of the gate now built and measured), triggers re-armed; IP signed off by the author 2026-10-06 (§5 status record, no trigger) — regulatory NOT_RUN is now the sole remaining distribution condition** | gate before any distribution |
