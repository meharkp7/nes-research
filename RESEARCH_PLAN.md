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
| **Strategy × model matrix, first pass (W1.3)** | 4 READY strategies × 3 models: **12/12 round trips BER 0.0, 12/12 robustness gates pass — only detectability separates**; LWE detector **0.50 on all three models** (3/3 wins), sign-family 0.59–0.84 everywhere (qae worst: 0.84/0.73); `neural` + `nf4_qae` excluded by name | `results/exp18_matrix_*.json` |
| **Adaptive routing as designed (W5.1)** | three models → **three different branches** (gemma σ=0.000448→lwe, Qwen σ=0.001554→neural, Llama σ=0.007736→sign); every round trip that ran holds BER 0.0; Qwen's neural route fails design-as-written (no trained model) and is **recorded as the design's own**, both available branches round-trip 0.0 as its fallback | `results/exp19_adaptive_*.json` |
| **Sign/parity split dial measured (W5.3)** | five parity shares × exp10's three axes: all round trips **BER 0.0**; BER@σ0.002 rises **0 → 0.0127** and detector falls **0.7875 → 0.50** with parity share — the stealth/robustness trade-off is a dial; pure parity **reproduces exp18's lwe cell exactly**, only it wins all gates | `results/exp20_split_dial_*.json` |
| **QAE encode + LWE read-out (W5.2)** | plan called it plausible; measured **0.5433 vs the 0.0 gate → FAIL (the finding)**, matched control **0.0**; the public correction returns **0.0** — `parity(v) = sign(v) ⊕ cell-parity(|v|)` measured: the "hybrid" is sign reading plus a public relabeling | `results/exp21_qae_lwe_*.json` |
| **NES round-trips at BER 0.0 through three 4-bit formats: NF4, GPTQ, AWQ** | GPTQ corr 0.9903 / AWQ corr 0.9941 (raw 0.9890) | `results/exp9_formats.json` |
| 9 model ids covered, 0 ERROR | 35 PASS / 6 FAIL / **0 NOT_RUN** | `results/experiment_manifest.json` |

**How those rows are kept honest.** `nes-llm/claim_audit.py` re-derives every
MEASURED row above from disk: **87/87 pass**, with `check_consistency.py` at
9/9 and a 42-test suite running green. Four claims in this document failed
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
../.venv/bin/python claim_audit.py                    # 87/87 must pass
```

`claim_audit.py` is this section made executable: each row is re-derived from
the artifact it names — counts included, because every claim that failed this
audit failed on a count while the values underneath stayed correct — and it
exits non-zero on anything it cannot verify. It passes **87/87** as written.

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
| **Strategy × model matrix (W1.3, first pass)** | exp10's three axes × 4 strategies × 3 models: round trip **12/12 BER 0.0**, robustness **12/12 pass** (σ0.001 all 0.0); detector — LWE **0.5000 ×3** (wins 3/3), sign 0.61–0.78, magnitude_aware 0.59–0.78, qae **0.59–0.84** (0/9 for the sign family); pipeline flag false for LWE (sign-only DecryptPipeline, known wiring gap) | `exp18_matrix_*.json` |
| **Adaptive routing as designed (W5.1)** | three models → **three different branches**: gemma σ **0.000448**→lwe, Qwen σ **0.001554**→neural (route fails design-as-written — no trained model — `EmbeddingError` recorded, forced sign/lwe fallbacks both **0.0**), Llama σ **0.007736**→sign; all four round trips that ran **BER 0.0**; each branch recomputes exactly from recorded σ + thresholds | `exp19_adaptive_*.json` |
| **Sign/parity split dial (W5.3)** | 5 fractions × Qwen2.5-3B: round trips **5/5 BER 0.0**, σ0.001 **0.0 everywhere**; detector **[0.7875, 0.6938, 0.70, 0.5875, 0.50]**, BER@σ0.002 **[0, 0.0034, 0.0067, 0.0095, 0.0127]** across parity share 0→1 (stealth↑ robustness↓); only pure parity wins; its detector **= exp18's lwe cell (delta 0.0)**, pure-sign endpoint +0.0375 from exp18's sign cell (fresh-key variance, mechanism verified bit-identical) | `exp20_split_dial_*.json` |
| **QAE encode + LWE read-out (W5.2)** | one embed, same stego, three readings: matched **0.0** (control), raw LWE parity **0.5433** (5,572/10,256 — gate 0.0, **verdict FAIL**), public cell-parity correction **0.0**; `parity(v)=sign(v)⊕cell-parity(|v|)` measured, not asserted; complement prediction missed (carriers all ≥ one grid width: sampled min 0.013 > 0.010) — miss recorded | `exp21_qae_lwe_*.json` |
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
verify. All 87 checks pass at the time of writing. Run it before citing
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
| `claim_audit.py` | `nes-llm/claim_audit.py` | 87 checks re-deriving every MEASURED claim — counts included — and exiting non-zero on any it cannot verify |
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
| Strategy × model matrix — **first pass measured (exp18, 3 models)** | breadth deliverable; widening to the full 7-model grid remains |
| Model surgery (LoRA merge, fine-tune, re-quantize, prune, merge) | determines viability |
| Pareto frontier | the strongest publishable framing |
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

#### W1.3 Full strategy × model matrix — **done (first pass), exp18**

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
(untrained, W1.2), `nf4_qae` (blocked, exp17). Widening to the full 7-model grid
remains open (NOT_RUN row annotated). Full record: `RESEARCH_LOG.md` §24.

#### W1.4 Dead or duplicate code

`real_residual_embedder.py` / `_v2.py` have no embed/extract. `residual_embedder.py`,
`_v2`, `_qcae`, `embedder.py` each reimplement `ResidualEmbedder` differently.
`loader.py` (3-value return) and `model_loader.py` (dict return) remain two
divergent residual implementations, and legacy `scripts/exp*.py` import the former.
**Decide: consolidate or delete.** Do not leave four embedders and two loaders.

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
- **W5.4 — per-layer strategy selection.** Different grid width per layer, keyed by
  layer noise. Layers differ: Qwen2.5-7B spans 0.0012–0.0130, Phi-3 spans
  0.0026–0.0031.

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

### W8.2 Statistical guards on the delta itself

If the delta is the deliverable, it should carry its own integrity metadata: hash,
carrier count, payload length. Prevents silent corruption.

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
| 6 | **W1.3** strategy × model matrix (3 models) — **done, exp18: one axis decides** | the breadth deliverable (first pass; 7-model widening remains) |
| 7 | **W5** hybrids — **7a done, exp19 (adaptive); W5.3 done, exp20 (parity-share dial); W5.2 done, exp21 (FAIL = finding: interop 0.5433 vs 0.0, parity ≡ sign ⊕ public relabeling)** | most interesting science — only W5.4 (per-layer selection) left |
| 8 | **W6** model surgery | determines viability |
| 9 | **W7** Pareto frontier | the strongest publishable framing |
| 10 | **W1.4** consolidation | cleanup; do before W8 |
| 11 | **W8** delta productization | only after 1–8 hold |
| — | misuse assessment | gate before any distribution |
