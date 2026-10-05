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
| Sign robustness under both noise gates | σ=0.001 ≤ 0.0012, σ=0.002 ≤ 0.0108; BER 0.0 on **6 of 7** models | `results/exp6_*.json` ⚠ see audit note |
| Sign fidelity 0.0053% PPL delta | 3-way protocol, Qwen3B only | `results/exp5_*.json` |
| Sign is detected at 70.5% | carrier-centered MLP | `results/exp7_neural_*.json` |
| Sign detection is structural | α×100, γ×5, payload×10 → 69.4–74.4% | `results/exp7_neural_parameter_study.json` |
| LWE grid encoding is extractable without the cover | BER 0.5036 → 0.0000 (`extract_needs_cover: false`) | `results/exp10_*.json` |
| LWE passes both gates on every model it was measured on | BER 0.0000, detector 50.00%, all measured models pass | `results/exp12_lwe_cross_model.json` ⚠ see audit note |
| **NES round-trips at BER 0.0 through three 4-bit formats: NF4, GPTQ, AWQ** | GPTQ corr 0.9903 / AWQ corr 0.9941 (raw 0.9890) | `results/exp9_formats.json` |
| 9 model ids covered, 0 ERROR | 35 PASS / 6 FAIL / **0 NOT_RUN** | `results/experiment_manifest.json` |

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
2. **Every detectability number uses a detector trained against the same scheme it
   tests.** Cross-scheme is untested.
3. **Clean and stego patches are both cut at carrier positions.** The detector is
   handed the carrier locations.
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
```

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
| LWE both gates | BER 0.0, detector 50.00%, all measured models pass ⚠ | `exp12_lwe_cross_model.json` |
| **GPTQ round trip** | BER **0.0**, 10,256/10,256 bits, corr 0.9903, 36/36 layers | `exp9_formats.json` |
| **AWQ round trip** | BER **0.0**, 10,256/10,256 bits, corr 0.9941, **35/36 layers** | `exp9_formats.json` |
| Suite coverage | 35 PASS / 6 FAIL / 0 NOT_RUN / 0 ERROR | `experiment_manifest.json` |
| AWQ reconstruction perplexity | 33.59 → 35.25 (+5%); layer-2 `up_proj` zeroed → 40.65; reference `up_proj` → **inf** | `RESEARCH_LOG.md` §16 (log entry, not a `results/` artifact) |

**Audit note on the ⚠ row (raised by this audit, not yet closed).** The
LWE cross-model result was written up as *"5 of 5 measured models pass
both gates"*, with a six-row table in `RESEARCH_LOG.md` §7 and a commit
message to match. `results/exp12_lwe_cross_model.json` contains **four
entries, three of them measured**: TinyLlama SKIPPED, then Qwen2.5-7B,
Llama-3.1-8B and gemma-2-9b. Qwen2.5-3B and Mistral-7B are absent
entirely — not skipped, simply not in the file — and no archived version
of this artifact has ever contained them. So the 5-model figure has no
artifact behind it, while the file that exists says 3 of 3.

Both numbers agree on the part that matters (every measured model
passes, BER 0.0000, detector 50.00%), which is why this is a coverage
discrepancy rather than a contradiction. The row above is worded so it
is true either way, and `exp12` is being re-run across all six cached
models to close the finding. Do not cite "5/5" until it does.

**Audit note on the ⚠ on robustness (raised, closed).** This row and
two others read *"BER 0 at σ=0.001"*. It is 0.0 on six of seven models;
**Mistral-7B is 0.00115 at σ=0.001 and 0.01074 at σ=0.002**, both
inside the 0.02 / 0.10 gates, so every cell still reads PASS and no
threshold moved. Only the claim was wrong — an average presented as a
maximum — and the row above now says what the seven numbers actually
are.

**Audit note on exp2's count (raised, count corrected, artifact open).**
Both documents said exp2 fails on **6 of 7** models, and `RESEARCH_LOG`
§4.1 said *"1/7 pass, six fail"*. The manifest says **4 of 7 fail**:
Qwen2.5-3B, Qwen2.5-7B, TinyLlama and Llama-3.1-8B fail; gemma-2-9b,
Phi-3-mini and Mistral-7B pass. Neither older figure came from the cells.

Two figures existed because two measurements exist.
`exp2_criterion_calibration.json` is a *separate* run over the legacy
`residual_profile_*.json` files — a set that contains `gemma-2-2b` and no
Phi-3 — written at a moment when gemma-2-9b and Llama-3.1-8B still carried
their pre-rerun values (0.000382 / 0.000879 against today's 0.003605 /
0.022642). Its own `finding` string also read *"Zero of 7"* while its own
`models_passing` read `1`, because "Zero" was hard-coded rather than
counted. The generator is fixed; regenerating the artifact is what remains,
and two `claim_audit.py` checks hold that open.

`python nes-llm/claim_audit.py` re-derives every MEASURED claim in this
section from `results/*.json` and exits non-zero on any it cannot verify,
including the still-open ⚠. Run it before citing any number here.

### FAIL — measured, gate did not pass, not rewritten

| cell | number | why it stands |
|---|---|---|
| `exp2` on 4 of 7 models | ≥ 80% of layers with mag_mean > 0.002 | threshold tracks quantization format; changing it is an author decision |
| `exp7_neural`, sign | 70.5% vs 55% | proved structural across a 100× α and 5× γ sweep |
| `exp8` cross-model | FAIL | correctly inherits the neural FAIL rather than averaging it away |
| AWQ `model.layers.2.mlp.down_proj` | corr 0.9337 | excluded from Exp9's residual set and named in `metrics.layers_excluded` |
| AWQ `model.layers.32` / `33` `self_attn.v_proj` | 0.9211 / 0.9147 | same; outside Exp9's module type entirely |

### IMPLEMENTED — exists, tested, no result depends on it

| item | where | note |
|---|---|---|
| GPTQ + AWQ dequantizers | `src/quantization/adapters.py` | 18 tests in `tests/test_quantization_adapters.py` |
| `verify_dequantization` gate | `adapters.py` | includes the absorbed-scale correction and the NaN-safe comparison |
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
| LWE perplexity on any model | undetectability is worthless if the model is damaged |
| Cross-scheme detector (train sign → test LWE) | the claim's main weakness |
| Blind-patch adversary | both patch classes are cut at carrier positions today |
| Strategy × model matrix | breadth deliverable; 3 of 7 models first |
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
| exp2 residual fingerprint | FAIL ×6, PASS ×1 | threshold tracks quantization format; unchanged |
| exp3 round trip | PASS ×7 | BER 0.0 |
| exp4 capacity | PASS | ≥ 10M bits at BER 0 |
| exp5 fidelity | PASS | +0.0053% PPL |
| exp6 robustness | PASS ×7 | under both gates; BER 0.0 on 6 of 7 models (Mistral-7B 0.00115) |
| exp7 statistical | PASS ×7 | below the 55% gate |
| exp7 neural | **FAIL** ×7 | 70.5% vs 55%; structural, not a fluke |
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

#### W1.1 Quantization-aware strategies (never run)

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

#### W1.2 Neural strategy (written, never run)

`train_sampled()` is implemented but untested. Blocked on `AdaptiveStrategy`, which
delegates to it.

**Known design concern [read]:** the objective rewards matching the value
distribution's mean and std. A sign flip satisfies both trivially, and nothing
penalises the sign leak. The encoder may converge to the same sign-based solution.
Running it answers this either way.

#### W1.3 Full strategy × model matrix

All viable strategies × 7 models, one table, four axes: extractability, BER,
robustness (BER @ σ=0.001/0.002), detectability.

**Cost control:** detectability is ~10 min/model/strategy. Full 7×5 ≈ 6 h of pure
compute that will thrash 26 GB RAM. **First pass on 3 models** (one per size class:
TinyLlama 1.1B, Qwen2.5-3B, Llama-3.1-8B), then widen. Run one model per process.

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

---

### W3. Adversaries

The detectability claim is only as good as the adversary.

| test | what it answers | cost |
|---|---|---|
| **W3.1 cross-scheme** | train on sign, test on LWE | moderate |
| **W3.2 blind patches** | adversary that doesn't know carrier positions | low |
| **W3.3 stronger detector** | wider/deeper, more epochs, more pairs | low |
| **W3.4 multi-model detector** | trained on model A, tested on B | high |
| **W3.5 active cleaner** | can an attacker remove the embedding? | high |

**W3.2 is the cheapest high-value item.** Both clean and stego patches are currently
cut *at carrier positions* — the detector is handed the answer. Carriers are ~0.001%
of positions, so a blind detector sees little per-patch signal, but could still
catch an aggregate distribution shift.

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
  to read bits, which breaks confidentiality.
- **W4.3 multiple payloads.** What if two payloads share a model? Collision and
  crosstalk behaviour is unknown.
- **W4.4 capacity limits.** What is the true maximum, and what breaks first —
  QACI allocation, PER, or detectability?
- **W4.5 malformed input.** Recovery from a partially-written payload.

**W4.2 is the one I'd do first.** It is a potential break of the security property and
it is cheap to test.

---

## 4. Phase C — optional extensions

Useful, publishable, but nothing currently claimed depends on them. Do not let
these displace Phase B.

### W5. Hybrids

Increasing ambition. Each is a separate artifact; none replaces an existing result.

- **W5.1 — `adaptive_strategy` as designed.** Noise-threshold routing to LWE/neural/sign.
  Cheap. Tests someone else's design and gives a baseline for anything better.
- **W5.2 — QAE encode + LWE read-out.** Quantization-aware placement, parity decode.
  Plausible: both mechanisms avoid sign flips.
- **W5.3 — sign/parity split.** Parity on a fraction of carriers, sign on the rest.
  Makes the stealth-vs-robustness trade-off an explicit dial rather than a
  per-scheme guess. **Most interesting scientifically.**
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

```bash
cd nes-llm

# one command, any subset
../.venv/bin/python run_nes_experiments.py --models Qwen/Qwen2.5-3B
../.venv/bin/python run_nes_experiments.py --models Qwen/Qwen2.5-3B --exp exp3 exp6

# audit, verify, report
../.venv/bin/python run_nes_experiments.py --audit
../.venv/bin/python check_consistency.py
../.venv/bin/python claim_audit.py            # every MEASURED claim vs disk
../.venv/bin/python tests/test_strategy_registry.py
../.venv/bin/python tests/test_quantization_adapters.py

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
| 1 | **W4.2** key/scale recoverability | cheap; potential break of the security property |
| 2 | **W3.2** blind-patch adversary | cheapest test of the central claim |
| 3 | **W2** LWE perplexity | biggest hole; invalidates the strategy choice if it fails |
| 4 | **W3.1** cross-scheme detector | the claim's main weakness |
| 5 | **W1.1** QAE adapter + round trip | adds two strategies cheaply |
| 6 | **W1.3** strategy × model matrix (3 models) | the breadth deliverable |
| 7 | **W5** hybrids | most interesting science |
| 8 | **W6** model surgery | determines viability |
| 9 | **W7** Pareto frontier | the strongest publishable framing |
| 10 | **W1.4** consolidation | cleanup; do before W8 |
| 11 | **W8** delta productization | only after 1–8 hold |
| — | misuse assessment | gate before any distribution |
