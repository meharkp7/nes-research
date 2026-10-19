# StableLM-2-1.6B Detectability Report (Gate D)

Run: `stablelm_qualification/detectability_v1/`. Status: **COMPLETE**.
Protocol frozen in `QUALIFICATION_PROTOCOL_STABLELM.md` before this run.
StableLM is evaluated **separately** and is not pooled with TinyLlama/Qwen.

## 1. Provenance

| Item | Value |
|---|---|
| Model | `stabilityai/stablelm-2-1_6b` @ `f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef` |
| Clean checkpoint | `ckpt/stablelm_clean` (`bff3b65e…`) |
| Codec sha256 | `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` |
| Method | `interior_nearest` |
| Payload sizes | 10,000 and 50,000 bits |
| Seeds | 20261016, 20261017, 20261018 |
| Tensors | 6 (L0/L11/L23 × q/k/v and up/down/gate) |
| Features | the 16 predeclared block-level features |
| Blocks sampled | 2,048 per artifact (seed 424242) |
| Grouping | `stablelm::<tensor>` (GroupKFold / leave-one-tensor-out) |
| Runtime (build) | 12.3 s |

## 2. Results

| Detector | AUC | AUC 95% CI (grouped bootstrap) | TPR@1% FPR | TPR@5% FPR | Perm. p | n perm |
|---|---|---|---|---|---|---|
| Gradient Boosting | 0.5005 | [0.4997, 0.5017] | 0.0100 | 0.0500 | 0.463 | 200 |
| Logistic Regression | 0.5009 | [0.5006, 0.5016] | 0.0100 | 0.0502 | 0.318 | 200 |
| MLP (exploratory) | 0.5000 | [0.4996, 0.5004] | 0.0097 | 0.0500 | 0.483 | 200 |

Clean-vs-clean control (Gradient Boosting): AUC **0.4848**, CI [0.4444, 0.5106] —
consistent with the expected null (~0.5).

- n_examples = 86,016 block rows; **n_groups = 6** independent tensor groups.
- TPR@1% and TPR@5% are exactly at the FPR operating points (no lift above chance).
- No detector exceeds the screening thresholds.

## 3. Effect sizes

Maximum absolute Cohen's d per tensor: 0.002–0.009 (all negligible). The largest
single feature effect is `parity_imbalance` on L0 q_proj at d = −0.009. No
feature shows a practically meaningful clean-vs-embedded separation.

## 4. Block coverage (proportion of blocks touched)

Per tensor, payload size, and seed, the fraction of blocks touched ranges from
**2.7% to 32.1%**. This is recorded because low touched-fraction can dilute a
detector's aggregate signal; here even the most-touched tensors show no signal.
Full per-artifact values: `block_sparsity.json`.

## 5. Gate D verdict

**PASS (screening criterion).** All three predeclared detectors are at chance
(AUC ≈ 0.500), TPR at the FPR operating points, permutation p far from
significance, clean-vs-clean control consistent with null.

**Limitations that bound this verdict (recorded, not hidden):**
1. **Only 6 independent groups.** All StableLM attention tensors are the same
   size (4,194,304 codes), so there is no small-capacity tensor as in the
   TinyLlama/Qwen small-capacity supplement. 86,016 block rows from 6 tensors
   must not be read as 86,016 independent samples.
2. **Power.** With 6 groups the grouped bootstrap CI is narrow but the number of
   independent units is small; a subtle signal could be missed. The verdict is a
   screening pass, not a proof of undetectability.
3. **No seed-vs-seed control was run** in this StableLM dataset (only
   clean-vs-clean). Noted as not-run.
4. **Untouched-block dilution.** Tensors are touched 2.7–32.1%; the detector was
   evaluated on sampled blocks including untouched ones, matching the prior
   methodology. This is consistent with, not evidence for, undetectability.

This result does **not** overturn the prior small-capacity finding
(Logistic Regression AUC 0.5343, permutation p = 0.0050 on four groups), which
remains recorded in `RESULTS_DETECTABILITY_SMALLCAP.md` and the evidence ledger.
StableLM simply behaves differently from that small-capacity configuration.

## 6. Relation to prior Track-3 detectability results

| Model / config | Detector | AUC | Perm. p | Verdict |
|---|---|---|---|---|
| TinyLlama+Qwen, small-capacity 50k (4 groups) | Logistic Regression | 0.5343 | 0.0050 | marginal signal (screening fail) |
| **StableLM 10k+50k (6 groups)** | Gradient Boosting | 0.5005 | 0.463 | pass |
| **StableLM 10k+50k (6 groups)** | Logistic Regression | 0.5009 | 0.318 | pass |
| **StableLM 10k+50k (6 groups)** | MLP (exploratory) | 0.5000 | 0.483 | pass |

The two results are reported side by side and not pooled. The prior marginal
signal is preserved.
