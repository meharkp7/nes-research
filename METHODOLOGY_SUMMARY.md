# NES Research — Methodology Summary

**Current closeout: 2026-10-16.** Start with the [status ledger](PHASE0_STATUS_LEDGER.md) and [source-backed final claims](PHASE3_CONSOLIDATED_SEVEN_METHOD.md). The reviewed package is uncommitted. Historical experiment gates, failed runs and unresolved provenance are retained.

| Research question | Existing evidence | Scope and limitation |
|---|---|---|
| Can residual methods recover payloads? | exp18: four methods × seven models, 28/28 bit round trips at BER 0.0. Separate saved corpus artifacts cover all five residual methods on seven IDs at 36,128 bits and exact recovery. | Method/model/receiver-specific. exp18's LWE bit recovery coexists with failure of its historical sign-only production decrypt path. |
| Can tested detectors distinguish residual embedding? | exp18: LWE-inspired grid/parity accuracy 0.50 on seven models; other methods often exceed the 0.55 gate. exp7 neural sign accuracy 0.705 (FAIL). | Separately fitted per-model detectors. TinyLlama LOLO is another protocol: sign/QAE near chance, three other methods higher. No universal undetectability claim. |
| Is the grid method cryptographically secure? | exp13: public width 0.01, ignored key, keyless phase-stream BER 0.0 (FAIL). | LWE-inspired, not a formal LWE construction. AES-GCM message authentication is a separate tested property (exp27). |
| Does sign embedding survive model modifications? | exp23: 8/12 zero-BER cells including control; NF4, GPTQ, AWQ re-quantization and half-merge fail exact recovery. | One Qwen model, selected down_proj weights, specified LoRA/prune/merge/fine-tune settings. Nonzero BER does not prove useful message survival. |
| How much payload and utility were tested? | exp4: maximum tested payload 10M bits at BER 0.0; exp26: 1k–50k curve with statistical detector. exp5 signed PPL change −0.0052866%; exp15 LWE +0.0076998% Qwen / +0.0501199% Gemma. | Different protocols and payload bands; no unlimited capacity or all-task utility claim. |
| What did packed-NF4 evaluation establish? | Later reference-assisted QSE and packed-code DCE corpus reports record exact 36,128-bit recovery. 20261011 model-grouped LR/HGB ROC-AUCs are 0.5004–0.5011. | Artifact bundles, not full HF checkpoints; 4/1/2 model-ID split, family overlap and correlated blocks. Historical QSE failures and the separate 20261010 detector remain visible. |
| What remains uncertain? | Source-revision matching UNVERIFIED; new model-held-out five-method detector comparison BLOCKED by available one-model features. | Do not claim verified checkpoint independence, full reproducibility or equal validation of all seven methods. |

Validation: **419 tests passed**, 3 subtests passed, 23 warnings; **152 claims verified**; **9/9 consistency checks**, 41 artifacts verified. Logs and original artifact fingerprints are in [the closeout record](artifacts/closeout_20261016/evidence.json). The claim audit preserves neural detection failure and the unsupported exp25 H3 public-re-run hypothesis.

The two tracks remain separate. Recovery, robustness, detectability, capacity, utility and message confidentiality each require their own evidence; one successful axis does not certify another.
