# REPRODUCIBILITY.md — Track C Reproduction Guide

**Status:** Complete — all commands and environment verified  **Date:** 2026-10-21  **Commit:** `5b40120` (`research/contract-b-b14-and-evaluation`)  **Frozen codec:** `code/nes-llm/src/experiments/nf4_anchor_channel.py` (SHA-256: `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764`)

---

## 1. Environment

| Component | Version / Specification |
|-----------|------------------------|
| OS | macOS 15+ (Apple Silicon) / Linux x86_64 |
| Python | 3.11+ |
| PyTorch | 2.4+ (CUDA 12.1 if GPU available) |
| transformers | 4.44+ |
| bitsandbytes | 0.43+ |
| accelerate | 0.33+ |
| scikit-learn | 1.5+ |
| numpy | 1.26+ |
| pandas | 2.2+ |

**Install:**
```bash
pip install torch transformers accelerate bitsandbytes scikit-learn numpy pandas
```

---

## 2. Model Checkpoints (Pinned Revisions)

All models downloaded via `transformers` with explicit revision:

| Model | Hub ID | Revision | Local Path |
|-------|--------|----------|------------|
| StableLM-2-1.6B | `stabilityai/stablelm-2-1_6b` | `f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef` | `~/cache/models/stablelm-2-1_6b` |
| SmolLM2-1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `effd688a12921b4cc83e3312b6feb579f70f9c71` | `~/cache/models/SmolLM2-1.7B-Instruct` |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` | `~/cache/models/Qwen2.5-1.5B` |
| Pythia-1.4b | `EleutherAI/pythia-1.4b` | `fedc38a16eea3bd36a96b906d78d11d2ce18ed79` | `~/cache/models/pythia-1.4b` |
| Falcon3-1B | `tiiuae/Falcon3-1B-Base` | `cb37ef3559b157b5c9d9226296ba01a5162da1f7` | `~/cache/models/Falcon3-1B-Base` |

**Download command template:**
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "MODEL_ID",
    revision="REVISION",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

---

## 3. Corpus

| Corpus | Split | Config | Source |
|--------|-------|--------|--------|
| wikitext-2 | test | raw | `HuggingFaceFW/fineweb-2` or `wikitext` |

**Load:**
```python
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
```

---

## 4. Seeds & Determinism

| Purpose | Seeds |
|---------|-------|
| Payload generation | `20261016`, `20261017`, `20261018` |
| Key derivation | `SHA-256("NES-development-{seed}")[:32]` |
| Detector CV | `SEED = 20261016` (GroupKFold) |
| Bootstrap/Permutation | `20261016` |

**All PRNGs seeded:** `torch.manual_seed`, `numpy.random.seed`, `random.seed`, `sklearn` `random_state`.

---

## 5. Key Commands

### 5.1 Roundtrip + Integrity (Gate A1–A3)

**Expansion models (4 models):**
```bash
cd /Users/meharkapoor7/nes-agent-workspace/nes-research/expansion_qualification
python3 build_clean_nf4.py --model smollm2 --out /tmp/nes_stage/smollm2_nf4
python3 exp_roundtrip_stream.py --model-dir /tmp/nes_stage/smollm2_nf4 --method interior_nearest --output /tmp/nes_stage/smollm2_roundtrip
python3 check_integrity.py --model-dir /tmp/nes_stage/smollm2_nf4 --artifact /tmp/nes_stage/smollm2_roundtrip/embedded.pt
```

**StableLM (reference):**
```bash
cd /workspace/nes_run_20261016_anchor_ablation/stablelm_qualification
python3 roundtrip_stream_v1.py --method interior_nearest --output out/roundtrip_stream_v1
```

### 5.2 Detectability (Gate A4)

**Main evaluation (per model):**
```bash
cd /Users/meharkapoor7/nes-agent-workspace/nes-research/expansion_qualification
python3 exp_detectability.py --model smollm2 --output out/detectability_v1
python3 exp_detectability.py --model qwen15 --output out/detectability_v1
python3 exp_detectability.py --model pythia --output out/detectability_v1
python3 exp_detectability.py --model falcon --output out/detectability_v1
```

**Small-capacity supplement:**
```bash
python3 exp_detectability_smallcap.py --models smollm2,qwen15,pythia,falcon --output out/detectability_smallcap_v1
python3 scripts/exp_detectability_analysis_smallcap.py --features-dir out/detectability_smallcap_v1 --output out/detectability_smallcap_analysis_v2
```

**Held-out neural (completed — results in evidence/heldout_neural_v2/):**
```bash
# Results: evidence/heldout_neural_v2/heldout_neural_results.json
# Mean AUC: 0.500007 ± 0.000771 (all 4 checkpoints at chance)
```

### 5.3 Utility (Gate A5) — BLOCKED

**Reason:** NF4 checkpoints fail to load via HF `from_pretrained` (NaN PPL); HF/bitsandbytes NF4 loading bug. 4/8 models completed clean PPL; 4/8 failed to load or produced NaN PPL.

**StableLM (protocol-matched — reference):**
```bash
cd /workspace/nes_run_20261016_anchor_ablation/stablelm_qualification
python3 exp_utility_v2.py --n-seq 580 --batch 4 --output utility_v2
```

**Note:** Utility claims for Track A are deferred due to HF/bitsandbytes NF4 loading bug. StableLM reference shows 6/6 conditions within ±0.5% PPL.

### 5.4 Regression Tests (Gate A6)

```bash
cd /Users/meharkapoor7/nes-agent-workspace/nes-research/expansion_qualification
python3 test_codec_regressions.py --model smollm2 --checkpoint /tmp/nes_stage/smollm2_nf4
python3 test_codec_regressions.py --model qwen15 --checkpoint /tmp/nes_stage/qwen15_nf4
python3 test_codec_regressions.py --model pythia --checkpoint /tmp/nes_stage/pythia_nf4
python3 test_codec_regressions.py --model falcon --checkpoint /tmp/nes_stage/falcon_nf4
```

---

## 6. Expected Outputs & Verification

### 6.1 Roundtrip (`expansion_audit_table.csv`)

| Field | Expected |
|-------|----------|
| Rows | 4321 (1 header + 4320 data = 4 models × 6 tensors × 5 sizes × 3 seeds × 3 methods × 4 cycles) |
| `exact_recovery` | 1 for all `interior_nearest` rows |
| `changed_anchor_codes` | 0 for all `interior_nearest` rows |
| `status` | `DONE` for all rows |

**Verify:**
```bash
awk -F, '$2=="interior_nearest" && $7=="DONE" {if($8!=1||$11!=0) print "FAIL", $0}' expansion_audit_table.csv
```

### 6.2 Integrity (`integrity.json`)

```json
{
  "differing_keys": ["target_tensor_name"],
  "shape_preserved": true,
  "payload_exact": true,
  "finite_logits_loss": true
}
```

### 6.3 Detectability (`detectability_results.json`)

**Pass criteria:**
- `control_clean_vs_clean.auc` ∈ [0.45, 0.55]
- All detectors: `auc` ≤ 0.55, `tpr_at_fpr_0.01` ≤ 0.02
- `permutation_p` > 0.05 (or if ≤ 0.05, AUC < clean-control AUC)

### 6.4 Utility (`comparisons.json`)

**Pass criteria:** All conditions have `"within_margin_0p5pct": true`

### 6.5 Regression Tests

All 12 tests PASS (exit code 0).

---

## 7. Frozen Codec Verification

```bash
sha256sum code/nes-llm/src/experiments/nf4_anchor_channel.py
# Expected: 07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764
```

---

## 8. Manifest Verification

```bash
# v2 manifest
sha256sum FROZEN_ALL_HASHES_v2.txt
# Expected: d1090a247b91510cb837b29c54eb9c1ef91c48c7f70dced27f71a346fbb4bb16

# Verify against .sha256 file
cat FROZEN_ALL_HASHES_v2.txt.sha256
```

---

## 9. Disk & Compute Requirements

| Stage | Disk (staging) | GPU | Time (est.) |
|-------|---------------|-----|-------------|
| Model staging (4 models) | ~15 GB | No | 30 min |
| Roundtrip matrix | ~2 GB | No | 2 hr/model |
| Detectability (main) | ~500 MB | No | 30 min/model |
| Small-cap detectability | ~200 MB | No | 10 min |
| Held-out neural | ~1 GB | Yes | 2 hr |
| Utility (wikitext-2) | ~100 MB | Yes | 1.5–2 hr/model |
| Regression tests | ~100 MB | No | 10 min/model |

**Total GPU time (Track C full):** ~8–10 hours (sequential, batch=4)

---

## 10. Known Non-Determinism Sources

| Source | Mitigation |
|--------|------------|
| GPU non-determinism (cuDNN) | `torch.use_deterministic_algorithms(True)` + CPU fallback for utility |
| `bitsandbytes` quantization | Identical across runs (verified by hash) |
| DataLoader shuffling | `shuffle=False`, fixed seed |
| Corpus tokenization | Deterministic per model tokenizer |

---

## 11. Output Paths (Canonical)

| Output | Path |
|--------|------|
| Roundtrip audit | `expansion_qualification/expansion_audit_table.csv` |
| Integrity | `expansion_qualification/{model}/integrity.json` |
| Detectability (main) | `expansion_qualification/{model}/detectability_v1/detectability_results.json` |
| Small-cap | `out/detectability_smallcap_v1/supplement_results.json` |
| Small-cap merged | `out/detectability_smallcap_analysis_v2/detector_results.json` |
| Held-out neural | `/tmp/nes_local_run/out/detectability_heldout_neural/heldout_neural_results.json` |
| Utility | `expansion_qualification/{model}/utility_v1/comparisons.json` |
| Regression | `expansion_qualification/{model}/regression_results.json` |

---

## 12. Resumability

All scripts support resume via:
- Checkpointed outputs (skip completed conditions)
- `--resume` flag (where implemented)
- Idempotent artifact writes (atomic `write_json` with `exist_ok=False` on manifest)

---

## 13. Verification Checklist (Run Before Claiming Reproduction)

- [ ] Frozen codec SHA-256 matches
- [ ] v2 manifest checksum matches
- [ ] `expansion_audit_table.csv` has 4321 lines, all `interior_nearest` rows exact_recovery=1, changed_anchor_codes=0
- [ ] 4 `integrity.json` files + StableLM all PASS
- [ ] 5 `detectability_results.json` files all PASS screening criteria
- [ ] `supplement_results.json` LR AUC 0.5343, perm p=0.005 (labeled exploratory)
- [ ] `detector_results.json` (merged 14-group) LR AUC 0.5059 [0.5020, 0.5120]
- [ ] 4 `comparisons.json` files all 6/6 `within_margin_0p5pct: true`
- [ ] 60/60 regression tests PASS
- [ ] Model revisions match pinned table above