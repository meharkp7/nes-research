# METHODS.md — Track C Method Specification

**Status:** Frozen — matches frozen codec `nf4_anchor_channel.py` (SHA-256: `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764`)  
**Date:** 2026-10-21  
**Commit:** `5b40120` (`research/contract-b-b14-and-evaluation`)

---

## 1. NF4 Quantization Layout (BitsAndBytes)

| Parameter | Value |
|-----------|-------|
| Quantization type | NF4 (NormalFloat-4) |
| Block size | 64 weights per block |
| Nested (double) quantization | True |
| Quantization storage | uint8 (4-bit packed) |
| Compute dtype | float16 |
| Quantile estimation | NormalFloat (mean=0, std=1) |

**Block structure:** Each block of 64 weights shares a single scale and zero-point. The 64 4-bit codes are packed into 32 bytes (2 codes per byte). Nested quantization quantizes the block scales themselves.

---

## 2. Carrier Eligibility: Interior Codes Only

### 2.1 Codebook Indices
NF4 codebook has 16 levels: indices 0–15.
- **Endpoints (anchors):** 0 and 15 — **never used as carriers, never rewritten**
- **Interior carriers:** indices 1–14 — eligible for payload embedding

### 2.2 Eligibility Mask
For each 64-weight block, compute a boolean mask:
```
eligible[i] = (code[i] >= 1) and (code[i] <= 14)
```
Only eligible positions participate in payload embedding.

---

## 3. Embedding Operation: Nearest-Same-Parity Interior Mapping

### 3.1 Payload Preparation
- Payload: arbitrary bit string of length `L` bits
- Framing: `MAGIC (32 bits) || length (16 bits) || payload (L bits) || checksum (16 bits CRC)`
- Total framed bits: `L + 64`
- Seeds: 20261016, 20261017, 20261018 (deterministic payload generation)
- Key derivation: `key = SHA-256("NES-development-{seed}")[:32]`

### 3.2 Carrier Selection (Deterministic)
For each embedded bit:
1. Derive carrier positions from key + tensor name + framed bit index (deterministic PRNG)
2. Only eligible (interior) positions are candidates
3. Carrier set is stable across requantization cycles (verified)

### 3.3 Embedding Operation (per carrier)
Let `c` be the original 4-bit code at a carrier position (`1 ≤ c ≤ 14`).
Let `b ∈ {0,1}` be the payload bit to embed.

**Target parity:** `p = b` (even for 0, odd for 1)

**Operation:**
```
if (c % 2) == p:
    c' = c  # already correct parity
else:
    # Move to nearest interior code with target parity
    if c == 1:  c' = 2   # only even neighbor
    elif c == 14: c' = 13  # only odd neighbor
    else:
        # Two candidates: c-1 and c+1 (both in 1..14)
        # Choose nearest by absolute difference
        if abs((c-1) - c) <= abs((c+1) - c):
            c' = c-1
        else:
            c' = c+1
```

**Key invariants:**
- `c' ∈ [1, 14]` (never 0 or 15)
- `(c' % 2) == p` (correct parity)
- `|c' - c| = 1` (minimal change)
- Endpoints 0 and 15 are **never** written

### 3.4 Baseline (for comparison): `pair_parity`
```
c' = (c & 14) | b   # Forces parity by masking LSB; can write 0 or 15
```
This baseline mutates endpoints and fails under requantization.

---

## 4. Extraction

### 4.1 Framing Verification
1. Load embedded tensor, unpack NF4 codes
2. Re-derive carrier positions from key + tensor name
3. Read parity bits at carrier positions → recover framed bitstream
3. Verify MAGIC, length, checksum
4. Return payload

### 4.2 Extraction Robustness
- Carrier positions recomputed from (stego codes, key, tensor name) — **stable under requantization**
- No clean cover or original weights needed for extraction
- Extraction fails (checksum mismatch) if artifact corrupted or wrong key

---

## 5. Requantization Protocol

### 5.1 Cycle Definition
One cycle = dequantize NF4 → float16 → quantize back to NF4 (same block size, nested stats)
- Implemented via `requantize(tensor, state)` in frozen codec
- Cycles 0–3: primary matrix (reported)
- Cycles 0–5: stress test (regression suite only)

### 5.2 Invariants Verified
| Invariant | Verification |
|-----------|--------------|
| Payload exact recovery (BER=0) | All cycles 0–3, all models, all sizes, all seeds |
| Carrier set stability | Carrier positions identical across cycles |
| Endpoint preservation | `changed_anchor_codes = 0` everywhere |
| Byte-idempotence | Identical artifact SHA across cycles 0–3 |
| Baseline failure | `pair_parity` fails from cycle 1 (checksum mismatch) |

---

## 6. Payload Parameters

| Parameter | Value |
|-----------|-------|
| Payload sizes | 1000, 5000, 10000, 20000, 50000 bits |
| Seeds | 20261016, 20261017, 20261018 |
| Framing overhead | 64 bits (MAGIC 32 + length 16 + CRC 16) |
| Seeds used | 20261016, 20261017, 20261018 |

---

## 6. Tensor Selection (6 per model)

Depth × Projection rule: early/mid/late × (attention q/k/v, mlp up/down/gate)

| Model | Layers | Tensor naming convention |
|-------|--------|--------------------------|
| SmolLM2-1.7B | 24 | `model.layers.{L}.self_attn.{q,k,v}_proj.weight`, `model.layers.{L}.mlp.{up,down,gate}_proj.weight` |
| Qwen2.5-1.5B | 28 | Same as above (Llama family) |
| Pythia-1.4b | 24 | `gpt_neox.layers.{L}.attention.query_key_value.weight` (fused QKV), `mlp.{up,down,gate}` |
| Falcon3-1B | 18 | Llama family naming, 18 layers |
| StableLM-2-1.6B | 24 | Llama family, `model.layers.{0,11,23}.self_attn.{q,k,v}_proj.weight`, `mlp.{up,down,gate}` |

---

## 7. Payload Sizes & Seeds

| Size (bits) | Seeds | Framed bits | Carrier bits needed |
|-------------|-------|-------------|---------------------|
| 1,000 | 3 | 1,064 | 1,064 |
| 5,000 | 3 | 5,064 | 5,064 |
| 10,000 | 3 | 10,064 | 10,064 |
| 20,000 | 3 | 20,064 | 20,064 |
| 50,000 | 3 | 50,064 | 50,064 |

Carrier utilization: 0.03%–1.2% of interior codes (well below capacity).

---

## 8. Frozen Codec Reference

**File:** `code/nes-llm/src/experiments/nf4_anchor_channel.py`  
**SHA-256:** `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764`

**Key functions:**
- `unpack(codes)` → unpack uint8 to 4-bit codes
- `pack(codes)` → pack 4-bit codes to uint8
- `eligible(codes)` → boolean mask of interior codes (1–14)
- `carrier_positions(codes, framed_bits, key, tensor, method)` → deterministic carrier positions
- `eligible(carrier_positions)` → filter eligible carriers
- `embed(raw, payload, key, tensor, method, codebook)` → returns (stego, info)
- `extract(stego, key, tensor, method)` → payload bytes
- `pack(codes)` / `unpack(codes)` — packing/unpacking 4-bit codes
- `requantize(tensor, state)` → dequantize + requantize cycle

**Frozen:** This file has not changed since commit `c81edda` (2026-10-17). SHA-256 verified at every pipeline stage.