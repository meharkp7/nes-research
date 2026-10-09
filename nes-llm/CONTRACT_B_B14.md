# Contract B B1.4 — 10,000-bit artifact-only NF4 pilot

## Scope and caveats

B1.4 tests whether a 10,000-bit application payload can be embedded in packed NF4 codes of one selected tensor and recovered from the serialized artifact using a fixed test key and protocol, without original FP16 weights, residual cache, or delta sidecar. The key and synthetic payload are test fixtures. The truncated SHA-256 envelope checksum detects accidental corruption but is not a cryptographic authentication mechanism. Do not claim confidentiality, stealth, utility preservation, robustness, or novelty from this pilot.

## Run from `nes-llm/`

Sender:

```bash
../.venv/bin/python scripts/contract_b_nf4_b14.py embed \
  --original ../cache/contract_b_nf4_probe_retry \
  --output-dir ../cache/contract_b_nf4_b14_10k
```

Inspect `../cache/contract_b_nf4_b14_10k_b14_sender_report.json`, then copy `payload_sha256` into:

```bash
../.venv/bin/python scripts/contract_b_nf4_b14.py receive \
  --stego-dir ../cache/contract_b_nf4_b14_10k \
  --output-payload ../cache/contract_b_nf4_b14_recovered.bin \
  --expected-sha256 PASTE_SENDER_PAYLOAD_SHA256_HERE
```

The receiver runs in a separate process and reloads the stego model on CPU by default. Do not use `--skip-model-reload` for an end-to-end PASS.

## Expected accounting

- Application payload: 10,000 bits = 1,250 bytes.
- Envelope: 4-byte magic + 4-byte length + payload + 8-byte truncated SHA-256 = 1,266 bytes.
- Carrier positions: 1,266 × 8 = 10,128 packed NF4 codes.
- The low bit of each selected code carries one bit; `code // 2` is preserved.

## Automated tests

From `nes-llm/`:

```bash
../.venv/bin/python -m unittest discover -s tests -p 'test_contract_b_nf4_b14.py' -v
```

Tests cover deterministic payload and envelope accounting, unique/prefix-stable carrier selection, nibble ordering, exact 10k round trip, pair-ID invariants, wrong-key rejection, corruption rejection, and synthetic SafeTensors parsing. They do not replace the actual cached-model run.

## Remaining phases

1. Run B1.4 on the local Qwen cache and retain both reports.
2. Add fresh-process negative tests and an artifact manifest/hash.
3. Capacity sweep at 1k/10k/25k/50k bits, repeated for reproducibility.
4. Model utility: original-versus-stego perplexity and task-level checks.
5. Detectability: baselines, held-out evaluation, confidence intervals.
6. Robustness: reload, fine-tuning, LoRA, pruning, merges, re-quantization; preserve failures.
7. Multi-model evaluation after the single-model path is stable.
8. Literature review, ablations, claim audit, reproducibility package, paper writing.

## Execution boundary

GitHub source access does not provide access to the user's local model cache or Mac runtime. The actual Qwen NF4 reload run must be executed locally; source review, synthetic tests, tracked artifact review and analysis can be done separately.
