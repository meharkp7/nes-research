#!/usr/bin/env python3
"""Requantization extension test: cycles 0-10 on one tensor per model."""
from __future__ import annotations

import sys
import json
import os
import contextlib
import io

# Suppress ALL stdout
sys.stdout = io.StringIO()

sys.path.insert(0, '/Users/meharkapoor7/nes-agent-workspace/nes-research/code/nes-llm')
sys.path.insert(0, '/Users/meharkapoor7/nes-agent-workspace/nes-research/code/nes-llm/scripts')

from scripts.run_nf4_anchor_pilot import load_tensor, requantize
from src.experiments.nf4_anchor_channel import embed, unpack, extract
from scripts.exp_detectability import METHOD, payload_for, key_for

SIZES = (10000, 50000)
SEEDS = (20261016, 20261017, 20261018)

MODELS = {
    "smollm2": {
        "checkpoint": "/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/ckpt_nf4/smollm2_clean",
        "tensor": "model.layers.11.self_attn.k_proj.weight",
    },
    "qwen15": {
        "checkpoint": "/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/ckpt_nf4/qwen15_clean",
        "tensor": "model.layers.13.self_attn.k_proj.weight",
    },
    "pythia": {
        "checkpoint": "/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/ckpt_nf4/pythia_clean",
        "tensor": "gpt_neox.layers.11.attention.dense.weight",
    },
    "falcon": {
        "checkpoint": "/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/ckpt_nf4/falcon_clean",
        "tensor": "model.layers.8.self_attn.k_proj.weight",
    },
}


def test_model(model_name, spec):
    source, state = load_tensor(spec["checkpoint"], spec["tensor"])
    raw = source.numpy().tobytes()
    cb = state.code.float().numpy()
    
    model_results = {}
    
    for size in SIZES:
        for seed in SEEDS:
            stego, _ = embed(raw, payload_for(size, seed), key_for(seed), spec["tensor"], METHOD, cb)
            payload_bytes = payload_for(size, seed)
            key = key_for(seed)
            
            cycle_results = {}
            current_raw = stego
            
            for cycle in range(11):
                try:
                    extracted = extract(current_raw, key, spec["tensor"], METHOD)
                    exact = (extracted == payload_bytes)
                except Exception:
                    exact = False
                
                cycle_results[f"cycle_{cycle}"] = bool(exact)
                
                if cycle < 10:
                    current_raw = requantize(current_raw, state)
            
            model_results[f"size_{size}_seed_{seed}"] = cycle_results
    
    return model_results


def main():
    output_dir = Path("/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/requant_10cycle")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_name, spec in MODELS.items():
        model_results = test_model(model_name, spec)
        all_results[model_name] = model_results
        
        for key, cycles in model_results.items():
            all_pass = all(cycles.values())
            print(f"  {key}: {'ALL PASS' if all_pass else 'FAIL'}", file=sys.stderr)
    
    # Save results
    output_file = output_dir / "requant_10cycle_results.json"
    output_file.write_text(json.dumps(all_results, indent=2))
    print(f"Results saved to {output_file}", file=sys.stderr)
    
    # Summary
    print("\n=== OVERALL SUMMARY ===", file=sys.stderr)
    for model_name, model_results in all_results.items():
        all_pass = all(
            all(cycles.values())
            for cycles in model_results.values()
        )
        print(f"  {model_name}: {'PASS' if all_pass else 'FAIL'}", file=sys.stderr)


if __name__ == "__main__":
    from pathlib import Path
    main()