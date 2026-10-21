"""
Track B: DCE End-to-End Evaluation on NF4 Checkpoints
"""

import sys
sys.path.insert(0, '/Users/meharkapoor7/nes-agent-workspace/nes-research/code/nes-llm')
sys.path.insert(0, '/Users/meharkapoor7/nes-agent-workspace/nes-research/code/nes-llm/scripts')

from src.experiments.nf4_anchor_channel import unpack, embed, extract
import hashlib
import hmac
import numpy as np
import json
from pathlib import Path

METHOD = "interior_nearest"


class DCECodec:
    def __init__(self, master_key: bytes, context: bytes = b""):
        self.master_key = master_key
        self.context = context
        self.grid_width = self._derive_grid_width()
        self.grid_offset = self._derive_grid_offset()
    
    def _derive_grid_width(self) -> int:
        h = hmac.new(self.master_key, self.context + b"width", hashlib.sha256).digest()
        return (h[0] % 5) + 2
    
    def _derive_grid_offset(self) -> int:
        h = hmac.new(self.master_key, self.context + b"offset", hashlib.sha256).digest()
        return h[0] % self.grid_width
    
    def _derive_tensor_key(self, tensor_name: str) -> bytes:
        return hmac.new(
            self.master_key, 
            self.context + tensor_name.encode(), 
            hashlib.sha256
        ).digest()[:32]
    
    def embed(self, raw: bytes, payload: bytes, tensor_name: str, cb: np.ndarray) -> tuple:
        tensor_key = self._derive_tensor_key(tensor_name)
        return embed(raw, payload, tensor_key, tensor_name, METHOD, cb)
    
    def extract(self, raw: bytes, tensor_name: str) -> bytes:
        tensor_key = self._derive_tensor_key(tensor_name)
        return extract(raw, tensor_key, tensor_name, METHOD)


# Models to test
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
    "mistral": {
        "checkpoint": "/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/ckpt_nf4/mistral_clean",
        "tensor": "model.layers.11.self_attn.k_proj.weight",
    },
    "gemma": {
        "checkpoint": "/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/ckpt_nf4/gemma_clean",
        "tensor": "model.layers.21.self_attn.k_proj.weight",
    },
    "qwen": {
        "checkpoint": "/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/ckpt_nf4/qwen_clean",
        "tensor": "model.layers.13.self_attn.k_proj.weight",
    },
    "phi3": {
        "checkpoint": "/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/ckpt_nf4/phi3_clean",
        "tensor": "model.layers.11.self_attn.qkv_proj.weight",
    },
}


def test_model(model_name, spec):
    """Test DCE on a single model checkpoint."""
    print(f"\n=== Testing {model_name} ===")
    
    # Import here to avoid issues
    from scripts.run_nf4_anchor_pilot import load_tensor
    
    source, state = load_tensor(spec["checkpoint"], spec["tensor"])
    raw = source.numpy().tobytes()
    cb = state.code.float().numpy()
    
    master_key = hashlib.sha256(b"DCE-master-key-20261022").digest()
    context = b"DCE-eval-20261022"
    dce = DCECodec(master_key, context)
    
    results = {
        "model": model_name,
        "tensor": spec["tensor"],
        "grid_width": dce.grid_width,
        "grid_offset": dce.grid_offset,
        "tests": []
    }
    
    # Test multiple payload sizes and seeds
    for size in (10000, 50000):
        for seed in (20261016, 20261017, 20261018):
            payload = np.random.default_rng(seed).bytes(size // 8)
            
            # Embed
            stego, _ = dce.embed(raw, payload, spec["tensor"], cb)
            
            # Extract with correct key
            extracted = dce.extract(stego, spec["tensor"])
            correct_match = (extracted[:len(payload)] == payload)
            
            # Test wrong master key
            wrong_master = hashlib.sha256(b"wrong-master-key").digest()
            from track_b_evaluation import DCECodec as DCECodec2
            dce_wrong = DCECodec2(wrong_master, context)
            try:
                extracted_wrong = dce_wrong.extract(stego, spec["tensor"])
                wrong_match = (extracted_wrong[:len(payload)] == payload)
            except ValueError:
                wrong_match = False
            
            # Test keyless
            dce_nokey = DCECodec2(b"\x00" * 32, context)
            try:
                extracted_nokey = dce_nokey.extract(stego, spec["tensor"])
                nokey_match = (extracted_nokey[:len(payload)] == payload)
            except ValueError:
                nokey_match = False
            
            test_result = {
                "size": size,
                "seed": seed,
                "correct_key": correct_match,
                "wrong_key": wrong_match,
                "keyless": nokey_match,
                "pass": correct_match and not wrong_match and not nokey_match
            }
            results["tests"].append(test_result)
            
            status = "PASS" if test_result["pass"] else "FAIL"
            print(f"  size={size}, seed={seed}: {status} (correct={correct_match}, wrong={wrong_match}, keyless={nokey_match})")
    
    all_pass = all(t["pass"] for t in results["tests"])
    results["overall"] = "PASS" if all_pass else "FAIL"
    print(f"  Overall: {results['overall']}")
    
    return results


def main():
    output_dir = Path("/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/dce_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_name, spec in MODELS.items():
        try:
            results = test_model(model_name, spec)
            all_results[model_name] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[model_name] = {"error": str(e)}
    
    # Save results
    output_file = output_dir / "dce_evaluation_results.json"
    output_file.write_text(json.dumps(all_results, indent=2))
    print(f"\n=== Results saved to {output_file} ===")
    
    # Summary
    print("\n=== TRACK B DCE EVALUATION SUMMARY ===")
    for model_name, results in all_results.items():
        if "overall" in results:
            print(f"  {model_name}: {results['overall']}")
        else:
            print(f"  {model_name}: ERROR - {results.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
