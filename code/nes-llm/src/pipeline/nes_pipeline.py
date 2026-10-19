"""
NES Real Model Pipeline — correct implementation using true residuals.

Pipeline:
    Load float16 model
        → Extract true residuals R = W_fp16 - W_nf4_dequant
        → Embed bits into R → R_embedded
        → Patch: W_new = W_nf4_dequant + R_embedded
        → Save / evaluate

Extract:
    Load embedded float16 model
        → Re-extract residuals from embedded weights
        → Run DecryptPipeline
"""

import json, os, secrets
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.residual_extractor   import ResidualExtractor, LLAMA_TARGET_MODULES
from src.model.weight_patcher       import WeightPatcher
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.extraction.decrypt_pipeline    import DecryptPipeline
from src.crypto.key_manager             import KeyManager
from src.core.types                     import EmbeddingConfig


class NESRealPipeline:
    """Full NES pipeline on a real float16 LLM."""

    def __init__(
        self,
        model_id:       str,
        target_modules: List[str] = None,
        layer_range:    Optional[Tuple[int, int]] = None,
        payload_bits:   int = 50000,
        cache_dir:      Optional[str] = None,
    ):
        self.model_id       = model_id
        self.target_modules = target_modules or LLAMA_TARGET_MODULES
        self.layer_range    = layer_range
        self.payload_bits   = payload_bits
        self.cache_dir      = cache_dir
        self.patcher        = WeightPatcher()
        self.km             = KeyManager()

    def _load_model(self, model_id: str, token: str = None):
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=self.cache_dir, token=token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto",
            cache_dir=self.cache_dir,
            token=token,
        )
        model.eval()
        print(f"[NESPipeline] Loaded {model_id} "
              f"(device={next(model.parameters()).device})")
        return model, tokenizer

    def embed(
        self,
        message:    str,
        hf_token:   Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Embed message into model using true NF4 residuals.

        Returns dict with key, carrier_indices, module_names, etc.
        """
        model, tokenizer = self._load_model(self.model_id, hf_token)

        extractor = ResidualExtractor(self.target_modules, self.layer_range)
        residuals, nf4_dequant, module_refs, module_names = extractor.extract(model)

        total_capacity = sum(t.numel() for t in residuals.values())
        print(f"[NESPipeline] Capacity: {total_capacity:,} carrier positions")

        config   = EmbeddingConfig(
            total_payload_bits=min(self.payload_bits, total_capacity),
            embedding_strategy="sign",
        )
        embedder = IntelligentEmbedder(config)
        result   = embedder.embed(message, residuals)

        print(f"[NESPipeline] Embedded {result.bits_embedded:,} bits "
              f"across {sum(1 for b in result.layer_allocation.values() if b>0)} layers")

        # Patch: W_new = W_nf4_dequant + R_embedded
        self.patcher.patch(module_refs, nf4_dequant, result.embedded_residuals)

        kid = self.km.add_key(result.key, model_id=self.model_id)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            self.km.save(os.path.join(output_dir, "nes_keys.json"))
            with open(os.path.join(output_dir, "nes_carriers.json"), "w") as f:
                json.dump({
                    "key_id":          kid,
                    "carrier_indices": {str(k): v for k, v in result.carrier_indices.items()},
                    "module_names":    {str(k): v for k, v in module_names.items()},
                }, f, indent=2)
            print(f"[NESPipeline] Saved to {output_dir}/")

        return {
            "key":             result.key,
            "key_id":          kid,
            "carrier_indices": result.carrier_indices,
            "module_names":    module_names,
            "bits_embedded":   result.bits_embedded,
            "output_dir":      output_dir,
        }

    def extract(
        self,
        model_dir:       str,
        key:             bytes,
        carrier_indices: Dict[int, List[int]],
        hf_token:        Optional[str] = None,
    ) -> str:
        """Extract message from embedded float16 model."""
        model, _ = self._load_model(model_dir, hf_token)

        extractor = ResidualExtractor(self.target_modules, self.layer_range)
        residuals, _, _, _ = extractor.extract(model)

        pipeline = DecryptPipeline(key=key)
        message, stats = pipeline.run(residuals, carrier_indices)

        if not stats.get("success"):
            raise RuntimeError(f"Extraction failed: {stats.get('error')}")

        print(f"[NESPipeline] Recovered message ({len(message)} chars)")
        return message