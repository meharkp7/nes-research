"""Configuration management for NES system."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any
import json


@dataclass
class ModelConfig:
    model_id:             str = "meta-llama/Llama-3-8B"
    num_layers:           int = 32
    hidden_dim:           int = 4096
    intermediate_dim:     int = 14336
    num_heads:            int = 32
    num_kv_heads:         int = 8
    head_dim:             int = 128
    vocab_size:           int = 128256
    dtype:                str = "float16"
    quantization_format:  str = "nf4"


@dataclass
class EmbeddingStrategyConfig:
    strategy:               str   = "sign"
    alpha:                  float = 0.25
    min_magnitude:          float = 0.001
    use_magnitude_weighting:bool  = True


@dataclass
class CarrierSelectionConfig:
    method:               str            = "magnitude"
    percentile_threshold: int            = 25
    use_qaci:             bool           = True
    layer_aware:          bool           = True
    adaptive_threshold:   bool           = True
    random_seed:          Optional[int]  = None


@dataclass
class QACIConfig:
    enabled:              bool  = True
    early_layer_factor:   float = 0.65
    late_layer_factor:    float = 0.65
    middle_layer_factor:  float = 1.0
    use_entropy_weighting:bool  = True
    use_variance_weighting:bool = True


@dataclass
class CryptoConfig:
    algorithm:     str = "aes"
    key_size:      int = 256
    cipher_mode:   str = "gcm"
    iv_size:       int = 12
    auth_tag_size: int = 16


@dataclass
class QuantizationConfig:
    format:                   str  = "nf4"
    block_size:               int  = 256
    use_double_quantization:  bool = True
    compute_dtype:            str  = "float16"
    storage_dtype:            str  = "nf4"


@dataclass
class EvaluationConfig:
    max_ppl_degradation:  float = 0.02
    max_accuracy_loss:    float = 0.01
    max_kl_divergence:    float = 0.05
    max_detector_accuracy:float = 0.55
    noise_tolerance_sigma:float = 0.002
    min_recovery_accuracy:float = 0.95
    perplexity_dataset:   str   = "wikitext-2"
    mmlu_shots:           int   = 5
    gsm8k_shots:          int   = 8
    eval_batch_size:      int   = 32
    num_eval_samples:     int   = 1024


@dataclass
class NESConfig:
    model:              ModelConfig              = field(default_factory=ModelConfig)
    embedding_strategy: EmbeddingStrategyConfig  = field(default_factory=EmbeddingStrategyConfig)
    carrier_selection:  CarrierSelectionConfig   = field(default_factory=CarrierSelectionConfig)
    qaci:               QACIConfig               = field(default_factory=QACIConfig)
    crypto:             CryptoConfig             = field(default_factory=CryptoConfig)
    quantization:       QuantizationConfig       = field(default_factory=QuantizationConfig)
    evaluation:         EvaluationConfig         = field(default_factory=EvaluationConfig)
    total_payload_bits: int            = 50000
    expected_noise_sigma:float         = 0.001
    use_gpu:            bool           = True
    device:             str            = "mps:0"
    verbose:            bool           = True
    random_seed:        Optional[int]  = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'NESConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            model=              ModelConfig(**data.get('model', {})),
            embedding_strategy= EmbeddingStrategyConfig(**data.get('embedding_strategy', {})),
            carrier_selection=  CarrierSelectionConfig(**data.get('carrier_selection', {})),
            qaci=               QACIConfig(**data.get('qaci', {})),
            crypto=             CryptoConfig(**data.get('crypto', {})),
            quantization=       QuantizationConfig(**data.get('quantization', {})),
            evaluation=         EvaluationConfig(**data.get('evaluation', {})),
            total_payload_bits= data.get('total_payload_bits', 50000),
            expected_noise_sigma=data.get('expected_noise_sigma', 0.001),
            use_gpu=            data.get('use_gpu', True),
            device=             data.get('device', 'mps:0'),
            verbose=            data.get('verbose', True),
            random_seed=        data.get('random_seed', None),
        )

    @classmethod
    def for_llama3_8b(cls) -> 'NESConfig':
        config = cls()
        config.model = ModelConfig(model_id="meta-llama/Llama-3-8B", num_layers=32)
        return config

    @classmethod
    def for_mistral_7b(cls) -> 'NESConfig':
        config = cls()
        config.model = ModelConfig(model_id="mistralai/Mistral-7B-v0.1", num_layers=32)
        return config

    @classmethod
    def for_testing(cls) -> 'NESConfig':
        config = cls()
        config.total_payload_bits          = 10000
        config.evaluation.num_eval_samples = 256
        return config


DEFAULT_CONFIG = NESConfig()