"""Type definitions for NES system."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import torch

TensorType  = torch.Tensor
BitSequence = List[int]
FloatList   = List[float]


@dataclass
class ResidualTensor:
    """FP16 - NF4 residual tensor with metadata."""
    tensor:      torch.Tensor
    layer_id:    int
    module_name: str
    magnitude:   float
    entropy:     float


@dataclass
class QuantizedWeights:
    """Quantized weights in NF4 or other format."""
    weights:             torch.Tensor
    quantization_format: str
    scale_factor:        Optional[torch.Tensor] = None
    zero_point:          Optional[torch.Tensor] = None


@dataclass
class LayerProfile:
    """Quality profile for a single layer (used by QACI)."""
    layer_id:         int
    module_name:      str
    num_params:       int
    mag_mean:         float
    mag_std:          float
    mag_max:          float
    entropy:          float
    quality_score:    float
    position_bias:    float
    adjusted_quality: float


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding process."""
    total_payload_bits:  int
    embedding_strategy:  str   = 'sign'
    carrier_selection:   str   = 'magnitude'
    use_qaci:            bool  = True
    model_family: str = 'llama' # NEW
    num_hidden_layers: int = 32
    alpha:               float = 0.25
    noise_level:         float = 0.001
    min_magnitude:       float = 0.001
    percentile_threshold:int   = 25
    gamma: float = 2.5
    # W5.3 split dial: fraction of carriers carrying LWE-style parity
    # (the rest carry sign flips). Only read by SplitStrategy.
    split_fraction: float = 0.5


@dataclass
class CarrierSelection:
    """Result of carrier selection."""
    selected_indices: Dict[int, List[int]]
    quality_scores:   Dict[int, torch.Tensor]
    layer_allocation: Dict[int, int]
    total_selected:   int


@dataclass
class EmbeddingResult:
    """Result of the embedding process."""
    success:          bool
    embedded_weights: Dict[int, torch.Tensor]
    carrier_indices:  Dict[int, List[int]]
    layer_allocation: Dict[int, int]
    metadata:         Dict[str, Any] = field(default_factory=dict)
    bits_embedded:    int   = 0
    total_bits:       int   = 0
    efficiency:       float = 0.0


@dataclass
class ExtractionResult:
    """Result of the extraction process."""
    success:           bool
    recovered_payload: Optional[bytes]      = None
    recovered_bits:    Optional[List[int]]  = None
    ber:               float = 0.0
    accuracy:          float = 0.0
    metadata:          Dict[str, Any] = field(default_factory=dict)
    bits_extracted:    int = 0
    bits_correct:      int = 0
    total_bits:        int = 0


@dataclass
class CryptoKey:
    """AES-256 key with metadata."""
    key:       bytes
    key_id:    str = ""
    created_at:str = ""
    model_id:  str = ""
    metadata:  Dict[str, Any] = field(default_factory=dict)

    @property
    def key_size(self) -> int:
        return len(self.key) * 8


@dataclass
class EncryptedPayload:
    """Encrypted bitstream."""
    iv:         bytes
    ciphertext: bytes
    auth_tag:   bytes
    key_id:     str = ""


@dataclass
class FidelityReport:
    """Model fidelity after embedding."""
    embedding_applied:      bool
    perplexity_baseline:    float
    perplexity_embedded:    float
    perplexity_degradation: float
    mmlu_baseline:          float
    mmlu_embedded:          float
    mmlu_loss:              float
    status:                 str

    def is_acceptable(self, max_ppl: float = 0.02, max_acc: float = 0.01) -> bool:
        return self.perplexity_degradation <= max_ppl and self.mmlu_loss <= max_acc


@dataclass
class DetectionMetrics:
    """Steganalysis detection metrics."""
    kl_divergence:     float
    entropy_diff:      float
    detector_accuracy: float
    is_detectable:     bool
    metadata:          Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationalEnvelope:
    """Operational constraints and guarantees."""
    payload_bits_min:        int   = 50000
    payload_bits_max:        int   = 100000
    noise_tolerance_sigma:   float = 0.002
    max_ppl_degradation:     float = 0.02
    max_accuracy_loss:       float = 0.01
    max_kl_divergence:       float = 0.05
    min_detection_resistance:float = 0.55


EmbeddingStrategy      = str
CarrierSelectionMethod = str
QuantizationFormat     = str