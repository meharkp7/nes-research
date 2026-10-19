"""Core abstractions and base classes for NES system."""

from .types import (
    TensorType, BitSequence, FloatList,
    ResidualTensor, QuantizedWeights, LayerProfile,
    EmbeddingConfig, CarrierSelection,
    EmbeddingResult, ExtractionResult,
    CryptoKey, EncryptedPayload,
    FidelityReport, DetectionMetrics, OperationalEnvelope,
)

from .interfaces import (
    Embedder, Extractor, FeatureExtractor,
    Selector, Scorer, Cipher,
    LayerProfiler, CarrierIntelligence,
    EmbeddingPipeline, ExtractionPipeline,
)

from .config import (
    ModelConfig, EmbeddingStrategyConfig,
    CarrierSelectionConfig, QACIConfig,
    CryptoConfig, QuantizationConfig,
    EvaluationConfig, NESConfig, DEFAULT_CONFIG,
)

from .exceptions import (
    NESException, CapacityExceeded, RecoveryFailed,
    SecurityViolation, QuantizationError, FidelityError,
    EmbeddingError, ExtractionError, CryptoError,
    AuthenticationError, ConfigurationError, ModelError,
)

__all__ = [
    "TensorType", "BitSequence", "FloatList",
    "ResidualTensor", "QuantizedWeights", "LayerProfile",
    "EmbeddingConfig", "CarrierSelection",
    "EmbeddingResult", "ExtractionResult",
    "CryptoKey", "EncryptedPayload",
    "FidelityReport", "DetectionMetrics", "OperationalEnvelope",
    "Embedder", "Extractor", "FeatureExtractor",
    "Selector", "Scorer", "Cipher",
    "LayerProfiler", "CarrierIntelligence",
    "EmbeddingPipeline", "ExtractionPipeline",
    "ModelConfig", "EmbeddingStrategyConfig",
    "CarrierSelectionConfig", "QACIConfig",
    "CryptoConfig", "QuantizationConfig",
    "EvaluationConfig", "NESConfig", "DEFAULT_CONFIG",
    "NESException", "CapacityExceeded", "RecoveryFailed",
    "SecurityViolation", "QuantizationError", "FidelityError",
    "EmbeddingError", "ExtractionError", "CryptoError",
    "AuthenticationError", "ConfigurationError", "ModelError",
]