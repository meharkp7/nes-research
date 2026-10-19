"""Abstract base classes for NES system."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
import torch

from .types import (
    EmbeddingResult, ExtractionResult, EmbeddingConfig,
    CryptoKey, EncryptedPayload, CarrierSelection,
    TensorType, BitSequence,
)


class Embedder(ABC):
    @abstractmethod
    def embed(
        self,
        residuals:        Dict[int, TensorType],
        bits:             BitSequence,
        selector_indices: Dict[int, List[int]],
    ) -> EmbeddingResult:
        pass

    @abstractmethod
    def get_bit_for_residual(self, residual: float, bit: int) -> float:
        pass


class Extractor(ABC):
    @abstractmethod
    def extract(
        self,
        weights:         Dict[int, TensorType],
        carrier_indices: Dict[int, List[int]],
    ) -> List[int]:
        pass

    @abstractmethod
    def get_bit_from_residual(self, residual: float) -> int:
        pass


class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, tensor: TensorType, window_size: int = 32) -> TensorType:
        pass


class Selector(ABC):
    @abstractmethod
    def select(self, scores: TensorType, num_carriers: int) -> List[int]:
        pass


class Scorer(ABC):
    @abstractmethod
    def score(self, features: TensorType, layer_quality: float = 1.0) -> TensorType:
        pass


class Cipher(ABC):
    @abstractmethod
    def encrypt(self, plaintext: bytes) -> EncryptedPayload:
        pass

    @abstractmethod
    def decrypt(self, payload: EncryptedPayload) -> bytes:
        pass


class LayerProfiler(ABC):
    @abstractmethod
    def profile_layer(
        self,
        layer_id:     int,
        residuals:    TensorType,
        module_name:  str,
        model_depth:  int,
    ):
        pass


class CarrierIntelligence(ABC):
    @abstractmethod
    def select_carriers(
        self,
        residuals:          Dict[int, TensorType],
        total_payload_bits: int,
        config:             EmbeddingConfig,
    ) -> CarrierSelection:
        pass


class EmbeddingPipeline(ABC):
    @abstractmethod
    def embed_message(
        self,
        message: bytes,
        model:   Any,
        config:  EmbeddingConfig,
    ) -> Tuple[EmbeddingResult, CryptoKey]:
        pass


class ExtractionPipeline(ABC):
    @abstractmethod
    def extract_message(
        self,
        model:           Any,
        key:             CryptoKey,
        carrier_indices: Dict[int, List[int]],
    ) -> Tuple[bytes, ExtractionResult]:
        pass