from dataclasses import dataclass
from typing import List


@dataclass
class LayerExtractionMetadata:

    layer: int

    module: str

    positions: List[int]

    recovered_bits: List[int]

    payload_size: int

    correct_bits: int

    incorrect_bits: int

    bit_error_rate: float

    accuracy: float


@dataclass
class ExtractionResult:

    recovered_bits: List[int]

    payload_size: int

    recovered_payload_size: int

    bit_error_rate: float

    accuracy: float

    layer_metadata: List[LayerExtractionMetadata]