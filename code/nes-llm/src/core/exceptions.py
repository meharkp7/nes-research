"""Exception hierarchy for NES system."""


class NESException(Exception):
    """Base exception for all NES errors."""
    pass

class CapacityExceeded(NESException):
    """Payload size exceeds layer carrier capacity."""
    pass

class RecoveryFailed(NESException):
    """Extraction/recovery failed (BER > threshold)."""
    pass

class SecurityViolation(NESException):
    """Security constraint violated (KL divergence too high)."""
    pass

class QuantizationError(NESException):
    """Quantization operation failed."""
    pass

class FidelityError(NESException):
    """Model fidelity constraint violated."""
    pass

class EmbeddingError(NESException):
    """Embedding operation failed."""
    pass

class ExtractionError(NESException):
    """Extraction operation failed."""
    pass

class CryptoError(NESException):
    """Encryption/decryption failed."""
    pass

class AuthenticationError(NESException):
    """AES-GCM authentication tag verification failed."""
    pass

class ConfigurationError(NESException):
    """Configuration is invalid."""
    pass

class ModelError(NESException):
    """Model loading or handling failed."""
    pass