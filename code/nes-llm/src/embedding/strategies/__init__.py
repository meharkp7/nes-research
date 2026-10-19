"""Embedding strategy registry."""

from src.embedding.sign_strategy_v2                            import SignEmbeddingStrategy
from src.embedding.strategies.magnitude_aware_strategy         import MagnitudeAwareStrategy
from src.embedding.strategies.lwe_strategy                     import LWEStrategy
from src.embedding.strategies.neural_strategy                  import (
    NeuralStrategy, NeuralEmbeddingModel, NeuralEmbeddingTrainer
)
from src.embedding.strategies.adaptive_strategy                import AdaptiveStrategy
from src.extraction.sign_extractor                             import SignExtractor
from src.extraction.magnitude_aware_extractor                  import MagnitudeAwareExtractor
from src.extraction.lwe_extractor                              import LWEExtractor
from src.extraction.neural_extractor                           import NeuralExtractor
from src.extraction.adaptive_extractor                         import AdaptiveExtractor


STRATEGY_REGISTRY = {
    "sign":             (SignEmbeddingStrategy,  SignExtractor),
    "magnitude_aware":  (MagnitudeAwareStrategy, MagnitudeAwareExtractor),
    "lwe":              (LWEStrategy,            LWEExtractor),
    "neural":           (NeuralStrategy,         NeuralExtractor),
    "adaptive":         (AdaptiveStrategy,       AdaptiveExtractor),
}


def get_strategy(name: str, config, **kwargs):
    """
    Get (embedder, extractor) pair by strategy name.

    kwargs for specific strategies:
        lwe:      secret_key=<bytes>
        neural:   model=<NeuralEmbeddingModel>, device=<str>
        adaptive: neural_model_path=<str>, secret_key=<bytes>,
                  force_strategy=<str>
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    EmbedCls, ExtractCls = STRATEGY_REGISTRY[name]

    if name == "lwe":
        secret_key = kwargs.get("secret_key", b'\x00' * 32)
        return EmbedCls(config, secret_key=secret_key), ExtractCls

    elif name == "neural":
        model  = kwargs.get("model")
        device = kwargs.get("device", "cpu")
        if model is None:
            raise ValueError("Neural strategy requires model=<NeuralEmbeddingModel>")
        return EmbedCls(config, model=model, device=device), ExtractCls(model=model, device=device)

    elif name == "adaptive":
        neural_model_path = kwargs.get("neural_model_path")
        secret_key        = kwargs.get("secret_key", b'\x00' * 32)
        force_strategy    = kwargs.get("force_strategy")
        embedder = EmbedCls(
            config,
            neural_model_path=neural_model_path,
            secret_key=secret_key,
            force_strategy=force_strategy,
        )
        return embedder, ExtractCls

    else:
        return EmbedCls(config), ExtractCls()


__all__ = [
    "SignEmbeddingStrategy", "MagnitudeAwareStrategy",
    "LWEStrategy", "NeuralStrategy", "AdaptiveStrategy",
    "NeuralEmbeddingModel", "NeuralEmbeddingTrainer",
    "SignExtractor", "MagnitudeAwareExtractor",
    "LWEExtractor", "NeuralExtractor", "AdaptiveExtractor",
    "STRATEGY_REGISTRY", "get_strategy",
]