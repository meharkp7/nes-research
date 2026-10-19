"""
Model Tensor Cache
==================

Production-grade disk cache for expensive NES preprocessing.

For every transformer layer we cache:

    1. FP16 reference weights
    2. Dequantized NF4 weights
    3. Quantization residual

Residual definition:

    R = W_FP16 - W_NF4_dequantized

Directory structure:

    cache/
    └── models/
        └── <model-name>/
            ├── metadata.json
            ├── layer_0000.pt
            ├── layer_0001.pt
            ├── ...
            └── layer_N.pt

The tensors are ALWAYS stored on CPU.

They are moved to the requested runtime device only when loaded.

This keeps the persistent cache independent of MPS/CPU.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import torch


class ModelTensorCache:
    """
    Persistent cache for model preprocessing tensors.

    The cache is model-specific and configuration-specific.

    A layer is considered valid only if:
        - its cache file exists
        - it can be loaded successfully
        - required tensors are present
        - tensor sizes are consistent
    """

    CACHE_VERSION = "1.0"

    def __init__(
        self,
        model_id: str,
        cache_root: str = "cache/models",
        quantization_type: str = "nf4",
        use_double_quant: bool = True,
        compute_dtype: str = "float16",
    ):
        self.model_id = model_id
        self.cache_root = Path(cache_root)

        self.quantization_type = quantization_type
        self.use_double_quant = use_double_quant
        self.compute_dtype = compute_dtype

        # ----------------------------------------------------------
        # Create a filesystem-safe model directory name.
        # ----------------------------------------------------------

        self.model_name = self._sanitize_model_id(model_id)

        self.model_cache_dir = (
            self.cache_root / self.model_name
        )

        self.model_cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_file = (
            self.model_cache_dir / "metadata.json"
        )

    # ==============================================================
    # PATH HELPERS
    # ==============================================================

    @staticmethod
    def _sanitize_model_id(model_id: str) -> str:
        """
        Convert HuggingFace model ID into a filesystem-safe name.

        Example:

            meta-llama/Llama-3.1-8B
                ↓
            meta-llama__llama-3.1-8b
        """

        safe_name = model_id.strip().lower()

        safe_name = re.sub(
            r"[^a-zA-Z0-9._-]+",
            "__",
            safe_name,
        )

        return safe_name

    def _layer_path(self, layer_id: int) -> Path:
        """
        Return cache path for one layer.
        """

        return (
            self.model_cache_dir
            / f"layer_{layer_id:04d}.pt"
        )

    # ==============================================================
    # METADATA
    # ==============================================================

    def _configuration(self) -> dict:
        """
        Configuration that determines whether a cache is compatible.
        """

        return {
            "cache_version": self.CACHE_VERSION,
            "model_id": self.model_id,
            "quantization_type": self.quantization_type,
            "use_double_quant": self.use_double_quant,
            "compute_dtype": self.compute_dtype,
        }

    def _write_metadata(self):
        """
        Atomically write metadata.

        A temporary file is used first so an interrupted write does
        not leave a corrupted metadata.json.
        """

        metadata = self._configuration()

        temp_file = self.metadata_file.with_suffix(
            ".tmp"
        )

        with open(temp_file, "w") as f:
            json.dump(
                metadata,
                f,
                indent=4,
            )

        temp_file.replace(
            self.metadata_file
        )

    def _metadata_matches(self) -> bool:
        """
        Check whether existing cache metadata matches the
        current model/configuration.
        """

        if not self.metadata_file.exists():
            return False

        try:
            with open(self.metadata_file, "r") as f:
                stored = json.load(f)

        except (OSError, json.JSONDecodeError):
            return False

        expected = self._configuration()

        return stored == expected

    # ==============================================================
    # LAYER VALIDATION
    # ==============================================================

    def validate_layer(self, layer_id: int) -> bool:
        """
        Check whether a cached layer is safe to reuse.

        Returns:
            True  -> cache exists and appears valid
            False -> recomputation required
        """

        if not self._metadata_matches():
            return False

        layer_file = self._layer_path(layer_id)

        if not layer_file.exists():
            return False

        try:
            data = torch.load(
                layer_file,
                map_location="cpu",
                weights_only=True,
            )

            required_keys = {
                "residual",
                "fp16_weight",
                "nf4_dequantized",
            }

            if not required_keys.issubset(
                data.keys()
            ):
                return False

            residual = data["residual"]
            fp16 = data["fp16_weight"]
            nf4 = data["nf4_dequantized"]

            # ------------------------------------------------------
            # All tensors must have matching number of elements.
            # ------------------------------------------------------

            if residual.numel() != fp16.numel():
                return False

            if residual.numel() != nf4.numel():
                return False

            # ------------------------------------------------------
            # Cache tensors must be CPU tensors.
            # ------------------------------------------------------

            if residual.device.type != "cpu":
                return False

            if fp16.device.type != "cpu":
                return False

            if nf4.device.type != "cpu":
                return False

            return True

        except Exception:
            # Any corrupted/unreadable cache is treated as a miss.
            return False

    # ==============================================================
    # SAVE
    # ==============================================================

    def save_layer(
        self,
        layer_id: int,
        residual: torch.Tensor,
        fp16_weight: torch.Tensor,
        nf4_dequantized: torch.Tensor,
    ):
        """
        Save one layer to disk.

        Tensors are detached and moved to CPU before serialization.

        The write is atomic:
            layer_x.pt.tmp
                    ↓
            layer_x.pt

        This prevents partially written cache files from being
        mistaken for valid cache entries.
        """

        layer_file = self._layer_path(
            layer_id
        )

        temp_file = layer_file.with_suffix(
            ".tmp"
        )

        # ----------------------------------------------------------
        # Validate before writing.
        # ----------------------------------------------------------

        if residual.numel() != fp16_weight.numel():
            raise ValueError(
                f"Layer {layer_id}: residual and FP16 "
                f"tensor sizes do not match."
            )

        if residual.numel() != nf4_dequantized.numel():
            raise ValueError(
                f"Layer {layer_id}: residual and NF4 "
                f"tensor sizes do not match."
            )

        # ----------------------------------------------------------
        # Always persist CPU tensors.
        # ----------------------------------------------------------

        payload = {
            "residual": residual.detach().cpu(),
            "fp16_weight": fp16_weight.detach().cpu(),
            "nf4_dequantized": (
                nf4_dequantized.detach().cpu()
            ),
        }

        # ----------------------------------------------------------
        # Atomic save.
        # ----------------------------------------------------------

        torch.save(
            payload,
            temp_file,
        )

        temp_file.replace(
            layer_file
        )

        # ----------------------------------------------------------
        # Metadata is created/updated after successful layer save.
        # ----------------------------------------------------------

        if not self.metadata_file.exists():
            self._write_metadata()

    # ==============================================================
    # LOAD
    # ==============================================================

    def load_layer(
        self,
        layer_id: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Load one cached layer.

        Returns:

            residual,
            fp16_weight,
            nf4_dequantized

        The tensors are moved to `device` after loading.
        """

        if not self.validate_layer(layer_id):
            raise RuntimeError(
                f"Layer {layer_id} cache is missing or invalid: "
                f"{self._layer_path(layer_id)}"
            )

        layer_file = self._layer_path(
            layer_id
        )

        data = torch.load(
            layer_file,
            map_location="cpu",
            weights_only=True,
        )

        residual = data["residual"].to(
            device
        )

        fp16_weight = data["fp16_weight"].to(
            device
        )

        nf4_dequantized = data[
            "nf4_dequantized"
        ].to(device)

        return (
            residual,
            fp16_weight,
            nf4_dequantized,
        )

    # ==============================================================
    # CACHE INFORMATION
    # ==============================================================

    def cached_layers(self) -> list:
        """
        Return all layer IDs currently present and valid.
        """

        layers = []

        for layer_file in sorted(
            self.model_cache_dir.glob(
                "layer_*.pt"
            )
        ):

            match = re.match(
                r"layer_(\d+)\.pt",
                layer_file.name,
            )

            if match is None:
                continue

            layer_id = int(
                match.group(1)
            )

            if self.validate_layer(
                layer_id
            ):
                layers.append(layer_id)

        return layers

    def is_complete(
        self,
        num_layers: int,
    ) -> bool:
        """
        Check whether every expected transformer layer
        has a valid cached representation.
        """

        if not self._metadata_matches():
            return False

        for layer_id in range(num_layers):

            if not self.validate_layer(
                layer_id
            ):
                return False

        return True

    # ==============================================================
    # CACHE SUMMARY
    # ==============================================================

    def summary(self) -> Dict:
        """
        Return cache status information.
        """

        layers = self.cached_layers()

        return {
            "model_id": self.model_id,
            "cache_directory": str(
                self.model_cache_dir
            ),
            "cache_version": self.CACHE_VERSION,
            "quantization_type": (
                self.quantization_type
            ),
            "double_quant": (
                self.use_double_quant
            ),
            "compute_dtype": (
                self.compute_dtype
            ),
            "cached_layers": layers,
            "num_cached_layers": len(layers),
        }

    # ==============================================================
    # CACHE CLEANUP
    # ==============================================================

    def clear_layer(
        self,
        layer_id: int,
    ):
        """
        Delete one cached layer.
        """

        layer_file = self._layer_path(
            layer_id
        )

        if layer_file.exists():
            layer_file.unlink()

    def clear(self):
        """
        Delete the complete model cache.

        Use carefully.
        """

        if not self.model_cache_dir.exists():
            return

        for file in self.model_cache_dir.iterdir():

            if file.is_file():
                file.unlink()

        # Keep the model directory itself.
        self.model_cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )