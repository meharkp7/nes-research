"""
Environment detection for the NES experiment suite.

Records the runtime facts that the research results depend on, so every
artifact is reproducible (§27) and so device-specific failure modes
(§26, MPS ``RuntimeError: Invalid buffer size``) are visible up front
rather than halfway through a multi-hour run.
"""

import os
import platform
import sys
from typing import Any, Dict

# Must be set before torch is imported anywhere in the suite.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def detect_device() -> str:
    """Return ``cuda``, ``mps`` or ``cpu``.

    The suite never assumes a device. The NF4 dequantization path keeps
    packed weights on CPU and must not push giant dequantized tensors to
    MPS, so the reported device describes where models execute, not where
    dequantization happens.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def software_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
    }

    for name in ("transformers", "bitsandbytes", "accelerate", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not_installed"

    return versions


def warnings_for_device(device: str) -> list:
    """Known platform caveats, recorded alongside every artifact."""
    notes = []

    if device == "mps":
        notes.append(
            "Apple Silicon/MPS: NF4 dequantization must stay on CPU. "
            "Moving large dequantized tensors to MPS raises "
            "RuntimeError: Invalid buffer size."
        )
        notes.append(
            "MPS has no native int8/uint8 dequantize; "
            "PYTORCH_ENABLE_MPS_FALLBACK is set to 1."
        )

    if device == "cpu":
        notes.append(
            "CPU-only execution: residual extraction and PPL are "
            "substantially slower than MPS/CUDA."
        )

    if device == "cuda":
        notes.append(
            "bitsandbytes must match the local CUDA version; "
            "mismatched builds fail at dequantize_4bit."
        )

    return notes


def describe() -> Dict[str, Any]:
    device = detect_device()

    return {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback_enabled": os.environ.get(
            "PYTORCH_ENABLE_MPS_FALLBACK"
        ),
        "torch_threads": torch.get_num_threads(),
        "software_versions": software_versions(),
        "platform_notes": warnings_for_device(device),
    }


def assert_ready() -> Dict[str, Any]:
    """Verify the suite can actually run, raising if a hard dep is absent.

    Called by the runner before any expensive model load so a missing
    package fails in seconds instead of minutes.
    """
    info = describe()

    missing = [
        name
        for name, version in info["software_versions"].items()
        if name in ("torch", "transformers") and version == "not_installed"
    ]

    if missing:
        raise RuntimeError(
            "Missing required package(s): " + ", ".join(missing)
        )

    return info