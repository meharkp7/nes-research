"""Small pure-PyTorch codec helpers for packed BitsAndBytes NF4 experiments.

The nibble convention matches the repository's audited SafeTensors/BitsAndBytes
pilot: even value index is the high nibble; odd value index is the low nibble.
"""
from __future__ import annotations

import torch


def unpack_codes(packed: torch.Tensor, n_values: int) -> list[int]:
    if n_values < 0 or n_values > packed.numel() * 2:
        raise ValueError("n_values must fit the packed tensor")
    raw = packed.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()
    return [((raw[i // 2] >> 4) & 15) if i % 2 == 0 else (raw[i // 2] & 15)
            for i in range(n_values)]


def pack_codes(codes: list[int]) -> torch.Tensor:
    if len(codes) % 2:
        raise ValueError("packed NF4 code count must be even")
    if any(not isinstance(code, int) or code < 0 or code > 15 for code in codes):
        raise ValueError("NF4 codes must be integers in [0, 15]")
    raw = bytearray(len(codes) // 2)
    for i in range(0, len(codes), 2):
        raw[i // 2] = ((codes[i] & 15) << 4) | (codes[i + 1] & 15)
    return torch.tensor(list(raw), dtype=torch.uint8)


def tensor_dequant(codes: list[int], codebook: torch.Tensor,
                   absmax: torch.Tensor, blocksize: int) -> torch.Tensor:
    if blocksize < 1:
        raise ValueError("blocksize must be positive")
    cb = codebook.detach().float().cpu().flatten()
    scales = absmax.detach().float().cpu().flatten()
    if cb.numel() != 16:
        raise ValueError("NF4 codebook must contain 16 values")
    if codes and (max(codes) > 15 or min(codes) < 0):
        raise ValueError("NF4 codes must be in [0, 15]")
    required_scales = (len(codes) + blocksize - 1) // blocksize
    if required_scales > scales.numel():
        raise ValueError("absmax scales do not cover the code sequence")
    return torch.tensor([float(cb[c]) * float(scales[i // blocksize])
                         for i, c in enumerate(codes)], dtype=torch.float32)
