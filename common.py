"""Shared building blocks: RMSNorm, rotary embeddings, SwiGLU feed-forward."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    original_max_seq_len: int = 4096,
) -> torch.Tensor:
    """Rotary frequencies, optionally stretched with YaRN (Section 4.3).

    DeepSeek-V3 extends 4K -> 32K -> 128K with YaRN applied *only* to the
    decoupled shared key k^R, using s = 40, alpha = 1, beta = 32.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    if scaling_factor > 1.0:
        # NTK-by-parts interpolation: low-frequency dims are interpolated,
        # high-frequency dims are left alone, with a linear ramp between.
        alpha, beta = 1.0, 32.0

        def correction_dim(rotations: float) -> float:
            return dim * math.log(original_max_seq_len / (rotations * 2 * math.pi)) / (
                2 * math.log(theta)
            )

        low = math.floor(correction_dim(beta))
        high = math.ceil(correction_dim(alpha))
        ramp = torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3),
            0.0,
            1.0,
        )
        smooth = 1.0 - ramp
        freqs = freqs / scaling_factor * (1 - smooth) + freqs * smooth

    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # (end, dim // 2), complex


def yarn_attention_scale(scaling_factor: float) -> float:
    """sqrt(t) = 0.1 * ln(s) + 1, the extra softmax scale YaRN introduces."""
    if scaling_factor <= 1.0:
        return 1.0
    return 0.1 * math.log(scaling_factor) + 1.0


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to (B, T, H, D) with freqs_cis of shape (T, D // 2)."""
    dtype = x.dtype
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xc * freqs_cis[None, :, None, :]).flatten(-2)
    return out.to(dtype)


class SwiGLUFFN(nn.Module):
    """The dense feed-forward network, and the body of every expert."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)   # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, dim, bias=False)   # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def causal_mask(q_len: int, kv_len: int, device, dtype) -> Optional[torch.Tensor]:
    """Additive mask; query i may attend to key j <= i + (kv_len - q_len)."""
    if q_len == 1:
        return None
    offset = kv_len - q_len
    i = torch.arange(q_len, device=device)[:, None]
    j = torch.arange(kv_len, device=device)[None, :]
    allowed = j <= (i + offset)
    return torch.zeros(q_len, kv_len, device=device, dtype=dtype).masked_fill(
        ~allowed, torch.finfo(dtype).min
    )
