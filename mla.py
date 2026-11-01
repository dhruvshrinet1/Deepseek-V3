"""Multi-head Latent Attention (Section 2.1.1).

The idea: instead of caching one key and one value vector per head per token,
compress keys and values *jointly* into a single low-rank latent c^{KV} of width
d_c, and carry position information in one extra decoupled key k^R that is shared
across all heads. Only those two things are cached during generation.

    DeepSeek-V3 config:  n_h = 128, d_h = 128, d_c = 512, d_h^R = 64
    standard MHA cache:  2 * n_h * d_h   = 32768 floats / token / layer
    MLA cache:           d_c + d_h^R     =   576 floats / token / layer   (57x less)

Two execution paths are provided and are numerically equivalent:

  * `absorb=False` -- literal transcription of equations (1)-(11). Up-projects
    the cached latents back to per-head keys and values.
  * `absorb=True`  -- the form you actually want at decode time. W^UK is folded
    into the query and W^UV into the output projection, so the big per-head keys
    and values are never materialised at all; attention is computed directly
    against the d_c-wide latent.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import RMSNorm, apply_rope, causal_mask, yarn_attention_scale
from config import ModelConfig


class LatentKVCache:
    """Holds exactly the two blue-boxed tensors of Figure 2: c^{KV}_t and k^R_t."""

    def __init__(self, cfg: ModelConfig, batch_size: int, max_len: int, device, dtype):
        self.c_kv = torch.zeros(batch_size, max_len, cfg.kv_lora_rank, device=device, dtype=dtype)
        self.k_rope = torch.zeros(batch_size, max_len, cfg.rope_head_dim, device=device, dtype=dtype)
        self.length = 0

    def append(self, c_kv: torch.Tensor, k_rope: torch.Tensor) -> int:
        b, t, _ = c_kv.shape
        s = self.length
        self.c_kv[:b, s : s + t] = c_kv
        self.k_rope[:b, s : s + t] = k_rope
        self.length = s + t
        return self.length

    def view(self, batch_size: int):
        return self.c_kv[:batch_size, : self.length], self.k_rope[:batch_size, : self.length]

    def bytes_per_token(self, n_layers: int, bytes_per_elem: int = 2) -> int:
        return (self.c_kv.shape[-1] + self.k_rope.shape[-1]) * n_layers * bytes_per_elem


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_rope = cfg.rope_head_dim
        self.d_c = cfg.kv_lora_rank
        self.d_cq = cfg.q_lora_rank

        # --- key/value path: eq. (1)-(5) ---
        self.w_dkv = nn.Linear(cfg.d_model, self.d_c, bias=False)              # W^{DKV}
        self.w_kr = nn.Linear(cfg.d_model, self.d_rope, bias=False)            # W^{KR}
        self.kv_norm = RMSNorm(self.d_c, cfg.norm_eps)                         # extra RMSNorm at the bottleneck
        self.w_uk = nn.Linear(self.d_c, self.n_heads * self.d_head, bias=False)  # W^{UK}
        self.w_uv = nn.Linear(self.d_c, self.n_heads * self.d_head, bias=False)  # W^{UV}

        # --- query path: eq. (6)-(9) ---
        self.w_dq = nn.Linear(cfg.d_model, self.d_cq, bias=False)              # W^{DQ}
        self.q_norm = RMSNorm(self.d_cq, cfg.norm_eps)
        self.w_uq = nn.Linear(self.d_cq, self.n_heads * self.d_head, bias=False)   # W^{UQ}
        self.w_qr = nn.Linear(self.d_cq, self.n_heads * self.d_rope, bias=False)   # W^{QR}

        # --- output: eq. (11) ---
        self.w_o = nn.Linear(self.n_heads * self.d_head, cfg.d_model, bias=False)  # W^O

        # 1 / sqrt(d_h + d_h^R), times the YaRN mscale when the context is stretched
        self.softmax_scale = (self.d_head + self.d_rope) ** -0.5
        self.softmax_scale *= yarn_attention_scale(cfg.rope_scaling_factor) ** 2

    # ------------------------------------------------------------------ #

    def _project(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """Everything that depends only on the *new* tokens."""
        b, t, _ = x.shape

        c_kv = self.kv_norm(self.w_dkv(x))                                  # (B, T, d_c)
        k_rope = apply_rope(self.w_kr(x).view(b, t, 1, self.d_rope), freqs_cis)
        k_rope = k_rope.view(b, t, self.d_rope)                             # shared across heads

        c_q = self.q_norm(self.w_dq(x))                                     # (B, T, d'_c)
        q_c = self.w_uq(c_q).view(b, t, self.n_heads, self.d_head)
        q_r = apply_rope(self.w_qr(c_q).view(b, t, self.n_heads, self.d_rope), freqs_cis)
        return c_kv, k_rope, q_c, q_r

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[LatentKVCache] = None,
        absorb: bool = False,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        c_kv, k_rope, q_c, q_r = self._project(x, freqs_cis)

        if cache is not None:
            cache.append(c_kv, k_rope)
            c_kv_all, k_rope_all = cache.view(b)
        else:
            c_kv_all, k_rope_all = c_kv, k_rope
        s = c_kv_all.shape[1]

        if absorb:
            # Fold W^UK into the query: q_c . k_c = (W^UK^T q_c) . c^{KV}
            w_uk = self.w_uk.weight.view(self.n_heads, self.d_head, self.d_c)
            q_latent = torch.einsum("bthd,hdc->bthc", q_c, w_uk)            # (B, T, H, d_c)
            scores = torch.einsum("bthc,bsc->bhts", q_latent, c_kv_all)
            scores = scores + torch.einsum("bthr,bsr->bhts", q_r, k_rope_all)
        else:
            k_c = self.w_uk(c_kv_all).view(b, s, self.n_heads, self.d_head)
            k = torch.cat([k_c, k_rope_all[:, :, None, :].expand(b, s, self.n_heads, self.d_rope)], -1)
            q = torch.cat([q_c, q_r], -1)
            scores = torch.einsum("bthd,bshd->bhts", q, k)

        scores = scores * self.softmax_scale
        mask = causal_mask(t, s, x.device, scores.dtype)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores.float(), dim=-1).to(x.dtype)

        if absorb:
            # Fold W^UV into the output: sum_j a_j v_j = W^UV (sum_j a_j c^{KV}_j)
            ctx_latent = torch.einsum("bhts,bsc->bthc", attn, c_kv_all)      # (B, T, H, d_c)
            w_uv = self.w_uv.weight.view(self.n_heads, self.d_head, self.d_c)
            out = torch.einsum("bthc,hdc->bthd", ctx_latent, w_uv)
        else:
            v = self.w_uv(c_kv_all).view(b, s, self.n_heads, self.d_head)
            out = torch.einsum("bhts,bshd->bthd", attn, v)

        return self.w_o(out.reshape(b, t, self.n_heads * self.d_head))
