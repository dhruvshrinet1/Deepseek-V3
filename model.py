"""The DeepSeek-V3 model: transformer blocks, MTP modules, and the loss.

Multi-Token Prediction (Section 2.2) is implemented here rather than in its own
file because it shares the embedding layer, the output head and the block type
with the main model, and that sharing is the point.

Unlike Gloeckle et al., who predict D extra tokens in parallel from independent
heads, each MTP module here runs *sequentially* and keeps the full causal chain:

    h'^k_i = M_k [ RMSNorm(h^{k-1}_i) ; RMSNorm(Emb(t_{i+k})) ]
    h^k    = TRM_k(h'^k)
    P^k_{i+k+1} = OutHead(h^k_i)

Depth k therefore predicts token i+k+1 while having seen a representation that
already consumed everything up to i+k. At inference the modules can be dropped
entirely, or reused as speculative-decoding drafters.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import RMSNorm, SwiGLUFFN, precompute_freqs_cis
from config import ModelConfig
from mla import LatentKVCache, MultiHeadLatentAttention
from moe import DeepSeekMoE, collect_balance_loss


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = MultiHeadLatentAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        # "We substitute all FFNs except for the first three layers with MoE layers."
        self.is_moe = layer_idx >= cfg.n_dense_layers
        self.ffn = DeepSeekMoE(cfg) if self.is_moe else SwiGLUFFN(cfg.d_model, cfg.ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[LatentKVCache] = None,
        absorb: bool = False,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis, cache=cache, absorb=absorb)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MTPModule(nn.Module):
    """One prediction depth. Holds a projection and a single transformer block."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.norm_h = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_emb = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)   # M_k
        self.block = TransformerBlock(cfg, layer_idx)

    def forward(self, h_prev: torch.Tensor, emb_future: torch.Tensor,
                freqs_cis: torch.Tensor) -> torch.Tensor:
        h = self.proj(torch.cat([self.norm_h(h_prev), self.norm_emb(emb_future)], dim=-1))
        return self.block(h, freqs_cis)


@dataclass
class ModelOutput:
    logits: torch.Tensor
    mtp_logits: List[torch.Tensor]
    loss: Optional[torch.Tensor] = None
    parts: Optional[Dict[str, float]] = None


class DeepSeekV3(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(TransformerBlock(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # MTP modules share self.embed and self.head with the main model. With
        # DualPipe the shallowest and deepest layers sit on the same pipeline
        # rank, so this sharing is physical: one copy of the parameters and one
        # copy of their gradients.
        self.mtp = nn.ModuleList(
            MTPModule(cfg, cfg.n_layers - 1) for _ in range(cfg.mtp_depth)
        )

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                cfg.rope_head_dim,
                cfg.max_seq_len,
                cfg.rope_theta,
                cfg.rope_scaling_factor,
                cfg.rope_original_max_seq_len,
            ),
            persistent=False,
        )
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    # ------------------------------------------------------------------ #

    def backbone(
        self,
        tokens: torch.Tensor,
        caches: Optional[List[LatentKVCache]] = None,
        absorb: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        t = tokens.shape[1]
        freqs = self.freqs_cis[start_pos : start_pos + t]
        h = self.embed(tokens)
        for i, layer in enumerate(self.layers):
            h = layer(h, freqs, cache=None if caches is None else caches[i], absorb=absorb)
        return h

    def forward(
        self,
        tokens: torch.Tensor,
        compute_loss: bool = False,
        run_mtp: bool = True,
        caches: Optional[List[LatentKVCache]] = None,
        absorb: bool = False,
        start_pos: int = 0,
    ) -> ModelOutput:
        """`tokens` is the full sequence; the loss is next-token prediction on it."""
        b, t = tokens.shape
        h = self.backbone(tokens, caches, absorb, start_pos)
        logits = self.head(self.norm(h))

        mtp_logits: List[torch.Tensor] = []
        mtp_losses: List[torch.Tensor] = []

        if run_mtp and len(self.mtp) > 0:
            h_prev = h
            for k, module in enumerate(self.mtp, start=1):
                if t - k < 1:
                    break
                # depth k consumes h^{k-1}_i together with Emb(t_{i+k}); both
                # run over positions i = 0 .. T-k-1
                h_in = h_prev[:, : t - k]
                emb_future = self.embed(tokens[:, k:])
                h_k = module(h_in, emb_future, self.freqs_cis[start_pos : start_pos + t - k])
                lg = self.head(self.norm(h_k))         # shared output head
                mtp_logits.append(lg)
                h_prev = h_k

                if compute_loss:
                    # position i predicts token i+k+1, so the last slot has no label
                    pred = lg[:, :-1]
                    tgt = tokens[:, k + 1 :]
                    mtp_losses.append(
                        F.cross_entropy(
                            pred.reshape(-1, pred.shape[-1]).float(), tgt.reshape(-1)
                        )
                    )

        loss, parts = None, None
        if compute_loss:
            main = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                tokens[:, 1:].reshape(-1),
            )
            balance = collect_balance_loss(self)
            mtp = (
                torch.stack(mtp_losses).mean() * self.cfg.mtp_loss_weight
                if mtp_losses
                else logits.new_zeros(())
            )
            loss = main + mtp + balance
            parts = {
                "main": main.item(),
                "mtp": float(mtp.detach()),
                "balance": float(balance.detach()),
                "total": loss.item(),
            }

        return ModelOutput(logits=logits, mtp_logits=mtp_logits, loss=loss, parts=parts)

    # ------------------------------------------------------------------ #

    def new_caches(self, batch_size: int, max_len: int, device=None, dtype=torch.float32):
        device = device or next(self.parameters()).device
        return [
            LatentKVCache(self.cfg, batch_size, max_len, device, dtype)
            for _ in range(self.cfg.n_layers)
        ]

    def kv_cache_report(self) -> str:
        return kv_cache_report(self.cfg)


def kv_cache_report(cfg: Optional[ModelConfig] = None) -> str:
    """Cache footprint per token, the headline reason MLA exists."""
    c = cfg or ModelConfig()
    mla = (c.kv_lora_rank + c.rope_head_dim) * c.n_layers * 2
    mha = 2 * c.n_heads * c.d_head * c.n_layers * 2
    gqa8 = 2 * 8 * c.d_head * c.n_layers * 2
    ctx = 128 * 1024
    return (
        f"KV cache per token (bf16, {c.n_layers} layers, {c.n_heads} heads)\n"
        f"  standard MHA : {mha / 1024:8.1f} KiB   "
        f"{mha * ctx / 2**30:7.1f} GiB at 128K context\n"
        f"  GQA, 8 groups: {gqa8 / 1024:8.1f} KiB   "
        f"{gqa8 * ctx / 2**30:7.1f} GiB\n"
        f"  MLA          : {mla / 1024:8.1f} KiB   "
        f"{mla * ctx / 2**30:7.1f} GiB   ({mha / mla:.0f}x smaller than MHA)"
    )
