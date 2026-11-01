"""DeepSeekMoE with auxiliary-loss-free load balancing (Section 2.1.2).

Three things are new here relative to a GShard-style MoE:

1. Fine-grained routed experts plus a small number of *shared* experts that
   every token passes through. Affinities use sigmoid, and the gating values of
   the selected experts are renormalised to sum to one.

2. Auxiliary-loss-free balancing. A per-expert bias b_i is added to the affinity
   *only when deciding the top-K*, never when computing the gate that multiplies
   the expert output. After each step, b_i moves down by gamma if expert i was
   overloaded and up by gamma if it was underloaded. Balance is therefore
   enforced without a gradient that fights the language-modelling objective.
   A tiny sequence-wise balance loss is kept underneath purely as a guard
   against pathological imbalance inside a single sequence.

3. Node-limited routing. Experts are laid out across nodes; each token may only
   reach at most M of them, chosen by the sum of the top K_r/M affinities on
   each node. This bounds cross-node InfiniBand traffic.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from common import SwiGLUFFN
from config import ModelConfig


class Router(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_routed = cfg.n_routed_experts
        self.top_k = cfg.n_activated_experts
        self.n_groups = cfg.n_expert_groups
        self.top_groups = cfg.n_limited_groups

        # e_i, the centroid vector of the i-th routed expert
        self.centroids = nn.Parameter(torch.empty(self.n_routed, cfg.d_model))
        nn.init.normal_(self.centroids, std=cfg.init_std)

        # b_i. A buffer, not a parameter: it is updated by a rule, not by SGD.
        self.register_buffer("bias", torch.zeros(self.n_routed))
        # running token count per expert, consumed by update_bias()
        self.register_buffer("load_counter", torch.zeros(self.n_routed))

    # ------------------------------------------------------------------ #

    def _node_limited_mask(self, biased: torch.Tensor) -> torch.Tensor:
        """Keep only experts living on the M best nodes for each token."""
        n_tokens = biased.shape[0]
        per_group = self.cfg.experts_per_group
        grouped = biased.view(n_tokens, self.n_groups, per_group)

        # "selected according to the sum of the highest K_r/M affinity scores
        #  of the experts distributed on each node"
        k_per_group = max(self.top_k // self.top_groups, 1)
        group_score = grouped.topk(k_per_group, dim=-1).values.sum(-1)       # (T, n_groups)

        keep = group_score.topk(self.top_groups, dim=-1).indices             # (T, M)
        mask = torch.zeros_like(group_score, dtype=torch.bool)
        mask.scatter_(1, keep, True)
        return mask[:, :, None].expand(-1, -1, per_group).reshape(n_tokens, self.n_routed)

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (n_tokens, d_model) -> (topk_idx, topk_gate, balance_loss)."""
        n_tokens = x.shape[0]

        # eq. (15): sigmoid affinity, not softmax
        scores = torch.sigmoid(x.float() @ self.centroids.float().t())        # (T, N_r)

        # eq. (16): the bias steers the top-K choice only
        biased = scores + self.bias

        if self.n_groups > 1 and self.top_groups < self.n_groups:
            node_mask = self._node_limited_mask(biased)
            biased = biased.masked_fill(~node_mask, float("-inf"))

        topk_idx = biased.topk(self.top_k, dim=-1).indices                    # (T, K_r)

        # eq. (13)-(14): the gate itself comes from the *unbiased* affinity
        topk_gate = scores.gather(1, topk_idx)
        topk_gate = topk_gate / topk_gate.sum(-1, keepdim=True).clamp(min=1e-20)

        if self.training:
            with torch.no_grad():
                counts = torch.bincount(topk_idx.flatten(), minlength=self.n_routed)
                self.load_counter += counts.to(self.load_counter.dtype)

        balance_loss = self._balance_loss(scores, seq_len)
        return topk_idx, topk_gate.to(x.dtype), balance_loss

    def _balance_loss(self, scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Complementary sequence-wise balance loss, eq. (17)-(20)."""
        if self.cfg.balance_alpha == 0.0 or not self.training:
            return scores.new_zeros(())

        n_tokens, n_r = scores.shape
        b = n_tokens // seq_len
        s = scores.view(b, seq_len, n_r)

        # f_i: fraction of this sequence's tokens routed to expert i, normalised
        # so that a perfectly balanced sequence gives f_i = 1. Uses raw affinity.
        chosen = s.topk(self.top_k, dim=-1).indices
        hits = torch.zeros_like(s).scatter_(-1, chosen, 1.0)
        f = hits.sum(1) * (n_r / (self.top_k * seq_len))                      # (B, N_r)

        # P_i: mean normalised affinity. This is the differentiable half.
        s_norm = s / s.sum(-1, keepdim=True).clamp(min=1e-20)
        p = s_norm.mean(1)                                                    # (B, N_r)

        return self.cfg.balance_alpha * (f.detach() * p).sum(-1).mean()

    @torch.no_grad()
    def update_bias(self) -> None:
        """Call once per optimizer step. gamma is set to 0 late in training."""
        gamma = self.cfg.bias_update_speed
        if gamma == 0.0 or self.load_counter.sum() == 0:
            self.load_counter.zero_()
            return
        mean_load = self.load_counter.mean()
        # down if overloaded, up if underloaded
        self.bias += gamma * torch.sign(mean_load - self.load_counter)
        self.load_counter.zero_()

    @torch.no_grad()
    def load_imbalance(self) -> float:
        """max_i load_i / mean load. 1.0 is perfect balance."""
        if self.load_counter.sum() == 0:
            return float("nan")
        return (self.load_counter.max() / self.load_counter.mean()).item()


class DeepSeekMoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.router = Router(cfg)
        self.routed = nn.ModuleList(
            SwiGLUFFN(cfg.d_model, cfg.expert_ffn_dim) for _ in range(cfg.n_routed_experts)
        )
        self.shared = (
            SwiGLUFFN(cfg.d_model, cfg.expert_ffn_dim * cfg.n_shared_experts)
            if cfg.n_shared_experts > 0
            else None
        )
        self.last_balance_loss: torch.Tensor = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        flat = x.reshape(-1, d)

        topk_idx, topk_gate, balance_loss = self.router(flat, seq_len=t)
        self.last_balance_loss = balance_loss

        out = torch.zeros_like(flat)
        # No token dropping: every selected (token, expert) pair is computed.
        for e in range(self.cfg.n_routed_experts):
            token_ids, slot = (topk_idx == e).nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            y = self.routed[e](flat[token_ids]) * topk_gate[token_ids, slot, None]
            out.index_add_(0, token_ids, y.to(out.dtype))

        out = out.view(b, t, d)
        if self.shared is not None:
            # eq. (12): shared experts are unconditional, every token hits them
            out = out + self.shared(x)
        return out  # the residual connection is applied by the enclosing block


def collect_balance_loss(module: nn.Module) -> torch.Tensor:
    total: Optional[torch.Tensor] = None
    for m in module.modules():
        if isinstance(m, DeepSeekMoE):
            total = m.last_balance_loss if total is None else total + m.last_balance_loss
    return total if total is not None else torch.zeros(())


def update_all_biases(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, Router):
            m.update_bias()
