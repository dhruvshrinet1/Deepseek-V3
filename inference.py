"""Inference: latent-cache decoding, MTP speculative decoding, redundant experts.

Section 5.4.3 reports an 85-90% acceptance rate for the second token and about
1.8x tokens per second. The draft here is free in the sense that the MTP module
is one extra transformer block, already trained, and it runs on the same hidden
state the main model just produced.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from model import DeepSeekV3
from moe import Router


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    probs = F.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sp, si = probs.sort(dim=-1, descending=True)
        keep = (sp.cumsum(-1) - sp) < top_p
        sp = sp * keep
        sp = sp / sp.sum(-1, keepdim=True)
        return si.gather(-1, torch.multinomial(sp, 1))
    return torch.multinomial(probs, 1)


@torch.no_grad()
def generate(
    model: DeepSeekV3,
    prompt: torch.Tensor,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Plain autoregressive decoding against the latent KV cache."""
    model.eval()
    b = prompt.shape[0]
    caches = model.new_caches(b, prompt.shape[1] + max_new_tokens)
    out = model(prompt, caches=caches, absorb=True, run_mtp=False)
    nxt = _sample(out.logits[:, -1], temperature, top_p)
    tokens = [nxt]

    for _ in range(max_new_tokens - 1):
        pos = caches[0].length
        out = model(nxt, caches=caches, absorb=True, run_mtp=False, start_pos=pos)
        nxt = _sample(out.logits[:, -1], temperature, top_p)
        tokens.append(nxt)

    return torch.cat([prompt] + tokens, dim=1)


@torch.no_grad()
def generate_speculative(
    model: DeepSeekV3,
    prompt: torch.Tensor,
    max_new_tokens: int = 32,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Greedy decoding with the depth-1 MTP module as the drafter.

    Each round:
      1. the main model's last hidden state gives token t1 and is also the input
         to the MTP module, which drafts t2;
      2. both are fed back in a single forward pass, which simultaneously
         verifies t2 and produces the next t1.
    A correct draft advances the sequence by two tokens for one forward pass.
    """
    assert len(model.mtp) >= 1, "speculative decoding needs at least one MTP module"
    model.eval()
    b = prompt.shape[0]
    assert b == 1, "this reference loop keeps the bookkeeping to batch size 1"

    caches = model.new_caches(b, prompt.shape[1] + 2 * max_new_tokens + 4)
    mtp_caches = model.new_caches(b, prompt.shape[1] + 2 * max_new_tokens + 4)

    seq = prompt
    proposed = accepted = forwards = 0

    def step(chunk: torch.Tensor, start: int):
        h = model.backbone(chunk, caches=caches, start_pos=start, absorb=True)
        return h, model.head(model.norm(h))

    pos = 0
    h, logits = step(seq, pos)
    pos = caches[0].length
    forwards += 1

    while seq.shape[1] - prompt.shape[1] < max_new_tokens:
        t1 = logits[:, -1:].argmax(-1)                           # main model's token
        seq = torch.cat([seq, t1], dim=1)

        # draft: feed the last hidden state plus the embedding of t1
        mtp = model.mtp[0]
        freqs = model.freqs_cis[pos - 1 : pos]
        h_draft = mtp(h[:, -1:], model.embed(t1), freqs)
        t2 = model.head(model.norm(h_draft))[:, -1:].argmax(-1)  # drafted token
        proposed += 1

        # verify t2 and get the next t1 in one pass over [t1, t2]
        h, logits = step(torch.cat([t1, t2], dim=1), pos)
        pos = caches[0].length
        forwards += 1

        verified = logits[:, 0:1].argmax(-1)                     # what t2 should be
        if torch.equal(verified, t2):
            seq = torch.cat([seq, t2], dim=1)
            accepted += 1
            # logits[:, 1] already conditions on t2, so the next round is free
            logits = logits[:, 1:]
            h = h[:, 1:]
        else:
            # roll the cache back past the rejected draft and continue from t1
            caches_rollback(caches, 1)
            pos -= 1
            logits = logits[:, 0:1]
            h = h[:, 0:1]

    stats = {
        "acceptance_rate": accepted / max(proposed, 1),
        "tokens_per_forward": (seq.shape[1] - prompt.shape[1]) / max(forwards, 1),
        "forwards": float(forwards),
    }
    return seq[:, : prompt.shape[1] + max_new_tokens], stats


def caches_rollback(caches, n: int) -> None:
    for c in caches:
        c.length = max(c.length - n, 0)


# ---------------------------------------------------------------------- #
# Redundant expert deployment (Section 3.4.1)
# ---------------------------------------------------------------------- #


def plan_redundant_experts(
    loads: torch.Tensor,
    n_gpus: int,
    n_redundant: int,
) -> Dict[int, List[int]]:
    """Decide which high-load experts to duplicate and where to put the copies.

    The real system recomputes this every ~10 minutes from live serving
    statistics. The rule: duplicate the busiest experts, then place each copy on
    whichever GPU is currently carrying the least traffic, so the duplicate
    actually relieves the hot spot instead of creating a new one.
    """
    n_experts = loads.numel()
    base = n_experts // n_gpus
    placement: Dict[int, List[int]] = defaultdict(list)
    gpu_load = torch.zeros(n_gpus)

    for e in range(n_experts):
        gpu = e // base
        placement[gpu].append(e)
        gpu_load[gpu] += loads[e]

    hottest = loads.argsort(descending=True)[:n_redundant]
    for e in hottest.tolist():
        target = int(gpu_load.argmin())
        placement[target].append(e)
        # a duplicated expert splits its traffic across its copies
        copies = sum(e in v for v in placement.values())
        gpu_load[target] += loads[e] / copies

    return dict(placement)


def expert_load_profile(model: DeepSeekV3) -> Optional[torch.Tensor]:
    """Per-expert token counts accumulated since the last bias update."""
    stacks = [m.load_counter.clone() for m in model.modules() if isinstance(m, Router)]
    return torch.stack(stacks).sum(0) if stacks else None
