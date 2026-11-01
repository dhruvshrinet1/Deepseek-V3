"""End-to-end training demo: all the mechanisms running together.

Trains the tiny config on a synthetic copy-and-shift task, which is learnable in
a couple of hundred steps on a CPU, and reports the three things the paper cares
about as training proceeds: the main loss, the MTP loss, and expert load balance.

    python train_demo.py
"""

import time

import torch

from config import ModelConfig
from fp8 import BF16AdamW
from inference import expert_load_profile
from model import DeepSeekV3, kv_cache_report
from moe import Router, update_all_biases


def synthetic_batch(cfg: ModelConfig, batch: int, length: int, period: int = 7):
    """A periodic sequence: token t+period is a fixed function of token t."""
    base = torch.randint(0, cfg.vocab_size // 4, (batch, period))
    reps = length // period + 2
    seq = base.repeat(1, reps)[:, :length]
    shift = torch.arange(length) % 3
    return (seq + shift) % cfg.vocab_size


def lr_at(step: int, total: int, peak: float = 2.2e-3) -> float:
    """Warmup then cosine decay to a tenth, the shape used in Section 4.2."""
    warmup = max(total // 20, 1)
    if step < warmup:
        return peak * step / warmup
    import math

    progress = (step - warmup) / max(total - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main(steps: int = 300, batch: int = 8, length: int = 64) -> None:
    torch.manual_seed(0)
    cfg = ModelConfig.tiny(mtp_depth=1, bias_update_speed=1e-3)
    model = DeepSeekV3(cfg).train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"tiny config: {n_params / 1e6:.1f}M params, {cfg.n_layers} layers, "
          f"{cfg.n_routed_experts} routed experts, {cfg.n_activated_experts} active\n")

    opt = BF16AdamW(model.parameters(), lr=2.2e-3)
    t0 = time.time()

    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps)

        tokens = synthetic_batch(cfg, batch, length)
        out = model(tokens, compute_loss=True)

        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # the bias update is a separate rule, applied after the gradient step
        imbalance = max(
            m.load_imbalance() for m in model.modules() if isinstance(m, Router)
        )
        update_all_biases(model)

        if step % 25 == 0 or step == 1:
            p = out.parts
            print(
                f"step {step:4d}  main {p['main']:.4f}  mtp {p['mtp']:.4f}  "
                f"balance {p['balance']:.2e}  worst expert load {imbalance:5.2f}x"
            )

    print(f"\ntrained in {time.time() - t0:.1f}s")

    # Does the second predicted token actually agree with the first? This is the
    # acceptance rate that speculative decoding converts into throughput.
    model.eval()
    with torch.no_grad():
        tokens = synthetic_batch(cfg, 4, length)
        out = model(tokens, run_mtp=True)
        main_next = out.logits[:, :-1].argmax(-1)          # position i -> token i+1
        draft = out.mtp_logits[0][:, :-1].argmax(-1)       # position i -> token i+2
        # the draft at i should match what the main model says about token i+2
        agree = (draft == main_next[:, 1:]).float().mean().item()
    print(f"draft agrees with the main model on {agree:.0%} of positions")

    loads = expert_load_profile(model)
    if loads is not None and loads.sum() > 0:
        print(f"expert load spread (max/mean): {loads.max() / loads.mean():.2f}x")

    print("\n" + kv_cache_report(ModelConfig()))


if __name__ == "__main__":
    main()
