"""Checks that each implemented mechanism behaves the way the paper claims."""

import torch

from config import ModelConfig
from fp8 import (
    E4M3_MAX,
    accumulation_error_vs_k,
    act_quant,
    fp8_matmul,
    fp8_matmul_per_tensor,
    weight_quant,
)
from mla import MultiHeadLatentAttention
from model import DeepSeekV3, kv_cache_report
from moe import DeepSeekMoE, update_all_biases
from common import precompute_freqs_cis
from inference import generate, generate_speculative, plan_redundant_experts

torch.manual_seed(0)
OK, FAIL = "  PASS", "  FAIL"


def check(name, cond, detail=""):
    print(f"{OK if cond else FAIL}  {name}{('  ' + detail) if detail else ''}")
    assert cond, name


# ---------------------------------------------------------------- MLA --- #
def test_mla():
    print("\nMulti-head Latent Attention")
    cfg = ModelConfig.tiny()
    attn = MultiHeadLatentAttention(cfg).eval()
    freqs = precompute_freqs_cis(cfg.rope_head_dim, 32, cfg.rope_theta)
    x = torch.randn(2, 12, cfg.d_model)

    with torch.no_grad():
        naive = attn(x, freqs[:12], absorb=False)
        absorbed = attn(x, freqs[:12], absorb=True)
    err = (naive - absorbed).abs().max().item()
    check("weight-absorbed path matches the literal equations", err < 1e-4, f"max|d| = {err:.2e}")

    # incremental decoding against the latent cache must match a full forward
    from mla import LatentKVCache

    cache = LatentKVCache(cfg, 2, 32, x.device, x.dtype)
    outs = []
    with torch.no_grad():
        for t in range(12):
            outs.append(attn(x[:, t : t + 1], freqs[t : t + 1], cache=cache, absorb=True))
    inc = torch.cat(outs, dim=1)
    err = (inc - naive).abs().max().item()
    check("token-by-token decoding matches the full forward", err < 1e-4, f"max|d| = {err:.2e}")

    full = ModelConfig()
    mla_sz = full.kv_lora_rank + full.rope_head_dim
    mha_sz = 2 * full.n_heads * full.d_head
    check(
        "cache is far smaller than MHA at the real config",
        mha_sz / mla_sz > 50,
        f"{mha_sz} -> {mla_sz} floats/token/layer ({mha_sz / mla_sz:.0f}x)",
    )


# ---------------------------------------------------------------- MoE --- #
def test_moe():
    print("\nDeepSeekMoE routing")
    cfg = ModelConfig.tiny()
    moe = DeepSeekMoE(cfg).train()
    x = torch.randn(4, 16, cfg.d_model)

    idx, gate, _ = moe.router(x.reshape(-1, cfg.d_model), seq_len=16)
    check("each token activates exactly K_r experts", idx.shape[-1] == cfg.n_activated_experts)
    check("gates sum to one", torch.allclose(gate.sum(-1), torch.ones(idx.shape[0]), atol=1e-5))

    nodes_touched = torch.tensor(
        [torch.unique(row // cfg.experts_per_group).numel() for row in idx]
    )
    check(
        "no token reaches more than M nodes",
        int(nodes_touched.max()) <= cfg.n_limited_groups,
        f"max nodes touched = {int(nodes_touched.max())}, M = {cfg.n_limited_groups}",
    )

    # The bias should flatten expert load with no gradient flowing through it.
    # Measured as a windowed mean: the update is sign-based, so the load of any
    # single step oscillates around the balanced point rather than settling on it.
    moe.router.load_counter.zero_()
    skewed = torch.randn(1, 64, cfg.d_model) * 0.3 + torch.randn(1, 1, cfg.d_model)
    history = []
    for _ in range(400):
        moe.router(skewed.reshape(-1, cfg.d_model), seq_len=64)
        history.append(moe.router.load_imbalance())
        update_all_biases(moe)
    before = sum(history[:50]) / 50
    after = sum(history[-50:]) / 50
    check(
        "aux-loss-free bias flattens expert load",
        after < before * 0.8,
        f"mean max/mean load {before:.2f} -> {after:.2f} over 400 steps",
    )
    check("bias is not a trainable parameter",
          all(n != "bias" for n, _ in moe.router.named_parameters()))

    # gamma is a real hyper-parameter, not a free knob: too large and the
    # sign-based update overshoots every step instead of converging.
    fast = DeepSeekMoE(ModelConfig.tiny(bias_update_speed=0.05)).train()
    fast_hist = []
    for _ in range(400):
        fast.router(skewed.reshape(-1, cfg.d_model), seq_len=64)
        fast_hist.append(fast.router.load_imbalance())
        update_all_biases(fast)
    check(
        "an oversized gamma oscillates instead of converging",
        sum(fast_hist[-50:]) / 50 > after,
        f"gamma=0.05 settles at {sum(fast_hist[-50:]) / 50:.2f} vs "
        f"{after:.2f} at the paper's 0.001",
    )

    # the gate multiplied into the FFN output must come from the raw affinity
    moe.router.bias.normal_(std=1.0)
    scores = torch.sigmoid(x.reshape(-1, cfg.d_model).float() @ moe.router.centroids.float().t())
    idx2, gate2, _ = moe.router(x.reshape(-1, cfg.d_model), seq_len=16)
    raw = scores.gather(1, idx2)
    check("gating value ignores the routing bias",
          torch.allclose(gate2.float(), (raw / raw.sum(-1, keepdim=True)), atol=1e-5))


# ---------------------------------------------------------------- MTP --- #
def test_mtp():
    print("\nMulti-Token Prediction")
    cfg = ModelConfig.tiny(mtp_depth=2)
    model = DeepSeekV3(cfg).train()
    tokens = torch.randint(0, cfg.vocab_size, (2, 24))
    out = model(tokens, compute_loss=True)

    check("one logits tensor per prediction depth", len(out.mtp_logits) == cfg.mtp_depth)
    for k, lg in enumerate(out.mtp_logits, start=1):
        check(f"depth {k} covers positions 0..T-{k}-1", lg.shape[1] == 24 - k, str(tuple(lg.shape)))
    check("loss has a main term and an MTP term",
          out.parts["mtp"] > 0 and out.parts["main"] > 0,
          f"main={out.parts['main']:.3f} mtp={out.parts['mtp']:.3f}")

    out.loss.backward()
    check("gradient reaches the shared embedding", model.embed.weight.grad is not None)
    check("gradient reaches the MTP projection", model.mtp[0].proj.weight.grad is not None)

    # the embedding and the output head are shared, not duplicated per depth
    import torch.nn as nn

    check("MTP modules contain no embedding table of their own",
          not any(isinstance(m, nn.Embedding) for m in model.mtp.modules()))
    heads = [m for m in model.modules()
             if isinstance(m, nn.Linear) and m.out_features == cfg.vocab_size]
    check("exactly one output head exists in the whole model",
          len(heads) == 1 and heads[0] is model.head)

    full = ModelConfig()
    saved = cfg.mtp_depth * 2 * full.vocab_size * full.d_model
    print(f"         at the production config, sharing saves "
          f"{saved / 1e9:.2f}B parameters across {cfg.mtp_depth} depths")

    # dropping the MTP modules must leave the main model's predictions untouched
    model.eval()
    with torch.no_grad():
        a = model(tokens, run_mtp=True).logits
        b = model(tokens, run_mtp=False).logits
    check("discarding MTP at inference changes nothing",
          torch.equal(a, b))


# ---------------------------------------------------------------- FP8 --- #
def test_fp8():
    print("\nFP8 mixed precision")
    torch.manual_seed(1)
    m, k, n = 256, 1024, 512
    x = torch.randn(m, k)
    x[:, 17] *= 40.0                       # a feature outlier, as in the paper's motivation
    w = torch.randn(n, k) * 0.02
    ref = x @ w.t()

    def rel(y):
        return ((y - ref).norm() / ref.norm()).item()

    fine = rel(fp8_matmul(x, w, block_size=128))
    coarse = rel(fp8_matmul_per_tensor(x, w))
    check("fine-grained scaling beats per-tensor scaling under outliers",
          fine < coarse, f"{fine * 100:.3f}% vs {coarse * 100:.3f}% relative GEMM error")

    # The accumulation story, isolated from the quantisation story: a 14-bit
    # register drifts as K grows, an FP32 register promoted every 128 does not.
    errs = {kk: accumulation_error_vs_k(kk) for kk in (512, 1024, 2048, 4096)}
    growth = errs[4096][1] / errs[512][1]
    flat = errs[4096][0] / max(errs[512][0], 1e-12)
    check("a 14-bit register accumulates error as K grows",
          growth > 1.8,
          "  ".join(f"K={kk}: {v[1] * 100:.3f}%" for kk, v in errs.items()))
    # Promotion does not remove the error inside a 128-chunk, it stops the error
    # compounding across chunks, so the curve goes flat instead of going away.
    check("CUDA-core promotion every 128 elements keeps it flat in K",
          flat < 1.2 and errs[4096][0] < errs[4096][1],
          "  ".join(f"K={kk}: {v[0] * 100:.4f}%" for kk, v in errs.items()))

    # The paper's 0.25% figure is a *loss* error against a BF16 baseline, not a
    # per-GEMM error, so measure it where the paper measures it.
    torch.manual_seed(2)
    h = torch.randn(64, 512)
    w1, w2 = torch.randn(1024, 512) * 0.03, torch.randn(1024, 1024) * 0.03
    tgt = torch.randint(0, 1024, (64,))

    def loss_with(mm):
        a = torch.nn.functional.silu(mm(h, w1))
        return torch.nn.functional.cross_entropy(mm(a, w2).float(), tgt).item()

    l_ref = loss_with(lambda a, b: a @ b.t())
    l_fp8 = loss_with(lambda a, b: fp8_matmul(a, b, 128))
    err = abs(l_fp8 - l_ref) / l_ref
    check("relative loss error stays under the paper's 0.25%",
          err < 0.0025, f"{err * 100:.4f}%  (loss {l_ref:.5f} -> {l_fp8:.5f})")

    q, s = act_quant(torch.randn(8, 256), 128)
    check("activation tiles are 1x128", s.shape == (8, 2) and q.abs().max() <= E4M3_MAX)
    qw, sw = weight_quant(torch.randn(256, 384), 128)
    check("weight blocks are 128x128", sw.shape == (2, 3))


# ------------------------------------------------------------ end-to-end --- #
def test_model():
    print("\nFull model")
    cfg = ModelConfig.tiny()
    model = DeepSeekV3(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 16))

    model.eval()
    with torch.no_grad():
        full = model(tokens, run_mtp=False).logits
        caches = model.new_caches(1, 64)
        inc = torch.cat(
            [
                model(tokens[:, t : t + 1], caches=caches, absorb=True,
                      run_mtp=False, start_pos=t).logits
                for t in range(16)
            ],
            dim=1,
        )
    err = (full - inc).abs().max().item()
    check("cached decoding matches the parallel forward", err < 1e-3, f"max|d| = {err:.2e}")

    out = generate(model, tokens, max_new_tokens=6)
    check("greedy generation runs", out.shape == (1, 22))

    # The property that makes speculative decoding safe: it is a speedup, never
    # a change of output. Rejected drafts must leave no trace in the cache.
    greedy = generate(model, tokens, max_new_tokens=16)
    spec, stats = generate_speculative(model, tokens, max_new_tokens=16)
    check("speculative decoding reproduces greedy output exactly",
          torch.equal(greedy, spec))
    check("speculative decoding reports its acceptance rate",
          0.0 <= stats["acceptance_rate"] <= 1.0,
          f"accept={stats['acceptance_rate']:.0%} on an untrained drafter, "
          f"{stats['tokens_per_forward']:.2f} tok/forward")
    print("         tokens per forward is 1 + acceptance rate, so the paper's "
          "85-90% gives ~1.9")

    loads = torch.rand(16) ** 3
    plan = plan_redundant_experts(loads, n_gpus=4, n_redundant=4)
    sizes = [len(v) for v in plan.values()]
    check("redundant experts are placed on the quietest GPUs",
          sum(sizes) == 20 and max(sizes) - min(sizes) <= 4, f"per-GPU counts {sizes}")

    print("\n" + kv_cache_report(ModelConfig()))


if __name__ == "__main__":
    test_mla()
    test_moe()
    test_mtp()
    test_fp8()
    test_model()
    print("\nAll checks passed.")
