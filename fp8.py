"""FP8 mixed-precision training, emulated (Section 3.3).

Real FP8 training needs Hopper tensor cores. What matters conceptually is
reproducible in fp32, so everything here is *fake quantisation*: values are
rounded onto the FP8 grid and immediately widened back, which reproduces the
numerics exactly while running anywhere.

Four ideas from the paper:

1. Fine-grained quantisation. Scales are per 1x128 activation tile (one token,
   128 channels) and per 128x128 weight block, instead of one scale per tensor.
   A single outlier can then only wreck its own tile.

2. Increasing accumulation precision. H800 tensor cores keep roughly 14 bits
   while accumulating FP8 products, which costs ~2% relative error at K=4096.
   DeepSeek accumulate inside the tensor core for N_c = 128 elements only, then
   promote the partial sum to an FP32 register on a CUDA core. `fp8_matmul`
   walks K in chunks of N_c and does exactly that, applying the per-group scales
   at the promotion point (which is also where dequantisation becomes free).

3. E4M3 everywhere. Prior work used E4M3 forward / E5M2 backward; fine-grained
   grouping already shares exponent range within a tile, so the extra mantissa
   bit is worth more than the extra exponent bit.

4. Online quantisation. Scales are computed from the current tile's amax rather
   than from a delayed history of past maxima.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

E4M3_MAX = 448.0
E5M2_MAX = 57344.0


def _round_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round onto the E4M3 grid (1 sign, 4 exponent, 3 mantissa bits)."""
    if hasattr(torch, "float8_e4m3fn"):
        try:
            return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).to(x.dtype)
        except (RuntimeError, TypeError):
            pass
    sign = torch.sign(x)
    a = x.abs().clamp(max=E4M3_MAX)
    exp = torch.floor(torch.log2(a.clamp(min=1e-30))).clamp(min=-6)  # min normal exponent
    step = torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype), exp - 3)
    return sign * torch.round(a / step).mul(step).clamp(max=E4M3_MAX)


def _truncate_mantissa(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Emulate a tensor core that only keeps `bits` mantissa bits when adding."""
    sign = torch.sign(x)
    a = x.abs()
    exp = torch.floor(torch.log2(a.clamp(min=1e-30)))
    step = torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype), exp - bits)
    return sign * torch.round(a / step) * step


def act_quant(
    x: torch.Tensor, block_size: int = 128, power_of_two_scale: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per 1x128 tile: one token, 128 channels. Returns (grid values, scales)."""
    *lead, k = x.shape
    assert k % block_size == 0, f"inner dim {k} must be a multiple of {block_size}"
    v = x.float().reshape(-1, k // block_size, block_size)
    amax = v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / E4M3_MAX
    if power_of_two_scale:
        # used for activations feeding the Linear after attention, and for the
        # pre-dispatch activations, so that re-tiling introduces no extra error
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    q = _round_to_e4m3(v / scale)
    return q.reshape(*lead, k), scale.reshape(*lead, k // block_size)


def weight_quant(w: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per 128x128 block: 128 input channels by 128 output channels."""
    n, k = w.shape
    assert n % block_size == 0 and k % block_size == 0
    v = w.float().reshape(n // block_size, block_size, k // block_size, block_size)
    amax = v.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale = amax / E4M3_MAX
    q = _round_to_e4m3(v / scale)
    return q.reshape(n, k), scale.reshape(n // block_size, k // block_size)


def _mma_accumulate(
    xq: torch.Tensor,
    wq: torch.Tensor,
    register_mantissa_bits: Optional[int],
    mma_group: int = 32,
) -> torch.Tensor:
    """Accumulate a chunk of FP8 products the way a tensor core would.

    "After aligning 32 mantissa products by right-shifting based on the maximum
    exponent, the Tensor Core only uses the highest 14 bits of each mantissa
    product for addition ... The accumulation of addition results into registers
    also employs 14-bit precision."

    With `register_mantissa_bits=None` the running sum is exact, which is what
    an FP32 CUDA-core register gives you.
    """
    if register_mantissa_bits is None:
        return xq @ wq.t()
    acc = torch.zeros(xq.shape[0], wq.shape[0], dtype=torch.float32, device=xq.device)
    for s in range(0, xq.shape[1], mma_group):
        acc = acc + xq[:, s : s + mma_group] @ wq[:, s : s + mma_group].t()
        acc = _truncate_mantissa(acc, register_mantissa_bits)
    return acc


def fp8_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    block_size: int = 128,
    register_mantissa_bits: Optional[int] = None,
    power_of_two_act_scale: bool = False,
) -> torch.Tensor:
    """y = x @ w.T, fine-grained FP8 with promotion to FP32 every N_c elements.

    The loop over k-blocks *is* the promotion: inside one N_c chunk the tensor
    core accumulates at whatever precision it has, and at the chunk boundary the
    partial sum moves to an FP32 register on a CUDA core, where the two per-group
    scaling factors are applied as a free dequantisation.
    """
    *lead, k = x.shape
    x2 = x.reshape(-1, k)
    n = w.shape[0]

    xq, xs = act_quant(x2, block_size, power_of_two_act_scale)
    wq, ws = weight_quant(w, block_size)

    out = torch.zeros(x2.shape[0], n, dtype=torch.float32, device=x.device)
    for kb in range(k // block_size):
        sl = slice(kb * block_size, (kb + 1) * block_size)
        partial = _mma_accumulate(xq[:, sl], wq[:, sl], register_mantissa_bits)
        w_scale = ws[:, kb].repeat_interleave(block_size)[:n]
        out += partial * xs[:, kb : kb + 1] * w_scale[None, :]

    return out.reshape(*lead, n).to(x.dtype)


def fp8_matmul_per_tensor(
    x: torch.Tensor,
    w: torch.Tensor,
    register_mantissa_bits: Optional[int] = None,
) -> torch.Tensor:
    """The baseline the paper improves on: one scale per tensor, no promotion.

    "the input distribution is aligned to the representable range of the FP8
    format by scaling the maximum absolute value of the input tensor to the
    maximum representable value of FP8 ... This method makes low-precision
    training highly sensitive to activation outliers."

    Per-group scaling and CUDA-core promotion go together: without a point in
    the loop where the partial sum leaves the tensor core, there is nowhere to
    apply a per-group scale. So this path accumulates the whole of K in one
    register, which is what `register_mantissa_bits` then degrades.
    """
    *lead, k = x.shape
    x2 = x.reshape(-1, k).float()
    xs = x2.abs().amax().clamp(min=1e-12) / E4M3_MAX
    ws = w.abs().amax().clamp(min=1e-12) / E4M3_MAX
    xq = _round_to_e4m3(x2 / xs)
    wq = _round_to_e4m3(w.float() / ws)
    acc = _mma_accumulate(xq, wq, register_mantissa_bits)
    return (acc * xs * ws).reshape(*lead, w.shape[0]).to(x.dtype)


def accumulation_error_vs_k(
    k: int,
    m: int = 128,
    n: int = 128,
    register_mantissa_bits: int = 13,
    seed: int = 0,
) -> Tuple[float, float]:
    """Relative error of the FP8 accumulation alone, with and without promotion.

    Both runs quantise identically, so the difference isolates the register
    precision from the quantisation error. Returns (promoted, unpromoted).
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g)
    w = torch.randn(n, k, generator=g)
    xq = _round_to_e4m3(x / (x.abs().amax() / E4M3_MAX))
    wq = _round_to_e4m3(w / (w.abs().amax() / E4M3_MAX))

    exact = xq @ wq.t()
    promoted = sum(
        _mma_accumulate(xq[:, s : s + 128], wq[:, s : s + 128], register_mantissa_bits)
        for s in range(0, k, 128)
    )
    unpromoted = _mma_accumulate(xq, wq, register_mantissa_bits)
    rel = lambda y: ((y - exact).norm() / exact.norm()).item()
    return rel(promoted), rel(unpromoted)


class _FP8LinearFn(torch.autograd.Function):
    """Fprop, Dgrad and Wgrad all in FP8, as in Figure 6."""

    @staticmethod
    def forward(ctx, x, w, block_size, pow2):
        ctx.save_for_backward(x, w)
        ctx.block_size, ctx.pow2 = block_size, pow2
        return fp8_matmul(x, w, block_size, power_of_two_act_scale=pow2)

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        bs = ctx.block_size
        g = grad_out.reshape(-1, grad_out.shape[-1])
        # Dgrad: dL/dx = grad @ w   -- the activation backward pass
        grad_x = fp8_matmul(g, w.t().contiguous(), bs).reshape(x.shape)
        # Wgrad: dL/dw = grad.T @ x -- the weight backward pass
        grad_w = fp8_matmul(g.t().contiguous(), x.reshape(-1, x.shape[-1]).t().contiguous(), bs)
        return grad_x, grad_w, None, None


class FP8Linear(nn.Module):
    """Drop-in nn.Linear whose three GEMMs run in FP8.

    The components the paper deliberately leaves in BF16/FP32 -- embeddings, the
    output head, MoE gating, normalisation and attention operators -- should keep
    using plain nn.Linear. Master weights stay FP32 here, as they do in the paper.
    """

    def __init__(self, in_features: int, out_features: int, block_size: int = 128,
                 power_of_two_scale: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.006)
        self.block_size = block_size
        self.power_of_two_scale = power_of_two_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FP8LinearFn.apply(x, self.weight, self.block_size, self.power_of_two_scale)


class BF16AdamW(torch.optim.Optimizer):
    """AdamW with BF16 moments and FP32 master weights (Section 3.3.3).

    "We adopt the BF16 data format instead of FP32 to track the first and second
    moments, without incurring observable performance degradation. However, the
    master weights and gradients are still retained in FP32."
    """

    def __init__(self, params, lr=2.2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    state["v"] = torch.zeros_like(p, dtype=torch.bfloat16)
                state["step"] += 1
                t = state["step"]
                g = p.grad.float()                       # gradients stay FP32
                m = state["m"].float().mul_(b1).add_(g, alpha=1 - b1)
                v = state["v"].float().mul_(b2).addcmul_(g, g, value=1 - b2)
                state["m"], state["v"] = m.bfloat16(), v.bfloat16()
                denom = (v / (1 - b2 ** t)).sqrt_().add_(group["eps"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.addcdiv_(m / (1 - b1 ** t), denom, value=-group["lr"])
        return loss
