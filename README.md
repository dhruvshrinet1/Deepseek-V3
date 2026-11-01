# DeepSeek-V3 innovations, implemented

A reference implementation in PyTorch of the mechanisms introduced in the
DeepSeek-V3 technical report (arXiv:2412.19437). Everything runs on a CPU at a
small config, and every claim the code makes about itself is checked by
`test_innovations.py`.

```
pip install torch
python test_innovations.py     # verifies each mechanism
python train_demo.py           # trains the tiny config end to end, ~1 min
```

## Where each idea lives

| Paper | Idea | Code |
|---|---|---|
| 2.1.1 | Multi-head Latent Attention, low rank joint KV compression, decoupled RoPE key | `mla.py` |
| 2.1.2 | DeepSeekMoE, shared plus fine grained routed experts, sigmoid affinity | `moe.py` |
| 2.1.2 | Auxiliary loss free load balancing via a routing bias | `moe.py`, `Router.update_bias` |
| 2.1.2 | Complementary sequence wise balance loss | `moe.py`, `Router._balance_loss` |
| 2.1.2 | Node limited routing, at most M nodes per token | `moe.py`, `Router._node_limited_mask` |
| 2.2 | Multi-Token Prediction with a preserved causal chain | `model.py`, `MTPModule` |
| 3.3.1 | FP8 mixed precision, which operators stay high precision | `fp8.py`, `FP8Linear` |
| 3.3.2 | Fine grained 1x128 and 128x128 quantisation, E4M3 everywhere, online scales | `fp8.py` |
| 3.3.2 | Promotion to CUDA cores every N_c = 128 for accurate accumulation | `fp8.py`, `fp8_matmul` |
| 3.3.3 | BF16 optimizer moments with FP32 master weights | `fp8.py`, `BF16AdamW` |
| 3.4.1 | Redundant expert deployment | `inference.py`, `plan_redundant_experts` |
| 4.3 | YaRN context extension applied to the decoupled key only | `common.py` |
| 5.4.3 | MTP reused for speculative decoding | `inference.py` |

Not implemented, because they are distributed systems work rather than model
code: DualPipe scheduling, the custom cross node all to all kernels with warp
specialisation, and the prefill/decode deployment split. The shapes those
optimise are visible in the code, but reproducing them needs a real cluster.

## What the tests actually show

MLA has two execution paths that agree to 3e-7. The literal path transcribes
equations 1 to 11. The absorbed path folds `W^UK` into the query and `W^UV` into
the output projection so the per head keys and values are never materialised,
and attention runs directly against the 512 wide latent. That is the form worth
using at decode time, and the reason the cache is small:

```
KV cache per token (bf16, 61 layers, 128 heads)
  standard MHA :   3904.0 KiB     488.0 GiB at 128K context
  GQA, 8 groups:    244.0 KiB      30.5 GiB
  MLA          :     68.6 KiB       8.6 GiB   (57x smaller than MHA)
```

The routing bias flattens expert load with no gradient attached to it, going
from 1.94x to 1.28x worst case load over 400 steps. The test also shows why
gamma is a real hyper-parameter: at gamma = 0.05 the sign based update
overshoots every step and settles at 3.90x instead of converging.

The FP8 results separate two effects the paper treats separately. Fine grained
scaling beats per tensor scaling once an activation outlier is present, 3.17%
against 3.61% relative GEMM error. The accumulation precision is a different
axis: with a 14 bit register the error grows with K (0.008% at K=512 up to
0.021% at K=4096), and promoting the partial sum to an FP32 CUDA core register
every 128 elements holds it flat at 0.004% regardless of K. Promotion does not
remove the error inside a chunk, it stops it compounding across chunks.

Note that the paper's 0.25% figure is a relative *loss* error against a BF16
baseline, not a per GEMM error. Measured where the paper measures it, a two
layer forward pass in FP8 lands at 0.0004%.

Speculative decoding with the MTP module reproduces greedy output exactly,
including rolling the latent cache back past a rejected draft. Tokens per
forward is 1 plus the acceptance rate, so the 85 to 90% the paper reports gives
roughly 1.9.

## Config

`ModelConfig()` carries the production hyper-parameters from Section 4.2: 61
layers, 7168 hidden, 128 heads of 128, d_c = 512, d'_c = 1536, d_h^R = 64, 1
shared and 256 routed experts with 8 activated, M = 4 nodes, gamma = 0.001,
alpha = 0.0001, MTP depth 1 with lambda = 0.3. Instantiating it needs a cluster.
`ModelConfig.tiny()` keeps every structural property at a size that runs here.

## Caveats

FP8 is emulated. Values are rounded onto the E4M3 grid and widened back, which
reproduces the numerics exactly but none of the speed. The MoE forward loops
over experts in Python, which is correct and slow; a real implementation
dispatches with grouped GEMMs. The redundant expert planner is the placement
rule, not a serving system.
# Deepseek-V3
