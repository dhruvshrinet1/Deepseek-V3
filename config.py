"""Configuration for the DeepSeek-V3 reference implementation.

Defaults mirror the hyper-parameters reported in Section 4.2 of the technical
report. `ModelConfig.tiny()` gives a small variant that actually runs on a CPU.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # ---- general ----
    vocab_size: int = 129280          # byte-level BPE, 128K + specials
    d_model: int = 7168               # hidden dimension
    n_layers: int = 61                # transformer blocks
    n_dense_layers: int = 3           # first N layers keep a dense FFN
    ffn_dim: int = 18432              # dense FFN intermediate size
    max_seq_len: int = 4096           # pre-training sequence length
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    init_std: float = 0.006           # "randomly initialized with std 0.006"

    # ---- Multi-head Latent Attention (Section 2.1.1) ----
    n_heads: int = 128                # n_h
    d_head: int = 128                 # d_h
    kv_lora_rank: int = 512           # d_c, the joint KV compression dimension
    q_lora_rank: int = 1536           # d'_c, the query compression dimension
    rope_head_dim: int = 64           # d_h^R, the decoupled RoPE dimension

    # ---- DeepSeekMoE (Section 2.1.2) ----
    n_shared_experts: int = 1         # N_s
    n_routed_experts: int = 256       # N_r
    n_activated_experts: int = 8      # K_r
    expert_ffn_dim: int = 2048        # per-expert intermediate size
    n_expert_groups: int = 8          # experts are spread over this many nodes
    n_limited_groups: int = 4         # M, the node-limited routing cap
    bias_update_speed: float = 1e-3   # gamma, set to 0.0 for the final 500B tokens
    balance_alpha: float = 1e-4       # alpha, the sequence-wise balance loss factor

    # ---- Multi-Token Prediction (Section 2.2) ----
    mtp_depth: int = 1                # D
    mtp_loss_weight: float = 0.3      # lambda (0.3 for the first 10T tokens, then 0.1)

    # ---- YaRN long-context extension (Section 4.3) ----
    rope_scaling_factor: float = 1.0  # s; 40 when extending 4K -> 128K
    rope_original_max_seq_len: int = 4096

    # ---- FP8 (Section 3.3) ----
    use_fp8: bool = False             # emulated; see fp8.py
    fp8_block_size: int = 128         # N_c, also the accumulation promotion interval

    @property
    def qk_head_dim(self) -> int:
        """Per-head query/key width: the compressed part plus the RoPE part."""
        return self.d_head + self.rope_head_dim

    @property
    def experts_per_group(self) -> int:
        assert self.n_routed_experts % self.n_expert_groups == 0
        return self.n_routed_experts // self.n_expert_groups

    @classmethod
    def tiny(cls, **overrides) -> "ModelConfig":
        """A small config with the same structure, for tests and demos."""
        cfg = cls(
            vocab_size=503,   # prime, so no other layer width collides with it
            d_model=256,
            n_layers=4,
            n_dense_layers=1,
            ffn_dim=512,
            max_seq_len=256,
            n_heads=8,
            d_head=32,
            kv_lora_rank=64,
            q_lora_rank=96,
            rope_head_dim=16,
            n_shared_experts=1,
            n_routed_experts=16,
            n_activated_experts=4,
            expert_ffn_dim=128,
            n_expert_groups=4,
            n_limited_groups=2,
            mtp_depth=1,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg
