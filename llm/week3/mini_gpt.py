"""第 3 周：从零手写一个带显式 KV cache 的 mini GPT。

为什么手写而不是直接读 gpt_fast？
  gpt_fast 的 model.py 已经很精简，但要真正建立 prefill/decode/KV cache 的心智模型，
  自己写一遍、能 print 出每一步 cache 的形状，比读代码更扎实。
  写完再回去读 benchmarks/gpt_fast/model.py，你会发现结构一一对应。

这个实现刻意做到：
  - 结构和真实 LLM 一致：RMSNorm + RoPE + MHA(带 KV cache) + SwiGLU MLP。
  - KV cache 是"预分配 + 按位置写入"的静态 cache —— 这正是引擎（和 CUDA Graph）需要的形态。
  - 用随机权重即可：本周关心的是机制、形状、时间，不是生成质量。

对照阅读：benchmarks/gpt_fast/model.py（KVCache、Attention、Transformer）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    max_seq_len: int = 2048
    ffn_hidden: int = 1536  # SwiGLU 中间维度


# ---------------------------------------------------------------------------
# KV cache：本周的主角。
# 预分配 [B, n_heads, max_seq_len, head_dim] 的 K/V 缓冲区，decode 时按 position 写入。
# 静态形状 + 固定地址是引擎能用 CUDA Graph 的前提（第 5 周会用到这个事实）。
# ---------------------------------------------------------------------------
class KVCache(nn.Module):
    def __init__(self, batch: int, n_heads: int, max_seq_len: int, head_dim: int,
                 dtype: torch.dtype, device: torch.device):
        super().__init__()
        shape = (batch, n_heads, max_seq_len, head_dim)
        # 用 buffer 预分配，地址在整个 generate 过程中不变
        self.register_buffer("k", torch.zeros(shape, dtype=dtype, device=device), persistent=False)
        self.register_buffer("v", torch.zeros(shape, dtype=dtype, device=device), persistent=False)

    def update(self, start_pos: int, k: torch.Tensor, v: torch.Tensor):
        """把新算出的 k/v 写入 [start_pos : start_pos+seq]，返回从头到当前的全部 k/v。

        - prefill：start_pos=0，seq=prompt_len，一次写入一大段。
        - decode ：start_pos=当前长度，seq=1，每步写入一格。
        """
        seq = k.shape[2]
        self.k[:, :, start_pos:start_pos + seq] = k
        self.v[:, :, start_pos:start_pos + seq] = v
        # attention 需要看到从 0 到当前位置的所有历史 K/V
        return self.k[:, :, :start_pos + seq], self.v[:, :, :start_pos + seq]


# ---------------------------------------------------------------------------
# RoPE：旋转位置编码。decode 时要按"当前绝对位置"取对应的 cos/sin。
# ---------------------------------------------------------------------------
def build_rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)  # [seq, head_dim/2]
    return torch.cos(freqs), torch.sin(freqs)  # 各 [seq, head_dim/2]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, n_heads, seq, head_dim]
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack([out1, out2], dim=-1).flatten(-2)
    return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.wqkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, start_pos, kv_cache: KVCache, cos, sin, causal: bool):
        B, seq, _ = x.shape
        q, k, v = self.wqkv(x).split(x.shape[-1], dim=-1)
        # [B, seq, dim] -> [B, n_heads, seq, head_dim]
        q = q.view(B, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, seq, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # 写入并取回全部历史 —— KV cache 的核心动作
        k, v = kv_cache.update(start_pos, k, v)

        # prefill 段内需要 causal mask；decode 单 token 对全部历史做 attention，不需要 mask
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=causal and seq > 1)
        attn = attn.transpose(1, 2).reshape(B, seq, -1)
        return self.wo(attn)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, start_pos, kv_cache, cos, sin, causal):
        x = x + self.attn(self.attn_norm(x), start_pos, kv_cache, cos, sin, causal)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.kv_caches: list[KVCache] | None = None
        self._rope = None  # (cos, sin) 全长度缓存

    def setup_caches(self, batch: int, device, dtype):
        """在 generate 前预分配所有层的 KV cache 和 RoPE 表。引擎里对应 'allocate KV blocks'。"""
        cfg = self.cfg
        head_dim = cfg.dim // cfg.n_heads
        self.kv_caches = [
            KVCache(batch, cfg.n_heads, cfg.max_seq_len, head_dim, dtype, device)
            for _ in range(cfg.n_layers)
        ]
        cos, sin = build_rope_cache(cfg.max_seq_len, head_dim, device)
        self._rope = (cos.to(dtype), sin.to(dtype))

    def forward(self, tokens: torch.Tensor, start_pos: int, causal: bool):
        """tokens: [B, seq]。start_pos 指定这段 token 在序列中的起始绝对位置。"""
        B, seq = tokens.shape
        x = self.tok_emb(tokens)
        cos, sin = self._rope
        cos = cos[start_pos:start_pos + seq]
        sin = sin[start_pos:start_pos + seq]
        for block, kv in zip(self.blocks, self.kv_caches):
            x = block(x, start_pos, kv, cos, sin, causal)
        x = self.norm(x)
        return self.lm_head(x)


def build_mini_gpt(device: torch.device, dtype: torch.dtype,
                   cfg: GPTConfig | None = None) -> MiniGPT:
    cfg = cfg or GPTConfig()
    model = MiniGPT(cfg).to(device=device, dtype=dtype)
    model.eval()
    return model
