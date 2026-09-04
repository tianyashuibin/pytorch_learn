"""
第 1 周的实验模型。

计划建议"始终带着一个真实模型学习":
- TinyMLP     —— 最小实验对象,便于验证正确性和快速迭代。
- TinyTransformer —— 一个极简的 decoder-only block 堆叠,更接近 LLM 推理负载,
  用来在后续几周观察 prefill / decode、动态 shape、kernel 数等现象。

这里刻意保持依赖最少(不引入 transformers 等库),方便反复读源码和 profiler 对照。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMLP(nn.Module):
    def __init__(self, dim: int = 1024, hidden: int = 4096, layers: int = 4):
        super().__init__()
        blocks = []
        d = dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """标准 pre-norm decoder block:MHA + MLP。"""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        b, t, c = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # 每个 [b, t, n_heads, head_dim]
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))  # -> [b, n_heads, t, head_dim]
        # causal SDPA:后续几周会重点看它如何走 FlashAttention / flex_attention
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(b, t, c)
        x = x + self.proj(attn)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int = 8192, dim: int = 512, n_heads: int = 8, n_layers: int = 6):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)


def build_model(name: str, device: torch.device, dtype: torch.dtype = torch.float32):
    """按名字构造模型和一个匹配的示例输入。"""
    if name == "mlp":
        model = TinyMLP().to(device=device, dtype=dtype).eval()
        example = torch.randn(32, 1024, device=device, dtype=dtype)
    elif name == "transformer":
        model = TinyTransformer().to(device=device).eval()  # embedding 走 long,保持默认精度
        if dtype in (torch.float16, torch.bfloat16):
            model = model.to(dtype=dtype)
        example = torch.randint(0, 8192, (4, 256), device=device)
    else:
        raise ValueError(f"未知模型: {name}(可选 mlp / transformer)")
    return model, example
