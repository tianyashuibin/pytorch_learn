"""第 2 周公共工具：设备选择 + 一个可复用的小模型加载。

第 2 周关注 eager 运行时本身（dispatcher / allocator），
模型只是载体，所以给两种选择：
  - 一个纯 Tensor 的 microbench（不依赖 transformers）；
  - 一个真实小 GPT（需要 transformers）。
"""

from __future__ import annotations

import torch

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def try_load_llm(device: torch.device, dtype: torch.dtype):
    """尝试加载真实小 GPT；失败（无 transformers / 无网络）则返回 None。"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, torch_dtype=dtype)
        model.to(device).eval()
        return model, tok
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 加载 {DEFAULT_MODEL} 失败，将退回纯 Tensor microbench：{e}")
        return None
