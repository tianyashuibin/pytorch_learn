"""第 1 周：模型与 tokenizer 加载。

默认用一个小 GPT，方便在单卡甚至 CPU 上跑通流程。
本周目标是"跑通 benchmark 方法"，不是"追求大模型性能"，所以模型越小越好调。
"""

from __future__ import annotations

import torch

# 默认小模型：无网络时可换成本地路径。TinyLlama 约 1.1B，单卡友好。
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon 上可用 mps 跑通流程（但 CUDA 相关计时/显存指标会是 nan）。
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        # 优先 bf16（数值更稳），不支持则 fp16。
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(model_name: str, device: torch.device, dtype: torch.dtype):
    """加载 HuggingFace 因果语言模型。

    需要 transformers：pip install transformers
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer
