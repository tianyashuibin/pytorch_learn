# LLM 推理优化端到端报告（第 9 周）

## 1. 模型与输入

- 模型：手写 MiniGPT，dim=512 layers=8 heads=8 max_seq_len=1024
- 负载：prompt_len=128, gen_len=64
- 环境：device=mps, dtype=torch.float16, torch=2.12.0

> 说明：MiniGPT 是可控负载，用于把方法跑通。要换真实模型（如 TinyLlama），
> 复用第 1 周 model.py 的加载器，指标口径不变。

## 2. 延迟 / 吞吐 / 显存

| 配置 | TTFT(ms) | TPOT(ms) | decode tok/s | 峰值MB |
| --- | --- | --- | --- | --- |
| 基线 batch=1 | 6.60 | 1.48 | 676.1 | 131 |
| batch=8 (并发) | 42.99 | 4.73 | 1692.7 | 391 |
| W4A16 (替换41个Linear) | 26.60 | 20.64 | 48.5 | 557 |

## 3. 正确性

- W4A16 vs FP 基线 prefill logits 相对误差：**8.4539%**
- （补：生成文本是否一致 / 前 K token 是否匹配）

## 4. 各维度实验（填结论）

- [ ] KV cache 增长（第 3 周公式，见终端输出）
- [ ] SDPA 后端对比（第 4 周，哪个最快、为什么）
- [ ] CUDA Graph 开关（第 5 周，仅 CUDA；TPOT 变化）
- [ ] W4A16 量化（第 8 周，显存↓ / 误差 / 速度）
- [ ] 高并发 batch 吞吐（第 7 周，continuous batching）

## 5. 瓶颈、优化理由、副作用

> 一句话说清：瓶颈在哪（访存/算力/launch）→ 用了什么优化 → 副作用是什么。
> 例：decode 访存受限 → KV/权重量化省带宽 → 代价是数值误差 X%，需验证下游质量。

## 6. 复现命令

```bash
python run_project.py
```
