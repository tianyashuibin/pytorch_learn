# 第 9 周：端到端综合项目 ★★★

## 目标

把第 1~8 周串成**一次可复现的 LLM 推理优化**，产出的不是几个孤立数字，而是一份能说清"瓶颈在哪、为什么这样优化、副作用是什么"的报告。

## 一条命令跑通所有维度

```bash
cd week9
python run_project.py          # 用带 torch 的 python（本机是 miniconda 的 python）
python run_project.py --big    # 更大模型档位（有 GPU 时更有代表性）
```

用第 3 周手写的 `MiniGPT` 作**可控真实负载**（不依赖联网/HF，随处能跑），在它上面一次跑齐：

| # | 维度 | 来自 | 脚本里怎么体现 |
| --- | --- | --- | --- |
| 1 | eager FP16/BF16 基线 | week1 | `bench()` 冷/热分离测 TTFT/TPOT |
| 2 | prefill/decode 分离指标 | week1/3 | prefill 一次吃 prompt，decode 逐 token 计时 |
| 3 | KV cache 增长 | week3 | 按公式算每 token / 每请求 KV 显存 |
| 4 | attention 后端对比 | week4 | `sdpa_backend_compare` 强制 MATH/FLASH/EFFICIENT |
| 5 | CUDA Graph 开关 | week5 | CUDA 上提示如何 capture；非 CUDA 优雅跳过 |
| 6 | 量化实验 | week8 | `apply_w4a16` 就地替换 Linear |
| 7 | 高并发调度 | week7 | batch=1 vs batch=8 吞吐对比 |
| 8 | 正确性验证 | — | W4A16 vs 基线 logits 相对误差 |
| 9 | 瓶颈解释 | 全部 | 脚本末尾按实测数据给结论 |

跑完终端出一张统一对比表，并写出 [report.md](report.md) 报告骨架（填结论即成交付）。

## 本机实跑结果（MPS, fp16, dim512/8层）参考

```
配置                          TTFT(ms)  TPOT(ms)  decode tok/s   峰值MB
基线 batch=1                      6.52      1.47         678.1      131
batch=8 (并发)                   42.78      4.77        1677.7      391
W4A16 (替换41个Linear)            26.30     20.54          48.7      557
```

**这张表里最值得琢磨的是 W4A16 那行反而更慢、显存更高**——这不是 bug，而是刻意保留的教学点：

- mini `W4A16Linear` 每次前向**把整块 int4 权重解回 fp16 再做 matmul**（未融合 dequant），于是既没省显存（反而多了一份 fp16 临时权重），又多了解包开销。
- 真实 W4A16（vLLM `awq_triton.py` 那类）用**融合 kernel 边解包边乘**，权重从不 materialize 成 fp16，才拿到"省显存 + decode 加速"的真实收益。
- 结论：**量化的收益依赖 kernel 实现，不是"位数变少就自动变快"**。这正是第 5 周 Triton、第 8 周"融合 dequant kernel"那句话的落地。

（batch=1→8 吞吐 2.5x 则如预期：批处理摊薄每步固定/launch 开销，呼应第 2、7 周。）

## 换成真实模型

要把 MiniGPT 换成 TinyLlama 等真实模型：复用第 1 周 `week1/model.py` 的加载器和 `benchmark_baseline.py` 的计时口径，指标定义完全一致。量化可直接用 `pip install torchao` 后 `quantize_(model, int4_weight_only())` 拿到融合 kernel 的真实收益，替代这里的 mini 实现。

## 固定实验模板（每个实验都填这张表）

| 项目 | 内容 |
| --- | --- |
| 模型与输入 | 模型、并发、prompt 长度、生成长度、dtype |
| 环境 | GPU、CUDA、torch/引擎 commit、关键配置 |
| 正确性 | 与基线的误差 / 一致性 |
| 延迟 | TTFT(P50/P99)、TPOT/ITL |
| 吞吐 | tokens/s、requests/s |
| 资源 | 峰值显存、GPU 利用率、KV 占用 |
| 内核 | 主要 kernel、attention 实现、CUDA Graph 是否命中 |
| 调度 | batch 组织、前缀缓存命中率、驱逐次数 |
| 结论 | 瓶颈、优化理由、副作用 |

**每次只改一个主要变量**——否则无法归因。

## 本周验收（能答出来才算过）

1. 你的报告里，decode 的瓶颈判断依据是什么数据？（TPOT、GPU 利用率、kernel 大小）
2. batch 增大后吞吐为什么涨、涨到哪会饱和？（摊薄固定开销 vs KV/算力上限）
3. 为什么本项目里 W4A16 反而变慢？真实引擎靠什么拿回收益？（融合 dequant kernel）
4. 一个优化的副作用你怎么验证？（量化误差如何量化、是否影响下游质量）
5. 如果只让你保留一项优化上线，你选哪个、用什么数据支撑？

## 承上启下

- 本周是第 1~8 周的收口：每一行对比表都能追溯到某一周的机制。
- 第 10 周把这套方法固化成《排障手册》，并挑一个引擎子系统（KV cache / 调度 / 一个 attention kernel）逐行读通，补全"引擎为什么手写而不交给编译器"的答案。
