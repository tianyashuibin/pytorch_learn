# LLM 推理排障手册

> 用法：从**症状**出发，按"可能原因 → 怎么验证 → 优化方向"逐条排除。每条都标了对应的学习周，回去看机制。
> 铁律：**每次只改一个变量**，改前先测基线，用数据归因，不靠猜。

## 0. 先分清 prefill 还是 decode 出问题

任何延迟问题，第一步把 TTFT（prefill）和 TPOT/ITL（decode）分开测（week1）。两者瓶颈完全不同：

| 阶段 | 典型瓶颈 | 一句话 |
| --- | --- | --- |
| prefill | 算力受限（compute-bound） | 大矩阵乘，GPU 算得满 |
| decode | 访存 / launch 受限（memory/launch-bound） | 每步只算 1 token，瓶颈是搬权重+读 KV、发 kernel |

搞反了会优化错方向（如给 decode 上激活量化、给 prefill 上 CUDA Graph 都是浪费）。

---

## 1. decode 慢 / TPOT 高

| 可能原因 | 怎么验证 | 优化方向 |
| --- | --- | --- |
| kernel launch 开销大（小 kernel 一大串） | profiler 看 GPU 有大量空隙、CPU launch 时间 > kernel 时间（week2） | **CUDA Graph** capture/replay 消 launch（week5）；`torch.compile(mode="reduce-overhead")` |
| 权重搬运受限 | decode 是 memory-bound，算 roofline / 看带宽利用 | **权重量化 W4A16**（week8），搬运量降到 1/4 |
| 读 KV cache 带宽受限（长上下文） | KV 显存大、TPOT 随 seq 增长 | **KV cache 量化 INT8/FP8**（week8） |
| batch 太小，固定开销摊不开 | batch=1 vs batch=N 吞吐对比（week9 实测 2.5x） | **continuous batching** 提高并发（week7） |
| attention 后端选得差 | 强制对比 SDPA 各后端（week4） | 选 FLASH/EFFICIENT，避免 MATH 后端 O(seq²) 物化 |

## 2. TTFT 高 / prefill 慢

| 可能原因 | 怎么验证 | 优化方向 |
| --- | --- | --- |
| prompt 太长、无前缀复用 | 相同 system prompt 反复重算 | **前缀缓存**：RadixCache / block-hash（week6），命中直接省 prefill |
| attention 是 O(seq²) 朴素实现 | 看是否走了 flash/efficient（week4） | 上 FlashAttention（tiling + online softmax，不物化 seq² 矩阵） |
| 长 prompt 阻塞其它请求 | 单个长 prompt 占满一步 | **chunked prefill** 切块跨步（week7），让 decode 请求也能挤进 batch |
| 激活是算力瓶颈 | prefill compute-bound | **激活量化 W8A8**（week8）用 int8 tensor core |

## 3. OOM / 显存不够

| 可能原因 | 怎么验证 | 优化方向 |
| --- | --- | --- |
| KV cache 占用爆炸 | 按公式 `2·layers·batch·heads·seq·head_dim·bytes` 估（week3） | KV 量化；减小 max batch；分页减少碎片（week6） |
| 显存碎片（reserved ≫ allocated） | `torch.cuda.memory_summary()`，reserved 远大于 allocated（week2） | 分页 KV（PagedAttention）；`expandable_segments`；避免频繁变 shape |
| batch/seq 分档 padding 浪费 | CUDA Graph 要求 static shape，padding 到大档 | 合理设分档粒度，别只留一个巨大档 |
| 权重本身太大 | 权重显存 = 参数量 × bytes | 权重量化 W4A16（week8，近 4x） |
| 忘了 `inference_mode` 存了 autograd 图 | 显存随步数涨 | `torch.inference_mode()`（week2） |

## 4. 吞吐上不去（throughput 低）

| 可能原因 | 怎么验证 | 优化方向 |
| --- | --- | --- |
| 并发度低、GPU 没喂饱 | GPU 利用率低、batch 小 | continuous batching，动态组 batch（week7） |
| 前缀缓存没命中 | 命中率低（多轮对话/共享 prompt 却没复用） | RadixCache 前缀树（week6），细到 token 复用 |
| 抢占频繁（KV 压力大） | 调度日志里 preemption 次数高（week7） | 加 KV 容量 / 降 max running / KV 量化腾空间 |
| 被长请求拖住（队头阻塞） | FCFS 下长 prompt 卡队首 | chunked prefill；必要时 priority 调度（week7） |

## 5. 正确性 / 数值异常

| 症状 | 可能原因 | 怎么验证 / 优化 |
| --- | --- | --- |
| 量化后输出漂移大 | 量化粒度太粗 / 位宽太低 | per-channel/group-wise、提位宽；GPTQ/AWQ 误差补偿（week8）。对比基线 logits 相对误差 |
| 量化"更慢/更费显存" | dequant 未融合，权重被 materialize 成 fp16 | 用融合 dequant kernel（week9 实测教训） |
| 输出 NaN | fp16 溢出 / mask 错 / 未初始化 KV | 检查 attention mask、KV buffer 初始化；必要时 fp32 兜关键算子 |
| CUDA Graph replay 结果错 | 违反三约束（地址/原地更新/shape） | 固定地址 KV、`copy_` 原地写输入、shape 归档（week5） |
| decode 结果和 prefill 不一致 | start_pos / RoPE 位置错、causal 标志错 | 检查 KV update 的 start_pos、RoPE 取 `cos[start_pos:...]`（week3） |

## 6. 通用诊断动作清单

1. **分离 prefill/decode 计时**，冷/热分开（week1）。
2. **profiler 归因**：CPU launch time vs GPU kernel time，谁大？（week2）
3. **显存快照**：allocated vs reserved，峰值在哪一步（week2/3）。
4. **roofline 直觉**：这个算子是 compute-bound 还是 memory-bound？决定优化方向。
5. **强制变量对比**：SDPA 后端、CUDA Graph 开关、量化开关、batch 大小——每次只改一个（week9）。
6. **正确性守门**：任何优化都对比基线输出误差，别只看速度。

---

*这份手册对应 `pytorch_learn_llm` 第 1~9 周的机制，每条都能回到具体某周的代码验证。*
