# 第 3 周：TorchDynamo 图捕获

对应学习计划「第 3 周」。目标：理解 `torch.compile` 如何把 Python frame 转成 FX Graph，以及 graph break 后如何恢复执行。

## 对应源码阅读（按顺序）

1. `torch/_dynamo/eval_frame.py` — PEP 523 frame 拦截、compile 入口
2. `torch/_dynamo/convert_frame.py`
3. `torch/_dynamo/symbolic_convert.py` — `InstructionTranslator`
4. `torch/_dynamo/output_graph.py` — FX Graph 构造
5. `torch/_dynamo/codegen.py`
6. `torch/_dynamo/resume_execution.py` — continuation function

重点概念：PEP 523 frame interception、`InstructionTranslator`、`VariableTracker`、FX Graph 构造、graph break 与 continuation function。

## 文件

| 文件 | 作用 |
|---|---|
| `models.py` | 4 个模型：`clean` / `control_flow`（数据相关分支）/ `side_effect`（print）/ `unsupported`（numpy 往返） |
| `explain_breaks.py` | `torch._dynamo.explain` 量化每个模型的子图数、break 次数、break 原因与代码位置 |
| `capture_backend.py` | 自定义 backend 截获每张 FX 子图并 `print_tabular`，直观看到 break 把图切成几段 |
| `fullgraph_vs_break.py` | 对比默认（允许 break）与 `fullgraph=True`（break 即报错） |

## 运行

```bash
python explain_breaks.py
python capture_backend.py --model control_flow
python fullgraph_vs_break.py

# 观察原始日志：
TORCH_LOGS="graph_breaks,graph_code" python explain_breaks.py
```

## 本周验收

> 能够从日志定位 graph break 对应的用户代码，并说明它如何增加 Python 与 kernel 调度开销。

- `explain_breaks.py` 的 `break_reasons` 直接给出原因 + `user_stack`（文件:行号）。
- 每次 break = 在断点处强制执行当前子图 + 可能的同步（如 `.item()`）+ 回落 Python + 重新进入编译，因此增加 Python 调度和 kernel launch 往返。
- 消除 break 的常见手段：把数据相关控制流改成 tensor 运算（`torch.where`）、去掉推理路径里的 print/日志、避免中途 `.cpu().numpy()` 往返。
