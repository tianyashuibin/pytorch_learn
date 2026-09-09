"""
第 3 周:用自定义后端截获 Dynamo trace 出的 FX 图,并观察 break 如何把它切成多段。

自定义 backend 的签名是 (gm: fx.GraphModule, example_inputs) -> callable。
Dynamo 每 trace 出一张子图就调用一次 backend。因此:
- clean 模型 -> backend 只被调用 1 次(1 张图)
- 有 graph break 的模型 -> backend 被调用多次(每段一张图)

这对应阅读:
- torch/_dynamo/eval_frame.py   (PEP 523 frame 拦截、compile 入口)
- torch/_dynamo/output_graph.py (FX Graph 构造)
- torch/_dynamo/codegen.py      (把图和 break 之间的胶水代码生成回 Python)

运行:
    python capture_backend.py --model control_flow
"""
import argparse

import torch
import torch._dynamo as dynamo

from models import build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="control_flow",
                        choices=["clean", "control_flow", "side_effect", "unsupported"])
    args = parser.parse_args()

    dynamo.reset()
    model, x = build(args.model)

    graphs = []

    def capture(gm: torch.fx.GraphModule, example_inputs):
        graphs.append(gm)
        idx = len(graphs)
        print("=" * 70)
        print(f"[子图 #{idx}] Dynamo 捕获的 FX 图")
        print("=" * 70)
        gm.graph.print_tabular()
        print()
        return gm.forward  # 交回 eager 执行

    compiled = torch.compile(model, backend=capture)
    with torch.inference_mode():
        compiled(x)

    print(f"结论:模型 '{args.model}' 被切成 {len(graphs)} 张子图。")
    if len(graphs) > 1:
        print("      >1 张 = 发生了 graph break,中间那段代码回落到 Python 执行。")
    else:
        print("      1 张 = 整图捕获,无 graph break。")


if __name__ == "__main__":
    main()
