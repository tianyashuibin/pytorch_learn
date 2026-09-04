"""
第 3 周:fullgraph=True 与默认(允许 break)的区别。

- 默认:遇到 graph break 就切图 + 回落 Python,程序照常跑,只是变慢。
- fullgraph=True:要求整段必须编译成一张图,遇到 break 直接抛错。
  这是推理部署里很有用的"守门"开关——强制暴露所有 break,逼你修掉。

运行:
    python fullgraph_vs_break.py
"""
import torch
import torch._dynamo as dynamo

from models import build


def try_fullgraph(name):
    dynamo.reset()
    model, x = build(name)
    compiled = torch.compile(model, fullgraph=True)
    print(f"--- {name} (fullgraph=True) ---")
    try:
        with torch.inference_mode():
            compiled(x)
        print("  ✓ 整图编译成功,无 graph break\n")
    except Exception as e:
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        print(f"  ✗ 触发 graph break,fullgraph 报错: {msg}\n")


def main():
    print("fullgraph=True 会把任何 graph break 变成硬错误,方便定位:\n")
    for name in ("clean", "control_flow", "side_effect", "unsupported"):
        try_fullgraph(name)
    print("实践建议:推理路径先用 fullgraph=True 跑一遍,把 break 全部逼出来再逐个修。")


if __name__ == "__main__":
    main()
