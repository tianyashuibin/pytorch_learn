"""
第 3 周实践:用 torch._dynamo.explain 量化每个模型的 graph break。

explain() 会真正 trace 一遍,返回:
- graph_count      : 被切成几张子图(1 = 无 break)
- graph_break_count: graph break 次数
- break_reasons    : 每次 break 的原因和用户代码位置

这直接对应验收:"能从日志定位 graph break 对应的用户代码"。

运行:
    python explain_breaks.py
    # 想看原始日志:
    TORCH_LOGS="graph_breaks,graph_code" python explain_breaks.py
"""
import torch
import torch._dynamo as dynamo

from models import REGISTRY, build


def analyze(name):
    dynamo.reset()
    model, x = build(name)
    print("=" * 70)
    print(f"模型: {name}")
    print("=" * 70)
    try:
        explanation = dynamo.explain(model)(x)
    except Exception as e:
        print(f"  explain 失败: {type(e).__name__}: {e}\n")
        return

    print(f"  子图数 graph_count       = {explanation.graph_count}")
    print(f"  graph break 次数         = {explanation.graph_break_count}")
    print(f"  编译的 op 节点数          = {explanation.op_count}")
    if explanation.break_reasons:
        print("  break 原因与位置:")
        for i, r in enumerate(explanation.break_reasons, 1):
            reason = getattr(r, "reason", r)
            print(f"    [{i}] {reason}")
            # user_stack 指向触发 break 的用户代码行
            for frame in (getattr(r, "user_stack", None) or [])[-2:]:
                print(f"         at {frame.filename}:{frame.lineno}  {frame.line}")
    print()


def main():
    print("解读:graph_count=1 且 break=0 -> 整图捕获;break 越多,Python<->kernel 往返越多。\n")
    for name in REGISTRY:
        analyze(name)
    print("提示:graph break 会强制在断点处执行、同步、再重新进入编译,")
    print("      因此增加 Python 调度和 kernel launch 开销,是推理优化要消除的目标之一。")


if __name__ == "__main__":
    main()
