"""第 7 周：continuous batching 调度器的最小模拟（引擎视角）。

这不是真实 vLLM，而是把 vLLM v1 Scheduler 的核心机制抽成一个可跑的模拟器，
先建立骨架，再去读真实源码 vllm/v1/core/sched/scheduler.py。

模拟的四个核心机制：
  1. continuous batching：每一步动态组 batch，把 running(decode) 和 waiting(prefill)
     的请求拼进同一步，而不是等一整个 batch 全部结束（那是静态 batching）。
  2. token budget：每步能处理的 token 总数有上限（受显存/算力约束）。
     prefill 一次吃很多 token，decode 一个请求只吃 1 token。
  3. chunked prefill：一个超长 prompt 不必一步吃完，可按 budget 切块跨多步。
  4. preemption：KV cache 显存不足时，把某个 running 请求踢回 waiting（释放它的 KV）。

对照真实：vllm/v1/core/sched/scheduler.py 的 Scheduler.schedule()。

运行：python scheduler_sim.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class Status(Enum):
    WAITING = "waiting"       # 还没开始 / 被抢占后回到这
    RUNNING = "running"       # 在解码中
    FINISHED = "finished"


@dataclass
class Request:
    rid: str
    prompt_len: int           # 需要 prefill 的 token 数
    gen_len: int              # 需要生成的 token 数
    status: Status = Status.WAITING
    prefilled: int = 0        # 已 prefill 的 token 数（chunked prefill 用）
    generated: int = 0        # 已生成的 token 数
    kv_blocks: int = 0        # 占用的 KV 块数（简化：1 块 = 1 token）

    @property
    def prefill_done(self) -> bool:
        return self.prefilled >= self.prompt_len

    @property
    def done(self) -> bool:
        return self.generated >= self.gen_len


class Scheduler:
    def __init__(self, token_budget: int, kv_capacity: int):
        self.token_budget = token_budget   # 每步 token 上限
        self.kv_capacity = kv_capacity      # KV cache 总容量（块）
        self.waiting: deque[Request] = deque()   # FCFS 队列
        self.running: list[Request] = []
        self.kv_used = 0
        self.step_no = 0

    def add(self, req: Request):
        self.waiting.append(req)

    def _kv_free(self) -> int:
        return self.kv_capacity - self.kv_used

    def _preempt(self):
        """KV 不足时，抢占最后进入 running 的请求（LIFO），把它踢回 waiting 队首。"""
        victim = self.running.pop()          # 简化策略：抢最年轻的
        self.kv_used -= victim.kv_blocks
        victim.kv_blocks = 0
        victim.prefilled = 0                 # 简化：被抢占后 KV 全丢，需重新 prefill
        victim.generated = victim.generated  # 已生成的 token 保留（真实里 KV 重算）
        victim.status = Status.WAITING
        self.waiting.appendleft(victim)      # 回到队首，优先恢复
        return victim.rid

    def schedule_step(self) -> dict:
        """模拟一步：组一个 batch，返回这步做了什么。核心对应 Scheduler.schedule()。"""
        self.step_no += 1
        budget = self.token_budget
        batch = []          # (rid, kind, tokens_this_step)
        preempted = []

        # ---- 1) 先调度 running 请求的 decode（每个吃 1 token）----
        for req in list(self.running):
            if budget <= 0:
                break
            if req.done:
                continue
            # decode 需要为新 token 申请 1 个 KV 块
            while self._kv_free() < 1 and self.running:
                if self.running[-1] is req and len(self.running) == 1:
                    break
                pid = self._preempt()
                preempted.append(pid)
            if self._kv_free() < 1:
                break
            req.kv_blocks += 1
            self.kv_used += 1
            req.generated += 1
            budget -= 1
            batch.append((req.rid, "decode", 1))

        # ---- 2) 再用剩余 budget 调度 waiting 请求的 prefill（可 chunked）----
        while self.waiting and budget > 0:
            req = self.waiting[0]
            remaining = req.prompt_len - req.prefilled
            # chunked prefill：这步最多吃 min(剩余 prompt, 剩余 budget) 个 token
            chunk = min(remaining, budget)
            # prefill 需要为这些 token 申请 KV 块
            if self._kv_free() < chunk:
                chunk = self._kv_free()      # 显存不够就少吃点
            if chunk <= 0:
                break
            self.waiting.popleft()
            req.prefilled += chunk
            req.kv_blocks += chunk
            self.kv_used += chunk
            budget -= chunk
            if req.prefill_done:
                req.status = Status.RUNNING
                self.running.append(req)
                kind = "prefill(done)"
            else:
                # 还没 prefill 完，放回队首，下一步接着吃（chunked）
                self.waiting.appendleft(req)
                kind = "prefill(chunk)"
            batch.append((req.rid, kind, chunk))
            if kind == "prefill(chunk)":
                break  # 这步 budget 用于这个长 prompt 的一块，下步继续

        # ---- 3) 清理已完成请求，释放 KV ----
        finished = []
        for req in list(self.running):
            if req.done:
                req.status = Status.FINISHED
                self.running.remove(req)
                self.kv_used -= req.kv_blocks
                req.kv_blocks = 0
                finished.append(req.rid)

        return {"step": self.step_no, "batch": batch, "preempted": preempted,
                "finished": finished, "kv_used": self.kv_used,
                "waiting": len(self.waiting), "running": len(self.running)}

    def all_done(self) -> bool:
        return not self.waiting and not self.running


def demo():
    # token_budget=8：每步最多处理 8 个 token；kv_capacity=20：KV 只能存 20 个 token
    sched = Scheduler(token_budget=8, kv_capacity=20)
    sched.add(Request("A", prompt_len=6, gen_len=3))
    sched.add(Request("B", prompt_len=10, gen_len=2))   # 长 prompt，会触发 chunked prefill
    sched.add(Request("C", prompt_len=4, gen_len=2))

    print("配置：token_budget=8/步, kv_capacity=20 块")
    print("请求：A(prompt6,gen3) B(prompt10,gen2) C(prompt4,gen2)\n")
    print(f"{'step':>4} | {'batch (rid,kind,tokens)':<48} | kv | wait run | 事件")
    print("-" * 92)
    while not sched.all_done():
        r = sched.schedule_step()
        batch_str = "  ".join(f"{rid}:{kind}:{tk}" for rid, kind, tk in r["batch"]) or "(空)"
        events = []
        if r["preempted"]:
            events.append(f"抢占{r['preempted']}")
        if r["finished"]:
            events.append(f"完成{r['finished']}")
        print(f"{r['step']:>4} | {batch_str:<48} | {r['kv_used']:>2} | "
              f"{r['waiting']:>4} {r['running']:>3} | {', '.join(events)}")
        if sched.step_no > 30:
            break

    print("\n要点：")
    print("  - 同一步里 decode 和 prefill 混在一个 batch（continuous batching），GPU 不空转。")
    print("  - B 的长 prompt 被 budget 切成多块跨步 prefill（chunked prefill）。")
    print("  - KV 逼近容量时触发抢占，把请求踢回 waiting 释放显存。")
    print("  - 对照真实源码：vllm/v1/core/sched/scheduler.py 的 Scheduler.schedule()")


if __name__ == "__main__":
    demo()
