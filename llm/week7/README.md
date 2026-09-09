# 第 7 周：continuous batching 与调度器（只读 vLLM）★★★

## 目标

理解引擎"每一步组一个 batch"的调度循环：如何把 **decode(running)** 和 **prefill(waiting)** 混进同一次前向、**token budget** 怎么约束、**chunked prefill** 怎么把长 prompt 切开跨步、**preemption** 在 KV 显存不足时怎么把 running 请求踢回 waiting。

> 本周源码路径基于你本地的 vLLM：`~/github/vllm-main/vllm/v1/`
> 本周**只看 vLLM**，不看 SGLang（两者调度思路一致，先吃透一个）。

## 先跑 mini 实现，再读真实源码

真实 `scheduler.py` 有 2600+ 行，混了 spec decode、encoder、P/D KV 传输、mamba、DP 平衡等大量分支，直接读会淹没在细节里。**先跑本目录的最小模拟器建立骨架**，再带着骨架去读真实源码（对照 [[feedback_code_reading_strategy]] 自顶向下）。

- `scheduler_sim.py` — 最小 continuous-batching 调度器：waiting/running 双队列、token budget、chunked prefill、KV 压力下 LIFO 抢占。

```bash
cd week7
python scheduler_sim.py
```

跑出来看三件事：

1. **同一步 batch 里 decode+prefill 并存**（如 `A:decode:1  B:prefill(chunk):7`）——这就是 continuous batching，GPU 不用等一整批结束。
2. **长 prompt B 被 budget 切块跨步 prefill**——chunked prefill。
3. **KV 逼近容量触发抢占**（`抢占['B']`），把请求踢回 waiting 释放显存，之后再恢复。

---

## vLLM v1 调度器源码阅读地图

阅读顺序（`v1/`）：

1. **`core/sched/interface.py:36`** `SchedulerInterface(ABC)` — 契约。核心抽象方法 `schedule()`（`:52`）的 docstring 一句话点破本质：调度产出一个 `{req_id: num_tokens}` 字典，指定这一步每个请求处理多少 token（新请求可以是整个 prompt，decode 请求就是 1，chunked prefill 介于两者之间）。`PauseState`（`:22`）UNPAUSED/PAUSED_NEW/PAUSED_ALL。
2. **`request.py`** — 请求与生命周期，先看状态机再看转移逻辑：
   - `Request`（`:59`）：关键字段 `num_computed_tokens`（已算到第几个 token）、`num_tokens_with_spec`、`status`、`is_prefill_chunk`（`:168`，标记非最后一块 prefill）、`num_preemptions`（`:175`）。`__lt__`（`:309`）按 `(priority, arrival_time, request_id)` 比较——优先级队列的排序依据。
   - `RequestStatus(enum.IntEnum)`（`:323`）：`WAITING / WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR / WAITING_FOR_REMOTE_KVS / WAITING_FOR_STREAMING_REQ / RUNNING / PREEMPTED / FINISHED_*`。妙处：`is_finished(status) = status > PREEMPTED`（`:345`）——用 IntEnum 顺序，一个比较就判完成。
3. **`core/sched/request_queue.py`** — waiting 队列**可插拔**：
   - `SchedulingPolicy(Enum)`（`:13`）：`FCFS` / `PRIORITY`。
   - `FCFSRequestQueue(deque)`（`:75`）：`pop_request = popleft()`，先到先服务。
   - `PriorityRequestQueue`（`:131`）：`heapq`，按 `Request.__lt__` 弹最小 `(priority, arrival_time)`。
   - `create_request_queue(policy)`（`:201`）：工厂，按 policy 选队列。
4. **`core/sched/output.py:181`** `SchedulerOutput` — `schedule()` 的产物（喂给 model runner）。核心字段：`num_scheduled_tokens: dict[str,int]`（`:193`，就是契约里那个字典）、`total_num_scheduled_tokens`（`:196`）、`scheduled_new_reqs` / `scheduled_cached_reqs`（新请求发全量、老请求发 diff，省进程间通信）、`preempted_req_ids`（`:219`）。
5. **`core/sched/scheduler.py`** — 主角：
   - `Scheduler(SchedulerInterface)`（`:68`）、`__init__`（`:69`）。队列：`self.running: list[Request]`（`:184`）、`self.waiting = create_request_queue(self.policy)`（`:181`）、`self.skipped_waiting`（`:183`，本步跳过的请求）。`self.max_num_scheduled_tokens`（`:109`）= 每步 token 上限。`self.policy`（`:175`）。
   - **`schedule()`（`:388`）**——本周核心，见下节。
   - `_preempt_request()`（`:1106`）——抢占，见下节。

---

## schedule() 到底怎么组一个 batch（`scheduler.py:388`）

开头有作者 woosuk 的关键注释（`:390-399`）：**调度器里没有"prefill 阶段"和"decode 阶段"之分**。每个请求只有 `num_computed_tokens` 和 `num_tokens_with_spec`，每步就是尽量让前者追上后者。这个抽象统一涵盖了 chunked prefill、prefix caching、spec decode。一次 `schedule()` = 一次前向。

流程：

1. `token_budget = self.max_num_scheduled_tokens`（`:408`）。PAUSED_ALL 时置 0（`:411`）。
2. **先调度 running（`:432` `while req_index < len(self.running) and token_budget > 0`）**：
   - `num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens`（`:463`），普通 decode 就是 1。
   - 受 `long_prefill_token_threshold` 和 `token_budget` 双重 cap（`:468-470`）。
   - 申请 KV：`allocate_slots(...)`（`:525`）。**返回 `None` = 没有空闲 KV 块 → 触发抢占**（见下）。
   - 记账：`num_scheduled_tokens[request_id] = num_new_tokens`（`:578`）、`token_budget -= num_new_tokens`（`:579`）。
3. **再用剩余 budget 调度 waiting（`:626` 起，`:629` `while (self.waiting or self.skipped_waiting) and token_budget > 0`）**：
   - 若这步已经发生过抢占（`preempted_reqs` 非空），整段 waiting 跳过（`:626` 的 guard）——先保住已在跑的。
   - `len(self.running) == self.max_num_running_reqs` 也停（`:630`）。
   - admit：`self.running.append(request)`（`:939`）、`token_budget -= num_new_tokens`（`:957`）、`status = RUNNING`（`:958`）。
4. **两个循环写进同一个 `num_scheduled_tokens` 字典、共享同一个 `token_budget`**——这就是 continuous batching 的实现：running 先占，剩下的额度让 waiting 挤进同一步的 batch。
5. 收尾断言：`assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens`（`:990`）、`assert token_budget >= 0`（`:992`）。

### chunked prefill（`scheduler.py:795-810`）

waiting 路径里 `num_new_tokens = request.num_tokens - num_computed_tokens`（`:795`），再被 `long_prefill_token_threshold`（`:796-798`）和 `min(..., token_budget)`（`:810`）切小。剩下的部分靠 `num_computed_tokens` 记住，下一步接着算。关键开关：`enable_chunked_prefill` 关闭且 prompt 超 budget 时，直接 `break` 不切（`:802-808`）——**chunking 只在显式开启时发生**。still-prefilling 标记：`num_computed_tokens + num_new_tokens < request.num_tokens`（`:961`）。

### preemption（`_preempt_request`, `scheduler.py:1106`）

触发点：running 循环里 `allocate_slots()` 返回 `None`（`:525-533`），即 KV 块耗尽。

- **选谁**：PRIORITY 策略选 `(priority, arrival_time)` 最大的（最低优先级，`:537-541`）；FCFS 则 `self.running.pop()`（`:562`，踢最新进 running 的）。对照模拟器里的 LIFO `self.running.pop()`。
- **怎么踢**（`:1106`）：断言原状态 RUNNING（`:1112`）→ 释放 KV 块和 encoder cache → `status = PREEMPTED`（`:1118`）→ `num_computed_tokens = 0`（`:1119`，KV 丢了要重算）→ `num_preemptions += 1`（`:1122`）→ `self.waiting.prepend_request(request)`（`:1127`，回到队首优先恢复）。
- **状态转移**：`RUNNING → PREEMPTED`，放回 waiting 队首；之后从 waiting 恢复时走 resumed 路径（`:946`）再 `→ RUNNING`（`:958`）。

## 生命周期状态机（一图记住）

```
                add_request
                    │
                    ▼
   ┌────────────  WAITING  ◄──────────────┐
   │            （队首恢复）                │ _preempt_request
   │  schedule() admit                     │ (KV 不足, RUNNING→PREEMPTED,
   │  token/KV 够                           │  num_computed_tokens=0)
   ▼                                        │
 RUNNING ─────────────────────────────► PREEMPTED
   │  update_from_output                    （回到 waiting 队首）
   │  命中停止条件 / 到 max_tokens
   ▼
 FINISHED_*  (is_finished: status > PREEMPTED)
```

（另有 `WAITING_FOR_REMOTE_KVS`(P/D KV 传输)、`WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`(结构化输出) 等阻塞态，由 `_try_promote_blocked_waiting_request`(`:2388`) 促回 WAITING/PREEMPTED，初学可先略过。）

---

## 本周验收（能答出来才算过）

1. 为什么 vLLM 说"调度器里没有 prefill/decode 阶段之分"？`num_computed_tokens` 追 `num_tokens_with_spec` 这个抽象如何统一了 chunked prefill、decode、spec decode？
2. 走一遍 `schedule()`：running 和 waiting 两个循环如何共享 `token_budget` 和 `num_scheduled_tokens`，从而把 decode 和 prefill 拼进同一个 batch？为什么先调度 running？
3. chunked prefill 靠哪个变量把长 prompt 切开、靠什么记住"切到哪了"？`enable_chunked_prefill` 关闭时会怎样？
4. 抢占什么时候触发？FCFS 和 PRIORITY 各选哪个请求当牺牲品？被抢的请求状态怎么变、KV 怎么处理、放回队列的哪个位置？（跑模拟器，用输出佐证）
5. `is_finished(status) = status > PREEMPTED` 这个技巧依赖 `RequestStatus` 的什么性质？

## 承上启下

- 第 6 周的 KV 分页/前缀复用 → 本周调度器在其之上做 continuous batching，`allocate_slots` 返回 `None` 正是第 6 周那套块分配器耗尽的信号。
- token budget + 固定 running 上限 → 呼应第 5 周 CUDA Graph 的 static shape：batch 归一到少数档位才能 replay。
- 抢占 → KV 显存是硬约束，把第 3-4 周"KV cache 有多贵"落到调度决策上。
