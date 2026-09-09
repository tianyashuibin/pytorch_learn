# 第 6 周：切入引擎——SGLang / vLLM 的 KV cache 管理 ★★★

## 目标

从 PyTorch 底座跨进真实引擎源码，理解 KV cache 的**分页存储**与**前缀复用**两件事，并看清 SGLang（radix 前缀树）和 vLLM（block 哈希链）在复用粒度上的核心差异。

> 本周源码路径基于你本地的：
> - SGLang：`~/github/sglang/python/sglang/srt/mem_cache/`
> - vLLM：`~/github/vllm-main/vllm/v1/core/`
> 对照笔记 [[project_sglang_vs_vllm_kvcache]]。

## 先跑 mini 实现，再读真实源码

真实源码有大量工程细节（多种 pool、LoRA namespace、分页对齐、锁引用计数），直接读容易迷路。**先跑本目录两个最小实现建立机制骨架**，再带着骨架去读真实源码，效率高得多（对照 [[feedback_code_reading_strategy]] 自顶向下）。

- `radix_cache_mini.py` — 最小 RadixCache（前缀树）：`match_prefix` 走最长公共前缀、`_split` 劈节点、LRU `evict_one`。
- `block_hash_cache_mini.py` — 最小 block-hash 缓存（vLLM 式）：固定 block_size、hash 链、物理块队列。

```bash
cd week6
python radix_cache_mini.py       # 看前缀树复用到 token 级
python block_hash_cache_mini.py  # 看 block 哈希只能整块复用
```

跑出来的关键对比：共享前缀 7 个 token，radix 复用到第 7 个，block-hash（block_size=4）只复用前 4 个。**这就是两者复用粒度差异的本质。**

---

## SGLang 源码阅读地图（radix 前缀树 + 张量 free-list）

阅读顺序（`srt/mem_cache/`）：

1. **`base_prefix_cache.py:230`** `BasePrefixCache` — 接口契约：`match_prefix / cache_finished_req / cache_unfinished_req / evict / inc_lock_ref / dec_lock_ref`。先看懂契约。
2. **`memory_pool.py`** — 两级存储，理解 "token pool vs KV pool"：
   - `ReqToTokenPool`（`:256`）：请求 → 每个位置的 token 槽位。核心是 2-D 张量 `req_to_token`，形状 `(size+1, max_context_len)`，`free_slots` 是 Python list。
   - `MHATokenToKVPool`（`:1755`）：真正存 K/V 张量，per-layer 的 `k_buffer`/`v_buffer`，`set_kv_buffer`（`:2327`）按槽位写入。（MLA 变体 `MLATokenToKVPool:3932`。）
3. **`allocator/token.py:28`** `TokenToKVPoolAllocator` — 发放 KV 槽位。free 结构是**1-D 张量** `free_pages = torch.arange(1, size+1)`；`alloc` 切片头部、`free` 用 `torch.cat` 拼回。
4. **`radix_cache.py`** — 复用主角：
   - `RadixCache`（`:303`）、`TreeNode`（`:238`，字段 `children/parent/key/value(KV槽位)/lock_ref/last_access_time`，`__lt__` 按 `last_access_time` 比较）。
   - 匹配：`match_prefix`（`:376`）→ `_match_prefix_helper`（`:678`），沿树走最长公共前缀，部分匹配时 `_split_node`（`:704`）在边界劈开节点。
   - 驱逐：`evict`（`:592`），对可驱逐叶子建**最小堆**（默认按 `last_access_time` = LRU），弹叶子、`free_segment` 释放 KV、把变成叶子的父节点再入堆。`inc/dec_lock_ref` 保护正在用的前缀不被驱逐。
5. **`chunk_cache.py:35`** `ChunkCache` — 对照组：`match_prefix` 返回空、`insert`/`evict` 都是 no-op（不复用）。看它理解"关掉前缀复用长什么样"。

## vLLM 源码阅读地图（block 哈希链 + 双向 free 队列）

阅读顺序（`v1/core/`）：

1. **`kv_cache_utils.py`** — 原语：
   - `KVCacheBlock`（`:118`）：块描述符，自身带 `prev_free_block/next_free_block` 指针（它就是链表节点）。
   - `FreeKVCacheBlockQueue`（`:179`）：手写双向链表 free 队列，带假头尾哨兵，O(1) 中间摘除；**LRU 顺序**（front 最久未用）。
   - `hash_block_tokens`（`:577`）：hash `(parent_block_hash, 本块token, extra_keys)` —— **哈希链**，前缀一致才产生相同链。`get_request_block_hasher`（`:673`）逐块填 `request.block_hashes`。
2. **`block_pool.py:144`** `BlockPool` — 物理块分配器：`free_block_queue` + `cached_block_hash_to_block`（hash→块 复用表，`:185`）。`get_new_blocks`（`:542`）从 LRU front 弹块并 `_maybe_evict_cached_block`；`free_blocks`（`:614`）无 hash 的块前插（优先驱逐）、有 hash 的后插；`touch`（`:597`）把复用块移出 free 队列。
3. **`kv_cache_manager.py:110`** `KVCacheManager` — 入口：`get_computed_blocks`（`:202`，命中前缀）、`allocate_slots`（`:244`，分配）。
4. **`single_type_kv_cache_manager.py:540`** `FullAttentionManager` — `find_longest_cache_hit`（`:542`）沿哈希链逐块查 `get_cached_block`，**第一个 miss 就 break**（哈希链保证后面不可能命中）。`req_to_blocks` 存请求→块映射。

---

## 两者对比（一句话）

| | SGLang | vLLM |
| --- | --- | --- |
| 复用结构 | radix 前缀**树** | block **哈希链** |
| 匹配方式 | 最长公共前缀路径 + 节点劈分 | 逐块查哈希，首个 miss 即停 |
| 复用粒度 | 细到 token（页对齐时按页） | 整块（block_size 对齐），尾部零头弃 |
| 驱逐 | 可驱逐叶子的最小堆（LRU/priority） | free 队列的 LRU（双向链表） |
| 槽位分配器 | 1-D 张量 free-list（`free_pages`） | 双向链表 free 队列（`FreeKVCacheBlockQueue`） |

## 本周验收（能答出来才算过）

1. SGLang 的 "token pool" 和 "KV pool" 各存什么？为什么要分两级（`ReqToTokenPool` vs `MHATokenToKVPool`）？
2. 走一遍 SGLang `match_prefix`：命中一半一个节点时为什么要 `_split_node`？
3. vLLM 的 block hash 为什么要把 `parent_block_hash` 纳入？为什么 `find_longest_cache_hit` 能"首个 miss 就 break"？
4. radix 树 vs block 哈希，复用粒度差在哪？各自的驱逐用什么数据结构？（跑两个 mini 实现，用输出佐证）
5. 结合第 4 周：这里的 KV 槽位/块，如何配合第 5 周 CUDA Graph 的固定地址要求？

## 承上启下

- 第 3 周连续预分配 KV cache → 第 4 周 paged demo 指出浪费 → 本周看真实引擎的分页 + 复用实现。
- 前缀复用的 KV 布局 → 第 7 周调度器要在这之上做 continuous batching。
- KV 槽位/块的固定物理地址 → 呼应第 5 周 CUDA Graph 三约束。
