# 源码深读：SGLang RadixCache 前缀缓存子系统

> 目标：挑一个引擎子系统**逐行读通**，写成能复述控制流的笔记。选 SGLang 的前缀缓存 `RadixCache`——它把"多请求共享 prompt 前缀、省掉重复 prefill"这件事落成一棵基数树，是第 6 周前缀缓存那一课的真实实现。
>
> 文件：`~/github/sglang/python/sglang/srt/mem_cache/radix_cache.py`（本地 863 行，2026-08-13 版本）。以下**行号全部核对过真实源码**。
> 读源码方式沿用你的习惯（[[feedback_code_reading_strategy]]）：先抓类和字段，再顺主流程 match→insert→evict，最后记与"教科书基数树"的差异。

---

## 0. 一句话定位

RadixCache 用一棵**页对齐的基数树**保存"token 序列前缀 → KV cache 在显存里的行号(indices)"。新请求进来先 `match_prefix` 找最长公共前缀直接复用其 KV（省 prefill），生成过程中/结束时把自己的 KV `insert` 回树里给后来者用；显存不够时 `evict` 按策略把没被引用的叶子驱逐。`lock_ref` 保证"正在用的前缀"不会被驱逐。

对照第 6 周：vLLM 是 **block-hash**（按块哈希查表），SGLang 是**基数树**（细到 token/页级共享，前缀树天然支持"分叉"）。见 [[project_sglang_vs_vllm_kvcache]]。

---

## 1. 两个核心类

### RadixKey（:59-235）—— 不是裸 token 列表

树的"键"不是 `str` 也不是 `list[int]`，而是包装类，这是和教科书基数树的第一个差异：

- `token_ids: array("q")`（int64 定长数组）+ `extra_key`（命名空间，如 LoRA adapter id）+ `cache_salt`（隔离用盐）+ `is_bigram`（EAGLE 投机解码的二元组视图）+ `limit`（逻辑截断，免拷贝）。
- `__len__`（:99）：普通模式 = token 数；bigram 模式 = token 数 - 1。
- `match(other, page_size)`（:181-215）：**求最长公共前缀长度**。用**指数跳跃 + 二分**找第一个分叉点（:194-206，`t0[lo:hi] != t1[lo:hi]` 一次 C 级比较，避免 Python 逐 token 循环），最后 `(matched // page_size) * page_size` **向下取整到页**（:213-215）。
- `child_key(page_size)`（:217-229）：取前 `page_size` 个逻辑单元做**可哈希的 dict 键**，并用 `extra_key`/`cache_salt` 命名空间化——所以相同 token 不同 adapter 永不共享节点。
- `page_aligned(page_size)`（:150-154）：长度截到 page_size 整数倍。

### TreeNode（:238-300）

`__init__(id=None, priority=0)`（:242），类级 `counter=0`（:240）自增分配 `id`（:265-266）。字段：

| 字段 | 行 | 含义 |
| --- | --- | --- |
| `children` | :243 | `defaultdict(TreeNode)`，键是 `child_key`（**整页**，不是单 token） |
| `parent` | :244 | 父节点 |
| `key` | :245 | 本节点持有的那段 `RadixKey` |
| `value` | :246 | 该段对应的 KV cache 显存**行号 tensor** |
| `lock_ref` | :247 | **驱逐保护引用计数**（>0 不可驱逐） |
| `last_access_time` | :248 | LRU 时间戳 |
| `hit_count` | :251 | 命中次数 |
| `host_ref_counter` / `host_value` | :254/:256 | CPU 分层(offload)用，本类里不激活 |
| `hash_value` / `event_hash_value` | :259/:261 | 每页 SHA，用于外部 KV 事件 |
| `priority` | :263 | 优先级感知驱逐用 |

- `evicted`（:268）= `value is None`；`backuped`（:272）= `host_value is not None`。
- `__lt__`（:299-300）：`self.last_access_time < other.last_access_time`——**只做堆里的 LRU 平局裁决**，真正的驱逐序来自可插拔的 `eviction_strategy`（见 §5）。

---

## 2. 初始化：根节点永久锁定

`RadixCache.__init__`（:304-331）从 `CacheInitParams` 取 `disable / req_to_token_pool / token_to_kv_pool_allocator / page_size / is_eagle / eviction_policy` 等；`get_eviction_strategy(...)`（:328）拿到可插拔驱逐策略；`evictable_leaves = set()`（:330）后 `reset()`。

`reset()`（:353-374）建根：
- `root = TreeNode(priority=-sys.maxsize)`（:355）——最小优先级，任何真实优先级都能盖过它。
- `root.key = RadixKey(array("q"))`、`root.value = []`、**`root.lock_ref = 1`**（:359，根永久锁定，永不被驱逐）。
- `evictable_size_ = 0`、`protected_size_ = 0`（:361-362）——两个计数器，见 §6。
- 预建 `_empty_match_result`（:364-373）：空 tensor + 三个 node 字段都指向 root（空命中的快速返回）。

---

## 3. match_prefix：查最长前缀（省 prefill 的关键）

`match_prefix(params)`（:376-434）：
1. `maybe_to_bigram_view`（:414）——EAGLE 时切二元组视图。
2. `disable` 或空键 → 直接返回 `_empty_match_result`（:416-417）。
3. `key.page_aligned(page_size)`（:419）——**先页对齐**。
4. `value, last_node = self._match_prefix_helper(root, key)`（:424）。
5. `value` 非空则 `torch.cat(value)`（:426）把各段行号拼成一维 int64 tensor。
6. 返回 `MatchResult(device_indices, last_device_node, last_host_node, best_match_node)`（:429-434）——**本基类里后三个 node 是同一个对象**（host 分层未区分，docstring :401 明说 "currently the same"）。

`_match_prefix_helper(node, key)`（:678-702）——逐层下降：
```
child_key = key.child_key(page_size)                    # :682 取首页做 dict 键
while len(key) > 0 and child_key in node.children:      # :685
    child = node.children[child_key]
    prefix_len = child.key.match(key, page_size)        # :688 本段能匹配多长
    if prefix_len < len(child.key):                     # :689 段内部分匹配
        new_node = self._split_node(child.key, child, prefix_len)  # 从中间切开
        value.append(new_node.value); node = new_node; break
    else:                                               # :694 整段命中，继续往下
        value.append(child.value); node = child
        key = key[prefix_len:]                          # 削掉已匹配部分
        child_key = key.child_key(page_size)
return value, node
```
关键点：**部分命中要 `_split_node` 把节点从中间切开**，让匹配边界精确落在节点边界上（这也是基数树相较哈希表的优势：能表达"共享一半"）。

---

## 4. _split_node：把一个节点从中间劈开（:704-727）

结构从 `parent → child` 变成 `parent → new_node → child`：
- `new_node = TreeNode(priority=child.priority)`（:707），继承 `hit_count`（:708）。
- `new_node.children = { key[split_len:].child_key(page_size): child }`（:709）——原 child 成为唯一孩子，按剩余段做键。
- `new_node.parent = child.parent`（:710）；**`new_node.lock_ref = child.lock_ref`**（:711，锁计数一并复制，保护不丢）。
- `new_node.key/value = child 的前 split_len`（:712-713，`value` 用 `.clone()`）。
- 改写 child：`child.parent = new_node`、`child.key/value = 后半段`（:714-716）。
- 祖父指向 new_node：`new_node.parent.children[key.child_key] = new_node`（:717）。
- `hash_value` / `event_hash_value` 若已算过则一并切分（:720-725）。

注意：`_split_node` 不动 `evictable_size_`，也不重跑 `_update_leaf_status`——因为 new_node 不是叶子，叶子集合不变。

---

## 5. insert：把新序列未缓存的尾巴写进树

`insert(params)`（:436-456）→ 页对齐、`value` 截到 `len(key)`（:447-448）→ `_insert_helper`。

`_insert_helper(node, key, value, priority, chunked)`（:737-790）：
- 沿路 `node.priority = max(node.priority, priority)`（:751，**优先级向上传播取大**）。
- 匹配下降循环（:758-776）：逐层匹配，`total_prefix_length += prefix_len`，削 `key`/`value`；段内部分匹配就 `_split_node`（:766-767）。
- **未缓存尾巴**（:777-789）：还有剩余 key，就新建叶子 `TreeNode(priority)`（:778），`value = value.clone()`（:781），挂到 `node.children[child_key]`（:783），**`evictable_size_ += len(key)`**（:784），对 parent 和新叶子都 `_update_leaf_status`（:785-786），发 store 事件（:788）。
- 返回 `(total_prefix_length, node)`——`prefix_len` 告诉调用者"这段里有多少是本来就在树里的重复"。

---

## 6. evict：显存不够时驱逐（LRU/优先级可插拔）

`evict(params)`（:592-620）：
```
leaves = list(self.evictable_leaves)                    # :598 维护好的叶子集合，不是现场 DFS
heap = [(eviction_strategy.get_priority(n), n) for n in leaves]; heapify   # :599-602
while num_evicted < num_tokens and heap:                # :605
    _p, x = heappop(heap)                               # :606 取"最该驱逐"的叶子
    token_to_kv_pool_allocator.free_segment(x.value, 0) # :609 真正释放 KV 显存行
    num_evicted += len(x.value)
    self._delete_leaf(x)                                # :611 从树上摘除
    if len(x.parent.children) == 0 and x.parent.lock_ref == 0:  # :613 父变叶子且没锁
        heappush(heap, (get_priority(x.parent), x.parent))      # 重新入堆，可继续驱逐
```
- **驱逐序**来自可插拔 `eviction_strategy.get_priority`（:600/:614），不是纯 LRU；`__lt__` 的 LRU 只做平局裁决。
- **锁保护**：被锁节点根本不在 `evictable_leaves` 里——`_update_leaf_status`（:820-833）在 `node.evicted or node.lock_ref>0`（:821）或"还有未驱逐孩子"（:826-830）时把节点移出叶子集合；父节点重入堆也显式要求 `lock_ref==0`（:613）。
- `_delete_leaf`（:810-818）：从父 `children` 弹出（带 assert），`evictable_size_ -= len(node.key)`（:815），并 `_update_leaf_status(parent)` 让父可能成为新的可驱逐叶子。

---

## 7. lock_ref：请求怎么"锁住"自己在用的前缀

这是把调度（第 7 周）和缓存连起来的关键——**正在被某请求使用的前缀不能被驱逐**。

`cache_unfinished_req(req, chunked)`（:515-583）——每步之间提交 KV：
1. 取 `token_ids = req.get_fill_ids()`、对应 `kv_indices`（:520-523），建页对齐 RadixKey。
2. `insert(...)` 得 `new_prefix_len`（:534-542）；把已在池中的重复段 `free_segment`（:544-547）。
3. **重新 `match_prefix`**（:550）拿到规范化后的 `new_indices, new_last_node`，断言长度一致（:555-557），写回 `req_to_token_pool`（:559-562）。
4. **锁交接**：`dec_lock_ref(req.last_node)` 再 `inc_lock_ref(new_last_node)`（:570-571）——放掉旧前缀锁，锁住新终点，让它熬过步与步之间的驱逐。
5. 更新 `req.prefix_indices` / `req.last_node`（:576-583）。

`cache_finished_req(req, *, kv_len_to_handle)`（:458-513）——请求结束：
- `disable_finished_insert` 时不插树（:463-464，确定性模式）。
- 正常：建 key、`insert`，`freed_end = result.prefix_len`；`free_segments` 释放重复段和未对齐尾巴（:501-509）。
- **`dec_lock_ref(req.last_node)`**（:512-513）——请求走了，松开它锁住的前缀，这部分前缀就变回可驱逐（但树节点还在，给后来者复用）。

`inc_lock_ref(node)`（:622-635）：从 node 一路走到 root，节点 `lock_ref` 从 0→1 时把 `len(node.key)` **从 `evictable_size_` 挪到 `protected_size_`**（:628-631），再 `lock_ref += 1`。
`dec_lock_ref(node)`（:637-656）：对称地在 1→0 时挪回来。

## 8. 两个计数器

- `evictable_size()`（:658）= `evictable_size_`：可被驱逐的 token 数。insert 时 +（:784），delete_leaf 时 −（:815）。
- `protected_size()`（:661）= `protected_size_`：被锁住不可驱逐的 token 数。
- 二者只在 `lock_ref` 的 0↔1 边界互相搬运。所以 `lock_ref>0` ⟺ 该节点 token 计入 `protected_size_` 且不在 `evictable_leaves` ⟺ `evict` 够不到它。**这就是"正在用的前缀不会被驱逐"的机制底座。**

---

## 9. 与"教科书基数树"的 7 个差异（重点记这个）

1. **页对齐、非逐 token 边**：键截到 `page_size` 整数倍，匹配向下取整到页（:213-215），孩子按整页 `child_key` 索引——分裂/匹配都发生在页边界。
2. **命名空间键**：`extra_key + cache_salt` 让同 token 不同 adapter/salt 的序列彼此隔离，永不共享节点（`_check_compatible` :169-179）。
3. **Bigram/EAGLE 视图**：投机解码时用 token 二元组覆盖，改变有效长度与 value 配对（:156-167）。
4. **Host/CPU 分层是留的钩子，本类未启用**：`host_value`/`host_ref_counter` 等字段在，但本基类 `last_host_node == last_device_node`（:432）；真正的 offload 在子类。
5. **驱逐可插拔，非纯 LRU**：序来自 `eviction_strategy.get_priority` + 节点 `priority` 字段，`__lt__` 的 LRU 只是平局裁决。
6. **维护式叶子集合，非现场 DFS**：`evict` 直接读 `evictable_leaves`，由 `_update_leaf_status` 增量维护。
7. **ChunkCache / `chunked` 标志**：本文件只体现为 `chunked` 参数（跳过 hit_count 膨胀，:729-735）与 `disable`/`create_simulated` 退化模式；ChunkCache 类在另一文件。

---

## 10. 读完能答什么

- 多轮对话为什么 SGLang 前缀复用强？→ 基数树能表达"共享一半再分叉"，`_split_node` 精确切边界，细到页级复用（呼应第 6 周、[[project_sglang_vs_vllm_kvcache]]）。
- "正在用的前缀不被驱逐"怎么实现？→ `lock_ref` + `protected_size_` + `evictable_leaves` 三者联动（§6-8）。
- 前缀缓存和调度怎么衔接？→ `cache_unfinished_req` 的 dec/inc_lock_ref 锁交接，跨步保住 KV（§7，接第 7 周调度器）。
- 与 vLLM block-hash 的取舍？→ 树更灵活（任意前缀分叉）但节点/指针开销大；块哈希更简单但共享粒度受块约束。
