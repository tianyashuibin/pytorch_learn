"""第 6 周：最小 RadixCache（SGLang 的前缀复用机制）。

SGLang 用一棵 radix tree（压缩前缀树）来复用 KV cache：
  - 树的每个节点存一段 token 序列（key）及其对应的 KV cache 位置（value）。
  - 新请求进来，从根沿着 token 匹配最长公共前缀 -> 命中的部分直接复用 KV，不重算。
  - 显存不够时按 LRU 从叶子驱逐。

关键特征（对照 vLLM 的 block-hash）：
  - 复用粒度 = 任意长度前缀（字符级/ token 级，可在节点内部“劈开”），更细。
  - 匹配 = 沿树走最长公共路径。

本文件是把机制讲清楚的最小实现（用 token id 列表模拟，KV 位置用占位整数），
读完再去看真实源码：sglang/srt/mem_cache/radix_cache.py。

运行：python radix_cache_mini.py
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TreeNode:
    # 这个节点相对父节点新增的一段 token（radix 树的“边”压缩成节点里的一段）
    key: list[int] = field(default_factory=list)
    # 这段 token 对应的 KV cache 槽位（真实里是 token->slot 索引；这里用占位）
    value: list[int] = field(default_factory=list)
    children: dict[int, "TreeNode"] = field(default_factory=dict)  # 按首 token 索引
    parent: "TreeNode | None" = None
    last_access: int = 0  # 用于 LRU 驱逐


class RadixCache:
    def __init__(self):
        self.root = TreeNode()
        self._time = 0

    def _tick(self) -> int:
        self._time += 1
        return self._time

    # ---- 匹配最长公共前缀 ----
    def match_prefix(self, tokens: list[int]) -> tuple[list[int], TreeNode]:
        """返回 (命中的 KV 槽位列表, 命中到的最深节点)。命中部分可直接复用，无需重算。"""
        node = self.root
        matched_value: list[int] = []
        idx = 0
        while idx < len(tokens):
            first = tokens[idx]
            if first not in node.children:
                break
            child = node.children[first]
            # 逐 token 比对这条边上的 key
            shared = _common_len(child.key, tokens[idx:])
            matched_value.extend(child.value[:shared])
            idx += shared
            child.last_access = self._tick()
            if shared < len(child.key):
                # 只匹配了边的一部分 -> 命中就到这，节点可被“劈开”（见 insert）
                return matched_value, child
            node = child
        return matched_value, node

    # ---- 插入新序列（把没命中的尾巴写进树）----
    def insert(self, tokens: list[int], values: list[int]) -> None:
        node = self.root
        idx = 0
        while idx < len(tokens):
            first = tokens[idx]
            if first not in node.children:
                # 直接挂一个新叶子
                leaf = TreeNode(key=tokens[idx:], value=values[idx:],
                                parent=node, last_access=self._tick())
                node.children[first] = leaf
                return
            child = node.children[first]
            shared = _common_len(child.key, tokens[idx:])
            if shared < len(child.key):
                # 需要劈开 child：公共段成为新中间节点，原 child 变其子节点
                self._split(node, child, shared)
                child = node.children[first]
            idx += shared
            node = child
            node.last_access = self._tick()
        # 完全被已有路径覆盖，无需新增

    def _split(self, parent: TreeNode, child: TreeNode, at: int) -> None:
        """把 child 在位置 at 劈成两段：前段留在树上，后段成为它的子节点。"""
        mid = TreeNode(key=child.key[:at], value=child.value[:at],
                       parent=parent, last_access=child.last_access)
        child.key = child.key[at:]
        child.value = child.value[at:]
        child.parent = mid
        mid.children[child.key[0]] = child
        parent.children[mid.key[0]] = mid

    # ---- LRU 驱逐（显存不足时从最久未用的叶子开始）----
    def evict_one(self) -> list[int] | None:
        """驱逐一个最久未访问的叶子，返回被释放的 KV 槽位（可回收）。"""
        leaf = self._lru_leaf(self.root)
        if leaf is None or leaf is self.root:
            return None
        freed = leaf.value
        del leaf.parent.children[leaf.key[0]]
        return freed

    def _lru_leaf(self, node: TreeNode) -> TreeNode | None:
        if not node.children:
            return node
        best = None
        for c in node.children.values():
            cand = self._lru_leaf(c)
            if cand and (best is None or cand.last_access < best.last_access):
                best = cand
        return best


def _common_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def demo():
    cache = RadixCache()
    # 请求 1：system prompt + 问题 A
    req1 = [1, 2, 3, 4, 10, 11]      # 1-4 是共享 system prompt
    cache.insert(req1, values=list(range(100, 106)))
    print("插入 req1:", req1)

    # 请求 2：同样的 system prompt + 不同问题 B
    req2 = [1, 2, 3, 4, 20, 21]
    matched, node = cache.match_prefix(req2)
    print(f"\nreq2 = {req2}")
    print(f"命中前缀 KV 槽位 = {matched}  （前 4 个 token 的 KV 直接复用，不重算！）")
    print(f"只需为新增的 {req2[len(matched):]} 计算 KV")
    cache.insert(req2, values=matched + [200, 201])

    # 请求 3：更长的共享前缀
    req3 = [1, 2, 3, 4, 10, 11, 12]  # 和 req1 共享 1-4-10-11
    matched, _ = cache.match_prefix(req3)
    print(f"\nreq3 = {req3}")
    print(f"命中前缀 KV 槽位 = {matched}  （连 10,11 也复用了）")

    print("\n要点：radix tree 按最长公共前缀复用，粒度细到 token；")
    print("      多轮对话 / 相同 system prompt 的高并发场景，命中率高、省大量 prefill。")
    print("      对照真实源码：sglang/srt/mem_cache/radix_cache.py")


if __name__ == "__main__":
    demo()
