"""第 6 周：最小 block-hash 前缀缓存（vLLM 的复用机制）。

vLLM 用固定大小的 block + 哈希表来复用 KV cache：
  - KV cache 切成固定 block_size（如 16 token）的块。
  - 每个 block 按“它覆盖的全部 token（含前缀）”算一个 hash。
  - 哈希表 hash -> 物理 block 号。新请求逐 block 算 hash，命中就复用那个物理 block。
  - 显存不足时 LRU 驱逐空闲 block。

关键特征（对照 SGLang 的 radix tree）：
  - 复用粒度 = 整块（block_size 对齐），更粗；不足一个 block 的尾巴无法复用。
  - 匹配 = 逐 block 查哈希表，是否命中取决于“从头到这块的全部 token 是否完全一致”。
  - 实现简单、和 PagedAttention 的物理 block 天然对齐。

本文件是最小实现，读完再看真实源码：
  vllm/v1/core/block_pool.py 和 kv_cache_utils.py（block hashing）。

运行：python block_hash_cache_mini.py
"""

from __future__ import annotations


def hash_block(prefix_hash: int, block_tokens: tuple[int, ...]) -> int:
    """block 的 hash 由 (前缀 hash, 本块 token) 共同决定 —— 保证前缀完全一致才命中。

    真实 vLLM 也是把前一个 block 的 hash 纳入，形成 hash 链（见 kv_cache_utils.py）。
    """
    return hash((prefix_hash, block_tokens))


class BlockHashCache:
    def __init__(self, block_size: int = 4, num_blocks: int = 64):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))   # 空闲物理块队列
        self.hash_to_block: dict[int, int] = {}       # block_hash -> 物理块号
        self.block_refs: dict[int, int] = {}          # 物理块号 -> 引用计数
        self._access = 0

    def _blockify(self, tokens: list[int]) -> list[tuple[int, ...]]:
        """把 token 切成对齐 block_size 的整块；不足一块的尾巴丢弃（无法复用）。"""
        n_full = len(tokens) // self.block_size
        return [tuple(tokens[i * self.block_size:(i + 1) * self.block_size])
                for i in range(n_full)]

    def match_and_allocate(self, tokens: list[int]) -> tuple[list[int], int]:
        """返回 (每个逻辑块命中的物理块号, 命中的 token 数)。"""
        blocks = self._blockify(tokens)
        physical = []
        prefix_hash = 0
        matched_tokens = 0
        hit_prefix = True  # 一旦断链，后面即使 hash 巧合也不算命中
        for blk in blocks:
            h = hash_block(prefix_hash, blk)
            prefix_hash = h
            if hit_prefix and h in self.hash_to_block:
                pb = self.hash_to_block[h]
                self.block_refs[pb] = self.block_refs.get(pb, 0) + 1
                physical.append(pb)
                matched_tokens += self.block_size
            else:
                hit_prefix = False
                if not self.free_blocks:
                    break
                pb = self.free_blocks.pop(0)
                self.hash_to_block[h] = pb
                self.block_refs[pb] = 1
                physical.append(pb)
        return physical, matched_tokens


def demo():
    cache = BlockHashCache(block_size=4, num_blocks=64)

    # req1：system prompt(8 token) + 问题(4 token) = 3 个 block
    req1 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
    phys1, hit1 = cache.match_and_allocate(req1)
    print(f"req1 分到物理块 {phys1}，命中复用 {hit1} token（首次全 miss）")

    # req2：相同 system prompt(8) + 不同问题 = 前 2 个 block 完全一致
    req2 = [1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23]
    phys2, hit2 = cache.match_and_allocate(req2)
    print(f"req2 分到物理块 {phys2}，命中复用 {hit2} token")
    print(f"  -> 前 2 块（8 token）命中，物理块号和 req1 相同：{phys2[:2]} == {phys1[:2]}")
    print(f"  -> 第 3 块不同，分到新物理块")

    # 对比粒度：block_size=4 时，7 个 token 的共享前缀只能复用 4 个（1 整块）
    print("\n=== 粒度对比 ===")
    cache2 = BlockHashCache(block_size=4)
    cache2.match_and_allocate([1, 2, 3, 4, 5, 6, 7])         # 存
    _, hit = cache2.match_and_allocate([1, 2, 3, 4, 5, 6, 7, 99])
    print(f"共享前缀 7 token，block_size=4 -> 只复用 {hit} token（尾部 3 token 不足一块，弃）")
    print("  radix tree 则能复用到第 7 个 token（粒度更细）。这就是两者的核心差异。")

    print("\n要点：block-hash 实现简单、和 PagedAttention 物理块对齐，但复用受 block_size 粒度限制；")
    print("      对照真实源码：vllm/v1/core/block_pool.py, kv_cache_utils.py")


if __name__ == "__main__":
    demo()
