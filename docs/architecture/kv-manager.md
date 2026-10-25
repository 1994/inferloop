# KV Cache Manager 与页缓存

KV 管理分为 scheduler 的逻辑页预算和 backend owner 的物理页 / 私有状态。共享前缀生命周期由 `PrefixCache<V, M>` 管理，`M: PrefixPolicy` 选择匹配与最近使用索引表示，
`PrefixReuse<V>` 实现后端的获取/恢复操作。`KvCacheManager<P>` 在此之上组合 BlockPool，
为共享页提供 lease 适配；CUDA 则提供连续设备快照适配。公共层不调用 CUDA / Metal API。

## 共享前缀生命周期与扩展

```text
PrefixCache<V, M>
  publication / exact-token guards / compatibility / byte & entry budget / LRU / counters
  M: PrefixPolicy
    BlockHash      块指纹查找 + 树形 recency（共享页默认）
    SnapshotScan   快照扫描 + 可复用容量的 recency 队列（CUDA）
  PrefixReuse<V>
    PageReuse      clone 快照句柄并原子 retain 共享页
    CUDA Restore   恢复 target/draft/FP8/hidden，settle 设备复制
```

- `PrefixSnapshot<V>` 只包含 tokens、字节报价和后端 payload；`V` 不要求 Clone 或页 lease。
  发布必须已越过设备完成边界。被拒绝、替换或淘汰的 payload 返回给 owner：页适配器 release
  cache 自己的引用，独占设备快照由 CUDA 正常销毁。CUDA 保持每 4096 tokens 的发布粒度。
- matcher 从只读 `PrefixSearch` 取得候选；公共候选迭代器先检查 maximum、完整 token 前缀相等
  和后端兼容性，再交给策略选择最长候选。CUDA 不需要为了共用生命周期而逐块哈希长 query。
- 复用按“匹配兼容候选 → 后端获取/恢复 → 成功提交 hit/reused_tokens/LRU”执行。
  后端返回错误或没有兼容项时，源快照仍在缓存，预算与 LRU 不改变；lookups 记录实际查询尝试。
  CUDA 的 unknown/poisoned/slot/home/非 logits 状态在查询前排除。
- hook 的成功边界由后端资源协议定义：共享页是获取有效 lease 后返回句柄，Metal 随后在 command
  buffer 编码复制；CUDA 当前同步恢复并 drain 后返回。失败时后端须回滚自己的引用/排队工作，
  或将部分恢复的 destination 标为 poisoned 并 settle 复制，公共层不虚构 GPU fence 完成。
- 不兼容的长快照不阻止选取较短的兼容快照。CUDA 不再 take/reinsert，也不在兼容性检查前增加 hit。
  两种策略使用相同 publication、资源 hook、统计和 LRU 语义。`PrefixRecency` 只定制
  最近使用索引的表示，不参与预算、资源释放或 hit 提交；CUDA 的成功查询不新增树节点分配。

## 所有权与操作

| 操作 | 合约 |
|---|---|
| `page_growth` / `prepare_append` | O(1) growth 报价；原子预留新增页与共享尾页 COW，返回复制计划 |
| `fork_pages` | 保留共享 lease，不复制完整 KV |
| `attach_prefix` | 匹配 namespace 与最长完整块前缀，原子取得引用 |
| `publish_prefix` | cache 取得独立引用，LRU 淘汰只释放自身所有权 |
| `pin` / `release` | GPU copy / reader 期间保留；最后一个 fence 与引用归零后可回收 |
| `inspect` / `check_owners` | 冷路径核对 unique block、refcount、generation、owner 与 cache |

lease 为 `(OwnerId, index, generation)`，跨池、旧 generation 和重复确认都会被拒绝。pool owner 不可 clone，页表使用设备 `u32` index。逻辑增页是批次事务：物理编码失败时回滚新增 lease 与页表；已发布的设备操作保留其 pin 直到 fence。引用计数校验使用固定 marker，append / COW 使用常驻 lease scratch。分配证据范围见 [CPU 性能测量](../guides/cpu-performance.md)。

## StateRecipe 与预算

加载期从编译图生成 checked `StateRecipe`，分别布局私有 conv / delta、hidden / logits、tokens / page table / probes、host mirror / readback / lease 与共享 KV block。Metal 用 recipe 分配和报价，reset 复用已分配 buffer。

准入保留私有状态与最大 token / page descriptor 容量，KV 按实际计算增页。单请求最大计算长度必须能独占页池，tenant 配额按最坏 token / page 数计算；只分配零 KV 的准入不能被当作零状态内存。

```text
available_blocks = free_blocks + unpinned_cache_only_blocks
```

active 与 cache 可以共享同一块，容量按 unique block 统计；pinned cache-only 不可回收。压力下先淘汰 cache-only，再由调度策略选择安全的重算 victim。固定驻留字节、私有状态、cache snapshot 与 host mirror 分别统计。

## Metal 布局、COW 与 prefix

- 每层 K/V plane 形状为 `[blocks, block_size, kv_width]`，sequence 用持久页表定位块；conv / delta / hidden / logits 为私有设备 buffer。attention 的 KV 写入与读取分 pass。
- 写入共享尾页时用 GPU blit 复制各层 K/V，保留源页 pin，然后原子切换写入方页表。失败时恢复旧页表与新增 lease；host 侧发布顺序不能替代 GPU 复制依赖。
- Prefix namespace 绑定精确 weights、backend、precision、layout 与实例内 token 指纹。只发布完成 GPU 边界的完整块，并保存对应 conv / delta / readout 状态；命中时共享 KV，并在 GPU 内恢复私有状态，普通路径不读回完整张量。最后 prompt token 需要重算以取得最终 logits。
- 缓存受条数 / 字节与 LRU 约束。Full snapshot 可用于 compact Generate，compact snapshot 不支持 Full readout。取消与淘汰独立于活动引用，不能释放仍被 flight 或 CPU reader 使用的状态。

## Checkpoint 与扩展边界

quiescent checkpoint 冷读回实际状态并记录共享页关联；恢复时先校验身份、shape、有限值、容量、generation 与 payload 一致性，再创建新池，同一共享页只分配一次。失败保留现有状态；prefix cache 与 debug history 不作为恢复缓存。

当前物理分配与淘汰的 owner 位于 backend。统一到 scheduler 决策 ledger、异构状态组、分层 offload 与远程 PD transfer 的目标见[能力边界](backends.md)。
