# 容器性能评估：保留调度树，优化前缀缓存

日期：2026-10-24（America/Los_Angeles）。代码与测量均在 45（archlinux）完成，
基点 `2bec30a92510d5a4b913a5a92b3112e3b0cc52ad`。

## 决策

调度队列继续使用预热节点的 `btree-slab 0.6.1`；不引入 bbtree 生产依赖。
前缀缓存的摘要索引从 `BTreeMap` 改为已有的 `ahash 0.8.12` 提供的 `AHashMap`。
其 `BTreeSet<(touch_clock, digest)>` LRU 保持有序，淘汰顺序不依赖哈希迭代。
其余容器没有足够证据支持替换，逐项理由见后文。

[bbtree 上游](https://github.com/XiangpengHao/bbtree)的通用性能说明不是本项目的性能结论。
本次测试 crates.io 的 **0.1.5 单线程 unsync 版本**，不是并发 BBTreeMap。
实际源码中 `clear()` 销毁节点，没有可直接替代当前预热 slab 的 reserve 接口；
可自定义 Allocator，但需要新增 unsafe 页池及容量证明，本次实测收益不足以支持该复杂度。
接口与实现见 [0.1.5 源码](https://docs.rs/crate/bbtree/0.1.5/source/src/unsync/map/mod.rs)。

## 测量方法

Intel Core i9-13900KF，Linux x86_64，Rust 1.99.0，release / thin LTO / codegen-units=1，
默认目标特性；使用与现有 CPU 门禁相同的 mimalloc 分配计数器。
绑定 CPU 8（性能核，SMT 同胞 9），保持 powersave governor、睿频与桌面任务现状，未独占整机。
构建在测量前完成，测量进程串行，无其他本任务的编译/测试并发。

容器对比使用 7 轮、每轮 100,000 次复合操作，轮换容器顺序。
前缀缓存的前后对比再执行 **5 组交替顺序的二进制配对**，每组两侧均有 7 轮；
表格报告各进程中位数的中位数。原始轮次、分配计数、checksum、编译器、源码及二进制
SHA256 保存在[证据 JSON](artifacts/container-performance-2026-10-24.json)。
基准直接编译生产 OrderedSet 与 PrefixCache；工具、场景边界和命令见
[容器基准说明](../../tools/bench/containers/README.md)。

基线只把 cache.rs 恢复到基点后构建，随后恢复候选源码并构建；两侧使用相同 harness、锁文件、
编译器与 profile。基线 manifest 保留同一个已存在的 ahash 依赖边（基线缓存代码不使用它）。
结果是 CPU 局部操作成本，含 SHA 链、token 比较与 LRU touch，**不等于模型或服务加速比**。

## 排序索引：bbtree 不满足当前替换条件

使用公平调度键 `(u64, u64, RequestId, usize)`；每次删除、插入、first、lower bound。
生产 OrderedSet 使用其完整的预热和范围查询实现。单位 ns/复合操作；分配数为每轮 100,000 次操作。

| 容量 | btree-slab | bbtree unsync | std BTreeSet | slab 分配 | bbtree 分配 | std 分配 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 42.0 | 48.5 | 31.0 | 0 | 0 | 0 |
| 64 | 52.9 | 54.3 | 32.9 | 0 | 4,166 | 14,286 |
| 1,024 | 66.2 | 69.5 | 58.2 | 0 | 4,399 | 16,617 |
| 8,192 | 81.7 | 83.6 | 74.3 | 0 | 4,399 | 16,659 |

当前调度树的关键约束是构造后持续增删不 alloc/realloc/dealloc。
只替换成标准库或 bbtree 默认 allocator 会破坏该约束；这一场景中 bbtree 也未得到明显速度收益。
多字段键按默认 Ord 比较，不专门实现 InlineKey；即使专门编码键能改善速度，仍需另解节点复用问题。
该工作负载是固定规模 churn，键逐渐递增；不把结果外推为所有键分布都更慢。

## 精确查找：前缀缓存适合哈希索引

32 字节分散键；一次复合操作为命中、未命中、删除、插入。单位 ns/复合操作。

| 容量 | std BTreeMap | bbtree | std HashMap + ahash | hashbrown + ahash | BoundedMap |
|---:|---:|---:|---:|---:|---:|
| 16 | 50.2 | 63.4 | 20.5 | 17.8 | 36.0 |
| 64 | 67.2 | 93.9 | 18.8 | 18.1 | 37.0 |
| 1,024 | 224.2 | 135.5 | 18.5 | 18.4 | 39.8 |
| 8,192 | 284.4 | 144.4 | 26.5 | 26.4 | 42.2 |

bbtree 对大摘要索引优于 std BTreeMap，但仍落后于哈希表。
std 与直接 hashbrown 都是 SwissTable 系列，不为一个没有整体证据的微小差别增加生产直接依赖。
随机哈希种子会影响局部小表结果，16 条目的差异不能推导为独立库的稳定整体收益。
两个预留容量的可增长哈希表在每轮 churn 都出现一次重新分配哈希表（1 alloc + 1 dealloc，realloc 为 0），因此不能直接替代零分配 BoundedMap。

实际 PrefixCache 保留哈希链及 token 校验，只替换摘要索引。单位 ns/调用：

| 条目 | 2 tokens lookup+touch 前→后 | 降幅 | 128 tokens / block16 lookup+touch 前→后 | 降幅 |
|---:|---:|---:|---:|---:|
| 16 | 88.1 → 80.5 | 8.6% | 720.0 → 689.3 | 4.3% |
| 64 | 102.3 → 78.6 | 23.2% | 808.0 → 700.8 | 13.3% |
| 1,024 | 206.7 → 101.4 | 51.0% | 1,138.6 → 722.2 | 36.6% |
| 8,192 | 248.9 → 123.9 | 50.2% | 1,397.2 → 784.9 | 43.8% |

2-token contains 在 1,024 / 8,192 条目下降 36.8% / 46.5%；16 条目为 53.2 → 53.6 ns（+0.7%，
视为持平）。128-token contains 的 SHA 成本占主导，降幅 0.9%–6.8%。
所有 5 组配对的 lookup+touch 都有改善；大表提升不依赖某一轮偶然结果。
本次未测实际模型服务的 prefix-hit 分布、RSS 或 P99，不能声称服务吞吐有同等提升。

## 其他容器检查

| 位置/类型 | 用途与约束 | 决策 |
|---|---|---|
| core BoundedMap / BoundedSet | 固定容量、错误返回、借用查询、序列化，churn 不增长或 rehash | 保留；普通 HashMap/hashbrown 的分配记录不满足契约；不换 FxHash/nohash 弱散列 |
| core Arena、请求/状态 slot Vec | generation 防 stale handle、固定容量 free list、直接下标 | 保留；slab/slotmap 需要另证明句柄/序列化语义，未见可归因收益 |
| scheduler BinaryHeap / CandidateScratch | 预分配、仅取优先级头部与 tenant merge，generation 标记去重 | 保留；不需要范围查询的堆没有更换为树的理由 |
| BlockPool free VecDeque、runtime pending/history、observe pages | FIFO、环形裁剪与已知容量，部分 Arc 页快照 | 保留；连续 ring 已适配访问模式，SmallVec 会改变内联体积及溢出行为 |
| core event rtrb | 固定容量 SPSC，生产者/消费者所有权分离 | 保留；没有多生产者需求，通用 MPMC/无界队列增加无用同步与分配 |
| PrefixCache LRU BTreeSet | 有序取最老、touch 与时钟回绕；显式保留 `(clock,digest)` tie break | 保留；哈希表不能取代有序淘汰，基准已包含其现有分配成本 |
| IR/config、compiler、model/package、agent registry 的 BTreeMap/Set | 构建/加载/校验、稳定输出和 checkpoint；部分公开字段与 serde 接口 | 保留；无热路径证据，批量改型会扩散 API/输出变化 |
| observe/runtime 的 parents、snapshot、trace 聚合树 | 追踪、历史导出或一致性检查 | 保留；先测实际规模与导出瓶颈，不能把请求队列微基准外推到这里 |
| CUDA/Metal executor 的 states/tensors/weights 树 | 设备状态、加载/图构建、部分确定性遍历；CUDA resident 热表另有预分配数组 | 保留；本次没有可归因的 GPU/服务证据；Metal 未在 45 执行 |
| CUDA cublaslt 的标准 HashMap | device+stream context 与图构建 activation scratch 缓存 | 保留；不是图 replay 的 per-token 查找，计时外 warmup 不值得盲目改型 |

新增生产依赖只有 infer-state 对工作区既有 ahash 的直接声明；生产锁文件不新增 package。
候选 bbtree/Verus 与 hashbrown 仅存在于独立 benchmark 锁文件，不进入发布依赖图。
`PrefixCache::values()` 明确为任意顺序；仓库调用方仅用它检查所有权与集合，不依赖摘要排序。
clock 回绕按 `(touched,digest)` 显式排序，继续保持确定性。

## 验证范围

新增随机 churn 与标准 LRU oracle 对照，覆盖替换、条目压力、miss、values 集合与排空；
另外直接测试时钟回绕后的全部淘汰顺序。既有 byte budget、namespace、最长前缀、COW/pinned 页所有权测试继续运行。
45 上通过：

- `cargo test --locked -p infer-core -p infer-state -p infer-scheduler -p infer-runtime`：170 passed，0 failed；
  2 个原有 NUMA 权限专项保持 ignored，未执行硬件权限验收。
- 上述四个 crate 的 all-targets Clippy、根目录 fmt；独立容器工具 fmt / all-targets Clippy。
- Rust 1.90 对上述四个 crate all-targets 检查，以及独立容器工具检查。
- `cargo run --locked --release -p infer-cpu-bench`：49 组计数记录的 alloc/realloc/dealloc 全部为 0。
- layout / policy / workspace 边界与其 7 个回归测试、测试收集检查，改动的 Python 通过 Ruff。

CPU 零分配门禁只按其原有范围解释，prefix 发布不在该门禁内。
本次未执行完整 `make check`、GPU 服务验收或 Metal 实机测试。
