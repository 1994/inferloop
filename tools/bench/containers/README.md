# CPU 容器对比

本工具只比较单线程容器操作，不代表模型吞吐、GPU kernel 或服务延迟。候选库固定为
bbtree 0.1.5、btree-slab 0.6.1、hashbrown 0.17.1、ahash 0.8.12。
独立 workspace 把 bbtree 的 Verus 依赖与候选 hashbrown 留在生产依赖图之外。
分配计数直接复用现有 CPU benchmark 的 mimalloc 计数器。

```sh
cargo build --locked --release --manifest-path tools/bench/containers/Cargo.toml
# Linux：按本机拓扑选择一个性能核；45 机器使用 CPU 8。
taskset -c 8 tools/bench/containers/target/release/infer-container-bench > containers.json
INFER_CONTAINER_PREFIX_ONLY=1 taskset -c 8 \
    tools/bench/containers/target/release/infer-container-bench > prefix.json
cargo clippy --locked --manifest-path tools/bench/containers/Cargo.toml --all-targets -- -D warnings
```

- 容量为 16、64、1024、8192；每个样本 100,000 次复合操作、7 轮，轮换容器执行顺序。
- 排序索引直接编译生产 `OrderedSet`，使用 `(u64, u64, RequestId, usize)` 公平队列键。
  每次删除、插入、取最小值与 lower bound；不宣称覆盖所有调度索引或键分布。
  bbtree 也先插入两倍容量再 clear；初始填充在计时外。
- 精确查找用分散的 32 字节摘要键，每次命中、未命中、删除、插入。
  比较 std BTreeMap、bbtree、std HashMap + ahash、hashbrown + 同一 ahash、生产 BoundedMap。
  键生成、初始填充与 live-key 缓冲均在计时外。
- PrefixCache 直接编译生产代码，覆盖单个 2-token block 与 128 tokens / 16-token block。
  每次实际 lookup + LRU touch 或 contains，保留所测版本的指纹计算和 token 比较成本。
- 相同场景比较 checksum，排序索引额外和标准库核对完整遍历与缺失键范围。
  分配、重分配、释放均包含在计时中；退出前的容器销毁不计入。
- 构建与计时应串行。比较改动前后时使用同一工具源码、锁文件、编译器、profile 与 CPU，
  保留两个二进制，至少 5 组交替运行，每组保留全部轮次与分配记录。
  基线 cache.rs 可从报告的 base commit 提取，在独立副本构建；不要覆盖带有未提交修改的文件。

输出 `ns_per_operation` 是上述复合操作的摊销时间，不是单次 get 延迟，也不是逐请求 P99。
默认编译选项不加 target-cpu=native；仅绑定逻辑 CPU，不关闭 SMT、睿频或桌面任务。
测量噪声与硬件范围必须随结果说明，不能把 README 的宣传倍数当作项目实测。

45 机器的结果、保留决策与证据见[容器评估](../../../docs/research/container-performance-2026-10-24.md)
与[前缀指纹评估](../../../docs/research/prefix-fingerprint-2026-10-24.md)。


## 共享前缀生命周期

```sh
cargo build --locked --release --manifest-path tools/bench/containers/Cargo.toml --example prefix_lifecycle
taskset -c 8 tools/bench/containers/target/release/examples/prefix_lifecycle > lifecycle.json
```

该 example 冻结旧 CUDA VecDeque 策略，并与实际共享管理器的 SnapshotScan、BlockHash 比较。
覆盖 4096/32768 tokens、1/8/32 条目、命中加 history 复制、首 token 未命中、maximum 截断，
以及预分配 token 输入后的去重替换/发布淘汰。查询每轮 4096 次，发布每轮 256 次；
64 次预热、7 轮轮换顺序。至少运行 5 个串行进程，使用进程中位数的中位数。
新策略查询断言不发生分配；输出包含每轮分配、释放和 checksum。
两种实现均使用 u64 payload，不能代表真实 CUDA Tensor 布局、设备复制或完整服务路径。
退役 token buffer 释放计入发布时间，输入分配与设备快照捕获不计入。

[评估与全部退化场景](../../../docs/research/shared-prefix-lifecycle-2026-10-24.md)
包含默认页缓存的独立基线对比与缺失证据。
