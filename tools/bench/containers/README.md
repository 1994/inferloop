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
  每次实际 lookup + LRU touch 或 contains，保留 SHA 链和 token 比较成本。
- 相同场景比较 checksum，排序索引额外和标准库核对完整遍历与缺失键范围。
  分配、重分配、释放均包含在计时中；退出前的容器销毁不计入。
- 构建与计时应串行。比较改动前后时使用同一工具源码、锁文件、编译器、profile 与 CPU，
  保留两个二进制，至少 5 组交替运行，每组保留全部轮次与分配记录。
  基线 cache.rs 可从报告的 base commit 提取，在独立副本构建；不要覆盖带有未提交修改的文件。

输出 `ns_per_operation` 是上述复合操作的摊销时间，不是单次 get 延迟，也不是逐请求 P99。
默认编译选项不加 target-cpu=native；仅绑定逻辑 CPU，不关闭 SMT、睿频或桌面任务。
测量噪声与硬件范围必须随结果说明，不能把 README 的宣传倍数当作项目实测。

45 机器的结果、保留决策与证据见[容器评估](../../../docs/research/container-performance-2026-10-24.md)。
