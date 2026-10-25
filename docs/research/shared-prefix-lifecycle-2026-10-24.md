# 共享前缀生命周期：后端保留物理恢复策略

工作分支先合入 main `62417eb`，候选实现 `3067284`。CUDA 原先在 executor 内维护另一套
匹配、take/reinsert、预算、LRU 和计数；Metal 经 KvCacheManager 使用共享 PrefixCache。
本改动把发布、匹配边界、兼容性过滤、预算、淘汰、成功命中计数收回 engine/state。
后端通过 PrefixReuse 定制页引用获取或 CUDA target/draft 恢复；通过 PrefixPolicy 选择索引。
这在 Rust 中用组合与 trait 表达共享流程和扩展点，不需要模拟类继承。

## 所有权与失败

PrefixSnapshot 是发布载体，payload 无需 Clone 或假装持有共享页。复用借用缓存中的源，
后端成功后才更新命中、复用 token 数与 LRU。失败不移除源或刷新顺序。
Metal 页适配器成功 retain 后返回拥有者；CUDA 保持原有设备恢复调用，失败时 poison
目标并 drain 已排队工作，源仍由缓存持有。新增 drain 仅在错误恢复路径。
最长候选必须先满足目标容量与 draft/readout 兼容性，允许退回较短兼容快照。

发布的拒绝、替换、淘汰 payload 返回给后端清理；单项返回值内联，批量淘汰才为溢出部分分配。
共享页适配器只释放缓存自己的引用。覆盖 failed retain、非 Clone payload、失败计数/LRU、
批量 retirement 清理、两种策略 3000 次随机操作一致性、时钟回绕与 ID 耗尽。

## 为什么保留两种索引

CUDA 快照少而大，提前未命中应只比较很少的 token。强制共用 BlockHash 会给 32K 首 token
未命中引入整段 hash 成本。因此 SnapshotScan 使用实例内不复用 ID、完整 token 去重、
小 VecDeque 保存 ID 的 recency；BlockHash 保留双 aHash 增量指纹及树形 recency。
这两者共用生命周期，只有索引表示不同。完整 token 比较始终是身份确认条件。
缓存与不可变加载的 backend 实例一同构造，不跨实例传递快照。

最短驻留长度的保守下界让过短 maximum 提前返回；淘汰后允许下界过小，但不能过大。
每次 hit 直接借用候选，避免第二次 map 查找以及 CUDA take 后再去重发布。

## 45 上的 CPU 实测

[完整证据](artifacts/shared-prefix-lifecycle-2026-10-24.json)记录源码与二进制身份、每轮时间、
checksum 和分配计数。i9-13900KF、Rust 1.99.0、release/thin LTO/codegen-units=1，
绑定 CPU 8；powersave、SMT、睿频和桌面未调整。5 个串行进程，每进程 7 轮轮换实现顺序。
本任务的构建和测试在计时外；同机其他工作分支曾有编译任务，后台负载未隔离，绝对值存在噪声。
下表为进程中位数的中位数，单位 ns，百分比为耗时变化，正数是退化。

CUDA 对比是旧 VecDeque 策略的冻结逻辑与实际共享管理器，payload 为 u64 占位，
**不含真实 CUDA Tensor 结构布局、设备分配/复制、模型执行或服务延迟**。
查询每轮 4096 次，预热 64 次，命中包含 host history 复制；发布每轮 256 次，
token 输入提前分配，计入退役 token buffer 的释放，不计设备捕获与 payload 析构。

| 场景 | tokens | 条目 | 原 CUDA 策略 | shared-scan | 耗时变化 |
|---|---:|---:|---:|---:|---:|
| hit | 4096 | 1 | 285.92 | 179.29 | -37.3% |
| hit | 4096 | 8 | 380.32 | 260.98 | -31.4% |
| hit | 4096 | 32 | 463.39 | 319.03 | -31.2% |
| hit | 32768 | 1 | 4314.31 | 2994.80 | -30.6% |
| hit | 32768 | 8 | 5621.42 | 3610.34 | -35.8% |
| hit | 32768 | 32 | 7135.54 | 5810.27 | -18.6% |
| early-miss | 4096 | 1 | 4.37 | 4.00 | -8.5% |
| early-miss | 4096 | 8 | 16.00 | 18.36 | +14.7% |
| early-miss | 4096 | 32 | 55.26 | 66.91 | +21.1% |
| early-miss | 32768 | 1 | 4.37 | 4.00 | -8.4% |
| early-miss | 32768 | 8 | 16.01 | 18.37 | +14.8% |
| early-miss | 32768 | 32 | 55.27 | 66.90 | +21.0% |
| capped | 4096 | 1 | 3.82 | 1.82 | -52.4% |
| capped | 4096 | 8 | 4.60 | 1.82 | -60.3% |
| capped | 4096 | 32 | 9.05 | 1.82 | -79.9% |
| capped | 32768 | 1 | 3.82 | 1.82 | -52.3% |
| capped | 32768 | 8 | 4.59 | 1.83 | -60.2% |
| capped | 32768 | 32 | 9.06 | 1.83 | -79.8% |
| publish-replace | 4096 | 1 | 277.43 | 284.90 | +2.7% |
| publish-replace | 4096 | 8 | 307.78 | 324.58 | +5.5% |
| publish-replace | 4096 | 32 | 351.02 | 389.19 | +10.9% |
| publish-replace | 32768 | 1 | 3391.84 | 3373.06 | -0.6% |
| publish-replace | 32768 | 8 | 4102.26 | 4205.74 | +2.5% |
| publish-replace | 32768 | 32 | 5310.34 | 5335.00 | +0.5% |
| publish-evict | 4096 | 1 | 13.82 | 31.15 | +125.3% |
| publish-evict | 4096 | 8 | 24.36 | 40.61 | +66.7% |
| publish-evict | 4096 | 32 | 66.23 | 105.68 | +59.6% |
| publish-evict | 32768 | 1 | 38.00 | 53.18 | +40.0% |
| publish-evict | 32768 | 8 | 51.06 | 60.91 | +19.3% |
| publish-evict | 32768 | 32 | 86.88 | 133.70 | +53.9% |

命中快 18.6%–37.3%；提前未命中在 8/32 条目时分别多约 2.4/11.6 ns，
冷发布淘汰多约 10–47 ns。4K 替换多 7–38 ns，32K 替换变化为 -0.6% 至 +2.5%。
这是保留共享 map、预算、成功提交和清理所有权所付的索引元数据成本；
明确保留这项权衡，不能把命中收益概括为所有操作提速。
所有 shared-scan 查询均无分配/重分配/释放；发布替换无新增分配，淘汰通常无新增分配，
32 条目 churn 的部分轮次有一次 8336 字节 map 重建分配。每轮固定释放 256 个退役 token buffer，
有 map 重建时再释放旧表。多项 retirement 与首建 map/queue 分配不由该热查询结论覆盖。

下面用 PR51 冻结二进制对比默认共享页 BlockHash 路径，缓存源码与合入 main 的版本一致，
已核对前一报告中的二进制与源文件 SHA。5 组交替运行，7 轮，每轮 100000 次。

| 场景 | 条目 | 原共享缓存 | 本次共享缓存 | 耗时变化 |
|---|---:|---:|---:|---:|
| prefix_contains_b16_t128 | 16 | 54.24 | 54.39 | +0.3% |
| prefix_contains_b16_t128 | 64 | 58.44 | 58.17 | -0.5% |
| prefix_contains_b16_t128 | 1024 | 59.44 | 59.22 | -0.4% |
| prefix_contains_b16_t128 | 8192 | 77.26 | 74.31 | -3.8% |
| prefix_contains_b2_t2 | 16 | 9.12 | 9.12 | -0.0% |
| prefix_contains_b2_t2 | 64 | 9.51 | 9.42 | -0.9% |
| prefix_contains_b2_t2 | 1024 | 12.07 | 12.18 | +0.9% |
| prefix_contains_b2_t2 | 8192 | 13.00 | 13.02 | +0.1% |
| prefix_lookup_touch_b16_t128 | 16 | 99.43 | 101.24 | +1.8% |
| prefix_lookup_touch_b16_t128 | 64 | 107.28 | 109.02 | +1.6% |
| prefix_lookup_touch_b16_t128 | 1024 | 131.49 | 133.97 | +1.9% |
| prefix_lookup_touch_b16_t128 | 8192 | 165.47 | 169.18 | +2.2% |
| prefix_lookup_touch_b2_t2 | 16 | 32.88 | 34.55 | +5.1% |
| prefix_lookup_touch_b2_t2 | 64 | 36.45 | 36.93 | +1.3% |
| prefix_lookup_touch_b2_t2 | 1024 | 60.07 | 61.61 | +2.6% |
| prefix_lookup_touch_b2_t2 | 8192 | 76.88 | 78.21 | +1.7% |

默认页缓存 contains 约 -3.8% 至 +0.9%，lookup+LRU 多约 0.5–3.7 ns（+1.3%–5.1%）。
这是增加统一成功提交和计数的可量化代价。两侧各场景 checksum 与分配计数一致；
树形 LRU 在中大缓存中的既有分配仍存在，本改动不宣称其查询零分配。

## 设备与容量边界

生产的 4096-token 快照间隔、字节预算、target/draft KV 复制与成功路径同步点保持原调用。
源码检查未引入成功路径 readback、额外 GPU 拷贝或图捕获。共享 CPU 元数据多一个 hash 表
及 ID recency；真实 CUDA payload 的主机布局/峰值、模型 TTFT/TPOT、并发与冷启动未测，
不能从 u64 微基准推断这些指标。实例初始化多两个随机 hash 状态；成本未单独测量。
45 无 Metal 设备，Metal 实机验收缺失。数值 kernel 本身未改动。

复现命令见[容器基准说明](../../tools/bench/containers/README.md)。最终检查结果记录在 PR；
host compile/test 与 CUDA 设备执行分别报告，不将 ignored 用例算作设备验收。
