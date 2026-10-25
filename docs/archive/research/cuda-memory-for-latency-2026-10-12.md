# 主机内存与共享工作区的推理加速方向

本记录先区分容量、带宽与数据复用。增加 RAM/显存占用本身不构成性能收益；
验收指标仍是服务 TTFT、TPOT、并发 wall/吞吐与原有数值/完整模型质量契约。
验证设备是 45；生产策略不得按 GPU 型号写死。

## 已观测内存

45 主机 `MemTotal=65,661,976 KiB`（约 62.6 GiB），一次模型间空闲观察中
`MemAvailable=55,550,880 KiB`（约 53.0 GiB），`Cached=49,069,788 KiB`（约 46.8 GiB）。
Linux 已在利用可回收文件缓存，不能把 MemAvailable 当作未使用的所有物理页。
原有 swap 占用属于全机状态；本次受保护测试的 swap 上限为零，未干预其他进程。

`gdn-tuning-f32-half-perf-v1` 第 0 轮硬件监测：native WY 进程 RSS 采样峰值约 1.135 GiB，
vLLM 进程树 RSS 求和峰值约 4.944 GiB。RSS 求和可能重复计入共享页；cgroup 口径应分别核对。
native 启动期 cgroup memory peak 约 4.846 GiB，vLLM 约 3.849 GiB，不能据此概括 Rust
在所有阶段内存都更低。模型权重使用 mmap/read_view，并加载到 GPU；额外复制一份 CPU
权重通常只增加加载阶段空间，不会减少每个 GPU decode step 的成本。

相同服务对比中 native 显存峰值约 30.08 GiB（30,806 MiB），总容量 32,607 MiB，显存容量已经接近上限。
nvidia-smi 的 `utilization.memory` 是内存活动率，既不是容量百分比，也不是测得的峰值带宽比例。
低活动率可能包含排队、计算、同步、低并行度或零填充成本，须用热路径成本验证归因。

## 可测方向 A：主机内存保存较冷的前缀

[当前前缀缓存](../../../crates/backend/cuda/src/executor/prefix.rs)全部保存 GPU Tensor。
[ProgramSnapshot](../../../crates/backend/cuda/src/executor/prefix_state.rs)保留 KV stride、完整 F32
递推状态、卷积等状态；恢复使用 D2D copy。GPU 准入压力通过逐出前缀释放显存，当前没有 RAM 次级缓存。

从实际模型配置与 serving runtime 的 FP8 KV readback 推导：
48 层 × 48 value head × 128 × 128 × 4 byte = 144 MiB 的 recurrent matrix，
16 层 attention × K/V × 4 KV head × 256 dim × 1 byte = 32 KiB/token。
因此 2048-token 边界仅这两部分就是 208 MiB；卷积、draft/hidden 等另计。
GDN 固定状态使短前缀的单位复用成本较高，应按省下的 prefill 成本/字节保留，不能只按 token 数排序。

设计应保持 GPU hot tier + 有界 RAM warm tier，普通页、固定页和 staging 路径由实测选择。
被 GPU 容量逐出的有价值前缀可搬入 RAM；命中时恢复真实覆盖位置的 KV、F32 递推与卷积状态，
同时恢复 draft/hidden。状态按原始字节搬运，不改 dtype，不将循环状态转为 BF16。
RAM 预算与 GPU budget 分开计价，RAM bytes 不可计入 GPU reclaimable bytes；
容量同时受用户上限、MemAvailable、cgroup 限制与其他活跃负载约束。
固定页缓冲大小须按主机/GPU/PCIe 身份测量，不能把验证机的赢家固化为全部设备策略。
异步 D2H/H2D 的完成事件、缓冲重用和 snapshot 生命周期必须有明确归属。

`host-prefix-spill-cost-v1` 已完成 256/511/1024/2048/8192-token 等效载荷的成本实验，
比较 D2D、pageable H2D、整份固定页 H2D、8/16/32 MiB 双缓冲，并测 D2H spill。
双缓冲固定页总量不超过已有 64 MiB readback batch 上限；整份固定页作为另一个成本参考，
不能视为当前 native 已支持的实现。传输实验逐字节验收，包含 CPU staging 与同步成本。
它不代替 native end-to-end 前缀命中测试，也不构成固定冷提示性能收益。

## 可测方向 B：共享更宽的 prefill 工作区

[BatchBuilder](../../../crates/backend/cuda/src/resident/batch.rs)每个 prompt graph 持有自己的
ActivationArena、projection workspace、recurrent scratch 和 readout。pooled 状态当前用窄图保留并发；
之前直接增加每序列宽图使第 4 个请求无法同时准入，并放大短提示填充工作，详见
[准入证据](cuda-pooled-admission-geometry-2026-10-12.md)和
[性能实验](cuda-performance-experiments.md)。

宽 chunk 可以增加 GEMM 并行度，并减少整套权重被重复遍历的次数。
更合适的空间用途是一份可复用宽工作区，再为不同 slot 的状态捕获或绑定各自图；
保留窄图处理短提示，确保共享 scratch 不与并行 replay 重叠使用。
graph 必须早于底层资源销毁，并在跨 slot 重用前等待上一使用者完成。
它首先是资源所有权改造，之后再对图宽度、slot 数和 arena 配额做硬件/工作量调优，
不能简单把 64 改成 256。

## 收益与验收边界

主机前缀层服务重复长前缀、会话重入和显存逐出场景；对每次都不同的冷提示，命中率为零。
不能用预热前缀混入当前冷提示 vLLM 差距对比。需要单独测试 cold miss、RAM hit、GPU hit、
缓存淘汰与多请求并发，并保留真实 cache readback。
迁移成本必须小于保存的计算；还要计入 miss 时 spill 成本，低复用率可能整体变慢。
共享宽工作区则有机会改善当前固定冷提示 TTFT，须保住四并发准入与短提示时延。
优先比较可节省的实际 GPU 工作，避免把所有权重展开到 RAM 后再逐 token 经 PCIe 读取。

## 主机传输成本实验结果

五个载荷、七次测量（各排除两次 warmup），所有搬入/搬出逐字节一致。state/cache 保留原始
F32/FP8 字节，不作 dtype 转换。下表计入 CPU staging 与同步，单位 ms；分配在计时外。

| token 边界 | 载荷 MiB | D2D ms | 普通 RAM→GPU ms | 整份固定页→GPU ms | 16 MiB 双缓冲→GPU ms | GPU→RAM（16 MiB 双缓冲）ms |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 152.000 | 0.223 | 8.529 | 3.321 | 11.253 | 12.069 |
| 511 | 159.969 | 0.226 | 8.949 | 3.498 | 11.876 | 12.579 |
| 1024 | 176.000 | 0.249 | 9.706 | 3.842 | 13.074 | 14.109 |
| 2048 | 208.000 | 0.293 | 11.774 | 4.549 | 15.430 | 16.488 |
| 8192 | 400.000 | 0.577 | 21.637 | 8.752 | 29.873 | 31.710 |

本次小固定页双缓冲原型没有收益：8/16/32 MiB 的三个配置恢复都慢于普通 pageable H2D。
208 MiB 载荷三者为 15.517/15.430/15.427 ms，对照普通 RAM 11.774 ms；
400 MiB 为 29.734/29.873/29.109 ms，对照 21.637 ms。不能预先把双缓冲当作生产赢家。
CPU copy、框架调用、event 等可能影响结果，当前没有分项归因证据；须在 native Rust 搬运路径
中验证后再选择。整份固定页恢复最快，但每条缓存锁住 152–400 MiB，不能把 53 GiB
MemAvailable 全转为固定页，也不能套用已有单批 64 MiB readback API 上限。

微基准使用冻结官方 Torch 2.13 CUDA runtime，验证机为 5090；参数只作为该设备证据。
Torch allocator 可能保留已释放固定页，ring 的申请总量最多 64 MiB 不代表全进程固定页峰值。
受保护测试 cgroup peak 约 2 GiB、swap 为零。检查并保留其他 GPU 进程；本实验不使用服务对照
的外部图形显存严格一致性资格，不宣称是合格 end-to-end 性能对比。
载荷只包含 recurrent matrix 和 KV，不包含卷积/draft/hidden、Tensor stride 和 native 调度开销。

证据目录 `host-prefix-spill-cost-v1`，`results.json` SHA256 为 `0a8a2af16b2bfb5e2c3dc5235986545d6cadf4401e44257296547bfbf8b96454`；
`plan.json`、`preparation.json`、原始 `run.log` 和逐次样本全部保留。
当前只完成搬运可行性筛选，native RAM prefix tier 尚未实现，也未获得完整服务收益。

从第一性原理，假设被缓存前缀的重算成本为 C，恢复为 R，额外 spill 为 W，命中率为 p，
忽略其余 bookkeeping 的必要收益条件是 `p(C−R) > (1−p)W`。
实际策略还应计入快照构建/分配、驱逐、页锁定与并发干扰。GPU hot hit 比 RAM hit 更便宜，
RAM 应承接有价值的 GPU 驱逐项，不应主动把热项全部搬离 GPU。
冷提示基准 p=0，新增主机缓存不能解决它的 TTFT 差距。

## 可测方向 C：GDN 准备结果换空间复用

当前 value 列分区重复计算该 head 的 QK interaction 与三角 inverse。
可以先为每个 32-token 子块写入共同准备结果，apply kernel 保持 F32 state，
在一个 CTA 内跨子块递推；不同 value 列分区再读取同一准备结果。
对于 `VH=48, LANES=256, chunk=32`，单个 `48×8×32×32` F32 表仅 1.5 MiB，
若需多个表则按实际个数计价；按执行顺序跨层复用 scratch 才能避免乘上 48 层的容量。
K/Q 点积可在共享 key-head 范围复用，但 beta/gate 随 value head 变化，inverse 不可直接跨它们共享。
该方向需要比较额外 global 读写/launch 与省下的计算，没有已证明的服务收益，
也不改变四分之一列宽尚未通过模型质量资格的事实。

用户新增的 [ReplaySSM](https://dao-lab.ai/blog/2026/replayssm/) 方向作用于 decode 和推测验证：
把近期校正向量保留在显存小窗口，减少整份 recurrent state 写回，并可能减少检查点容量。
它与主机前缀层可以结合；当前独立探索见 [ReplaySSM 研究记录](cuda-replayssm-gdn-2026-10-12.md)，不混入本搬运成本结果。
