# 多卡并行：DP、TP 与跨设备边界

状态：方案设计完成；P0.1（设备序号选择）随本 PR 落地，P0.2 起待实现，P1 的实机验收需要两卡。事实基点为 main `e34d7b2a`（2026-11-19），引用的行号以该提交为准。

本文回答一个问题：**不推翻现有 runtime / scheduler / state 分层，怎样支持多卡部署。** 结论是把“多卡”拆成三条成本完全不同的路线：

| 路线 | 解决的问题 | 改动面 | 顺序 |
|---|---|---|---|
| DP：每卡一个完整副本 | 吞吐 | 部署、路由、实例身份；Engine 不改 | 先做 |
| TP：一个模型切到 N 卡 | 容量（单卡装不下）或单请求延迟 | 加载、IR/图、kernel、后端、KV、checkpoint | 第二步 |
| PP / PD 分离 | 阶段异构、超长上下文、跨机 | 跨卡 KV 传输、调度阶段放置、故障域 | TP 之后 |

判断规则很直接：**模型单卡装得下时，吞吐的第一答案是 DP 而不是 TP**——多副本没有逐 token 通信，路由按前缀指纹粘会话；TP 只有在容量或单请求延迟构成瓶颈时才值得付出集合通信的代价。这与 [CUDA 性能实验](../../archive/research/cuda-performance-experiments.md) 第八节的结论一致。

本文负责方案与背景；P0.1（设备序号）已随本 PR 落地，P0.2 起的实现由后续 PR 承接，交接入口见第 11 节。本文的 P0/P1/P2 是本方案内部的实施顺序；与[路线图](../README.md)的全局优先级分别维护，不改变现有任务编号。

## 1. 背景

单卡路径已经具备多卡方案必须依赖的契约：常驻图、paged KV、池化准入、前缀缓存、MTP/DFlash2、checkpoint 与故障隔离。问题不在“缺一个并行框架”，而在单卡假设分散在入口、加载、执行和观测四层：

- 入口曾把设备固定为 0 号（P0.1 已改为 `--device`），进程只能拥有一个 backend；
- 加载按整份权重报价并上传，没有分片视图；
- 一个 backend 只有一个 compute flight、一个 ticket、一套物理状态；
- 准入、成本、超时按单设备报价；
- checkpoint 身份是单个 backend 字符串；
- 服务层一个进程一个 Engine、一个监听端口，Responses 状态是进程内的。

因此本方案不新增“并行 runtime”，而是把 TP 实现成**一个逻辑后端**（rank group），让 `Engine<B: BackendProvider>` 原样工作；DP 则保持“一进程 = 一设备 = 一 Engine”，把复杂度放在网关和实例身份上。

**非目标：** EP、远程 PD、分层 offload、跨 backend checkpoint、动态 C ABI/WASM 加载。这些都不在本方案的 P0–P2 内。

## 2. 现状中可以直接复用的契约

| 契约 | 位置 | 为什么多卡可以直接用 |
|---|---|---|
| Engine 泛型化于 `BackendProvider` | `crates/engine/runtime/src/engine/construction.rs:41` | TP 后端对 Engine 透明，调度、请求生命周期、输出交付都不需要 rank 概念 |
| 逻辑状态与物理状态分离 | `docs/architecture/kv-manager.md` | 逻辑页仍是全局身份，TP 只是让每个 rank 各持有一份物理分片 |
| 能力与要求模型 | `docs/architecture/backends.md` | rank group 上报交集能力即可复用现有校验 |
| ticket 绑定 owner 与 generation | `crates/backend/cuda/src/executor/submission.rs:17` | group ticket 沿用同一身份规则，拒绝跨 owner / 重复消费 |
| 两阶段资源准备 | `crates/foundation/spi/src/backend.rs:138` | `prepare_state_reservation` 已经是“先探测、后提交”的钩子，TP 在此基础上做跨 rank 原子事务 |
| 后端身份 + checkpoint 校验 | `crates/engine/runtime/src/persistence/snapshot/validation.rs:46` | 升级为 group 身份后，分片错配会在恢复前被拒绝 |
| 单 compute flight、单 device-owner 线程 | `docs/architecture/backends.md`、`construction.rs:213` | TP 的并行度来自 N 个 rank 各自一个 owner，而不是打破单 owner 边界 |

**要保留的不变量：** 调度器不出现 rank；backend 不感知调度策略；引擎不重建 `ModelIr`；所有设备 buffer 留在 backend；一次提交只有一个有所有权的 flight。

## 3. 单卡假设清单

P0.1 已把入口的硬编码设备换成显式序号；下表其余项仍是当前实现的假设。

| 假设 | 位置 | 对多卡的影响 | 阶段 |
|---|---|---|---|
| 设备序号隐式 | `crates/service/cli/src/backend/mod.rs:129`（能力探测只问 0 号）、`tuning/preparation.rs:32`（调优身份只问 0 号） | DP 只能靠 `CUDA_VISIBLE_DEVICES`；进程内多设备无法表达 | P0.1 已加 `--device`；P0.3 收口调优身份 |
| 加载按整份权重报价 | `crates/model/package/src/storage/loader.rs:75` | TP 无法只读自己那一片，显存收益消失 | P1 |
| 一个 `ExecutionProgram`、一个 ticket | `construction.rs:69`、`spi/backend.rs:335` | submit/poll 需要扇出与组完成语义 | P1 |
| 一套 `StateRecipe` | `crates/foundation/ir/src/state_recipe/mod.rs:56` | head 数、KV 布局要按 rank 参数化 | P1 |
| 物理页与前缀缓存属于单个 backend | `crates/engine/state/src/cache.rs:27` | 跨 rank 的 attach 必须 all-or-nothing | P1 |
| 准入/成本/超时按单设备 | `construction.rs:64`、`crates/engine/runtime/src/config.rs` | TP 的并发、成本、超时要按最慢 rank 归约 | P1 |
| checkpoint 身份是一个字符串 | `executor/mod.rs:319`、`validation.rs:46` | 需要 tp 度数、各 rank 身份、分片计划 | P1 |
| JIT identity store 是进程级单 scope | `crates/backend/cuda-kernels/src/tile/identities.rs:50` | 进程内多设备会共用缓存文件；按 ordinal 取身份即可 | P0.3 |
| 一个进程一个 Engine、一个监听 | `crates/service/cli/src/commands/server.rs:107` | DP 需要网关与实例身份 | P0.2 |
| Responses 状态是进程内 | `crates/service/frontdoor/src/responses_store.rs:4` | DP 下 `previous_response_id` 需要粘性路由 | DP 部署 |

## 4. DP：多副本

### 4.1 部署形态

每张卡一个进程、一个 Engine、一个监听端口；进程之间互不共享状态。`CUDA_VISIBLE_DEVICES` 决定进程可见集合，`--device` 选择集合内的序号，并把选择写进启动日志与调优身份（`/health` 见 P0.2）：

```text
CUDA_VISIBLE_DEVICES=0 infer model --device 0 --listen 127.0.0.1:8001
CUDA_VISIBLE_DEVICES=1 infer model --device 0 --listen 127.0.0.1:8002
```

两者互补：`--device` 能表达“可见集合里的第 N 张卡”，并在异构机上避免日志与调优身份含糊。**不建议**在没有网关的情况下做进程内多副本：CUDA 的 `fresh()` 尚未安装（`crates/service/cli/src/backend/selected.rs:64`），JIT identity store 也还是进程级单 scope。

### 4.2 路由

网关按请求的 prompt 前缀指纹做一致性哈希，把同一会话送到同一副本；副本过载时按显式负载信号降级到轮询，而不是静默换桶：

- 前缀键取系统提示与首个 user 轮次的稳定前缀；不需要完整 tokenizer，也不需要读取 KV。
- 负载信号用 `/health` 的 `ready` 与有界队列深度，避免把请求送进正在加载或已隔离的实例。
- `previous_response_id`、流式重连与取消必须保持同一副本；不满足时返回明确错误，不能悄悄切到另一个副本丢上下文。
- 不做跨副本前缀缓存。路由只能“买回”命中率，不能承诺副本间共享。

### 4.3 实例身份与观测

- P0.2 起启动日志、`/health`、`/native/v1/runtime` 带 device 序号与硬件身份（name/arch/显存），网关据此判断“同一张卡上的两个实例”与“两台机器”。
- 每副本独立上报 TTFT/TPOT、队列深度、KV 使用与故障状态；聚合指标不能替代 per-replica 视图。
- 计费、租户配额、`max_num_seqs` 保持 per-instance 语义，不引入跨副本协调。

## 5. TP：rank group 后端

### 5.1 逻辑后端

`RankGroup<B>` 实现 `BackendProvider`，每个 rank 是一个已经线程化的 backend：

```text
Engine<RankGroup<ThreadedBackend<CudaBackend>>>
  ├─ rank 0: ThreadedBackend<CudaBackend>（自己的 owner 线程与 stream）
  └─ rank 1: ThreadedBackend<CudaBackend>
```

**组合顺序很重要。** 现有 `into_threaded` 把整个 backend 包成一个 owner 线程（`construction.rs:213`）。如果先分组再线程化，N 张卡会串行编码，TP 的收益被 launch 延迟吃掉。正确路径是每个 rank 先 `ThreadedBackend`，再在它们之上建 group；`into_threaded` 需要增加 group-aware 入口，或由 CLI 先构造 rank 再交给 Engine。

group 对 Engine 暴露与单卡相同的语义：

- `capabilities()`：交集（相同 backend kind、公共 dtype、最小显存）；异构卡显式 Unsupported，不按最小卡静默降级。
- `execution_profile()`：并发取最小、延迟取最大；否则准入会超卖。
- `state_reservation_bytes` / `free_state_bytes` / `state_capacity_bytes`：分别取最大 / 最小 / 最小。
- `submit` / `submit_borrowed`：向所有 rank 广播；全部编码成功才允许任何一张卡开始执行；任一 rank 失败则回滚并向 Engine 报错，且不留下未确认的设备工作。
- `poll`：只有所有 rank 的 fence 都确认才返回完成；一张卡完成、另一张仍在飞行时不得发布结果。
- `completion_timing` / `device_timing` / `kv_cache`：分别取最大、合并、合并。
- `identity()`：包含 tp 度数、各 rank 设备身份与分片计划，供 checkpoint 与诊断使用。

### 5.2 图与集合通信

集合通信必须进入 `DataflowGraph`，而不是散落在 backend 的录制逻辑里：

```text
ModelIr → DataflowGraph（含 Collective 节点）→ compiler 选 kernel → ExecutionProgram（rank 无关）
                                                               ├─ rank 0 绑定自己的权重与 KV
                                                               └─ rank 1 绑定自己的权重与 KV
```

- 在 IR 增加 AllReduce / AllGather / ReduceScatter / Send / Recv 的算子描述；单卡注册 no-op kernel，因此**单卡与 TP 共用同一份图构造代码**。
- collective 节点的 kernel 选择仍由能力与 shape 决定；TP 后端把它降成 NCCL 或自研 P2P，单卡保持直通。
- `ExecutionProgram` 不携带 rank；各 rank 只在权重绑定、KV 布局与 kernel 参数上不同。
- 图捕获顺序、宽窄档选择、tuning 决策必须是 `(设备身份, 模型, 配置, tp 度数)` 的纯函数；否则各 rank 捕获到不同的图，集合通信会在错误的节点上等待。

### 5.3 权重与状态分片

- **加载期视图**：在 `WeightLoadPlan` 的 binding 之上加 shard view（切轴、偏移、长度），只从 safetensors/GGUF 读取本 rank 的切片，`resident_bytes` 与预检按分片计算。
- **块对齐**：FP8/NVFP4/GGUF 的量化块必须按块边界切；除不尽直接拒绝，不能加载后再补齐。
- **指纹**：`payload_fingerprint` 升级为“分片计划 + 各 rank payload”的组指纹；checkpoint 恢复时缺一个 rank 或分片错配都必须失败。
- **head 分片**：`ModelIr` 的 attention 已带 `kv_heads`（`crates/foundation/ir/src/model.rs:18`）；加载期校验 `query_heads % tp == 0`、`kv_heads % tp == 0` 或显式 KV 复制策略，并把 `heads_per_rank` 写进 `StateRecipe`。
- **词表**：embedding / lm_head 按 vocab 切或复制，logits 的 all-gather 与采样 rank 必须固定；同一 prompt 在不同 rank 上采样结果不同，属于协议错误，不是数值误差。
- **KV**：逻辑块 id 仍是全局身份；每个 rank 持有自己的物理块。`reusable_prefix` 取各 rank 的最小值，`reuse_prefix` 与 `reserve_state` 都必须 all-or-nothing：先在所有 rank 上 prepare，再统一 commit，部分失败时 release 并隔离。

### 5.4 决策、成本与超时

TP 下每个动态决策都是一次全局同步，因此：

- 所有动态决策只读一份归约后的快照；执行器不得读 rank 本地状态。
- 成本模型按 group 重新校准；batch 打包与 SLO 预测使用 group 成本，而不是单卡成本。
- `submission_timeout_us` / `resource_timeout_us` 覆盖最慢 rank 加通信；straggler 是 TP 的常态，不是异常。
- 准入按 group 最小容量计算；一张卡装不下的请求在准入阶段就拒绝，不能等到 reserve 才发现。

### 5.5 故障、排空与观测

- **单 rank 故障 = 全组故障。** 任一 rank 无法确认 fence，group 进入隔离；继续 poll 所有 rank，直到资源真正排空。不能只释放坏 rank 而让其余 rank 继续服务。
- 看门狗要区分“慢”和“挂”：NCCL 超时必须映射为 group fault，而不是无限等待；shutdown 的 5 秒收敛语义对 group 同样适用。
- 观测增加有界 `rank` 维度与 per-rank 队列/显存；collective 时间作为独立 segment，不能混入 kernel 时间；group 级 trace 用同一父 span。
- checkpoint 逐 rank 捕获执行状态，恢复时校验 schema、tp 度数、各 rank 设备身份、分片计划与 payload；不完整的分片拒绝恢复。

## 6. 纵向不变量

1. **逻辑后端**：任何代码不得假设“一进程 = 一设备”；TP 时同一个 `BackendProvider` 由 rank group 实现。
2. **集合通信只走图**：单卡 no-op、TP 真实通信，图构造代码只有一份。
3. **决策唯一**：所有动态决策读归约后的快照，执行器禁止读 rank 本地状态。
4. **分片是加载期视图**：模型 provider 对 TP 无感；分片计划由 backend 按 tp 度数声明。
5. **状态按 rank 参数化**：heads、KV 布局、recurrent 状态的切分规则进 IR/recipe，不从 config 直取。
6. **确定性**：tuning、捕获顺序、种子、并列规则在 rank 间一致；否则死锁或数值发散。
7. **原子资源事务**：prepare → commit → release，任何 rank 上都不能半提交。
8. **显式拒绝**：异构卡、head 除不尽、量化块不整除、缺 rank 的 checkpoint，一律 Unsupported。

## 7. 实施计划

### P0：单卡可完成

| 编号 | 改造 | 验收 | 状态 |
|---|---|---|---|
| P0.1 | 设备序号：`--device`、`Selection.device`、加载与日志、Metal 显式拒绝 | 默认 0；非法序号明确报错；启动日志可读出序号；单卡默认路径无行为变化 | **本 PR** |
| P0.2 | 实例身份：`/health` 与 inspection 带 device 与硬件身份 | 两台不同设备/副本可通过 `/health` 区分；单卡输出兼容 | 待做 |
| P0.3 | 设备键收口：JIT identity store 与 tuning stamp 按 ordinal 取设备身份 | 异构机上 `--device N` 的调优缓存不与 0 号卡混用 | 待做 |
| P0.4 | 组协议主机测试：fake 2-rank backend 验证 group ticket、两阶段 reserve、min-prefix 与故障传播 | 不依赖 GPU；`cargo test` 覆盖回滚与 fence | 待做 |

P0.4 的 fake backend 只放在测试里，不新增产品 feature；它验证的是协议，不是设备能力。

### P1：TP=2

| 编号 | 改造 | 依赖 | 验收 |
|---|---|---|---|
| P1.1 | `RankGroup` + group ticket + 故障隔离 | P0.4 | 主机语义测试通过；两卡提交/轮询不再串行 |
| P1.2 | 图内 collective（单卡 no-op） | P1.1 | 单卡图与输出逐字节不变；两卡 all-reduce 正确 |
| P1.3 | 分片加载 + `StateRecipe` 按 rank 参数化 | P1.2 | 单卡装不下的模型可加载；显存按 rank 下降 |
| P1.4 | 两卡验收：逐 token 一致性、容量、TTFT/TPOT、单 rank kill | NVLink 实机 | 独立参考 + 显式容差；逐 case 性能；隔离不假 healthy |

P1 的硬件前提是 NVLink 机型；5090 这类无 NVLink 的机器不用于 TP 的开发或验收。

### P2：TP=4/8、PP、PD

- TP=4/8 先复用 P1 的同一套 group 协议，不新增决策路径；
- PP 需要调度阶段放置与多 flight，先证明单 stage 的收益；
- PD 需要跨卡 KV 传输、远程状态所有权与传输计费，必须单独设计，不能借用 TP 的 group ticket。

## 8. 性能与验收

| 场景 | 必测 | 不接受 |
|---|---|---|
| DP | 副本线性吞吐、路由开销、前缀命中率变化、单副本故障时的 p99 | 用聚合吞吐掩盖某个副本的尾延迟 |
| TP | batch 1 的 TTFT/TPOT、高并发吞吐、长上下文、all-reduce 占比、峰值显存 | 用总吞吐掩盖 batch 1 变慢；只报平均值 |
| 故障 | kill 一个 rank、NCCL 超时、慢 rank | 把隔离报告为 idle 或 healthy |
| 正确性 | 单卡参考的完整 token 序列、拒绝后状态、KV/recurrent 回滚 | 只比首 token；用更宽容差替代参考 |

性能证据按[性能基线](../performance/baseline.md)的 release、配对、逐 case 规则执行；CUDA 设备行为还需要 `make test-cuda` 的目标硬件窗口。单卡改动（P0.1–P0.3）不改变默认执行路径，PR 需说明“无执行影响”的依据，而不是省略性能一节。

## 9. 风险

| 风险 | 表现 | 约束 |
|---|---|---|
| rank 本地决策发散 | 集合通信等待错误的节点，表现为 hang | 决策唯一 + 确定性捕获顺序 |
| 部分提交 | 一张卡持有 lease、另一张回滚，后续恢复泄漏 | prepare/commit/release 两阶段 |
| straggler | TP 尾延迟被最慢卡支配 | 超时覆盖通信；性能逐 case 报告 |
| 异构卡 | 静默按最小卡运行，数值/容量都与预期不符 | 能力交集 + 显式 Unsupported |
| 分片 checkpoint | 缺 rank 或错分片被当成完整状态恢复 | 组身份校验；拒绝部分恢复 |
| DP 粘性缺失 | `previous_response_id` 落到别的副本，上下文丢失 | 网关粘性；失败显式报错 |

## 10. 明确不做

- 不在 scheduler / state 中加入 rank、tp、并行度概念。
- 不用环境变量做并行度开关；配置只走显式参数与配置字段。
- 不在没有两卡验收机型时合入 NCCL 路径；单卡 no-op 可以先合。
- 不为 DP 引入跨副本共享前缀缓存或全局调度器。
- 不把 CPU 测试后端的“并行”当作设备并行能力。

## 11. 交接

后续实现按 P0.2 → P0.3 → P0.4 → P1.1 → P1.2 的顺序推进；每一项可独立合并、独立回退。

**先读：**

- 本文第 2、3 节：可复用的契约与仍需处理的单卡假设。
- [Backend 能力与扩展](../../architecture/backends.md)：能力交集、ticket / owner、单 flight 与故障隔离。
- [KV Manager](../../architecture/kv-manager.md)：逻辑页 / 物理页、prefix 生命周期与 COW。
- [Runtime](../../architecture/runtime.md)：完成协议、隔离与排空。

**P0.2 实例身份**

- 入口：`Engine::inspect` 已经读取 `backend.capabilities()`（`crates/engine/runtime/src/engine/inspection.rs:88`）。把 `device` 与硬件身份加进 `RuntimeInspection` 和 `/health`，新字段用 `#[serde(default)]` 保持旧输出兼容。
- 验收：两个不同设备/副本可通过 `/health` 区分；单卡输出不变；不新增逐 token 字段或热路径拷贝。

**P0.3 设备键收口**

- 入口：`CudaDevice::enable_kernel_cache()` 与 `tuning/preparation.rs::device_identity()` 现在固定读取 0 号设备；把 `Selection.device` 传进去，让 JIT identity store 与 tuning stamp 使用同一设备身份。
- 验收：异构机上两个进程分别以 `--device 0/1` 启动时不共用设备身份；单卡启动时间不退化（缓存命中路径不变）。

**P0.4 组协议主机测试**

- 入口：在 `crates/engine/runtime/tests/support` 放一个只实现 `BackendProvider` 的 fake 2-rank backend，复用现有 Engine 测试构造；不新增产品 feature，不改 `src/` 的公开 API。
- 验收：group ticket 在所有 rank fence 前不发布；reserve 部分失败可回滚；prefix 取各 rank 最小值；任一 rank 错误进入隔离且继续 poll 其余 rank。

**P1 的前置**

- 两卡 NVLink 机型与 P0.4 的协议测试先就位；P1.1 只做 `RankGroup` 与 group ticket，P1.2 再改图与 collective，避免同一 PR 同时改协议和图。
