# CUDA 架构收口方案

状态：实施中，第一批已合入（见第 8 节）。事实基点为 main `c249c0ff`（PR #129），复核日期 2026-10-31；下文的行号与计数都以该提交为准，开工前先复核。
决策依据见 [ADR-0005](../../adr/0005-cuda-architecture-goals.md)，整体排期见[路线图](../README.md)的 A1。

本文回答一个问题：**怎样让性能调优更方便、kernel 更容易更换、新模型更容易接入、CUDA 版本兼容更容易。** 结论是不新增层，改两件事：**收口 CUDA 内部的算子层**（kernel 选择、lowering、调优框架、host 对 cuTile 的依赖），并**把 CUDA 越权持有的决策交回 engine**（时间、准入、页、前缀、推测）。两件事都是主线，只做前者，CUDA 仍然是“什么都管的后端”。

本文不复制其他方案。`Buffer` / `Kernels` / cell 的契约细节见 [kernel provider 方案](cuda-kernel-provider.md)；KV 分期见 [KV 方案](../performance/kv-cache-paged-attention.md)；推测协议见 [SPI 方案](../speculation/spi.md)。

## 1. 动机

公共层（请求生命周期、调度策略、`DataflowGraph`、compiler 校验）基本够用。成本集中在 `crates/backend/cuda`：它既是后端，也是一个隐藏的内部 compiler，同时还持有本该属于 engine 的账。

| 目标 | 现在为什么难 | 证据 |
|---|---|---|
| 换 kernel | 选择逻辑写在录制路径里，失败时才回退；kernel crate 没有统一的数值验收 | prefill 回退链 mma → oxide → tiled → legacy（`resident/capture_attention.rs:244` 起），`record` 用返回 `false` 表示回退；`cuda-kernels/tests/unit` 只有 `aot_artifact`、`target` |
| 调优 | 每个算子族各有一套 key、schema、证据和缓存字段，测量还有自己的 launch | `tuning/` 约 1 万行；8 个族有独立 `Key` 或证据结构；`tuning/attention/capture.rs` 自带 launch；kernel 指纹 `l2_cache_key` 随源码位置变化（见 PR #129） |
| 多模型 | 同一算子在多条录制路径上各写一遍，还有形状硬限制 | `TensorOp::Delta` 出现在 12 个文件，`Conv` 9 个，`Rope` 7 个；`dflash2/mod.rs` 有自己的 `record_projection`、`record_attention_block`；attention 要求 `head_dim` 为 2 的幂（`capture_attention.rs:79`）；embedding 要求 BF16（`loading/bindings.rs:187`） |
| 版本兼容 | host 不用 cuTile 就编不过；host 级的 `cuda-aot` feature 还不存在 | 88 个文件用 `Tensor`，43 个文件用 `cuda_core`；现有 `gate.sh cuda-aot` 只覆盖 kernel crate；tuning 候选依赖 JIT 特化；`attention_oxide.rs:60` 的制品形状固定 |

### 1.1 CUDA 层干了太多

上表是算子层的问题。第二个根因是职责错位：engine 已有契约，CUDA 却用私有实现应付，调度和准入因此用不上真数。

| 账 | 应归属 | 现状（CUDA 里） | 后果 |
|---|---|---|---|
| 时间 | 成本模型 | 没有实现 `completion_timing`，默认返回 `None`，`record_execution_completion` 不产生成本观测 | `CalibratedCosts` 在 CUDA 上得不到校准，SLO 预测和批次打包缺真实依据 |
| 准入 | runtime / scheduler | `executor/admission_pool.rs` 的 `pool_funding`、`defer_request_pool` 决定何时捕获多大的池；`loading/budget.rs` 958 行 | 双重决策，且准入不计准备中的预留：KV 方案已复现 5×8000 并发时第 5 条超时 |
| 页 | `infer-state` / scheduler | `SlotLedger` 和 `Paging` 私有；`shared_blocks`、`cow_copies` 恒为 0，`cow_tail` 恒为 false，页按 capacity 预付（`executor/provider.rs:88-114`） | engine 的按块准入对 CUDA 不成立；Metal 用 `KvCacheManager`，CUDA 是另一套，`cpu-runtime.md` 的目标 owner 与现状相反 |
| 前缀 | `infer-state` | 生命周期已用公共 `PrefixCache`，但复用单位是整段设备快照，`PREFIX_GRANULARITY` 写在 executor | 只能整段拷贝，engine 无法按页共享 |
| 推测 | `infer-workloads` | `mtp` 和 `draft` 共出现在 36 个文件，`executor/execution.rs` 2018 行、126 行涉及；`SpeculationPlan` 有两个同名类型（`ir/workload.rs:211`、`spi/model.rs:81`） | 每加一种 draft 都要改 executor 的策略分支 |

判断边界的规则：**设备事实归 CUDA，决策归 engine。** 例如 `loading/concurrency.rs` 从实际图和设备算 roofline 是设备事实，留在 CUDA 并上报；据此决定收多少请求归 engine。

这部分同样服务四个目标：新模型和新 draft 不用再碰 executor 策略；加设备时不用重验一套私有账本；调优能看到真实设备时间。

## 2. 目标形态

```text
DataflowGraph            模型语义，一 token 一张图，F32；不进入设备几何
      │
      ▼
host lowering            唯一入口：(算子, 模式, 权重, tuning 结果) → Plan
                         负责图几何、KV、slot、融合；不出现 cuTile 类型
      │
      ▼
Kernels { projection, attention, norm, … }
                         构造期选定 tile 或 aot；record(plan, buffers) 只录制不等待
      │
      ▼
runtime                  submit / poll，不重建拓扑
```

| 决策 | 归属 | CUDA 只负责 |
|---|---|---|
| 逻辑页、引用、COW、淘汰 | `infer-state` / scheduler（复用 `KvCacheManager`） | 物理块分配、拷贝、fence |
| 推测提议、接受规则、资源描述 | `infer-workloads` | 草稿图、物理回滚、追赶 |
| 逻辑提交、预算、公平性、准入决策 | runtime / scheduler | 上报真实设备时间和物理容量事实 |
| 融合、宽窄图、kernel 选择 | CUDA host，加载期定成 Plan | `Kernels` 只录制 |

**明确不做：**

- 不把 `cuda-kernels` 合回 `cuda`，不把 CUDA 图几何抬进 IR，不让公共 compiler 做融合或选核。
- 不让 `KernelRegistry::select` 成为运行时选核器，不为 12.9 新增 `BackendKind`。
- 不用环境变量分路，不把已验收的 cuTile kernel 仅为统一语言而重写成 `.cu`。
- 不用多 stream、后台线程或无界队列冒充执行重叠。
- 不做静态融合表 `(pattern_id, dtypes, mode)`：现有融合依赖图形状、权重和实测 tuning（如 `QuantPairPlan::new(graph, weights)`），静态 key 表达不了。用识别函数返回 `FusionPlan`。

## 3. 主线

顺序表示依赖和建议次序。每项可单独合并，单独回退。

### M0 门禁与基线

- **做什么：** 在 `tools/check/layout.py` 加棘轮，统计 host 的 `Tensor<`、`.partition(`、`.generics(`、`cuda_core`，只许下降；拦截 CUDA 与 runtime 中的 `WorkloadKind` 分支；新增 host 级 `cuda-aot` 编译 lane，以错误数做棘轮，允许先不通过。
- **基线（`c249c0ff`，`crates/backend/cuda/src`）：** `Tensor<` 786 处；`.partition(` 324 处；`.generics(` 207 处；用 `Tensor` 的文件 88 个；用 `cuda_core` 的文件 43 个。
- **注意：** 不用 `cutile::` 引用数做指标，56 个文件用 glob 导入，190 这个数会严重低估。`cuda-layers.md` 写的 `Tensor` 1,492 处与上述统计对不上，落地棘轮时统一口径并修正文档。
- **验收：** 新增调用让门禁失败，有对应测试；不改产品行为。

### M1 第一条纵切

- **做什么：** 按 kernel provider 方案 Phase 1 落地 `Kernels`、`Buffer`、`Device` / `Stream`，打通 projection。host 不再 `TileProjection::new`（`resident/pool_gemv.rs:80`），选择一次定完。
- **为什么先于其他算子族：** 现有契约是 `unsafe fn` 加 `DevicePtr`，不先定 `Buffer` 就迁移，unsafe 面会随算子族增长。
- **验收：** 13.x 模型输出逐位一致；固定负载性能不回退；记录 kernel 指纹的失配范围及其对已选配置的影响。

### M2 cell 验收工具

- **做什么：** 每个算子族一套验收，覆盖 CPU 参考对照、容差、有限值、单算子性能；tile 与 aot 共用同一入口。
- **验收口径：** 同一 cell 的迁移要求逐位一致。跨 profile 只要求参考值加容差，因为归约顺序可以不同（见 `aot/projection/pool_gemv.cu` 头注释）。设备未运行、零用例或全 ignored 不算通过。

### M3 统一调优框架

- **做什么：** 候选就是 Plan，只保留一种 key、证据和驱动，tuning 与服务走同一录制路径。key 使用内容身份（kernel 内容、合同、形状、设备），不依赖源码位置，否则每次搬移都会让缓存失效。从 `pool_dense` 开始逐族迁移，迁移时处理 `tuning.json` 与缓存的兼容或失效。诊断入口用 example 程序，不加环境变量。
- **参考：** PR #89 对 GEMV 缓存身份和可推导参数的审查，直接相关，开工前先读。
- **验收：** 新增候选不需要新 key 结构或新缓存字段；一条命令产出配对报告；迁移后默认选择不变，或变化有证据。

### M4 单一 lowering（风险最高）

- **做什么：** 用一个入口把 `(算子, 模式)` 降低成 Plan，`CaptureMode` 成为 Plan 参数。`Capture`、`batch.rs`、`batch_projection.rs`、`slot_verify*`、`prefill_projection.rs`、`dflash2/mod.rs` 复用该入口。attention 回退链收成加载期选定的 Plan，`record` 不再用 `Ok(false)` 表示回退。优先级应高于只改 capture.rs：自 10-15 起 `batch.rs` 改动 41 次，`program.rs` 27 次，`slot_batch.rs` 26 次，`slot_verify.rs` 24 次，capture.rs 未进前 12。
- **attention 回退链（已完成）：** 选择本来就在捕获前定好；被选中的 kernel 现在要么录制，要么让捕获失败，不再用 `Ok(false)` 静默退到慢路径。27B 和 0.6B 上均未触发，输出与 main 一致。
- **分批：** 先 projection，再 attention（与 C1A 合并推进），再 norm 和 recurrent。融合、宽图如果合不进通用入口，保留为显式 Plan 特化，但必须有通用路径可退。
- **前置：** PR #127 改动 `batch.rs`、`arena.rs` 和 executor，等它合并或定案后再动这些文件。
- **验收：** 每批 13.x 逐位一致且性能不回退；`resident/**` 的 `.generics(` 单调下降；未融合时在生效配置里可见。

### M5 通用 cell

- **做什么：** 每个算子族至少一个通用 cell，任意受支持形状可跑但可以慢；其他形状在加载期显式返回 `Unsupported`。AOT kernel 优先用运行时几何参数，`pool_gemv.cu` 是模板，避免“模型 × 架构 × 形状”的制品组合膨胀。注册表 `KernelRegistration` 随之改为真实能力，这需要给 provider 提供设备信息，因为 `CudaKernels` 目前是单元结构体且全部注册 F32 占位。
- **验收：** 现有非 2 幂形状限制逐条变为通用 cell 支持或加载期错误。

### M6 host 去 cuTile

- **做什么：** 未迁移的模块（vision、dflash2 等）在模块边界用 `Option` 能力隔离并返回 `Unsupported`，调用路径上不加 `cfg`。tuning 支持 AOT 候选，AOT 下候选是已有制品或运行时参数。
- **范围：** 主推理路径必须迁完；其他模块可隔离后按需迁，不追求一次全量清零。
- **验收：** `--features cuda-aot` 无工具链可构建 host；13.x 行为不变。

### M7 第二模型家族验收

- **时机：** 先做一次探测，不等 M4、M5（理由见第 9.3 节）：探测失败的项就是这两步的工作清单，完成时再验收一次。
- **做什么：** 选一个普通稠密 attention 解码器（无 Delta/Conv、无 MTP），先用微型 fixture 和独立参考。这是架构验收，不代表扩大正式支持范围。
- **验收：** 只改 provider；`resident/**` 零 diff；走通用路径能运行；自动调优能接管。做不到时，差距就是下一批 M4/M5 的工作清单。

### M8 L20 实机验收

按 kernel provider 方案第 10 节执行。编译通过不代表设备支持。

**状态：等设备。** 目前没有 L20，本项跳过，不算通过；在拿到设备之前，12.9 的结论只到第 9.4 节的文档推演。

## 4. 职责归还

与第 3 节同为主线，两条线基本独立，可以并行。**是否归还不由性能测量决定**：测量只决定第 5 节容量优化的先后。这里的每一项都先保持行为不变或走配对验收，不把换所有者和换策略放进同一次改动。

### O1 时间账

- **做什么：** 实现 CUDA `completion_timing`，使用设备时间，按 prefill、verify、draft 区分，灌入 `CalibratedCosts`。
- **已核实：** 观测在下一个 tick 开头才应用，而完成和 `dispatch_prepared` 在同一个 tick，所以样本不会让为它预规划的批次作废（`a_measured_completion_keeps_the_batch_prepared_for_it`）。线上用 `prepared_dispatched` 与 `prepared_discarded` 观察。
- **验收：** 会改变调度结果，走 B3 配对验收；SLO 预测误差下降，prepared 命中率不恶化。

### O2 准入账

- **竞态已不复现：** 现有代码在资源进展后重新计时；2026-10-31 在 27B 5×8000（16 槽）和 0.6B 14×8000（12 槽）上全部成功，后者整轮 48 秒，超过 30 秒超时。下一次复现之前不加预留账。
- **做什么：** 把 `admission_pool.rs` 拆开：CUDA 只报价和上报物理事实，何时要资源、按什么计价归 runtime。完成 KV 方案 P3 的按块计价。第二 flight 的预算以后也从这个入口计入。
- **验收：** 5×8000 竞态用例不再超时；CUDA 里不再有独立的准入判断；报价分项可读回。

### O3 页账

- **做什么：** `infer-state` 持有页账本，复用 `KvCacheManager`，不新建共享 arena（KV 方案第八节已否证）。`SlotPool` 改成执行镜像，只负责物理块分配、设备拷贝和 fence；统一 `block_size` / `bytes_per_block` 为同一报价单位。
- **第一步只换所有者：** 预付语义不变，所以 13.x token 和状态逐位对照可判定。按需增长和块共享是后续的策略改变，见第 5 节。
- **SPI：** 如需调整，只改现有 `kv_cache`、`state_page_growth`、reservation，不新增第二套容量接口。
- **注意：** KV 只是显存的一部分；递推状态按序列保存，仍由 CUDA 物理实现，engine 只记其字节数。
- **验收：** `kv_cache()` 的数据来自公共账本而非 CUDA 自算；CUDA 没有私有 free list 决策；取消和错误路径下页不泄漏、不提前释放。

### O4 推测账

- **前置：** M4 之后、O3 之后。回滚语义依赖页所有权，lowering 收口后迁移成本更低。
- **做什么：** 先拆开两个 `SpeculationPlan`（可立即做）。先把 MTP 和已合入的 DFlash2 都对照一遍公共契约，再按 [SPI 方案](../speculation/spi.md)步骤③把提议、接受、池宽和能力策略搬到 `infer-workloads`。此前冻结 DFlash2 的新接线。
- **验收：** MTP token、状态、资源行为与迁移前一致；新增 draft 不需要改 executor 的策略分支。

### O5 不立项的部分

Workload 边界目前是干净的：CUDA 和 runtime 中查不到 `WorkloadKind` 分支，`plain_greedy` 在 IR。只加门禁（M0），不做迁移。

## 5. 性能线

账变真之后才有意义的优化，由测量决定做不做、谁先做。C1 的现有工作不停，但遵守第 7 节的规则。

### P1 测量（可立即开始）

- **GPU 空泡归因（已测）：** 27B，闭环负载，C1/C4/C8/C16：设备时间占墙钟 99.0% 到 99.4%，步间间隙每步 0.17 到 0.27 ms，步长 20 到 41 ms，吞吐 105 到 739 tok/s。设备时间是步首到步尾，会把步内空闲算成忙，所以用 `nvidia-smi` 100 ms 采样交叉验证：活跃期利用率中位数 98%。
- **显存分项（C1M）：** 拆出 arena、KV、递推状态、draft、图和 workspace。

### P2 容量与重叠

**当前判定（2026-10-31，27B）：** 空泡潜在收益远低于 5%，按下方规则**暂缓 depth 2**；KV 只占报价的 6% 到 17%（16 槽 0.33 到 1.14 GiB，报价 5.85 到 6.66 GiB），**暂缓按需增长**，容量工作先看 target 池里的递推状态和 worker arena。这个判定只覆盖 27B 和这组负载，换模型或负载要重测。


依赖 O2、O3。默认顺序：

1. **按需增长：** decode 向 engine 要页，同时补齐缺页时的抢占或失败路径。
2. **D2-a：** 独立请求重叠，加第二 flight 的准入预算。
3. **块级前缀（KV 方案 P4）：** 递推状态无法按块共享，前缀由 KV 页加 recurrent checkpoint 组成，`hot_long` 不得回退。
4. **D2-b：** 同请求占位回填。已决定采用 ADR-0004 的占位回填，因为同请求连续 decode 是吞吐的主要场景；落地时同步修改 `cpu-runtime.md` 中“同一请求等上一步结果”的约束。

**决策规则：**

- C1M 显示显存不紧且空泡占主要份额：先做 D2-a。
- 空泡归因后潜在收益低于 5%（沿用非阻塞评审的最小有用收益）：暂缓 depth 2，记录否决证据。
- KV 不是显存大头：暂缓按需增长，先处理 arena。旧账目（27B，capacity 8192）是每槽 262 MiB KV 加 149 MiB 递推状态，并记录报价大头是激活 arena，这些数字需要重测。

**默认先增长的原因：** captured graph 绑定固定地址，第二条 flight 很可能需要第二份 arena（读代码推断，设计时证实），这会直接占用并发容量。按需增长和 depth 2 都改变 KV 的可见时机，不能同时做，否则结果无法归因。

**已有基础：** `prepare_next` 已在 flight 未完成时规划下一批（`pipeline/mod.rs:176`）。缺的是占位回填和第二次 submit，所以 depth 2 的验收是“第二批在第一批 fence 前提交，且空泡下降”。

### P3 多卡

最后做，按“一卡一 scheduler”实现。前置是 O3：SPI 中只有一套容量账。TP/PP 作为 program 几何，不进 capture 分路。

## 6. 完成判定

| 目标 | 判定 |
|---|---|
| 换 kernel | 只改 cell、制品清单和一行候选登记；验收工具自动给出数值和性能报告；host 零 diff |
| 调优 | 加候选不新增 key 或缓存字段；一条命令产出配对报告；测量和服务走同一路径 |
| 多模型 | 算子已覆盖的新模型只改 provider，`resident/**` 零 diff，走通用路径可运行，再由自动调优提速 |
| 兼容 | 加架构只改 `Target`、制品和 baseline，host 零 diff；`cuda-aot` 无工具链可构建；L20 通过实机验收 |
| 层边界 | CUDA 只有设备事实和物理执行：成本模型收到 CUDA 设备时间；页账本和准入决策在 engine；executor 无推测策略分支；门禁防止回流 |

**做完后仍然不会容易的部分：** 新算子语义和新量化格式仍要写 kernel、参考和验收；新模型的峰值性能仍要专门投入；每个新架构仍需制品和设备窗口。方案只保证这些工作有固定落点。

## 7. 工作规则

- **提交分三格：** cell（kernel、制品、Plan 参数、tuning 候选）、host（图几何、KV、准入、融合）、契约迁移（两边都可动，但必须逐位不变）。约束落在提交级，每个提交都要能编译；PR 内仍可顺手修 bug。
- **冻结：** 不新增 attention 回退分支、host 中的 `#[cutile::module]`、DFlash2 接线；新路径必须先有 Plan。CUDA 不新增私有账本或策略决定，缺接口时先改 SPI。
- **设备窗口：** 冲突时按 P1 测量、C1 性能验收、架构迁移的逐位对照排序。O3 第一步保持预付语义，先用 CPU 与逐位对照推进，不阻塞 C1。
- **单一变量：** 性能 A/B 每次只改变一个可归因策略，引用对应 baseline；架构收口若逐位不变，用逐位对照加固定负载验收，不等完整矩阵。
- **证据口径：** release 同条件测量；分别报告已运行、仅编译、未运行和设备验收范围。CPU 通过不代表设备支持。

## 8. 接手入口

**已落地：** G1–G3；M4 的 attention 回退链；M7 探测；P1 测量；M0 棘轮和 `WorkloadKind` 拦截；O1 设备时间上报；O4 第一步（模型侧类型改名 `DraftHead`）；C1M 的池显存分项日志。host 级 `cuda-aot` lane 随 M6 落地，现在加只是一条必失败的 lane。

**下一步：**

1. **M1：** `Buffer` 与多个算子族一起设计。host 目前只有一个契约调用点，不单独包 `Kernels`。
2. **O3：** 页账本换所有者，保持预付语义。

**27B 显存（2026-10-31，16 槽，容量 8192，报价 5.85 GiB）：** target 池 3.40 GiB，其中 KV 只有 0.33 GiB；target worker 0.95 GiB，draft worker 0.64 GiB，draft 池 0.32 GiB。按第 5 节的决策规则，KV 不是大头，**按需增长暂缓，先看 arena 和递推状态**。

**未知项：**

- O3 是否需要调整 SPI，`SlotPool` 的物理块能否直接镜像 `KvCacheManager`（Metal 布局不同，文档记录为各自保留）。
- 第二条 flight 是否确实需要第二份 arena，需要多少显存。
- M4 能合并到什么程度，融合路径可能有多少保留为特化。第一批迁移后更新本节。
- kernel 指纹改为内容身份后，现有 `tuning.json` 条目如何迁移。

**本方案的证据范围：** 第 1 至 7 节基于代码与文档阅读；O1、O2 的核实和第 8 节的显存数字来自 RTX 5090 release 实测。其余性能数字来自引用文档，未重测。

## 9. 文档推演：四条接入路径走一遍

L20 不在手边，所以用另一种办法检验架构：按操作手册的步骤对照 main `ef4d662` 的代码逐步走，看每一步要改哪里。这是读代码的推演，**不能代替实机**。结论分三栏：今天怎么样、方案做完后怎么样、推演发现的新问题。

### 9.1 调优：加一个新候选

| 步骤 | 今天 | 方案做完后 |
|---|---|---|
| 定义候选 | 在所属算子族的 `tuning/` 下写，每族有自己的 `Key`、证据和缓存字段 | 候选就是一个 Plan（M3） |
| 测量 | 族内自带 capture 和 launch | 与服务同一条录制路径 |
| 看服务收益 | `fixed-work-serving.py`；每步设备时间已可读（`gpu_execution`） | 一条命令产出配对报告 |
| 判定 | 逐位相同或走质量门禁 | 不变 |

**判断：路径合理。** 新问题：M3 的“候选即 Plan”依赖 Plan 类型，目前只有 `PoolGemvPlan`，所以 M3 不能先于 M1。现有顺序已满足。

### 9.2 新硬件：一块有 cuTile 的新 GPU（如 sm_100）

| 步骤 | 今天 | 问题 |
|---|---|---|
| 识别设备 | `Other("sm_100")`，加载继续 | **有两套身份：** host 的 `CudaTarget` 和 kernel 的 `Arch` |
| 选 kernel | FA4、oxide 按架构开放，其他架构回退通用路径 | 18 个 host 文件出现架构名，违反 kernel provider 方案的放置规则 |
| 算并发 | `loading/concurrency.rs` 的 `Rates` 对 `Other` 返回 `None` | 算力拐点要改 host，违反第 6 节“加架构 host 零 diff” |
| 调优、预算 | 按设备身份自动重新测量 | 无 |

**判断：今天能靠回退路径跑起来，这条是好的；但拿到峰值要改 host。** 补进方案：

- **G1（已完成）：** 合并 `CudaTarget` 和 `Arch`，只留 `contract::target` 一套。
- **G2（已完成）：** host 只有制品目录可以写架构名，门禁拦截其余位置。
- **G3（已完成）：** 算力参数搬到 `Arch::tensor_rates`，host 只读取。

### 9.3 新模型：一个普通稠密 attention 解码器

| 步骤 | 今天 | 问题 |
|---|---|---|
| provider、图 | 只改 `providers/` 和 recipes，`decoder::lower` 可复用 | 无，这层已经干净 |
| 能否执行 | `registry.rs` 全部注册 F32 占位，编译器挡不住 | 要到录制时才失败（M5） |
| 形状限制 | attention 要求 `head_dim` 为 2 的幂，embedding 要求 BF16 | 已知（M5） |
| 融合 | 按图形状识别，不命中就走通用路径 | 只慢不错，要确认未融合时可见（M4） |
| 调优、显存 | 按形状和图自动推导 | 无 |

**探测结果（2026-10-31，RTX 5090，12 种微型 Qwen3 几何，随机权重走真实服务路径）：** 头维度 64、128、256，无 GQA，单层，宽 MLP，`hidden` 与头数不匹配，F16、F32 全部能出 token，单独和并发的输出一致；头维度 48、96、100 原本能启动，却在第一个请求隔离引擎，现已改为加载期返回 `Unsupported`。随机权重只能证明路径通，不证明数值对，数值对照属于 M2。尚未覆盖的：非绑定权重、非 Qwen3 的 provider、Delta 和 Conv。

**判断：** 模型只要形状落在 2 幂范围内，今天就能只靠 provider 接入。M5 的真实缺口收窄为非 2 幂的通用 cell。因此调整顺序：

- **G4：** M7 提前做一次探测。微型 fixture 只需 CPU 参考和一台现有 GPU，失败项直接成为 M4、M5 的工作清单，比先改 2700 行的 `batch.rs` 再验收更便宜。

### 9.4 CUDA 12.9：L20（sm_89，无 FP4，只能用 SASS）

| 步骤 | 今天 | 问题 |
|---|---|---|
| 无工具链构建 | `cuda` feature 强制带上 cutile、cuda-core、`infer-cuda-kernels/tile` | host 没有 `cuda-aot`（M6） |
| kernel | `AotProjection` 已在 sm_120 上通过捕获与回放；只有一个 entry | 其他算子族都要写（10.1 清单） |
| 图录制 | host 用 cuTile `Scope`（54 个文件涉及 capture 或 Scope） | `Recording` 已有两种形式，但捕获的发起方还在 host |
| 执行 | `run_op`、`fence` 基于 `cutile::cuda_async`（19 个文件） | **方案原来没有点名** |
| 设备时间 | O1 的 `StepClock` 用 `cuda_core::Event` | 要搬到 `Device` 后面 |
| 数值策略 | `PrecisionPolicy::Preferred` 按能力解析，无 FP4 自动退到 BF16 | 无 |
| 调优 | 键含设备和驱动，不混用 | AOT 候选没有 JIT 特化（M3、M6） |
| 准入、KV、调度 | 不依赖 kernel 来源 | 无 |

**判断：engine 和数值策略这两层已经满足要求；难点全在 host 的 cuTile 依赖，方向正确但工作量不小。** 补进方案：

- **G5（M1）：** `Device` / `Stream` 契约必须同时覆盖**捕获**和**异步执行**，不能只覆盖 `Buffer` 和 kernel launch。这是 M6 最大的风险，M1 设计时要先定。
- **G6：** kernel provider 方案第 3.5 节和第 10 节写着 `tile`、`aot` 不许同时编，与第 3 节、第 3.6 节的“可叠加”矛盾。以可叠加为准，已修正。

### 9.5 结论

| 目标 | 推演结论 |
|---|---|
| 调优 | 合理，M1 后推进 M3 |
| 新硬件 | 能用的路径今天已通；峰值路径要补 G1–G3 |
| 新模型 | 模型层已干净；CUDA 侧未经验证，用 G4 探测 |
| CUDA 12.9 | 分层合理；G5 是最大未知，L20 实机仍未验证 |

engine、SPI、模型层的边界经得起这四条路径，**没有发现需要新增层或推翻方向的问题**；缺口都落在第 3 节已有的 M 项内。
