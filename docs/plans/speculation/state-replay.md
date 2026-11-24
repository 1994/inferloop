# Recurrent 状态重放与回滚分层方案

状态：设计方案，基于 2026 年 10 月 9 日的 `7c79d99`。本文的新增契约和模块均为拟议设计，未修改实现。

整体排期与任务状态见 [路线图](../README.md) 的 R1。本文维护本项设计与验收。

ReplaySSM 的输入记录思想可以用于多种 recurrent 模型；具体记录格式、输出计算和状态转移则依赖算子。公共层统一“提交接受前缀”的状态协议，backend 负责选择和执行物理回滚策略。MTP 与 DFlash2 共用 target 回滚能力，算法 provider 只提出验证布局和状态需求。

CUDA gated-delta 的 ReplaySSM 物理策略（缓存修正值 u 与 k、检查点加环形缓冲、chunked 验证）见 [ReplaySSM 设计](../performance/replayssm-gdn.md)。

接口全貌见 [推测解码 SPI 方案](spi.md)，容量、数值与参考性能见 [Infernix 调研](../../research/infernix-performance.md)，CUDA 执行机制见 [性能优化方案](../performance/cuda.md)。性能实验执行 [性能基线方案](../performance/baseline.md)。本文维护分层与通用性边界，不重复性能估算。

## 已实施：gated-delta 接受前缀参考（E1 删除执行器前保住的数值对照）

第一阶段的 Record/Fold 需要一份**不依赖被测实现**的数值参考。它原本只存在于 CPU host 执行器的
`delta()` 里，而 E1 会删掉那个 crate，所以现在把公式与数值固化成登记制品：

- 导出器 `tools/fixtures/export-delta-state-golden.py`（冷路径、纯标准库）重新实现该步：
  q/k 归一化（`sqrt(sum(x²)+eps)`，q 另乘 `sqrt(key_dim)`）、`beta = sigmoid(row1)`、
  `decay = exp(-exp(row3) * softplus(row2 + row4))`、`state *= decay`、
  `delta = (v - k @ state) * beta`、`state += outer(k, delta)`、`out = q @ state`；
  **f64 累加、状态每步回落到 f32**——与被删掉的参考实现一致（不一致就无法作为对照）。
- 制品 `examples/recurrent-delta/golden.json`：2 个 key head / 4 个 value head / key_dim = value_dim = 4、
  3 步，记录几何、公式、确定性输入、基点状态、每步状态与输出、容差与理由；重复运行逐字节一致。
- 同时记录 R1 成立的前提：**从基点重放记录输入能复现最后状态**（`replay_property.holds = true`），
  这正是"提交接受前缀时可以不物化每个候选快照"的依据。
- `crates/engine/state/tests/recurrent_reference.rs` 把制品绑回代码：几何必须与记录的状态长度
  一致、基点必须为空、每步状态与输出必须有限、最后状态不得等于基点、重放性质必须为真。

CUDA 的 delta 算子接进来后直接对照这份 golden 即可，不必再依赖已删除的执行器。

**参考实现现在也在代码里**（不只是 JSON）：`crates/engine/workloads/src/recurrent.rs` 提供
`DeltaGeometry` / `ConvGeometry` 的 `step()` 与 **`fold()`**——后者就是本方案的提交语义
（从基点只重放**已接受前缀**，卷积则从基点窗口与已接受输入中 gather 末尾窗口）。它落在
`infer-workloads`，正是方案表格里"通用接受规则与**独立正确性参考**"所指的位置，也符合刚落地的分层
门禁（该 crate 的生产依赖只有 core/ir/spi/models）。算术刻意保留原参考的舍入点：状态 f32、累加 f64，
卷积用**分开的乘加**而非 FMA（否则数值会偏离已记录的 golden，因此带理由的 lint expectation 说明了
这一点）。`crates/engine/workloads/src/recurrent_reference_tests.rs` 的 6 条用例逐位对比两份 golden（1e-6 以内）并覆盖
`fold` 的正例与"接受前缀超出记录窗口"的反例。

**Causal conv 的窗口同样已固化**（方案表格里"记录输入，提交时从基点历史与接受输入选取末尾
history"那一行）：导出器 `tools/fixtures/export-conv-history-golden.py` 按参考实现的顺序重放
（每通道一行 kernel 权重、f64 累加、SiLU 后回落 f32、窗口左移并追加本次输入），制品
`examples/recurrent-conv/golden.json` 记录几何、公式、确定性输入、基点窗口、每步窗口与输出、容差，
并**验证**了 R1 的提交方式：从基点窗口与已接受输入中 gather 末尾窗口，与逐步重放得到的窗口一致
（`replay_property.holds = true`）。`recurrent_reference.rs` 里另有两条用例把它绑回代码。

## 已实施：验收矩阵中"不需要设备"的用例已落到参考测试里

R1 的验收要求覆盖"**零提交、首候选拒绝、部分/全接受、padding、拒绝尾部改写、串行/池化、取消、重复完成及
多轮状态传播**"✓。其中**设备/verifier 才有的部分**（拒绝尾部改写、串行/池化、取消、重复完成 ✓）留给
D2/设备窗口 ✓；**参考层就能证明的部分现在补齐** ✓：

- `crates/engine/workloads/src/recurrent_reference_tests.rs` 原有"复现记录步"与"全接受 = 记录末态" ✓，
  但**零提交**与**多轮传播**没有覆盖 ✗，而"部分接受"只是 `assert_ne!` ✗（只能说明"不一样" ✗）。
- 现在对 **Delta 与 Conv 各自**逐一核：`accepted = 0..=n` **每一个前缀**都必须落在该记录点上的状态/窗口 ✓
  （零提交回到 base ✓，全接受回到末态 ✓，中间每个 k 都对齐 ✓）；并核**多轮状态传播** ✓ ——
  `fold(k)` 之后把剩余步按普通 `step` 跑完 ✓，终点必须与不中断的一整轮**逐值相同** ✓（1e-6 ✓）✓，
  这正是"拒绝尾部后可以继续、无需重放整段历史"所依赖的性质 ✓。

新增三个用例 ✓（Delta 前缀矩阵 ✓、Delta 传播 ✓、Conv 前缀矩阵 ✓），既有用例不回归 ✓。

## 通用的是思想，适配单位是状态算子

对于 `S_t = F(S_{t-1}, u_t)`，可以保留基点 `S_0` 和短窗口输入记录 `u_1…u_T`，接受长度确定后只重放有效前缀，避免为每个候选保留完整状态。记录的是驱动状态更新的实际中间输入，不是原始 token；重放不应重新运行整套模型投影与 attention。

适合采用该策略的条件是：记录包含恢复转移所需的全部信息，其大小明显小于完整状态，状态更新可重复执行，重放成本能被内存或服务收益抵消。外部依赖、随机输入或量化尺度参与转移时，也必须固定或记录。不能由 `StateKind::Recurrent` 这样的类别名称自动推断支持。

[ReplaySSM 原作者说明](https://tridao.me/blog/2026/replayssm/) 以 Mamba-2 为例，并给出 Gated DeltaNet 的适配；除了输入记录，还包括不物化每步状态的 output-only 计算与延迟 flush。后两者依赖具体递推结构，并可能改变浮点计算顺序。因此本项目分两阶段：先做保留有限精度转移的接受前缀 Record/Fold，再独立研究普通 decode 的 output-only 与跨轮延迟 flush。

| 状态或算子 | 建议策略 | 通用性边界 |
|---|---|---|
| GDN / 当前 `TensorOp::Delta` | raw key/value、门控输入记录，加同一状态转移的 Fold | 第一阶段的具体实现；按 head 几何、dtype、归约和 cast 边界编译 |
| Causal conv history | 记录输入，提交时从基点历史与接受输入选取末尾 history | 不是 GDN 的公式；可以采用精确 gather，不必重算卷积输出 |
| Mamba 等 SSM | 独立的输入格式、递推和可能的 output-only/flush | 技术思想适用，当前项目没有因此自动获得对应算子支持 |
| 其他 recurrent state | 经算子证明和测量后选择 replay 或 snapshot | 完整状态小或更新昂贵时，replay 未必有收益 |
| Full attention KV | 有效长度截断、页引用与 COW | 原有回滚机制足够，不必套用 recurrent Record/Fold |

Hybrid 模型可以在同一次提交中对不同状态区域采用不同策略。Layer 的状态类型、更新语义与公共提交位置应独立描述，不用一个全模型 `use_replayssm` 开关代替。

## 与 MTP、DFlash2 的互补关系

MTP 与 DFlash2 改变候选的生成方式，ReplaySSM 改变 target recurrent 状态的记录与提交方式。候选进入共同 target verifier 后，可以使用同一个 replay 计划。兼容性取决于 target 的状态算子、精度、验证布局与资源，而非 proposal 算法的名称。

| 组合 | 互补作用 | 条件与边界 |
|---|---|---|
| MTP + Record/Fold | 减少 MTP verification 的逐位置快照，帮助较深候选或多 slot 保持池化 | 不改变 MTP 候选评分；更深提议是否划算仍看完整周期和实际接受长度 |
| DFlash2 + Record/Fold | 降低较宽 block verification 的回滚容量，缓解本项目原生 8 列、多 slot 的准入阻塞 | 草稿权重、features、draft KV 与 arena 仍需报价；不直接提高接受率 |
| MTP / DFlash2 分轮选择 + Record/Fold | 两个 provider 复用 target 回滚实现和已提交状态 | 切换仍需检查各自 draft 状态与 target feature 覆盖，计入重建成本 |
| MTP + DFlash2 同轮组合 + Record/Fold | 组合 provider 产生候选后，共同 verifier 只提交一条权威接受路径 | 同轮组合算法需独立证明收益；树形布局还需 replay 的分支与路径 gather 能力 |

在保持本次 target 有限精度转移与提交语义的条件下，Record/Fold 不要求改变 proposal 分布或接受算法。正确性仍要验证状态轨迹；它不能补偿候选条件、随机接受 `q` 或 draft 追赶中的错误。

对本项目，DFlash2 的宽验证显存约束使这项互补尤其有价值。MTP 也可受益，但较浅窗口原先能装下时，额外 Fold 可能抵消部分时间收益。应分别比较同一个 provider、同一宽度和 slots 下的 snapshot 与 replay，再评估是否扩大验证宽度或并发；不要同时改变算法和回滚策略后把全部收益归给 ReplaySSM。

纯 attention target 没有这项 recurrent 回滚收益，仍可使用 MTP 或 DFlash2。三种技术能够共存也不意味着加速倍数相乘：最终依据完整周期时间除以实际新发布 token 数判断，且组合模式需对照已使用相同回滚优化的最佳单 provider。

建议先在现有 MTP 中验证 Record/Fold 的数值与 serial/pooled 复用，再接 DFlash2，最后评估分轮选择和同轮组合。ReplaySSM 作为共同 target 能力，在这些阶段中持续复用。

## 分层与职责

这里的 backend 表示设备执行层，包含 CUDA、Metal 等独立实现。ReplaySSM 的思想与公共提交契约不绑定 CUDA；CUDA 的私有模块建议只是第一套实现的归属。MTP/DFlash2 的公共算法 provider 与模型图归属见 [SPI 分层方案](spi.md)，共享协议不要求共享设备 kernel。

| 层 | 职责 | 本项目对应位置 |
|---|---|---|
| 模型与执行 IR | 声明 Delta/Conv/SSM 等状态更新语义、依赖、几何和精度要求 | 模型 provider；[ModelIr](../../../crates/foundation/ir/src/model.rs)、[DataflowGraph](../../../crates/foundation/ir/src/dataflow.rs) |
| IR / SPI 公共契约 | 表达验证布局、每序列提交位置、能力与资源报价；返回事务 receipt | [BackendProvider](../../../crates/foundation/spi/src/backend.rs) 与拟议 speculation 执行计划 |
| Runtime / state / scheduler | 请求和逻辑前缀所有权、epoch、预算、提交确认及取消 | `infer-runtime`、`infer-state`、`infer-scheduler` |
| Backend 执行与物理状态 | 编译回滚计划，管理基点、records、destination、graph、fence；协调 KV/recurrent/features 的提交 | CUDA executor 与 resident program；Metal 独立实现 |
| Backend 状态算子 | Record、状态转移、Fold、conv history gather；保证数值轨迹 | CUDA 的 [recurrent](../../../crates/backend/cuda/src/resident/recurrent.rs)、[capture_state](../../../crates/backend/cuda/src/resident/capture_state.rs) 等 |

主要实现放在 backend 的物理状态执行与 recurrent 算子之间。CUDA 可以新增私有 `resident/recurrent_replay` 模块，让 serial [batch](../../../crates/backend/cuda/src/resident/batch.rs) 与 pooled [slot_verify](../../../crates/backend/cuda/src/resident/slot_verify.rs) 共用；executor 只协调一次验证事务和接受结果。

公共层不持有 CUDA tensor/stream/graph，不逐 token 调用重放虚接口。MTP/DFlash2 provider 不自行恢复 target 状态，也不复制一份 GDN Fold。`infer-state` 管理共有生命周期和逻辑状态协议，设备数值计算继续由 backend owner 完成；不因为技术可通用就新增一个依赖 CUDA 的公共 crate。

模型图保留 `Delta`、`Conv` 等原语。第一阶段不添加全模型 `TensorOp::ReplaySSM`：重放是这些状态算子的执行策略，不是更换模型结构。以后如果支持新的 SSM，模型 IR 仍需表达其实际更新语义，不能直接复用 Delta 的数学。

## 公共契约表达结果，backend 编译策略

在现有 speculation 执行计划上补足状态回滚需求，在 backend 冷路径编译物理计划。下面是契约内容，不要求新增同名 trait 或热路径调用：

| 契约内容 | 要求 |
|---|---|
| 验证与提交 | 明确线性/树形布局、每个序列的基点和实际消费位置，提交范围包含 seed 与接受候选 |
| 正确性 | 物理状态等于本次实际 verify 对应有效前缀的状态；第一阶段 Record/Fold 直接检查 recurrent/conv bits |
| 能力 | 按已绑定 program、状态算子、精度和布局声明支持，不作为设备型号的全局布尔能力 |
| 策略与报价 | backend 在逐位置 snapshot、输入 replay、稀疏 snapshot 间选择；报告实际策略、slots/width、增量峰值字节和限制 |
| 身份与生命周期 | state/config epoch、事务身份、设备 owner、完成 fence；records 不跨越未完成写者释放 |

资源选择发生在加载、计划编译或容量档位建立时，图捕获前确定有界存储。热路径可以把 Record、设备接受判断与 Fold 编进图，使用设备上的每序列接受长度，不要求将阶段拆成主机往返。

现有 [StateRecipe](../../../crates/foundation/ir/src/state_recipe/mod.rs) 描述 retained sequence state，注释明确将 per-batch activation scratch 与权重分开。第一阶段继续用它报价 committed recurrent/conv 状态；每轮 records、临时 destination 和回滚图 workspace 纳入执行计划的增量资源报价。Sequence 独占的临时资源随 sequence lease，pool 共享的资源随 pool/flight；持久分配但短生命周期的 buffer 也必须计算实际峰值，且不与原基点重复计费。

若未来跨普通 decode 多轮保留尚未 Fold 的输入日志，它将成为 live sequence state 的一部分，必须扩展 retained recipe、迁移、导出和 prefix restore 契约。这是第二阶段，不能把第一阶段的 transient records 直接当作可持久化 checkpoint。

## 第一阶段执行过程

1. 建立事务：绑定 committed 基点、位置与 epoch，预留有界记录及可能的 destination。
2. 验证并记录：基点保持只读；保存本次状态转移实际消费的输入位，输出与特征仍由 target 计算。
3. 决定：生成每序列实际消费长度。首个草稿被拒绝时通常仍需消费 seed；取消时则按协议恢复基点。
4. 提交：Fold seed 与接受前缀，conv history 做相应选取，KV 截断到同一消费位置，失效拒绝尾部 features，返回 receipt。
5. 完成：逻辑 owner 确认 receipt 后发布结果；记录与临时状态在最后读者 fence 完成后才可复用。Draft provider 按自身规则追赶。

全接受也需要 Fold，因为 Record 未更新 committed 基点。可以另做保留 final destination 的全接受快路径，但需要独立资源与数值验收。Partial Fold 中发生错误时，若基点或 destination 已受到不可恢复的修改，标记请求状态不可复用；不能用未确认的 receipt 发布逻辑提交。

这套机制解决 target 的接受前缀状态提交，不替代 MTP 的 target-conditioned hidden 追赶，也不替代 DFlash2 的 feature/KV 生命周期。MTP 与 DFlash2 同轮组合仍由 [组合方案](composition.md) 定义。

## 实施顺序与复用验收

先为当前 Delta/Conv 建立 backend 私有 replay 计划和共享状态转移，接到现有 MTP 的 serial/pooled verifier；再让独立 DFlash2 使用同一 target verifier。对尚不支持 replay 的状态算子保留 snapshot 选择，容量不足时显式报告或按已有策略缩小档位。

树形验证单独声明能力：输入记录需要 parent/ancestor 布局与接受路径 gather，不能将“线性链 replay 已支持”解释为树形也已支持。Metal 可先沿用 snapshot，在能力和报价允许后独立实现重放，不依赖 CUDA 的记录格式。

数值验收按“状态算子 × 实际精度 × 接受长度”组织，公共层测试位置、所有权与 receipt。独立数学 oracle 和设备 Fold 对同一 verify 的状态比较放在现有测试体系，不新建测试推理后端。覆盖零提交、首候选拒绝、部分/全接受、padding、拒绝尾部改写、串行/池化、取消、重复完成及多轮状态传播。

通用性验收要求更换 proposal provider 后 target 回滚实现仍可复用；复用 target 回滚时不新增 MTP/DFlash2 条件分支。性能验收绑定共同契约下已验收的 profile 与 baseline ID，分别报告 records 与完整预算、Fold 时间、完整周期时间和池化成功率，避免把协议复用当作性能已提高。
