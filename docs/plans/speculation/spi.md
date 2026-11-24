# 推测解码 SPI 与状态协议方案

状态：设计方案。基于 2026 年 10 月 9 日的 `7c79d99`；下文新增类型与接口均为拟议契约。

整体排期与任务状态见 [路线图](../README.md) 的 I1。本文维护本项设计与验收。

MTP 与 DFlash2 各自负责草稿生成，共用目标模型验证、状态提交、资源管理与观测协议。SPI 描述算法需求和执行计划，backend 将计划编译为设备执行路径。首先迁移现有 MTP，保留其行为；随后接入 DFlash2，最后验证组合算法。

接入细节见 [DFlash2 接入方案](dflash2.md)，组合语义见 [MTP 与 DFlash2 组合方案](composition.md)，设备执行优化见 [CUDA 性能优化方案](../performance/cuda.md)。性能实验的构建、配置与基线条件统一执行 [性能基线方案](../performance/baseline.md)。

## 当前边界与需要调整的部分

现有 [SpeculationProvider](../../../crates/foundation/spi/src/extensions.rs) 只有规划入口。实际草稿、验证、接受判断与追赶逻辑集中在 CUDA 的 [execution](../../../crates/backend/cuda/src/executor/execution.rs) 和 [drafting](../../../crates/backend/cuda/src/executor/drafting.rs)，加载参数、池化宽度和 capability 也以 `mtp_depth` 为中心。接入第二种算法会继续扩大这些条件分支。

同时存在两个 `SpeculationPlan`：[模型声明](../../../crates/foundation/spi/src/model.rs) 表示 MTP 权重前缀、层数与融合槽；[执行 IR](../../../crates/foundation/ir/src/workload.rs) 表示 proposal、verification、hidden taps 与接受策略。迁移时应区分模型绑定与执行计划，避免把外部草稿模型强行解释为 MTP head。

| 所属层 | 拟议职责 | 需要保留的边界 |
|---|---|---|
| `infer-ir` | 算法标识、特征需求、候选布局、采样契约、提交结果与资源描述 | 不暴露 CUDA tensor、stream 或 graph 类型 |
| `infer-spi` | provider 的能力声明、绑定需求与规划接口 | 不要求逐 token 虚调用或主机同步 |
| `infer-models` 与 `infer-model-recipes` | target、MTP head、外部 draft 的配置、权重槽与算子拓扑 | 模型图继续由模型 provider 定义 |
| `infer-workloads` | 内置 MTP、DFlash2 与组合 provider 的规划实现、通用接受规则与独立正确性参考 | 不拥有设备状态或 CUDA graph |
| `infer-runtime`、`infer-state`、`infer-scheduler` | 请求所有权、逻辑提交、资源票据、预算和公平性 | 不直接依赖具体算法执行器 |
| backend | 草稿与验证图、设备 buffer、KV、recurrent state、物理回滚和追赶 | 具体融合、录制和执行留在设备 owner |

## 现状清点（I1 第一步，已完成）

两个同名计划仍然并存，这正是方案要求区分"模型绑定"与"执行计划"的地方：

| 类型 | 位置 | 字段 | 语义 |
|---|---|---|---|
| `infer_spi::SpeculationPlan` | `crates/foundation/spi/src/model.rs:81` | `prefix`、`layers`、`fusion` | 模型绑定：草稿 head 的权重前缀、层数与融合槽 |
| `infer_ir::SpeculationPlan` | `crates/foundation/ir/src/workload.rs:158` | `proposal`、`verification`、`hidden_state_taps`、`max_candidates`、`acceptance_policy` | 执行计划：proposal/verify program、抽头与接受策略 |

算法策略在 backend 的分布（按 `mtp*`/`draft*`/`speculat*` 标识符在代码中的出现次数统计，
`crates/backend/cuda/src`，注释不计）：

| 文件 | 次数 | 承担的职责 |
|---|---:|---|
| `executor/execution.rs` | 48 | 草稿图装载、slot pool、验证与状态结算 |
| `loading/bindings.rs` | 46 | 张量绑定与槽位映射 |
| `loading/mod.rs` | 39 | 加载参数与预算 |
| `executor/drafting.rs` | 15 | 草稿步进、采样与位置映射 |
| `executor/state.rs` | 14 | 草稿状态与预算记账 |
| `executor/provider.rs` | 11 | capability 声明 |
| `executor/mod.rs` | 11 | 执行器装配 |
| 其余 11 个文件 | 17 | 前缀复用、池化、profiling、常量、resident 批/验证 |

合计 **211 处、18 个文件**——这就是"接入第二种算法会继续扩大条件分支"的具体规模。
其中 `resident/slot_verify.rs` 正是另一个 session 当前在改的文件，因此实际拆分要等那块安静下来。

公共层目前只剩 4 处算法名，已作为 I1 的**移除清单**由门禁 ratchet 冻结：

- `crates/foundation/ir/src/diagnostics.rs`：`mtp_depth`——公共诊断字段以单一算法命名；
- `crates/foundation/spi/src/model.rs`：`mtp_layers`——字段与构造参数。

`tools/check/policy.py` 新增 `check_algorithm_neutrality()`：任何**新增**的算法名都会让门禁失败
（注释与 `#[cfg(test)]` 块除外，规则针对会分支的代码），清单条目一旦被移除而条目还在，也会失败，
强制同步清理。当前 34 个 tools 单测通过，其中 5 个覆盖这条规则。

**公共层的算法名已清零（I1 步骤①完成）**

E1 删除执行器后，公共层只剩两处算法命名 ✗：`infer-ir` 诊断里的 `mtp_depth` 与 `infer-spi` 模型绑定里的
`mtp_layers`。现在分别改为 **`draft_depth`**（"已解析的**草稿**深度，无论由哪个算法提供提议" ✓）与
**`draft_layers`**（"配置声明的**草稿**层数" ✓），生产端（`infer-models` 的 qwen provider ✓）与其测试
同步改名 ✓；CUDA 自身那个**属于设备执行**的 `MtpOptions::mtp_depth` 保持原名 ✓（按分层它本就该由
backend 持有 ✓），只把构造 `infer_ir::ExecutionProfileInspection` 的字面量字段改为 `draft_depth` ✓。

因此**算法中立棘轮清零** ✓：`policy.py` 的 `ALGORITHM_NAME_BASELINE` 从四条变成 **空表** ✓，规则退化为
"公共层不得出现任何算法名" ✓（`infer-ir` / `infer-spi` / `infer-workloads` ✓），而"条目过期即失败"的
机制保留着，供将来的迁移继续使用 ✓。

**两条边界现在由门禁强制，而不只是写在方案里**

方案要求"公共算法实现仅依赖共有 IR/SPI 和模型描述，不反向依赖 `infer-backend-cuda`、
`infer-backend-metal`、cuTile 或 Metal API"，并要求 runtime/state/scheduler"不直接依赖具体算法执行器"。
`tools/check/layout.py` 过去只覆盖了 foundation 与少数几个 crate，**`infer-workloads`（未来 provider 所在
层）与 `infer-runtime` 都没有这条约束** ✗。现在：

- `crates/engine/workloads` 的生产依赖只允许 `{core, ir, spi}` 加模型描述（`infer-models`、
  `infer-model-recipes`）——与它当前实际的四个依赖一致，因此是防回归；
- `crates/engine/runtime` 除原有规则外，生产依赖不得出现任何 `infer-backend-*`（设备执行器）。
  规则只看**生产依赖**（`production_dependencies`），测试执行器作为 dev-dependency 不受影响。

layout 单测新增了对应案例（workloads 依赖 cuda/metal/kernel-api/runtime 均被拒，runtime 依赖
cuda/metal 被拒）。

**下一步（按风险从低到高）**：① 把 `mtp_depth`/`mtp_layers` 改成 draft 语义（公共 API 变更，需要
后端同步改，故此步要与后端改动一起排）；② 把 `acceptance_policy: String` 换成结构化契约；
③ 按上表把 drafting/pool/capabilities 的策略移入 `crates/engine/workloads` 的 provider，backend
只留算子与设备执行。

## 算法语义与设备实现的分层

MTP、DFlash2 和 ReplaySSM 都不在公共架构上绑定 CUDA。当前 MTP 热路径集中在 CUDA，是已有实现的位置；DFlash2 和 ReplaySSM 在本项目仍属于接入与优化方案。公共接口与设备实现需要分开设计，不能从 CUDA 先落地推断它们是 CUDA 专属功能，也不能据接口存在宣称 Metal 已支持。

MTP 与 DFlash2 各自包含“模型结构”和“推测策略”两部分。模型 provider 描述权重绑定、拓扑、mask、feature taps 与 selector 数学；算法 provider 描述候选预算、依赖阶段、采样和提交规则、追赶需求及组合策略。Device backend 执行绑定后的计算和状态操作。

| 技术 | 公共层内容 | Backend 专属实现 |
|---|---|---|
| MTP | 内置 head 的模型绑定与图；短链提议计划、hidden 条件与位置映射、接受和追赶语义 | head forward、设备候选与采样、verify batching、KV 更新、graph 与融合 |
| DFlash2 | 独立草稿 package 与图；block mask、target taps、selector 定义、候选与特征生命周期 | 特征捕获、动态卷积、非因果 attention、selector/top-k、设备 buffer 与验证执行 |
| ReplaySSM | 目标状态更新原语、接受前缀回滚契约、能力和资源描述 | 算子专属 record layout、Record/Fold、history gather、workspace 和设备同步 |

沿用现有 crate 分组，拟议目录和依赖如下；这些目录尚未创建：

- `crates/foundation/ir`：计划、候选布局、状态提交、能力和资源描述。它承载数据与验证，不实现算法 provider 或 GPU 调用。
- `crates/foundation/spi`：`SpeculationProvider` 与 backend 的边界契约，不放内置 MTP/DFlash2 的具体策略。
- `crates/engine/workloads/src/speculation/`：内置 `mtp`、`dflash2`、`composition` provider、注册与共有策略；现有 `speculative.rs` 接受规则作为迁移基础。
- `crates/model/package` 与 `crates/model/recipes`：target/MTP/DFlash2 的配置、权重映射和声明式模型图。
- `crates/model/compiler` 与 backend 的冷路径编译：将声明式模型、算法计划与具体能力绑定，产出可执行 program 和完整报价。
- `crates/backend/cuda`、`crates/backend/metal`：各自实现算子、设备执行、状态与同步；ReplaySSM 的物理实现分别归属各 backend。
- `crates/engine/runtime`、`crates/engine/state`、`crates/engine/scheduler`：使用 provider 计划和 backend receipt 协调生命周期、选择与预算。

公共算法实现仅依赖共有 IR/SPI 和模型描述，不反向依赖 `infer-backend-cuda`、`infer-backend-metal`、cuTile 或 Metal API。设备要求按算子、dtype、状态布局和实际精度表达；backend-specific lowering 可做特化，但公共算法能力选择不以 `backend == cuda` 代替能力匹配。

把策略迁出 CUDA 不要求把数值热路径搬到 CPU。公共 provider 在冷路径输出声明式阶段、依赖和状态需求，backend 将其编译为常驻计算、设备判定和提交路径。Backend 可以包含特定算子的专门实现；新 provider 若仅复用已有算子与计划语义，应通过绑定即可接入。需要新增动态卷积或 selector 等语义时，先扩展模型/执行原语，再由 backend 实现，避免通用 executor 按算法字符串分支。

Metal 可以独立实现 MTP、DFlash2，或先采用 snapshot 而后增加 recurrent replay。可用性取决于绑定 program 所需的算子、精度、验证与状态能力，以及完整资源预算；缺失能力时显式报告或按 `auto` 策略回退。首次实现优先 CUDA，不构成公共契约中的厂商限制。

ReplaySSM 的具体分层、两阶段范围和持久状态边界由 [Recurrent 状态重放方案](state-replay.md) 维护。算法 provider 把接受前缀交给共同 target 状态协议，由 backend 选择物理策略；不把 replay 变成第三个草稿 provider。

## Provider 与计划契约

拟议 provider 在组合根显式注册，以算法 ID 解析。MTP 和 DFlash2 可以同时注册；实际启用仍须通过模型兼容性、采样语义、设备算子和资源预算检查。算法 provider 组合模型配方，不能接管模型 provider 的拓扑职责。

| 拟议契约 | 关键内容 |
|---|---|
| `MtpHeadBinding` | 将现有模型侧 `SpeculationPlan` 明确命名为内置 head 绑定，保留迁移适配 |
| `DraftModelBinding` | 独立 package、模型版本、共享 embedding/head 的兼容声明及精度策略 |
| `SpeculationDescriptor` | 算法 ID、线性或树形候选布局、可用采样模式、特征需求、状态追赶方式 |
| `TargetFeatureRequirement` | layer ID、抽头位置、归一化状态、dtype、位置映射、历史保留范围 |
| `SpeculationPlan` | 已绑定的 proposal/verify program、宽度、状态 recipe、预算、接受策略与图签名 |
| `ProposalBatch` | backend buffer handle、每请求候选数量、位置、候选来源及采样所需概率信息 |
| `VerifyDecision` | 接受数量、目标状态消费数量、停止原因、后续采样所需结果与特征覆盖范围 |
| `CommitReceipt` | 请求和事务身份、state/config epoch、实际物理游标及资源归属 |

`ProposalBatch` 使用设备或共享 buffer 的不透明句柄，不强制转换成 `Vec<u32>`、完整 logits 或 hidden 的主机数组。句柄必须声明设备 owner、完成 fence、读者生命周期和有界容量。

接受策略由字符串改为明确的类型及 capability。第一版沿用现有 greedy 能力；随机采样仅在 provider 能提供实际有效 proposal 分布时开放。采样契约统一 repetition penalty、词表过滤、grammar、EOS、tie break 和随机数索引；没有实现相同语义的设备路径显式拒绝或回退。

模型 package 解析、权重绑定、资源报价、规划与图录制发生在冷路径。热路径按逻辑执行 `propose → verify → decide → settle`，这些阶段可以在 backend 内融合，不要求阶段之间回到 CPU。提交与完成继续复用 `BackendProvider::submit/poll` 和现有 ticket 协议。

## 已实施：执行计划的接受策略不再是字符串（I1 步骤②第一刀）

执行 IR 的 `SpeculationPlan` 里，`acceptance_policy` 是一个 **`String`** ✗ —— 方案要求的正是"结构化契约" ✓
（第 19 行：`infer-ir` 承载"算法标识、特征需求、候选布局、**采样契约**、提交结果与资源描述" ✓）。字符串既
不能表达"验证器能接受哪种采样"，也**把拼写错误当成合法策略** ✗，而这份文件里其余字段都是类型的 ✓。

现在它是 `AcceptancePolicy` 枚举 ✓：`Greedy`（只有与验证器 argmax 一致才接受 ✓）与 `Residual`（分歧从残差
重采样 ✓ —— 正是 capability 里 `greedy_only` 那条注释所依赖的区分 ✓），serde 以 `snake_case` 记录 ✓，
`Default = Greedy` ✓。三个用例（在 `crates/foundation/ir/src/workload/tests.rs` ✓）
覆盖：按名字往返 ✓、**未知策略被拒绝而不是被带下去** ✓、默认值 ✓。

**这一步的边界要说清楚**：执行 IR 的 `SpeculationPlan` 目前**没有生产者** ✗（方案已把它列为步骤③"provider
迁移"的内容 ✓），所以本次是**把既有契约收紧** ✓，而不是新增无人使用的脚手架 ✓；生产端接入时不必再改类型 ✓。

## 验证与提交的共同协议

目标模型的验证结果定义权威接受前缀。backend 提交物理状态并返回 receipt；runtime 校验身份后，只提交一次逻辑结果，再交付输出。provider 不能自行把未验证的草稿写入公共前缀。

以线性候选链为例：target 已物化前缀长度为 `N`，下一次输入是已经确定的 seed token `t0`；provider 再提议 `m` 个 token。验证输入为 `[t0, c1, …, cm]`，宽度为 `m + 1`。

若接受 `k` 个候选，target 物理游标推进到 `N + 1 + k`。后续校正 token 或全接受后的 bonus token 可从最后有效 target row 采样，但尚未进入 target KV/recurrent state，作为下一轮 seed。遇到 EOS 或输出上限时，不再生成多余 token。保留当前“backend 返回接受候选和尾部 logits，由 runtime 完成后续采样”的适配路径，再单独引入设备采样。

因此需要分别记录以下位置，不能用一个 `position` 代替全部语义：

- 客户端已发布的逻辑输出前缀。
- target 已物化的输入前缀和待消费 seed。
- 每个 provider 的 draft 状态覆盖位置。
- 每种 target feature 的有效位置范围。

MTP 当前存在一 token 的状态错位关系，应由 provider 的状态 recipe 声明。DFlash2 的特征历史、block 和 draft KV 使用自身位置映射，公共 coordinator 不写死 `MTP_KV_OFFSET`。

## 回滚与草稿追赶

| 阶段 | 状态约束 |
|---|---|
| 开始 | 保存事务基点；校验 target、draft、feature 的 epoch 与覆盖位置 |
| 提议 | 只推进 provider 私有的暂存状态；候选不进入公共前缀 |
| 验证 | target 可暂时消费整条链；未接受尾部保持不可见 |
| 决定 | 生成接受前缀、停止信息与物理消费数量；不直接发布客户端输出 |
| 提交 | 保留接受前缀，截断或恢复其余状态；返回可校验的 receipt |
| 追赶 | 用已验证 target 特征修复活跃 draft；未活跃 provider 可标记失效并延迟重建 |

full attention KV 可通过逻辑长度与页所有权截断，但 recurrent state 必须恢复到准确的消费位置。物理实现可以选择逐 lane 快照、基点加接受前缀重放或稀疏快照；统一协议只约束最终状态与释放时机。

ReplaySSM 的分层与通用性由 [Recurrent 状态重放方案](state-replay.md) 维护：公共 IR/SPI 表达接受前缀、事务、能力与预算；backend 编译并执行状态算子专属的 Record/Fold。MTP、DFlash2 共享 target 回滚能力，provider 不持有具体重放实现。

全接受也不能默认免除 draft 追赶。现有 MTP 提议期间用了 draft hidden，下一轮需要恢复为 target-conditioned 的状态。DFlash2 则按其参考实现维护 feature 和 draft KV，不继承 MTP 的追赶规则。

拒绝尾部的 target taps、草稿缓存和页引用必须同步失效。发生取消、超时或 GPU 错误时，沿用现有 owner/fence 协议：可证明恢复到基点才允许复用；无法恢复的请求状态标记 poisoned 并在完成 fence 后释放。超时不能被解释为设备完成。

## Prefix 与模式切换

target prefix 的缓存身份包括 target 权重、精度、位置编码和 state recipe。provider 的附属缓存另带算法、draft 权重、feature recipe 与版本身份。缺少可复用的附属缓存时，从 target 前缀重建所需特征或显式缩小复用范围，不能把只有 target KV 的命中视为 draft 已初始化。

切换 provider 时，优先保留 target 状态，校验新 provider 的特征覆盖和预算；未追赶的 draft 在重新激活前重建。不得为每一轮强制维护全部 provider，也不得省略重建成本或无限保留多个 provider 的状态。

## MTP 迁移与验收

第一步拆开模型绑定与执行计划，提供旧参数和旧声明的适配。`num_speculative_tokens` 继续表示 seed 之外的候选数量，`mtp_depth` 留在 MTP provider 内。verification width 来自实际候选布局，scheduler 使用 backend 报告的有效宽度和预算。

第二步把 serial 与 pooled 路径的接受前缀计算接到共同契约，保留两者设备实现及当前数值路径。第三步统一 commit receipt、draft 追赶与 prefix 附属状态。扩展能力用只声明计划的局部 provider 样例验证，确认注册无需增加通用执行器的算法条件分支；样例不加载权重、不执行模型或维护真实设备状态。独立 DFlash2 的实际接入归 D2，I1 不等待第二个真实模型完成。

验收覆盖零候选、全接受、首 token 拒绝、中间拒绝、EOS、输出额度不足，以及 serial/pooled 切换、prefix 恢复、取消、超时、重复完成与 stale epoch。每个用例都检查输出和 target/draft/feature 的位置，而不只检查接受数量。

现有 [verify_draft](../../../crates/engine/workloads/src/speculative.rs) 可作为随机接受规则的 CPU 正确性参考；它目前不能直接证明 CUDA 或 DFlash2 已支持随机推测解码。第一阶段以现有 greedy MTP 的 token、状态和服务性能不回退为迁移标准；性能判断绑定按共同契约验收的 native MTP baseline ID，后续设备优化独立验收。
