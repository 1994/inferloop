# 工程整理与推理优化路线图

状态：部分完成，逐行维护。初始源码基点为 `7c79d99`，整理日期为 2026 年 10 月 9 日；状态列最后复核于 2026-10-25。本页统一维护任务优先级、前置条件、并行边界和完成状态；详细方案维护接口与本项验收。排期只表示资源争用时的先后，完成状态以状态列注明的证据为准。

性能优先的执行重叠与 AOT 决策见 [ADR-0004](../adr/0004-performance-first.md)。

B1/B2 已完成，B3 的复测样本与持续/混合负载仍待补；工程整理 E1/E2 已完成，Agent 长度与基础工具闭环 S1/S2 已交付。当前主线是 C1 的长上下文 attention 与资源画像；推测解码按 I1、R1、D1 的前置逐项推进。vLLM/SGLang 作为机制参考，服务参数按实际使用场景取舍。

**准确 baseline 是性能工作的硬前置条件。** release、实际配置对齐、计时统计和证据规则由 [性能基线方案](performance/baseline.md) 维护。三条基线已签发，但对应 profile 的完整性能判定尚未通过，历史倍率不能确认新收益。修复、采集、正确性参考和工程迁移可以先做；新的性能 A/B 与默认性能策略切换必须引用对应配置已验收的 baseline ID。

## 优先级与任务清单

P0 优先消除正确性和测量阻塞；P1 交付工程、服务与公共协议基础；P2 接入主要优化和第二种 provider；P3 在单项稳定后验证组合或按需扩展。优先级决定资源争用时谁先做，不代表所有 P1 都要等 P0 全部结束。

任务编号用于交接与查找。审查文档里的 P1/P2 是缺陷严重程度，与这里的排期优先级分别维护。

| 优先级与编号 | 任务与方案 | 启动条件 | 完成标准 | 状态 |
|---|---|---|---| --- |
| P0 · B1 | [CUDA 三项修正](../reviews/cuda-commits-2026-10-09.md) | 可立即启动 | handle 设备/stream 归属、FP8 缓存身份、capture 前预热通过对应设备回归；保留数值规则 | **已完成**：`46ed038`（2026-10-10）修正 handle 设备/stream 归属、FP8 workspace 身份与 capture 前预热并进入 main，数值规则保留 |
| P0 · B2 | [采集与比较门禁修正](performance/baseline.md) | 可立即启动，与 B1 分开开发 | release 身份、配置回读、工作量、缓存、计时与统计可核验；审查中的无效对比不能误判通过 | **已完成**：release 身份、配置回读、工作量、缓存、计时与统计均有门禁；并加上 profile 版本边界，旧基线仍可复核 |
| P0 · B3 | [重建并固化 baseline](performance/baseline.md) | B1、B2 的对应修正验收；固定可复现的 release 制品 | 按共同契约完成对应 profile 的完整矩阵、独立复测与复算，保存证据并签发 baseline ID | **部分**：三条基线已签发并可复核；独立复测样本（2→5）与持续/混合负载需要设备窗口 |
| P1 · E1 | [测试归属](engineering/tests.md)与[删除测试后端](engineering/remove-test-backends.md) | 可立即启动，先登记消费者和断言清单 | 按 crate 迁移测试并替换消费者；删除两个完整 CPU 测试执行器及产品 feature/CLI/IR 分叉；关键断言有去向 | **已完成**：两个执行器与产品 feature/CLI/IR 分叉已删除，见[验收对照](engineering/remove-test-backends.md) |
| P1 · E2 | [构建与依赖收敛](engineering/build-and-dependencies.md) | 盘点、计划设计可立即启动；受影响消费者/feature 随 E1 收敛后集成 | 主 workspace 版本来源统一，native/cross 共用计划，host/device/model/performance 入口与 CI 职责清楚；不顺带升级依赖 | **已完成**：盘点、计划、依赖图、计数、参考环境、入口动词与 workspace 边界均有门禁或实测证据 |
| P1 · S1 | [总上下文与有效服务配置](serving/agent-api.md) | CPU 规则可立即启动；设备容量验收使用受控窗口 | 所有入口共享长度预算，输出 cap 与总上下文区分，实际配置与来源可读回；只增加实际需要的参数 | **已完成**：`--max-model-len`、输出与序列 cap 使用共同解析路径，生效配置可读回；设备容量验收按 C1M/基线窗口另行执行 |
| P1 · S2 | [基础 Agent 工具闭环](serving/agent-api.md) | parser、模板与协议可立即启动；完整接入依赖 S1 和相关测试归属约定 | typed 工具历史、结果关联、none/auto、文本 SSE 与增量 parser 通过 SDK 闭环；流式/完整返回一致 | **已完成**：JSON/function 两种方言、参数 deltas、截断合同与 SDK 工具往返通过；`required`/指定 function/`strict` 已交付，真实模型与 GPU 严格工具验收未做 |
| P1 · I1 | [公共 SPI 与 MTP 迁移](speculation/spi.md) | 契约设计可先做；迁移验收需要 B3、受影响 E1 迁移及 E2 的构建/feature 合同稳定 | 模型绑定、算法 provider、设备执行分开；serial/pooled 的 token、状态、资源和服务性能不回退 | **部分**：步骤①（公共层算法名清零）与步骤②第一刀（接受策略成类型）已完成；其余契约与 provider 迁移待做 |
| P1 · D1 | [DFlash2 package、模型模块与独立参考](speculation/dflash2.md) | 配置/权重检查、参考制品和资源估算可先做；独立设备模块另排窗口，共同接线等 I1 契约稳定 | 架构/词表/层抽头/窗口兼容性清楚，BF16/完整 head 的 draft 图与 selector 对照独立参考；完整资源报价可解释 | **部分**：配置/权重/几何/融合/卷积/selector/token/mask/报价已完成；arena 与图报价、tokenizer 逐 token 对照缺制品或设备 |
| P2 · R1 | [Target Record/Fold 状态重放](speculation/state-replay.md) | I1 的共同 verifier/提交契约稳定；迁移后 MTP 参考已验收 | Delta/Conv 接受前缀状态直接对照通过，serial/pooled 共用；容量、Fold 成本和完整周期有证据 | **部分**：两组 golden 与 Rust 参考已完成；设备对照与 Fold 成本需要窗口 |
| P2 · D2 | [DFlash2 接入](speculation/dflash2.md)与[请求/轮次选择](speculation/composition.md) | D1、I1，以及目标设备可兑现的回滚和草稿资源能力；本轮小显存接入需 R1 与草稿预算一起解决 | 线性 greedy、拒绝后状态、池化与切换通过；实际启用算法、追赶成本和回退可见；两个 provider 分别有有效对照 | 未开始：前置 D1、I1 未满足 |
| P2 · C1 | [按 profile 选择 CUDA/服务优化](performance/cuda.md) | 对应 B3 baseline 与阶段计时可用；涉及草稿时还需该 provider 已验收 | 每次只改变一个可归因策略，完整服务矩阵有收益，尾延迟/显存/数值通过；可早于 D2 做普通 decode 优化 | **部分**：精确 GPU token readout 与[pooled 图几何准入](../archive/research/cuda-pooled-admission-geometry-2026-10-12.md)各通过五组 native 服务配对；后者批量完成时间下降 37.8%，保留 TTFT/TPOT 取舍；最新 vLLM 跨引擎矩阵及后续 profile 优化待验收；2026 年 10 月 17 日：PR #10 已合并，vector 五组 native 20 项门通过；当前主线转为 attention，原 vLLM 质量和稳定长矩阵未验收，见下方 C1 子任务 |
| P3 · X1 | [MTP 与 DFlash2 同轮组合](speculation/composition.md) | D2 的单 provider 和切换路径稳定，组合模型条件及资源报价明确 | 优于同负载下最佳单 provider，使用相同回滚策略；不能只与关闭推测比较 | 未开始：前置为 D2 单 provider 验收 |

B3 不等待全部 E2 或完整 Agent 能力完成。先用经过修正的固定制品和当前实际配置建立基线；S1 或后续策略改变共同约束时，按基线合同创建新 profile 和交接结果。I1 不必等待整套 CI、打包和不相关 crate 的目录整理，但其涉及的测试、feature、构建合同必须已经稳定。共享 IR 的测试 feature 删除需要相关消费者一并迁移，不能只在某个局部构建中隐藏旧分叉。

R1 与草稿量化解决不同的显存开销。D1 先建立 BF16/完整 head 参考，再按目标设备报价决定是否需要草稿量化或 shortlist；不能把这两项排到 D2 之后才发现无法准入，也不能因此改变 target 精度。大显存设备上的 snapshot 正确性参考，不代替目标设备的容量与性能验收。

## CUDA 长上下文任务交接

截至 2026 年 10 月 17 日，PR #10 已合并；vector 候选 32K TTFT 五组配对降低 8.41%，四请求整组吞吐提高 7.97%，相对父版本的 20 项门通过。该结果不代表新的跨引擎基线通过。已验证事实、失败方案、证据路径和详细验收见 [CUDA 长上下文性能结论与任务交接](../research/cuda-vllm-performance-handoff-2026-10-17.md)。

| 编号 | 可独立执行的任务 | 启动条件与交接 | 当前状态 |
|---|---|---|---|
| C1Q | 模式一致性与发布质量 | 原题目、scorer、预算及失败保持；定位通用数值或状态差异，按原合同验收冻结二进制 | 原严格门失败；同模式 256 题仅完成归因，最新 vector 尚无完整 1,024 题验收 |
| C1A | Target attention 完整成本 | 新 profile/NCU/真实输入已具备；量可削减成本后，一个候选依次数值、算子、自动 tuning、服务、质量验收 | 当前主线；attention 占 71.8%；单独 K 写入冲突候选原门否决，不重测 |
| C1M | 真实显存与多会话驻留 | 先编译运行已准备的 LoadedModel 分项采集，再决定 arena、工作区或页级复用 | 当前实际分项尚未采集；旧 allocation、槽数不能承诺 32K 驻留数 |
| C1V | 稳定 vLLM 长矩阵 | 固定制品与实际配置，解释旧 C4 波动后冻结新匹配矩阵，保留全部样本与独立复算 | 约 6.04 倍只是两轮 32K C1 诊断数据相除，稳定 C4 跨引擎倍率未验收 |
| C1H | H/B 实机验收 | 设备到位后逐架构验 ABI、数值、资源、自动选择/回退及完整服务 | 六个 AOT 离线变体已核验；用户当前无 H/B 机器，性能待实机 |
| C1D | 分支验收与合并 | 对应质量、服务、CI 和持久证据齐备，形成可 review PR 并处理反馈 | PR #10 已完成；后续分支未合入，原 vLLM 质量失败保持未完成 |

实施方案见 [vLLM 长上下文性能追平实施方案](performance/vllm-parity.md)。

C1Q、C1A、C1M 的 CPU 审计可独立拆解；同一设备的构建与 GPU 测量使用原锁串行。C1A/C1M 局部结果不代替 C1Q 或 C1V，C1D 不继承 PR #10 的个别合并批准。已落地的量化复用、设备 greedy、旧池回收不重复立项，也不依据旧 GDN 画像覆盖当前测量顺序。

## 当前可以同时推进的工作

建议按测量、服务与协议三条线交接。下面只列尚未完成、可实际启动的范围；B1/B2、E1/E2、S1/S2 已完成，不再重新排期。

| 并行线 | 当前可执行范围 | 暂时不做 | 交接点 |
|---|---|---|---|
| 测量与正确性 | B3 的独立复测与持续/混合负载；C1A target attention 成本；C1M 实际显存分项 | 不用历史倍率代替新 profile；未验收前不切换默认性能策略 | 冻结制品与 profile、完整样本、独立复算 |
| 工程整理 | E1/E2 已交付；只维护门禁和新消费者接入 | 不再做全仓目录或依赖大搬家 | 新测试与构建入口沿用既有合同 |
| Agent 服务 | S1/S2 已交付；按需做真实模型与 GPU 严格工具验收 | 不扩大 vLLM 参数清单；不支持的 schema 保持显式错误 | 设备验收与推测组合按 I1 合同接入 |
| 协议与模型准备 | I1 公共契约、R1 状态重放、D1 draft 参考与完整资源报价 | 不在共同协议未定时各写一套 verifier | target taps、状态 receipt、候选布局由共同契约定义 |

S1/S2 已交付；新测试继续遵守 E1 的归属约定。严格工具的设备验收和推测组合在 I1 合同稳定后接入，不另建测试 backend。

## 前置完成后还能并行哪些工作

| 前置状态 | 可以并行的工作 | 必须串行的交接 |
|---|---|---|
| 对应 profile 的 B3 已验收 | I1 的公共接口迁移、C1 单项普通 decode 优化 | 修改同一 feature/manifest/请求提交接口时先固定合同；正式 GPU 测量分别排窗口 |
| I1 verifier/提交契约稳定 | R1 的 backend Record/Fold 与 D1 的 draft 图、selector 独立数值实现 | target taps、状态 receipt、候选布局由共同契约定义；D2 最后合流 |
| R1 与 D1 正确性/预算具备 | D2 的 provider 接入，独立 C1 候选 | D2 集成先冻结一版 verifier/资源计划；各候选使用独立制品和单项 A/B |
| D2 单 provider 已验收 | X1 组合算法研究，按真实瓶颈继续 C1；按需扩展生成约束 | 组合与单 provider 分别验收，不能在同次 A/B 混入多个改变 |

D1 的 host 检查和独立参考可以提前。依赖 I1 的设备接线、共同提交和服务性能不能因此提前宣布完成。R1/D1 可以同时开发不同的设备模块，并行开发不意味着在同一 GPU 同时执行验收。

## 排期规则：必须保持的串行顺序

以下顺序包含已完成阶段，保留作为交接规则；已完成阶段不再重复执行，但后续改动仍受其合同约束。

- 测量链：B1 与 B2 修正通过 → 固定制品/配置 → B3 独立复测与固化 → 对应 profile 的 C1 或算法性能 A/B。
- 工程链：清单与归属 → 消费者替换及测试迁移 → 删除执行器/feature 分叉 → 收敛受影响依赖/构建 → 最后替换 CI 重复入口。可按 crate 分批，不做一次全仓大搬家。
- 算法链：受影响工程合同与 B3 可用 → I1 MTP 迁移 → R1 和 D1 的设备部分分别验收 → D2 集成与选择 → X1 同轮组合。
- Agent 链：模型方言/消息/长度合同 → 增量输出与工具结果回传接入 → SDK 闭环 → 有需求时接严格约束。基础闭环不等待算法链完成。

CPU 协议或数学参考的准备可以早于这些验收点；实际设备接入、默认切换和性能结论遵守对应前置，不能用“已并行开发”代替“已验收”。

## 并行时的冲突与资源边界

| 共享边界 | 协作规则 |
|---|---|
| workspace、lockfile、features、构建驱动 | E2 已统一收口；后续修改公共构建合同先固定 feature/依赖边界，不同时升级依赖 |
| 同一 crate 的测试目录与挂载 | E1 已完成路径、收集与 assertion 映射；功能修改不再搬动测试目录，避免重写断言 |
| ModelIr、SPI、verifier、receipt、target taps | I1 先固定公共语义；R1/D1 按模块边界实现，D2 集成时只保留一个提交/回滚入口 |
| frontdoor、长度 resolver、输出事件 | S1/S2 已固定长度与输出事件合同；严格工具与推测组合沿用同一入口，不另起分叉 |
| release 制品与实验配置 | 每条线绑定明确源码、二进制和配置身份；B3/C1 使用冻结证据包，不测会被另一条线覆盖的构建产物 |
| GPU、CPU、内存与 I/O | 同一 GPU 的模型/设备/性能运行排队。正式服务测量独占实验约定的整机资源窗口，不同时跑重编译、host suite、参考导出或模型搬运；其他机器可继续开发 |

增加人手可以并行准备代码和参考，不能使共享 GPU 的正式测量并行。正确性验证与性能样本分别留存；新的共同配置、算法或精度需要新 profile，旧证据不能混入新结果。

## 后续储备与启动条件

这些能力保留设计，只有出现实际场景或测量证据时再排期，不作为 I1、R1 或 D2 的默认阻塞项。

| 储备 | 启动条件 |
|---|---|
| 严格工具/schema 扩展（`$ref`、`$defs`、pattern、数值/长度边界等） | 调用方实际需要，或格式失败造成实测重试；先验收非推测约束，再依共同 SPI 接推测状态回滚 |
| 树形验证、copy proposal、可变 DFlash2 block | 线性 verifier 稳定，相关模型条件、布局和状态能力齐备；完整asd成本可能优于已有单 provider |
| 随机推测与随机组合 | 能导出实际条件 q，接受/残差采样及分布/状态验收完整 |
| 普通 decode output-only 与延迟 flush | R1 第一阶段稳定，数值变化、跨轮记录和 checkpoint 契约清楚 |
| Target 重新量化、低精度 attention、改变数值规则的融合 | 独立制品与按任务/context 的质量对照可用；与保留 target 精度的优化分别报告 |
| PDL、新设备或更大 prefill 策略 | 工具链/能力具备，profile 支持选择；不能用 kernel/launch 局部收益代替完整服务收益 |

服务参数只选择实际 SDK、部署和测量需要的部分；历史别名、特殊默认模式和无调用方的开关不进入任务队列。同轮组合 X1 已有明确后续位置，其余储备不会因参考框架支持就自动升级为必须交付。

## 按问题查详细方案

| 任务 | 文档 | 主职责 |
|---|---|---|
| E2 | [构建与依赖](engineering/build-and-dependencies.md) | 命令、目标合同、workspace、依赖与 CI |
| E1 | [测试组织](engineering/tests.md) | 目录、私有性、收集与设备/性能入口 |
| E1 | [删除测试后端](engineering/remove-test-backends.md) | 消费者替代、删除范围与断言迁移 |
| B2、B3 | [基线与 vLLM 对齐](performance/baseline.md) | release、真实配置、矩阵、计时统计、证据和基线更新 |
| S1、S2；按需约束 | [Agent 工具调用与服务配置](serving/agent-api.md) | 工具闭环、增量 parser、按需生成约束、总上下文与实用配置 |
| I1 | [SPI 与 MTP 迁移](speculation/spi.md) | 公共 provider、模型/算法/backend 分层与提交协议 |
| R1 | [Recurrent 状态重放](speculation/state-replay.md) | 通用回滚契约、backend Record/Fold 和状态生命周期 |
| D1、D2 | [DFlash2 接入](speculation/dflash2.md) | 模型、特征、selector、block 与独立接入验收 |
| D2、X1 | [选择与组合](speculation/composition.md) | 分请求/轮次选择和待验证的同轮组合 |
| B1、C1 | [CUDA 执行优化](performance/cuda.md) | 计时、完整预算、设备驻留、算子与服务验收 |

## 审查与实验依据

| 资料 | 用途 |
|---|---|
| [近期 CUDA 提交审查](../reviews/cuda-commits-2026-10-09.md) | 三项源码缺陷、近期优化判断及验证范围 |
| [vLLM 对比测试审查](../reviews/vllm-benchmark-methodology-2026-10-09.md) | 对比方法缺口、可复现探针与“当前差距”的证据条件 |
| [Infernix 调研](../research/infernix-performance.md) | 参考机制、模型精度变化及本项目容量/收益估算 |
| [CUDA 性能实验记录](../archive/research/cuda-performance-experiments.md) | 历史 A/B、失败实验和已落地改动；其中早期待办以本路线图为准 |

日常排期只看本页的任务表与并行边界，再进入对应方案。审查用于定位缺陷，研究与历史实验用于核对机制和证据，不再承担当前待办清单。任务状态、跨模块依赖与全局顺序只在本页维护；每项交付通过验收后更新当前架构/使用文档，并更新这里的完成状态。
