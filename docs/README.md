# 文档导航

文档说明当前接口、操作方法与能力边界。实现细节以源码为准，命令参数以 `--help` 为准。

## 后续工作

先读 [工程整理与推理优化路线图](plans/README.md)。该页统一维护 P0–P3 优先级、任务编号、启动条件、并行范围与完成标准；按任务进入详细方案，不必逐份阅读历史研究。工程基础、Agent 服务、推测解码与 CUDA 性能方案归档在 `plans/` 的对应分组。

Agent 接入、tool parser、文本流式、max length 与严格工具约束的实现边界和后续设计见 [Agent 工具调用与服务配置方案](plans/serving/agent-api.md)。该方案按实际使用场景取舍，借鉴 vLLM/SGLang 的机制；当前接口范围仍以使用文档与源码为准。

准备测试或性能对比时，先读 [性能基线与 vLLM 对齐方案](plans/performance/baseline.md)。release、当前配置对齐、矩阵、计时统计和证据规则只在该文维护；模块方案补充自己的接口与数值/资源验收。基线验收完成前，历史矩阵只用于调查线索。

## 使用与开发

| 文档 | 内容 |
|---|---|
| [开发与运行](guides/development.md) | 构建、CLI、HTTP/SSE、checkpoint 与测量 |
| [构建与打包](guides/packaging.md) | Zig 平台矩阵、归档、校验与 GitHub CI |
| [模型执行](guides/model-execution.md) | 模型包、权重、prefill/decode |
| [新增模型](guides/adding-a-model.md) | ModelProvider、执行图与注册 |
| [图像模型](guides/vision.md) | 视觉塔、MRoPE、独立参考与支持限制 |
| [OpenAI 兼容接口](guides/openai-api.md) | 路由、参数与错误约定 |
| [Linux 放置](guides/linux.md) | cpuset、NUMA 与硬件验收 |

## 测试与测量

| 文档 | 内容 |
|---|---|
| [MoE 单卡验证](guides/moe-validation.md) | 5090 缩小模型、独立参考与当前支持范围 |
| [质量门禁](guides/quality-gates.md) | 本地/CI 检查、Attention 对照与验收要求 |
| [CPU 性能测量](guides/cpu-performance.md) | 分配计数、延迟与测量范围 |
| [CUDA 性能测量](guides/cuda-performance.md) | 现有算子/模型/服务工具、能力探针与测量边界 |
| [新硬件与新模型调优](guides/tuning-new-hardware-and-models.md) | 加载期自动调优、检查表、诊断工具与待改为推导的常数 |
| [性能基线与 vLLM 对齐方案](plans/performance/baseline.md) | 统一测试条件、比较与验收契约；已签发基线见文内 |
| [Serving 稳定性](guides/serving-stability.md) | 故障半径、排队与过载行为、生产差距 |
| [工具导航](../tools/README.md) | 门禁、参考导出与基准入口 |

## 架构

| 文档 | 内容 |
|---|---|
| [代码布局](architecture/layout.md) | 职责、依赖方向、模型与 kernel 边界 |
| [Runtime](architecture/runtime.md) | 执行流水线、故障与观测 |
| [CPU 资源协议](architecture/cpu-runtime.md) | owner、队列、内存、fence 与回收 |
| [调度](architecture/scheduling.md) | 就绪队列、准入与成本模型 |
| [请求生命周期](architecture/request-lifecycle.md) | 状态提交、取消与恢复 |
| [KV Manager](architecture/kv-manager.md) | 页所有权、StateRecipe、COW 与 prefix |
| [Backend](architecture/backends.md) | 设备能力、选择、扩展与支持限制 |
| [Agent](architecture/agent.md) | JSON-RPC 诊断与配置实验 |
| [架构决策](adr/README.md) | 已采纳决策及其理由 |

## 审查与研究

| 文档 | 内容 |
|---|---|
| [前缀指纹优化](research/prefix-fingerprint-2026-10-24.md) | SHA-256 的必要性、增量内存指纹、碰撞校验与 45 配对实测 |
| [容器性能评估](research/container-performance-2026-10-24.md) | 45 实测：bbtree、哈希索引、固定容量与其他容器的保留依据 |
| [近期 CUDA 提交审查](reviews/cuda-commits-2026-10-09.md) | 2026-10-09 基点的源码缺陷、优化判断与验证范围 |
| [vLLM 对比测试审查](reviews/vllm-benchmark-methodology-2026-10-09.md) | 配置、计时、数值与统计缺口；历史差距的可信范围 |
| [Infernix 性能调研](research/infernix-performance.md) | 参考机制、精度变化与收益估算 |
| [CUDA 长上下文性能结论与任务交接](research/cuda-vllm-performance-handoff-2026-10-17.md) | 已合并 PR #10、五组正式收益、当前 attention 瓶颈、失败方案、质量边界与可拆任务 |
| [CUDA kernel 工程整理与 45 重建记录](research/cuda-kernel-engineering-2026-10-23.md) | AOT 制品清单、统一校验/重建入口与固定工具链 |
| [Backend 标准化与接入 SOP 研究](research/backend-standardization-2026-10-25.md) | CUDA/Metal 接入面盘点、Tier 分层、可共享契约与分阶段验收 |

过期的服务矩阵、10-13 性能画像和测试迁移清单已删除；2026-10-09 至 10-14 的完整 A/B 与逐轮调查记录移入[历史归档](archive/README.md)，保留源码基点与测量条件供复核，不代表当前状态，也不进入日常排期。

## 维护约定

- 已实现的使用方法写入 `guides/`，跨模块契约写入 `architecture/`，已采纳决策写入 `adr/`。
- 待实施工作归入 `plans/`；任务状态、依赖和全局顺序只在路线图维护，详细方案维护本项设计与验收。性能与测试的共同条件由基线方案维护；同一规则由一个主文档维护，其余链接引用。
- 审查与研究分别归入 `reviews/`、`research/`，保留源码基点与实验条件。已完成工作更新当前使用/架构文档并更新路线图，历史实验不再承担当前排期。
- 结论被取代或不再承担当前排期的一次性实验记录移入 `archive/`，归档只改变位置、保留可追溯路径；当前导航只列仍在使用的文档。
- 运行暂存、临时报告和 profiler 输出放入 `artifacts/`；已验收基线的 manifest、输入、原始样本和可复算汇总持久保存，不能只有 `artifacts/` 副本。具体留存与更新按基线方案执行；现有 `benchmarks/` 微基准保持各自测量范围。
- 删除或移动文档时修复链接，并运行 `python3 tools/check/layout.py`。
