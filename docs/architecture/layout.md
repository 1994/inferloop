# 代码布局与调用链

目录按职责分组。`lib.rs` 只做模块声明与显式 API 导出，实现放在职责模块中；测试位于所属 crate 的 `tests/` 或 `src/**/tests.rs`。

## Crate 职责

| 目录 | Crate | 职责 |
|---|---|---|
| `crates/foundation/core` | `infer-core` | ID、错误、生命周期、arena、credit、事件与有界容器 |
| `crates/foundation/ir` | `infer-ir` | 请求、模型、执行、设备能力、计划与共享输出 |
| `crates/foundation/spi` | `infer-spi` | backend、state、调度、模型、workload、资源与扩展接口 |
| `crates/backend/api` | `infer-gpu-api` | 设备描述、传输与批次槽协议 |
| `crates/backend/kernel-api` | `infer-kernel-api` | 通用算子描述（含 attention）、kernel 注册与能力选择 |
| `crates/backend/metal` | `infer-backend-metal` | Metal 设备、加载、forward、提交、完成与物理状态 |
| `crates/backend/cuda` | `infer-backend-cuda` | 原生 Rust/cuTile 算子、设备能力选择、融合与常驻图执行；非默认成员。分层与改动入口见 [CUDA 后端分层](cuda-layers.md) |
| `crates/model/package` | `infer-models` | HF 包、索引、Safetensors、内存预检、tokenizer 与模板；模型 provider 注册表（按架构解析） |
| `crates/model/recipes` | `infer-model-recipes` | 可复用的模型执行配方，定义 decoder/hybrid 的算子连接、权重槽与状态读写 |
| `crates/model/compiler` | `infer-compiler` | 接收模型提供的 dataflow，校验、规划生命周期并选择 kernel |
| `crates/engine/state` | `infer-state` | 逻辑状态、块池、prefix cache、KV manager 与物理对照 |
| `crates/engine/scheduler` | `infer-scheduler` | 多队列、调度策略、公平性、预算与成本校准 |
| `crates/engine/workloads` | `infer-workloads` | 采样、embedding、rerank、decision 与 projection |
| `crates/engine/runtime` | `infer-runtime` | 请求所有权、阶段推进、资源交接与故障处理 |
| `crates/diagnostics/observe` | `infer-observe` | 指标、trace、事件、源码映射与诊断视图 |
| `crates/diagnostics/quality` | `infer-quality` | golden、进展检测、校准与性能测量 |
| `crates/service/media` | `infer-media-input` | 可复用的多模态来源读取和媒体解码 |
| `crates/service/frontdoor` | `infer-frontdoor` | 输入准备、单所有者 actor、HTTP/SSE 与交付 |
| `crates/service/agent` | `infer-agent` | 诊断工具、证据查询、实验与建议 |
| `crates/service/cli` | `infer-cli` | 参数解析、命令分发、backend 选择与运行支持 |
| `crates/testing/cpu/reference-ops` | `infer-reference-ops` | 标量参考算子：设备 kernel 的独立数值对照基准 |

## 依赖方向

基础类型与 IR 位于依赖底层；SPI 描述接口；模型编译、state、scheduler 与 workload 各自实现职责。runtime 组合这些模块并通过 SPI 驱动 backend；frontdoor、agent、cli 位于入口层。runtime 不依赖 Metal 或 CUDA 实现，backend 不依赖服务入口，默认 CLI 不依赖 CPU 测试执行器。

- 队列与调度策略归 `scheduler`；runtime 的 `pipeline/scheduling` 只负责把请求、就绪视图和资源结果接到策略，不另建一套队列。
- KV 的逻辑分配与所有权归 `state`；backend 负责设备内存与复制执行；runtime 负责资源票据与请求生命周期的协调。

## 模型执行与设备实现的边界

```text
ModelProvider::import / graph / draft_graph
  → ModelPackage / QuantizedPackage   配置、执行图与权重绑定
  → backend loading                  绑定图中的权重与设备状态
  → BackendProvider::execution_graph 返回已经绑定的模型图
  → compiler::lower / compile        校验、生命周期和 kernel 注册选择
  → backend execution                设备录制、融合、加速与执行
```

模型 provider 决定算子顺序、连接、位置编码以及 speculative draft 拓扑；可复用 `model/recipes`，也可以生成自己的 `DataflowGraph`。没有实现执行配方的 provider 明确返回 unsupported，不自动套用 decoder。compiler 不依赖模型包或配方；runtime 从 backend 获取加载时绑定的图，不从 `ModelIr` 重建拓扑。线程包装与 CLI backend 选择必须转发这份图。调度器只消费计划、预算、状态与完成信息，KV 逻辑生命周期归 state，设备分配和数据搬运归 backend。

通用 attention 描述位于 `kernel-api/attention.rs`，不包含模型名、GPU 型号、cuTile 或 Candle 类型。CUDA 的 `attention` 目录提供计划和 kernel 实现，按维度、mask 和设备能力选择路径。模型里的 attention、MLP 等组合关系属于模型图；等价融合和具体加速策略属于后端。Candle 仅作为隔离性能基线，不进入生产 runtime。

`ModelPackage` 是内部统一名称；`QwenPackage` 只保留为包 facade 的兼容别名。

## 推理调用链

```text
CLI / HTTP / SSE
  → frontdoor::ingress / cpu            有界接入与异步输入准备
  → frontdoor::actor                    单所有者驱动、命令与输出交付
  → runtime::preparation / pipeline::admission
  → runtime::pipeline::scheduling       就绪视图、阶段与资源状态衔接
  → scheduler::queue / policy           队列选择、公平性、预算与计划
  → runtime::pipeline::dispatch         计划、resource、任务与 backend 交接
  → BackendProvider                     submit / poll / 资源操作
  → runtime::pipeline::completion       完成反馈与 CPU 输出阶段
  → runtime::pipeline::lifecycle        终止、取消、超时与回收
  → frontdoor::actor::delivery          HTTP/SSE 输出与背压
```

prefill / decode 适配保留在 `runtime/src/stages`，runner 管理 backend 线程与交接。观测由 `runtime/src/observation` 收集并通过 `infer-observe` 输出；checkpoint、snapshot 与 replay 归 `runtime/src/persistence`。

## 模块内组织

- `model/package/src/providers`：架构注册、配置解释、权重命名和模型配方入口。
- `model/package/src/storage`：索引、Safetensors、权重绑定、量化存储、内存预检和有界加载。
- `model/package/src/input`：文本、图像、位置与提示准备；生成参数单独归 `generation`。
- `model/recipes/src/decoder.rs`：模型侧 decoder/hybrid 配方；不依赖 compiler、runtime 或后端。
- `runtime/src/engine`：类型、构造、检查与不变量，不混入阶段执行。
- `runtime/src/pipeline`：admission、scheduling、dispatch、completion、lifecycle。
- `runtime/src/requests`：请求 record、冷热存储与 tenant 统计。
- `state/src/logical`：reservation、prefix、lifecycle、inspection；`blocks` / `cache` / `kv` 保持独立。
- `metal/src/executor`：loading、setup、state、prefix、forward、submission、execution、completion、checkpoint；`provider` 汇集 SPI 适配。
- `spi/src`：按 provider、model、workload、scheduling、state、backend、kernel、protocol、observer 与扩展合约拆分。
- `frontdoor/src/actor`：启动、驱动、命令、准入与交付；`http` 按 native、text、protocol、observability、server、error 拆分。
- `cli/src/commands`：按命令领域组织；`support` 按 engine、JSON、fixtures、execution、server、agent 组织。

公开 API 由 facade 显式导出，依赖在使用处导入，helper 使用最窄可见性。

## 文档与工具

`docs/guides` 放操作、测量方法与验收边界，`docs/architecture` 放当前实现契约，`docs/adr` 放已采纳的架构决策。模型样例与测试资产归 `examples`，可复用基线归 `benchmarks`，生成日志、临时报告与负载结果归 `artifacts`；不另设过程计划或会话进度目录。

`tools/check` 是统一门禁与架构规则，`tools/validation` 是服务与负载验证，`tools/fixtures` 是独立参考导出，`tools/bench/cpu` 是隔离的分配测量 crate；`tools/attention` 保存独立正确性 oracle 和性能判定，`tools/bench/attention` 是隔离的 Candle 基线 crate。`Makefile` 与 CI 调用同一门禁入口。

新增 crate 必须落在职责分组下，新增 backend 必须落在 `backend/<name>`，CPU 对照只能位于 `testing/cpu`。目录门禁校验分组、路径依赖、facade 大小、实现层反向依赖与文档链接，并检查 target 条件依赖，禁止绕过模型/编译器/调度边界；Rust 门禁继续校验公开 API、feature 隔离与全部 lint。
