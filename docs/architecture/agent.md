# Agent 控制面

`infer-agent` 为 Coding Agent 提供引擎诊断、控制与实验能力，自身不调用外部语言模型。依赖方向为 CLI → Agent → Runtime / SPI，核心不反向依赖 Agent。

模型生成的工具调用属于文本服务协议。当前尚无输出 tool parser 或完整 OpenAI 工具闭环；这项工作见 [Agent 工具调用与服务配置方案](../plans/serving/agent-api.md)，不能用本页的诊断命令注册代替。

## 分层

| 层 | 职责 |
|---|---|
| protocol / service | JSON-RPC envelope、通知、顺序 batch、错误与审计 |
| transport | 有界 NDJSON，独立于业务分发 |
| registry | typed 参数、规范名称 / 别名、效果与事务注册 |
| context | 控制会话、Engine、审计与实验记录 |
| commands | runtime、diagnostics、state、quality、observability 分域处理器 |
| AgentBackend | backend / kernel catalog、trace / probe 与 timing adapter |
| graph / experiment | 源码关联、隔离 baseline / candidate 与验收 |

新增工具调用使用 `AgentDispatcher::register`，新增设备实现实现 `AgentBackend`。snapshot 复用 Runtime checkpoint。

## 协议与限额

JSON-RPC 2.0：通知不回复，显式 `id:null` 才回复；batch 按输入顺序执行，不提供跨命令事务。解析 / 请求 / 方法 / 参数错误对应 `-32700` / `-32600` / `-32601` / `-32602`，领域错误保留 `error.data.code`。规范见 [JSON-RPC 2.0](https://www.jsonrpc.org/specification)。

- 单行 ≤ 1 MiB，batch ≤ 64，审计 ≤ 256 条，实验历史 ≤ 16。
- 超大行完整丢弃后继续读取，不执行截断命令；历史公开 dropped 计数。
- 每进程一个本地 stdin 会话与一个 Engine。
- `agent.discover` 返回实际方法、别名、参数与 `read_only` / `control` / `isolated_execution` 效果；效果不是权限机制。
- typed 参数拒绝未知字段，复合 IR 由自身 validator 检查，当前不自动生成完整 JSON Schema。

## 命令

| 入口 | 内容 |
|---|---|
| `agent.discover` / `agent.inspect` | 注册契约与审计 |
| `runtime.*` / `request.*` / `scheduler.*` | inspect / query / explain、submit / tick / cancel、`request.trace`、`scheduler.inspect` |
| `state.*` / `kv.*` / `cache.*` / `snapshot.*` | 状态统计、ownership、checkpoint、restore / replay |
| `runtime.graph` / `runtime.diagnose` | Request→Decision/Step/State→Program/Op→Kernel/Source；生命周期与所有权不变量 |
| `gpu.profile` / `kernel.profile` / `executor.probes` | 后端时间、注册来源、实际执行 op 与有界层探针 |
| `accuracy.verify` / `benchmark.run` / `benchmark.compare` | 数值验证、closed 测量与候选验收 |
| `experiment.run` / `inspect` / `list` | 隔离配置实验及有界结果 |

精确参数与兼容别名以 discovery 为准。关联图来自真实保留记录，并公开 trace / journal / event 的 gap；Source 来自 kernel registration。Metal op elapsed 是 CPU encoding 时间，GPU command 时间单独统计。

最小发现请求：

```json
{"jsonrpc":"2.0","id":1,"method":"agent.discover","params":{}}
```

会话示例见 [examples/agent.jsonl](../../examples/agent.jsonl)。

## 配置实验

`experiment.run` 接收 baseline / candidate `RuntimeConfig`、同一组 CanonicalRequest 与约束。两侧共享模型身份，使用 fork 后的 workload provider，创建独立的 Engine / state / cost / request 空间，顺序运行后释放，不改写当前会话。

- 实际 token、排名和结构严格比较，浮点按 atol / rtol。
- 可提供覆盖全部请求的 `reference_outputs`；未提供时只报告 baseline / candidate parity。
- `experiment.run` 不接受调用者声称的 `correctness_passed`，而是比较实际输出；`benchmark.compare` 中的同名字段仍只是外部证据。
- 约束包括成功率、SLO goodput、P99 E2E、最低 goodput 比例与可选 state bytes 上限；要求字节上限却没有测量证据时拒绝。
- 每侧上限为 30 秒或一百万 tick，最多 256 个请求。
- 结果保存 ExperimentId、workload SHA256、配置、测量、正确性与接受 / 拒绝原因。

当前实验只比较运行时配置，不执行代码制品变更。自动性能根因、远程 adapter、持久实验仓库与硬件 counters 不属于当前接口支持范围。
