# Agentic 服务层方案

状态：A1 已实施（crate 建立与 Responses 生命周期迁移），A0、A2 及之后未开始。源码基点 `44b5a257`，日期 2026 年 11 月 24 日。参考 [vllm-project/agentic-api](https://github.com/vllm-project/agentic-api/tree/41a0fc23275883786fc512e6192dbf1f8b3d1f01)（固定 `41a0fc2`，PyPI `0.10.0` 同期），只借鉴机制，不照搬参数集合与默认值。排期见[路线图](../README.md) S4；现有 Responses 状态契约见 [Responses 状态与扩展 SPI](../../architecture/responses-state.md)，工具协议与严格约束见 [Agent 工具调用与服务配置方案](agent-api.md)。

## 目标

客户端发一次 Responses 请求，服务端完成历史合并、工具循环、续接与上下文管理，只返回最终结果；中间的思考和工具执行详情供人异步查看。

1. **执行优先。** 轮次之间不重算已有上下文、不重复渲染和 tokenize 历史、不重新排队；工具一旦参数闭合就派发。
2. **观测不阻塞执行。** 展示、detokenize、SSE 编码和事件写入都在关键路径之外；慢读者只会滞后，不会拖慢 loop。
3. **语义无状态，实现有状态。** 对话记录（store 中的 item）是唯一事实来源，任何请求都可以从记录完整重算；引擎持有的 KV、recurrent 与 parked 状态只是缓存，丢失后退回重算，结果不变，只是更慢。

非目标：不重新实现 tokenizer 与模板（继续使用 `infer-models` 的 `TextAssets` 和方言）；不做通用插件平台；首期不做独立网关进程、Anthropic Messages、computer use 和多 agent；不在引擎进程内执行任意代码，code interpreter 一类工具由外部沙箱实现 executor。

## 与 vLLM agentic-api 的对照

agentic-api 是位于 vLLM 前面的独立 Rust 网关：`agentic-server-core` 负责领域逻辑（类型、事件、loop、工具、存储），不依赖 axum；`agentic-server` 负责 HTTP/WebSocket；上游调用 vLLM 的无状态 Responses HTTP 接口。

| 机制 | agentic-api 做法 | 本方案 |
|---|---|---|
| 分层 | core 不依赖 HTTP；transport 单独 crate | 采用：`infer-agentic` 不依赖 axum/HTTP，`frontdoor` 保留 HTTP/SSE 适配 |
| 回合决策 | `RoundDecision::{Continue, Done, RequiresClientAction, Incomplete, UpstreamTerminal}` | 采用同样语义的 typed enum |
| 工具归属 | gateway / client / provider 三类；未知形态不执行 | 采用 engine / client 两类；未知形态显式拒绝，不执行 |
| 轮数上限 | `MAX_GATEWAY_TOOL_ROUNDS = 10`，到达后 `incomplete` | 采用上限与 `incomplete` 语义；数值按本项目文档化常量与请求 `max_tool_calls` 确定 |
| 并发执行 | `GatewayScheduler` 有界 fan-out，结果按模型调用顺序回填 | 采用；复用有界 CPU 池，不按请求开任务 |
| 混合轮次 | client 调用返回给客户端，gateway 调用照常执行 | 采用 |
| 存储内容 | 只存面向客户端的 item；面向模型的形态每次请求重建 | 采用；另加引擎侧 token 账本作为缓存，见下文 |
| 流式 | 多轮共用一个 SSE 生命周期，序号重排；终态事件在持久化之后发出 | 采用 |
| 存储后端 | 内置 sqlx（SQLite/Postgres） | 不内置数据库；沿用 `ResponseStore` SPI，后端由外部实现 |
| 上游接口 | HTTP 调用无状态 Responses | **不同**：进程内 typed 接口，携带会话提示与 token 账本 |
| background | 拒绝（其 #357） | 已有非流式 background；本方案补齐流式回放 |
| 压缩 | `POST /v1/responses/compact`，上游重新 prefill 整段历史 | 后期：从已有 KV fork 生成摘要，不重新 prefill |

本质区别在于上游接口。网关只能看到 HTTP 请求和文本，以下几项只有进程内才做得到：在精确位置保留混合模型的 recurrent 状态、知道会话即将返回从而转移到主机内存而不是淘汰、按 token ID 检测 CallEnd、提高续接轮次的调度优先级、基于现有 KV fork。这些是本方案相对 agentic-api 的主要收益来源。

## 分层

```text
infer-cli
  └─ infer-frontdoor            HTTP/SSE 路由与线格式 DTO；实现 Upstream（准入、模板编码、parser）
       └─ infer-agentic         新增：会话、loop、工具调度、观测账本、上下文策略；不依赖 HTTP 与 backend
            ├─ infer-spi        ResponseStore（已有）、ToolExecutor（新增）
            └─ infer-core / infer-ir
engine（ir / state / scheduler / runtime / backend）
                                会话提示、精确快照、parked 状态、主机内存分层、fork、续接优先级
```

- 依赖方向：frontdoor 依赖 agentic，agentic 不依赖 frontdoor。agentic 通过自己定义的 `Upstream` trait 提交生成，frontdoor 用现有 `RuntimeHandle`、`TextAssets` 和 parser 实现它。测试用 mock upstream 替代，这是 trait 的实际第二个实现。
- 位置为 `crates/service/agentic`，crate 名 `infer-agentic`，属于 `service` 分组，实现层 crate 不能依赖它。已有的 `crates/service/agent`（`infer-agent`）是引擎诊断控制面，和本 crate 无关，[Agent 控制面](../../architecture/agent.md) 需要写明两者区别。
- agentic 不持有 KV、页或设备状态，只通过 `SessionHint` 表达意图；逻辑分配仍归 state，生命周期协调仍归 runtime，物理状态仍归 backend。
- Responses 编排从 `frontdoor/src/http/openai/responses/` 的 `mod.rs`、`execute.rs`、`state.rs` **迁移**到 agentic，不在两处各保留一份；frontdoor 保留 `request.rs`（线格式解析）、`events.rs`、`stream.rs`、`output.rs`（线格式编码）。持久化 payload 的格式仍归 frontdoor：agentic 只通过 `ResponseDocument` 读取类型化元数据并保存编码后的字节，不知道 JSON 形态；迁移后 payload 字节不变，`payload_version` 不升级。参考 store `InMemoryResponseStore` 也移入 agentic，frontdoor 为兼容继续重新导出。

### 模块

| 模块 | 职责 |
|---|---|
| `items` | 面向客户端的 typed item：message、function_call、function_call_output、reasoning，以及后续的内置工具 item；带 serde，作为持久化格式 |
| `session` | 由 `previous_response_id` / conversation 合并历史；维护会话 token 账本与 `SessionHint` |
| `turn` | 一次请求内的多轮 loop、`RoundDecision`、预算与上限 |
| `tools` | 工具目录、归属分类、有界调度、调用顺序回填、审批 |
| `ledger` | 只追加的执行记录与游标读取，供异步展示和事件回放 |
| `persist` | 基于 `ResponseStore` 的快照、条件写和每轮 item 写入；迁移现有 `StoreSink`/`StoreLease` |
| `context` | 压缩触发与摘要请求；后期实现 |

## 执行路径

```text
hydrate ─▶ round ─▶ (CallEnd 立即派发) ─▶ Finished ─▶ decide ─┬─ Continue ─▶ 追加结果 ─▶ round …
                                                              ├─ RequiresClientAction ─▶ 返回客户端调用
                                                              ├─ Done ─▶ 最终结果
                                                              └─ Incomplete(reason) ─▶ 上限或截断
```

1. **hydrate。** 读取 `previous_response_id` 对应的记录，合并历史。若记录带有引擎会话提示且引擎仍持有对应状态，本轮走续接；否则按完整历史编码，结果相同。
2. **round。** 调用 `Upstream::generate`，接收 typed 事件。中间轮次的文本不 detokenize 给客户端，只写入账本。
3. **CallEnd 立即派发。** engine 归属的调用在参数闭合时立即提交到工具池，不等 EOS；client 归属的调用只登记。工具名必须属于本次声明的工具集，否则这一轮失败，不执行。
4. **decide。** 生成结束后按调用顺序收集结果：
   - 只有 engine 调用：`Continue`；
   - 含 client 调用：`RequiresClientAction`，同轮 engine 调用的结果一并作为 item 返回；
   - 无调用且正常停止：`Done`；
   - 长度截断、轮数或工具次数到上限、墙钟超时：`Incomplete`，带原因。
5. **Continue。** 追加 `function_call_output` item，下一轮带 `SessionHint` 提交。引擎命中时只 prefill 增量（停止 token、模板标记、工具结果）。

上限：每次请求的轮数和工具调用总数取请求 `max_tool_calls` 与服务 cap 的较小值；单个工具超时由 executor 声明；总输出预算继续使用 `max_new_tokens` 的统一规则，按轮次累计。上限常量按数值策略命名并写明来源，不做成环境变量。

取消：客户端取消或断连时，依次取消正在进行的生成、未完成的工具调用（executor future 被 drop），并释放会话提示对应的引擎状态。每个 decode 步和每次工具返回都检查取消。

## 观测路径

- loop 只向 `ledger` 追加记录：生成的 token ID、调用开始与结束、工具结果字节。这是 O(1) 的追加，不做 detokenize、格式化或序列化。
- 读者持有游标，在独立的有界 CPU 池里 detokenize 和编码事件：
  - 前台 SSE：一个 response 生命周期跨越多轮，`sequence_number` 单调；
  - background + stream：补上目前返回 501 的路径，通过 `ResponseStore::append_event` / `events(after)` 回放，后加入的读者从游标 0 开始；
  - 只等最终结果的客户端：不创建读者，账本只在持久化时使用。
- 读者队列有界。满了就丢弃这段展示事件并标记 gap，之后从账本补齐；账本本身完整，所以只损失实时性。账本也有字节上限，超限按资源错误结束请求，不静默截断。
- 每轮结束时把该轮的 item 写入 store（background 运行时可以通过 GET 看到进度）。终态事件在 terminal 快照写入之后才发出。

## 引擎接口

```rust
/// Submits one model round. Implemented by frontdoor over the runtime; tests use a mock.
pub trait Upstream: Send + Sync {
    /// # Errors
    /// Returns admission, encoding, or capacity errors before any token is produced.
    fn generate(&self, round: Round<'_>) -> Result<RoundStream>;
}

pub struct Round<'a> {
    pub items: &'a [Item],
    pub tools: &'a ToolSet,
    pub options: &'a GenerationOptions,
    /// Engine-side state the previous round left behind; absent on a cold start.
    pub session: Option<SessionHint>,
}

pub enum RoundEvent {
    Tokens(TokenSpan),
    CallStart { index: CallIndex, name: ToolName },
    CallArguments { index: CallIndex, delta: Box<str> },
    CallEnd { index: CallIndex },
    Finished(RoundFinish),
}

pub struct RoundFinish {
    pub reason: FinishReason,
    pub usage: RoundUsage, // input, cached, output tokens
    /// Present when the engine retained state that a continuation can attach to.
    pub session: Option<SessionHint>,
}
```

- `SessionHint` 是 newtype，包含 owner、index、generation，由引擎签发。陈旧、外部或重复的提示一律拒绝，并退回完整编码，不报错。
- 续接校验："重新渲染出的完整历史 token == 引擎账本 ++ 增量"。不一致时退回最长公共前缀。已知会导致不一致的情况：
  - 当前模板在工具回传时跳过 reasoning item（见 [OpenAI 接口指南](../../guides/openai-api.md#输入-item)），而 KV 里有 thinking token；
  - Qwen3 模板在最后一个 user 消息之前剥掉 assistant 的 thinking。

  因此续接时以引擎账本为准追加增量，而不是重新渲染整段历史；校验只用于判断能否续接。
- 停止 token 采样后没有经过前向计算，增量必须从它开始。
- 现有 `RuntimeHandle` 的提交、取消和背压语义不变；`Upstream` 在 frontdoor 内是对它的薄封装。

## 工具

```rust
/// Executes tools the engine owns. Installed by the deployment like `ResponseStore`.
pub trait ToolExecutor: Send + Sync {
    fn capabilities(&self) -> ToolExecutorCapabilities; // names, effect class, timeout, output byte cap
    fn call(&self, call: ToolInvocation<'_>) -> ToolFuture<'_, ToolOutput>;
}
```

- 和 `ResponseStore` 一样放在 `infer-spi`，future 不绑定 Tokio，可以被 drop 取消，不留下部分写入。
- 归属：请求里的 function 工具归 client；executor 声明的工具名归 engine。两者同名时请求被拒绝。未知的内置工具类型继续返回 501。
- 效果类别：`ReadOnly` 直接执行；`SideEffect` 需要部署方预先授权，否则这一轮返回审批请求 item，状态为 `incomplete`。批准后通过 `previous_response_id` 续接，期间引擎状态处于 parked，续接不需要重算。语义参照 OpenAI 的 MCP approval item。
- 资源：executor 并发受一个有界池约束；单个调用超时和输出字节上限由 capabilities 声明，服务端再加一个上限。超时、错误和超限都作为 tool 输出 item 回传给模型，带明确错误文本，不伪造成功。
- 首期只提供 SDK 安装接口和测试 executor。MCP（streamable HTTP）客户端和记忆检索放在后期，新依赖按[依赖方案](../engineering/build-and-dependencies.md)评估。

## 引擎侧加速

agentic crate 只依赖下面这些能力的契约，实现分别归各自的层，按收益排序：

| 编号 | 能力 | 归属 | 效果 |
|---|---|---|---|
| E1 | `finish=tool_calls` 时在精确位置发布快照，尾部半页 COW | state + runtime + backend | 现在 decode 出的 token 不进前缀缓存，快照只在 4096 对齐点，请求结束时调用 `release_state`；修复后每轮的重复 prefill 接近 0 |
| E2 | `SessionHint`：parked 状态不占 slot，必要时转到主机内存，续接时预取 | ir + state + runtime + backend | 续接不需要全量 tokenize；长时间等待（审批、慢工具）的会话也不丢状态 |
| E3 | 续接轮次在 scheduler 中优先 | scheduler | 同一任务的下一轮按 TTFT 处理，不排在新请求后面 |
| E4 | 按 token ID 检测 CallStart / CallEnd，只在 CallEnd 时解码参数 | frontdoor upstream + models | 增量 detokenize 移出关键路径 |
| E5 | 基于现有 KV fork 生成摘要；面向 agent 的推测解码（从上下文复制起草）；并行 COW 分支 | state + workloads + backend | 后期；各自另立方案 |

工具耗时很短时序列可以不 park，直接留在 slot 里。是否 park 的阈值根据加载时测得的快照导出与绑定成本推导，不做成配置项。

## 分期与验收

| 阶段 | 内容 | 完成标准 |
|---|---|---|
| A0 | agent 回放基准：`tools/bench/` 下回放录制的多轮 trace，工具用固定延迟的 mock | 按阶段统计 decode、prefill、重算、工具等待和开销；统计 thinking 占比和可从上下文复制的 token 占比；2B 上可在 30 分钟内跑完 |
| A1 | 建立 `infer-agentic`，迁移 Responses 编排，行为不变 | **已完成**：租约、链式校验、后台任务与取消移入 agentic，`Upstream` 首版只含 `complete`/`cancel`；frontdoor 现有 HTTP 测试不改动全部通过，agentic 用脚本化 upstream 覆盖生命周期；payload 字节不变 |
| A2 | engine 工具 loop：`ToolExecutor` SPI、`RoundDecision`、上限、混合轮次、单一 SSE 生命周期、账本与 background 流式回放 | mock executor 覆盖：调用顺序回填、上限触发 `incomplete`、取消与断连回收、终态在持久化之后、未知工具拒绝、慢读者不影响 loop；OpenAI SDK 端到端跑通 |
| A3 | E1 + E2 + E3 | A0 基准中每轮重复 prefill token 接近 0；2B 上 C1 每轮 TTFT 和 C16 agent 吞吐对比 A2 有提升；续接与冷启动输出逐 token 一致 |
| A4 | 审批流程、Conversations API（新增 `ConversationStore` SPI）、上下文策略 | 审批前后状态续接无重算；压缩只在策略触发时发生 |
| A5 | MCP executor、记忆检索、E5 | 各自另立方案与验收 |

A1 是纯迁移，A2 只在 CPU 侧，可以先于 A3 合并。A3 需要设备验收，先在 2B 上做，27B 只做最终抽查；对比遵守[性能基线方案](../performance/baseline.md)。

## 测试

- 单元测试放在 `crates/service/agentic/tests/`，使用 mock `Upstream` 和 mock `ToolExecutor`，执行时间可控、无网络。
- HTTP 测试留在 frontdoor，只覆盖线格式和路由。
- 性能用 A0 基准；非 agentic 路由（Chat、decisions）每个阶段都要确认不回退。
- 门禁：`cargo fmt --all --check`、`make check-rust`；涉及 CUDA 的 A3 在 45 上加跑 `make test-cuda-hosted` 和对应设备测试。

## 风险

| 风险 | 处理 |
|---|---|
| 模板重渲染与 KV 不一致 | 续接以引擎账本为准，校验失败退回最长公共前缀；按模板和 tokenizer 指纹测试 |
| parked 会话占用显存 | 计入准入预算和资源报价；有数量、字节和 TTL 上限；内存紧张时先转主机内存再淘汰 |
| 迁移破坏现有 Responses 行为 | A1 只迁移不改行为，以现有测试和 payload 字节一致作为门槛 |
| 有副作用的工具无人审批 | 默认需要审批；只有部署方显式授权才自动执行 |
| 多轮共用 SSE 生命周期时序号或终态乱序 | 序号由账本统一分配；终态事件只在 terminal 快照写入后发出 |
| 多副本 | 共享 store 时历史可以跨副本读取，引擎状态只在本进程；未命中按冷启动处理，结果一致 |
