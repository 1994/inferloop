# Agentic 服务层方案

状态：A1 已实施（crate 建立与 Responses 生命周期迁移），A0 基准已建立并完成首轮测量，A2a（工具 SPI 与 loop 核心）已实施，A2b 起未开始。源码基点 `44b5a257`，日期 2026 年 11 月 24 日。参考 [vllm-project/agentic-api](https://github.com/vllm-project/agentic-api/tree/41a0fc23275883786fc512e6192dbf1f8b3d1f01)（固定 `41a0fc2`，PyPI `0.10.0` 同期），只借鉴机制，不照搬参数集合与默认值。排期见[路线图](../README.md) S4；现有 Responses 状态契约见 [Responses 状态与扩展 SPI](../../architecture/responses-state.md)，工具协议与严格约束见 [Agent 工具调用与服务配置方案](agent-api.md)。

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

1. **hydrate。** 读取 `previous_response_id` 对应的记录，合并历史，按模板完整渲染。是否接上引擎状态只取决于前缀缓存命中，见"有状态逻辑"；结果与冷启动相同。
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

- **一个序号空间。** SSE 的 `sequence_number`、后台回放的事件序号、`RecordStore` 中该响应记录的日志序号，三者是同一个序号。
- **进行中的响应** 在内存里有一个有界的环，loop 只做 O(1) 追加，读者按游标读取。
- **每轮结束和终态时** 把这一段批量追加到存储日志，终态事件在终态快照写入之后发出。后加入的读者（后台 + 流式）先从存储日志读到最新已落盘的序号，再切到内存环继续，不会漏也不会重复。
- **持久化的粒度是轮，不是 token**，存储写入次数与轮数成正比。
- 读者在独立的有界 CPU 池里 detokenize 和编码事件，覆盖三类使用方：
  - 前台 SSE：一个 response 生命周期跨越多轮，`sequence_number` 单调；
  - background + stream：补上目前返回 501 的路径，通过 `ResponseStore::append_event` / `events(after)` 回放，后加入的读者从游标 0 开始；
  - 只等最终结果的客户端：不创建读者，账本只在持久化时使用。
- 读者队列有界。满了就丢弃这段展示事件并标记 gap，之后从账本补齐；账本本身完整，所以只损失实时性。账本也有字节上限，超限按资源错误结束请求，不静默截断。

## 有状态逻辑：跨轮次的连续性

agent 的每一轮都是"上一轮的全部上下文 + 一点新内容"。体验好坏几乎完全取决于这一点：新的一轮能否接着上一轮的引擎状态，只计算新增的部分。A0 测得，现在 4096 以下的上下文每轮重算 68–77%。

### 原则：语义无状态，状态只是缓存命中

- **每一轮都按模板完整渲染整段历史**，引擎按 token 前缀匹配缓存。
  - 结果与冷启动逐 token 一致，因为输入本来就相同；
  - 缓存丢失只会变慢，不会变错。
- **同一个机制服务三类使用方，不需要各自的通路**：
  - 只发完整历史的无状态客户端（Chat Completions 一类 harness）；
  - 用 `previous_response_id` 续接的客户端；
  - 服务端工具 loop 的下一轮。
- **命中的前提是渲染忠实**：重新渲染出的历史，必须等于模型上一轮实际看到并生成的 token。凡是让两者分叉的地方，既是缓存缺陷，也是语义缺陷，比如模型丢掉了自己上一步的推理。所以修复它们同时改善速度和质量。

旧方案主张"以引擎 token 账本为准拼接增量"，不采用。拼接会让模型看到与模板渲染不同的 prompt。Qwen3.5 的模板在新的 user 轮到来时剥掉旧 thinking，拼接会把它们留下，于是违反 A3 自己的验收条件（续接与冷启动逐 token 一致），还要多维护一条"账本 + 增量渲染"的通路。token 账本只用于诊断：记录续接时实际命中到哪个位置。

### 让渲染忠实

| 分叉来源 | 现状 | 处理 |
|---|---|---|
| reasoning 不重放 | `ChatMessage` 没有 reasoning 字段：Chat 的 `reasoning_content` 被接受后丢弃，Responses 的 reasoning item 被跳过。两个出货模板（Qwen3.5、Qwen3.8）都读取 `message.reasoning_content`：前者为最后一条 user 之后的 assistant 消息渲染 `<think>…</think>`，后者默认保留全部历史的 thinking。现在模型在工具步之间丢掉自己的推理，渲染出的空 `<think>` 也和 KV 分叉 | `ChatMessage` 增加 reasoning，按模板约定传入；Chat 的 `reasoning_content` 和 Responses 的 reasoning item 都重放到对应的 assistant 消息上 |
| 工具参数重新序列化 | 参数按生成的原文保存和回传，模板侧解析为对象再渲染 | 保持原文；一致性测试覆盖"解析 → 渲染"与生成 token 的对齐 |
| 生成 token 不是规范切分 | 模型可能生成与重新 tokenize 不同的 token 序列 | 无法也不应强制。缓存在分叉点之前命中，命中位置进入指标 |
| 模板按设计改写旧历史 | Qwen3.5 在新的 user 轮剥掉旧 thinking | 这是模型的训练格式，照做；代价是新 user 轮时重算上一轮的 assistant 与工具消息 |

### 缓存断点：在每一轮的末尾留下可复用的状态

前缀缓存现在只在 4096 的倍数和 `prefix_boundary` 处发布快照，decode 出的 token 从不进入缓存，请求结束时状态整体释放。两个断点覆盖一轮的全部内容：

1. **prompt 末断点（C2）。** 多轮形态的请求（历史里有 assistant 或 tool 消息，或者声明了工具）把 `prefix_boundary` 设在渲染后 prompt 的末尾，引擎按页向下取整后在那里发布快照。下一轮只需重算这一轮的生成内容和新输入。这是引擎已有的能力，只需要 frontdoor 设置。代价是每个请求多一次快照发布，按 A0 和固定负载测量后决定是否对所有多轮请求开启。
2. **decode 末快照（E1）。** 请求以 `tool_calls`（或部署声明的续接条件）结束时，在精确位置发布包含生成 token 的快照，尾部半页 COW。对 thinking 模型这是大头：一轮的 reasoning 常有数千 token，没有 E1 下一轮就要全部重算。停止 token 采样后没有经过前向计算，所以快照止于它之前，下一轮从它开始重算。

断点只是"可以复用"的提示，复用仍然走现有的兼容性检查、身份匹配、LRU 和预算。不存在"必须命中"的路径。

### E1 的契约（实施前定稿）

现状的三个约束决定了 E1 的形状：

1. **共享前缀缓存只接受整块快照**：`publish_prefix` 要求长度是 32 token KV 块的倍数。CUDA 生产路径上的序列是槽位常驻的，快照按块表从槽位导出。
2. **混合模型（Qwen3.5 的 GDN 层）的 recurrent 状态只存在于当前位置**，不能退回到对齐点。所以 decode 末快照要么恰好在当前位置，要么不发布。
3. **快照捕获现在会同步设备**（`PREFIX_GRANULARITY` 的注释记录了每次约 0.12 s / 32K 的代价）。C2 在并发下首轮 TTFT +9% 很可能来自这里；在 decode 末同步还会拖住同批其他请求的 decode。

因此：

- **触发由请求声明**：`CanonicalRequest` 增加"结束时保留"，frontdoor 对声明了工具或带工具历史的 chat 请求设置。引擎不解析输出，不知道一轮是否以工具调用结束，也不应该知道。
- **位置精确，快照由整块加一个部分尾块组成**：整块照常进入前缀缓存，部分尾块由快照独占；复用时尾块拷贝到新序列的私有页（COW），整块按引用共享。前缀缓存的长度规则从"整块"放宽为"整块 + 至多一个由快照携带的部分块"，匹配仍要求完整 token 前缀相等。
- **只走异步捕获**：在完成路径上排入捕获，用 fence 保留源页与 recurrent 状态，直到拷贝完成；`release_state` 推迟到 fence 之后。owner 线程不等待设备，同批其他请求不受影响。C2 的 prompt 末快照一并改走这条路径，作为对那 50 ms 首轮代价的修复来验收。
- **各层落点**（读代码后确定）：
  - **ir**：`CanonicalRequest` 增加结束时保留的意图（类型化，不是布尔开关），frontdoor 按上面的条件设置。
  - **spi**：`release_state` 带上释放方式，`Discard` 或 `PublishPrefix`。发布时机只能在释放时：现在的发布点在执行任务之后，并排除了 decode 任务，而 decode 末快照只能在请求结束时拍。不支持的 backend（Metal 目前）按 `Discard` 处理并在能力里声明，runtime 不假设它发生。
  - **runtime**：完成路径按请求意图选择释放方式；推迟释放期间状态的所有权留在 backend，直到捕获的 fence 完成。
  - **state**：通用 `PrefixCache` 的"只接受整块"改为策略属性。`BlockHash` 按块建索引，仍要求整块；`SnapshotScan` 逐条比较完整 token 前缀，允许精确长度。页账本（`KvPages`）仍然只接受整页。
  - **CUDA，私有序列**：快照本来就是连续拷贝 KV 的前 `covered` 行，加上当前位置的完整 recurrent 状态，任意长度在物理上已经成立。
  - **CUDA，槽位常驻序列**（池化服务的默认路径）：整页照常发布到页账本共享；不足一页的尾部 KV 行（含 FP8/窄格式缓存）和 recurrent 状态拷进快照自带的小缓冲。恢复时先挂上整页，再把尾行写入槽位自己的下一页。那一页本来就由槽位私有，于是 COW 不需要新的页共享机制。
- **预算**：尾块与 recurrent 状态计入前缀缓存的字节预算，按现有 LRU 淘汰。等待中的捕获占用的页在 fence 前不可回收，这部分计入资源报价。
- **验收**：
  - 正确性：恢复后的完整 token 序列与冷启动逐 token 一致（含 thinking 与工具调用）；
  - 收益：A0 生成历史模式下，每轮重复 prefill 接近 0；
  - 并发：8 路并发下首轮与后续轮 TTFT 都不回退于 C2，decode TPOT 不回退；
  - 先在 2B 上验收，27B 只做抽查。

### E1 第一版的实施与实测

- **已实施**：
  - `FinishedState` 请求意图与 SPI `finish_state`（默认就是释放，Metal 不变）；
  - runtime 只在正常结束时传递意图，线程化 runner 的释放队列也携带它；
  - 前缀缓存的整块规则改为策略属性；
  - CUDA 在释放前捕获精确位置的快照，复用现有导出路径。
- **槽位常驻序列**：decode 在槽位里进行，prompt 程序的位置停在 prompt 末尾；实际状态在槽位里，与 `history` 一致。所以快照的位置由调用方给出，不能读 prompt 程序。第一版漏掉这一点时，快照被判为不合格，诊断事件给出了 `covered=318 position=290`。
- **推测中的序列（MTP）也保留**：草稿只影响提议，验证保证输出与 target 等价；草稿快照只需位置与导出的行数一致（`covered - MTP_KV_OFFSET`），由调用方显式给出。2B 开 `--num-speculative-tokens 2` 实测：保留成功、无恢复错误，thinking 关时续接与冷启动逐字相同。
- **还没做**：
  - 异步捕获。释放本来就会先 idle、再 drain 整个设备，所以同步捕获只增加拷贝本身，8 路并发下测不出开销；
  - "整页共享 + 尾块"省去整段拷贝。

  两者都留作后续优化。

实测（RTX 5090，Qwen3.5-2B，基线为 C2，两组交替，各一个新进程，只取第一次重复）：

| 场景 | 重复 prefill / 输入 | 后续轮 TTFT |
|---|---|---|
| 生成历史，thinking 开 | 16% → 0%（3 个 token） | 均值 97 → 68 ms；第 4 轮 142 → 61 ms |
| 生成历史，thinking 关 | 14% → 6% | 两边轨迹不同（见下），不可直接比较 |
| 录制历史长会话（E1 不命中，测开销） | 不变 | 不变 |
| 8 路并发录制历史（测开销） | — | 后续轮 p50 1476 → 1426 ms；首轮 p50 616 → 615 ms；decode 3.61 → 3.62 ms/token |

**数值性质与验收的修订。** 之前的前缀缓存（4096 对齐点与 C2 的 prompt 末断点）复用的都是 prefill 算出的状态，与冷启动在 bit 级上一致。E1 是第一个复用 decode 算出的状态的机制：上一轮生成的 token 的 KV 与 recurrent 状态来自 decode kernel，冷启动时则由 prefill kernel 重算，两者在 BF16 下累加顺序不同。

实测：工具对话的第二轮在 thinking 关闭时逐字相同；thinking 开启时，热缓存与冷启动各自完全可复现，但在 1461 个字符的第 117 个字符处分叉（"as a parameter" / "as input"）。所以 E1 下续接的输出是确定的，但不保证与冷启动重算逐 token 相同；vLLM、SGLang 的前缀缓存也有同样的性质。

"续接与冷启动逐 token 一致"对 C1、C2 仍然是验收条件，对 E1 改为：同一快照上的续接确定可复现，质量按准确率门槛验收。这是对输出语义的改变，合并前需要明确同意。

### 会话提示与保留（E2/E3）

- **`previous_response_id` 链** 把"这一轮很可能马上续接"告诉引擎。引擎据此提高这条前缀的保留优先级；在 E2 中，长时间等待（审批、慢工具）的会话转到主机内存，续接时预取，不必淘汰重算。
- **续接轮次在 scheduler 中按 TTFT 优先**（E3），不排在新请求后面。
- 提示由引擎签发，带 owner、index、generation；陈旧、外部或重复的提示一律忽略，退回普通匹配，不报错。

### 服务端工具 loop 复用同一机制

`RoundUpstream`（A2a）在 frontdoor 的实现每一轮都完整渲染会话、正常准入一次生成。轮间的状态连续性完全来自上面的断点，loop 不持有任何引擎状态，也不需要专门的续接接口。同一请求内的多轮于是和多个请求的多轮走同一条路径、同一套验收。

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
| C1 | 忠实重放：reasoning 进入模板，工具参数保持原文 | models + frontdoor | 模型在工具步之间保有自己的推理；渲染不再在本轮第一条 assistant 消息处与 KV 分叉 |
| C2 | prompt 末断点：多轮请求在 prompt 末尾发布快照 | frontdoor（引擎已有 `prefix_boundary`） | 下一轮只重算本轮生成内容与新输入 |
| E1 | decode 末快照：`finish=tool_calls` 时在精确位置发布含生成 token 的快照，尾部半页 COW | state + runtime + backend | thinking 模型不必每轮重算上一轮的 reasoning；配合 C1、C2，每轮重复 prefill 接近 0 |
| E2 | `SessionHint`：链式续接提高保留优先级；parked 状态转到主机内存，续接时预取 | ir + state + runtime + backend | 长时间等待（审批、慢工具）的会话也不丢状态 |
| E3 | 续接轮次在 scheduler 中优先 | scheduler | 同一任务的下一轮按 TTFT 处理，不排在新请求后面 |
| E4 | 按 token ID 检测 CallStart / CallEnd，只在 CallEnd 时解码参数 | frontdoor upstream + models | 增量 detokenize 移出关键路径 |

按体验收益排序：C1 与 C2 只在服务侧，先做并用 A0 测量；E1 需要引擎与设备验收，紧随其后；服务端工具 loop（A2b）复用同一机制，不依赖 E1。
| E5 | 基于现有 KV fork 生成摘要；面向 agent 的推测解码（从上下文复制起草）；并行 COW 分支 | state + workloads + backend | 后期；各自另立方案 |

工具耗时很短时序列可以不 park，直接留在 slot 里。是否 park 的阈值根据加载时测得的快照导出与绑定成本推导，不做成配置项。

## 分期与验收

| 阶段 | 内容 | 完成标准 |
|---|---|---|
| A0 | agent 回放基准：`tools/bench/` 下回放录制的多轮 trace，工具用固定延迟的 mock | **已完成**：[agent-replay.py](../../../tools/bench/agent-replay.py) 与两条 trace，2B 上全部模式不到 3 分钟；首轮结果见下文 |
| A1 | 建立 `infer-agentic`，迁移 Responses 编排，行为不变 | **已完成**：租约、链式校验、后台任务与取消移入 agentic，`Upstream` 首版只含 `complete`/`cancel`；frontdoor 现有 HTTP 测试不改动全部通过，agentic 用脚本化 upstream 覆盖生命周期；payload 字节不变 |
| A2 | engine 工具 loop：`ToolExecutor` SPI、`RoundDecision`、上限、混合轮次、单一 SSE 生命周期、账本与 background 流式回放 | mock executor 覆盖：调用顺序回填、上限触发 `incomplete`、取消与断连回收、终态在持久化之后、未知工具拒绝、慢读者不影响 loop；OpenAI SDK 端到端跑通。拆为三步：**A2a 已完成**（`infer-spi` 的 `ToolExecutor`/`ToolSpec`，`infer-agentic` 的 `ToolLoop` 与 `RoundUpstream`：参数闭合即派发、进程级有界并发、调用顺序回填、轮数/工具次数/输出预算上限、超时与输出超限作为失败输出回传、drop 即取消）；A2b frontdoor 实现 `RoundUpstream` 并接入单一 SSE 生命周期；A2c 账本与 background 流式回放 |
| A3 | 连续性：C1、C2（**已完成**，结果见下文），E1（第一版已实施，数值性质待确认，见上文），然后 E2、E3 | C1/C2：A0 生成历史模式（thinking 开启）每轮重复 prefill 下降、单轮 TTFT 不回退；E1：重复 prefill 接近 0；全程续接与冷启动输出逐 token 一致；2B 上 C1 每轮 TTFT 与 C16 agent 吞吐提升 |
| A4 | 审批流程、Conversations API（新增 `ConversationStore` SPI）、上下文策略 | 审批前后状态续接无重算；压缩只在策略触发时发生 |
| A5 | MCP executor、记忆检索、E5 | 各自另立方案与验收 |

A0 首轮结果（2026 年 11 月 25 日，RTX 5090，Qwen3.5-2B，`6ebde549`，每条 trace 重复 3 次，温度 0，thinking 关闭）：

| trace | 上下文 | 排队+prefill | decode | 工具等待 | prefix 复用 / 输入 | 重复 prefill / 输入 | 可复制输出 |
|---|---|---|---|---|---|---|---|
| `coding-fix`（录制历史） | ≤2.0k | 9.6% | 56% | 34% | 0 | 77% | 5% |
| `coding-fix`（模型生成历史） | ≤2.4k | 7.7% | 70% | 22% | 0 | 68% | 28% |
| `rust-review-long`（录制历史） | ≤15.5k | 55% | 43% | 1% | 65% | 9% | 35% |

- 4096 粒度以下的上下文完全不复用，链式 `previous_response_id` 与每轮全量重发结果相同；E1 对短会话把重复 prefill 从约 70% 降到接近 0，但短会话 prefill 只占墙钟 8–10%。
- 长会话 prefill 占一半以上墙钟，其中重复部分只占输入的 9%；更大的瓶颈是长前缀之后的 prefill 吞吐：12k 前缀后新增 3.2k token 用时约 1.06 s（约 3k token/s），无前缀时 2.7k token 约 0.32 s。这一项应在 A3 之前单独 profile。
- 可复制占比（≥32 字符的上下文原文片段）在模型自己写工具参数和引用代码时达到 28–35%，支持把 copy-from-context 投机放在 E1 之后。
- 2B 模型在生成历史模式下常偏离录制路径（先读文件、错误补丁、输出截断），录制历史模式用于跨构建对比，生成历史与链式模式用于验证引擎侧复用。

C1 与 C2 的首轮结果（2026 年 11 月 25 日，RTX 5090，Qwen3.5-2B，release，基线为同一栈的上一步 ST3，两组交替运行，每组一个新进程；只取第一次重复，避免后续重复命中完全相同的 prompt 而夸大收益）：

| 场景 | 重复 prefill / 输入 | 后续轮 TTFT | 首轮 TTFT |
|---|---|---|---|
| `coding-fix` 生成历史，thinking 关 | 68% → 14% | 中位 165 → 77 ms | 117 → 120 ms |
| `coding-fix` 生成历史，thinking 开（effort medium） | 55% → 16% | 中位 139 → 75 ms | 101 → 101 ms |
| `rust-review-long` 录制历史 | 19% → 0.2% | 均值 974 → 708 ms | 98 → 100 ms |
| 8 条互不共享前缀的会话并发（录制历史，两轮交替） | — | p50 2224 → 1376 ms，p90 4188 → 3267 ms，p99 4343 → 3853 ms；总墙钟 19.2 → 15.0 s | p50 556 → 604 ms，p90 784 → 840 ms |

- 剩下的 14–16% 是上一轮的生成内容，由 E1 覆盖；长会话剩余的时间主要是长前缀之后的 prefill 吞吐，单独 profile。
- 并发下首轮 TTFT 高约 50 ms：每个请求多发布一个快照的代价，8 个首轮同时到达时更明显；每组只有 16 个首轮样本，差异尚未确认。换来的是之后每一轮少约 850 ms，以及总时长少 22%。
- thinking 开启时两边都产生了 reasoning item；`usage` 的 `reasoning_tokens` 恒为 0，是已知的计数缺口，不代表 thinking 未开启。

A1 是纯迁移，A2 只在 CPU 侧，可以先于 A3 合并。A3 需要设备验收，先在 2B 上做，27B 只做最终抽查；对比遵守[性能基线方案](../performance/baseline.md)。

## 测试

- 单元测试放在 `crates/service/agentic/tests/`，使用 mock `Upstream` 和 mock `ToolExecutor`，执行时间可控、无网络。
- HTTP 测试留在 frontdoor，只覆盖线格式和路由。
- 性能用 A0 基准；非 agentic 路由（Chat、decisions）每个阶段都要确认不回退。
- 门禁：`cargo fmt --all --check`、`make check-rust`；涉及 CUDA 的 A3 在 45 上加跑 `make test-cuda-hosted` 和对应设备测试。

## 风险

| 风险 | 处理 |
|---|---|
| 模板重渲染与 KV 不一致 | 让渲染忠实（C1），无法消除的分叉（非规范 token 切分、模板按设计改写旧历史）只降低命中，不影响正确性；命中位置进入指标 |
| parked 会话占用显存 | 计入准入预算和资源报价；有数量、字节和 TTL 上限；内存紧张时先转主机内存再淘汰 |
| 迁移破坏现有 Responses 行为 | A1 只迁移不改行为，以现有测试和 payload 字节一致作为门槛 |
| 有副作用的工具无人审批 | 默认需要审批；只有部署方显式授权才自动执行 |
| 多轮共用 SSE 生命周期时序号或终态乱序 | 序号由账本统一分配；终态事件只在 terminal 快照写入后发出 |
| 多副本 | 共享 store 时历史可以跨副本读取，引擎状态只在本进程；未命中按冷启动处理，结果一致 |
