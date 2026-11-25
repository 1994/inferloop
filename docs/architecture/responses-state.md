# Responses 状态与扩展 SPI

`POST /v1/responses` 的有状态能力由 `infer-spi` 的 `ResponseStore` 扩展点提供。引擎不把 response 存储、后台任务或取消策略写进协议层；`infer-agentic` 负责生命周期编排（租约、链式校验、后台任务、取消），frontdoor 只负责 OpenAI 线格式，外部实现负责持久化、原子状态迁移和跨进程共享。

## 分层

```text
frontdoor /v1/responses           线格式；ResponseBody 实现 ResponseDocument，TextUpstream 实现 Upstream
  → infer_agentic::Responses        prior / retain / complete / spawn_background / retrieve / cancel
  → infer_spi::ResponseStore        get / put / delete / transition / events
  → 调用方安装的 Arc<dyn ResponseStore>
       ├─ infer_storage::RecordResponseStore   Responses 语义只写一次
       │    └─ Arc<dyn RecordStore>             MemoryRecordStore（参考，有界）或任何记录后端
       └─ 直接实现 ResponseStore 的外部 SDK
```

`infer-spi` 不依赖 frontdoor、Tokio、数据库或 `serde_json`。存储记录里的协议 payload 是不透明字节，只有 frontdoor 知道其 JSON 含义；这使 wire format 演进不必改动每个存储后端。

## 公共类型

| 类型 | 作用 |
|---|---|
| `ResponseId` | 响应标识；1..128 字节 ASCII token；无 store 时 `resp_<request>`，写入 store 时带进程实例前缀 `resp_<instance>_<request>`，避免持久后端重启后复用标识 |
| `ResponseStatus` | `queued` / `in_progress` / `completed` / `incomplete` / `failed` / `cancelled` |
| `IncompleteReason` | `max_output_tokens`、`max_messages`、`content_filter` |
| `ResponseError` | 失败响应的 `code` 与 `message` |
| `ResponseUsage` | input/output/reasoning/cached token 计数 |
| `StoredResponse` | 响应元数据 + 不透明 `request_input` 与 `response` payload |
| `ResponseEvent` | 后台流事件序号与不透明 SSE payload（预留） |
| `ResponseStoreCapabilities` | retrieve/chaining/background/cancel/event replay、持久/共享、单记录字节上限 |
| `ResponseStore` | 异步、对象安全的存储与原子迁移契约 |
| `ResponsesExtensions` | 安装一个 `Arc<dyn ResponseStore>` 与可选 `Arc<dyn ToolExecutor>` 的扩展集合 |
| `ToolExecutor` / `ToolSpec` / `ToolName` | engine 归属工具：名称、参数 schema（JSON 文本）、效果类别、超时与输出字节上限；返回装箱 future，drop 即取消，失败以 `ToolOutput::Failed` 回传模型。`validate` 拒绝重名和 `SideEffect`（审批流程未实现） |

`StoredResponse::validate` 只校验类型化元数据：活动状态不能带 terminal 字段，`completed` 需要 `completed_at`，`incomplete` 需要原因，`failed` 需要错误，payload 版本必须匹配。payload 内容由 frontdoor 校验。

## 契约

```rust
fn capabilities(&self) -> ResponseStoreCapabilities;
fn get(&self, id: &ResponseId) -> ResponseFuture<'_, Option<StoredResponse>>;
fn put(&self, response: StoredResponse, condition: WriteCondition) -> ResponseFuture<'_, ()>;
fn delete(&self, id: &ResponseId) -> ResponseFuture<'_, ()>;
fn transition(
    &self,
    id: &ResponseId,
    allowed: &[ResponseStatus],
    next: ResponseStatus,
) -> ResponseFuture<'_, StoredResponse>;
fn append_event(&self, id: &ResponseId, event: ResponseEvent) -> ResponseFuture<'_, ()>;
fn events(&self, id: &ResponseId, after: Option<u64>) -> ResponseFuture<'_, Vec<ResponseEvent>>;
```

- `ResponseFuture` 是 `Pin<Box<dyn Future<Output = Result<T>> + Send>>`；SPI 不引入 runtime 依赖。
- 完成、失败与断连清理都使用 `WriteCondition::UnlessTerminal`：任何已写入的终态优先，后到的写不会覆盖 cancelled/failed/completed。条件不满足是成功的 no-op。
- `transition` 必须原子地检查 `allowed` 与 `ResponseStatus::can_transition_to`；失败分别返回 not-found / conflict。
- `get`/`delete` 的缺失记录分别是 `Ok(None)` / no-op；`transition` 的缺失记录是 not-found。
- 扩展实现必须 `Send + Sync`，并保证返回的 future 可被取消（drop）而不留下部分写入。
- `event_replay` 与 `background` 是独立能力；不支持时返回 `unsupported`，不能静默丢弃。

## 状态机

当前 frontdoor 的后台请求在准入完成后直接写 `in_progress`；`queued` 留给外部 store 在真正排队时使用。

```text
queued ──→ in_progress ──→ completed
   │            ├────────→ incomplete
   │            ├────────→ failed
   │            └────────→ cancelled
   ├─────────────────────→ failed
   └─────────────────────→ cancelled
```

`completed` / `incomplete` 是前台正常终态；`failed` 表示生成或后台错误；`cancelled` 只由取消迁移产生。`previous_response_id` 只接受 store 中保留的 `completed` 与 `incomplete`，且模型必须匹配；活动、失败、取消记录返回冲突。

## 请求流程

### 前台

1. 解析并校验请求；`previous_response_id` 先从 store 读取并合并历史。
2. 走共享准入与编码路径；分配 request id。
3. 安装 store 时先写入 `in_progress` 快照（空 output、usage null）。
4. 非流式完成后写 terminal 快照；流式在终末事件前写 terminal 快照。
5. 初始 store 写入失败时取消已提交的 generation 并返回错误；已发布流前缀不回写。

### 后台

1. 请求先在当前请求内完成准入与编码，随后写入 `in_progress` 快照。
2. 从有界后台许可池取一个 permit；池满返回 429，不提交新 generation。
3. 任务在独立 Tokio 任务中运行，permit 随任务释放。
4. 完成/失败写 terminal 快照；`UnlessTerminal` 保证已写入的终态优先。
5. 客户端用 `GET /v1/responses/{id}` 轮询。

后台流式（`background=true, stream=true`）当前返回 501；事件存储契约已预留，后续可接事件回放路由。

### 读取与取消

- `GET /v1/responses/{id}`：读取记录，并用元数据的 status/completed_at/error/incomplete/usage 覆盖不透明 body 中的对应字段。这样取消迁移不需要重写 payload。
- `POST /v1/responses/{id}/cancel`：读取记录，原子迁移到 `cancelled`，然后调用 `RuntimeHandle::cancel(request_id)`。终态取消返回 400；取消后后台任务的 terminal 写入被条件写拦下。

## 参考实现与边界

参考后端由两层组成（见[存储层方案](../plans/engineering/storage.md)）：`infer-storage` 的 `RecordResponseStore` 把 Responses 语义写在任意 `infer_spi::RecordStore` 之上，`MemoryRecordStore` 是有界的进程内记录存储。默认关闭，通过 `--enable-response-store` 安装二者的组合：

| 限制 | 默认 |
|---|---|
| 记录数 | 256 |
| 总 payload | 64 MiB |
| 单记录 | 4 MiB |
| 单响应事件 | 4096 |
| 终态保留 | 1 小时（LRU） |
| 活动记录 | 不因 TTL/LRU 驱逐 |

- 每个响应是一条记录，状态标签就是响应状态，后台事件是记录的日志。终态快照"不覆盖已取消/已终态的记录"是一次带 `StateNotIn` 条件的写入；状态迁移是"读版本 → 按版本条件写"，冲突时重试。
- 活动响应写成 `Retention::Pinned`，终态响应写成带 TTL 的 `Retention::Disposable`。记录存储只按保留等级淘汰：容量不够时按 last-used 丢弃 disposable 记录，pinned 记录从不被驱逐，只有它们挡路时写入才失败。
- 替换记录保留它的日志，删除和过期才连同日志移除，所以后台响应的终态快照不会抹掉已经产生的事件。
- 记录与日志都在同一把短临界区锁内处理，不在 `.await` 中持锁。
- 这是单进程、非持久、非共享实现。需要跨重启或多副本时，实现一个 durable/shared 的 `RecordStore` 即可复用同一份 Responses 语义；也可以像以前一样直接实现 `ResponseStore`。两种实现都跑 `crates/testing/storage/conformance` 的同一套测试；`spi_version` 和 `payload_version` 用于兼容校验。
- 存储包含 prompt 与输出文本；外部实现负责加密、租户隔离、访问控制和保留策略，不能把 payload 写入普通日志。
- 前台请求或 SSE 客户端提前断开时，`StoreLease` 会尽力把仍活动的记录写成 failed（取消优先时保持 cancelled）；它不阻塞断连清理，也不保证进程崩溃后的清理。

## 安装方式

```rust
let store: Arc<dyn infer_spi::ResponseStore> = Arc::new(MyStore::new()?);
let extensions = infer_spi::ResponsesExtensions::new().with_store(store);
extensions.validate()?;
infer_frontdoor::serve_with_text_media_and_responses(
    listener, handle, assets, media, extensions, shutdown,
).await?;
```

CLI 的 `--enable-response-store` 安装 `RecordResponseStore` + `MemoryRecordStore`。没有安装 store 时，`store:true`、`previous_response_id`、`background`、GET 与 cancel 都返回明确错误，不会被静默忽略。
