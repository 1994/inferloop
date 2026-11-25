# 存储层方案

状态：方案，未实施。源码基点 `44b5a257`，日期 2026 年 11 月 25 日。取代 PR #313 中"每个数据库各实现一遍 `ResponseStore`"的结构：#313 的 SQLite/PostgreSQL 驱动代码、一致性测试与 CLI 入口按本方案重组后保留。排期见[路线图](../README.md) E3。

## 问题

引擎和服务层要持久化的数据越来越多：Responses 记录与后台事件（已有）、Conversations 与审批（[agentic 方案](../serving/agentic.md) A4）、parked 会话与前缀的 KV/recurrent 状态（E2 的主机内存层再往下）、quiescent checkpoint（`RuntimeSnapshot` 与 backend `execution_state`，目前只返回给调用方），以后还有记忆检索。部署形态也不止一种：单机用本地文件，多副本共享 PostgreSQL 或 MySQL，KV 放本地 NVMe，系统提示的预计算前缀从只读文件加载，冷数据进对象存储。

如果每个领域对每个后端各写一份实现，工作量是"领域数 × 后端数"，每份都要重复过期、条件写、事件顺序和取消竞争这些语义，任何一处细节不一致都会表现为某个后端上的偶发错误。#313 已经出现了这个形状：SQLite 和 PostgreSQL 两个文件各写了一遍完整的 Responses 语义。使用方也不应该知道数据最终落在哪里。

## 目标

1. **使用方只依赖领域接口。** frontdoor、agentic、engine state 只拿 `ResponseStore`、`StateArchive` 这类按用途命名的 trait，不出现 URI、SQL、文件路径或驱动类型。
2. **领域语义只写一份。** 每个领域在一个适配层里实现一次，所有后端共用；后端只实现少数几个按访问模式划分的存储原语。工作量变成"领域数 + 后端数"。
3. **异构与分层是配置，不是代码。** 每个领域可以落在不同后端；同一领域可以叠加多层（主机内存 → 本地文件 → 对象存储、可写层 + 只读种子层），组合体本身也实现同一个原语。
4. **两级扩展。** 第三方可以只实现一个原语（例如 MySQL 的 `RecordStore`）并注册 URI scheme，自动获得所有领域；也可以继续直接实现领域 trait（现有的 `ResponsesExtensions::with_store` 不变）。
5. **一套一致性测试。** 原语和领域各有一套可复用的测试，任何后端（包括第三方的）跑同一套。
6. **不碰执行热路径，不破坏所有权契约。** 存储 IO 不在 scheduler、device owner 或 tokio 服务线程上执行；KV 的导出与恢复仍遵守 fence、quiesce 与身份校验规则，存储层只搬字节。

非目标：通用 ORM 或查询语言；跨 key 的事务；后端之间的数据自动迁移；替换已有的机器本地派生缓存（调优表、kernel 身份、cuTile 缓存，见"数据分类"）；在本方案内实现远程 PD 传输协议。

## 数据分类

存储原语按访问模式划分，不按数据库划分。把 GB 级 KV 塞进 SQL，或者要求对象存储支持事务，都是原语选错了。

| 类别 | 例子 | 大小 | 访问模式 | 一致性 | 生命周期 |
|---|---|---|---|---|---|
| 记录 | Responses、后台事件、Conversations、审批、KV 归档的 manifest 与索引 | KB 到 MB | 按 key 点读点写；按状态条件写；每个 key 一条有序追加日志 | 单 key 线性一致；多副本共享 | TTL，删除时连同日志 |
| 大对象 | KV 页、recurrent/hidden 状态、checkpoint 的 `execution_state` | MB 到 GB | 写一次不改；流式写、按范围读；读写直连 pinned 主机内存 | 提交前不可见，提交后不可变 | 容量预算 + 淘汰 |
| 派生缓存 | 调优表、kernel 身份、cuTile/tileiras 缓存 | KB 到 MB | 进程启动时读，偶尔追加 | 可丢弃、可重建 | 随构建或设备失效 |
| 检索索引 | 记忆检索（A5） | 不定 | 相似度查询 | 最终一致 | 按策略 |

本方案定义前两类的原语。派生缓存保持现状：它们按构建和设备身份失效，丢了只是变慢，放进共享存储反而引入版本耦合；需要在机群间共享时，再作为大对象的一个用途接入。检索索引的访问模式与前两者不同，到 A5 时单独定义原语，不硬塞进记录。

## 分层

```text
使用方         frontdoor / agentic                      engine state / scheduler
                    │                                           │
领域接口       ResponseStore   ConversationStore(A4)      StateArchive        ── infer-spi
(按用途)            │                │                         │
领域适配       RecordResponseStore  RecordConversationStore  BlobStateArchive  ── infer-storage
(语义各一份)        └────────┬───────┘                    ┌────┴────┐
存储原语                  RecordStore                     BlobStore  RecordStore   ── infer-spi
(按访问模式)               │                               │
组合体          Namespaced · Instrumented · Encrypted      Tiered · ReadOnly · Instrumented
                           │                               │
后端            memory · file · SQLite · PostgreSQL ·      memory · file · 对象存储
                MySQL · Redis …                            (opendal 等，后续)
                           │                               │
                infer-storage(memory/file)                 infer-storage(memory/file)
                infer-storage-sql(SQLite/PG/MySQL)
部署层          StorageProfile：URI → 后端实例 → 组合 → 领域句柄；能力校验；凭据处理  ── infer-cli
```

- **依赖方向。** 领域接口和存储原语都在 `infer-spi`（foundation），没有实现依赖。engine 只依赖 `infer-spi` 的 `StateArchive`，具体实现由 CLI 注入，所以 engine 不依赖任何存储实现 crate，也不会出现 engine → service 的依赖。
- **crate 划分。**
  - `infer-storage`（service 组）：领域适配、memory 与 file 后端、组合体、`StorageProfile` 与 scheme 注册。不依赖任何数据库驱动。
  - `infer-storage-sql`（service 组）：SQL 家族的 `RecordStore`。内部共用一层 `SqlDatabase`/方言，SQLite、PostgreSQL、MySQL 各自只做驱动映射。#313 的 `infer-response-store` 改名并入这里。
  - `crates/testing/storage-conformance`（dev-only）：原语与领域的一致性测试套件，第三方后端也可以依赖它。
  - 以后的对象存储、Redis 等驱动较重的后端各自成 crate，部署按需链接。
- **为什么领域适配放在 `infer-storage`，而不是 `infer-agentic`。** Responses 的生命周期编排（租约、链式校验、后台任务）留在 agentic；"一个响应记录怎样编码、状态迁移怎样映射到条件写"是存储语义，和 ConversationStore、StateArchive 的适配放在一起，可以共用编码、重试和过期规则。

## 存储原语一：`RecordStore`

带版本的记录，加每个 key 一条追加日志。

```rust
/// Durable keyed records with optimistic versions and one append-only log per key.
pub trait RecordStore: Send + Sync {
    fn capabilities(&self) -> RecordCapabilities;
    fn get<'a>(&'a self, key: &'a RecordKey) -> StoreFuture<'a, Option<Record>>;
    fn put<'a>(&'a self, key: &'a RecordKey, write: RecordWrite<'a>) -> StoreFuture<'a, PutOutcome>;
    fn delete<'a>(&'a self, key: &'a RecordKey, condition: Precondition) -> StoreFuture<'a, bool>;
    fn append<'a>(&'a self, key: &'a RecordKey, entry: LogEntry<'a>, limit: LogLimit)
        -> StoreFuture<'a, ()>;
    fn read_log<'a>(&'a self, key: &'a RecordKey, after: Option<LogSequence>, limit: usize)
        -> StoreFuture<'a, Vec<OwnedLogEntry>>;
}

pub struct RecordKey { namespace: Namespace, id: RecordId }   // 校验过的有界 ASCII
pub struct Record { version: Version, state: RecordState, expiry: Expiry, value: Vec<u8> }
pub struct RecordWrite<'a> { value: &'a [u8], state: RecordState, expiry: Expiry, condition: Precondition }
pub enum Precondition { Any, Absent, Version(Version), StateNotIn(StateSet), StateIn(StateSet) }
pub enum PutOutcome { Written(Version), Rejected { current: Option<(Version, RecordState)> } }
pub enum Expiry { Never, At(UnixMillis) }
```

**语义**

- **单 key 线性一致，没有跨 key 事务。** 盘点下来，所有领域操作都能落成"一个 key 上的条件写"加"该 key 日志上的追加"。不要求跨 key 事务，原语就能映射到 SQL、Redis（WATCH/Lua）、单进程文件（锁 + WAL）、DynamoDB 一类 KV，以及支持条件 PUT 的对象存储。
- **`Version`** 由 store 分配，每个 key 单调递增，删除后重建从新的序列开始，不复用旧版本号。
- **`RecordState`** 是领域定义的小整数标签，比如 Responses 的状态枚举。store 不理解它的含义，只用它判断条件。`StateNotIn(终态集合)` 让"终态快照不覆盖已取消或已完成的记录"这类常见写入在 SQL 上一条语句完成。不支持状态谓词的后端（比如只有 ETag 条件写的对象存储）在 `capabilities` 里声明，由 trait 的默认方法退化成"读版本 → 按版本条件写 → 冲突重试"。重试次数有上限，是命名常量；同一 key 的竞争方只有所有者和取消者，上限很小。
- **过期由写入方给出时间点**，store 只负责：过期记录对读不可见，并连同日志删除。"活动记录不过期、终态记录最后一次写入后 TTL 过期"这条 Responses 规则放在适配层，按状态选 `Expiry`，store 不用知道。时间点用写入方时钟，TTL 是小时级，副本间秒级时钟偏差不影响语义；文档写明这一点，不引入服务端时钟。
- **日志** 附属于记录：
  - 序号严格递增，超出 `LogLimit` 返回 capacity；
  - 记录不存在或已过期时追加返回 not-found；
  - 替换记录（新版本）保留日志，删除和过期才连同日志移除。这条修正了 #313 发现的参考 store 缺陷。
- **值不透明。** store 不解析 value。领域适配层用带版本号的信封编码，信封格式见"领域适配"。

**能力声明**：`durable`、`shared`（多副本可共享）、`state_predicates`、`max_value_bytes`、`max_log_entries`。部署层按领域要求校验，比如多副本部署的 Responses 要求 `shared`；不满足就启动失败，不静默降级。

## 存储原语二：`BlobStore`

不可变大对象。

```rust
/// Immutable large objects written once and read by range into caller-owned host buffers.
pub trait BlobStore: Send + Sync {
    fn capabilities(&self) -> BlobCapabilities;
    fn create<'a>(&'a self, key: &'a BlobKey, length: u64) -> StoreFuture<'a, Box<dyn BlobWriter + 'a>>;
    fn read(&self, key: &BlobKey, range: Range<u64>, buffer: IoBuffer) -> IoFuture<'_, IoBuffer>;
    fn stat<'a>(&'a self, key: &'a BlobKey) -> StoreFuture<'a, Option<BlobInfo>>;
    fn delete<'a>(&'a self, key: &'a BlobKey) -> StoreFuture<'a, ()>;
    fn list<'a>(&'a self, prefix: &'a BlobPrefix) -> StoreFuture<'a, Vec<BlobInfo>>;
}

pub trait BlobWriter: Send {
    fn write(&mut self, buffer: IoBuffer) -> IoFuture<'_, IoBuffer>;
    /// Make the object visible atomically. Dropping an uncommitted writer discards it.
    fn commit(self: Box<Self>, checksum: Checksum) -> StoreFuture<'static, BlobInfo>;
}
```

**缓冲区所有权是这个原语最关键的设计。** 取消一个 future 不代表 IO 已经结束：线程池里的 `pread` 或 io_uring 可能还在往缓冲区写。所以读写都转移缓冲区的所有权：调用方交出 `IoBuffer`，完成后由 future 交还。future 被丢弃时，store 一直持有缓冲区，直到底层 IO 真正结束，再按 `IoBuffer` 自带的归还方式释放。这和仓库"取消与超时不证明设备或 reader 完成"的规则一致。`IoBuffer` 可以包装 backend 的 pinned 主机内存，DMA 直接进出，不做中间拷贝。

**其他规则**

- **提交前不可见。** 文件后端写临时文件，fsync 后 rename；对象存储用 multipart 加 complete。崩溃不会留下半个可见对象，未提交的临时数据在启动清扫时删除。
- **完整性。** 每个对象带 crc32c 校验和（硬件指令，GB/s 级），读后由调用方或适配层核对。不用内容寻址的哈希做 key：在 spill 路径上对 GB 级数据算密码学哈希代价太高，而去重本来就在 manifest 层按前缀身份完成。
- **对齐与直接 IO。** `capabilities` 声明对齐粒度，调用方按此分配缓冲。文件后端是否用 `O_DIRECT` 由加载期测量决定，不做成开关。
- **并发与背压。** 每个 store 限制在途字节数（credit），超出时 `create`/`read` 等待；调用方据此决定放弃 spill 还是等待。

## 领域接口与适配

### `ResponseStore`（已有，接口不变）

`RecordResponseStore<R: RecordStore>` 实现一次：

| `ResponseStore` 操作 | 映射 | 往返 |
|---|---|---|
| `put(Always)` | `put(condition: Any)` | 1 |
| `put(UnlessTerminal)` | `put(StateNotIn(终态))`；无状态谓词时用 CAS 循环 | 1 |
| `put(UnlessStatus(s))` | `put(StateNotIn({s}))` | 1 |
| `transition(allowed, next)` | `get` → 检查 `allowed` 与 `can_transition_to` → 改写状态 → `put(Version(v))`，冲突时重读重试 | 2 |
| `append_event` / `events` | `append` / `read_log` | 1 |
| `get` / `delete` | 同名 | 1 |

迁移操作只出现在取消和后台任务上，多一次往返可以接受；每个存储请求的两次快照写入都是一次往返。

信封编码：`[格式版本 u16][元数据长度 u32][元数据][request_input][response]`。元数据是 `StoredResponse` 除两个 payload 之外的字段，用 serde 编码；两个 payload 原样拼接，不做 base64，也不二次 JSON 转义。格式版本独立于 `RESPONSES_PAYLOAD_VERSION`：前者是存储编码，后者是 frontdoor 的线格式。

现有的 `InMemoryResponseStore` 变成 `RecordResponseStore<MemoryRecordStore>`。条目数和总字节的 LRU 预算移进 `MemoryRecordStore`，只对终态记录生效，活动记录不被驱逐。保留旧名作为类型别名，CLI 行为不变。

### `StateArchive`（新增）

服务两件事：parked 会话和前缀的 KV/recurrent 状态下沉到本地盘（E2 主机内存层的下一层），以及从只读种子目录加载预计算的系统提示前缀。quiescent checkpoint 的持久化是第三个用途，排在后面。

```rust
/// Saves and restores engine sequence state; the engine owns when and what.
pub trait StateArchive: Send + Sync {
    fn capabilities(&self) -> ArchiveCapabilities;
    /// Longest archived prefix of `tokens` compatible with `identity`.
    fn lookup<'a>(&'a self, identity: &'a StateIdentity, tokens: &'a [u32]) -> StoreFuture<'a, Option<ArchiveEntry>>;
    fn save<'a>(&'a self, manifest: StateManifest, segments: SegmentSource<'a>) -> StoreFuture<'a, ArchiveEntry>;
    fn restore(&self, entry: &ArchiveEntry, destination: SegmentSink) -> IoFuture<'_, SegmentSink>;
    fn evict<'a>(&'a self, entry: &'a ArchiveEntry) -> StoreFuture<'a, ()>;
}
```

- **`StateIdentity`** 包含权重摘要、模型 revision、program/provider/backend 身份、精度、KV layout、`StateRecipe` 版本和 readout 兼容性，与 `PrefixCache` 的兼容规则相同。lookup 只返回身份完全匹配的条目。
- **`StateManifest`** 列出 token 数、前缀摘要、各段（KV 页、recurrent、hidden、draft）的对象 key、偏移、长度和校验和。manifest 与前缀索引写进 `RecordStore`，段数据写进 `BlobStore`。两边可以是不同后端，比如段放 NVMe 文件、索引放 SQLite。
- **租户范围。** 默认按租户划分命名空间。跨租户共享前缀会泄露"别人问过什么"的时序信息，只有部署显式声明共享的前缀（比如公共系统提示）才进共享空间。
- **`BlobStateArchive<B, R>`** 实现一次，负责：
  - 先写段、后写 manifest，崩溃时只会留下无主段，启动时清扫；
  - 恢复时核对校验和；
  - 按容量预算做 LRU 淘汰（manifest 记录最近使用）。

**与 engine 的分工**，遵守 [KV Manager](../../architecture/kv-manager.md) 与 [ADR-0001](../../adr/0001-runtime-boundaries.md)：

| 环节 | 归属 |
|---|---|
| 何时下沉、何时恢复、恢复还是重算 | scheduler 策略。恢复成本 = 字节数 ÷ 加载期测得的读带宽，与 prefill 成本模型比较，取小者 |
| 页的逻辑分配、引用与 COW | state |
| 越过设备 fence 后导出到 pinned 主机缓冲；把读回的缓冲写回新预留的页 | backend（物理所有者） |
| 搬字节、校验和、提交可见、淘汰 | `StateArchive` |
| 恢复前校验 schema、权重、program、provider、游标与 payload | state 层；不恢复 native ticket |

- 存储 IO 跑在一个新的有界 IO 池上，完成后作为资源回复交给 owner，沿用现有 resource command/reply 协议，所以 owner 线程从不等 IO。
- spill 是尽力而为：IO 池满或存储变慢时直接放弃这次下沉，那段状态退回"需要时重算"，不阻塞准入。
- 归档里的前缀成为 `PrefixCache` 候选里的另一层：命中后先异步恢复到新预留的页，再走正常的 attach，复用现有的兼容性检查、hit 提交和 LRU 语义。
- 现有的 `StateTransferProvider`（目前只有 `LocalOnlyTransfer`）是远程 PD 的入口，不在本方案内。它以后可以复用同一个 `StateManifest` 格式，传输层另行设计。

### `ConversationStore`（A4）

与 Responses 同构：一个 conversation 是一条记录（元数据加状态），item 写进它的日志。适配层到 A4 再写，原语不需要改。

## 组合体

组合体实现与内部相同的原语，使用方和领域适配都看不出区别。

| 组合体 | 原语 | 作用 |
|---|---|---|
| `Tiered<Upper, Lower>` | Blob | 读先查上层，未命中读下层并按策略回填；写入策略为 write-through 或只写上层，由领域需要声明。只用于不可变对象，没有一致性问题 |
| `ReadOnly<S>` | Blob、Record | 只读种子层，比如随部署分发的系统提示前缀目录；写入返回 unsupported |
| `Namespaced<S>` | Record、Blob | 多个领域或租户共用一个后端实例时隔离 key 空间 |
| `Instrumented<S>` | 两者 | 每个操作一个 tracing event（操作、命名空间、字节数、耗时、结果），延迟直方图进 observability；不在逐字节路径上分配 |
| `Encrypted<S>` | 两者 | 静态加密，密钥由部署提供；value 和 blob 整体加密，key 与元数据保持明文以便检索 |

记录不提供 `Tiered`：多层可变记录需要缓存一致性协议，而记录的读写量本来就小，用单一后端加副本由数据库自己负责。

## 部署配置

使用方拿到的只是领域句柄，配置只在部署层出现：

```toml
[storage.backends]
main = { url = "postgres://infer@db.internal/infer", password_file = "/run/secrets/pg" }
nvme = { url = "file:///nvme/infer" }
seed = { url = "file:///opt/infer-seed", read_only = true }

[storage.domains]
responses = "main"
state = { blobs = ["nvme", "seed"], index = "nvme" }   # 列表即分层，上层在前
```

CLI 简写：`--storage responses=postgres://… --storage state=file:///nvme/infer`。`--response-store` 和 `--enable-response-store` 保留为 `responses=` 的别名。

- **`StorageProfile`** 按 URI scheme 查已注册的工厂，构造后端实例（同名实例只建一次、可被多个领域共用），套上组合体，再按领域要求校验能力：
  - 多副本 Responses 要求 `shared`；
  - state 的段存储要求支持范围读；
  - 只读层不能当最上层的可写层。

  校验失败就启动失败。
- **扩展。** 内置工厂注册 `memory`、`file`、`sqlite`、`postgres`、`mysql`；SDK 使用方在构建服务时注册自己的 scheme。直接实现领域 trait 的旧方式继续可用。
- **凭据。** 优先从 `password_file` 读；URL 里的密码也可以用，但不进入 `Debug`、日志、调优准备键或指标标签。远程连接默认要求 TLS，明文需要显式 `sslmode=disable`。
- **生效配置可读回。** 每个领域实际选用的后端、组合层和能力出现在 `/native/v1/runtime`，凭据除外。

## 后端

| 后端 | 原语 | 要点 |
|---|---|---|
| memory | Record、Blob | 参考实现；有界，LRU 只淘汰终态记录和未 pin 的对象 |
| file | Record | 单进程：一把短锁 + 追加式 WAL + 周期快照；文件锁防止两个进程同时打开 |
| file | Blob | 一对象一文件，写临时文件后 rename；目录按 key 前缀分桶；启动清扫临时文件与无主对象 |
| SQLite | Record | 一个 WAL 连接由专用 actor 线程独占，每个操作一个事务，主机单写；适合单机 |
| PostgreSQL | Record | deadpool 连接池加 rustls TLS；状态谓词与版本条件写各一条语句；建表与迁移在 advisory lock 下执行 |
| MySQL | Record | 与 PostgreSQL 共用 SQL 家族层；`ON DUPLICATE KEY UPDATE` 与条件 `UPDATE` 由方言提供，`RETURNING` 的缺失由方言用同一事务内的 `SELECT … FOR UPDATE` 补齐 |
| 对象存储 | Blob | 后续；候选是 opendal（一套接口覆盖 S3/GCS/Azure），引入前按依赖方案评估 |
| Redis | Record | 按需；版本与状态谓词用 Lua 脚本实现原子性 |

SQL 家族的表结构是通用的 `records(namespace, id, version, state, expires_at, value)` 和 `record_logs(namespace, id, sequence, value)`，各后端用自己的迁移列表管理版本。数据库里的版本比代码新时拒绝启动，旧时按顺序迁移。人工排查时，`namespace`、`state` 和 `expires_at` 是明文列；value 的元数据段是可读的 serde 编码。

## 一致性测试

`storage-conformance` 提供：

- `record_store_suite`：
  - 条件写的每种 `Precondition`；
  - 版本单调；
  - 状态谓词与 CAS 退化路径结果一致；
  - 两个连接同时迁移只有一个成功；
  - 日志的顺序、上限、替换保留与删除清除；
  - 过期不可见；
  - 超限值与非法 key。
- `blob_store_suite`：
  - 未提交不可见；
  - 丢弃 writer 不留下对象；
  - 范围读与校验和；
  - 读 future 被丢弃后缓冲区仍在 IO 结束时归还；
  - 在途字节上限。
- `response_store_suite`：现有 #313 用例，跑在 `RecordResponseStore<各后端>` 上，再加直接实现 `ResponseStore` 的参考 store。
- `state_archive_suite`：
  - 身份不匹配不返回；
  - 最长兼容前缀；
  - 崩溃在段与 manifest 之间时清扫正确；
  - 淘汰与容量。
- **文件后端的崩溃测试**：在提交的各个点上杀掉子进程，重启后不出现半个对象或半条记录。
- **数据库后端的测试**：需要服务器的用例在 CI 里用容器跑，本地缺少服务器时这些用例失败，不算通过。

## 性能

- **不在执行路径上。** 记录操作只发生在请求开始、结束、取消和后台任务上，状态归档的 IO 在独立的有界 IO 池里，owner 线程不等 IO。每个阶段都要确认 Chat、decisions 等非存储路由不回退。
- **往返次数。** 存储一个 Responses 请求的两次快照各一次往返，与 #313 相同；迁移多一次往返。
- **状态归档。**
  - 验收看 TTFT：长前缀从归档恢复的 TTFT 必须低于重算，否则策略应选择重算；
  - 恢复带宽、spill 对在途 decode 的影响（PCIe 争用）和尾延迟按[性能基线方案](../performance/baseline.md)测量；
  - 先在 2B 上验收，27B 只做最终抽查。
- **内存。** pinned 缓冲来自 backend 已有的主机镜像预算，不另开一池；在途字节上限计入资源报价。

## 实施顺序

| 步骤 | 内容 | 验收 |
|---|---|---|
| ST1 | `infer-spi` 加 `RecordStore`/`BlobStore` 原语与能力类型；`infer-storage` 加 `RecordResponseStore`、`MemoryRecordStore`、memory blob；`storage-conformance` 套件；`InMemoryResponseStore` 改为别名 | 现有 Responses 与 agentic 测试不改动通过；三套原语/领域测试在 memory 上通过 |
| ST2 | #313 重组为 `infer-storage-sql`：通用表、迁移、SQLite actor、PostgreSQL 连接池加 TLS | 一致性测试在 SQLite 与 PostgreSQL 17 上通过；端到端重启取回与链式续接 |
| ST3 | `StorageProfile`、scheme 注册、`--storage` 与旧参数别名、能力校验、生效配置读回 | 错误配置启动失败并给出原因；凭据不出现在任何输出 |
| ST4 | file 后端（Record + Blob）与 `Tiered`/`ReadOnly` 组合 | 一致性测试与崩溃测试通过 |
| ST5 | `StateArchive` 与 `BlobStateArchive`；engine 接入（IO 池、PrefixCache 归档层、恢复/重算决策），与 agentic E2 一起做 | 恢复与重算的完整 token 序列一致；归档命中的 TTFT 低于重算；spill 不使 decode TPOT 回退；需要设备窗口 |
| ST6 | MySQL；对象存储与 Redis 按需求排期 | 同一套一致性测试 |

ST1–ST3 只在 CPU 侧，可以先合；ST5 是唯一需要设备验收的步骤。

## 考虑过但不采用

| 方案 | 不采用的原因 |
|---|---|
| 每个后端直接实现每个领域 trait（#313 现状） | 工作量随领域数和后端数相乘增长，语义在每份实现里各写一遍，已经出现过事件被替换写入抹掉的分歧 |
| 用 sqlx 或 ORM 作为存储抽象 | 只覆盖 SQL，不覆盖文件、对象存储、Redis 和大对象；sqlx 0.9 要求 Rust 1.94，高于仓库 MSRV 1.90，0.8 已停更一年半，依赖量是驱动直连的约两倍。它可以作为某个 SQL 后端的内部实现，不作为抽象层 |
| 记录与大对象用一个统一的存储 trait | 语义冲突：记录要条件写和日志，大对象要流式、范围读和缓冲区所有权转移。合在一起，两边都只能取最小公约数 |
| 原语支持跨 key 事务 | 所有领域操作都是单 key；要求事务会排除 Redis、对象存储、文件和大多数 KV，换不来任何领域能力 |
| 原语只有版本 CAS，没有状态谓词 | 每次条件写都要先读再写，多一次往返；状态谓词让 SQL 后端一条语句完成，不支持的后端由默认方法退化成 CAS，语义不变 |
| 大对象用内容哈希做 key | spill 路径上对 GB 级数据算哈希代价高，去重已在 manifest 层按前缀身份完成；完整性改用 crc32c |
| 可变记录也做多层缓存 | 需要缓存一致性协议，而记录访问量小，收益不抵复杂度 |
| 把调优表等派生缓存一并迁入 | 它们按构建与设备身份失效、可重建，放进共享存储会引入版本耦合 |

## 待定

- **记录信封的元数据编码**：serde JSON（可读）或 postcard（紧凑）。倾向 JSON，因为元数据只有几百字节，可读性有利于排查。
- **file 记录后端的快照周期**：从 WAL 长度和写入速率推导，还是固定为条目数阈值。
- **归档层的 pinned 缓冲**：是否需要独立于主机镜像的预算。取决于 E2 的 parked 会话是否与归档同时占用主机内存，在 ST5 设计时用实测回答。
