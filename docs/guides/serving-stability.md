# Serving 稳定性现状与生产差距

调查日期：2026-10-10；2026-10-25 复核并更新 OpenAI 流式与错误映射。其余结论和行号基于调查时源码，引用前以当前代码为准。

**定位**：性能背景见 [CUDA 性能实验记录](../archive/research/cuda-performance-experiments.md)；后续工作以 [路线图](../plans/README.md) 的范围和顺序为准，正式性能验收按 [基线方案](../plans/performance/baseline.md) 执行。本文档结论来自逐行代码调查，标记【确证】（代码确证）
/【推断】。

**总评**：稳定性骨架是生产级的 —— 全链路有界队列、防饥饿、断连干净、无 panic 路径、
无 hang 路径。短板集中在三处：OpenAI 超时与 QoS 旋钮、过载段的客户端体验
（语义与可见性）、fail-closed 隔离策略需要编排层兜底。

## 一、故障爆炸半径

| 故障类别 | 半径 | 机制 |
|---|---|---|
| 畸形请求（JSON/参数/超长/未知模型） | 只死请求（400/404/413/429/501） | 双层校验：frontdoor + engine 准入（`ir/src/request.rs:90-157`）；状态码总表 `http/error.rs:12-24` |
| 准入期资源不足（OOM/队列满） | 只死请求（429） | Capacity 一律视为背压，不进 fault |
| 输出/准备阶段错误（采样 provider panic、NaN logits） | 只死请求 | 输出 worker 与 CPU 准备池均有 per-job `catch_unwind`（`stages/worker.rs:203-206`、`frontdoor/cpu.rs:179-180`）；非 Invariant 错误不 isolate |
| **执行期设备错误**（graph replay / kernel launch 失败） | **该 step 全部请求失败 + 引擎永久隔离** | step 序列标 poisoned（`execution.rs:96-102`）；drain 再失败 → `fatal`（`execution.rs:385-390`）→ `isolate()` 闩锁、全库无清除路径（`observation/mod.rs:309-328`）；新提交 503（`admission.rs:228-232`），`/health` 503（`inspection.rs:68`）。**恢复只有重启进程** |
| 内部不变量破裂 | 毒化引擎 | 任何 tick 错误都 isolate |
| engine actor / device worker 线程 panic | 引擎瘫痪、进程存活 | 无 catch_unwind、无重启；channel 断开 → 新请求 503；在飞 step 由 30s submission 超时兜底转隔离 |
| HTTP task panic | 只死该连接 | tokio 默认隔离 |

panic 残余面【确证】：生产代码 panic/unwrap/expect 由 lint 门禁清零
（`Cargo.toml:106-139`，`unfulfilled_lint_expectations=deny`）；唯一豁死在内部页表
（`engine/state/src/logical/lifecycle.rs:42-48`），客户端输入不可达；release 无
`panic=abort`；~80 处索引/切片抽样均被准入校验或前置 guard 覆盖（抽查非穷举）。

**fail-closed 是刻意设计**（设备错误后状态可能腐败，宁可停机）。生产含义：单次
CUDA hang/错误 = 全站 503 直到替换实例 —— 必须有编排层（liveness probe + 自动重
启）配合，或做 §五.6 的引擎重建路径。

客户端断连【确证】：连接断开 → `cancel_closed_streams` 主动 cancel（actor 自醒上限
250µs）；在飞请求置 `pending_finish`，fence 落地即释放 KV/state，不占 slot 空转；
另有 `DELETE /native/v1/requests/{id}`。

## 二、排队机制

**全链路有界，无一处无界队列**【确证】：

| 队列 | 容量 | 满时行为 |
|---|---:|---|
| ingress 准入信用 | `max_jobs`（CPU workers×2） | 429 |
| CPU 准备池 | `max_jobs` + 30s 超时 | 429 |
| actor 命令 / 控制通道 | 256 / 64 | 429 |
| 设备控制通道 | 32 | 429 |
| 请求表 / 调度队列 | 256 / 256 | 429 |
| 设备提交环 | 2 | 软背压（下轮重试） |
| 每请求输出通道 | 16 事件 | 慢消费者 → cancel |

**公平与防饥饿**【确证】：ready 队列三级候选 —— aging（20ms 未服务升最高类）、紧迫
deadline（10ms 窗口）、租户间 WFQ（virtual_finish + weight 摊薄）；硬兜底 30s 队列
等待上限（到期 Failed 而非挂死）；支持 recompute 抢占（backend 声明后生效）。

**deadline/SLO 执行现状**【确证】：`qos.deadline_us` 硬执行（每 tick 扫描，到期
Failed 并通知客户端）；`ttft_slo_us`/`tpot_slo_us` **只参与排序、不强制执行**；
`reject_infeasible_slo`（不可行 SLO 拒入）存在但**默认关闭**；native API 客户端可
全权设 QoS，**OpenAI API 硬编码 `Qos::default()`（全 None）**。

## 三、过载时的客户端可观察行为

| 阶段 | 表现 |
|---|---|
| 入口（ingress/CPU/通道/配额满） | 快速 429（语义正确） |
| quote 等 GPU 超 5s | **503**（语义错误：过载应报 429 + Retry-After） |
| 已准入、排队等资源 | TTFT 线性增长，**客户端零排队可见性**，最长 30s → Failed |
| 设备卡死 | 30s watchdog → 在飞全部失败 + 整机隔离 → 全站 503 |
| OpenAI 面 | 容量 429、模型 404、未实现 501、backend 503；尚无请求级 timeout 旋钮（`openai/error.rs`） |

## 四、hang 风险排查结果（均无 hang 路径）【确证】

- step 卡死：30s submission watchdog → 隔离（不 hang）；
- parked 请求事件丢失：250ms 资源 ack 超时兜底；ticket 丢弃有 abandon 补偿；
- 长 step 控制面假死：engine/actor 线程不假死（tick 全非阻塞、actor 250µs 自醒），
  仅设备工单延迟到 step 间隙（chunked prefill 使窗口 = 单 chunk 时长）；
- 死锁/活锁：ProgressGuard（连续 3 次空调度 → 隔离）；全 try_send/try_recv，无锁环。

## 五、生产差距 → 改进项

| # | 项 | 落点 | 优先级 |
|---|---|---|---|
| 1 | ~~OpenAI 流式缺失~~ **已交付**：`stream=true` 返回 SSE，含 usage 与 tool deltas | 见 [OpenAI 接口指南](openai-api.md) | 已完成 |
| 2 | **OpenAI 无超时/deadline 旋钮**（QoS 全 None） | `openai/execution.rs` 的 `Qos::default()`；映射到 `qos.deadline_us`（执行机制已有） | P0 |
| 3 | 过载信号语义：quote 超时 503 → 429 + Retry-After | `actor/handle.rs:57-69`、`openai/error.rs` | P1 |
| 4 | 排队 early rejection：开 `reject_infeasible_slo`；按排队深度预测 TTFT 必超则拒入 | `ir/src/scheduling.rs:220-227`（默认关）、`scheduler/src/admission.rs:99-108` | P1 |
| 5 | TTFT/TPOT SLO 强制执行（现仅排序） | `ready.rs:131-143` 的 latency_deadline 消费方 | P2 |
| 6 | **隔离不可恢复**：引擎重建路径（或文档化编排要求） | `observation/mod.rs:309-328`、`pipeline/admission.rs:228-232` | P1 |
| 7 | 排队可见性：queue position / ETA（响应头或轮询端点） | actor 队列深度已在手，暴露即可 | P2 |
| 8 | **过载 soak 测试缺失**：429 风暴下延迟/拒绝率回归 | 按 [测试组织方案](../plans/engineering/tests.md) 登记端到端场景；负载与指标执行 [基线方案](../plans/performance/baseline.md#服务矩阵与判定范围) | P1 |

## 六、测试覆盖现状

已有：`runtime/tests/capacity.rs`（背压重试不丢请求）、`control_path.rs`（超时隔离/
deadline/livelock）、`frontdoor/tests/isolation.rs`（owner 阻塞下控制面存活）、
`output.rs`（输出超时）、`http/tests.rs`（断连取消、慢消费、quarantine 健康检查）、
`scheduler/tests/lifecycle.rs`（wait deadline）。

缺口【推断】：无端到端"N 客户端打爆 → 断言 429 比例/TTFT 分布"的过载 soak；无
quarantine 后编排恢复的演练测试。

## 七、请求终态日志与指标契约

每个请求恰好有一条终态行，目标都是 `infer::request`，并带有 `phase` 字段，所以一条查询就能覆盖所有结局。被引擎接收的请求还会先输出一条 `phase="accepted"` 的接收行，它不算终态行。字段名是稳定契约，可以直接用来 grep 或配置告警。

| 字段 | 取值 | 含义 |
|---|---|---|
| `request` | u64 | 请求 ID，同时出现在 `infer::http` 行上 |
| `phase` | `finished` / `rejected` / `abandoned` | `finished`：引擎里的请求到达终态；`rejected`：请求在准备阶段或准入时被拒，引擎里没有记录；`abandoned`：请求在准备阶段被放弃 |
| `reason` | `Length` / `Eos` / `Completed` / `Cancelled` / `Deadline` / `Failed(..)` | 仅 `finished` 行有，即 `FinishReason` |
| `cancel` | `client_disconnected` / `shutdown` / `requested` / `backpressure` / `none` | 谁取消了请求：输出流被关闭 / 运行时关停 / 显式调用 cancel 端点 / 客户端读得太慢导致有界输出队列溢出 |
| `stage` | `media` / `preparing` / `admission` / `queued` / `prefill` / `decode` / `output` | 结局被决定时请求所处的阶段；延迟到 fence 之后才完成的结局，也按决定时的阶段记 |
| `error_code` | `invalid_input` / `unsupported` / `capacity` / `not_found` / `conflict` / `invariant` / `backend` / `none` | `Failed` 和拒绝背后的错误码，不再压平成文本 |
| `output_tokens` / `cached_tokens` | usize | 生成的 token 数 / 复用前缀命中的 token 数 |
| `ttft_us` / `max_tpot_us` / `e2e_us` | u64 | 来自现有的 measurement |
| `queue_us` | u64 | 从接收到首次 dispatch 的时间；从未离开队列的请求没有这个字段 |
| `last_token_age_us` | u64 | 最后一个 token 到终态的时间，可以区分请求是卡住了还是客户端刚走 |
| `elapsed_us` / `bytes` | u64 / usize | 仅准备阶段的行有：准备耗时，以及仍占用的 ingress 字节数 |

各退出路径对应的终态行：

| 路径 | `phase` | `cancel` | `stage` | `error_code` |
|---|---|---|---|---|
| 客户端断开（已准入，流式） | `finished` | `client_disconnected` | 当时的阶段 | `none` |
| 服务端关停 | `finished` | `shutdown` | 当时的阶段 | `none` |
| 显式 cancel（已准入） | `finished` | `requested` | 当时的阶段 | `none` |
| 慢消费者导致有界输出队列溢出 | `finished` | `backpressure` | 当时的阶段 | `none` |
| 准备阶段客户端断开 | `abandoned` | `client_disconnected` | `media` / `preparing` | `none` |
| 准备阶段显式 cancel | `abandoned` | `requested` | `media` / `preparing` | `none` |
| 准备失败（编码、约束构建等） | `rejected` | `none` | `media` / `preparing` | 对应码 |
| 准入拒绝（quarantine、容量、不支持） | `rejected` | `none` | `admission` | 对应码 |
| 已提交给引擎、等准入回复时客户端断开 | `finished` | `client_disconnected` | `queued` 等 | `none`（引擎准入后发现回复无人接收，就取消它；若引擎拒绝，则为上面一行） |
| 队列等待超时 | `finished` | `none` | `queued`（被抢占后重新排队的请求为当时的阶段） | `capacity` |
| deadline | `finished` | `none` | 当时的阶段 | `none`（`reason=Deadline`，另有 `DeadlineExceeded` 诊断） |
| 执行期后端错误、提交失败、非法 completion | `finished` | `none` | `prefill` / `decode` | 后端给出的码 |
| 隔离 drain | `finished` | `none` | 当时的阶段 | 故障本身的码 |
| 输出 worker 失败或超时、资源准备失败或超时 | `finished` | `none` | `output` 或当时的阶段 | 对应码（超时为 `backend`） |

`infer::http` 行带 `request` 字段（如果该 HTTP 交换预留过请求 ID），另外有 `stream`、`client_gone`、`headers_ms`、`elapsed_ms`。长度已知的响应在响应头就绪时记一条。流式响应（SSE）在 body 结束时记一条：客户端提前离开时以 warn 级别记为 `client_gone=true`，不再像以前那样在响应头时刻就记一个 `200`。

指标从 `Finished` 事件计数，不另做记录：

- `infer_requests_cancelled_total{backend,cause}`：按 `cancel` 取值计数。
- `infer_request_failures_total{backend,code}`：按 `Failed` 的错误码计数。
- 现有的 `infer_requests_failed_total{backend}` 保持原义：所有非成功结局，包括取消和 deadline。为了不改变这个已有序列，按错误码拆分的计数另外用了上面的新名字。

排查示例：

```sh
# 某个请求的接收行、终态行和 HTTP 行
grep -E 'request=13( |$)' server.log
# 最近因客户端断开而结束的请求，以及它们停在哪个阶段
grep 'infer::request' server.log | grep 'cancel="client_disconnected"' | grep -o 'stage="[a-z]*"' | sort | uniq -c
```

限制：

- 上表的诊断字段只为日志服务，不进入 checkpoint。从 checkpoint 恢复的请求，这些字段从恢复时刻重新开始记录。
- 预留请求 ID 之前就被拒的请求（ingress 预算已满、请求 ID 重复）没有 `infer::request` 行，只有 `infer::http` 行上的 429/409。
- 提交命令一旦送达引擎，终态行就由引擎负责写，frontdoor 不再写，以免同一请求出现两条终态行。命令没送达的情况（actor 已停止）按准备失败记，阶段是 `preparing`。
