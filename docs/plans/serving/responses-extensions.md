# Responses 状态扩展与外部 SDK 方案

状态：SPI、参考 store、前台/后台非流式状态路由已交付；外部 SDK 与持久化后端适配是后续工作。实现契约见 [Responses 状态与扩展 SPI](../../architecture/responses-state.md)，公开接口见 [OpenAI 兼容接口](../../guides/openai-api.md#responses)。

本项解决的是 Responses API 的状态归属，而不是把 OpenAI 的托管服务照搬进引擎：引擎提供协议、生命周期与原子写契约，部署方决定 response/message/event 存在哪里、保留多久、是否跨副本共享。

## 参考实现

| 实现 | 默认行为 | 存储 | 已知边界 |
|---|---|---|---|
| vLLM | `VLLM_ENABLE_RESPONSES_API_STORE` 默认关闭；关闭时静默把 `store` 置 false | `response_store` / `msg_store` / `event_store` 进程内 dict，无 TTL/LRU | 官方源码标注 HACK/FIXME 与内存泄漏；有插件化 store 与有界 store 的未合并方案 |
| SGLang | `--enable-response-store` 默认关闭；关闭时 `previous_response_id`/`background`/GET/cancel 显式 400 | `response_store` / `msg_store` 进程内 dict，无 TTL；后台 queued → in_progress | PD 禁止开启；存储不跨节点；关闭时前台 `store=true` 仍接受但不保留，响应回显请求值 |
| 本项目 | 无 store 时显式 501；安装后 `store` 默认 true | frontdoor 参考 `InMemoryResponseStore`：有界 LRU + TTL + 字节预算 | 参考实现单进程；后台流式暂未开放；外部 SDK 负责持久/共享 |

参考来源：[vLLM serving.py](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/responses/serving.py)、[SGLang serving_responses.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/openai/serving_responses.py)、[SGLang #39122](https://github.com/sgl-project/sglang/pull/39122)。

本项目的取舍：

- 不用“静默降级”伪装成功。没有 store 就返回 501；安装 store 后才接受状态字段。
- 不把无界 dict 当默认。参考 store 有记录数、总字节、单记录、事件数和 TTL 上限。
- 不把协议 JSON 放进 SPI。`StoredResponse` 的 `request_input`/`response` 是 frontdoor 拥有的不透明字节，元数据是类型化字段。
- 写入 store 的 response、item 与 call id 带进程实例前缀，避免持久后端在重启/多副本后复用标识。
- 原子性进入 store 契约。取消、失败、断连与完成竞争由 `UnlessTerminal` 和 `transition(allowed,next)` 解决，而不是 frontdoor 外部锁。
- 不绑定 runtime。`ResponseFuture` 让 Redis/Postgres 实现使用自己的异步客户端；SPI 不引入 Tokio。
- `previous_response_id` 只接受 store 中已保留的 completed/incomplete 记录，并要求模型一致；失败/取消记录不可作为上一轮。

## 交付范围

已交付：

- `infer-spi::ResponseStore`、`StoredResponse`、状态/能力/错误/usage 类型、SPI 版本与 payload 版本。
- `ResponsesExtensions` 注入点与 `validate`。
- `InMemoryResponseStore` 参考实现：LRU、TTL、总字节/单记录/事件数限制、条件写、原子迁移。
- frontdoor `store`（默认安装时 true）、`previous_response_id`、`GET /v1/responses/{id}`、`POST /v1/responses/{id}/cancel`、`background`（非流式）。
- `--enable-response-store`；`router_with_text_media_and_responses` / `serve_with_text_media_and_responses` 供 SDK 注入自定义 store。
- 前台完整/流式与后台非流式都写 terminal 快照；取消优先不覆盖；前台失败写 failed 快照。
- 单元与 HTTP 测试覆盖存储条件写、迁移、驱逐、TTL、事件、store 关闭、链式输入合并、后台轮询、取消、GET 与流式落库。

未交付（明确边界）：

- 后台流式 `background=true, stream=true` 返回 501；SPI 已预留事件读写，后续接事件回放路由。
- `conversation` 仍 501；Conversations API 未实现。
- 内置工具（web search、file search、MCP、code interpreter）与其执行器未进入本项；它们属于独立的工具扩展面，不能用宽松反序列化假装支持。
- Redis/Postgres 等生产后端、SDK crate、认证/多租户与运维配置不在本仓库内实现。
- 多副本取消语义：共享 store 可以共享读取/链式上下文，但 `RuntimeHandle::cancel` 只作用于本进程；跨副本取消需要后续控制面或分布式任务归属。

## 外部 SDK 面

后续 SDK 建议只依赖 `infer-spi` 与 frontdoor 的构造入口，不复制协议解析：

```rust
pub trait ResponseStore: Send + Sync {
    fn capabilities(&self) -> ResponseStoreCapabilities;
    fn get(&self, id: &ResponseId) -> ResponseFuture<'_, Option<StoredResponse>>;
    fn put(&self, response: StoredResponse, condition: WriteCondition) -> ResponseFuture<'_, ()>;
    fn transition(
        &self,
        id: &ResponseId,
        allowed: &[ResponseStatus],
        next: ResponseStatus,
    ) -> ResponseFuture<'_, StoredResponse>;
    // delete / append_event / events
}
```

SDK 需要提供：

- 后端 adapter（Redis/Postgres/DynamoDB/托管存储），实现原子条件写与 TTL/租户隔离。
- 配置加载与 `ResponsesExtensions` 构造，调用 `validate` 后交给 frontdoor。
- 可选的工具执行器扩展（`ToolServer` 等价面）：输入 tool call、输出 output items/usage；该契约尚未定稿，本项不预留空 trait。
- 观测：store 延迟、命中、驱逐、失败原因；不能记录 prompt/output payload。
- 版本策略：`RESPONSES_SPI_VERSION` 精确匹配；`StoredResponse.payload_version` 由 frontdoor 校验；破坏性变更提升版本并拒绝旧实现。

## 验证

- `cargo test -p infer-frontdoor --lib` 覆盖无 store 的 501、默认 store、GET、链式输入、store=false、后台轮询、取消、流式落库与无 store 的断连清理；状态 store 的断连保底写由 `StoreLease` 实现，外部后端接入时应补真实网络断开测试。
- `cargo test -p infer-spi --lib` 覆盖 ID/状态/记录校验/扩展版本。
- 参考 store 单测覆盖条件写、迁移冲突、LRU、活动保护、字节上限、TTL、事件顺序与容量。
- 未做真实 Redis/Postgres 或跨进程重启验收；外部 SDK 接入时必须补后端级原子性、故障恢复和并发取消测试。
