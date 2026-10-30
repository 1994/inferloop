# OpenAI 兼容接口

`serve --package` 在模型包包含 `tokenizer.json` 时启用 `/v1` 兼容路由；Chat Completions 还需要包内聊天模板。接口直接调用本地推理引擎，不需要 OpenAI 账号或 API key。

响应格式参考官方 [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) 与 [Completions](https://developers.openai.com/api/reference/resources/completions/methods/create)，但兼容范围是明确限定的。

已交付长度预算、有效配置读回、typed 工具历史、工具闭环与参数增量输出，以及非推测解码中的 `required`、指定 function、`strict` schema 生成约束。设计与后续推测组合见 [独立方案](../plans/serving/agent-api.md)。扩展按实际使用场景选择，以下只说明当前能力。

## 路由

| 方法与路径 | 行为 |
|---|---|
| `GET /v1/models` | 返回当前 engine 的单个模型，ID 为生效的部署名 |
| `GET /v1/models/{model}` | 查询部署名对应的模型；不存在时返回 OpenAI 格式的 404 |
| `POST /v1/chat/completions` | 用包内模板编码 messages，返回 `chat.completion` 或 SSE |
| `POST /v1/completions` | 编码单个字符串 prompt，返回 `text_completion` 或 SSE |
| `POST /v1/decisions` | 用同一生成模型回答 predicate / choice / score 问题，见「决策」 |

部署名默认是内部 ModelId 的十进制字符串，可用 `--served-model-name` 覆盖；样例模型默认名为 `"1"`。请求未加载的模型返回 404。包内没有 tokenizer 时上述路由不存在，原生 token API 仍可使用。

## 请求参数

| 参数 | 约定 |
|---|---|
| `model` | 必填字符串，必须匹配 `/v1/models` 返回的部署名 |
| `messages` | 仅 chat，1..256 条；role 为 system/user/assistant/tool，见下节 |
| `prompt` | 仅 completions，非空字符串 |
| `max_tokens` | 正整数，默认 16 |
| `max_completion_tokens` | 仅 chat，与 `max_tokens` 互斥 |
| `temperature` | 0..2；省略时由模型生成配置与模式预设解析，0 为 greedy |
| `seed` | 非负 u64；省略时使用解析后的默认值 |
| `n` | 仅支持 1 |
| `stream` | 布尔值；`true` 返回 SSE，见「流式」 |
| `stream_options` | 仅 `stream=true` 时接受；`include_usage` 追加终末 usage |
| `top_p` | (0,1]；省略时使用解析后的默认值 |
| `top_k` | 非负整数，0 禁用过滤；省略时使用模型默认 |
| `min_p` | 0..1；省略时使用解析后的默认值 |
| `presence_penalty` | -2..2；省略时使用解析后的默认值 |
| `repetition_penalty` | 正有限数；省略时使用解析后的默认值 |
| `enable_thinking` | 布尔值，控制模板与受支持模型的模式预设 |
| `tools` | 仅 chat，至多 128 个 function 工具，名称唯一，1..64 个 ASCII 字母、数字、下划线或短横线 |
| `tool_choice` | `"none"`、`"auto"`、`"required"` 或指定 function；指定未声明的 function、缺少工具的 required 或错误的 type 返回 400 |
| `parallel_tool_calls` | 默认 `true`；`false` 通过生成约束限制至多一次调用 |
| `mm_processor_kwargs` | 仅 chat 的图片输入；支持 resize 和像素范围选项，见[视觉服务路径](vision.md#服务路径) |

可选字段允许为 `null`，按未提供处理。服务已读取包内 `generation_config.json`，并从 generation config / model config 解析 EOS；识别到的模型还会应用模式预设，显式请求字段最后覆盖。模型包配置了 EOS 时会参与停止判断，未配置时不凭空增加 EOS。请求级 `eos_token` 覆盖使用原生 API。

### 消息与工具历史

`content` 可以是字符串、`null`，或 `{"type":"text","text":...}` 组成的列表；其他 content part 类型返回 501。`null` 按空文本处理，这是只有工具调用的 assistant 消息的常见形式。

assistant 消息可以带 `tool_calls`（`null` 按空列表处理），每项为 `{"id":..., "type":"function", "function":{"name":..., "arguments":...}}`，`arguments` 是该调用收到或产生的原始 JSON 文本。`tool` 消息必须带 `tool_call_id`，并关联到同一段历史中更早的一次 assistant 调用。id 重复、关联不存在的历史、或把 `tool_calls` / `tool_call_id` 放错 role 都返回 400。历史可以包含本次未声明的工具调用，本次声明集合只约束新产生的调用。

### 严格工具调用

遵循 [OpenAI 工具选择语义](https://developers.openai.com/api/docs/guides/function-calling)：required 至少一次调用，指定 function 恰好一次指定调用；parallel=false 至多一次。

`strict: true`、`required`、指定 function 或 `parallel_tool_calls: false` 会在 CPU 准备池中编译 llguidance 1.9.1 grammar，逐 token 在采样前屏蔽非法候选。greedy 与随机采样共用约束；top-k/top-p/min-p 在合法集合上计算。普通 `auto` 且无严格工具的请求保留原有路径。

严格 schema 支持 object、array、string、number、integer、boolean、null、nullable type 数组、enum 和嵌套 anyOf。根必须为 object；严格 object 必须设置 `additionalProperties: false`，并在 required 中列出全部 properties。支持 description/title；`$ref`、`$defs`、pattern、数值/长度边界等未实现关键字返回 501，不会忽略。function/parameter 方言不支持根 enum/anyOf，参数名不可有首尾空白、尖括号或换行。

要求 byte-level decoder、模板中的受支持工具方言和 tokenizer 特殊 EOS。约束与推测解码组合返回 501；活动 matcher 的 checkpoint/replay 也显式拒绝。长度上限仍可截断调用，终末原因保留 `length`，不能当作完整调用执行。

资源边界：工具声明合计不超过 64 KiB，schema 深度 16、每个 schema 256 个节点；每个请求预留 32 MiB 主机信用，受约束输出最多 4096 字节（含 reasoning 与标签），超过后返回生成错误。共享缓存至多保留 4 个已编译 grammar，每个请求独立 matcher；grammar 大小、lexer 状态和单步计算也有上限。严格模式保证受支持 schema 的格式，不保证参数语义或工具选择正确。

### 长度预算

`--max-model-len` 是单序列总上下文（prompt 与 output 合计），`--max-output-tokens` 是服务输出 cap，`--max-num-seqs` 是并发序列 cap。三者与输入 cap 分别表达，实际生效值和来源可从 `/native/v1/runtime` 的 `lengths` 字段读回。准入检查 `P <= input cap`、`G <= output cap`、`P + G <= L`；显式输出预算超过服务 cap 返回 400，不会静默削短 prompt，显式配置超出模型上限则启动失败。省略输出预算时使用有界默认值，再受服务 cap 限制。

不要把当前输出默认 16 当作上下文上限，也不要假设省略采样参数一定得到 greedy。

不支持的字段（如 `stop`、`response_format`、`logprobs`）会被拒绝，不会静默忽略。多模态输入的范围见[视觉服务路径](vision.md#服务路径)。Responses、Embeddings、鉴权、服务端存储均未实现。

## 决策

`POST /v1/decisions` 实现 [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions) 的文本子集。它不需要单独的决策模型或额外 head：服务生成的同一份权重也回答决策，`model` 与其他路由一样填部署名。

```json
{
  "model": "1",
  "input": "I was charged twice for my order.",
  "questions": [
    {"type": "predicate", "name": "refund", "instructions": "Does the customer ask for a refund?"},
    {"type": "choice", "name": "department", "instructions": "Which department should handle this?",
     "choices": [{"value": "billing", "description": "Payments and refunds."}, {"value": "other"}]},
    {"type": "score", "name": "severity", "instructions": "How severe is this issue?",
     "levels": [{"label": "Cosmetic"}, {"label": "Workaround available"}, {"label": "Fully blocked"}]}
  ]
}
```

每个问题是一次独立的 prompt 读取：共享 `input` 在前，问题与按 A、B、C… 标注的选项在后，prompt 停在模型回复开始处（thinking 关闭）。引擎只做 prefill，读取最后位置的 LM logits，在选项字母上归一化，不生成 token。问题按部署的并发宽度分批提交，答案顺序与 `questions` 一致。

| 类型 | 选项 | 返回 |
|---|---|---|
| `predicate` | `false`、`true` 两项 | `probability`：`true` 的概率 |
| `choice` | 2..26 个 `choices`，`value` 为字符串或布尔值，类型不同即不同值 | `choice`、各值 `probabilities`、`confidence` |
| `score` | 2..10 个由低到高的 `levels` | `score`：以 0 起始的档位序号按概率加权的期望，各档 `probabilities`、`confidence` |

`confidence` 是胜出选项在整个词表上的概率（选项内概率 × 选项总质量），模型更想输出选项以外的 token 时它会偏低；它未按 OpenAI 模型校准，阈值需用自己的标注数据确定。`name` 原样回显，未命名为 `null`。

`input` 可以是字符串，或依次包含 user / developer / system / assistant 消息、`function_call` / `function_call_output`、`custom_tool_call` / `custom_tool_call_output` 的数组。developer 与 system 消息合并为聊天的 system 轮；只有 user 消息时按原文拼接，否则以「User: / Assistant: / Function call name (call_id):」等前缀保留说话方。工具输出必须对应更早的同类调用。`input_json` 只能出现在 user 消息中，以紧凑 JSON 文本呈现，限制为 32 层、10,000 个值、256 KiB。

1 期范围与限制：

- 1..400 个问题；choice 最多 26 个选项（字母标签），协议允许的更多选项返回 400。
- 图片（`input` 中或问题 `instructions` 中的 `input_image`）与 `item_reference` 返回 501。
- 多个问题共享的前缀（system 轮与 `input`）只 prefill 一次：服务把共享 token 前缀作为请求的 `prefix_boundary`，第一个问题在该处保留快照，其余问题从快照恢复后只 prefill 自己的问题文本，`usage.input_tokens_details.cached_tokens` 报告这样恢复的 token 数。RTX 5090 上 Qwen3.8-27B 以约 2,500 token 的输入问 64 个问题，耗时从 60 s 降到 2.5 s，答案概率不变。快照按 32 token 的 KV 块对齐，在 GPU 前缀缓存预算内按 LRU 淘汰，可以被之后相同前缀的请求复用。
- `prompt_cache_key`、`prompt_cache_options`、`prompt_cache_breakpoint`、`safety_identifier` 会校验后接受，但不改变上述行为；`cache_write_tokens` 恒为 0。
- 从不返回 `refusal` 答案；`output_tokens` 为 0，`input_tokens` 是所有问题 prompt token 之和。
- 包的回复前缀必须结束在 token 边界，且每个字母在回复开头是一个独立 token；否则返回 501。服务按包内模板实际探测，而非按模型名单判断。

## 响应与错误

成功响应包含单个 choice、创建时间和 `usage`。prompt token 数包含实际聊天模板，completion token 数来自完成结果，`finish_reason` 为 `length`、`stop` 或 `tool_calls`。

模型输出按包内模板绑定的方言解析：`Qwen3-VL` 使用 `<tool_call>{"name":...,"arguments":{...}}</tool_call>`，`Qwen3.8-27B` 使用 `<tool_call><function=F><parameter=K>v</parameter></function></tool_call>`。模板未声明受支持方言时，带工具的 `auto` 请求在准入前返回 501。JSON 参数保留原始字节；function 方言在每个参数闭合时转换为 JSON，完整的 function 闭合标签不可省略。

解析不修补模型输出：未闭合的块被扣留，未能解码或未声明的名称在发布调用之前留在 `content` 中作为文本。参数已经流出后才发现格式错误，则终止流并发布错误，不撤回或改写前缀。正常停止时仍有未闭合的结构区域，完整返回报 503，SSE 发布错误；引擎报告长度上限时保留 `length`。声明了工具且模型确实调用时，choice 形如：

```json
{
  "index": 0,
  "message": {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_1_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
      }
    ]
  },
  "finish_reason": "tool_calls",
  "logprobs": null
}
```

调用 id 由请求 ID 与序号组成，流式与完整返回一致。开启 thinking 时 reasoning 文本单独放在 `reasoning_content`，不计入 `content`；模板已在生成前缀打开思考区时，parser 从该状态开始处理新 token。引擎因长度上限停止时 `finish_reason` 仍为 `length`，即使停止前已产生完整调用；调用方应按终末原因决定是否执行。

```json
{
  "error": {
    "message": "model is not served; see /v1/models",
    "type": "invalid_request_error",
    "param": null,
    "code": "model_not_found"
  }
}
```

schema 或输入错误返回 400，模型不存在 404，body 超限 413，容量不足 429，已知未实现选项 501，backend 失败 503，无效 Content-Type 保留 415。body 上限为 2 MiB。

## 流式

`stream=true` 返回 `text/event-stream`，每行 `data:` 一个 JSON chunk，末尾为 `data: [DONE]`。Chat 的首个 chunk 发布 `delta.role`，随后是 `choices[0].delta.content`（completions 为 `choices[0].text`，不包含 `delta`），工具调用以 `delta.tool_calls` 发布，最后是一个带 `finish_reason` 的 chunk；`stream_options.include_usage` 会在终末原因与 `[DONE]` 之间追加一个 `choices` 为空、带 `usage` 的 chunk；开启时其余 chunk 都带 `usage: null`。Chat Completions 和 Completions 均支持此选项。流式错误使用与非流式相同的 `error.message/type/param/code` 结构。

已发布的 `id`、`index` 与 arguments 前缀不会被回写。JSON 方言在名称确定后直接追加已扫描的对象参数字节；字符串包装的 JSON 对象需要等块闭合再解码。`function-parameter` 方言在名称确定时发布调用，在每个参数闭合时追加对应 JSON 字段，function 闭合时追加右括号。无需等整个工具块到齐才收到首段参数。

有效输出的 content、reasoning 与每个调用的 arguments 分别按顺序拼接后，与完整返回相同。usage 按引擎提交的 token 计数，UTF-8 跨 token 缓冲和 SSE 分段不改变计数。客户端只在收到成功的终末 `tool_calls` 后执行调用；部分参数、`length` 或错误不表示可执行成功。

文本流式要求包内 tokenizer 使用 byte-level decoder，才能保证增量解码的前缀稳定；其他 decoder（例如按空格连接 token）会让分段解码与整体解码不一致，这类包请求 `stream=true` 返回 501，而不是发布随后需要更正的字符。

## 实现说明

请求 ID 来自 `RuntimeHandle` 的单调分配器：显式原生请求会推进计数器，路由重建不会重置；混用两套接口时可用 `RuntimeHandle::allocate_request_id` 分配 ID，避免手动 ID 冲突。文本编码与结果解码分别运行在有界 preparation / delivery 池中：流式路径同样把每个已提交 token 送进 delivery 池做增量 detokenize 与解析，兼容层继承原生准入、背压和断连取消机制。

[SDK 验收工具](../../tools/validation/openai-tools.py) 在已启动的服务上验证 `none/auto`、参数前缀、usage、流式一致性及工具结果回传。使用安装了 OpenAI Python SDK 的验证环境运行 `python tools/validation/openai-tools.py --base-url http://127.0.0.1:8080/v1 --output artifacts/openai-tools.json`；SDK 不属于服务生产依赖。
