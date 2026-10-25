# Agent 工具调用与服务配置方案

状态：S1/S2 已交付：基础工具协议、增量 parser、总上下文与有效配置、非推测生成约束均已实施，HTTP 严格工具模式按支持子集开放；真实模型与 GPU 严格工具验收、推测组合仍属后续工作。初始调研源码基点为 `7c79d99`，日期为 2026 年 10 月 9 日；基础实现基于 `3a188f4`，严格工具约束随 `a1a11fa` 交付。当前公开能力以 [OpenAI 接口指南](../../guides/openai-api.md) 为准。

整体排期与任务状态见 [路线图](../README.md) 的 S1、S2。本文维护本项设计与验收。性能对比的 release、配置、统计和证据规则沿用 [统一基线方案](../performance/baseline.md)，不另设一套门槛。

当前已实现 typed 工具历史、结果关联、`none/auto`、文本 SSE，以及 JSON 与 function/parameter 两种方言的增量输出。`--max-model-len`、输出与序列 cap 及有效配置读回已有共同解析路径。`infer-agent` 继续负责引擎诊断；外部工具由调用方执行。`required`、指定工具、`strict` schema 与禁止并行调用使用 llguidance 1.9.1；不支持的 schema 与推测组合显式拒绝。编译 grammar 和 tokenizer trie 按模型缓存，请求独占 matcher 随 CPU 输出任务移交，正常逐 token 路径不加共享锁；未请求约束时保留 backend greedy 快速路径。

外部源码以 vLLM `v0.31.0`（`db9527a46873454610df6dbedf79a36d6bf1a7f6`）、SGLang `59eb71a831f6b87babc2400737c7faabed5ee01c` 为调研参照。`latest` 文档用于理解能力范围；实施和验收固定实际参考版本。这些实现提供机制与反例，不决定本项目的参数集合和默认值。

## 能力取舍原则

本项围绕三个实际场景选择能力：常用 SDK 的 Agent 工具调用闭环、单设备部署的上下文与资源控制、可复算的性能测量。每个新增参数必须对应具体调用者或运维动作，语义清楚、有实际执行路径，并能验证效果；不能因为 vLLM 存在同名字段就加入待办。

对外保留常用 OpenAI 消息、工具调用和流式协议，使调用方能接入。内部采用一套 typed 配置和规范化契约；模型方言优先由 package/profile 绑定，后端几何和执行策略优先自动推导。只有实际需要人工控制时才增加覆盖入口。

性能基线继续对齐实际工作量、质量条件与资源约束；参数名字、默认值和配置形式允许不同。已有参数不因本次方案改写而删除或改变语义，新方案也不承担完整 vLLM CLI/API 兼容。

## 当前实现与缺口

| 范围 | 已有实现 | 需要补齐 |
|---|---|---|
| Agent 控制面 | [infer-agent](../../architecture/agent.md) 提供 JSON-RPC 诊断、控制和实验 | 不承担模型生成内容的解析；不能把模型输出直接分发到 `AgentService::register` |
| 工具提示词 | [TextAssets / ChatOptions](../../../crates/model/package/src/input/text.rs) 编码工具定义、typed assistant 调用历史及关联结果；模板绑定输出方言 | 后续方言需补真实模板、tokenizer 与输出样本 |
| OpenAI 请求 | [GenerationRequest](../../../crates/service/frontdoor/src/http/openai/request.rs) 支持文本、nullable assistant content、工具历史、none/auto、required/named/strict、parallel=false 与 SSE；未知字段拒绝 | 更完整的 schema 及推测组合 |
| OpenAI 输出 | [execution](../../../crates/service/frontdoor/src/http/openai/execution.rs) 与 [stream](../../../crates/service/frontdoor/src/http/openai/stream.rs) 共用 parser；支持 reasoning、tool_calls 与参数 deltas | 新方言或约束接入时复用当前事件契约 |
| 结构化生成 | [IR FeaturePlan](../../../crates/foundation/ir/src/workload.rs) 有 grammar 元数据；Decision workload 用于评分/分类 | 工具约束通过独立 matcher 接入，通用 response_format 尚未开放；不能把 Decision 当作 JSON Schema 生成能力 |
| 模型上下文 | [Qwen provider](../../../crates/model/package/src/providers/qwen.rs) 导入 `max_sequence`；服务、workload 与 runtime 共用 `P+G<=L` 预算 | 后续扩展上下文能力另行验收 |
| 输入上限 | [RuntimeConfig](../../../crates/engine/runtime/src/config.rs) 将输入 cap 与总上下文分别表达 | 保持各入口共同规则 |
| 输出上限 | OpenAI 两种输出预算字段互斥，统一为 `max_new_tokens`，受服务输出 cap 与剩余上下文限制 | 后续能力保持来源与冲突规则 |
| 服务配置 | [CLI](../../../crates/service/cli/src/arguments.rs) 支持总上下文、输出/序列 cap、部署名；有效值和来源可读回 | 需要新覆盖时先确定真实调用场景 |

当前采样并非始终 greedy：[GenerationDefaults](../../../crates/model/package/src/generation/mod.rs) 已加载包内 generation config / EOS，应用识别到的模型模式预设，再覆盖请求参数。基线需要保存最终参数和来源，不能只比较客户端是否传了同名字段。

## 严格工具实施进度

2026-10-23 开始接入生成约束。首步在 `infer-workloads` 增加借用的词表 bitset
`TokenMask`、`sample_constrained` 与 `probabilities_constrained`：合法集合先于
top-k/top-p/min-p 过滤；penalty 仍按原始 token ID 处理；greedy 与随机采样均不能选中
被禁止的 token（包括 EOS）。空合法集合、词表尺寸不匹配和非有限 logits 显式失败，
不回退无约束采样。全允许 mask 保持原来的确定性采样轨迹；已有调用者继续走无约束路径。

现已接通 `infer-models` 的 llguidance grammar 编译、模型 tokenizer 对齐与有界缓存，`infer-spi` 的请求 matcher 契约，以及 preparation/output worker 的所有权移交。严格工具请求禁用绕过 mask 的 backend token 快速路径；未实现的推测组合、活动 matcher checkpoint/replay 显式拒绝。

HTTP 完整与 SSE 路径共用这些约束；CPU 测试覆盖 required/named/strict、调用数量、嵌套/nullable schema、标记字符串、EOS、长度截断与缓存状态隔离。支持范围和资源上限见接口指南。尚未进行真实模型或 GPU 严格工具验收，不把协议测试当作设备性能证明。

## 可以参考哪些实现

| 参考 | 已核实的机制 | 对本项目的价值与边界 |
|---|---|---|
| vLLM Rust parser | [ToolParser](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/rust/src/parser/src/tool/mod.rs) 提供每请求状态、增量 feed、按序 text/tool 事件、finish/reset；[基准](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/rust/src/parser/benches/qwen3_coder.rs) 覆盖 parser 创建/复用、短 chunk、长文本和工具参数 | 最贴近当前 Rust 服务。借鉴状态机和测量方法；不能凭 Rust 语言或基准文件宣称最快。该 [Qwen Coder 实现](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/rust/src/parser/src/tool/qwen_coder.rs) 要等完整 tool block 才发 arguments，不代表所有参数都能逐 token 流出 |
| SGLang detector | [FunctionCallParser](https://github.com/sgl-project/sglang/blob/59eb71a831f6b87babc2400737c7faabed5ee01c/python/sglang/srt/function_call/function_call_parser.py) 以模型方言选择 detector，分别提供完整与流式解析 | 借鉴模型兼容矩阵和解析样本；这里核实的是 Python detector，未测得可用于本项目排序的性能数字 |
| Infernix | 本地基点 `fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de` 的 [tool contract](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/src/models/qwen3_5/frontend/tool_contract.h) 与 [output parser](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/src/models/qwen3_5/frontend/tool_call_parser.cpp) 共享规则，处理工具结果历史、标记和代码围栏 | 借鉴 contract 与 grammar/parser 共用规则；其工具区域主要在 finish 后解析，不作为低延迟 arguments delta 的证明 |
| XGrammar | [结构标签](https://xgrammar.mlc.ai/docs/latest/structural_tag/tool_calling_and_reasoning.html) 可表达普通文本、reasoning、工具调用与 schema 的组合；[论文](https://arxiv.org/abs/2411.15100) 研究约束编译、缓存及 CPU/GPU 重叠 | 属于生成时的 token 约束，不是生成后的 tool parser。适合高复用 schema 的候选评估；C++ 集成、冷编译和 tokenizer 对齐需要计入完整成本 |
| llguidance | [官方实现](https://github.com/guidance-ai/llguidance) 为 Rust 约束引擎，结合 lexer、Earley parser 与 token trie | Rust 接入值得优先试验。作者报告 128K tokenizer 下平均约 50µs/CPU 核的 token mask 计算；这是其 maskbench 条件的 mask 耗时，不是 tool parser、HTTP 或本项目 248K 词表的结果，不能直接换算服务收益 |

先采用纯 Rust 的增量输出解析设计；生成约束后续在 llguidance 与 XGrammar 之间用实际模型、schema 和冷/热 workload 选型。vLLM parser crate 还有 tokenizer、grammar 等依赖，不直接把整个 vLLM workspace 引入项目；依赖、MSRV、制品体积和许可按 [依赖方案](../engineering/build-and-dependencies.md) 评估。

parser 名称不能替代方言验证。同属 Qwen 的模型、模板可能使用 JSON 包装或 function/parameter 标记。每个支持组合绑定模型 profile、模板指纹、tokenizer 指纹、标记 token 与真实输出样本；当前目标模型先完成这项确认，再选择适配器。

## 分层与公共契约

| 层 | 拟议职责 |
|---|---|
| model/package | 模型输出方言、特殊标记、模板与工具历史的序列化规则；保持与设备无关 |
| service/frontdoor 的文本输出路径 | typed OpenAI 协议、增量 detokenizer、reasoning/tool parser、SSE 编码；每请求独立可变状态 |
| common workloads / sampling 与 SPI | 规范化的生成约束、token 可接受集合、推进/checkpoint/rollback 协议；不引用 OpenAI HTTP 类型 |
| CUDA / Metal backend | 应用 mask、设备存储与 fence；报告执行能力及资源成本 |
| 调用方 Agent | 执行工具、处理工具错误、把关联结果回传、决定下一轮；服务返回结构化调用，不自动运行外部工具 |

tool parser 与 reasoning parser 都在 CPU，不绑定 CUDA。生成约束的规则与状态也独立于设备，mask 的实际应用可以在 GPU。已有诊断 Agent 保持其控制面职责；不为解析几个辅助类型新增一个独立 crate。

当前 [StreamEvent](../../../crates/model/package/src/input/tool/types.rs) 契约为有序的 `Text`、`Reasoning`、`CallStart(index, name)`、`CallArguments(index, arguments)`、`CallEnd(index)`；`Call` 保留完整对象的兼容发布。parser 只接收新提交文本，HTTP 层分配 ID 并解释结束原因；不能修改已发布的前缀。实际提示词只渲染与编码一次，并将生成前缀是否已打开思考区传给两种响应路径，避免正文泄漏推理。方言、扫描与参数转换留在 package，parser 不引用 HTTP 或 backend。目标 tokenizer 的工具/思考标记均为非 special token，当前 skip-special 解码保留它们；新增 tokenizer 需重新核对标记身份。

共享不可变工具 schema、模板和已编译约束；detokenizer、parser、grammar matcher、调用 index 与发送游标均归单个请求所有。串行处理同一请求的片段，不同请求可由有界 CPU 池并行；保留现有取消、内存预算和背压。reasoning 与 tool 区域通过统一 span 所有权区分，避免两个 parser 重复消费同一标记。

## 工具调用闭环与兼容范围

首批交付文本 function tools、单 choice 的 Chat Completions、none/auto 模式，完整返回与 SSE 使用同一 parser 和组装契约。请求消息增加 typed variants：普通文本消息、assistant 的 nullable content 与 `tool_calls`、带 `tool_call_id` 的 tool 结果。模板必须能重放一整轮 assistant 调用和对应结果，不只支持首轮 tools 注入。下面的 required/named/strict 语义供实际需要这些模式时使用，不全部列为首批交付条件。

闭环验收顺序为：客户端传工具定义 → 模型返回 name、arguments 与稳定 ID → 客户端执行 → tool 消息关联 ID 回传 → 模型继续回答或再次调用。新生成的名称必须属于本次声明工具，ID 唯一且结果关联正确；历史可以包含过去工具集合中的调用，不能因本次集合变化而改写历史，也不能把任意 tool 文本当关联结果。

| 请求模式 | 必须满足的语义 | 放行方式 |
|---|---|---|
| `tool_choice=none` | 允许普通回答，不发布结构化调用 | 模板和解析策略共同落实；不能仅传个字段便认为禁用生效 |
| `auto` | 模型可回答或调用声明工具 | 基础 parser + 对应模板；schema 合法性另行报告，不能默认承诺 strict |
| `required` | 至少一个调用 | 需要该方言的生成约束或等价可证明机制；只有后解析时明确拒绝此模式 |
| 指定 function | 调用指定名称 | 生成约束限制名称与结构；不通过改写模型已生成的名字伪造满足条件 |
| function `strict=true` | 按声明 schema 约束参数 | 只有 schema、parser 方言与约束语言一致时启用；不支持的 schema 关键字应拒绝 |
| `parallel_tool_calls=false` | 至多一个调用 | 生成约束限制次数，或该模式明确不支持；不能截掉第二个调用后声称满足契约 |

vLLM 将调用义务、grammar 激活和 schema 严格程度分别处理，且受具体 parser 能力限制；`auto` 在支持的严格配置中也可以启用约束，不能固定理解为完全无约束。[工具调用说明](https://docs.vllm.ai/en/latest/features/tool_calling/) 的行为须结合固定版本源码验收。

SSE 的 tool index 与 ID 一旦发出就稳定，function name 只在确定后发布；arguments 是可追加的字符串。JSON 对象按新扫描字节发送，字符串包装对象在块闭合时解码；function 方言按参数边界转换并追加 JSON。正常停止时结构未闭合是 backend 错误，长度截断保留 `length`；已发布调用的后续校验失败不能降级成文本。

流式实现先发送 role，随后发送 content/reasoning/tool deltas，再发送结束原因；支持 `stream_options.include_usage` 时发送终末 usage 和 `[DONE]`。正常生成完成且含完整调用时使用 `tool_calls`；引擎因长度上限停止仍保留 `length`，不因 parser 恰好收到闭合括号而改写原因。流式中已发出的部分调用不可撤回，SDK 集成必须按终末原因判断是否执行；非流式保留错误/不完整状态，不自行补括号、修参数或伪造成功。取消和 backend 失败按终末错误契约处理。

首批不扩张到 Responses、内置执行器、MCP 执行、custom tools、多 choice 或多模态工具内容。这些能力可以后续单独定义，不能通过宽松反序列化默认“支持”。

## Max length 的统一语义

采用 vLLM 常用名称 `--max-model-len` 表达单序列总上下文，包含 prompt 与 output；`max_tokens` / `max_completion_tokens` 表达请求输出上限。[vLLM CLI](https://docs.vllm.ai/en/latest/cli/serve/) 将这几类预算分别定义；HF 的 `max_length` 不是 OpenAI Chat 请求字段，不新增一个含义模糊的请求别名。

拟议 `ResolvedLengthLimits` 在模型 metadata 解析后、backend 资源规划前解析一次：

- `M`：模型包经 provider 解析并验证的可支持上下文上限。
- `L`：服务生效的 `--max-model-len`，首批要求正整数，省略时来自模型上限，`L <= M`。
- `P`：最终编码的输入 token 数，包含模板、工具定义、调用历史、结果及实际支持的媒体占位。
- `G`：请求解析后的输出 token 上限；reasoning、工具标记、参数等实际生成 token 都计入输出预算。

准入检查 `P <= input_cap`、`G <= output_cap`、`P + G <= L`，采用 checked arithmetic。输入上限、服务输出上限、总上下文与物理容量分别报告；先检查轻量请求/字节预算，再完成有界模板编码和 token 检查，最后申请昂贵输出/state 资源。不能移除当前已经存在的 input+output 检查；参照版本的 [TokenizeParams](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/renderers/params.py) 也按总预算减去请求输出预算校验输入。

显式输出预算超限返回参数错误，不静默削短请求；省略输出预算时使用明确、有界的服务默认值，再受服务 cap 与剩余上下文限制，并公开最终值。当前默认 16 是否调整，由 Agent 任务的输出需求决定，调整时给出迁移说明；不为了复刻外部默认值而自动用满剩余上下文。

内部只使用一个 `max_new_tokens`。对外已有 `max_tokens` / `max_completion_tokens` 归一到它，两者同时传入沿用当前拒绝规则，不新增别名或照搬 vLLM 的隐式优先级。固定版本的 [外部协议](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/entrypoints/openai/chat_completion/protocol.py) 仅作为行为参照；SDK 测试验证本项目公开的字段合同。

不通过修改 ModelIr 的架构上限实现部署 cap。解析结果必须传给 TextAssets、所有请求入口、workload plan、runtime 准入与 backend 容量规划；checkpoint 恢复也校验当前限制。KV、recurrent 状态、graph arena、推测 scratch 和 seed/bonus 所需临时空间单独报价，不能把 scratch 隐藏到用户输出 token 数里。

首批不接受超出模型能力的上下文扩展。省略配置时使用模型上限并验证容量能否兑现；资源不足时给出可调整的预算错误，显式整数配置无法兑现则启动失败，不能暗中收缩。自动缩小上下文只有在容量报价可信且存在实际需求时另行设计，不引入 `-1` 这类特殊值或另一套自动探测开关；若实际需要友好的单位输入，再单独定义一致的单位规则。聊天历史不自动截断，尤其不能丢掉工具调用或结果中的一半。

`--max-num-batched-tokens` 是每步计算预算，`--max-num-seqs` 是运行序列限制，`--block-size` 是 KV 分块设置；它们都不作为 max length 的替代入口。

## 首批需要的服务配置

本节是选定能力，不是 vLLM 参数总表；新 CLI 名称只在部署确实需要时确定，已有配置文件能够表达的内容不重复增加入口。

| 能力 | 实际用途 | 拟议取舍 |
|---|---|---|
| 总上下文 `--max-model-len` | 限制 Agent 多轮历史和单请求资源 | 新增统一入口，按上一节落实 |
| 运行序列 cap / step token budget | 控制并发容量、延迟和基线条件 | 使用现有 RuntimeConfig 与 batch budget；必要的 CLI 入口映射同一配置，公开实际 slot / graph 宽度 |
| 显存预算 | 约束单卡驻留量和资源准入 | 保留已有显存比例语义，校验并报告实际字节上限；不为了跟随 vLLM 改写 0 的现有含义 |
| 可控的 prefix cache | 多轮 Agent 复用，以及冷/热基线 | 有明确启用/禁用策略，读回复用量；配置形式服从本项目，禁用必须真实生效 |
| 对外模型名称 | SDK 使用稳定名称而不依赖内部数字 ID | 提供一个服务名称，保持内部 ModelId；不先扩展多别名、远端拉取与 revision 参数体系 |
| 采样默认与覆盖 | 解释模型行为、固定实验条件 | package defaults、模型模式、服务覆盖和请求覆盖由统一 resolver 解析并记录来源；服务覆盖使用 typed 配置，不新增 `auto/vllm/path` 三套默认模式 |
| 模板与输出方言 | 正确呈现工具历史、thinking 与调用输出 | package/profile 默认绑定；模板编译缓存，特殊部署按需提供显式覆盖；不要求用户逐项开启多个 parser 开关 |

已有 `--listen` 足以描述监听地址，不为模仿部署命令新增 `--host` / `--port` 别名。已有 block size 和 GPU blocks override 保留校验与观测，作为高级配置使用，不列为 Agent 使用流程的一部分。配置读回反映最终生效值与来源，不靠命令长得相同判断两边等价。

## 首批需要的请求协议

| 能力 | 实际用途 | 拟议取舍 |
|---|---|---|
| `model` / typed `messages` | SDK 会话与工具结果回传 | 单模型、文本消息，支持 assistant tool_calls 和 tool_call_id |
| 输出 token 上限 | 约束单轮生成与多轮成本 | 现有两个外部字段归一到一个内部预算；字段冲突显式报错 |
| 已有采样字段与 `seed` | 控制生成及复现实验 | 保留当前能力，解析最终值；top_k 禁用仍用 0，seed 保持非负 u64，不引入负数特殊值 |
| `tools` / `tool_choice` / `parallel_tool_calls` | Agent 工具选择与调用数量 | 先支持 none/auto；受约束模式按实际需求和能力开放，不支持组合显式报错 |
| `stream` / 终末 usage | 低延迟展示与调用成本统计 | 文本、reasoning、tool delta 同源；保证终末原因、usage、背压和取消 |
| `stop` | 调用方指定终止边界 | 按需增加常用字符串/字符串列表，跨 chunk 匹配、有界保留后缀；不能只在最终文本裁剪 |
| `enable_thinking` | 实际目标模型的思考模式 | 保留明确的请求控制；不增加任意模板 kwargs 或多档 reasoning_effort 映射 |

常用 SDK 的消息与 delta 格式参照 [Chat protocol](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/entrypoints/openai/chat_completion/protocol.py)，以真实客户端闭环测试验收。协议适配留在 adapter 层，不能使 HTTP 参数类型渗入公共采样或 backend；影响生成语义却尚未实现的字段明确拒绝，不能静默忽略。

## 有实际场景再引入的能力

| 能力 | 引入条件 | 对外形式 |
|---|---|---|
| required / 指定工具 / strict schema / JSON 输出约束 | 调用方需要调用义务或 schema 保证，或实测格式错误导致明显重试 | 公共生成约束 + 常用工具/response_format 协议；backend 选择优先自动，不搬用 vLLM 的约束配置集合 |
| MTP / DFlash2 选择与组合 | 公共 SPI 接入及资源验收完成 | 本项目 typed 推测策略，仅包含已实现 provider 与真实组合；不复刻整个 speculative-config JSON |
| 精度或量化覆盖 | 已有对应执行能力与独立质量/性能证据 | package 精度计划与受控覆盖；不给未实现的 dtype / KV / quantization 值先建通用参数 |
| logprobs / logit bias / 新采样策略 | 评分、诊断或实际 Agent 明确需要，且成本可测 | 只增加所需 sampler/输出合同，避免无调用方的 token 读回 |
| eager / chunked prefill 的人工选择 | 故障诊断或 profile 证明需要，并存在可执行路径 | 本项目执行策略/诊断配置；默认按能力推导，不接收无效果的开关 |

这些条目是设计储备，不是自动进入排期的兼容待办。多设备并行、多 choice、批量 prompt、任意请求级模板、隐式 prompt 截断、各种 stop/EOS 组合和外部框架历史别名均不因 vLLM 支持而引入；已有原生诊断能力继续服务自己的场景。固定长度基准的停止策略在测量合同中处理，不要求将所有诊断旋钮开放到 Agent API。

## 性能设计与预期收益

当前没有这条完整路径的性能基线，不填写加速倍数。增量输出 parser 的目标是避免累计重解析的 CPU 成本，提前提供工具调用并改善高并发尾延迟；收益幅度待测，它本身不会让 CUDA GEMM 或模型 forward 加速。生成约束可能减少无效 JSON 与重试，但也增加编译、mask 和同步成本，要分别测量格式成功率与任务正确率。

1. **只处理增量。** 增量 detokenizer 消费已提交 token，处理 UTF-8 和 tokenizer 边界；不在每个 token 上解码整个历史或重新解析累计 JSON。验证目标 tokenizer 的稳定前缀，只保留必要后缀；无法证明有界增量解码的方言/decoder 明确记录退化成本或拒绝流式，不能发布后来需要回写的字符。tool 标记保留语义身份，MTP 一次发布多个 token 时按有序批次 feed。
2. **状态机与有界内存。** marker 扫描、字符串转义、嵌套深度和当前调用状态持续保存；约束单请求工具数、参数字节、深度与暂存量。XML 转 JSON 的延迟与必要缓冲按方言计入预算。普通长文本、标记长前缀和恶意碎片输入都测复杂度，避免累计重扫导致平方增长。
3. **模板/schema 编译缓存。** 模板只编译一次；schema/grammar cache key 包含 tokenizer/词表、模板、方言、schema、strict 模式和 backend 版本。编译缓存有界，区分冷请求和命中请求；cache 只共享不可变对象，matcher 不共享。可评估 `encode_fast` 避免无用 offsets，但先验证 token ID 一致性。
4. **有界 CPU 并行。** preparation 与 delivery 的排队、编码、解析、写 socket 分别计时；[当前 delivery](../../../crates/service/frontdoor/src/actor/startup.rs) 只有一个 worker，工具/文本流式接入后先 profile，再决定 worker 数。每请求保持顺序，并发扩展要服从统一 CPU 与内存预算，不在异步 I/O executor 上做无界解析或编译。
5. **约束成本完整核算。** 比较 llguidance 与 XGrammar 的冷编译、缓存命中、每 token mask、复制/FFI、GPU 应用和 P99。mask 使用稳定存储与正确 fence；不能为了 CPU/GPU 重叠使用尚未就绪的 mask。词表大小、复杂 schema、长参数和多调用是必测条件。

XGrammar 的 token marker 模式要求标记能由 tokenizer 的特定 token 表示；字符串模式可能允许多个普通 token 拼出同一标记。parser 与 grammar 必须识别同一语言，不能让 grammar 接受普通 token 拼写，而 parser 只认一个特殊 ID，或反过来。

后解析只改变响应组织，不改变 target 权重；生成约束会改变允许的输出分布，不能宣称等价于完全自由生成。schema 正确也不保证工具选择或参数语义正确，端到端任务质量需要独立验收。

## 与 MTP、DFlash2 的组合

tool/reasoning parser 只消费共同提交点之后的新 token。草稿、被拒绝 token、重复 seed 和尚未提交的 bonus 都不能进入输出 parser；因此通常无需让输出 parser 跟随每条草稿回滚。提交点沿用 [公共 SPI](../speculation/spi.md) 与 [选择/组合方案](../speculation/composition.md)。

生成约束则位于采样过程内部：每个候选前缀有独立 matcher 状态，支持 fork/checkpoint、推进、接受前缀提交和拒绝回滚。验证 target 时必须使用候选前缀对应的合法集合，不能用一张 mask 覆盖整个 verify block；随机推测的接受/残差公式必须使用各自实际受约束的条件分布 `p` 与 `q`。

先完成非推测 greedy 约束与 CPU 状态对照，再接 MTP / DFlash2 的线性 verifier。未支持的“约束 + 推测”组合明确拒绝，或在已声明策略下回退普通 decode，并报告实际算法和原因；不能参数显示已开 MTP、实际关闭却仍作为 MTP 基线。纯后解析的 auto 模式无需因此全部关闭推测。上述 grammar 状态不同于 [recurrent 状态重放](../speculation/state-replay.md)，各自有提交/回滚契约，不能混成 CUDA 私有 parser。

## 验收与测量

测试目录和正式入口沿用 [测试组织方案](../engineering/tests.md)。下面补充本项场景，不另建一个只放单个 test 的目录或新的测试 backend。

2026 年 10 月 11 日的基础闭环验收使用 OpenAI Python SDK 3.24.0，在 45 上分别运行 Qwen3-VL 2B 与 Qwen3.8 27B NVFP4。两种方言均通过工具定义、参数 deltas 拼接、关联结果回传、最终回答及 none 模式；greedy 下完整/SSE 的 content、调用参数与 usage 相同。27B 还通过 `--enable-thinking` 复跑，验证提示词预开思考区时 reasoning 与正文分离。[验收工具](../../../tools/validation/openai-tools.py) 可复跑，原始响应与服务日志保存在 `artifacts/openai-parser-review/`。

parser 的 release 微基准使用同一 8 KiB ASCII 轨迹、新建 parser 和 feed/finish，7 组各 30 次取中位数，分别测 1/32/256 字节片段。32 字节片段的完整事件模式中，JSON 从 9.22µs 降至 5.10µs，function 从 8.76µs 降至 6.87µs，长空白从 426.58µs 降至 19.23µs。当前参数增量模式单独测得 JSON 9.79µs、function 7.84µs；旧版没有此事件模式，不能将其与完整事件模式混作同一 A/B。`before.csv`、`after.csv`、`after-deltas.csv` 与两个 probe 源码保留在同一证据目录。这些数字只描述解析成本，未作为 CUDA 推理或完整服务吞吐收益。

| 类别 | 必须验证 |
|---|---|
| 协议闭环 | SDK 实际传入工具定义、收到调用、回传结果并继续；未知工具、重复 ID、错误关联、nullable content、模型别名、unsupported 组合和错误码 |
| parser 正确性 | 对每个标记和 JSON/XML 结构切分 chunk；UTF-8、转义、嵌套、代码围栏、普通文本中的相似标记、多个调用、长参数、EOS、length、取消、失败；禁止未声明的自动修复 |
| 流式一致性 | 合并 deltas 后与非流式的 text/reasoning/tool 对象和顺序一致；index/ID 稳定、参数前缀不可回写；终末原因、usage、背压与断连正确 |
| 模式约束 | 首批 none/auto 与未支持模式的显式错误；已引入的 required/named/strict、parallel=false 再验证实际语义，不支持的 schema 与 parser capability 在运行前拒绝 |
| 长度和预算 | `P+G=L`、`L+1`、输入 cap、工具/模板/结果增加 P、reasoning 消耗 G、字段冲突、省略输出预算、整数溢出；所有入口与恢复路径使用相同生效限制 |
| 推测组合 | 接受 0/部分/全部前缀、拒绝后的 matcher 状态、多 token 发布、marker 跨轮与分轮切换；不发布拒绝内容，与非推测约束生成对照 |

CPU 微基准使用真实目标 tokenizer 和工具 schema：普通长文本、8KiB/更长 arguments、多个调用、单字符与实际 token chunk、冷/热 grammar cache、新建/复用 parser。分别测初始化和 feed/finish；vLLM Criterion 的 `iter_batched` setup 不计入主测量，不能根据 `create_parser` 名称就认为包含创建耗时。记录分配次数、峰值暂存、总 CPU 时间、每片段延迟与长输入的增长趋势。

服务验收使用固定 baseline ID。两边对齐模型/模板/tokenizer、工具 schema、history、thinking、stop、最终采样、总/输出预算、并发、缓存和实际约束能力，记录 parser 与 grammar backend 的版本/指纹。先用固定已提交 token 轨迹测文本解码→解析→SSE 的独立成本，再用真实模型测端到端；固定轨迹测量不能冒充 CUDA 推理吞吐。

除统一基线指标外，额外记录首次 content、首次 tool name、首次 arguments、完整可执行调用到达时间、格式/任务成功率、每请求 CPU/分配/暂存、delivery 排队、约束冷编译与稳态 mask 成本。SSE 事件数不当作 token 数；工具 name/argument 事件不冒充逐 token ITL。客户端工具执行耗时单独记录，不并入服务自身加速倍数。

工具选择与参数质量可按 vLLM 官方工具调用基准方式建立 BFCL 或固定 Agent 场景；两边必须用同一工具定义、结果桩和判定器。多轮自然生成导致工作量不同，保留实际 token 数、回合数、重试与成功率，不能只挑耗时更短但任务失败的样本。

## 交付边界

本项先交付长度 resolver、实际部署所需的有效配置与参数错误合同，再交付文本流式、工具历史和基础 parser；生成约束、严格工具模式及推测组合有具体场景时，按已具备的共同接口接入。基础工具闭环不等待 DFlash2，也不阻塞当前 token 诊断基线建立。

参数只在实际路径和验收完成后进入当前使用文档。交付以实际 Agent 闭环、预算生效和测量可复算为标准，不以兼容参数数量为标准。


### 约束成本复测

`INFER_CONSTRAINT_PACKAGE=/path/to/package cargo run --locked --release -p infer-cpu-bench` 使用指定包的实际 tokenizer，对固定严格 `ping` 调用轨迹测量冷编译（不计 TextAssets 加载）、缓存命中与 mask/commit 的 p50/p95，以及分配次数。基准包括特殊 EOS，遇到不合法轨迹立即失败；不执行 GPU、不测网络或完整推理吞吐，也不代表与 XGrammar 的对比。当前 matcher 仍有库内分配，尚未实现 GPU 重叠或强制 token 快进；不能宣称严格模式零分配或已获得端到端加速。
