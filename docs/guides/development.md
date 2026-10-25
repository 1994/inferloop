# 开发与运行

本页说明如何构建、运行和调试 inferloop。环境要求与最低版本见 [README](../../README.md)，模型包格式见[模型执行](model-execution.md)。

以下为现有入口；构建、测试与 CI 的分工见 [构建与依赖方案](../plans/engineering/build-and-dependencies.md)。正式测试、设备验收和性能对比执行 [共同 release 与基线要求](../plans/performance/baseline.md)，不能把现有门禁通过当作服务基线已经验收。

## 构建

```sh
cargo build --locked --release -p infer-cli
```

直接调用 Cargo 时，feature 仍由调用方负责：Linux CUDA 需要 `--features cuda` 和 [CUDA 环境](../../crates/backend/cuda/README.md)。如果希望按本机平台自动选择生产 backend，使用：

```sh
make local-build
```

它在 Linux 上启用 CUDA，在 macOS 上使用 Metal；Linux 还会自动把 GCC 的标准头目录加入 bindgen 的 include 路径。`auto` 只选择已编译且可用的 GPU，不会回退到 CPU；Cargo 本身不会根据运行时 GPU 自动启用 feature。没有 CPU 推理后端；无 GPU 的协议、单元与数值参考检查由 `make test` 运行。

完整检查见 `make check`，提交前至少运行 `make check-rust`。

## 默认启动

```sh
infer /path/to/model
```

默认启动 `127.0.0.1:8080` 的 HTTP 服务，无需指定 backend、`serve`、`--package` 或配置文件。未加入 PATH 时使用 `target/release/infer /path/to/model`。同名子命令优先解析为命令，模型目录若叫 `serve` 或 `doctor`，使用 `./serve` / `./doctor` 路径。

- 后端按已编译且可用的 CUDA → Metal 选择。
- CUDA 使用设备容量、当前可用显存与利用率计算预算；Metal 使用设备建议 working set 与利用率，替代固定 512 MiB 默认预算。
- 无配置文件时，workspace 由已绑定的模型图编译结果确定，输入上限受模型上下文限制；提供物理 KV 池的后端同步逻辑页预算，资源等待超时采用设备提交超时以覆盖冷图捕获。
- CUDA 投影默认自动选 tile，优先使用匹配的测量缓存，缺失时测量；支持且已验证的 CUDA 配置在省略执行选项时自动选择 BF16 attention 与 MTP，显式 `--num-speculative-tokens 0` 关闭推测。
- `--listen`、`--backend` 等仍可覆盖默认选择；`--config` 显式保留配置中的 runtime 预算，`--block-size` 和 `--max-num-batched-tokens` 作为命令行覆盖。旧 `serve --package` 入口保留兼容。

自动参数保证按契约配置与容量校验，最优性能仍需目标硬件和实际负载验证。调优与门槛见 [CUDA 性能指南](cuda-performance.md)。

## CLI

```sh
infer [options] <model_path>
infer [--backend auto|metal|cuda] <command> [options]
```

全局选项：

| 选项 | 默认值 | 说明 |
|---|---:|---|
| `--backend` | `auto` | `auto` 按 CUDA → Metal 选择，无可用 GPU 时报错 |
| `--gpu-memory-utilization` | 0.9 | 允许 engine 使用的显存比例（vLLM 语义，含权重） |
| `--host-memory-mib` | 自动 | 兼容预算覆盖；未指定时 CUDA/Metal 按设备推导 |
| `--num-gpu-blocks-override` | 自动 | 物理 KV block 数 |
| `--block-size` | 自动 | 每个 KV block / 逻辑页的 token 数 |
| `--max-num-batched-tokens` | 自动 | 单次 GPU prefill 的最大 token 行数 |
| `--upload-staging-mib` | 自动 | 权重上传暂存上限，不计入驻留 |

常用命令：

```sh
# 检查包内权重绑定
target/release/infer inspect-package --package examples/qwen-hybrid-tiny \
  --host-memory-mib 32768

# 执行请求文件（Metal 示例）
target/release/infer --backend metal run --package examples/qwen-hybrid-tiny \
  --requests examples/requests.json --config examples/runtime.json

# 对照 golden 验证数值
target/release/infer --backend metal verify --package examples/qwen-hybrid-tiny \
  --golden examples/qwen-hybrid-tiny/golden.json --atol 0.000002 --rtol 0.00002

# 启动服务与诊断 Agent
target/release/infer --backend metal serve --package examples/qwen-hybrid-tiny \
  --listen 127.0.0.1:8080
target/release/infer --backend metal agent --package examples/qwen-hybrid-tiny \
  --probe-memory-mib 4
```

请求格式见 [examples/requests.json](../../examples/requests.json)，调度与配额见 [examples/scheduling.json](../../examples/scheduling.json)，Agent 会话见 [examples/agent.jsonl](../../examples/agent.jsonl)。`doctor` 输出后端目录与可用性。

## HTTP 与 SSE

服务启动后提供原生接口；模型包包含 `tokenizer.json` 时额外启用文本接口。

| 路径 | 用途 |
|---|---|
| `GET /health` | ready、fault 与 pending release；隔离时返回 503 |
| `GET /native/v1/runtime` | runtime 与状态检查 |
| `GET /native/v1/observability` | 指标、资源、session 与 trace 完整性 |
| `GET /native/v1/events` | 按 after/limit/request cursor 查询事件 |
| `GET /native/v1/diagnostics` | 故障与准入证据 |
| `GET /native/v1/timeline` | Chrome Trace / Perfetto JSON |
| `GET /native/v1/traces/otlp` | OTLP JSON |
| `POST /native/v1/requests` | CanonicalRequest → 完成结果 |
| `POST /native/v1/stream` | `token` / `finished` / `error` SSE |
| `POST /native/v1/text` | prompt / messages 与多工作负载 |
| `DELETE /native/v1/requests/{id}` | 取消请求 |

请求体上限 2 MiB，准备与输出队列有界。断连或输出背压会取消对应请求，在途资源按 reader / fence 生命周期回收。

领域错误返回 `code` / `message` JSON：schema 错误 422、业务输入 400、未实现 501、重复 ID 409、容量或配额 429；JSON、body 与 Content-Type 错误由框架处理，未知字段一律拒绝。accepted `RequestId` 在同一 Engine 内不复用。

OpenAI 兼容路由、参数与错误见 [OpenAI 兼容接口](openai-api.md)。

## Checkpoint、回放与测量

```sh
target/release/infer --backend metal run --package examples/qwen-hybrid-tiny \
  --requests examples/requests.json --config examples/runtime.json \
  --journal artifacts/journal.json --snapshot artifacts/checkpoint.json

target/release/infer --backend metal replay --package examples/qwen-hybrid-tiny \
  --snapshot artifacts/checkpoint.json

target/release/infer --backend metal profile --package examples/qwen-hybrid-tiny \
  --journal artifacts/journal.json --config examples/runtime.json \
  --output artifacts/profile.json

target/release/infer --backend metal benchmark --package examples/qwen-hybrid-tiny \
  --requests 8 --input-tokens 6 --output-tokens 5 --output artifacts/benchmark.json
```

- `run --op-trace <path>` 额外导出算子编码记录。
- checkpoint 要求 quiescent 状态；restore 会校验 schema、weights/program/provider 身份与物理数据，协议见[请求生命周期](../architecture/request-lifecycle.md)。
- journal 截断不可视为完整记录，回放会报告 dropped。
- `benchmark` 为 closed 测量；`compare` 用正确性、goodput 与 P99 约束评估候选。
- 固定到达 HTTP 验证由 `make test-metal` 执行，测量范围见[质量门禁](quality-gates.md)。

## 仓库约定

- 提交 `crates/`、`tools/`、`docs/`、`examples/`、CI 与根配置。
- 主 workspace 与 CPU benchmark 各自的 `Cargo.lock`、微型模型权重、tokenizer 与 golden 都是必要资产，需要保留。
- 完整模型下载、本机环境、`target/`、Python 缓存和 `artifacts/` 由 [.gitignore](../../.gitignore) 排除。
- `make clean-artifacts` 删除可重建的生成输出，保留文档引用的证据与模型资产；先用 `python3 tools/check/clean-artifacts.py --dry-run` 预览。
- 提交前用 `git status --short` 与 `git diff --cached --stat` 核对实际文件。
