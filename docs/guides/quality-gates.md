# 质量门禁

所有检查共用统一入口 `make check`，任一步失败即退出。CI 与本地使用同一脚本，`quality-gate` job 汇总 Rust、工具、MSRV、依赖安全、CUDA（无 GPU 的头文件门禁）与四平台打包结果，见 [打包指南](packaging.md)。

本文描述现有门禁。正式测试与服务对比的共同要求见 [性能基线方案](../plans/performance/baseline.md)；release 构建身份、当前 native/vLLM 配置和证据校验已由基线方案的门禁覆盖，但现有 gate 成功退出不代表对应 profile 已通过性能判定。入口与测试收集的约定见 [构建与依赖方案](../plans/engineering/build-and-dependencies.md)。

## 检查项

| 范围 | 要求 |
|---|---|
| 目录与依赖 | crate 分组、路径注册、小型 facade、禁止实现层反向依赖服务或具体 backend、本地文档链接有效 |
| Rust lint | `cargo fmt --check`；Clippy all / pedantic / nursery 全部 `-D warnings`；`clippy::absolute_paths` 要求三段及以上的项路径先用 `use` 引入（阈值 2，见 `clippy.toml`） |
| 长度与复杂度 | `too_many_lines` ≤ 100、`cognitive_complexity` ≤ 25，不允许调高阈值 |
| 数值常量 | `crates/*/*/src` 生产代码里的数字字面量必须写成带 `///` 说明的具名常量；白名单 `0`/`1`/`2`、数组下标、`const`/`static` 定义、字符串与字符字面量、注释、`#[cfg(test)]`/`#[cutile::module]` 段以及测试文件除外 |
| 错误路径 | 生产代码禁止 `unwrap` / `expect` / `panic` / `todo` / `unimplemented` / `dbg` 与忽略 `Result` |
| unsafe | 默认 `deny`；仅经局部安全证明的 OS/设备 FFI、CUDA PDL 与隔离的 benchmark allocator 开放，并说明安全条件 |
| Feature 隔离 | 默认 CLI / IR / workspace 与 CUDA feature 分别构建；CUDA 在打包和硬件门禁单独启用，生产依赖图不含 CPU 执行器 |
| API JSON | `crates/service/frontdoor/src/http` 禁止 `json!`；响应与 SSE 代码禁止 `serde_json::Value`，只允许 `tools/check/policy.py` 列出的 wire 请求 DTO 与协议不透明输入 |
| 编译与文档 | release 构建、unit / integration / doctest、独立 golden、Rustdoc 警告失败 |
| CPU | release 协议、KV primitives、Engine tick 分配门禁，范围见 [CPU 性能测量](cpu-performance.md) |
| MSRV | Rust 1.90 检查 workspace（不含 CUDA）与隔离的 CPU benchmark |
| 依赖 | frozen lockfile；advisory / yanked、license / source allowlist、重复与通配版本、闲置声明 |
| 凭证 | Gitleaks 扫描源码；在 Git 仓库中另扫历史 |
| 工具 | Ruff 检查与格式化、actionlint、workspace lint 继承 |
| Linux | 原生 affinity / cpuset / 失败恢复与 owner 启动；Mac 上交叉编译检查 |
| Metal | 真实 CLI / Agent / HTTP / SSE、golden、页压力、checkpoint、固定到达负载与资源排空 |

## 执行入口

```sh
make check            # 全部（macOS 上包含 Metal 实机）
make check-rust       # 目录、lint、测试、文档、release、CPU 分配与 golden
make check-tools      # Ruff 与 actionlint
make check-security   # cargo-deny、cargo-audit、Gitleaks
make check-msrv       # Rust 1.90
make bench            # 隔离的 CPU 分配计数（别名 check-cpu）
make test-metal       # Metal 实机验收（仅 macOS，别名 check-metal）
make test-cuda-hosted # CUDA 编译、lint 与 CPU 侧测试：需 Toolkit 头文件，不需 GPU
make test-cuda        # CUDA 实机 kernel 验收（需 Toolkit、GPU 与 tileiras，别名 check-cuda）
make check-attention  # 自研 Attention 对 Candle 基线：数值与性能不达标均非零退出
make check-linux      # Linux 原生放置测试
make check-linux-numa # Linux NUMA 硬件验收，见 Linux 指南
```

`check-rust` 对 CUDA crate 的无 feature 版本只检查编译隔离：整个库被排除，
该入口报告 `compiled`，不声称执行了 CUDA 测试。纯主机用例也随 CUDA 库启用，
在 `test-cuda-hosted` 中执行。需要 Toolkit 的部分分成两半：`test-cuda-hosted` 用 CUDA 13.2 开发头文件编译、lint 全部 target 并运行 CPU 侧测试，因此托管 CI 可以运行，也是唯一会编译 cuda-gated `#[cfg(test)]` 模块的门禁；`test-cuda` 追加设备验收（`cuda-kernels` 示例与需 GPU 的用例）。缺少 GPU 或相应 syscall 权限时，硬件门禁明确失败，不会跳过。

## 小迭代与大版本分级

小迭代默认使用 `make check-iteration`：一个真实 native 新进程配对，
每边各做一次 C16 和短请求预热、一次计时，再跑候选的 64 题质量抽检。
性能与质量共用 600 秒总预算（包含启动和最多 10 秒清理），运行锁忙时立即退出，
不排队、不重试、不扩样。入口使用已有 release 二进制；编译与针对改动的回归单独执行。

`summary.json` 同时给出基线/候选的耗时、TTFT、TPOT、相对变化、正确题数、
质量变化百分点、实际执行配置和资源配额。任一性能指标退化超过默认 10% 即快速失败；
质量沿用原任务集、原始答案、EOS、512 cap 和 16/64 拒绝检查点。
当两边执行几何、报价和实际分配完全相同，且显存额度波动小于同一 KV 物理分配块时，
迭代门禁记录波动并继续；缺少可验证的块大小、波动达到一整块或实际资源不同时仍拒绝。
成功只表示迭代筛查通过（`iteration_accepted: true`），不签发完整质量或发布证书。
64 题报告保持 `completed: false`，不能作为后续完整质量基线。

```sh
make check-iteration PYTHON=/path/to/python ITERATION_ARGS='\
  --model /path/to/model --baseline-executable /path/to/base/infer \
  --executable /path/to/candidate/infer --suite /path/to/full-suite.json \
  --baseline /path/to/full-baseline/report.json --tuning-seed /path/to/tuning.json \
  --workload /path/to/c16.json --short-workload /path/to/short1.json \
  --output artifacts/iteration-unique'
```

五对以上独立进程的完整性能矩阵、长上下文、长短混跑、受限显存，以及完整 1,024 题质量，
属于大版本发布验收，不串入小迭代入口。
`make check-release` 先按冻结 checklist 校验所有已采集的正式进程配对，
全部通过才启动完整质量采集。按发布矩阵逐项重复 `--performance CHECKLIST PAIRS`；
缺少、失败、重复或来源不同的样本不能放行。长测采集沿用既有 `serve-compare.py`、
`experiment-checklist.py` 和[性能基线方案](../plans/performance/baseline.md)，只在发布验收显式安排。
发布前仍需通过代码、依赖、目标平台与设备门禁。

```sh
make check-release PYTHON=/path/to/python RELEASE_ARGS='\
  --performance /path/to/default-checklist.json /path/to/default-pairs.json \
  --performance /path/to/constrained-checklist.json /path/to/constrained-pairs.json \
  --suite /path/to/full-suite.json --model /path/to/model \
  --executable /path/to/candidate/infer --baseline /path/to/full-baseline/report.json \
  --output artifacts/release-quality-unique'
```

原始完整采集入口仍用于建立基线与发布验收；日常使用上面的迭代入口。

## 通用 Attention 门禁

生产运行时与 kernel 保持 Rust/cuTile 路线；Candle 0.11.0 仅作为独立 benchmark 基线。
`make check-attention` 不切换实现，不允许用精度优势掩盖性能差距，也不因基线速度快就放宽自研精度。

通用语义位于 `infer-kernel-api::attention`：Q/KV 长度和 head 数、独立 QK/V 维度、scale、dtype、
causal query offset、滑动窗口与分段 mask。CUDA `DenseAttentionPlan` 接受调用方的 buffer/graph，
不管理模型、RoPE、内存池或调度。当前 dense provider 支持 F32、padded head ≤256；
其他 dtype/宽度显式拒绝。paged KV、量化 KV、MLA、稀疏/线性 attention 不能冒充已覆盖的 dense SDPA。

- 数值矩阵：31 个用例，覆盖 head 32/64/72/128、短序列、尾块、长序列、MHA/GQA/MQA、
  cross attention、causal prefill/decode、负 query offset 的全遮蔽行、滑动窗口，以及 RoPE/分帧回归。
  显式 F64 matmul/mask/softmax/matmul oracle；全遮蔽行定义为零，不使用自动选择的 SDPA reference。
- 自研正确性：输出有限、长度与 padding 正确，
  `max_abs_error <= 1e-6 + 1e-4 * max_abs_reference`。独立测试还覆盖 QK/V 维度不同。
- Candle 基线正确性：FP16 使用 atol=2e-4/rtol=2e-3，BF16 使用 atol=2e-3/rtol=2e-2；
  记录各自实际误差。这是不同精度下的性能目标，不宣称两边采用相同计算精度。
- 性能：同机 release 二进制，5 轮交替顺序、100 次图 warmup、每轮 60 个 event 样本；
  每次图执行含 8 次算子调用，按调用平均，以摊薄主机提交开销。
  每个形状的配对中位数与 P95 相对 Candle 均不得慢超过 5%，至少 4 轮满足中位数容差；
  跨形状几何平均延迟不得高于 Candle，轮间中位数漂移超过 10% 判为需重测。
- 两种计时：完整流程计入 dtype/layout 转换与恢复 F32 输出，RoPE 回归案例还计入 RoPE；
  dense SDPA 另比较预先准备好输入的核心调用，不能靠 Candle 转换开销隐藏自研内核落后。
  Candle 核心调用仍使用其公开 Rust wrapper，含其图内分配节点，不冒充裸 CUDA launch。
  upload/JIT 在计时外；普通 SDPA 原生路径已是核心调用，两个计时共用样本。
- 证据：校验 fixture SHA256、设备 UUID、驱动、Toolkit、运行 ID、精度标签、完整形状与轮次；
  记录二进制 SHA256、manifest 与原始日志，结束时再次校验输入和二进制未变化。
  缺失、重复、NaN、构建/执行失败为 `invalid`，数值失败为 `reject`，性能差距或噪声为 `review`；
  只有 `pass` 返回零。任何一组形状落后都不能被其他形状的平均加速覆盖。

CPU 门禁判定测试已进入 `check-rust`。Candle 在 `tools/bench/attention` 独立 workspace，
锁定依赖，不进入生产依赖图。报告在 `artifacts/attention-gate/results-*`，每次独立保存；
`INFER_ATTENTION_GATE_OUT` 可改目录，`ATTENTION_PYTHON` 可指定已安装 torch/safetensors 的 Python。
默认通过 uv 运行 fixture generator。主机 bindgen 参数沿用 [CUDA 构建说明](../../crates/backend/cuda/README.md)。
基准不按 GPU 型号选择算子；每台设备独立生成证据，单机结果不外推为所有硬件通过。

## 任务质量采集与基线复用

固定的、已分词任务集可用 [run-task-quality.py](../../tools/bench/run-task-quality.py)
采集 CUDA native 输出。它复用现有服务工具和
[score-task-quality.py](../../tools/bench/score-task-quality.py)，不改变题目、数值契约、
评分公式、置信水平或通过阈值，也不代替独立的 kernel 数值与服务性能验收。

使用已安装 Transformers 的 Python、预先构建的 release 二进制和本地模型：

```sh
# 基线只采集一次：一个服务进程按顺序跑完整任务集。
bash tools/bench/safe-run.sh python3 tools/bench/run-task-quality.py \
  --suite artifacts/task-suite.json --model /path/to/local-model \
  --executable /path/to/baseline/infer --output artifacts/task-baseline

# 后续候选复用明确指定的完整基线，只运行候选模型一次。
bash tools/bench/safe-run.sh python3 tools/bench/run-task-quality.py \
  --suite artifacts/task-suite.json --model /path/to/local-model \
  --executable /path/to/candidate/infer --output artifacts/task-candidate \
  --baseline artifacts/task-baseline/report.json
```

默认并发为 16，生成长度和 EOS 来自任务集；仅支持声明的 greedy 合约。
每次采集启动一个服务，不再为任务集分片重复加载模型。
复用前重新核对完整原始答案、任务集 SHA256、模型身份、设备与驱动、
数值契约和已记录的采集设置；模型身份沿用现有工具的配置/tokenizer 哈希与权重分片大小，
不是权重内容的完整哈希。改变模型包后必须重新建立基线。
记录并校验基线文件与候选二进制 SHA256，不自动选择基线或沿用旧的通过标记。

每批验证实际 prefix 复用计数没有增加、所有请求完整结束后，
立即保存并同步 `results.jsonl`，同时更新 `progress.json`。
中途失败会保留已验证的批次和不完整报告，不能用于放行；当前入口不自动恢复采集。

提供基线时，在完成 16、64、256 条后的固定检查点做质量快速失败检查
（仅检查小于完整任务数的点）。每个检查点同时检查整体与每种题型，
用反向配对 Tango score 上界和按原题聚类的 Rao–Scott 修正识别明确超过原退化容差的损失；
全部预定检查点和题型共同分配原置信水平的错误预算（Bonferroni）。
并发超过 16 时也先完成首批 16 条再检查。
明确退化立即以非零退出、正常关闭服务，保留原始结果、不完整报告与
`screening.json`；不确定时扩大批量。每批耗时写入报告，便于定位慢批次。
快速检查只允许拒绝，不能提前放行；最终仍对原顺序的全部任务运行原评分器，
完整通过阈值、题目数与生成预算均不变。
输出目录必须不存在，避免覆盖旧证据。
只有显式提供基线才生成 `gate.json`，质量未通过时返回非零；
单独采集成功只表示输出收集完整，质量状态为 `not_evaluated`。

减少的是重复加载和基线重跑，不是任务数量。具体耗时仍取决于模型、输入和硬件；
不得据此宣称推理吞吐提升。仅修改采集工具时，执行 bench Python 测试、
`make check-tools` 和文档检查；工具行为用小规模实机采集验证，无需为工具改动重跑整套模型质量集。

## 工具链与版本

| 工具 | 版本 |
|---|---|
| Rust（固定） | 1.99.0，由 `rust-toolchain.toml` 提供 |
| Rust（MSRV） | 1.90.0 |
| Python | 3.12+ |
| uv | 任意近期版本 |
| Go | 用于 `go run` 执行 actionlint 与 gitleaks |
| cargo-deny / cargo-audit | 0.20.2 / 0.22.2 |
| Ruff / actionlint / Gitleaks | 0.15.7 / 1.7.7 / 8.24.3 |

```sh
rustup toolchain install 1.90.0 --profile minimal
cargo install --locked cargo-deny --version 0.20.2
cargo install --locked cargo-audit --version 0.22.2
make check
```

## 例外

- [deny.toml](../../deny.toml) 记录 `paste 1.0.15`（RUSTSEC-2024-0436）、CUDA 构建依赖的精确许可证审批与重复版本例外，原因和移除条件以配置为准。
- `tools/check/policy.py` 另外校验 workspace lint 继承、闲置依赖，以及 API 响应的 typed JSON 规则。
- Metal 间接依赖 `block 0.1.6` 有已知 future-incompatibility 报告，作为上游迁移项保留，不用全局 `RUSTFLAGS` 隐藏。
- 测试代码允许 `unwrap` / `expect` / `panic` 用于断言，helper 仍返回 `Result`；其他数值或接口例外需在局部说明。
- 可选的官方 tokenizer parity 测试通过 `INFER_QWEN_TEXT_PACKAGE` 指向外部包，普通 CI 使用仓库内微型资产。

## 验收口径

能力声明需要分别提供执行、正确性、benchmark、profiling、observability 与 Agent 证据。编译成功、微型 golden 或单个 kernel 通过，只证明相应范围。性能未达门槛时保留失败结论，不通过放宽误差或删掉慢形状放行。

Attention 当前保留的设备结果为 `review`，详见 [CUDA 性能指南](cuda-performance.md#通用-attention-对照)。GPU 与 NUMA 专项检查必须在对应硬件运行；hosted CI 上传的包仍保留 `gpu_inference_accepted: false`。
