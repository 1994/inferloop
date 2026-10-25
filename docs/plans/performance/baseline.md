# 性能基线与 vLLM 对齐方案

状态：采集与门禁已实施，已签发三条有效基线（B2 达到验收；B3 的复测样本数与负载类型仍是待补项，见文末"已实施"）。源码基点为 `7c79d99`，整理日期为 2026 年 10 月 9 日。

整体排期与完成状态由 [路线图](../README.md) 的 B2、B3 维护。本方案统一规定 release、当前 native/vLLM 配置对齐、比较身份、计时与统计、原始证据和基线更新。CUDA、MTP、DFlash2、ReplaySSM 的性能工作都引用本方案；模块各自维护数值、状态与资源验收，不另定采样规则或基线门槛。

现有缺口见 [对比方法审查](../../reviews/vllm-benchmark-methodology-2026-10-09.md)，CUDA 正确性前置见 [提交审查](../../reviews/cuda-commits-2026-10-09.md)。旧服务矩阵已删除，不能再作为基线身份或当前差距的依据。

本方案定义比较与证据的硬门禁，采集、比较和冻结工具已按本节实现。仍未完成的是 B3 的独立复测样本、持续/混合负载与部分 profile 的完整矩阵；这些缺口不能用历史倍率或单次对比替代。基线是否准确与引擎是否足够快分别判定：即使 native 明显慢于 vLLM，只要比较条件与证据通过，仍可形成有效基线；无效对比即使显示加速，也不能形成基线或批准优化。

## 已实施：采集、门禁与已签发基线

本节记录**实测**状态，便于下一位维护者不必重新考古。核验命令写在条目里，全部可在本机无 GPU 执行，
只有标注"设备窗口"的项需要真机。

**工具与门禁**

| 工具 | 职责 |
|---|---|
| `tools/bench/experiment-checklist.py` | 一个 profile 只声明一次；`--check` 用清单核验单次运行（缺任何条件即失败），`--compare` 打印**配置对照表**（native/reference 逐项实际生效值）与性能对比 |
| `tools/bench/compare-results.py` | 有效性门禁：release 构建身份、同一模型制品与硬件、cache 状态与资源约束一致、矩阵完整、无失败或截断请求；通过后才做性能判定 |
| `tools/bench/freeze-baseline.py` | 不可覆盖地冻结基线与证据，`--verify` 重算全部哈希 |
| `tools/bench/hardware-monitor.py` | 随测量记录设备遥测（而非在间隙里采样） |

这些模块的 `tools/bench/test_*.py` 由 `tools/check/gate.sh` 收集（gate.sh 里有一行注释说明这是审查发现
"某个模块没有被任何地方收集"之后的修正），因此采集与门禁的实测行为在 CI 里跑。

**已登记 profile 与已签发基线**

`benchmarks/profiles/{2b-mtp0,27b-mtp0,27b-mtp2}.json` 三个 profile 对应方案要求的模型范围
（qwen3vl-2b 与 Qwen3.8-27B-NVFP4）与 MTP 档位（0 与 2）。据此签发三条基线：

| baseline ID | profile | 核验结果 | 性能判定 |
|---|---|---|---|
| `2b-mtp0-serving-v1` | 2b-mtp0 | 8 个文件哈希一致、`identity_verified`、`cache_effective`、四个基础场景矩阵完整 | `gate.passed: false` |
| `27b-mtp0-serving-v1` | 27b-mtp0 | 同上 | `gate.passed: false` |
| `27b-mtp2-serving-v1` | 27b-mtp2 | 同上 | `gate.passed: false` |

核验命令（本机可复现，无需 GPU）：

```sh
for b in 2b-mtp0-serving-v1 27b-mtp0-serving-v1 27b-mtp2-serving-v1; do
    python3 tools/bench/freeze-baseline.py --verify benchmarks/baselines/serving/$b
done
```

**B2 的五项可核验条件分别落在哪里**

| 条件 | 证据字段（`benchmarks/baselines/serving/*/…json`） |
|---|---|
| release 身份 | `identity.release_profile`（实测 `lto=thin`、`codegen-units=1`）、`identity.build`（`sha256`、`bytes`、`image`、`debug_sections: []`、`release_like: true`）、`identity.source`（revision + dirty） |
| 配置回读 | `config_readback`（与 CLI 传入值分开记录的实际生效值） |
| 工作量 | `inputs_sha256`、`max_new_tokens`、`matrix`、`hot_prefix_tokens_reused` |
| 缓存 | `prefix_cache_enabled` 与清单核验出的 `validity.cache_effective` |
| 计时与统计 | `trials`、`completed`、`statistical_basis`（`paired_units`、`declared_max_drift`、逐项 drift）、`telemetry` |

无效对比不能误判通过这一点由 `compare-results.py` 的有效性门禁与它的单测保证：身份、模型/硬件、
cache、资源、矩阵与请求完整性任一不满足就不签发。

**验收入口按方案使用 release 并报告计数**

方案要求"正式 Rust 测试、性能对比和设备/模型验收统一使用 release"，并"每个测试入口报告收集、执行、
ignored 和失败数量"。审查发现**设备验收入口是例外**：`make test-cuda` 的测试步骤跑的是 debug，
`make check-linux-numa` 同样是 debug ✗；而 `make test-metal` 早已通过 release 二进制验收 ✓、
`make bench` 也走 release ✓。

现在两个入口都改为 `cargo test --locked --release`，并接入上一条的计数包装器（`--entry cuda` /
`--entry numa`）✓。CUDA 测试本身对两种 profile 都有准备——套件里有三处 `cfg!(debug_assertions)`
分支、没有任何 `debug_assert!`，所以 release 是它们预期要支持的形状 ✓。

顺带修掉包装器的一个真 bug：它原先删除**所有** `--`，会把 `cargo test … -- --ignored` 里的内层分隔符
一起吃掉 ✗，使 `--ignored` 退化成测试名过滤 ✗。现在只删除紧跟选项的那一个 `--`，并补了一条专门的测试。

**profile 身份从"记录了"变成"可证明"**

方案要求"执行清单保存构建命令、完整源码版本、features、target、编译器、**实际 profile 设置**和二进制
哈希；不能仅凭文件位于 `target/release/` 就认定构建正确"，并明确"任何优化级别、LTO、codegen units、
debug assertions、Rust flags 与 CUDA 编译选项的覆盖均要记录"。

审查发现报告只记录了 **manifest 里显式写出的键**（本仓库只有 `lto`、`codegen-units`）✗ —— 有效优化
级别、是否带 debug assertions、是否 incremental 全都不在报告里 ✗，而门禁只检查 `release_like` 布尔
✗，于是一份 profile 被改坏的报告也能通过 ✓→✗：

- `release_profile()` 现在给出**有效**配置：cargo 的 release 默认值叠加 `[profile.release]`，
  同时保留 `explicit`，并记录会改变构建的环境变量（`RUSTFLAGS`、`CARGO_ENCODED_RUSTFLAGS`、
  `CUDAARCHS`、`NVCC_CCBIN`，仅在设置时记录）；
- `compare-results.py` 新增 `require_release_profile()`：原生报告必须带有效 profile，且
  `debug-assertions`、`overflow-checks` 必须为 false、`opt-level` 不能为 0、`incremental` 必须为
  false、`debug` 不能为真——即"debug、混合 profile 或无法证明构建身份的结果不进入正式 baseline"
  这条被真正执行；
- 三条已签发基线仍逐条 `--verify` 通过（8 个文件哈希一致 ✓，`--verify` 只重算哈希、不重跑当时
  的门禁 ✓），测试夹具同步更新并新增 3 个反例用例。

**数值契约与采样种子的两条有效性规则**

方案有两句话过去只写在文档里、没有任何检查 ✗：*"共同数值契约不能只写 BF16/FP8/NVFP4 名称，要记录
执行中的权重处理、activation/accumulation、KV、recurrent state 与采样规则"*，以及请求对齐项里的
*"同一 token 输入、输出策略、**采样参数、seed**、EOS/stop 条件"*。

`experiment-checklist.py` 现在强制这两条：

- `numeric` 的 `weights`/`activation`/`kv` 必须是描述规则的字符串，**纯 dtype 名一律拒绝**
  （`bf16`/`fp16`/`fp32`/`f32`/`fp8`/`nvfp4`/`mxfp4`/`int8`/`w4a16`/`w8a8` 等，大小写不敏感），
  空值同样拒绝；
- `temperature > 0` 时必须声明 `seed` ✗→✓：没有固定种子的采样，两边哪怕温度相同也不是同一个实验。

写规则时立刻得到了印证：**测试夹具自己就写着 `weights: "nvfp4"`、`activation: "fp8"`** ✗，
正好是方案点名的那种写法 ✓，已改为描述性契约。三条已签入 profile 的温度都是 0 且数值字段本就是
描述性文字，因此**无需改动、继续有效** ✓。新增 6 个用例（纯 dtype、带有大小写的变体、空值、
需要种子、带种子、贪心无需种子），bench 单测 78 → **84 全过** ✓。

**已签发的数字可以从冻结证据重新算出（实测）**

`freeze-baseline.py verify` 只核**哈希** ✓ —— 它说明证据没被改动，却**不说明 manifest 里的数字来自这些证据**
✗（抄写错误、签发后被编辑的报告、或比较口径漂移都不会被发现 ✗）。新增 `tools/bench/verify-baseline.py` ✓
（接入 `make check-tools` ✓，10 个用例 ✓）做**复算** ✓：取 manifest 记录的 baseline/candidate 两份报告 ✓，
用**与签发时同一个 `compare()`** ✓ 重算每个 case 的 baseline/candidate 中位数与 ratio ✓，再与 manifest 里
`gate.performance.metrics` 逐项比对（含 `available` 与总判定 ✓，容差 1e-9 ✓），多算或少算都算失败 ✓。

同一条检查还核**契约来自 profile 而不是签发当天** ✓（这些都不需要设备 ✓）：

- 冻结目录里的 profile 副本 `profile_id` 与 manifest 一致 ✓；
- **矩阵完整** ✓：`validity.matrix` 与**每份报告**的 `matrix` 都必须等于 profile `workload.matrix` 声明的
  case/并发/重复次数 ✓（漏掉一个 case、事后改并发都会失败 ✓）；
- **阈值来自声明** ✓：`gate.max_latency_ratio` 必须等于 profile `measurement.max_ratio` ✓（事后放宽门槛
  会被抓住 ✓）；
- **工作量逐份对齐** ✓：每份报告的 `max_new_tokens`/`temperature`/`mtp_depth`/`eos_tokens`/输入哈希都要
  与 profile 一致 ✓；
- **独立复测** ✓：必须有 `reproductions` ✓、每个单元都 `within_declared_tolerance` ✓、
  `paired_units` 与单元数自洽 ✓；**记录的漂移值本身也要能重算** ✓ —— 用该单元的两份 g2 报告重跑同一个
  `compare()` ✓，逐项复算 `drift` ✓ 与 `worst_drift` ✓ 并核对 `within_declared_tolerance`/`passed` ✓
  （"记录了 12 个漂移值" ✗ 不等于"它们仍来自那两份报告" ✓）；**活动窗口的子集不变量**也一并核 ✓
  （窗口样本数不得超过整轮 ✓、均值不得超过峰值 ✓）；**若少于方案要求的 5 ✓，打印提示而不是判失败** ✓（现状 2 ✓，manifest
  自己也如实记着 ✓ —— 这是需要设备窗口的缺口 ✓，不是可以含糊过去的 ✓）；
- **测量本身可信** ✓：每份报告必须来自**干净的树** ✓（`source.dirty` 为 false 且有 revision ✓）、
  checklist **通过且无 mismatch** ✓ 且 profile 对得上 ✓、profile 启用 prefix cache 时 `hot_*` case 必须
  **真的复用了 token** ✓（否则"热"这个 case 名不副实 ✓）；
- **checklist 就是那份 profile，而且是现在的它** ✓：每份报告（含复测报告 ✓）记录的 `checklist.sha256`
  必须等于冻结目录里 profile 副本的实际哈希 ✓；并且该哈希必须仍等于 `benchmarks/profiles/` 里**现行**
  profile 的哈希 ✓ —— 方案要求"共同约束变化时**新建 profile 并重测**"✓，所以**事后编辑现行 profile** ✗
  会立刻失败 ✓（"沿用旧结果"允许 ✓，"悄悄改掉旧基线所依据的约束"不允许 ✗）。若该 profile 已从
  `benchmarks/profiles/` 退场 ✓，只打印提示不判失败 ✓（旧结果仍要能读 ✓）；
- **同一台机器才谈得上比值** ✓：manifest 声明了签发时的设备 ✓，每份报告也带自己的身份 ✓ —— 现在核两者
  **逐设备一致** ✓（uuid/名称/驱动 ✓）：不同设备、不同驱动、或身份未经验证（`identity.verified` ✓）都会失败 ✓；
  另外要求 `identity.build.sha256` 存在 ✓ 且与报告的 `binary_sha256` **一致** ✓（同一个制品被两处记录却互相
  矛盾，正是"可复现制品"要防的 ✓）；
- **硬件证据仍然成立** ✓：签发时 `require_telemetry` 会拒绝没有遥测的报告 ✓；冻结之后没人再核 ✓，现在补上
  —— 报告里的 `telemetry.sha256` 必须与**旁边那份 `.telemetry.jsonl`** 的实际内容一致 ✓（这是 manifest 的
  文件哈希**表达不了**的一层 ✓：报告与其遥测日志之间的绑定 ✓）、日志必须能按 JSON Lines 解析 ✓ 且以
  metadata 开头 ✓、其中 `sample` 行数必须等于报告记录的 `samples` ✓（实测 251 = 251 ✓），并且仍有
  `active_window` ✓ 与带 `gpu` 的 trial ✓；
- **配置可核验** ✓：每份报告要说明配置从哪读 ✓（`config_readback.source` ✓）。**这里有一处如实的不对称** ✓：
  native 侧是真回读 ✓，而 **vLLM 侧只记录命令行**（`readback: null` ✓）—— 命令本身在报告的 `command` 里 ✓，
  所以配置仍可核验 ✓；工具**打印这条提示** ✓ 而不是假装字段是满的 ✓，也不因此判失败 ✓。

**实测**：三条已签发基线**全部通过** ✓ —— 每个指标都能从其冻结报告重新算出，判定与记录一致 ✓，矩阵、
阈值、工作量与独立复测也都对得上 ✓；并对三条都打印了 `2 paired units, the plan asks for 5` 的提示 ✓。这条证据
与上一节的"版本边界"互补 ✓：一个保证**旧基线现在仍可读** ✓，一个保证**记录的数字确实来自它的证据** ✓。

**profile 规则有版本边界，旧基线仍可复核（实测）**

上面那条规则加进来之后出现一个副作用 ✗：**已签发的三条基线无法被当前门禁复核** ✗ —— 它们的报告早于规则，
只有 manifest 的显式键，于是 `compare-results.py` 直接报 "records no effective release profile" ✗。
而方案明确要求"**保留旧 profile 的结果和交接对照**"✓，所以这不是"旧报告无效"的问题，而是**版本边界**问题 ✓：

- `release_profile_state()` 现在给出三态：**`proven`**（带有效 profile ✓）、**`legacy`**（字段缺失，
  规则之前写的 ✓）、以及**直接抛错**（profile 存在但被改坏：debug assertions / 未优化 / incremental ✓）；
- `compare()` 默认要求**两侧都** `proven` ✓；新增 `--allow-legacy-baseline` ✓ 表示"这是一次**历史对照**"
  （复核冻结对、或用新候选对照冻结参考 ✓），此时任一侧可以是 `legacy` ✓ —— 但**新测量**在默认路径下
  仍必须自证 profile ✓，规则强度没有被稀释 ✓；
- **实测复核**：三条已签发基线用该开关重新跑一遍有效性门禁 ✓ —— **各自 12 个指标全部复核通过** ✓，
  且 verdict（`passed = false`）与其 manifest 记录**一致** ✓，说明冻结证据至今仍可被当前门禁读懂 ✓。

**性能结论如实记录，不作签发条件**

三条基线的 `gate.passed` 都是 `false`：native 目前慢于 vLLM（例如 2b 的 batch4 `wall_seconds`
1.58×、`ttft_seconds` 2.25×，上限为 1.10×）。按方案自身的规定，**测得退化仍是有效基线**——有效性
（比较条件与证据）与"引擎是否够快"分开判定，`manifest.json` 如实写 `false`，不因为结果难看而拒绝签发，
也不因为好看而放宽有效性。

**能力限制不是缺口，而是登记项**

方案要求未支持的算法档位"明确登记能力限制"，不允许用普通 decode 的回退结果充当该算法验收。
`benchmarks/profiles/README.md` 因此登记了每个模型可测的档位与依据：Qwen3.8-27B-NVFP4 有 MTP head
（`mtp_num_hidden_layers = 1`），可测 0 与 2；qwen3vl-2b 的配置**不含任何 `mtp_*` 键**，只可测 0。
`experiment-checklist.py` 现在强制这条：清单声明正 MTP 深度而模型未声明 MTP head 时直接拒绝。

**B3 的待补项（都需要设备窗口）**

1. **独立复测样本数**：现值 `paired_units: 2`，方案要求至少 5。证据包自己在 `statistical_basis.note`
   里写明了这一点，消费方可以据此判断区间可信度。
2. **负载类型**：四个基础场景（short/long/batch4/hot_long）已覆盖；方案的**服务产品验收**还要求
   "持续与混合负载、代表性任务与长度分布"，目前尚未采集。
3. 各引擎分别调优的产品实验需要另设 profile，不替代当前"当前配置对齐"基线。

以上三项完成后，B3 才算按共同契约完成完整矩阵与固化；在那之前，本方案的有效性结论仅适用于已签发的
三个 profile 与四个基础场景。

## Release 与当前配置是必检条件

正式 Rust 测试、性能对比和设备/模型验收统一使用 **release**：CLI、backend driver、example 和 host/device 测试 target 均按对应 target 的 release 配置构建，Rust 测试使用 `cargo test --locked --release`。Python 工具测试不使用 Cargo profile，其依赖的 Rust/GPU 可执行文件仍遵守此要求。执行清单保存构建命令、完整源码版本、features、target、编译器、实际 profile 设置和二进制哈希；不能仅凭文件位于 `target/release/` 就认定构建正确。debug、混合 profile 或无法证明构建身份的结果不进入正式 baseline。

release 配置沿用当前仓库的 `[profile.release]`，任何优化级别、LTO、codegen units、debug assertions、Rust flags 与 CUDA 编译选项的覆盖均要记录。vLLM 使用固定版本的优化发行构建，保存安装来源及 PyTorch/CUDA/原生扩展的版本与构建信息；Python 服务不套用 Cargo 的 profile 名称，但不能用 debug 原生扩展或调试执行环境参与正式对比。额外的 profiler、同步调试与算子诊断运行单独保存。

这里的配置对齐指实际工作量、质量条件和资源约束等价，不要求本项目复制 vLLM 的 CLI 字段、默认值、历史别名或配置形式。对照工具将各自配置映射到共同实验清单；缺少同名参数不能成为放宽测量条件的理由。产品配置的取舍见 [Agent 服务方案](../serving/agent-api.md)。

vLLM 配置以**当前待测 native 的生效配置快照和共同实验清单**为依据逐项对齐，不复用脚本中的旧默认值。正式运行前输出两边的配置对照表并验证实际生效值：

| 对齐项 | 要求 |
|---|---|
| 模型与数值 | 同一模型制品及 revision；明确权重处理、计算/累加和 KV/state 精度；再量化或不能保持相同数值规则的差异按独立 profile 与质量契约验收 |
| 请求 | 同一 token 输入、输出策略、采样参数、seed、EOS/stop 条件、并发与到达模式；文本对照还核验 template/tokenizer 与输出接口工作量 |
| 推测算法 | 普通 decode 两边均关闭；MTP 对照使用相同方法、深度与 draft 制品身份，记录实际启用情况；不能仅传入参数就认定启用 |
| 调度与容量 | 对齐当前上下文上限、最大运行序列数、每步 token budget 与服务约束，记录实际 graph/chunk 宽度和驻留容量；不继续单方面给 vLLM 使用默认 8192/16/2048 |
| 缓存与资源 | 对齐 cache 开关、冷/热准备、设备与显存字节上限；核查实际复用/重算量、KV/state 分配和资源回退 |
| 执行与观测 | 固定 graph/compile、autotune 和日志/计时策略，记录实现差异；正式样本使用相同的测量边界与预热标准 |

不同引擎的内部图宽度、块粒度或算法实现可以不同，但必须在相同服务约束下解释，不能靠同名参数或相同 dtype 标签假定等价。对照条件不能映射、不能实际生效或存在未声明差异时，该 profile 不通过同条件 baseline 验收；只能保存为有明确限制的诊断结果。

只修改 native 内部实现的候选沿用固定 vLLM 参考配置与 ID。若当前 MTP 深度、token budget、精度或其他共同约束发生变化，就为两边建立新的配置 profile 并重测，同时保留旧 profile 的结果和交接对照；不能拿新 native 配置除以旧 vLLM 数字。各引擎分别调优的产品实验另设 profile，不替代本轮要求的当前配置对齐基线。

## 三种对照各有固定身份

| 对照 | 用途 | 必须固定或明确说明的条件 |
|---|---|---|
| native 改动前后 | 判断本次工作的收益 | 同一数值/质量契约、workload、硬件、资源上限和计时；绑定基线源码与候选源码，只改变本次声明的策略 |
| native/vLLM 计算与调度诊断 | 解释差距在哪里 | 相同 token 工作量，实际重算长度；先测 MTP=0、无缓存，再增加指定 MTP 与缓存场景；计算精度和配置差异逐项列出 |
| native/vLLM 服务产品 | 判断用户体验与容量 | 等价用户接口、共同质量要求、同卡资源上限和相同负载；以当前配置对齐为主，各自调优另设 profile；返回文本、停止规则与输出工作量可核查 |

每条对照用独立 profile ID，不能把 token 接口诊断的延迟用于文本服务结论。MTP 首先相对本引擎 MTP=0 的基线测增益，再跨引擎对照；ReplaySSM、DFlash2 与组合也沿用同一规则。

共同数值契约不能只写 BF16/FP8/NVFP4 名称，要记录执行中的权重处理、activation/accumulation、KV、recurrent state 与采样规则。若跨引擎不能保持相同数值规则，就以固定 token 轨迹做诊断，另以预先固定的任务集与质量阈值验收产品对照。保留 target 数值的 native 改动还需输出及状态对照；改变数值的候选使用新 profile，并保留原 profile 结果，不能直接替换旧基线。

## 服务矩阵与判定范围

初始模型范围沿用 Qwen3.8-27B-NVFP4 与 qwen3vl-2b；具体制品、revision 和数值身份由实验清单固定，不能仅凭模型名认定同一输入。每个模型覆盖普通 decode 与当前已支持的 MTP 档位；不支持的算法明确登记能力限制，不能把回退成普通 decode 的结果当作该算法验收。

| 场景 | 必须核查的工作量与用途 |
|---|---|
| short | 单请求短输入；固定实际输入/输出长度，观察低负载 TTFT 和 decode |
| long | 单请求非复用输入；保存实际长度与重算量，不把历史 511 token 外推为长上下文结论 |
| batch4 | 四个实际同时发出的请求；校验四个 slot、发出偏斜与完成情况 |
| hot_long | 明确准备的长前缀；逐轮核查命中和重算量，不能只看累计命中非零 |
| 持续与混合负载 | 固定并发/到达率和输入输出分布，覆盖 prefill/decode 共存；测吞吐、排队、尾延迟、成功率与 SLO goodput |

四个基础场景用于与既有工作交接，服务产品验收还包含持续与混合负载、代表性任务及长度分布。模型支持的上下文范围、期望 case/slot/样本数和未覆盖范围写入清单，局部矩阵只给局部结论。CUDA 图宽度边界、状态接受长度等额外扫描由各模块方案定义，使用相同采集与判定规则。

native/vLLM 的既有延迟目标保留为 ≤1.10x；该目标是性能要求，不是有效基线的资格要求。TPOT、吞吐和尾延迟分别判断；多个模型、场景与指标不合成一个掩盖退化的平均加速倍数。主指标、SLO、允许的取舍和区间判定由本方案的统计要求落实到实验清单。

## 按顺序交付

| 步骤 | 交付 | 放行条件 |
|---|---|---|
| 1. 修正采集与门禁 | 修复报告身份、release 构建校验、真实配置、缓存开关、concurrency 和计时语义；覆盖已审查的错误输入 | debug/构建身份未知、不同模型制品/资源约束、缺字段、单 slot batch4、未生效 cache 开关、失败或截断请求不能被误判通过；合法突发与单 token 输出按指标可用性处理 |
| 2. 固定实验清单 | 每个 profile 的 release 构建、模型、数值/质量条件、资源、输入、输出策略、矩阵、预热与统计规则，以及当前 native/vLLM 配置对照表 | 两边读回的实际配置与清单一致；vLLM 跟当前 native 的共同约束对齐；每项差异有归因范围；质量与数值检查通过 |
| 3. 重建矩阵 | 正确性修复后的 native 与固定版本 vLLM 的完整逐请求结果和遥测 | 期望 case/slot/请求量全部覆盖；实际工作量、cache 状态、完成原因和计时通过校验 |
| 4. 独立复测与复算 | 新进程复测、配对不确定性、从原始结果重建汇总 | 在预先声明的波动与误差要求内复现；采集结果与独立复算一致；不可判定格子明确保留 |
| 5. 固化基线 | 不可覆盖的证据包、机器可读 manifest、baseline ID | 文件齐全、哈希正确且可读取；本契约所有有效性检查通过；后续候选可以直接引用该 ID |

步骤 1 的测试随 [测试组织方案](../engineering/tests.md) 统一收集，避免只运行报告比较测试而漏掉请求采集测试。步骤 3 不沿用旧 `perf-r40` 汇总或手填缺失字段；可恢复的旧证据单独归档，不冒充新基线。

## 清单与运行必须一致

拟议 manifest 至少包含以下字段，缺少必需信息的正式对比判为无效：

| 组别 | 需要保存与验证的内容 |
|---|---|
| 身份 | baseline/run/profile ID、完整源码版本；release 构建命令、features/target、实际 profile/编译选项与二进制哈希；vLLM 及其依赖的固定版本和发行构建来源；模型 shard/config/tokenizer/generation config 的制品清单与指纹 |
| 数值与质量 | 实际权重编码及再量化、activation/accumulation/KV/state dtype、数值规则；参考输出/state/logits 检查及预先固定的任务质量门槛 |
| 硬件与环境 | GPU UUID、型号、显存、driver/工具链、CPU/运行环境；设备频率、功耗、温度和其他 GPU 占用的前后/运行中记录 |
| 生效配置 | 完整命令、影响执行的环境变量、服务端回读配置与 native/vLLM 对照表；上下文容量、token budget、prefill/verify width、slot 数、MTP 深度、cache 状态、预算分解与回退原因 |
| 工作量 | 原始 token 输入与哈希、期望 case/slot、实际长度；固定长度或自然 EOS 策略、采样与停止条件；每请求实际输出、复用及重算量 |
| 测量与判定 | 计时起止点和单位、原始事件时间与 token 数、完整周期、预热记录、配对顺序、统计规则、原始样本、失败原因、区间和质量结果 |

手填的启动参数不是生效证明。两边同为 0.88 的显存比例需要记录实际字节上限和分配；自动缩小图、关闭池化或退化为串行仍要保留结果，并解释其资源代价。硬件 UUID 不同或约束改变时建立新 profile，不能继续称为同一配对。

冷计算要求 cache 确实关闭或缓存被可靠重置，逐请求验证复用为零；热场景先按清单填充缓存，再核查每轮复用和重算量。诊断对照的重算工作量需相同；产品对照允许缓存粒度差异，但要单独解释。一次预热完成不表示所有图、shape、算法与并发路径都已稳定；按正式矩阵预热并保存稳定性证据，加载/JIT/capture 与稳态分开报告。

固定工作量诊断必须达到相同的实际可见输出数量，EOS 策略在两边真实生效；自然 EOS 的任务对照保留完整输出与长度分布，不通过补 token 或删样本伪造等量工作。两类结果分别汇总。文本服务工作量尚不能对齐时，该 profile 不签发服务产品基线，已有 token 诊断按自己的测量边界验收。

工具调用与结构化生成 profile 另保存 tool schema/history、parser 方言、模板、grammar backend 及其版本/指纹、strict/parallel/choice 的实际能力与约束冷/热状态；场景和额外指标见 [Agent 服务方案](../serving/agent-api.md)。工具解析、grammar mask、外部工具执行分别计时，不能将 SSE 事件数当作 token 数。

## 计时与统计不能产生假收益

请求 payload 提前准备；并发组用共同开始屏障，并保存各请求实际发出时间和可用的到达记录。统一定义 request TTFT、最后可见 token 时间、完成事件时间与 group makespan。吞吐按共同窗口的实际总输出计算，TPOT 按每请求摊销，事件 ITL 按原始流式事件计算。短输出、EOS 和一轮发布多个 token 的情况必须由确定时间的协议样本验证公式。

客户端延迟、CPU 阶段和 GPU 周期分别记录，观测开启产生的开销也要对照。需要 profiler 的归因运行与正式计时运行分开，不能拿受 profiler 影响的数值替代服务基线。任何失败、取消、超时、截断、缺失请求或错误 workload 都显式报告，不能从分母或样本中静默删除。

正式测量至少五个独立配对单位作为起点，包含新进程复测并交替 AB/BA 顺序；配对单位、样本量、允许漂移、置信水平、扩样与停止规则在运行前固定，按误差要求继续采样。持续负载与尾延迟使用充分请求量，报告实际样本数；不把同一轮多个 token 当独立证据，不挑最快轮或有利 case。无法达到稳定性要求时，不签发基线。

性能判定保留 `通过 / 未通过 / 尚不能判定`：对延迟上限 ≤1.10x，区间整体支持阈值才放行，跨门槛记为尚不能判定；改动是否有改善也要看相对固定 native 基线的区间和已声明的最小有用收益。有效性检查与性能胜负分开，准确测得退化也是有效结果。主指标、完整矩阵的多指标判定及允许的取舍提前固定，不能事后改门槛。

### 默认服务候选的冻结合同

默认服务候选采用同一 profile 内至少五个 fresh native process pairs，逐对交替 AB/BA；每个进程的 run ID 唯一，baseline/candidate 的模型、输入输出、到达方式、tuning 和资源字节上限一致。C16 makespan 是主指标，其配对 95% 下界必须显示超过 2% 改善；C16 TPOT、short1 wall、32K TTFT、long-to-short 和 constrained 任一指标的对应上界不得超过 2% 退化，跨门槛保留“尚不能判定”。进程内 repeat 不是独立样本，禁止删除失败对或选择性扩样。

C16 与 32K 必须使用不同 profile ID。各自的 context、资源比例和输入哈希独立冻结，不得把不同条件的数字合成一个严格 native/vLLM 对照；启动失败、线程失败和 KV 容量失败不产吞吐。C16 的 69,632 物理 KV token 仅是该固定 `16×(4096+256)` 工作负载的容量下限，不得写进通用规划器。最终跨引擎目标仍为 native/vLLM ≤1.10x。

完整质量沿用原 1,024 例、scorer、EOS、512 cap 和 capped 计数，overall/zero-shot/eight-shot 的原单侧 CP95 gross-regression <2% 必须分别通过。`tools/bench/experiment-checklist.py` 校验 profile 中的进程合同，`tools/bench/process-pair-gate.py` 从全部配对报告重算区间；`tools/bench/serve-compare.py` 在模型加载前拒绝超 context 或并发上限的 workload，并记录参考引擎的线程环境。所有阈值来自 profile，不在工具中按 case 名硬编码。

## 证据保留与基线更新

拟将已验收的 manifest、逐请求样本、输入清单与可复算汇总保存到 `benchmarks/baselines/serving/<baseline-id>/`，供版本管理和后续实验引用。大日志、遥测与 profiler 文件采用可读取的持久存储并记录内容哈希；`artifacts/` 仅作运行暂存，不能成为正式基线证据的唯一副本。现有投影基线保持自己的测量层级，不用来补齐服务基线。

同一 run ID 禁止覆盖，保留失败与未通过的实验。每个候选报告同时记录固定 native baseline ID、固定外部参考 ID、本次唯一改变的因素和各指标原始量，明确“相对自己改善多少”和“与 vLLM 相差多少”。

模型/数值规则、workload/输出策略、资源约束、计时方法、参考引擎或工具链升级时重建相应 profile。新旧条件都可运行时做交接对照，再建立新 baseline ID，旧证据保留；若无法桥接，标为不能直接横比。候选完成正确性、质量、完整性能矩阵与独立复测后，才可登记为新的 native 基线，不能运行完一次就自动更新参考。
