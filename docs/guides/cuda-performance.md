# CUDA 性能测量与调优

CUDA 执行器使用原生 Rust/cuTile kernel 和设备驻留图。测量按算子、执行图、模型和服务分层；算子加速不能替代完整模型数值、状态与服务验收。构建要求见 [CUDA 后端](../../crates/backend/cuda/README.md)。

本文说明现有工具和测量边界。正式测试与性能工作的 release、当前 native/vLLM 配置对齐和基线验收统一按 [性能基线方案](../plans/performance/baseline.md) 执行；采集与比较门禁已实施，B3 的复测样本与持续/混合负载仍待补。现有服务对比有 [已审查缺口](../reviews/vllm-benchmark-methodology-2026-10-09.md)，历史矩阵不作为当前已验收 baseline。

## 默认优化策略

通过独立数值、完整模型质量与服务性能验收的优化进入默认执行路径；设备/shape 的合法 tile
由内部测量自动选择。用户不需要判断某个优化算法应开还是关。
尚未通过验收的算术与库候选保留在内部测试或独立诊断程序中，生产用户不能通过算法开关启用它们。
profiling、冻结缓存和测试的 A/B 控制是开发诊断工具，不能用来绕过生产资格。

生产 CLI 不再提供 `--chunked-recurrent`；生产环境也不读取 `INFER_CUDA_CHUNKED_RECURRENT`
或 `INFER_CUBLAS_PROJECTIONS` 切换算术。Chunk 与 BF16 Lt 委派由独立诊断程序显式构造，
完成资格后才纳入默认策略。历史实验命令须使用其冻结制品，不能当成当前用户配置建议。

### MTP 按能力选择（显式启用）

默认和显式 `--num-speculative-tokens 0` 都关闭推测解码。显式
`--num-speculative-tokens auto` 才检查 CUDA provider 的单层 dense MTP head、fusion
和 draft graph 绑定；完整且支持时选深度 2，缺失、不完整、不支持、Metal 或显式
prefill cap 为 1 / 2 时选 0。数字深度继续走现有 loader 校验，不静默降级。

这是能力选择，不是质量或吞吐保证。原 1024 题质量门未通过，MTP2 在该 RTX 5090
配置中还使 pooled KV token 容量从 98,304 降至 72,992（约 25.8%）；因此保留显式
启用，默认 attention 仍为 canonical。原失败证据见
[固定质量合同](../archive/research/artifacts/automatic-fast-profile-quality-2026-10-22.json)。

### 实验数值 profile：目标 attention BF16（未获准入）

当前 `main` 已提供 `--cuda-attention-profile canonical|bf16|fp16`；接口可用不代表实验
profile 已获默认准入。固定任务质量门的结论仍是未通过，默认保持 canonical。

- 默认 `canonical`：保持既有精确 attention 路径；目标图的执行与结果不变。
- 显式 `bf16`：只允许目标图的 **FP8-KV prefill** 使用已有 FA4 BF16 QKV / F32 累加 /
  BF16 输出核。目标 F32-KV、不支持的几何、以及没有可用调参/回退证据的路径仍走原有路径。
  混合图中即使某个 FP8 节点使 MMA workspace 存在，目标 F32-KV 节点也按节点级资格回退，
  不会把整个图按 `any` 资格错误地送进 BF16 路径。
- draft/MTP proposal 的既有 BF16 行为不受该开关影响；它不是自动启用 target BF16 的入口。

该 profile 是实验数值选择，**尚未通过固定任务质量门**，不自动启用，也不表示已获质量或
生产准入。目标 BF16 调参使用独立的 `fa4-bf16` schema 与 `resident_mma_attention` 缓存区，
不与 canonical FP8-KV 的 `resident_attention` 缓存混用；加载期 tuning、workspace 定价与
admission 共用同一纯 eligibility 判断。

### 实验数值 profile：目标 attention FP16（仅 SM120，未获准入）

在同一开关上新增 `--cuda-attention-profile fp16`。它只接入已通过算子探针的 SM120
plain-FP16 FA4 制品（`fa4-attention-f16`：FP16 QKV / F32 累加 / FP16 输出），并且只允许
目标图的 FP8-KV prefill 在已验证的 24/4/256 全因果几何上使用。其它架构、tile、几何、
目标 F32-KV、缺少制品或加载失败的路径都显式回退 canonical；draft/MTP proposal 仍保持
原有 BF16 行为，不受该 profile 影响。该 profile 没有新增性能矩阵，也不自动成为默认。

FP16 的 scratch、`prepare_fa4_f16`、AOT kernel、`finish_fa4_f16` 和 host ABI 共用一套
FP16 dtype；host 制品缓存按 `sm120-f16` 名称和字节哈希隔离，tuning/cache 使用独立的
`fa4-f16` schema 与 artifact identity，不能消费 `fa4-bf16` 或 canonical 条目。加载日志和
执行 readback 必须同时证明实际选中并记录了 FP16 kernel；只传 CLI flag 不算路由证据。

## 投影基线

```sh
mkdir -p artifacts
bash tools/bench/cuda-baseline.sh 17408 5120 artifacts/cuda-baseline
```

输出目录必须尚不存在。脚本默认使用 release，保存二进制与源码 SHA256、工具链、构建日志、设备快照和原始样本。矩阵尺寸应取自模型投影清单；`cuda-baseline --catalog /path/to/model` 只解析元数据，不执行完整模型。

BF16 GEMV 候选先对照独立 CPU 点积，再用 CUDA events 测量。A/B 交替、预热、冷 L2 与原始样本的具体参数由工具记录。这条同步投影基线包含分配、清零和提交成本，不能标成裸 kernel 时间。已有可复用测量数据见 [基线目录](../../benchmarks/baselines/README.md)。

## 模型与服务

```sh
cargo build --locked --release -p infer-backend-cuda --features cuda \
  --example cuda-model-smoke
bash tools/bench/safe-run.sh target/release/examples/cuda-model-smoke \
  /path/to/model '只输出17乘23的结果。' 32 \
  --device-graph --prefill-batch 32 --mtp 0 --thinking false --temperature 0
```

模型诊断入口与生产 CLI 服务分别验收。服务运行方式见 [开发指南](development.md)，服务与 MTP 对照工具见 [工具导航](../../tools/README.md)。固定模型、prompt、seed、采样和输出长度；比较 MTP 开关时核对完整 token 序列、接受率、拒绝后的 KV/recurrent state 回滚、EOS 与容量边界。

| 层级 | 必须区分的指标 |
|---|---|
| 单算子 | 数值误差、dtype、shape、预分配范围、CUDA event 延迟 |
| 执行图 | 捕获/JIT、首次运行、稳态 replay、主机提交、prefill/decode/draft/verify |
| 完整模型 | TTFT、TPOT、有效 tokens/s、峰值显存、MTP 接受率 |
| 并发服务 | 到达率、并发、拒绝/取消、P95/P99、满足 SLO 的 goodput |

`safe-run.sh` 默认限制任务主机内存为 32 GiB、禁用任务 swap，并要求限额之外保留至少 16 GiB 可用内存；`--memory-gib` 可降低限额。报告与 profiler 输出先放在 `artifacts/`；正式基线按 [证据留存规则](../plans/performance/baseline.md#证据保留与基线更新) 持久保存，不写入使用文档作为逐轮日志。

## 通用 Attention 对照

`make check-attention` 将原生实现与隔离的 Candle FP16/BF16 基线比较。它同时限制数值误差、完整流程、核心调用、逐形状延迟、P95 与跨轮稳定性，详细契约见 [质量门禁](quality-gates.md#通用-attention-门禁)。

保留的验收结论：2026-10-07 RTX 5090 / CUDA 13.4.92 / Candle 0.11.0，31 个用例、5 轮测量中，完整流程几何平均加速约 1.25×，核心调用约 0.69×；长序列退化，结果为 `review`（非零退出），性能未放行。加速比为 Candle 耗时 / 原生耗时。原生补偿 F32 与 Candle FP16/BF16 的精度契约不同；该结果不能外推其他设备，也不能用完整流程收益掩盖核心退化。

## 能力与精度

设备能力由 driver 查询，模型 provider 声明 `PrecisionPolicy`，加载时选择合法存储和计算路径。支持原生 FP4 与支持某个高效 Tensor Core tile 是两件事；数值通过也不证明编译器生成了预期指令。NVFP4→BF16 加载转换不会恢复原始权重精度，并会增加显存需求。Hopper 转换路径不等于已完成 H200 实机验收。

NVFP4 lowering 探针为 `cuda-mma-probe`，用 `tools/bench/inspect-cutile-cache.py` 检查 cubin/SASS；固定设备上的能力证据保存在 `benchmarks/capabilities/`。变更 tile 或工具链需重做数值、lowering 与性能验证。

MLP 的 `MlpConfig.pdl` 默认关闭，仅控制 SiLU×up → down projection 依赖边。生产者 signal 必须依赖 store token，消费者 load 必须依赖 wait token，graph 保持缓冲区生命周期；具体安全证明在 `mlp/pdl.rs` 与 `mlp/pdl_consumers.rs`。显式启用前校验工具链和架构，使用 `cuda-mlp-check --pdl` 验证。正确性通过不代表已证明 PDL 加速。

## 调优规则

1. 候选先校验独立参考、尾块、非整除维度与状态语义，再测量。
2. 按实际 dtype、shape、layout、执行阶段和设备测量；prefill、decode、MTP 不共享未经验证的性能结论。
3. 构建身份、配置对齐、冷/热条件、采样统计与证据按性能基线方案核验。
4. 搜索结束后独立复测；驻留 GEMM 用 AB 搜索、反序 BA 确认，不能用最快一次样本替换默认实现。
5. 用 profiler 定位访存、提交、occupancy 或 spill，再决定融合与流水化；功能边界由通用 API 保持。
6. 解码 slot 数与 KV 容量改变会影响常驻显存、准入与并发，必须连同完整服务负载重测。

### 驻留 GEMM 自动择优

驻留 NVFP4 与按通道缩放的 FP8 GEMM 在实际图捕获前选择配置，使用同一 `tuning.json` 的独立
typed section，避免与标量 GEMV 策略混淆。缓存和迁移规则见
[CUDA 后端](../../crates/backend/cuda/README.md#自动-tile-选择)：GEMV 本地测量优先于 shipped 基线，
旧单设备条目迁移到多设备表，缺失设备身份的记录不复用。

驻留 key 隔离硬件事实、CUDA API 与真实 NVRM driver build、编译器及 kernel fingerprint、
实际 `M/N/K`、激活模式、scale/布局与数值合同；Lt 另外隔离库版本和 typed 算法配置。
同形状可复用首份真实权重的性能搜索，但每份实际权重和 scale 仍需通过所选配置的完整输出
finite/bitwise 验收。验收记录关联仍存活的权重 owner，不靠形状或可复用裸地址跳过新权重检查。

默认原生 NVFP4 搜索现有 row/column tile 及 plain/prefetch main loop，保留 K 步长、block scale
和量化算术；原生 FP8 搜索同一 K 累积顺序的 tile。选择由实测决定，原先按 SM wave 阈值放大
NVFP4 列 tile 的策略已删除。设备能力、合法布局、对齐、workspace 和容量/显存上限仍是硬约束，
不能通过测量绕过。原生 gated norm 的编译器 occupancy 也按同一实测择优流程选择，其算子、完整模型
与 native 服务验收见[占用率调优证据](../archive/research/artifacts/cuda-gated-norm-occupancy-2026-10-15.json)。
Attention、图宽度和 GEMV/GEMM crossover 尚未全部纳入这个 chooser。

搜索按固定 AB 顺序成对测量，独立确认按 BA 反序成对测量，并使用新的 capture。两轮分别覆盖
冷 L2 和暖 L2，非 baseline 配置在四组比较中都至少优于保守配置 2%。原始 samples、两臂名称、
中位数、比值和顺序随策略保存，读缓存时校验完整性。计时范围是预分配的“激活量化 + GEMM”
graph replay，不含模型加载、分配或首次 JIT。单候选的 capture、数值或测量失败不终止其余合法
候选；基线失败仍回退，不能据此假设 CUDA 错误已恢复。

CUDA 服务 CLI 默认先在准备进程中调优并保存缓存，再由干净的服务进程加载这些配置，避免
搜索候选留下的设备模块挤占 KV 容量。准备进程不监听服务端口；父进程退出会终止整个准备
进程组，包括 JIT 子进程。准备最多运行 10 分钟；失败或超时后，服务仍可使用已验证缓存或
保守配置。`--no-autotune` 跳过准备进程。没有可写缓存时保留进程内调优；直接使用 SDK 的
`LoadOptions` 也保持进程内调优语义。两阶段启动需要额外加载一次模型，不能承诺启动更快。

自动调优默认开启。每次模型加载的 target、draft 和 program 共享 30 秒 GEMM 新搜索软预算，
首次新搜索才启动；greedy、norm、recurrent、attention 各有独立的 30 秒软预算。已开始的
JIT/测量不可中断，因此它们不是整体启动耗时的硬上限。搜索超时或证据未完成时回退，不把
未测 baseline 当成完整调优证据缓存。

GEMM 缓存身份解析和已缓存配置的当前真实权重数值验收不受新搜索预算限制，避免一个未缓存
形状耗尽预算后，使后续已验证配置无法生效；它们仍可能增加启动时间。配置必须匹配完整身份，
并通过当前权重的完整 finite/bitwise 验收，才能用于捕获。`--no-autotune` 仅禁止新的性能搜索，
仍自动复用有效缓存并完成这些数值验收。arena 预算和解码分片由 `tuning/`
按设备资源推导，量化 tile 由 GEMM chooser 选择，不提供环境变量覆盖器。宽 prompt 的 FP8
投影默认使用设备库查询到的 cuBLASLt 算法；Lt 独立诊断仍只存在于 CUDA 测试程序。

### 行相关 prompt GEMM tile

prompt GEMM 的保守基线按输出行数分档：≤16 行用 16×64，17–127 行用 64×64，≥128 行用
128×64；≥512 行的 W4A4 图默认使用 128×128 和 prefetch 主循环。128 行 tile 让 256 行 chunk 变成两个 CTA 而不是四个，减少同一 weight tile 的重复
读取；不足 128 行的尾块仍用 64 行，避免把半个 CTA 算在空行上。decode 与 MTP verification
仍走 16 行，不继承 prompt tile。

在 RTX 5090、32K C1、BF16 target、MTP2、表 B 的 A B B A 诊断中，把 prompt tile 全局改为
128 行（当时使用的手工 tile 诊断覆盖器现已移除）使 TTFT 配对均值 −0.54 s（12 对，中位 −0.47 s），
反序确认 −0.50 s（18 对）；同一覆盖也把 decode 降到 128 行 tile，decode 中位从 57.1 降到
46.9 tok/s，因此生产实现只对 ≥128 行的图取宽 tile。历史覆盖实验的结果不写入调优证据；
128 行仍是自动搜索的候选，搜索可在具体 shape 上改回 64 行。

这是性能诊断，不是质量、release 或 vLLM 追平结论；固定 64-token greedy 流在所有臂一致。
证据保留在 `staging/prompt-gemm-tile-20261018/`。

### FP8 Lt 内部候选验证

生产程序自动选择已验收的原生 FP8 tile，不读取 `INFER_CUDA_FP8_LT_AUTOTUNE` 或
`INFER_CUDA_VENDOR_FP8` 来改变算法。Lt 候选搜索和独立诊断仅保留在 CUDA 测试程序中。
完整输出 finite/bitwise、真实权重四种 activation pattern、冷/暖 AB 与反序 BA 确认仍用于
内部筛选；SDK 不保证相同 K 累积顺序，这些局部检查不能替代完整模型和服务验收。
候选合格后才能纳入自动选择，不让用户决定是否接受候选的数值差异。

下列 release 入口是独立候选 sweep，可保存逐形状冷/热 L2、三轮 A/B 交替的 CUDA event 样本；
它不替代 chooser 的反序确认、实际权重验收或服务对照。
计时范围为预分配的“激活量化 + GEMM” graph replay，不含加载、分配或首次 JIT：

```sh
bash tools/bench/safe-run.sh --memory-gib 8 cargo test --locked --release \
  -p infer-backend-cuda --features cuda --lib nvfp4_prefill_performance_comparison \
  -- --ignored --nocapture --test-threads=1
```

投影自动 tile 选择与缓存规则见 [CUDA 后端](../../crates/backend/cuda/README.md#自动-tile-选择)。缓存匹配不能替代工具链或 kernel 变更后的回归验证。

### 完整模型任务质量复算

改变 GDN 运算顺序的候选保持 F32 状态，先通过独立数值参考，再用完整模型生成结果验收任务质量。
共同实验条件沿用 [性能基线方案](../plans/performance/baseline.md)，本次单步定位和 WY 数值范围见
[F32 GDN 验收记录](../archive/research/cuda-gdn-f32-quality-contract-2026-10-12.md)。

`tools/bench/score-task-quality.py` 接收预先固定、含标准答案与输入 token 的 suite，以及两份完整报告：

```sh
python3 tools/bench/score-task-quality.py \
  --suite /path/to/tokenized-suite.json \
  --baseline /path/to/scalar-results.json \
  --candidate /path/to/wy-results.json \
  --output /path/to/new-quality-comparison.json
```

工具从原始文本的最终 `#### number` 行重新提取 Decimal 答案，验证逐题 ID、输入、完成原因、零前缀复用、
profile 与模型/设备身份；缺题、重排或手填分数与文本不符均失败。长度上限样本保留在分母，并单列截断及
格式缺失数量。置信水平和允许的正确率下降来自 suite，采用单侧 Clopper–Pearson 回归率上界；
新增答对不能掩盖原先答对而候选答错的题。报告为 `pass` 或 `not_demonstrated`，后者非零退出。

这项判定适用于声明的算术任务子集与生成预算。它独立于 kernel 数值、构建身份和服务性能门禁；
`quality_accepted` 不证明全面模型能力或性能目标已达到。`tools/check/gate.sh` 自动收集其反例与统计参考测试。

宽提示图的调参覆盖与完整模型质量须分别核验。同一调参入口的窄图通过，不能证明新宽图列分块
也通过；关闭 autotune 的冻结缓存没有某形状条目时，会使用保守配置。最新
[长提示调参覆盖与独立准备记录](../archive/research/cuda-wide-prefill-tuning-2026-10-13.md)
保留完整缓存的计时收益、未通过的质量回归，以及 GEMM/GDN 拆分控制。
