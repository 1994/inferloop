# CUDA 近期提交代码审查

审查日期：2026 年 10 月 9 日。源码基点：`7c79d99`。本文保留该基点的审查结论与验证范围。三项问题已在 `46ed038`（2026-10-10）修正并进入 main；后续工作见 [CUDA 优化方案](../plans/performance/cuda.md)。

## 审查结论

性能方向总体正确。按实际行数选择量化 GEMM tile、保持 prompt chunk 完整、选择最窄够用的 prompt 图，均针对固定图重放的实际浪费。256 行图限制在量化 recurrent 模型且受 arena 预算约束，也符合已有实验结果。应继续沿着形状分派、减少重放和完整资源预算推进。

当前实现有一项 P1 和两项 P2：cuBLAS handle 的设备与 stream 归属不正确；FP8 workspace 缓存遗漏量化模式；投影对照在图外预热之前捕获 vendor GEMM。前两项影响特定配置下的正确性，第三项影响基准的冷启动可复现性。性能收益不能替代这些修正。三项均已在 `46ed038` 修正；本节保留当时的严重程度与判断依据，设备回归按 CUDA 验收入口复核。

BF16 cuBLAS 投影和量化模型的 chunked recurrent 继续保留显式实验开关。它们已有性能收益记录，同时存在 token 差异，chunked recurrent 还存在 short/hot_long 回退。NVFP4 cuBLASLt 只有隔离实验，尚未接入生产执行路径，不能将其微基准倍率写成服务收益。

## 范围与证据

审查范围为 `f7b3ab0..7c79d99`，共 25 个提交、54 个文件，新增 3353 行、删除 384 行，包含文档和测试。提交时间从 2026 年 10 月 8 日 23:26 至 10 月 9 日 10:16，均为北京时间。重点检查 CUDA 图捕获、投影与量化 workspace、prompt 调度、MTP priming 和资源回退。

源码问题来自该范围的 diff 与当前调用链检查。下文性能数值引用仓库的 [性能实验记录](../archive/research/cuda-performance-experiments.md)，本轮没有在 CUDA 设备上独立复测。不同优化记录使用不同阶段的基线，其百分比不能相加。

## P1 cuBLAS context 绑定首次设备和 stream

位置：[cublaslt.rs](../../crates/backend/cuda/src/device/cublaslt.rs) 第 143–146 行、第 180 行和第 509–521 行；[device.rs](../../crates/backend/cuda/src/device.rs) 第 178–182 行。相关提交：`d2e0a88`、`eda4ac7`、`ac7988b`。

`context_for(device)` 使用进程级 `OnceLock<Option<Context>>`，只有第一次调用执行 `build(device)`。handle、device scalar 和 stream 因此都属于首次初始化的设备实例。后续 `CudaDevice::new` 会创建新 stream，但仍取得同一个 context；`GemmBf16::execute` 忽略传入的 `ExecutionContext`，直接通过原 handle 发射 GEMM。

触发条件是开启 `INFER_CUBLAS_PROJECTIONS`，在同一进程中先创建实例 A，再在实例 B 上执行或捕获投影。即使 A、B 使用同一卡，stream 也可以不同。B 的 cast 和其他图节点在 B 的 stream 上，GEMM 却进入 A 的 stream，破坏预期的捕获范围和执行顺序；跨卡时还会出现 handle 与设备指针归属不匹配。具体表现可能是 capture 失败、缺失依赖或错误结果。本轮确认了源码中的归属错误，尚未在硬件上复现其具体表现。

NVIDIA 规定 cuBLAS context 与创建时的 CUDA device 关联，`cublasSetStream` 决定后续调用使用的 stream。当前缓存没有表达这两项归属。[cuBLAS 官方文档](https://docs.nvidia.com/cuda/cublas/index.html)

修正方向是让 handle、scalars 与实际 device/context/stream owner 一起管理，并校验发射上下文。不能仅在共享 handle 上无同步地重复设置 stream，否则并发调用仍可能互相改变绑定。默认关闭实验开关限制了影响范围，但开启后的多实例路径仍需修复。

验收覆盖同卡两个独立实例、不同 stream 的普通执行和 graph capture，以及可用多卡环境。同时检查顺序执行与重叠执行，验证 GEMM 被捕获在正确图中、输出正确、资源在对应 owner 的生命周期内有效。

## P2 FP8 workspace 缓存遗漏量化模式

位置：[workspace.rs](../../crates/backend/cuda/src/resident/nvfp4_gemm/workspace.rs) 第 18 行、第 54–72 行和第 164–167 行。相关提交：`0784220`。

FP8 workspace 只以 `(rows, columns)` 为 key，但 per-channel 与 block-scaled FP8 的 activation scale 布局不同。前者分配 `[rows, 1]`，后者分配 `[rows, columns / 128]`。构造时遇到已有 key 就跳过分配，录制时也只按行列数查找。

同一 program 中两种 FP8 投影使用相同输入几何时，先遍历到的投影决定共享 scale buffer 的形状。以 `columns=5120` 为例，两者分别需要 `[rows, 1]` 与 `[rows, 40]`。后续投影取得的 scale extent 与自身量化 kernel 不匹配，可能造成捕获或执行错误，不能依赖图中投影的排列顺序保证正确。

这是混合量化模式下的条件性缺陷；全程仅使用一种模式的模型不因该项必然出错。本轮确认了 cache key 与布局的冲突，尚未进行混合模式 GPU 复现。

修正方向是将量化模式和必要的 scale layout 纳入 key，或分别管理两种 workspace。验收构造相同输入几何的 channel/block 混合投影，交换分配及捕获顺序，对照独立数值参考，确认两条路径均取得正确布局。

## P2 投影对照在预热之前捕获 cuBLAS

位置：[bench_check.rs](../../crates/backend/cuda/tests/unit/resident_prefill_gemm_bench_check.rs) 第 65–86 行；对照生产路径的 [cublaslt.rs](../../crates/backend/cuda/src/device/cublaslt.rs) 第 335–339 行和 [program.rs](../../crates/backend/cuda/src/resident/program.rs) 第 362 行、第 967 行起。相关提交：`fd32ec5`、`ac7988b`。

对照用例先创建包含 vendor GEMM 的 `gemm_graph`，随后才重放图进行 warmup。新进程首次遇到配置时，cuBLAS 可能需要在主机侧初始化算法或 workspace；图捕获期间不允许这类工作。捕获后的 warmup 无法解决发生在捕获阶段的失败。

生产路径已通过 `warm_delegated` 在 capture 前执行同形状 GEMM。仓库实验记录也明确记载未预热时出现 `operation not permitted when stream is capturing`。基准路径遗漏了同一个前置条件，已有进程内缓存可能掩盖该问题。这里的结论是冷启动基准不可靠，并非所有 CUDA/cuBLAS 配置下必然失败。

修正方向是在真实 stream 上、图外完成每个待测形状、dtype 和 layout 的预热，再捕获、检查输出并测量重放。预热使用已初始化的有效输入，计时仍计入必要的 F32→BF16 cast 成本。

验收从新进程重复运行全部模型形状，检查首次 capture、数值对照和后续计时。不能只在生产路径已预热的进程中执行该用例。

## 各项性能改动的判断

下表的测量依据均来自 [性能实验记录](../archive/research/cuda-performance-experiments.md)，保留审查所需的代表性结果。原始配置、样本和完整矩阵由该记录维护。

| 改动与提交 | 判断依据 | 审查意见 |
|---|---|---|
| 量化 GEMM 按 rows 分派，`b13e169` | 27B long replay 46.9→43.1 ms，linear 27.3→23.1 ms；token 一致，但部分 wall 指标回退 0.3–2.8% | 方向正确。保持小行数和宽行数分派；更宽 tile 在隔离基准获胜仍可能使服务回退，需要同时看完整矩阵 |
| Prompt chunk 原子化，`7d6fe8e` | 5 轮交错 A/B，27B/2B batch4 TTFT 均约下降 19%；2B TPOT 比值 1.549→1.584，约回退 2.3% | 方向正确，消除了碎片付出的完整 replay。不能表述为所有指标无回退；补混合 prefill/decode、尾延迟和预算退化场景 |
| 路由到最窄够用的 prompt 图，`938867b` | 4 轮交错 A/B，27B long TTFT 约下降 21%、wall 约下降 12%，token 一致 | 有服务侧支持。收益来自长块少重放、短块少空跑；继续验证宽度边界和并发负载 |
| 量化 recurrent 的 256 行阶梯，`0e8d531` | 27B long TTFT 0.3061→0.2929 s，约下降 4.3%；两轮、3 条 long 序列一致。2B dense 的 256 行实验无收益 | 限制适用模型且按 arena 预算选择合理，当前样本较少。加入 draft 权重和状态后重新核算宽图与并发预算 |
| BF16 大投影委托 cuBLAS，`d2e0a88` 至 `ac7988b` | 2B 各场景 TTFT 约下降 3.6–10.7%，wall/TPOT 总体变化小；21 条序列中 5 条不同 | 大投影按形状选择合理。先修 P1，继续显式启用；数值门禁通过前不改默认 |
| Chunked recurrent 与 gap tests，`fa27a0a` 至 `141259a` | 27B long TTFT 约下降 24.4%，但 short/hot_long wall、TPOT 回退约 6–11%；21 条序列均与默认不同 | 隔离成量化路径的显式数值选择合理。不能仅凭 long TTFT 改默认，也不能把小 ULP 差异当作 greedy 输出不变 |
| Block-scaled FP8，`0784220`、`5e09f67` | 扩展 checkpoint 格式并加入 block 计算与参考对照；外部引擎记录包含权重再量化差异 | 能力扩展合理，先修 workspace 缓存。外部 token 差异要区分权重处理和算术制度，既不能单独判定错误，也不能代替本实现的正确性验证 |
| 池化失败原因与 OOM 回收，`3a233c2`、`1172e98` | 记录池化关闭原因，在 driver OOM 后回收 cached memory 再尝试 | 方向正确，有助于解释深度升高后的并发性能突变。回收重试不能取代包含权重、回滚状态、图和 workspace 的完整准入预算 |
| MTP draft priming 按捕获宽度分块，`0339a3c` | 使用 draft 实际 prefill 图宽度组织 priming，减少小块重复付出固定图成本 | 方向合理；保持 target/draft 的位置、hidden 范围及 pending seed 协议，不能由少重放推断任意深度和模型都更快 |

### 两条数值路径的边界

BF16 投影委托包含 F32 activation 收窄为 BF16，并改变 GEMM 的计算与累加过程。记录中的 token 差异不能只归因于 vendor 累加顺序。后续应分别比较 cast 误差、算子输出、整模型 logits 和最终 token，再决定是否调整默认数值制度。

Chunked recurrent 当前的短序列回退机制仍未解释。源码与已有纠正记录表明：量化模型的开关作用于宽 prefill 构建，batch/slot/verify 使用逐 lane 路径。不能沿用“verify 也切到 chunked 导致 TPOT 回退”的旧解释。gap map 记录支持存在末位差异；本轮未独立验证 cuTile 编译器收缩是否为唯一原因。

### NVFP4 cuBLASLt 仍是接入候选

`a137cce` 记录的冷 L2 隔离对照支持继续研究 vendor GEMM，但要明确比较基线和投影形状。例如 `(m,n,k)=(12,17408,5120)` 中，出厂 tile 为 0.0630 ms、最佳 native tile 为 0.0427 ms、cuBLASLt 为 0.0369 ms：约 1.71 倍是相对出厂 tile，相对最佳 native 约 1.16 倍。

Prompt 也不能统一概括为“只快 1.1 倍”。记录中 `(64,17408,5120)` 相对出厂 tile 约 1.10 倍，而 `(64,5120,17408)` 约 1.58 倍。收益随投影形状变化。

当前源码接入的是 BF16 `cublasGemmEx` 委托；它与 NVFP4 cuBLASLt 接入是两项工作。后者还需适配权重与 activation scale 的 swizzle 布局，把在线量化、布局转换及 workspace 成本计入，再与最佳 native 和真实服务路径比较。冷 L2 结果也需补充 serving 缓存状态下的对照。

## 验证范围

此前审查已执行 `cargo test -p infer-scheduler -p infer-models`，79 个测试通过，并检查了 diff 空白问题。这支持本地覆盖到的调度与模型逻辑，不覆盖上述 CUDA 条件性缺陷。

本轮未执行 Linux CUDA 构建、GPU kernel、graph capture、多卡或端到端服务复测，也未完成构建改动的全平台验收。因此三项问题分别标明了源码证据与待做硬件验证，性能数据明确作为已有实验记录引用。文档补录只修改 Markdown。

## 修复与验收入口

三项修复进入 [路线图](../plans/README.md) 的 P0 任务 B1；本文保留 `7c79d99` 基点的缺陷与证据，原 P1/P2 缺陷等级保持不变，实际完成状态由路线图维护。

具体实施与验收维护在 [CUDA 优化方案](../plans/performance/cuda.md)，其他详细方案按路线图的工程基础与推测解码分组查阅。
