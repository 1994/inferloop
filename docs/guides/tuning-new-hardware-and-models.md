# 新硬件与新模型的调优流程

本文面向 coding agent 和开发者。换一块 GPU，或者接入一个新模型时，按这里的顺序检查、测量、调优，并把结果落成**自动默认行为**。用户不需要也不应该配置任何调优开关。

本文依据 2026-10-25（America/Los_Angeles）的 `main` 与 PR #47 的实际代码整理。代码与本文不一致时以代码为准，并顺手更新本文。

## 原则

1. **推导优先，实测其次，常数最后。** 能由设备属性（SM 数、显存、带宽、L2、寄存器与共享内存预算）和模型属性（权重字节、每序列状态、每 token KV 字节、MTP 深度）推导的量，用公式算出来。推导不了的，交给加载期的自动调优实测。两者都不行时才写常数，并在注释里写明测量来源、设备和形状。
2. **不加用户开关。** 实测更快的路径直接成为调优器的默认候选，或者成为默认实现。`INFER_CUDA_*` 环境变量只用于开发诊断（见“诊断工具”），不能作为功能开关交付。
3. **数值不变的优化，用逐位证据放行；数值改变的优化，用质量门禁放行。**
   - 逐位不变的改动：要求 GPU 单元测试比较新旧内核的输出字节，并要求端到端运行的 token 摘要完全一致（`fixed-work-serving.py` 会记录）。这类改动不需要跑 1,024 题。
   - 改变数值的改动（精度、累加顺序、批宽导致的数值差异）：必须通过 [任务质量门禁](cuda-performance.md#完整模型任务质量复算)，即配对正确率非劣。
4. **性能结论只来自同机、新进程、交替顺序的测量。** 方向筛查可用每臂两个新进程、BAAB 交替，每个进程丢弃 1 组 warmup，再取 3 组的中位数；这不构成正式性能放行。正式资格按[基线契约](../plans/performance/baseline.md)预先冻结至少五对新进程、运行顺序、指标和停止规则，由 `process-pair-gate.py` 判定；不能把筛查结果或单算子提速当成跨负载保证。profile 运行会扰动计时，只能用来定位瓶颈。

## 加载期已经自动完成的部分

以下内容不需要人工操作，但在新硬件上要确认它们确实生效了。

| 项 | 机制 | 落盘位置 / 日志 |
|---|---|---|
| 设备身份 | `DeviceProfile`：型号、架构、SM 数、显存、显存时钟、位宽、L2；驱动版本和驱动构建号 | 所有调优键的一部分，换硬件或驱动会自动重新测量 |
| 调优预热 | 服务启动前先起一个独立的预热子进程，超时 600 s，测完后释放资源；服务进程复用缓存 | 日志 `CUDA tuning preparation complete` |
| 调优缓存 | `$XDG_CACHE_HOME/infer-cuda/tuning.json`，未设置时为 `~/.cache/infer-cuda/tuning.json` | 每个条目带配对计时证据 |
| NVFP4 投影 tile | `tuning/resident_gemm.rs`：按形状 (M, N, K) 搜索 tile_m、tile_n 和主循环（Plain / Prefetch8），AB 搜索加 BA 确认 | `resident GEMM selected ... source=` 为 `measured`、`table-cache` 或 `autotune-disabled` |
| FP8 投影 | `tuning/fp8_gemm.rs`：原生 tile 和 cuBLASLt 候选一起实测 | `FP8 GEMM choice ... selected=` |
| attention、递推、norm | `tuning/attention`、`tuning/recurrent`、`tuning/gated_norm`：各自按形状实测 | `preparing hardware attention choices` |
| 显存预算 | 预算 = `min(显存总量 × gpu_memory_utilization, 空闲 − 1 GiB)` | `device budget ... of ... total` |
| 槽位与 KV | `loading/budget.rs`：扣除固定开销后为 KV 页面分配预算；槽数 = KV token ÷ 每槽保底 | `planned pooled CUDA serving storage planned_width=... kv_tokens=...` |
| prompt 宽度 | `loading/mod.rs`：按 arena 预算在若干档宽度中选择 | `prompt width rung width=...` |

**新硬件上要检查的日志：**
- 没有出现 `GEMM shape search budget exhausted`。每个 resident GEMM shape 的 allowance 由候选数和 paired measurement 上限推导，先前 shape 不消耗后续 shape；若出现，日志列出未完成的 family/shape。非 GEMM 类别仍有各自独立的有限额度。
- 没有出现 `conservative fallback`。
- 槽数和 KV token 数与显存相符。

## 新硬件检查表

按顺序执行，每一步都保存证据（日志、JSON、二进制的 sha256）。

1. **编译。** `cargo build --locked --release -p infer-cli --features cuda`。注意：`make check-rust` 会把 `target/release/infer` 重新编译成不带 CUDA 的版本，测性能前要重新编译。
2. **看架构相关的内核是否可用。** 以下实现按架构开放，其他架构会自动回退到通用路径：
   - FA4 prompt attention：`resident/attention_mma/catalog.rs`，目前覆盖 sm_90、sm_100、sm_103、sm_120；
   - oxide attention 与 recurrent prefill：`resident/attention_oxide*.rs`、`resident/recurrent_prefill/oxide.rs`，只支持 sm_120；
   - FP16 FA4：`resident/attention_mma/f16.rs`，只支持 sm_120。

   新架构上要确认回退路径正确，并记录哪些内核需要补充产物。NVFP4 W4A4 依赖 FP4 tensor core（Blackwell）；在 Hopper（sm_90）上需要另行确认 NVFP4 checkpoint 实际走的是哪条路径。
3. **冷启动调优。** 换一个干净的缓存位置（只为开发诊断设置 `INFER_CUDA_TUNING_TABLE=<新文件>`），用零参数启动一次，等预热结束。然后按上面的日志检查表确认每个形状都是 `measured`，没有预算耗尽或回退。
4. **正确性。** 运行 `make test-cuda-hosted`。和本次改动相关的 `#[ignore]` GPU 测试要在 safe-run 里单独运行：先用 `cargo test --no-run` 编出测试二进制，再执行 `tools/bench/safe-run.sh --memory-gib 24 <测试二进制> <名称过滤> --ignored`。
5. **性能基线。** 用 `tools/bench/fixed-work-serving.py`，分别跑零参数默认配置和与 vLLM 对齐的配置，按 BAAB 交替。用 `--profile` 跑一次，再用 `tools/bench/graph-profile-summary.py` 得到每张图、每类算子的 GPU 时间。
6. **和参照引擎逐形状对比。** 对占比最高的投影形状，测参照引擎的内核（如 vLLM 的 CUTLASS NVFP4 / FP8 GEMM）。计时要包含激活量化，并且**在多份权重副本之间轮换，使总量超过 L2**，否则小 M 形状会读到 L2 缓存，得出高于显存带宽的错误结论。
7. **质量。** 只有当数值路径改变时，才跑 1,024 题质量门禁。
8. **复查推导项。** 对照下文的“待改为推导的常数”，确认这些常数在新硬件上是否仍然合理；不合理的，按公式改写，不要只换一个新数字。

## 新模型检查表

1. **模型图。** 拓扑由 `crates/model/recipes` 构建（参考 [新增模型](adding-a-model.md)）。确认：
   - 量化格式在支持范围内：NVFP4 W4A4（`weight_global_scale` 加 `input_global_scale`）、FP8 按通道、BF16 稠密；
   - MTP head 是否存在、能否被自动选中；
   - recurrent 状态精度：例如 Qwen3.5 系列 config 中 `mamba_ssm_dtype: float32`，vLLM 也按 F32 存储。
2. **按图结构匹配的融合是否生效。** 以下融合都通过识别图中的特定模式来触发，新模型的结构稍有不同就可能不再生效：
   - gate/up 加 SwiGLU 加 FP4 量化融合（`prefill_projection.rs` 的 `MlpPlan`）：要求 gate 与 up 读同一输入、激活缩放相同，输出只被 SiLU 和 multiply 读取；
   - GDN `in_proj_b`/`in_proj_a` 合并（`batch.rs` 的 `gdn_ba_pairs`）：按图结构识别同一 `Delta` 节点的 beta/alpha 生产者，再核对同层、同输入和投影形状；prefill 与 verify 的批处理共用同一对 kernel；
   - norm 与 FP8 行最大值融合（`norm_maximum.rs`）、同输入的量化复用（`nvfp4_gemm/quant_reuse.rs`）。

   检查方法：用 profile 的节点数和各算子耗时，对比融合生效与不生效时的预期。也可以在对应的计划函数里打印命中数。
3. **显存规划。** 每序列固定状态（recurrent、draft）和每 token KV 字节由图推出（`loading/budget.rs`），一般不需要改。要确认的是零参数默认下的槽数是否合理：槽数偏少时，先找能压缩的固定开销，不要调高显存占比。
4. **质量与性能。** 同新硬件检查表的第 5–7 步。

## 诊断工具

这些只用于开发诊断，不是用户配置。

| 工具 | 用途 |
|---|---|
| `tools/bench/fixed-work-serving.py` | 单进程固定输出量计时：零参数或指定参数启动，两种引擎都忽略 EOS，记录 token 摘要，`--profile` 采逐节点事件 |
| `tools/bench/graph-profile-summary.py` | 汇总逐节点事件，按图（prefill、slot_verify、slot_decode……）和算子类型排序 |
| `tools/bench/score-task-quality.py` | 配对正确率非劣质量门禁 |
| `tools/bench/safe-run.sh` | 带主机内存上限、并发互斥与时长保护地运行 GPU 进程（每个进程最长 30 分钟） |
| `INFER_CUDA_PREFILL_PROFILE` / `INFER_CUDA_EXECUTION_PROFILE` | 逐节点 CUDA 事件 / host 阶段记录（由 `--profile` 设置） |
| `INFER_CUDA_PROFILE_GRAPH_ONLY` | 仅在图边界记录事件，配合 prefill profile；不改变算法 |
| `INFER_CUDA_ATTENTION_PRECISE_PREFIX` | 恢复前若干位置的精确 attention 诊断；不改变默认数值路径 |
| `INFER_CUDA_TUNING_TABLE` | 指定调优缓存文件，用于冷启动调优和隔离实验 |
| `INFER_CUDA_VENDOR_FP8` / `INFER_CUDA_FP8_LT_AUTOTUNE` / `INFER_CUDA_TILED_ATTENTION` | **仅测试构建**可读；生产 `cfg(not(test))` 不读取，不是产品开关 |
| `INFER_CUDA_CHUNKED_RECURRENT` | 已撤下；生产不读取，chunk 候选只在独立诊断 loader 中 |

## 待改为推导的常数

下列值目前写死，换硬件或换模型时可能不再合适。改动时用推导替换，或者交给调优器实测，不要只换数字。

| 常数 | 位置 | 现状 | 应改为 |
|---|---|---|---|
| `DEFAULT_MAX_BATCH = 16` | `engine/runtime/src/config.rs`、`spi::SchedulingRecommendation`、`executor/mod.rs` | CLI 自动路径从 backend 取得实际建议；直接构造 `RuntimeConfig` 的可移植默认仍为 16。自动路径消费 `SchedulingRecommendation = min(N_mem, N_knee, 显式 max_num_seqs, CB16/提交 ABI)`。`N_mem` 用实际 pooled quote 的固定 shared owners + 每序列 state + 最小 KV block 推导，不重复计入私有验证图；`N_knee=(实测部署模式 compute FLOP/s / DRAM B/s) × (实际权重字节 / 实际逻辑参数量) / 2 / 每步 token 数`。compute 输入复用 resident GEMM 调优器中已实测的最大 compute-bound NVFP4/W4A4 样本（含实际 kernel identity、mode、shape、时间与 FLOPs）；未知/无实测模式返回 `None`，不伪造成标量 FP32 峰值。 | backend 提供 `SchedulingRecommendation` 并记录 compute mode/kernel/shape/ms 与全部 raw inputs；显式 `--max-num-seqs` 始终优先，未知硬件保留诚实 fallback |
| `VALUE_SPLIT_*`（按 SM 数 63/127 分档） | `resident/slot_recurrent.rs` | 已删除 SM 分档 | 由实际 SM、covered value heads、槽数和 value 维合法除数推出；目标是一波 CTA，尾宽只接受可整除的 split |
| pooled attention `MAX_BATCH = 4`、切分数上限 64 | `resident/attention_decode/pooled.rs`、`tuning/attention/decode.rs` | 每组 4 个槽串行；切分数已按 SM 数、KV 头数与每组槽数推导，尚未结合上下文长度 | 切分数按上下文长度推导（每份至少若干 KV 块），部分和 workspace 随之缩小，从而一次启动处理全部槽 |
| `SWIGLU_TILE_ROWS/COLUMNS = 64` | `resident/nvfp4_gemm/workspace.rs`、`tuning/swiglu.rs` | 加载期真实权重 tuner：合法候选 64x64/128x64/128x128，完整 codes/scales 逐字节对照 baseline，量化-inclusive AB+BA 配对与持久 device/kernel/geometry/weight 证明；64x64 为无证据 fallback | 128 行只有在 exact proof 和两种顺序都超过确认阈值时才被选中；搜索失败/缓存损坏回退并记录缺口 |
| `GDN_BA_SHAPE = [48, 5120]` | `resident/prefill_projection.rs` | 按形状识别 b/a 投影对 | 按图结构识别（同输入、相邻、同为稠密小投影），不依赖具体形状 |
| narrow dense N16 exact `(64,48,5120)` | `resident/prefill_projection.rs`、`tuning/prefill.rs` | 原按完整 shape 硬匹配 | 已改为 48-row、64-multiple reduction 的小投影结构契约，prompt width 不参与匹配；GPU 性能/数值仍需设备 |
| `MAX_POOL_ROWS = CB_DECODE_SLOTS` | `resident/pool_gemv.rs`、`tuning/pool_gemv.rs` | 16 仍是未测量/失败时的 legacy fallback；加载期已按实际 Dense BF16 权重和 F32 输入测量 GEMV vs blocked GEMM | 按实际 rows/columns/outputs/权重证明持久化选择；cold+warm AB+BA，数值仅允许既有 F32 边界并保留最终 1024 门禁 |
| `CATEGORY_SEARCH_BUDGET = 30 s` | `tuning/resident_gemm.rs` | 非 GEMM 类别的独立有限额度 | Resident GEMM 已改为按 family/shape 的候选数与测量时长推导；其余类别保留独立限制并在日志中记录未完成项 |
| 每槽保底 KV = 单槽容量 ÷ 2 | `loading/budget.rs` 的 `funded_kv_tokens` | 已删除；每槽只保留一个物理页，实际 prompt+generation 页由 admission 按请求租用 | 保留 fixed recurrent/draft state 和 pooled backpressure；长短请求、取消/退款与 prefix 复用仍需设备矩阵 |

## 参考：一次完整调优的记录

PR #47 是一个完整的例子：把逐算子 profile、逐提交二分、和 vLLM 逐形状对比、质量门禁与固定输出量计时组合起来，把 RTX 5090 上零参数默认 C16 的耗时从 17.4 s 降到约 13.3 s。各步骤的证据路径见 PR 描述。

## 已测 split-K 实验的边界

[RTX 5090 的 NVFP4 split-K 筛查](../archive/research/artifacts/nvfp4-split-k-screen-2026-10-25.json)
覆盖 48 行的 gate/down 投影、三种 tile 和 2/4/8 分片，使用三个独立进程。
数值尾部回归通过，但合成权重微基准中 gate 最佳候选仍慢约 15.9%，down 最佳候选快约 11.3%。
因此未启用通用 split-K 默认策略。down 专项仍需实际权重选择、临时内存报价、服务性能和任务质量验收；
不能把这组算子收益当作服务收益或正式性能资格。实验源码与全部样本身份见记录。
