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
| 槽位与 KV | `loading/budget.rs` 与 `executor/admission_pool.rs`：扣除固定开销后按实际请求长度报价；冷启动先报告上限，reservation 选择实体池 | `planned pooled CUDA serving storage planned_width=... kv_tokens=...` |
| prompt 宽度 | `loading/mod.rs`：按 arena 预算在若干档宽度中选择 | `prompt width rung width=...` |

**新硬件上要检查的日志：**
- 没有出现 `resident projection search budget exhausted: missing shapes`。投影预算按实际录图会使用的唯一形状及候选计时窗口估算；若 JIT、数值验证或设备开销使估算耗尽，日志会列出未完成的 `(类型, M/N/K)`，这些形状继续使用保守默认配置，不保留不完整证据。
- 没有出现 `conservative fallback`。
- 槽数和 KV token 数与显存相符。

投影准备先分配既有量化缓冲区，再用录图本身的 MLP 融合计划确定活跃投影。
融合 gate/up 的 prompt 宽度不搜索独立 gate/up GEMM；decode、verify、额外消费者或不满足融合条件的图仍保留独立投影。
搜索额度为每个唯一 `(数值路径, M/N/K)` 的候选 AB 轮次、独立 BA 确认及必要的 native 确认之和，
每轮按冷/热两种状态乘以现有 `WARMUP_MS + REP_MS` 测量窗口计算。
重复权重不续期；新增宽度或新增候选只增加相应工作量，不重启计时。
只累计新搜索本身的用时；缓存命中的实际权重校验和其他准备工作不消耗搜索额度。
这是加载期计时工作量估算，正在执行的 CUDA JIT 与数值验证不能被中断；外层准备进程继续遵守既有时长保护。
attention、recurrent、norm、greedy 的独立 allowance 保持原有边界。

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
7. **迭代与发布。** 普通改动运行 `make check-iteration`：一次原生进程对、性能指标和分批快速失败的 64 条原始质量任务，总时限十分钟。1,024 题、五对以上进程与长上下文/受限资源矩阵只在明确的大版本验收中通过 `make check-release` 运行；64 条筛查不构成正式质量资格。
8. **复查推导项。** 对照下文的“待改为推导的常数”，确认这些常数在新硬件上是否仍然合理；不合理的，按公式改写，不要只换一个新数字。

## 新模型检查表

1. **模型图。** 拓扑由 `crates/model/recipes` 构建（参考 [新增模型](adding-a-model.md)）。确认：
   - 量化格式在支持范围内：NVFP4 W4A4（`weight_global_scale` 加 `input_global_scale`）、FP8 按通道、BF16 稠密；
   - MTP head 是否存在、能否被自动选中；
   - recurrent 状态精度：例如 Qwen3.5 系列 config 中 `mamba_ssm_dtype: float32`，vLLM 也按 F32 存储。
2. **按图结构匹配的融合是否生效。** 以下融合都通过识别图中的特定模式来触发，新模型的结构稍有不同就可能不再生效：
   - gate/up 加 SwiGLU 加 FP4 量化融合（`prefill_projection.rs` 的 `MlpPlan`）：要求 gate 与 up 读同一输入、激活缩放相同，输出只被 SiLU 和 multiply 读取；
   - GDN `in_proj_b`/`in_proj_a` 合并（`batch.rs` 的 `gdn_ba_pairs`）：目前按形状 `[48, 5120]` 硬匹配，见待改清单；
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
| `INFER_CUDA_TUNING_TABLE` | 指定调优缓存文件，用于冷启动调优和隔离实验 |

## 内部诊断开关与生产边界

生产程序不读取下列路径选择实验开关；它们只存在于内部 CUDA 测试程序，普通推理
无需设置任何 `INFER_CUDA_*` 变量：

| 内部测试变量 | 诊断用途 |
|---|---|
| `INFER_CUDA_ATTENTION_PRECISE_PREFIX` | 对照 canonical 前缀与完整 FA4 prefill |
| `INFER_CUDA_RECURRENT_VALUE_SPLIT` | 对照 recurrent value 分块，不能覆盖生产加载期实测选择 |
| `INFER_CUDA_DEVICE_GREEDY` | 对照 GPU token 读出与原始读出 |
| `INFER_CUDA_TILED_ATTENTION` | 比较尚未通过完整质量门槛的 tiled attention |
| `INFER_CUDA_FP8_LT_AUTOTUNE`、`INFER_CUDA_VENDOR_FP8` | 独立 vendor FP8 搜索与数值诊断 |

生产保留的变量仅用于开发者记录与隔离：`INFER_CUDA_PREFILL_PROFILE`、
`INFER_CUDA_EXECUTION_PROFILE`、`INFER_CUDA_PROFILE_GRAPH_ONLY` 控制性能记录，
`INFER_CUDA_TUNING_TABLE` 指定隔离的调优缓存路径。默认推理自行解析硬件能力、
图结构和调优结果，不依赖这些变量选择可用算法。历史实验文档里的开关命令描述
当时的冻结版本，不能作为当前生产配置步骤。

## 待改为推导的常数

下列值目前写死，换硬件或换模型时可能不再合适。改动时用推导替换，或者交给调优器实测，不要只换数字。

| 常数 | 位置 | 现状 | 应改为 |
|---|---|---|---|
| 默认并发（已推导） | `loading/concurrency.rs`、`executor/execution.rs`、runtime config | CUDA 自动模式取完整显存报价与计算/带宽拐点的较小值，录图成功后报告实际槽数 | runtime 在分配前应用 backend 的建议及更小的显式上限；64 仅是完成协议容量，不是性能默认值 |
| GDN value 切分（已推导） | `resident/slot_recurrent.rs` | 读取设备 SM/warp 数和当前节点的 value 头数、维度、槽数 | 在每个 value 分块不宽于设备 warp 的约束内，选择最接近一条 SM wave 的合法网格；按 value 维的二次幂整除约束取值，不再按 SM 数分档 |
| pooled attention `MAX_BATCH = 4`、切分数上限 64 | `resident/attention_decode/pooled.rs`、`tuning/attention/decode.rs` | 每组 4 个槽串行；切分数已按 SM 数、KV 头数与每组槽数推导，尚未结合上下文长度 | 切分数按上下文长度推导（每份至少若干 KV 块），部分和 workspace 随之缩小，从而一次启动处理全部槽 |
| 融合 SwiGLU tile（已接入调优） | `tuning/resident_gemm/swiglu` | 加载期按实际融合形状比较合法行/列 tile，独立缓存 | 完整 codes/scales 逐位校验，冷/暖两种状态与反向独立确认后选用；预算不足或候选失效保留原 `64×64` 基线 |
| 稠密投影识别（已推导） | `resident/prefill_projection/dense.rs` | 从图声明的输入与权重维度识别相邻、同层、同输入的稠密投影对和小输出投影 | 不再匹配固定模型维度；录图时复核实际稠密权重、尺寸与别名，保留 vendor、稀疏和单行 logits 路径 |
| pooled 稠密投影交叉点（已接入调优） | `tuning/pool_dense` | 对实际 decode/verify 行数比较逐行 GEMV、四行共享权重分块与原小 GEMM | 按设备、驱动、内核、M/N/K 与实际 GEMV tile 缓存冷/暖正反确认结果；新路径保持私有 F32 归约，只有原已使用小 GEMM 的行数允许保留该算法 |
| resident 投影搜索预算（已推导） | `tuning/projection_budget.rs` | 按实际录图保留的 FP4/FP8 形状、候选数和独立确认轮次估算 | 同一模型跨层和 serving state 去重；耗尽时列出尚未完成的形状，保留 native 路径 |
| 每槽半容量 KV 保底（已删除） | `loading/budget.rs`、`executor/admission_pool.rs` | 每请求按实际 prompt + 最大输出长度向上取整到页；保留一个完整上下文的隔离能力 | 冷启动纯报价，reservation 按实际长度选择计划；只在无活动状态、所有 lease 归还后调整实体池 |

CUDA 默认并发按两个界限推导。N_mem 通过现有完整 serving 报价求最大可资助宽度，
包括 target/draft pool、worker、slot home、workspace、KV 与 driver headroom。
每槽固定半容量保底已删除；KV 需求由实际 prompt + 最大输出长度逐页取整。
冷启动的并发是完整报价可资助的上限，实体 worker 尚不存在时不报告其宽度。
首次 reservation 及后续空闲边界再按实际请求需求选择池；活动状态、lease 或 flight
存在时不重建。运行时的初始并发仍是上限，实体槽或物理页不足的请求等待资源归还。
N_knee 用实际线性图的
权重/scale 字节、参数数、计算格式、设备 SM/时钟/带宽及每序列验证行数估算。
混合精度按各算子的峰值计算时间求和，再与权重读取时间相除；同精度时退化为
(峰值算力/带宽)×(权重字节/参数量)/(2×每序列行数)。只用 dense 吞吐与实际
F32 累加格式，不能混入稀疏或 FP16 累加的宣传峰值。

架构常数依据 [NVIDIA RTX Blackwell 白皮书表 3](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
及 [Hopper Tensor Core 说明](https://developer.nvidia.com/blog/?p=45555)；
SM 数、时钟与带宽来自驱动，不按显卡型号匹配。未知架构或专家路由图没有完整
计算估计时，明确记录该限制，仅使用显存及协议边界。该估计不是饱和点的实测认证。
显式 CLI 并发上限绕过自动 roofline 选择，仍受实际显存与协议容量限制。

runtime 默认先持有协议容量；收到 backend 报告的 concurrency 后，在分配主机缓冲与
scheduler workspace 前收紧。未提供建议的兼容 backend 只受原有资源准入及给定上限
约束，不把协议上限宣称为已调优并发。Metal 尚未提供这套 roofline 建议。
槽位上传环按实际池宽分配，池及验证的结构上限与完成协议保持一致。

加载期只登记实际 pooled decode/verify 形状，不为此测单行或 prompt 宽度。每个新候选
须在冷、暖缓存下都超过原路径至少 2%，并通过独立反向顺序确认；四行分块还会对
新的实际权重执行完整逐位校验。结果随 workspace 绑定，图重放期间不搜索、不读回。
未测量、预算耗尽或关闭调优且没有可用缓存时，才保留原有 16 行分界的兼容选择。
此处仍限于已验收的小输出 BF16 权重族，不把算子收益外推为服务收益。


GDN 的 wave 目标还要服从分块的资源边界：一个 verify CTA 要处理三个候选 lane，
过宽的 value 分块会让寄存器和共享内存压力上升，不能只以 CTA 数更少作为优化依据。
实现从驱动读取 warp 宽度限制连续 value 列，再用实际头数与槽数寻找最近的合法网格；
设备数值、状态与性能检查分别验证所选分块，不把网格推导本身当作性能证明。

## 参考：一次完整调优的记录

PR #47 是一个完整的例子：把逐算子 profile、逐提交二分、和 vLLM 逐形状对比、质量门禁与固定输出量计时组合起来，把 RTX 5090 上零参数默认 C16 的耗时从 17.4 s 降到约 13.3 s。各步骤的证据路径见 PR 描述。

## 已测 split-K 实验的边界

[RTX 5090 的 NVFP4 split-K 筛查](../archive/research/artifacts/nvfp4-split-k-screen-2026-10-25.json)
覆盖 48 行的 gate/down 投影、三种 tile 和 2/4/8 分片，使用三个独立进程。
数值尾部回归通过，但合成权重微基准中 gate 最佳候选仍慢约 15.9%，down 最佳候选快约 11.3%。
因此未启用通用 split-K 默认策略。down 专项仍需实际权重选择、临时内存报价、服务性能和任务质量验收；
不能把这组算子收益当作服务收益或正式性能资格。实验源码与全部样本身份见记录。

小输出投影的窄布局仍受已验证的 SM120、64 个 prompt lane 和既有 tile 宽度约束；
这些是设备与 kernel 布局边界，不是模型维度。图结构识别只在录图时执行，
不增加逐 token 查图；扩大适用形状后仍需独立数值与性能验收。

融合 SwiGLU 与独立 FP4 GEMM 分开登记搜索工作，跳过已被融合吸收的独立 gate/up，
不把同一形状的每一层都重新计时。不同权重对、输入 scale 和 down scale 仍分别进行完整
量化输出校验；缓存绑定硬件、驱动、编译器及量化器/融合 kernel 身份。候选失败记录原因，
不通过确认的配置不得覆盖基线。现有融合资格保持不变；当前 27B 的两个 prompt 形状
在 RTX 5090 上仍选中原 tile，其他形状的算子改善不能外推成该模型的服务收益。


按请求长度调整 KV 时，quote 阶段不能捕获图；实际 reservation 在既有 owner 和有界
异步资源协议中执行。实体池至少能保留一个完整上下文；长请求仍沿用独立报价、
空闲池回收和后续恢复路径。KV inspection 的空闲页来自真实物理页，不再按
“槽数 × 逻辑容量”虚增；请求已保留到其实际最大长度的页，不在 decode 时重复收费。

首次捕获可能从加载阶段移到首个请求。性能证据必须同时保留 ready 时间、首次
warmup 的完整耗时以及两者之和，不能只把 ready 提前当成启动加速。
普通迭代门禁按每个 workload 的 warmup 后读回实际计划，并要求测量期间保持一致；
默认仍拒绝资源计划变化。资源策略本身的修改须通过
[迭代工具](../../tools/bench/iteration-gate.py)的 `--resource-change-reason`
在证据中明确声明，执行精度、上下文、报价和实际分配的预算检查仍保留。
