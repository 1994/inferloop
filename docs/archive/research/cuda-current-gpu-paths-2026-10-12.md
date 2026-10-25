# 当前 GPU 执行路径诊断（2026-10-12）

本轮直接从 vLLM 官方 main 拉取源码，并在相同 27B checkpoint 上采集预热后的 native 与 vLLM GPU 路径。长提示的主要方向仍是减少重复的 prompt 矩阵计算、改进逐 token GDN 更新；小输出投影分块实验未通过逐位准入，没有进入主线。性能接近 vLLM 的目标仍未达到。

## 源码与运行版本

45 的 `/home/r/code/vllm` 从官方 `https://github.com/vllm-project/vllm.git` 快进到 `cc9b0f097d49abfa6fc94d1d773dca1cc2b9619c`，工作区干净。该次提交只改变 MiniMax RMS norm 的 CUDA IPC handle 处理，Qwen/GDN 路径和 native extension 源码均未改变。

本轮 vLLM 追踪保留上一轮已经验证的官方 `3fe917abb13160d60bea5330399f5c09ca282348` wheel，版本 `0.31.1rc1.dev289+g3fe917abb`。准备时查询 cc9 提交的官方索引与预计 wheel 地址均返回 404；不将 3fe 运行结果标为 cc9 构建。对 wheel 与最新 checkout 的 2,639 个共有 Python 文件逐个比对，仅 `_version.py` 与 MiniMax IPC 文件不同。二进制与依赖身份继续引用[服务对照报告](vllm-main-serving-comparison-2026-10-12.md)。

Native 使用已合入 serial 快照融合的 `b02a4fe9` CLI，SHA256 为 `052e351a8b1a97d5896f21779404e9ea08b64505743ad899f36280e96f44b819`。冻结硬件 tuning 表不变，显式开启 `INFER_CUDA_DEVICE_GREEDY=1`。本轮源码主线基点是 `38232b85`，runtime 算法没有变更。

## 采集条件与边界

每个引擎在新服务进程中执行 short、511-token long、batch4 三个冷场景；每场景一次排除的 warmup、三次测量，只有第一个测量重复开启追踪。每请求输出 64 token，MTP 深度 2。完整九个测量 trial 和所有预热 trial 均保留。测试保留用户/桌面进程，只清理由本轮启动的服务。

Native 使用 Nsight Systems 2026.5.1 的 CUDA graph node 追踪，分别保存三个 report；导出成功。三个 report 都包含 `Not all CUDA events might have been collected.`，因此下述 native 数字只描述已观察活动，不用来计算 GPU 总占比、空闲比例或完整 kernel 数量。

Nsight 注入的 vLLM 服务未能启动，在等待 pipe 时停滞超过正常启动时间；记录已保留，只停止本轮自有服务。随后采用 vLLM 内置 Torch profiler，服务 38.14 秒正常启动，六次 start/stop HTTP 调用全部返回 200，三个 CPU/CUDA trace 完整写出。开启 shapes，关闭 stack/memory，忽略 frontend，在 warmup 后启动记录。

两种 profiler 的开销与范围不同，native activation 为 F32、vLLM 为 BF16，生成 token 路径不同。这些轨迹用于源码归因，不是新的同条件延迟基线。前一轮未插桩的[五组服务对照](vllm-main-serving-comparison-2026-10-12.md)及其后台/质量/热复用限制继续有效。

## 长提示的实际工作

| 已观察路径 | Native | vLLM |
|---|---|---|
| 511-token target prompt | 窄 prompt 图分段执行，GDN 记录到 512 行（含 padding） | `execute_context_1(511)_generation_0(0)` |
| GDN prompt kernel | 首 token 到达前 24,576 个 `delta_entry`，已记录执行时间 110.35 ms | 48 层各执行 CP TPrecompute、MNPrecompute、FixupHmma、Prefill，共 192 个 kernel |
| vLLM 四类 GDN kernel 时间 | 不适用 | 四组总计 6.81 ms；仅该次插桩诊断 |
| 生成阶段 | native pooled 验证保留固定 pool 的 row/state 操作 | long 记录到 23 次三行 MTP context；short 为 29 次 |

数量与时间不能直接换算成端到端加速倍数：包括数值契约、不同接受轨迹、padding、采集完整性和 profiler 差异。但代码与活动都支持同一个优化方向：长 prompt 的 GDN 应转向经过数值/模型验收的分块矩阵算法，projection 应减少重复读取权重。仅扩大 GEMM tile 或调整编译优化等级无法消除逐 token 更新的成本。

当前对照也再次看到 native decode 的 `dense_entry` 以 `[1,2,1]` grid 执行小投影；short 记录 2,688 次、46.90 ms，约 17.45 μs/次。结合 checkpoint 与 capture 调用，这对应两路独立的 `[48,5120]` BF16 B/A 投影。vLLM 合并 B/A 后走 BF16 cuBLAS/WMMA。应分别评估减少调用与改善小 N 并行度，不能把不同 batch/精度的原始 kernel 时间直接当成等价收益。

Batch4 中 vLLM 记录到三行、六行、九行、十二行 MTP contexts，其中十二行 25 次；四个 HTTP 请求确实形成了跨序列验证批次。这比仅从 HTTP 并发数推断底层 GDN/GEMM 批宽更有依据。

## 小投影分块实验

隔离源码新增 8/16 列候选，保留原 K-loop 顺序和高 BF16/低 residual 两项，原 32 列 kernel 不变。读取全部 96 个真实 checkpoint B/A 权重，预设逐位验证 1、3、4、12、32 行，再进行冷热缓存 AB/BA 测量。

Release 编译成功，实验 binary SHA256 `3c335f2b979ff3061148fb6ef164e5112cdcd48770e8022023d7555400be3ed4`。但首个权重、1 行输入、8 列候选的全部 48 个输出位模式不同，原逐位 gate 返回 101，停止在性能测量之前。16 列候选和后续权重未执行，不能宣称它们失败或成功。

没有采集误差幅度和任务质量，位模式不同本身不证明数值结果无效；它只否定本候选满足逐位保持的准入假设。没有性能数据，不宣称小 N 分块可提速。候选源码与失败日志保留，隔离 build source 恢复为主线版本，生产 kernel、默认路径和 tuning 表未修改。

## 后续实施约束

1. 先定位 GDN chunk 的首个不同计算阶段，保持原数值门槛。既有 WY/Neumann 和编译等级扫描的失败记录仍有效，不能重新当作已验收方案。
2. Prompt 宽度需根据 workload 与硬件选择，并按实际保留的 graph/arena 计预算。历史显式 256 试验改善 long 的部分工作，但 short TTFT 明显退化，见[16.14](cuda-performance-experiments.md#1614-图宽的无插桩反例与真实模型数值门禁2026-10-11)，不全局固定到最大宽度。
3. 小投影的候选须先获得适用数值契约，再测真实权重和完整模型。确认的硬件选择接入现有 tuning，并保留保守 fallback；不按 5090 型号写死参数。
4. 推测解码的接受轨迹与执行批宽必须同时记录；降低一个验证循环的成本不等于相同输出吞吐提高。

## 原始证据

45 上共同根目录：

`/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles/`

| 目录/文件 | 内容 |
|---|---|
| `latest-official-vllm-main-2026-10-12.json` | 官方 main 拉取版本、干净状态、改动路径、wheel 查询结果 |
| `current-main-critical-path-v1/`、`-v2/` | 被 workload/CLI 门槛拒绝的早期采集尝试，未运行有效 GPU 测量 |
| `current-main-critical-path-v3/` | 三个 native report、SQLite、导出日志、host phase、失败 vLLM Nsight startup |
| `current-main-vllm-torch-critical-path-v1/` | 内置 profiler 配置、启停记录、三个 trace、九个 trial、源码逐文件核对和 activity analysis |
| `small-dense-tile-v1/` | 实际权重 manifest、候选/原源码、编译身份、逐位失败与恢复记录 |

`current-main-vllm-torch-critical-path-v1/evidence-manifest.json` 固定主要数据的 SHA256，其中 `activity-analysis.json` 为 `abc81a195a6da564112282ebd1b88ad0be2ca8a3e152557ea535788f27c5d34e`，`latest-checkout-source-proof.json` 为 `1ac31aa4c3d9847422cda5652d096a8fef9abf340fdb3f62a0d173814e6caaba`，小投影 `outcome.json` 为 `33b15a577e9b866d53f71fc33f9603132a2c24b6a5868f8e3d40e284ec39bdc0`。
