# Grouped-int kernel family and kernel observability

状态：**格式边界、packer、kernel 家族、多行 rows kernel 已实现**（第 1–4、7 步）；Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K 走同一条路径。kernel manifest 与可选设备计时（第 5–6 步）尚未实现。性能问题、测量与后续方案见 [GGUF grouped-int 性能](../performance/gguf-grouped-int.md)。

## 已实现的单一来源

| 事实 | 唯一位置 |
|---|---|
| 格式标签与几何（code 位宽、group 大小、zero point / bias） | `infer_ir::GroupedIntFormat` |
| GGML block 的位布局 | `infer_format::gguf::grouped`（`decode_span`），参考 `dequantize` 也走它 |
| GGUF 类型 → 格式 | `GgufType::grouped_int` |
| 设备布局（packing、tile、term 下标） | `infer_cuda_kernels::contract::{CodePacking, GroupedIntLayout}` |
| host 打包 | CUDA `loading::GroupedIntTiled`（按 packing 分支，不按格式） |
| kernel | `nibble::{gemv, gemv_zero, gemm, gemm_zero}`、`high::{…}`、`byte::{…}`；group、zero point、高位数与行块高度是 const generic |

`ExecutableEncoding::GroupedInt(format)` / `WeightEncoding::GroupedInt(format)` / `WeightRef::GroupedInt(GroupedIntWeight)` 各只有一个变体，`Projection::grouped_int` 是唯一入口。新增一个与现有 packing 相容的格式只需：`GroupedIntFormat` 一个变体 + 几何、`decode_span` 一个分支、CUDA provider 声明一行。

### 设备存储代价

所有格式共用 f32 factor（有 min 的格式另加 f32 bias）。4-bit 码两个一字节（`Nibble`）；5/6-bit 码的低 4 位
同样两个一字节，其上 1/2 位放在整矩阵 nibble 之后的高位平面（`NibbleHigh`，每 span `[32·高位数, 4]` 个
`u32`，位序见 `CodePacking::NibbleHigh`）；8-bit 码按字节存放：

| 格式 | GGUF bits/值 | 设备 bits/值 |
|---|---|---|
| Q4_0 | 4.5 | 5 |
| Q4_1 | 5 | 6 |
| Q4_K | 4.5 | 6 |
| Q5_0 | 5.5 | 6（字节存放时 9） |
| Q5_1 | 6 | 7（10） |
| Q5_K | 5.5 | 7（10） |
| Q6_K | 6.56 | 8（10） |
| Q8_0 | 8.5 | 9 |

剩余差距来自 f32 term（每值 1–2 bits）；改 f16 需要先测 K-quant `d·sc` 的舍入，见性能文档 P4。

Q2_K/Q3_K/IQ 系列尚未接入，加载时显式 `Unsupported`。

## 既有设计基线

这份方案沿用现有分层，不新增平行的 kernel 选择/观测机制：

- [Backend 架构](../../architecture/backends.md)：`ModelIr`/`DataflowGraph` 是共有语义，`KernelRegistration`/`ExecutionProgram` 绑定 backend；公共 trace/probe/stats 留在 IR。文档也明确：GPU profile 是 command-buffer 时间，OpTrace 是 CPU encoding，二者都不是逐 kernel GPU counter。
- [代码布局](../../architecture/layout.md)：host 调度、平台契约、backend 实现分层；tile layout 和 kernel 属于 backend。
- [性能基线约定](../../plans/performance/baseline.md)：先测瓶颈、区分算子/模型/服务，不把微基准当成服务证明。
- 现有 `infer-kernel-api` 的 `KernelRegistry` 已用 `KernelRegistration.estimated_ns` 做静态选择；`tile::runtime::Timing`/`record_event` 已提供成对微基准的设备计时；`phase`/`tracing` 已有加载阶段计时。

真正的缺口是：captured graph 没有把每个 launch 的静态描述和可选设备时间归因到 projection/kernel，而不是缺少一个全新的计时框架。因此下面的 manifest/timing 是对 `KernelRegistration` 和 `profile` 的扩展。

## 问题

现在的 Q4_K / Q6_K 路径在三个层面重复：

1. host 布局：`Q4KTiled` / `Q6KTiled` 各自被 loader 和 `ProjectionWeight` 分支处理；
2. 调用点：`grouped_int_row` / `grouped_int6_row` 两个 trait 方法，`row` 和 `rows` 里各自 match；
3. kernel：Q4/Q6、N16/N64、单层/两层 int8 各写一份完整 kernel。

其中 2、3 的一部分是历史交付顺序造成的，不是 GPU 本身要求。Q4_K/Q6_K 的 nibble 解包和 scale/min 语义不同，内层循环必然特化；但布局、调度、epilogue 和观测不应该重复。

## 目标

- 新增一个 grouped-int 格式时：
  - host 只增加一个 `GroupedIntKind` 变体和布局构造；
  - 契约只增加一个 kernel 实现/特化；
  - loader、binding、rows 路径、inspect、profile 不改分支。
- 所有 serving 的 prefill 必须走 multi-row kernel；逐行 GEMV 只允许 M=1 decode。
- 每个 kernel launch 都有可查询的静态描述；可选地有设备时间。
- 默认路径不增加 per-launch 分配、同步或事件开销。

## 设计

### 1. 格式描述

`GroupedIntFormat`（`infer-ir`）是唯一格式标签，提供与布局无关的几何：`code_bits`、`group_values`、`span_groups`、`offset()`（`Zero(z)` 或 `Bias`）与参考取值 `value(code, factor, bias)`。每个格式都解码为每值一个无符号 code 加每组一个 factor（和 bias）。

`decode_span`（`infer-format`）把 256 值的存储 span 解成 `GroupedSpan`；这是唯一知道各格式位布局的代码。参考 `dequantize` 与 CUDA packer 都调用它。

`GroupedIntTiled` 是 **CUDA backend 内部**的 host 布局：

```rust
struct GroupedIntTiled { format, rows, columns, codes, factors, biases: Option<_> }
```

它按 `CodePacking`（nibble / byte）写 code，按统一下标 `((tile·spans + span)·groups + g)·16 + row` 写 term，与格式名无关。Metal 不实现它。

### 0. 平台侧与 backend 的边界

| 层 | 放在哪里 | 内容 |
|---|---|---|
| 容器 | `infer-format` | GGUF metadata、block bytes、参考 decode |
| 模型 | `infer-models` | provider、config、slot 映射、`GroupedIntKind` 几何、raw block |
| 平台契约 | `infer-spi` / `infer-kernel-api` | kernel registration、capability、plan 语义、manifest/timing schema |
| CUDA | `cuda` / `cuda-kernels` | tile layout、Tiled 类型、kernel、CUDA event |
| Metal | `metal` | 自己的 layout/kernel/event，只实现平台契约 |

`GgufPackage` 只返回 raw block；CUDA 自己 pack。Metal 接入时复用 `decode_span` 与 `GroupedIntFormat`，只写自己的 packer 与 kernel。

### 2. 投影契约

`Projection` 只保留一个 grouped-int 入口：

```rust
fn grouped_int(&self, recording, plan: &GroupedIntPlan, out, x, weight: GroupedIntWeight<'_>) -> Result<()>;

struct GroupedIntWeight<'a> {
    format: GroupedIntFormat,
    codes: Buffer<'a, u8>,
    factors: Buffer<'a, f32>,
    biases: Option<Buffer<'a, f32>>, // 恰在 format.has_biases() 时存在
}
```

实现用 `GroupedIntLayout` 校验长度，再按 `(CodePacking, biases)` 选 4 个 entry 之一；group 大小与 zero point 作为 generic 传入。调用方不按格式分支。MMQ 接入时 `GroupedIntPlan` 再加 `kernel: Gemv | Mmq`。

### 3. Kernel 家族

不追求“一个 kernel 处理所有格式”。按家族生成特化：

```text
grouped_int_gemv<BITS, FORMAT>(...)       // M=1
grouped_int_mmq<BITS, TILE_N, TWO_TERM>(...) // M>=16
```

共享部分：

- code/nibble 解包的结构；
- factor/bias epilogue；
- activation 主项/残差量化；
- tile 输出、边界和 tail 处理。

特化部分：

- 位宽和 code 布局；
- scale/min 语义；
- MMA 指令和 tile 形状。

第一版可以用 const generic + 小型宏生成 Q4/Q6 两个模块，保留独立的 entry；目标是消除复制粘贴，而不是牺牲指令选择。

### 4. rows 路径

```text
M == 1          -> gemv / gemv_zero
M >= 2          -> gemm / gemm_zero，[32, 16] 输出块，末块谓词化
```

`Mode::Prompt`、`Mode::Rows`、`Mode::Slots` 都走 `RowsKernel::GroupedInt`，一次 launch 覆盖全部行块；不再有逐行循环。2–12 行的谓词化开销见性能方案 P6。

### 5. Kernel 观测

现有基础：

- `tile::runtime::Timing`：成对微基准，CUDA event；
- `phase` / `tracing`：加载阶段时间；
- `infer profile`：CPU/控制时间线。

缺口：captured graph 没有 per-launch GPU 时间，也没有统一的 kernel manifest。设计中分两层：

#### 静态 manifest（默认开启，零设备开销；复用 KernelRegistration 描述）

schema 放在 `infer-spi` / `infer-kernel-api`，复用 `KernelRegistration.estimated_ns` 作为静态估算；CUDA 和 Metal 都填同一份 manifest，capture 时记录每个 launch 的 `KernelDescriptor`。

```rust
struct KernelDescriptor {
    family: KernelFamily,       // Projection, Attention, Mlp, ...
    encoding: WeightEncoding,
    kernel: &'static str,       // "grouped_int_mmq_n64"
    rows: usize,
    outputs: usize,
    columns: usize,
    tile: [usize; 2],
    weight_bytes: u64,
    quantization: Option<QuantMode>,
}
```

写入 `DeviceProgram` 的只读 manifest；`inspect-package` 和 `profile` 可导出。它不需要事件、不需要同步，默认不改变执行路径。

#### 可选设备计时

设备计时由 backend 实现（CUDA event / Metal counter），但报告结构和聚合在平台侧。启动参数 `--kernel-timing`（或 feature）时，在每个 launch 前后插入 backend event：

```text
event(start) -> kernel -> event(end)
```

replay 后读取，聚合为：

- count；
- min/median/max；
- 总时间；
- 有效 GB/s（用 descriptor 的 bytes）。

约束：

- 默认关闭；
- 事件数量有上限，超过后降级为 manifest + 采样；
- graph replay 不重新分配事件；
- 计时结果写入 JSON/Chrome trace，不打印到请求路径；
- 计时开销单独测量，不能把开启计时的数字当作性能基线。

### 6. 验收

- 新增一个 grouped-int 格式只改 2 个 host 文件和 1 个 kernel 文件；
- Q4_K/Q6_K 的 M=1/M=16 精度与当前一致；
- prefill 不再出现逐行 launch；
- manifest 覆盖每个 captured launch；
- 设备计时默认零开销，开启后报告 overhead；
- 在 5090 上比较 Q4_K GEMV、MMQ N64、NVFP4 和 llama.cpp 的分项时间。

## 实施顺序

1. `GroupedIntTiled` + loader 收口（已完成）；
2. `Projection::grouped_int` 单入口、rows 单路径（已完成；rows 仍逐行调用 GEMV）；
3. kernel 家族按 packing × offset 模板化（已完成）；
4. 多行 rows kernel（BF16 MMA，`[32, 16]` 输出块）接入 prefill 与 pooled/gathered rows，删除逐行循环（已完成；int8 MMQ 见性能方案 P5）；
5. `KernelDescriptor` manifest；
6. 可选 CUDA event 计时；
7. 接入 Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q5_K（已完成，llama.cpp 量化的 Qwen3-VL-2B 端到端对照见 PR）。

## 非目标

- 一个 kernel 运行时支持所有格式；
- 默认开启 per-kernel 事件；
- 用通用抽象替代格式特化。
