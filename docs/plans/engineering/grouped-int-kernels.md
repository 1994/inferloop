# Grouped-int kernel family and kernel observability

状态：**设计 + 第一阶段实现**。目标是在接 Q8_0 / Q5_K / IQ 系列之前，把 grouped-int 的 host 边界、kernel 选择和观测收成一套；新增格式只增加格式描述和特化 kernel，不再改一圈调用点。

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

`GroupedIntKind` 是唯一格式标签，提供与布局无关的几何：

```rust
block_values()       // 256
code_shape(rows, cols)
has_biases()
```

`GroupedIntTiled` 是 **CUDA backend 内部**的 host 布局，不放在 model 或 kernel-api：

```rust
enum GroupedIntTiled { Q4(Q4KTiled), Q6(Q6KTiled) }
```

它提供 `rows/columns/codes/factors/biases/code_shape`。CUDA loader 只调用 `from_blocks(kind, ...)` 和这些访问器，不再知道 Q4/Q6 的内部字段。

Metal 不实现这个 enum。平台侧只拥有：

- `GroupedIntKind`；
- `code_shape` / `has_biases` 等与 backend 无关的几何；
- raw block 访问；
- plan / capability / manifest。

新增格式的 host 工作：

```text
GroupedIntKind::Q8_0
CUDA: GroupedIntTiled::Q8(...)
CUDA: from_blocks 增加一个分支
```

### 0. 平台侧与 backend 的边界

| 层 | 放在哪里 | 内容 |
|---|---|---|
| 容器 | `infer-format` | GGUF metadata、block bytes、参考 decode |
| 模型 | `infer-models` | provider、config、slot 映射、`GroupedIntKind` 几何、raw block |
| 平台契约 | `infer-spi` / `infer-kernel-api` | kernel registration、capability、plan 语义、manifest/timing schema |
| CUDA | `cuda` / `cuda-kernels` | tile layout、Tiled 类型、kernel、CUDA event |
| Metal | `metal` | 自己的 layout/kernel/event，只实现平台契约 |

当前过渡状态：`GgufPackage::q4k/q6k` 仍直接返回 CUDA tiled 类型；接 Metal 前必须改成返回 raw `GroupedIntBlocks`（kind、shape、bytes），由 backend 自己 pack。

### 2. 投影契约

`Projection` 只保留一个 grouped-int 入口：

```rust
fn grouped_int(
    &self,
    recording: &Recording<'_>,
    plan: &GroupedIntPlan,
    out: Buffer<'_, f32>,
    x: Buffer<'_, f32>,
    weight: WeightRef<'_>,
) -> Result<()>;
```

`GroupedIntPlan` 携带：

```rust
struct GroupedIntPlan {
    kernel: GroupedIntKernel, // Gemv | Mmq { tile_outputs }
    rows: usize,
    columns: usize,
}
```

实现内部按 `WeightRef` 分派到具体特化；调用方不再按 Q4/Q6 分支。

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
M == 1          -> grouped_int_gemv
M >= 16         -> grouped_int_mmq
1 < M < 16      -> pad 到 16 或回退到 GEMV（显式策略，不隐式逐行）
```

`Mode::Prompt` 不再产生 `PromptGrouped` 的逐行循环；没有 multi-row kernel 的格式在 load 阶段返回 `Unsupported`，而不是在 serving 中静默降级。

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

1. `GroupedIntTiled` + loader 收口（本 PR 已开始）；
2. `Projection::grouped_int` 单入口、rows 单路径；
3. Q4/Q6 kernel 家族模板化；
4. multi-row MMQ 接入 prefill，删除 `PromptGrouped` fallback；
5. `KernelDescriptor` manifest；
6. 可选 CUDA event 计时；
7. 再接入 Q8_0/Q5_K 等新格式。

## 非目标

- 一个 kernel 运行时支持所有格式；
- 默认开启 per-kernel 事件；
- 用通用抽象替代格式特化。
