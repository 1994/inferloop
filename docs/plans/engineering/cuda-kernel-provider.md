# CUDA AOT kernel provider 与 12.9 设备接入方案

状态：部分实施。kernel 已搬入 `infer-cuda-kernels`，`pool_gemv_bf16` 已在推理路径走契约，AOT 版本在 sm_120 上通过捕获与回放；host 迁移和 L20 实机未开始。首个目标设备为 L20（sm_89 / CUDA 12.9 / 驱动 550.144）；现有 13.x 路径不回退。
host 收口的整体顺序见 [CUDA 架构收口方案](cuda-architecture.md)。整体排期见[路线图](../README.md)，后端能力边界见 [Backend](../../architecture/backends.md)，CUDA 现有实现见
[CUDA 说明](../../../crates/backend/cuda/README.md)。

本文回答一个问题：**没有 Tile IR 的 CUDA 环境（12.9 / sm_89）如何跑同一套 engine，同时不改 host 的
KV、图、调优和准入逻辑。** 结论是新增一个编译期可选的 AOT kernel provider，host 只依赖
`Buffer` / `Kernels` 契约；设备接入退化为“加 profile、加制品、过验收”。

## 0. 启动前提

本方案的首要目的是**在工程结构上规范不同 CUDA 版本的差异**；L20 是第一个实例。项目判断 L20 不走 cuTile，
L20 需要的 kernel 用 `.cu` 重新实现，因此 AOT 是主线，不以 `cuda-compat` 前向兼容验证为前提。

v1 范围以 dense BF16/FP8 为起点，先实现目标模型实际用到的算子族，不按 146 个 cuTile entry 逐个翻译。

12.9 的驱动前提是 CUDA 12.x minor version compatibility（最低 525），550.144 满足。但 **550 的 PTX JIT
只到 12.4（PTX 8.5）**，因此 12.9 制品必须是 sm_89 的 SASS，不能只给 PTX；这一条是硬约束，见第 8 节。

## 1. 目标

一句话：**host 只认 `Buffer` / `Kernels` 契约，kernel 的来源（cuTile 或 AOT）由 feature 决定是否编入，构造时选定一次；新增设备 =
新增 profile + 制品 + 验收。**

可验收结果：

- `--features cuda`（13.x）的行为、性能、数值与今天一致；
- `--features cuda-aot`（12.9）在 L20 上跑通 dense BF16/FP8 模型，数值过 reference-ops，CUDA Graph 可捕获；
- 两个构建不共享运行期分支，host 代码零 `cfg(feature)`；
- `cuda-aot` 的构建不需要 CUDA 工具链头文件，也不需要 `tileiras`。

## 2. 非目标

- 不做运行期 kernel 切换，不新增 `BackendKind`、CLI `--backend` 或第二套 `BackendProvider`。
- 不复制 `resident/`、`tuning/`、KV 或准入逻辑；这些只保留一份。
- v1 不覆盖 NVFP4（L20 无 FP4）、MoE、vision；未覆盖算子显式 `Unsupported`。
- FlashInfer 不是默认依赖，不进入 `cuda` 构建。
- 不改变 13.x 的制品、数值 contract 或调优身份。

## 3. 设计原则

1. **host 不出现 `cutile` / `cuda_core` / `libloading` 类型。** 设备句柄、buffer、stream、artifact 加载都在
   `cuda-kernels` 内。
2. **来源可叠加，选择只发生一次。** `tile` 与 `aot` 可同时编入，没有 cuTile 的设备只编 `aot`；每个算子族在构造期选定实现，调用路径上没有版本或架构分支。
3. **一个 provider、一套 KV/图/调优/准入。** kernel 只是实现，不决定 page 格式、workspace 或调度。
4. **制品与数值 contract 是一等公民。** 每个 kernel 有 manifest、hash、arch、toolchain 和 contract ID。
5. **`cuda-aot` 构建零工具链依赖。** 驱动与 cuBLASLt 全部 dlopen，制品由固定工具链离线产出，不入库。
6. **数据布局由 host 定义，kernel 适配 host。** 外部 kernel 库（FlashInfer）只能适配既有布局，不能反向
   决定 KV 格式；边界见 8.5 节。

### 3.1 维度模型

不要用 CUDA 版本做组织维度。版本是二维的（工具链 × 硬件），按版本建目录/feature 会把矩阵压成一维，
每加一个组合就多一条分支。用五个正交维度：

| 维度 | 取值 | 谁感知 |
|---|---|---|
| `Target` | `Arch`（sm_75…sm_121）+ 设备类别 | `contract/target.rs`、能力探测 |
| `Profile` | `tile` / `aot` | Cargo feature、实现目录 |
| `Family` | projection/attention/norm/rope/recurrent/moe/vision/state/sampling | trait |
| `Contract` | `canonical-f32` / `bf16-f32acc` / `fp8-kv-f32acc` / `vendor-*` | registration + manifest |
| `Artifact` | `(profile, arch, family, contract)` → 文件 + hash + 工具链 + 格式 | `tools/cuda/kernels.json` 的 `sidecar` + 构建 manifest |

一个 **cell** = `(profile, arch, family, contract)`。host 只看到 `(family, contract)`；矩阵在构造期解析成
`Kernels`，调用路径上没有版本/架构分支。

### 3.2 版本策略

- CUDA 小版本不是维度，是 profile 的**构建属性**：`tile` 需要 CUDA 13.2+ 与 `tileiras`，`aot` 需要支持目标 arch
  的 nvcc；具体版本 pin 在 manifest，不 pin 在代码。
- 升小版本 = 更新构建配方 pin、重产制品、重验；不改 `cfg`、不加模块。
- 升大版本或 Tile IR 变化 = 新增 profile 版本（如 `tile-v2`）或新 artifact variant；host 不变。
- 驱动下限是 artifact 属性（PTX 版本 / SASS ABI + arch），加载时校验并明确报错。
- 兼容性只有三类：**T**（运行期需要 toolkit/JIT）、**A**（运行期只需驱动）、**V**（需要厂商 `.so`）。
  用能力探测决定，不用版本号分支。

### 3.3 放置规则（CI 可检查）

| 内容 | 只能出现在 |
|---|---|
| `cutile::`、`#[cutile::module]` | `cuda-kernels/tile` |
| `libcuda` 符号、`cuLaunchKernel`、`cuModuleLoadData` | `cuda-kernels/aot` |
| 制品清单与工具链 pin | `tools/cuda/kernels.json`；sidecar 制品本身不入库 |
| 架构名（`sm_89`…） | `contract/target.rs`、`aot/<family>/<arch>.rs`、tuning 参数 |
| 工具链版本 pin | artifact manifest、`tools/cuda` registry |
| `cfg(feature)` | `cuda-kernels`；host 只保留既有的 `cuda_backend` 平台门控 |
| `BackendProvider`、图、slot、tuning | `cuda` host |

### 3.4 扩展 recipe

- **新硬件**：`Arch` variant → 能力探测 → 每族选“复用 / 新写 / Unsupported” → 制品 → baseline → CI lane。
  host 零改动。
- **新 CUDA 版本**：能力是否变化？不变 → 更新 pin、重产、重验；变 → 新 profile 版本。host 零改动。
- **新算子**：trait + `Plan` → 两个 profile 各自实现或 `None` → registry 注册 + contract → reference-ops →
  制品 → tuning。host 只有一个调用点。
- **优化**：要么只改 cell（kernel/参数/制品），要么只改 host（图/融合/调度），不要混在一个提交。

### 3.5 治理

- 由 manifest 生成 support matrix（`profile × arch × family × contract`）并入库；CI 检查每个
  `KernelRegistration` 都有对应 cell 和制品 hash。
- 禁止：host 出现 `cfg`/arch/version、未登记制品、跨 cell 复用 tuning。
- 新增 cell 必须同时带：manifest、contract、reference-ops 对照、实机证据。

### 3.6 cuTile 与 `.cu` 的关系

不按语言统一，按契约统一。host 只认算子族 trait，由谁实现是 cell 的事。

- **现有 cuTile 不为统一而重写：** 它们的数值和调优已经过验证，重写只带来回归风险。
- **没有 cuTile 的设备写 `.cu`：** 按算子族实现，以独立 CPU 参考和实机结果验收，不按 entry 逐个翻译。
- **按算子族收敛：** 同一算子族的 `.cu` 在 13.x 设备上数值通过、服务级性能不差时，才退役对应的 cuTile 实现。
  这样双份维护只是过渡，且每一步都有证据。
- **`tile` 与 `aot` 可叠加：** 13.x 构建今天已经同时使用 cuTile 与预编译 PTX，按算子族收敛也需要两者共存，
  所以它们描述 kernel 来源，不互斥。

`KernelRegistration` 按 `Operation` 回答编译器“这个操作能不能执行”，不是实现选择器：host 只注册当前设备有实现的
操作，同一操作选哪个实现仍由算子族构造和 tuning 决定，不新增第二套注册表。

## 4. 模块布局

目标不是"再加一个 backend"，而是让 **同一份 host 在两种工具链下编译，差异全部落在 kernel 实现和制品上**。
布局按"契约 / 实现 / host"三层组织，实现内部再按算子族镜像。

### 4.1 目标树

```text
crates/backend/cuda/                         # host：一个 provider，没有第二个 backend
  src/
    lib.rs
    device.rs  device/                       # Device 门面；不再直接 use cuda_core
    executor/                                # provider/owner/submission/state/pool/prefix
    loading/                                 # 权重、量化、workspace 预算
    resident/                                # 图、capture、slot、workspace；不含 #[cutile::module]
    tuning/                                  # 选择、测量、baseline
    vision/                                  # 视觉 host 编排
    strategy.rs  target.rs  registry.rs  constants.rs  ...
  kernels/                                   # 迁移期保留的 13.x 制品
  baselines/tuning.json

crates/backend/cuda-kernels/                 # 新增：设备运行时 + 算子契约 + 实现
  src/
    lib.rs
    contract/
      buffer.rs  layout.rs  device.rs  stream.rs  plan.rs  target.rs
      kernels/                               # 每个算子族一个 trait
        projection.rs  attention.rs  norm.rs  rope.rs  recurrent.rs
        moe.rs  vision.rs  state.rs  sampling.rs
    tile/                                    # feature "tile"
      mod.rs                                 # TileKernels::new -> contract::Kernels
      projection.rs  attention.rs  norm.rs  rope.rs  recurrent.rs
      moe.rs  vision.rs  state.rs  sampling.rs  support.rs
    aot/                                     # feature "aot"
      mod.rs                                 # AotKernels::new -> contract::Kernels
      driver.rs  module.rs  launch.rs  cublaslt.rs  artifacts.rs
      host/                                  # C ABI shim（.cpp/.cu）
      projection.rs
      attention/  mod.rs  sm89.rs  sm120.rs
      norm.rs  rope.rs  recurrent.rs  state.rs  sampling.rs
    flashinfer/                              # feature "flashinfer"
  tests/
```

`.cu` 源码放在 `aot/<family>` 旁边，AOT 制品是构建产物，不进目录树。`crates/backend/cuda/kernels/` 里已入库的
13.x 制品保持现状，它们有独立的 manifest 和校验；是否迁移单独决定，不属于本方案。

### 4.2 镜像规则

同一算子族在 host、`tile/`、`aot/` 三处使用**同一个名字**，读者可以并排 diff：

```text
host/resident/attention.rs        -> 计划、workspace、capture
cuda-kernels/tile/attention.rs    -> cuTile 实现
cuda-kernels/aot/attention/mod.rs -> AOT 实现（按 arch 选 sm89/sm120）
```

host 的文件名不要求与实现一一对应（例如 `attention_decode.rs`、`attention_prefill.rs` 都属于
`Attention` 族），但实现侧必须收敛到族级，避免"一个 host 文件一个 kernel 文件"的碎片化。

### 4.3 差异吸收矩阵

| 差异 | 吸收位置 | 不应出现 |
|---|---|---|
| 工具链 13.x / 12.9 | Cargo feature + `tile/` / `aot/` | host 的 `cfg(feature)` |
| 架构 sm89 / sm90 / sm120 | `Target` + `aot/<family>/<arch>.rs` + tuning 参数 | host 的 `match arch` |
| 算子覆盖 | `Option<Arc<dyn Trait>>` + `KernelRegistration` | 调用点静默 fallback |
| dtype | trait 方法（`dense_bf16` / `dense_fp8`） | host 泛型或 dtype 分支 |
| AOT 制品 | `target/kernels/aot/<arch>/` + manifest，随包 sidecar | host 里的 `include_bytes!` |
| 数值契约 | `Contract` 绑定到 registration | kernel 体内的特例 |
| 厂商库 | `aot/cublaslt.rs`（`.so.12` / `.so.13` 回退） | host 直接 dlopen |

### 4.4 选择只发生一次

```rust
// contract/kernels/mod.rs：一个 Kernels 类型，两种构造
pub struct Kernels {
    pub projection: Arc<dyn Projection>,
    pub attention: Arc<dyn Attention>,
    pub norm: Arc<dyn Norm>,
    pub rope: Arc<dyn Rope>,
    pub recurrent: Option<Arc<dyn Recurrent>>,
    pub moe: Option<Arc<dyn Moe>>,
    pub vision: Option<Arc<dyn Vision>>,
    pub state: Arc<dyn State>,
    pub sampling: Arc<dyn Sampling>,
}
```

```rust
// tile/mod.rs、aot/mod.rs
pub fn new(device: &Device, target: Target) -> Result<Kernels>;

// lib.rs：feature 决定用哪个构造器，host 只看 Kernels
#[cfg(feature = "tile")] pub use tile::new as kernels;
#[cfg(feature = "aot")]  pub use aot::new as kernels;
```

`aot` 内部允许一次 `match target.arch` 选择 `sm89` / `sm120` 子模块；这是**构造期**的一次选择，不在
kernel 调用路径上。host 不感知 arch，也不感知 feature。

### 4.5 现有文件怎么搬

下表是迁移的来源映射；落地时目标统一为 `tile/<family>/`，同一算子族下保留多个文件，host 路径同步更新。

| 现在 | 目标 | 搬法 |
|---|---|---|
| `src/kernels.rs`（4 entry，linear） | `tile/projection.rs` | 整块移动，launch 改 trait |
| `src/resident/kernels.rs`（12 entry，aux） | `tile/support.rs` + `tile/norm.rs` + `tile/rope.rs` | 按算子拆 |
| `src/attention/kernels.rs` | `tile/attention.rs` | 整块移动 |
| `src/mlp/{kernels,pdl,pdl_consumers}.rs` | `tile/mlp.rs` | 合并到族 |
| `src/vision/kernels.rs`、`src/device/vision_position.rs` | `tile/vision.rs` | 合并 |
| `src/resident/dflash2/kernels.rs` | `tile/speculation.rs` | 整块移动 |
| `src/device/vendor_fp8.rs`、`vendor_fp8/nvfp4/kernels.rs` | `tile/quantized.rs` | 整块移动 |
| `src/resident/attention_mma/host.rs`（FA4） | `tile/attention/fa4.rs` | 仅 13.x 编译 |
| `src/resident/{attention,attention_decode,attention_prefill}.rs` | host 保留；`#[cutile::module]` 搬 `tile/attention.rs` | 拆文件 |
| `src/resident/{nvfp4_gemm,fp8_gemm,prefill_gemm,linear_batch,small_gemm,pool_gemv}.rs` | host 保留；kernel 搬 `tile/projection.rs` | 拆文件 |
| `src/resident/{recurrent_prefill,slot_recurrent,recurrent,recurrent_checkpoint,conv_prefill}.rs` | host 保留；kernel 搬 `tile/recurrent.rs` | 拆文件 |
| `src/resident/{moe,slot_convolution,slot_rope,norm_*_kernel,fp8_quant_reduction,metadata,prompt_logits,greedy}.rs` | host 保留；kernel 搬对应族 | 拆文件 |

原则：**先搬已经独立的 `kernels.rs`（7 个），再拆混合文件**；每搬一族，13.x 数值逐位不变，作为一个可
回退的提交。

### 4.6 迁移阶段

| 阶段 | 13.x 制品 | AOT 制品 | host 依赖 | 12.9 构建 |
|---|---|---|---|---|
| 迁移期 | `cuda/kernels/` 入库 | 无 | `cuda`（cuTile） | 不可用 |
| 契约落地 | 不变 | 构建产物 | `cuda` + `cuda-kernels/tile` | 不可用 |
| AOT 落地 | 不变 | 构建产物，sidecar | `cuda` 或 `cuda-aot`（host 迁移后） | 可用 |

只拆一个 crate 而不是两个：契约与两个实现共享同一组类型，拆成 HAL + kernels 会多出第三个 manifest 和
跨三处的 feature 组合；一个 crate 内的 feature 提供同等隔离，只有一个边界要维护。

## 5. 契约

host 只认这一层。先定层，再迁移：下面的基础层已落地并有数据，其余是按模板的机械工作。

### 5.1 四层

| 层 | 类型 | host 怎么用 | 为什么零成本 |
|---|---|---|---|
| 内存 | `Allocation<T>`（持有）、`Buffer<'a, T>`（借用） | 分配，取 `buffer()` 传给 kernel | `Buffer` 是地址加长度，元素类型只在编译期存在；host 无 `unsafe` |
| 录制 | `Device::capture(\|recording\| ...)`、`Graph` | 在闭包里调用算子族 | 只在加载和建池时发生，这里允许 `dyn` 和分配 |
| 执行 | `Device::run(Launch)`、`Launch::then`、`Device::fence` | `device.run(graph.launch()).await` | 具体类型，编译期选定，`Launch<O>` 是 cuTile 操作的 newtype |
| 算子族 | `Kernels { projection, rope, ... }` | 填 Plan 和操作数 | kernel 的 JIT 特化参数不变，机器码相同 |

规则：

- **回放路径不经过 `dyn`。** `Kernels` 只在录制时用。
- **图持有内存。** 录制时算子族调 `recording.retain(buffer)`，`Graph` 持有这些 owner，回放不碰它们。host 丢掉句柄后图仍可回放。
- **`Allocation` 在 tile 实现下持有 cuTile `Tensor`，** 所以跨 stream 独占和资源跟踪原样保留。
- **形状不进 `Buffer`。** Plan 给形状，实现校验长度。
- **host 不写 `DeviceOp`。** 自定义 PTX 和 `.so`（MMA、oxide、cuBLASLt）随算子族搬进 `infer-cuda-kernels`。
- **无效组合构造不出来。** Plan 用枚举，每个变体只带它需要的字段。

**实测（RTX 5090，release，同一 `pool_gemv` 图交替回放，每块 2000 次，11 块中位数）：** 裸 cuTile 5.20 µs，`Device::run` 5.20 µs，比值 1.0002。见 `tests/unit/resident_runtime_replay.rs`，超过 1.03 用例失败。

### 5.2 算子族

149 个 cuTile 入口按下表归入算子族。每个 trait 一个方法 `record(&self, recording, &Plan, Operands)`，模板是 `Rope`：`RopeMode` 一个变体对应一个入口，操作数一个结构体。

| 算子族 | 入口来源（`tile/`） | 入口数 |
|---|---|---|
| `Projection` | `projection/*`（含 FP4、FP8、prompt logits、厂商库） | 43 |
| `Attention` | `attention/*`，MMA、oxide 的 PTX 和 `.so` | 21 |
| `Recurrent` | `recurrent/*`，oxide GDN | 31 |
| `Norm` | `norm/*`、`state/aux` 的 norm 和 fusion norm、`mlp::norm` | 7 |
| `Mlp` | `mlp/*`（激活、残差） | 6 |
| `Rope` | `rope/*`、`state/aux` 的 rope | 5 |
| `Moe` | `moe/*` | 5 |
| `State` | `state/aux` 的 zero、unary、binary、split，`state/metadata` | 6 |
| `Sampling` | `sampling/*` | 2 |
| `Speculation` | `speculation/dflash2` | 14 |
| `Vision` | `vision/*` | 9 |

`attention/*` 的 `KvView` 是 host 的只读视图（页表、scale、长度），kernel 不得改布局。

### 5.3 迁移模板

每一步都能单独合并，都要过棘轮和配对验收：

1. **图换类型（已完成）：** host 产品代码不再出现 `CudaGraph`，棘轮归零。`Graph::scope` 和 `Launch` 的 `DeviceOp` 实现是过渡接口，让未迁移的录制和 `.sync_on`、`.then`、`run_op` 不用改写。
2. **逐算子族迁录制：** 在 `contract` 加 Plan 和 trait，把 kernel 调用搬进 `tile/<family>/contract.rs`，host 留一个调用点。
3. **内存换类型：** `Tensor<T>` → `Allocation<T>`，`partition`、`generics` 只留在 `tile`。
4. **删过渡接口：** host 不再出现 cuTile 类型，棘轮归零。

每批验收：27B、0.6B、MTP、DFlash2 输出与 main 逐位一致；调优缓存全部命中；配对吞吐不回退，每步设备时间不回退。

## 6. Host 改造

| 文件/模块 | 改动 |
|---|---|
| `src/device.rs` | `CudaDevice` 持有 `cuda_kernels::Device`；删除 `cuda_core::sys` 直接引用；profile 查询走 `Device` |
| `src/kernels.rs`、`src/resident/**` | `Tensor<T>` → `Buffer<T>`；launch → `kernels.<family>.<op>(plan, ...)`；删除 partitioned 入参 |
| `src/loading/**` | 上传/转换走 `Device::upload`；量化 scale 作为 `Buffer` 传给 kernel |
| `src/tuning/**` | 计时走 `Device` 事件；候选由 `Plan` 表达，不感知实现 |
| `src/strategy.rs`、`src/target.rs` | 增加 `Target`（sm + kernel family）；`CudaTarget` 继续作为策略类型 |
| `src/resident/attention_mma/host.rs` | FA4 的 `.so` 加载归入 `tile` 实现；`aot` 构建不编译该路径 |
| `Cargo.toml` | 移除 host 对 `cutile`/`cuda-core`/`sha2` 的直接依赖，改为 `infer-cuda-kernels` feature |

迁移顺序按纵向切片，每步保持 13.x 行为不变：

1. `dense`（最小闭环，验证 Buffer/Stream/Plan）；
2. `gemv` + `norm` + `rope`（高频、无状态）；
3. `attention`（prefill/decode/verify，涉及 `KvView`）；
4. `recurrent`（GDN，涉及状态与 checkpoint）；
5. `state` + `sampling`（收口）。

每一步的退出条件是 13.x 全量测试与 smoke 通过、该算子数值逐位不变（canonical contract 下）。

## 7. 构建、feature 与 CI

### 7.1 Feature

```toml
# crates/backend/cuda/Cargo.toml
[features]
default = []
cuda = ["infer-cuda-kernels/tile"]
cuda-aot = ["infer-cuda-kernels/aot"]
flashinfer = ["cuda-aot", "infer-cuda-kernels/flashinfer"]

# crates/service/cli/Cargo.toml
cuda = ["dep:infer-backend-cuda", "infer-backend-cuda/cuda"]
cuda-aot = ["dep:infer-backend-cuda", "infer-backend-cuda/cuda-aot"]
```

`cuda` 与 `cuda-aot` 是 host 的两种构建配置，kernel crate 的 `tile`、`aot` 可以同时编入。

| 构建 | feature | 构建期需要 | 运行期需要 |
|---|---|---|---|
| 13.x / Blackwell | `cuda` | CUDA 13.2+ 头文件、`tileiras`（JIT） | 13.x 驱动 |
| 12.9 / L20 | `cuda-aot` | **无 CUDA 工具链** | 12.9 驱动（550 + 12.x 兼容） |
| 可选 FlashInfer | `flashinfer` | 无（shim 为预编译制品） | 12.9 驱动 + shim |

### 7.2 CI

- 现有 `cuda` job 不变（13.2 headers，`make test-cuda-hosted`）。
- 新增 `cuda-aot` job：普通 ubuntu runner，**不安装任何 CUDA 包**，运行
  `cargo clippy -p infer-backend-cuda --features cuda-aot --all-targets` 与 CPU 侧测试。这个 job 是
  “构建零工具链依赖”的证明，不能改成安装 headers 后通过。
- 设备 lane 拆分：`cuda-device`（现有 Blackwell 机器）与 `l20-device`（自托管）跑同一 `gate.sh`，只换
  feature 与制品 registry。
- 制品校验：`tools/cuda/kernels.py verify` 继续覆盖入库的 13.x 制品；AOT 制品在加载时校验 manifest 与 digest。
- `cargo --all-features` 继续排除 `infer-backend-cuda`；两个 feature 各自单测，禁止合并构建。

### 7.3 打包

- `build.rs` 的 plan 增加 `kernels` 字段（`tile` / `aot`），来源 `INFER_PACKAGE_KERNELS`，默认 `tile`；
  `--plan` 输出同步扩展。
- `tools/package/package.py` 依据该字段传 `--features cuda` 或 `cuda-aot`，归档名区分
  `linux-cuda` / `linux-cuda-aot`。
- `Makefile` 增加 `platform_targets linux-cuda-aot` 与 `test-cuda-aot`，保留 `test-cuda` 作为 13.x 别名。
- [打包指南](../../guides/packaging.md) 更新产物矩阵与“12.9 产物不含 Tile IR/JIT”的说明。

## 8. 制品：源码入库，制品出库

### 8.1 原则

Git 只提交 AOT 的源码（`.cu`）和 `tools/cuda/kernels.json` 的 `sidecar` 段（条目与每个架构的工具链 pin）。**AOT 的 cubin 不提交**；它们是构建产物，由本地 `make kernels` 或 CI 矩阵产出，进包，
不进仓库。

13.x 现有制品不变。"本机构建后入库"的模型解决的是"运行期不需要工具链、可逐字节复现"，但代价是二进制进 git、PR 噪音、
以及"谁的本机构建"无法作为 provenance。改为：

| 关注点 | 旧模型 | 新模型 |
|---|---|---|
| 源码 | git | git |
| 制品 | git | 构建产物（本地/CI） |
| 身份 | manifest 内嵌 hash | 工具链 pin + 构建 manifest |
| 复现 | 逐字节重建 | 固定 toolchain 镜像 + 同一配方 |
| 变更 | 大二进制提交 | 配方/lock 的 diff |

### 8.2 构建

```sh
make build-kernels-aot ARCH=sm_89 CUDA_HOME=/opt/cuda-12.9
# -> target/kernels/aot/sm_89/manifest.json + *.cubin
```

- 工具链 pin 在 `tools/cuda/kernels.json`，版本不符直接拒绝；manifest 记录实际 nvcc 版本。
- 12.9 / sm_89：`--cubin`（SASS），禁止 PTX-only；浮点开关沿用现有逐项固定。
- 13.x：`tile` profile 继续 JIT，本方案不改变它。
- 每个 cell 一个 manifest：符号、源码 hash、工具链、arch、contract ID、构建命令。

### 8.3 CI 与打包

```text
kernels job (matrix: profile x arch)  ->  upload artifact (manifest + cubins)
package job                           ->  download into target/kernels/<profile>/<arch>/
                                      ->  build CLI, assemble archive
```

- kernels job 不需要 GPU；只需要固定版本的 toolchain。
- package job 不重新编译 kernel，只消费已构建且 hash 已核对的制品。
- 归档内布局：

```text
infer-<version>-<target>-<profile>/
  bin/infer
  lib/infer/kernels/<profile>/<arch>/manifest.json + *.cubin
```

制品以 **sidecar** 形式随包分发，不 `include_bytes!` 进二进制：一个二进制支持多 arch，制品变更不触发
host 重编。需要单文件部署时再提供可选的内嵌模式。

### 8.4 加载与选择

- 启动时 `Target::detect(device)` 得到 `(arch, capabilities)`；读取 sidecar manifest；按
  `(profile, arch, family, contract)` 选 cell；`cuModuleLoadData` 加载。
- 没有对应 cell 时在加载期返回 `Unsupported`，不拖到第一个 token。
- 不读环境变量覆盖；`INFER_KERNEL_DIR` 只允许测试使用。
- 校验：manifest 的 hash 与文件一致、`cuobjdump` 的 arch 与目标一致，否则拒绝加载。

### 8.5 FlashInfer shim（可选）

- 默认不引入，不做运行时 FFI，自写优先。只有某个算子自写实现经测量明显落后，才按算子评估。
- 引入时只取算法或单个 kernel 源码，由同一流水线编成 cubin，pin 源码 commit 与产物 hash，挂在同一个算子族
  trait 后面，host 和架构都不变。
- contract ID 为 `vendor-flashinfer`，不与 `canonical-f32` 比较。
- 只允许出现在 `Recurrent` 与 `Moe` 两个 trait 后面；attention/GEMM 不接。
- shim 缺失时该能力返回 `Unsupported`，不静默回退。

### 8.6 现有 PTX 与 sidecar 的关系

只有一份清单 `tools/cuda/kernels.json`，一个入口 `kernels.py`；差异只在交付方式。

| 交付 | 适用 | 制品 | 校验 |
|---|---|---|---|
| 内嵌（`kernels`） | 13.x 上已验证的 PTX | 入库，编译时嵌入 | `kernels.py verify` 校验哈希与来源 |
| Sidecar（`sidecar`） | 需要 SASS 的设备 | 构建产物 | 工具链 pin，加载时校验 manifest 与 digest |

现有 PTX 不迁出仓库：打包 CI 没有 nvcc，oxide 和 FA4 的生成器难以进入 CI，制品哈希也是调优缓存身份的一部分；
总量约 4.4 MB，迁移收益不足以抵消这些代价。L20 需要的现有 `.cu` 以它为起点，在 sidecar 里按 sm_89 单独登记。

### 8.7 开发驱动与生产驱动

开发机（如 4090 / sm_89）可以使用比 550 新的驱动，生产 L20 保持 550。cubin 由固定的 12.9 nvcc 产出，新驱动可以加载。
新驱动会掩盖两类问题，所以需要单独把关：

- **PTX JIT：** 新驱动能 JIT 的 PTX 版本，550 不一定能；AOT 路径只发 cubin。
- **驱动符号：** `aot/driver.rs` 在加载时解析全部符号，新驱动才有的符号在 550 上会让设备打不开。

因此性能和调优结果只在 L20 上算数，发布前必须在 550 上跑完验收；开发机结果只说明功能可用。

## 9. 数值契约

| Contract | 定义 | 对照方式 |
|---|---|---|
| `canonical-f32` | 现有 F32 顺序与舍入规则 | 对自身冻结 golden 逐位一致 |
| `bf16-f32acc` | BF16 输入、F32 累加、BF16 输出 | reference-ops 容差 + 冻结 golden |
| `fp8-kv-f32acc` | FP8 KV、F32 累加、canonical 反量化 | reference-ops 容差 + 冻结 golden |
| `vendor-flashinfer` | 库自身语义 | 容差 + 端到端质量，不跨路径逐位 |

规则：

- 12.9 与 13.x **不做逐位对比**；各自对自己冻结的 golden。
- 跨路径只做容差与端到端质量；`INFER_CUDA_*` 诊断开关不进入生产判定。
- 每个 kernel registration 带 contract ID；manifest 与 registration 不一致时 `validate_program` 拒绝。

## 10. 设备接入 SOP

给拿到本方案的同学。以 L20 为首个实例，后续 sm_90 / sm_120 AOT 走同一流程。

1. **探测能力。** 记录 sm、显存、FP4/FP8、driver 版本、是否 550、是否数据中心卡（前向兼容只对数据中心卡）。
2. **选构建。** 设备能用 cuTile 走 `cuda`；不能用则 `cuda-aot`，此时不编 `tile`。
3. **定义 profile。** 在 `target.rs` 增加 `Arch`，在 `KernelRegistration` 里只注册真实
   可执行的算子与精度；未覆盖的返回 `Unsupported`。
4. **产出制品。** 按第 8 节生成 SASS、manifest、hash，运行 `kernels.py verify`。
5. **实现 trait。** 按算子族实现；先 `dense`，再 `norm/rope`，最后 `attention/recurrent`。
6. **数值。** 每个算子接 `reference-ops`；整模型接 golden；contract ID 写进 manifest。
7. **图与状态。** 验证 capture/replay、KV append/commit、checkpoint 恢复、取消/排空。
8. **实机验收。** `model smoke`、长上下文、并发 slot、显存报价；记录设备与驱动身份。
9. **更新文档。** [Backend](../../architecture/backends.md) 支持矩阵、[CUDA 说明](../../../crates/backend/cuda/README.md)
   限制、验收记录与路线图状态。

### 10.1 L20 v1 覆盖清单

| 算子 | 13.x 现状 | 12.9 实现 | 说明 |
|---|---|---|---|
| embedding | cuTile | 自写 PTX | gather |
| dense BF16/FP8 | cuTile | cuBLASLt | 逐通道 scale 需确认 |
| decode GEMV | cuTile | 自写 PTX | 带宽 bound |
| RMSNorm / gated | cuTile | 自写 PTX | |
| RoPE / MRoPE | cuTile | 自写 PTX | |
| prefill attention | cuTile / FA4 | 自写 `.cu` | FlashInfer 边界见 8.5 节 |
| decode attention | cuTile / `.cu` | 现有 `.cu` 重编 sm_89 | 保留 canonical F32 |
| GDN | cuTile / oxide | FlashInfer 或自写 | oxide PTX 需 12.8，550 不可用 |
| state append/commit | cuTile | 自写 PTX | |
| sampling | cuTile | 自写 PTX | |
| NVFP4 / MoE / vision | cuTile | 不覆盖 | 返回 Unsupported |

### 10.2 完成定义

```text
[ ] `cuda` 与 `cuda-aot` 各自能独立构建，`cuda-aot` 不链接 cuTile
[ ] host 无 cutile/cuda_core/libloading 类型
[ ] 13.x 全量测试、smoke、数值不变
[ ] 12.9 制品全部为 SASS，verify 通过
[ ] 每个注册算子有 reference-ops 对照与 contract ID
[ ] L20 实机 smoke、图 replay、KV 生命周期、显存报价通过
[ ] 支持矩阵与限制写入文档
[ ] CI 增加无工具链的 cuda-aot job 与 l20-device lane
```

## 11. 阶段与退出条件

| 阶段 | 内容 | 退出条件 | 可回退点 |
|---|---|---|---|
| Phase 1 | `cuda-kernels` 骨架 + `Buffer/Device/Stream` + `tile` 实现 + `dense` 迁移 | 13.x 全量测试通过，`dense` 数值与性能不变 | feature 关闭，回到现状 |
| Phase 2 | 其余算子迁到 trait；host 清除 cuTile 类型 | 13.x smoke 与数值逐位一致 | 按算子回退 |
| Phase 3 | `aot` 实现：驱动 dlopen、cubin 加载、cuBLASLt、sm_89 制品 | 无 CUDA 工具链构建通过；L20 smoke 通过 | `cuda-aot` 不发布 |
| Phase 4 | 打包、CI lane、L20 验收 | 两个归档可复现；L20 数值/图/显存通过 | 只发 13.x 归档 |
| Phase 5（可选） | FlashInfer 接 GDN/MoE | 与自写实现对比有明确收益与数值合同 | 关闭 `flashinfer` feature |

## 12. 风险与反模式

**风险**

- HAL 重构触及热路径：用纵向切片和 13.x 逐位不变作为闸门，不接受“先重构后补测试”。
- cuTile 的 future/graph 语义与裸 stream 不完全等价：`tile` 实现必须自己保证 capture 期间的顺序，
  不能依赖 host 做隐式同步。
- 两个 feature 的依赖可能被其他成员合并启用：CI 必须显式检查 `cuda` 与 `cuda-aot` 各自的图，不能只跑
  `--all-features`。
- 打包 plan 的 `kernels` 字段会影响归档身份与验收：改动必须同步 `build.rs`、`package.py`、Makefile 和文档。
- FlashInfer 迭代快：pin commit、pin 制品 hash、运行期无 JIT；升级按独立变更验收。

**反模式（评审直接拒绝）**

- host 里出现 `if cuda12 { ... } else { ... }` 或运行期 kernel 选择；
- 保留 `cutile::Tensor` 作为 host 的公共类型；
- 两套 KV/page 布局，靠 repack 维持；
- 为了省事提交 PTX-only 制品（550 无法 JIT）；
- 让 FlashInfer 的 paged 格式反向决定 `state` 层的布局；
- 复制 `resident/`、`tuning/` 或 `BackendProvider` 做第二份。

## 附录 A：文件清单

新增：

```text
crates/backend/cuda-kernels/Cargo.toml
crates/backend/cuda-kernels/src/lib.rs
crates/backend/cuda-kernels/src/contract/**
crates/backend/cuda-kernels/src/tile/**
crates/backend/cuda-kernels/src/aot/**
crates/backend/cuda-kernels/src/flashinfer/**
tools/cuda/build_aot.py
```

修改：

```text
Cargo.toml                         workspace members / dependencies
crates/backend/cuda/Cargo.toml     feature 与依赖
crates/backend/cuda/src/**         Tensor → Buffer，launch → traits
crates/service/cli/Cargo.toml      cuda-aot feature
build.rs                           plan 的 kernels 字段
tools/package/package.py           归档与 feature
Makefile                           platform_targets / test-cuda-aot
tools/check/layout.py              BACKENDS 增加 infer-cuda-kernels
tools/check/workspaces.py          如新增独立 workspace 才需要
docs/guides/packaging.md           产物矩阵
docs/architecture/backends.md      支持矩阵
```

## 附录 B：与现有规范的关系

- [Backend 能力与扩展](../../architecture/backends.md)：本方案不新增 backend，只新增 kernel provider，
  因此不触碰 `BackendKind`、CLI 选择或一致性套件的 backend 契约。
- [ADR-0004](../../adr/0004-performance-first.md)：AOT 覆盖是既定方向；本方案把 12.9 的 AOT 作为第一个
  完整实例，而不是第二个 backend。
- [Kernel 制品工程](../../../crates/backend/cuda/kernels/README.md)：现有 manifest/verify/重建规则继续适用，
  AOT 使用同一清单的 `sidecar` 段。
- [CUDA 后端分层](../../architecture/cuda-layers.md)：当前目录职责、launch 路径与改动入口；本方案第 6 节
  列出它与目标分层的差异。
- [代码布局](../../architecture/layout.md)：新 crate 落在 `crates/backend/<crate>`，依赖方向保持
  foundation → contracts → implementation。
