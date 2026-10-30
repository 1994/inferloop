# CUDA AOT kernel provider 与 12.9 设备接入方案

状态：设计，未实施。首个目标设备为 L20（sm_89 / CUDA 12.9 / 驱动 550.144）；现有 13.x 路径不回退。
整体排期见[路线图](../README.md)，后端能力边界见 [Backend](../../architecture/backends.md)，CUDA 现有实现见
[CUDA 说明](../../../crates/backend/cuda/README.md)。

本文回答一个问题：**没有 Tile IR 的 CUDA 环境（12.9 / sm_89）如何跑同一套 engine，同时不改 host 的
KV、图、调优和准入逻辑。** 结论是新增一个编译期可选的 AOT kernel provider，host 只依赖
`Buffer` / `Kernels` 契约；设备接入退化为“加 profile、加制品、过验收”。

## 0. 启动前提

本方案不是默认路线。启动前先完成两项验证，任一成立则不需要它：

1. **前向兼容验证（1–2 天）。** 在 L20 + 550.144 上安装 `cuda-compat-13-4`，用现有 cuTile kernel 跑通
   最小形状与一个模型 smoke。成立则保留现有栈，只补 sm_89 profile，见第 11 节 Phase 0。
2. **范围确认。** L20 v1 是否只需要 dense BF16/FP8。需要 GDN / MoE / vision 时，FlashInfer 从可选变为
   必选，边界见第 9 节；不需要时不要引入。

12.9 的驱动前提是 CUDA 12.x minor version compatibility（最低 525），550.144 满足。但 **550 的 PTX JIT
只到 12.4（PTX 8.5）**，因此 12.9 制品必须是 sm_89 的 SASS，不能只给 PTX；这一条是硬约束，见第 8 节。

## 1. 目标

一句话：**host 只认 `Buffer` / `Kernels` 契约，kernel 用编译期 feature 选 cuTile 或 AOT；新增设备 =
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
2. **编译期二选一。** `cuda` 与 `cuda-aot` 互斥，`compile_error!` 守卫；不存在运行期分支。
3. **一个 provider、一套 KV/图/调优/准入。** kernel 只是实现，不决定 page 格式、workspace 或调度。
4. **制品与数值 contract 是一等公民。** 每个 kernel 有 manifest、hash、arch、toolchain 和 contract ID。
5. **`cuda-aot` 构建零工具链依赖。** 驱动与 cuBLASLt 全部 dlopen，工件离线产出并入库。
6. **数据布局由 host 定义，kernel 适配 host。** 外部 kernel 库（FlashInfer）只能适配既有布局，不能反向
   决定 KV 格式；例外处理见第 9 节。

## 4. 分层

```text
crates/backend/
  cuda/                  host：provider、executor、resident、loading、tuning、strategy
  cuda-kernels/          新增：设备运行时 + kernel 契约 + 两个互斥实现
    src/lib.rs            Buffer / Device / Stream / Plan / traits（无驱动依赖）
    src/tile/             feature "tile"：cuTile 实现，沿用现有 kernel 体（13.x）
    src/aot/              feature "aot"：SASS 制品 + cuBLASLt（12.9）
    src/flashinfer/       feature "flashinfer"：可选 shim，仅 GDN / MoE
```

| 层 | 职责 | 依赖 |
|---|---|---|
| `infer-backend-cuda` | 能力选择、加载、常驻图、调优、`BackendProvider` | foundation、model、engine 契约、`infer-cuda-kernels` |
| `infer-cuda-kernels` | buffer/stream/图捕获、kernel 契约与实现、制品加载 | foundation（`infer-core`、`infer-ir`） |
| `infer-cuda-kernels/tile` | cuTile 编译与 launch | `cutile`、`cuda-core`（仅 13.x） |
| `infer-cuda-kernels/aot` | 驱动 dlopen、cubin 加载、cuBLASLt | 无 CUDA 构建依赖 |

只拆一个 crate 而不是两个：契约与两个实现共享同一组类型，拆成 HAL + kernels 会多出第三个 manifest 和
跨三处的 feature 组合；一个 crate 内的互斥 feature 提供同等隔离，只有一个边界要维护。

`crates/backend/cuda/kernels/` 继续放 AOT 制品；新增 `kernels/aot/` 子目录与独立 registry，见第 8 节。

## 5. 契约

### 5.1 Device / Stream / Buffer

```rust
pub struct Device { /* feature 内部实现 */ }
pub struct Stream { /* 同一 stream 语义，两种实现 */ }
pub struct Buffer<T> {
    ptr: DevicePtr,
    layout: Layout,
    owner: Arc<Allocation>,
    _dtype: PhantomData<T>,
}

impl Device {
    pub fn open(ordinal: usize) -> Result<Self>;
    pub fn alloc<T: Pod>(&self, shape: &[usize]) -> Result<Buffer<T>>;
    pub fn upload<T: Pod>(&self, dst: &Buffer<T>, src: &[T]) -> Result<()>;
    pub fn download<T: Pod>(&self, src: &Buffer<T>) -> Result<Vec<T>>;
    pub fn begin_capture(&self) -> Result<()>;
    pub fn end_capture(&self) -> Result<Graph>;
    pub fn sync(&self) -> Result<()>;
}

impl<T> Buffer<T> {
    pub fn shape(&self) -> &[usize];
    pub fn view(&self, shape: &[usize]) -> Result<Buffer<T>>;
    pub fn transpose(&self, a: usize, b: usize) -> Result<Buffer<T>>;
    pub fn slice(&self, axis: usize, range: Range<usize>) -> Result<Buffer<T>>;
}
```

约束：

- `Buffer` 不向 host 暴露裸指针；需要指针的 launch 只在 impl 内。
- `partition` 只存在于 impl。host 不再持有 partitioned view，改为向 kernel 传 `Plan`，由 impl 计算
  grid、tile 和 offset；这是与现状最大的语义变化，迁移时以编译器错误驱动。
- 分配、拷贝、capture、事件必须由同一实现对；两个 impl 共享同一组契约测试（见 10.8）。
- `Device` / `Stream` / `Buffer` 是具体类型而不是 trait 对象，避免泛型方法与非对象安全。

### 5.2 Kernel traits

按算子族拆分，全部 object-safe，参数只允许 `Buffer`、`Plan` 和标量：

```rust
pub trait Projection: Send + Sync {
    fn dense(&self, plan: &LinearPlan, out: &Buffer<f32>, x: &Buffer<f32>, w: &Buffer<f32>) -> Result<()>;
    fn dense_bf16(&self, plan: &LinearPlan, out: &Buffer<f32>, x: &Buffer<f32>, w: &Buffer<Bf16>) -> Result<()>;
    fn dense_fp8(&self, plan: &LinearPlan, out: &Buffer<f32>, x: &Buffer<f32>, w: &Buffer<Fp8>) -> Result<()>;
    fn gemv(&self, plan: &LinearPlan, out: &Buffer<f32>, x: &Buffer<f32>, w: &Buffer<Bf16>) -> Result<()>;
}

pub trait Attention: Send + Sync {
    fn prefill(&self, plan: &PrefillPlan, q: &Buffer<f32>, kv: &KvView<'_>, out: &Buffer<f32>) -> Result<()>;
    fn decode(&self, plan: &DecodePlan, q: &Buffer<f32>, kv: &KvView<'_>, out: &Buffer<f32>) -> Result<()>;
    fn verify(&self, plan: &VerifyPlan, q: &Buffer<f32>, kv: &KvView<'_>, out: &Buffer<f32>) -> Result<()>;
}

pub trait Norm: Send + Sync { /* rmsnorm / gated / rows */ }
pub trait Rope: Send + Sync { /* prefill / mrope / decode */ }
pub trait Recurrent: Send + Sync { /* gdn chunk / decode / checkpoint */ }
pub trait Moe: Send + Sync { /* route / combine */ }
pub trait Vision: Send + Sync { /* conv / layernorm / gelu */ }
pub trait State: Send + Sync { /* append / commit / checkpoint */ }
pub trait Sampling: Send + Sync { /* greedy / top-k / top-p */ }

pub struct Kernels {
    pub projection: Arc<dyn Projection>,
    pub attention: Arc<dyn Attention>,
    pub norm: Arc<dyn Norm>,
    pub rope: Arc<dyn Rope>,
    pub recurrent: Arc<dyn Recurrent>,
    pub moe: Option<Arc<dyn Moe>>,
    pub vision: Option<Arc<dyn Vision>>,
    pub state: Arc<dyn State>,
    pub sampling: Arc<dyn Sampling>,
}

pub fn kernels(device: &Device, target: Target) -> Result<Kernels>;
```

约束：

- 不允许 `cutile::Tensor`、`at::Tensor`、FlashInfer 类型出现在签名里。
- dtype 用独立方法而不是泛型，保持 object-safe，错误信息可读。
- 每个方法对应一条 `KernelRegistration`，`source` 指向真实实现；不支持的算子返回
  `Error::unsupported`，**不注册**，避免“注册后内部 fallback”。
- `KvView` 是 host 的只读视图（page table、scale、长度），由 `state` 层构造；kernel 不得修改布局。

### 5.3 Plan 类型

`LinearPlan`、`PrefillPlan`、`DecodePlan`、`VerifyPlan`、`NormPlan`、`RopePlan`、`StatePlan` 放在
`cuda-kernels`，由 host 的 `strategy` / `tuning` 填充。Plan 只描述几何、dtype 和数值模式，不含设备句柄，
便于在两种构建间共享同一份 host 决策。

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

```rust
#[cfg(all(feature = "cuda", feature = "cuda-aot"))]
compile_error!("cuda and cuda-aot are mutually exclusive; build one kernel provider per artifact");
```

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
- 制品校验：`tools/cuda/kernels.py verify` 同时覆盖 `kernels.json` 与新增 `kernels-aot.json`。
- `cargo --all-features` 继续排除 `infer-backend-cuda`；两个 feature 各自单测，禁止合并构建。

### 7.3 打包

- `build.rs` 的 plan 增加 `kernels` 字段（`tile` / `aot`），来源 `INFER_PACKAGE_KERNELS`，默认 `tile`；
  `--plan` 输出同步扩展。
- `tools/package/package.py` 依据该字段传 `--features cuda` 或 `cuda-aot`，归档名区分
  `linux-cuda` / `linux-cuda-aot`。
- `Makefile` 增加 `platform_targets linux-cuda-aot` 与 `test-cuda-aot`，保留 `test-cuda` 作为 13.x 别名。
- [打包指南](../../guides/packaging.md) 更新产物矩阵与“12.9 产物不含 Tile IR/JIT”的说明。

## 8. 制品规范

### 8.1 目录

```text
crates/backend/cuda/kernels/aot/
  manifest.json
  sm89/
    attention-decode.cubin
    dense.cubin
    norm-rope.cubin
  sm89/flashinfer/
    libinfer_flashinfer_shim.so
```

### 8.2 规则

| 项 | 规则 |
|---|---|
| 工具链 | nvcc 版本写入 manifest 并断言；12.9 与 13.x 制品不共用目录 |
| 架构 | `-arch=sm_89`；**`--cubin`（SASS）**，禁止 PTX-only |
| 浮点 | 沿用现有逐项固定：`--fmad=false --ftz=false --prec-div=true --prec-sqrt=true` |
| 身份 | 每个 cubin 记录符号、源码 hash、工具链版本、arch、contract ID |
| 校验 | `cuobjdump`/ELF 检查 arch 与符号；hash 不匹配拒绝加载 |
| 加载 | `cuModuleLoadData` + `cuModuleGetFunction`；不 JIT、不读环境变量 |

### 8.3 FlashInfer shim（可选）

- 离线编译，入库为 `.so` + hash；运行期只 `dlopen`，不带 Python、不带 torch、不做 JIT。
- contract ID 为 `vendor-flashinfer`，不与 `canonical-f32` 比较。
- 只允许出现在 `Recurrent` 与 `Moe` 两个 trait 后面；attention/GEMM 不接，理由见第 9 节。
- shim 缺失时该能力返回 `Unsupported`，不静默回退到别的实现。

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
2. **选构建。** 有 Tile IR（13.x）走 `cuda`；没有则 `cuda-aot`。不允许两边都编。
3. **定义 profile。** 在 `target.rs` 增加 `Target`（arch + family），在 `KernelRegistration` 里只注册真实
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
| prefill attention | cuTile / FA4 | 自写 `.cu` 或 FlashInfer | 见第 9 节边界 |
| decode attention | cuTile / `.cu` | 现有 `.cu` 重编 sm_89 | 保留 canonical F32 |
| GDN | cuTile / oxide | FlashInfer 或自写 | oxide PTX 需 12.8，550 不可用 |
| state append/commit | cuTile | 自写 PTX | |
| sampling | cuTile | 自写 PTX | |
| NVFP4 / MoE / vision | cuTile | 不覆盖 | 返回 Unsupported |

### 10.2 完成定义

```text
[ ] `cuda` 与 `cuda-aot` 各自能独立构建，编译期互斥
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
| Phase 0 | `cuda-compat-13-4` 前向兼容验证 | L20 上现有 cuTile 最小 kernel 与模型 smoke 通过 | 成立则取消后续阶段 |
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
crates/backend/cuda-kernels/src/buffer.rs
crates/backend/cuda-kernels/src/device.rs
crates/backend/cuda-kernels/src/stream.rs
crates/backend/cuda-kernels/src/plan.rs
crates/backend/cuda-kernels/src/tile/**
crates/backend/cuda-kernels/src/aot/**
crates/backend/cuda-kernels/src/flashinfer/**
crates/backend/cuda/kernels/aot/manifest.json
crates/backend/cuda/kernels/aot/sm89/**
tools/cuda/kernels-aot.json
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
tools/cuda/kernels.py              支持第二份 registry
docs/guides/packaging.md           产物矩阵
docs/architecture/backends.md      支持矩阵
```

## 附录 B：与现有规范的关系

- [Backend 能力与扩展](../../architecture/backends.md)：本方案不新增 backend，只新增 kernel provider，
  因此不触碰 `BackendKind`、CLI 选择或一致性套件的 backend 契约。
- [ADR-0004](../../adr/0004-performance-first.md)：AOT 覆盖是既定方向；本方案把 12.9 的 AOT 作为第一个
  完整实例，而不是第二个 backend。
- [Kernel 制品工程](../../../crates/backend/cuda/kernels/README.md)：现有 manifest/verify/重建规则继续适用，
  新增 registry 复用同一入口。
- [CUDA 后端分层](../../architecture/cuda-layers.md)：当前目录职责、launch 路径与改动入口；本方案第 6 节
  列出它与目标分层的差异。
- [代码布局](../../architecture/layout.md)：新 crate 落在 `crates/backend/<crate>`，依赖方向保持
  foundation → contracts → implementation。
