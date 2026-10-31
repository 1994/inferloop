# CUDA 后端分层与改动入口

读这一页可以定位三件事：`crates/backend/cuda` 每个目录负责什么、一次 kernel launch 的完整路径、
以及“我要改 X 该动哪里”。目标分层（host / kernel provider / AOT 制品）和 12.9 接入计划见
[CUDA AOT kernel provider 与 12.9 设备接入方案](../plans/engineering/cuda-kernel-provider.md)；
本页只描述当前实现（13.x / cuTile）。

## 1. 分层总览

```text
infer-cli / runtime
  │  BackendProvider / KernelProvider
  ▼
infer-backend-cuda
  ├─ executor/     provider、owner、submit/poll、状态、资源、prefix
  ├─ resident/     常驻图、capture、slot、各算子程序
  ├─ loading/      权重、量化、workspace 预算
  ├─ tuning/       自动选择、测量、baseline
  ├─ device.rs     驱动、stream、内存、设备能力
  ├─ device/       cuBLASLt、回读、上传、图像预处理、厂商 FP8/NVFP4
  ├─ vision/       视觉塔、MRoPE、图像会话
  └─ kernels/      cuTile 模块（当前与 host 同文件）
        │
        ▼
  CUDA driver / stream / graph
        +
  kernels/**       AOT 制品（.cu → PTX / cubin / .so）
```

当前 host 与 kernel **没有物理分离**：37 个文件里内联了 38 个 `#[cutile::module]`、146 个 entry，
108 个文件出现 `Tensor<`（1,492 处）。这是本页最重要的结构事实，也是目标分层要解决的耦合。

## 2. 目录职责

| 路径 | 职责 | 规模 | 目标层 | 什么时候动它 |
|---|---|---|---|---|
| `src/lib.rs` | 模块声明、`cfg(cuda_backend)`、导出 | 27 行 | host | 新增模块或 feature |
| `src/device.rs` | 设备打开、stream、内存、能力查询、GEMV 边界 | 812 行 | host + HAL | 设备生命周期、分配、stream 语义 |
| `src/device/` | `cublasLt`、回读、上传、异步、厂商 FP8/NVFP4、图像预处理、视觉位置 | 5,693 行，22 个文件 | HAL / IO | 厂商库签名、回读、nvJPEG、图像 staging |
| `src/executor/` | provider、owner、submit/poll、提交、状态、pool、prefix、profile、校验 | 5,352 行，17 个文件 | host | 提交协议、资源、状态报价、取消 |
| `src/resident/` | 常驻图、capture、slot 批处理、attention/GEMM/MoE/DFlash2/卷积程序 | 28,183 行，81 个文件 | host + kernel | 图几何、KV 布局、算子程序、checkpoint |
| `src/tuning/` | 候选、测量、自动调优、pool dense、NVFP4 Lt、投影预算、SwiGLU | 10,385 行，36 个文件 | host | 选择策略、baseline、fingerprint |
| `src/attention/` | 通用 dense attention 的计划与 kernel | 553 行，5 个文件 | kernel | 通用 attention 契约与路径选择 |
| `src/mlp/` | MLP、PDL、投影 | 499 行，5 个文件 | kernel | 融合与 PDL 候选 |
| `src/vision/` | 视觉塔、block MLP、device tower、footprint、MRoPE | 3,503 行，8 个文件 | host + kernel | 视觉模型接入 |
| `src/kernels.rs`、`src/registry.rs` | 通用 cuTile 模块、kernel 注册 | 141 行 | kernel / host | 新算子与注册 |
| `src/strategy.rs`、`src/target.rs` | 几何候选、架构 profile | 222 行 | host | 新架构、tile 候选 |
| `src/tuning.rs`、`src/benchmark.rs`、`src/constants.rs`、`src/attention_profile.rs`、`src/nvfp4.rs`、`src/quantized.rs` | 调优入口、基准、常量、profile、量化类型 | 1,394 行 | host | 对应策略 |
| `kernels/` | AOT 制品、manifest、编译/重建/校验脚本 | 10 组制品 | artifact | 新增或重建 AOT kernel |
| `baselines/tuning.json` | 按设备身份缓存的实测配置 | — | artifact | 新增设备或新配置实测 |
| `examples/` | 16 个设备/模型/诊断入口 | 38 个文件 | 验证 | 新增探针或验收入口 |
| `tests/` | 单元与支持模块，按 `#[path]` 挂载 | 142 个文件 | 验证 | 新增回归 |

## 3. 一次 launch 的路径

### 3.1 cuTile 路径（当前默认）

```text
api::zeros / api::meta          device.rs, loading/**
  → kernels::<family>::<op>     src/kernels.rs 或 resident/**/kernels.rs
  → .generics(vec![...])        launch 特化（dtype、tile、K）
  → .sync_on(&stream)           cuda_core::Stream，由 CudaDevice 持有
```

capture 期间的顺序由 `resident/capture.rs`、`resident/capture_*.rs` 保证；图几何和 arena 在
`resident/program.rs`、`resident/arena.rs`、`resident/capacity.rs`。调优在 `tuning/`，测量使用同一
stream 的事件。

### 3.2 AOT 路径（FA4 / oxide / nvcc / nvJPEG）

```text
resident/attention_mma/host.rs
  → libloading 打开内嵌 .so（hash 校验后落盘缓存）
  → C ABI infer_fa4_initialize / infer_fa4_forward
  → host.cpp.in 负责 TMA 描述符、cluster launch
```

oxide、nvcc 和图像预处理的 PTX 由 `kernels/` 内嵌，运行时通过 `cuModuleLoadData` 加载。AOT 路径
不经过 cuTile 编译器，但也没有统一的 kernel 契约，是目标分层要收口的部分。

## 4. 制品与工具链

| 项 | 当前事实 |
|---|---|
| 统一清单 | [tools/cuda/kernels.json](../../tools/cuda/kernels.json)，10 组制品 |
| 工具链 | nvcc 13.4.92；PTX `.version 9.4`（16 个中 13 个），8.7 两个，7.8 一个 |
| 目标架构 | `sm_120`、`sm_90a`、`sm_100a`、`sm_103a`、`sm_70`（portable 图像预处理）；4090（sm_89）没有 AOT attention |
| 重建 | 只写新目录，拒绝覆盖；字节一致校验与 provenance 在 manifest |
| 校验 | `make check-cuda-artifacts`、`tools/cuda/kernels.py verify`，也进入 `make check-rust` 与 `test-cuda-hosted` |
| 生成器 | nvcc（`.cu`）、cuda-oxide（Rust SIMT）、CuTe DSL（FA4 离线 AOT） |

12.9 / sm_89 的制品规则不同：**只接受 SASS cubin，不接受 PTX-only**（550 的 PTX JIT 只到 12.4），
并使用独立清单 `tools/cuda/aot-kernels.json`。设计见[接入方案](../plans/engineering/cuda-kernel-provider.md)第 8 节。

## 5. 改动入口

| 我要…… | 动这里 | 不要动 |
|---|---|---|
| 加一个通用算子 | `src/kernels.rs` 或对应 family 的 `kernels.rs`；`src/registry.rs` 注册；host 调用点 | 不在 host 文件里新写 `#[cutile::module]` |
| 改一个 AOT kernel | `kernels/<name>/*.cu` + `compile.py` / `regenerate.py` + manifest；`kernels.py rebuild` 到新目录 | 不覆盖已有 PTX/cubin；不改 hash 绕过校验 |
| 加一个架构 | `src/target.rs`；`kernels/` 新架构目录；`tuning/` 候选与 baseline | 不在 host 里按 GPU 型号分支 |
| 改 KV/状态 | `src/executor/state.rs`、`src/loading/kv_scales.rs`；逻辑生命周期归 `engine/state` | 不在 backend 重建页表或 prefix 语义 |
| 改图/capture | `src/resident/capture.rs`、`program.rs`、`arena.rs`、`capacity.rs` | 不复制第二套 slot/图 |
| 改提交/资源 | `src/executor/provider.rs`、`owner/`、`submission.rs`、`execution.rs` | 不绕过 `BackendProvider` 契约 |
| 改图像/视觉 | `src/device/image_preprocess/`、`src/vision/`、`kernels/image-preprocess*` | 不在模型 provider 里写设备格式转换 |
| 改调优 | `src/tuning/**`、`baselines/tuning.json` | 不用设备名或型号做分派 |
| 加设备/构建 | [接入方案](../plans/engineering/cuda-kernel-provider.md) | 不在 CLI 加第二个 CUDA backend |

## 6. 与目标分层的差异

| 当前 | 目标 |
|---|---|
| host 直接持有 `cutile::Tensor`（1,492 处） | `cuda-kernels::Buffer`，host 不出现 cuTile 类型 |
| kernel 模块内联在 37 个 host 文件 | `cuda-kernels/src/{tile,aot}` |
| host 直接用 `cuda_core`（13.0+ 头文件构建门禁） | `Device` / `Stream` 契约，AOT 路径可零工具链构建 |
| 一个 `cuda` feature | `cuda` / `cuda-aot` 编译期二选一 |
| 一份制品 registry | 13.x 与 12.9 两份，工具链与哈希隔离 |
| 12.9 不可用（无 Tile IR） | SASS + cuBLASLt；FlashInfer 仅可选补 GDN/MoE |

迁移按[接入方案](../plans/engineering/cuda-kernel-provider.md)第 11 节的阶段执行。在此之前，新代码尽量
落在目标一侧（kernel 进 family 模块、host 不新增 cuTile 类型、不新增第二套 KV 布局）。

## 7. 新同学阅读顺序

1. `src/lib.rs`（27 行）→ `src/device.rs` → `src/device/`
2. `src/kernels.rs` → `src/registry.rs`（一个算子如何注册与选择）
3. `src/executor/provider.rs` → `owner/` → `execution.rs`（提交、完成、资源）
4. `src/resident/capture.rs` → `program.rs` → `batch.rs`（常驻图与批处理）
5. `src/attention/` → `src/resident/attention_*`（attention 的两条路径）
6. `src/tuning/`（如何选 tile 与配置）
7. [kernels/README.md](../../crates/backend/cuda/kernels/README.md) → [tools/cuda](../../tools/cuda)

## 8. 维护约定

- 新增算子：kernel 放 family 模块，`KernelRegistration` 只注册真实可执行的精度与形状；未覆盖显式
  `Unsupported`。
- 新增架构：改 `target.rs`、加制品、补 baseline，不在 host 引入型号分支。
- 制品：只通过 `tools/cuda/kernels.py` 生成和校验，不手写 hash，不覆盖既有目录。
- 文档：当前契约进 `docs/architecture/`，待实施进 `docs/plans/`；改动后运行
  `python3 tools/check/layout.py`。
