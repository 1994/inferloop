# CUDA kernel 制品工程

本目录保存主机程序嵌入的 AOT 设备制品。通用 Rust/cuTile kernel 位于 `../src/`，
跟随 Cargo 构建；本目录通过统一的 [制品清单](../../../../tools/cuda/kernels.json) 管理。
开发工具入口是 [kernels.py](../../../../tools/cuda/kernels.py)，不进入服务运行依赖。

## 技术选择

| 职责 | 采用的实现 | 边界 |
|---|---|---|
| 设备生命周期、buffer、CUDA Graph、调度 | Rust + CUDA Rust host runtime | 共用 stream、内存和错误处理 |
| 通用 tile 算子、投影、融合 | cuTile Rust | 新增 tile 算子的首选，沿用现有 Cargo 工具链 |
| 必须控制 warp、共享内存、向量访存和运算顺序的 SIMT | CUDA C++ / nvcc AOT | 新增手写 SIMT 的默认入口；只生成 PTX，无 C++ host runtime |
| 已有 Rust SIMT attention/GDN | 固定 cuda-oxide + nightly | 保留已验证实现；有实测收益且工具链可复现时允许扩展 |
| FA4 tensor-core attention | 固定上游 CuTe DSL 离线 AOT | 作为第三方库管理；Python 只用于生成，运行时是固定 C ABI |
| cuBLASLt 等厂商库 | 已有 Rust adapter | 使用库算法，不复制为新的自研 kernel 系统 |

cuTile 和 cuda-oxide 属于 [NVIDIA CUDA Rust](https://nvidia.github.io/cuda-rust/) 的两种
编程模型；混用本身不是错误。当前自写 SIMT 已存在 nvcc 实现，而 oxide 还需要独立 nightly、
编译器后端及 Git SDK。新增 SIMT 默认优先 nvcc，以控制独立编译链的维护成本；存在明确实测收益且可复现时允许 Rust 实现。
已有 oxide 保留为已验证实现，不能仅因为语言统一就替换其 PTX。
45 的历史 GDN 对照中，同算法和启动配置的 C++ 流水线比 oxide 慢约 2.2%～5.0%；
它不是所有算子的结论，也没有测量 C++ host FFI 开销。具体证据见
[技术选型复核](../../../../docs/research/cuda-kernel-engineering-2026-10-23.md#技术选型复核)。
编译器文件丢失是制品管理问题，不是 Rust 性能或语言能力的证据。

不同 dtype、KV 布局、硬件能力和数值 profile 仍然需要不同算法。`shared`、`vector`、
`combined` 是同一 canonical 合同下的实现变体；BF16/FP16 FA4 是不同数值合同。
本次工程整理不改变运行时选择、默认 profile、调优缓存身份或已发布制品。

## 两种交付方式

同一份[制品清单](../../../../tools/cuda/kernels.json)管理两种交付方式，差异只在制品是否入库：

| 交付 | 适用 | 制品 | 校验 |
|---|---|---|---|
| 内嵌（`kernels`） | 13.x 上已验证的 kernel | PTX 入库，编译时嵌入 | `kernels.py verify` 校验哈希与来源 |
| Sidecar（`sidecar`） | 需要 SASS 的设备，比如 sm_89 / 驱动 550 | cubin 是构建产物，不入库 | 工具链 pin，加载时校验 manifest 与 digest |

内嵌制品保持入库：打包 CI 没有 nvcc，oxide 和 FA4 的生成器也难以进入 CI，而且制品哈希是调优缓存身份的一部分。

## 清单与验收

```sh
make list-cuda-kernels
make check-cuda-artifacts
python3 tools/cuda/kernels.py verify --kernel oxide-attention
```

一个清单登记 8 组制品的生成器、源文件、PTX、目标架构、符号和数值 profile。
新增制品目录或 PTX 未登记会失败。源码和产物 SHA256 继续由原有 provenance/manifest
持有，避免复制第二套哈希。5 个旧 `verify-artifact.py` 仅兼容转发到统一入口。

所有平台检查源文件、PTX、上游依赖身份与 AOT bundle 哈希；Linux 还执行 FA4 的
架构指令、ELF、C ABI 和动态依赖校验。macOS 明确报告该步骤需要 Linux，不宣称 ABI 已验收。
两个 oxide 制品在没有生成器、Toolkit 或 GPU 的 CI 上也会接受完整性检查。
`make check-rust` 和 `make test-cuda-hosted` 都调用同一入口。

## 重建

重建只写入新的输出目录，已有目录拒绝覆盖；不会覆盖仓库里的 PTX、SO 或资格记录。
每次保存 `result.json`、构建日志及执行命令。失败记录保持 `completed=false`。
所有构建均为离线编译，不执行 GPU kernel。

CUDA C++ 使用固定 nvcc 13.4.92，浮点开关逐项固定，不使用 fast-math。
这些开关会改变运算行为，见 [NVIDIA 浮点说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)。

```sh
python3 tools/cuda/kernels.py rebuild --kernel vector-attention-combined \
  --cuda-root /opt/cuda --output artifacts/kernel-rebuild/vector-v1
```

CUDA C++ PTX 必须逐字节一致。oxide 只允许共享符号 namespace 的一致重命名，
声明、布局、参数及其他 PTX 字节必须一致；不是通过删除注释或反汇编重排放宽比较。

```sh
python3 tools/cuda/kernels.py rebuild --kernel oxide-attention \
  --oxide-backend /toolchains/librustc_codegen_cuda.so \
  --output artifacts/kernel-rebuild/oxide-v1
```

oxide 需要 `cargo oxide` 在 PATH 中、仓库固定的 nightly 和 provenance 指定哈希的后端库。
构建在副本里进行，并检查 Cargo.lock 未变化。缺少固定编译器明确失败，不回退到最新版。
2026-10-23 在 45 上核查时，原实验目录里的后端 `.so` 已丢失；现有 PTX 可验证，
本轮未声称这两组已完成源码重建。恢复工具链或迁移算子应单独建立数值与性能证据。

FA4 的 `--upstream` 必须包含固定版本的 `flash_attn/cute` 源码，逐文件验证哈希。
`--sdk-site` 是带 nvidia-cutlass-dsl 4.8.0 的 site-packages；Python 环境还需 Torch 和
CUDA Python。quack-kernels 0.6.5、apache-tvm-ffi 0.1.11 可安装在独立目录，传入
`--quack` 和 `--tvm-ffi`，不能依赖已清理的 uv 临时缓存。

```sh
python3 tools/cuda/kernels.py rebuild --kernel fa4-attention-f16 \
  --python /path/to/python --cuda-root /opt/cuda \
  --upstream /path/to/flash-attention --sdk-site /path/to/site-packages \
  --quack /path/to/build-deps --tvm-ffi /path/to/build-deps \
  --output artifacts/kernel-rebuild/fa4-f16-v1
```

FP16 的生成脚本已从机器实验目录收回到 `tools/cuda/compile_f16.py`，完整重建检查
PTX、cubin、host header/object、SO 与 bridge。BF16 使用相同入口选择 `fa4-attention`，
委托现有冻结的六目标生成器；其脚本也在原 manifest 哈希链内，不能随意格式化使身份失效。
host object/SO 也要求逐字节一致，所以不同 C++ 编译器可能失败；使用原记录中的工具链，
或将新产物作为需要重新验收的变更，不能通过修改哈希绕过检查。

## 修改与准入

1. 先修改源码及相应测试，登记/更新构建参数和数值合同。
2. 在新输出目录生成候选，保留完整日志。`rebuild` 用于复现冻结制品；源码改变时应在
   独立候选目录运行相应生成器，不把 `verify` 失败解释为可以刷新所有哈希。
3. 对候选进行独立数值、CUDA Graph、完整模型质量和服务性能验收；方法见
   [CUDA 性能指南](../../../../docs/guides/cuda-performance.md)。
4. 审查源码、产物和新证据后更新 provenance/manifest，以及必要的运行时 fingerprint。

完整性校验与字节重建证明的是来源和复现能力。它们不会自动提升质量资格，也不能代替
硬件运行或证明新设备上的性能。
