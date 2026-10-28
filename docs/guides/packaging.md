# Zig 跨平台构建与打包

平台契约统一在根目录 [`build.rs`](../../build.rs)。CLI 的 Cargo manifest 将它注册为 build script；打包入口还将同一文件编译为宿主机工具，提前获取 target/backend/features 计划和 SDK 检查结果。不存在另一份 Python 平台矩阵。

`build.rs` 负责目标平台、glibc 参数、生产 feature 组合、macOS SDK 前置条件和 `OUT_DIR/infer-build.json`。它不递归启动 Cargo，也不在二进制链接前制作归档。Makefile 只传参数；`tools/package/package.py` 负责调用目标构建命令、核对 Cargo 回传的构建契约、组装归档和验证产物。

## 工具与目标

```sh
make setup-build                     # 安装 cargo-zigbuild 0.23.4；检查 Zig
rustup target add aarch64-unknown-linux-gnu
# macOS 交叉目标：
rustup target add aarch64-apple-darwin x86_64-apple-darwin
```

需要仓库指定的 Rust、Python 3.12+、Git 和 Zig。CI 使用 cargo-zigbuild 0.23.4 / Zig 0.16.0。`setup-build` 不替换系统 Zig；可通过 `CARGO_ZIGBUILD_ZIG_PATH` 选择已有版本。依赖和 ABI 机制见 [cargo-zigbuild 官方说明](https://github.com/rust-cross/cargo-zigbuild)。

| TARGET | 平台 | 生产 feature |
|---|---|---|
| `x86_64-unknown-linux-gnu[.<glibc>]` | Linux x86_64 / CUDA | `--no-default-features --features cuda` |
| `aarch64-unknown-linux-gnu[.<glibc>]` | Linux AArch64 / CUDA | 同上 |
| `x86_64-apple-darwin` | macOS Intel / Metal | `--no-default-features` |
| `aarch64-apple-darwin` | macOS Apple Silicon / Metal | 同上 |

省略 `TARGET` 使用 Rust host；显式指定时**允许跨 OS/架构**。平台别名和 target 冲突时失败，不修改 target 迁就宿主机。Linux 默认 glibc 请求基线为 2.28，可显式选择 2.17 / 2.28 / 2.31 / 2.34 / 2.35 / 2.36 / 2.39 / 2.41；不接受未知版本，避免 Zig 静默回退。Windows、musl、universal2 暂未提供生产打包契约。

## 一键流程

```sh
make package
make package-linux-cuda TARGET=x86_64-unknown-linux-gnu.2.28
make package-linux-cuda TARGET=aarch64-unknown-linux-gnu.2.28
SDKROOT=/path/to/MacOSX.sdk make package-macos-metal TARGET=aarch64-apple-darwin
make package-macos-metal TARGET=x86_64-apple-darwin

# 单独执行阶段；同样支持 TARGET
make build TARGET=aarch64-unknown-linux-gnu
make test-package TARGET=aarch64-unknown-linux-gnu
```

流程：契约/SDK 预检 → 目标检查 → release 编译 → ELF/Mach-O 架构核对 → 归档 → 解压完整性校验 → 本地产物目录。

宿主套件（布局、策略、格式、Clippy 与工作区测试）由同一条流水线的 Rust 与 tools job 负责，
`make package` 不再重复执行：打包 job 有 4 个目标，重复一次就等于把最贵的检查乘以 4。
`make test-package` 仍是打包流程的显式宿主套件入口（宿主目标执行套件；交叉目标编译其测试）；
仓库自身的测试入口是 **`make test`**（等于 `gate.sh test`，只跑工作区测试条目，不触发打包）。

- target 等于 host 时：启动解压后的 CLI `--version` / `--help`。
- 交叉 target：用目标构建命令**编译目标 CLI 测试，但不运行**；归档后验证文件摘要和实际机器码架构，不尝试在宿主机执行异构程序。manifest 明确记录 `cross-cli-tests-compiled-not-executed`、`smoke_executed: false`，以及这份包
依赖的 `source_checks`（`make check-rust`/`make check-tools`）与 `commit`/`dirty`。
- 在目标机器上再运行 `verify-package` 即可执行启动检查。GPU 推理需要下面的独立 `accept`。

`CARGO_TARGET_DIR`、`CARGO_BUILD_JOBS` 和 CUDA 工具链变量仍有效；工具从 Cargo JSON artifact 读取真实二进制路径，不猜测输出目录。显式保留 `--target`，不注入 `target-cpu=native`。macOS 主机构建 Apple Silicon 或 Intel macOS 目标时使用 `cargo build` 和 Apple 链接器，避免 Zig 对依赖动态库导出符号参数的处理错误；其他目标继续使用 `cargo zigbuild`，保留 Linux glibc 基线。

## 本机开发构建

`make local-build` 走与打包同一条路径：用同一个计划源解析宿主 target 的 backend 与 features，
跑同一个 preflight，再由同一个驱动按计划构建 release CLI。因此"本机构建"与"同一宿主 target 的包"
不会在 backend/features 上产生分歧——这正是方案要求的 native/cross 共用一个计划来源。

具体实现里只有一处环境发现：Linux + CUDA 时把 `gcc -print-file-name=include` 交给
`BINDGEN_EXTRA_CLANG_ARGS`（bindgen 自己找不到主机 C 标准头）。它现在在驱动的 preflight 旁边，
不再由 Makefile 按平台手写；找不到 `stddef.h` 时不注入，行为与之前一致。

## CUDA 与 Metal 的边界

Zig 提供 C/C++ 交叉编译和链接，不提供 NVIDIA 或 Apple SDK。CUDA build 仍需可用的 CUDA headers/libclang；必要时以 `CUDA_TOOLKIT_TARGET_DIR` 选择目标 Toolkit 的 `targets/` 子树。运行设备需驱动与 cuTile JIT 工具链，具体设置见 [CUDA 后端说明](../../crates/backend/cuda/README.md)。不把构建机的 CUDA 动态库复制进包。

从非 macOS 主机构建 Metal，必须设置有效的 `SDKROOT`，其中需有 Metal/Foundation frameworks；缺少 SDK 会在启动 Cargo 前明确失败。macOS 主机可使用 Xcode Command Line Tools 自动发现 SDK。SDK 链接成功也不等于已经在目标 GPU 完成验收。

## 产物与验证

默认输出 `artifacts/packages/infer-<version>-<zig-target>-<backend>/`，包含 `.tar.gz`、`SHA256SUMS` 与外部 `manifest.json`。包内包含 `bin/infer`、独立运行说明、LICENSE、Cargo.lock 和 manifest。

manifest schema 2 合并 `build.rs` 生成的 target/arch/backend/features/ABI 请求，附加宿主信息、Rust/Zig/cargo-zigbuild 版本、Git commit/dirty 状态、文件摘要和实际验证范围。target 字段来自 Cargo 目标，不使用宿主 `platform.machine()` 冒充目标架构。旧 schema 1 包不冒充新的交叉构建契约，验证时明确拒绝。

已有同名产物拒绝覆盖；用新的 `DIST_DIR` 保存再次构建。临时文件失败即清理，成功后原子移动产物目录。不会上传到外部服务。

```sh
make verify-package PACKAGE_FILE=/path/to/infer-....tar.gz
sha256sum -c SHA256SUMS        # macOS: shasum -a 256 -c SHA256SUMS
```

校验器拒绝路径穿越、链接、重名文件、遗漏文件、摘要不符和实际架构错误。同 target 上的启动验证在仓库外进行。校验和不是发行者签名；只对可信包运行启动验证。

在**目标机器**上以包内二进制做 GPU golden 验收：

```sh
make accept PACKAGE_FILE=/path/to/infer-....tar.gz \
  MODEL=/path/to/model GOLDEN=/path/to/golden.json
```

异构宿主、缺失模型/golden、设备不支持均报错。CUDA 继续通过 `safe-run.sh` 的内存和互斥保护运行，需要 systemd user session；Metal 直接运行。打包不自动修改 `gpu_inference_accepted: false`，不把未执行的检查写成通过。完整 Rust 门禁与 attention 性能仍分别执行 `make check-rust` / `make check-attention`。

## GitHub CI

[工作流](../../.github/workflows/ci.yml) 在 push、pull request、merge queue 和手动触发时运行。Rust 检查覆盖 Ubuntu 24.04 与 macOS 15；格式、Clippy、单元/golden、MSRV、工具和安全检查全部通过后才开始打包。

打包矩阵为 Linux CUDA x86_64 / AArch64（glibc 2.28）和 macOS Metal Apple Silicon / Intel。Linux 安装 CUDA 13.2 开发头文件，无需 GPU driver；macOS 使用 runner 的 Xcode SDK。两者使用 Python 3.12、Zig 0.16.0 和 Makefile 固定的 cargo-zigbuild。

`make package` 执行目标检查、release 构建和归档校验；与宿主同架构时执行 CLI 冒烟，交叉目标只做编译和静态检查。成功后上传归档、SHA256SUMS 和 manifest 作为 Actions 制品，保留 14 天。

推送与 workspace 版本一致的标签（例如版本 `0.1.0` 对应 `v0.1.0`）会运行完整门禁；全部通过后自动创建 GitHub Release，附上两个 Linux CUDA 架构的归档、各自的校验和与 manifest。发布前会核对版本、目标、CUDA feature、源码 commit 和归档校验和。已有同名 Release 不会被覆盖。下载同一架构的归档和 `.SHA256SUMS` 后，可用 `sha256sum -c <文件名>.SHA256SUMS` 验证。发布包仍需目标机器上的 NVIDIA driver 和 CUDA/cuTile JIT 工具链。

最终 `quality-gate` 要求所有质量检查和打包项通过，可用作分支保护的必需检查。hosted CI 不证明真实 GPU 的正确性或性能，manifest 保留 `gpu_inference_accepted: false`。设备验收使用 `make accept`、`test-cuda` / `test-metal` 和独立的 Candle attention 门禁。
