# NVIDIA CUDA 后端

`infer-backend-cuda` 使用 NVIDIA cuTile Rust 0.4.0 实现设备 kernel、驻留执行图、权重加载和 `BackendProvider`。模型拓扑由 provider 提供，后端按设备能力、dtype 和形状选择具体实现，不按模型名分派。

## Kernel 工程入口

通用算子沿用 Rust/cuTile；手写 SIMT 新增实现默认优先 CUDA C++ AOT，已有 oxide 保留为
已验证兼容实现，FA4 按固定版本的第三方 AOT 库管理。具体技术边界、清单、重建和准入流程见
[Kernel 制品工程](kernels/README.md)。`make check-cuda-artifacts` 在无 GPU 主机验证全部
嵌入制品，`make list-cuda-kernels` 查看统一登记。

## 构建与运行

需要 Linux、Rust、CUDA Toolkit 开发头文件、libclang 及标准 C 头文件；执行还需要 NVIDIA driver 与支持目标设备的 `tileiras`。打包 CI 使用 CUDA 13.2 开发头文件，本地设备验证使用 CUDA 13.4。构建通过不代表目标硬件与 JIT 工具链已验收。

```sh
export CUDA_TOOLKIT_PATH=/path/to/cuda
cargo build --locked --release -p infer-cli --features cuda
make test-cuda-hosted   # 编译、Clippy、Rustdoc 与 CPU 侧测试；不需要 GPU
make test-cuda          # 追加 GPU kernel 验收
```

bindgen 找不到 C 标准头时，将实际主机 include 路径传给 `BINDGEN_EXTRA_CLANG_ARGS`，例如 `-isystem /path/to/gcc/include`；该路径不能照搬其他机器。默认 CLI 不启用 CUDA feature，普通 hosted Rust 门禁不需要 Toolkit；需要 Toolkit 的固定检查集中在 `make test-cuda-hosted`。

## 条件编译边界

CLI 与 CUDA crate 的 build script 共用 [backend_cfg.rs](../../../tools/build/backend_cfg.rs)，
按 Cargo 目标平台和当前 crate 的 `cuda` feature 生成 `cuda_backend`、`metal_backend` 与
`native_backend` 条件。CUDA 当前仍只在 Linux 启用，整个库在 `lib.rs` 用
`#![cfg(cuda_backend)]` 一次性控制，内部模块不再重复判断 CUDA 是否启用。
CLI 在模块入口选择完整 CUDA 加载实现或不可用实现，已受父模块保护的代码不再重复条件。

未启用 CUDA 时该库不暴露 API；原先无 feature 可用的 `target`、`strategy`、`nvfp4`、
`registry` 与 `AttentionProfile` 也随整个后端启用。调用方应启用 `cuda` feature，
并满足平台及工具链要求。无 feature 门禁只检查编译隔离；原有纯主机测试仍保留，
由 `test-cuda-hosted` 在启用 CUDA 后实际执行。

新增平台时先补齐依赖与平台适配，再更新共享能力条件和 Cargo 的 target 依赖。
FA4 内嵌 ELF、动态库名称、FFI 与进程管理等真实 OS/ABI 限制仍留在各自边界，
不能仅扩大 `cuda_backend` 条件就声明支持 Windows。

## 执行边界

- BF16/FP8/NVFP4 投影、设备驻留 prefill/decode、状态/图复用与受限连续解码批处理。
- 可选贪心 MTP，由能力声明、draft 深度、容量与采样参数共同约束；输出仍需与关闭 MTP 的 target 路径对齐。
- 通用 dense attention 位于 `attention`，契约来自 `infer-kernel-api`；Candle 只存在于隔离 benchmark。
- 图像视觉塔与 MRoPE 通过专用示例验证，使用范围和复现方法见 [图像指南](../../../docs/guides/vision.md)。

当前同步 provider、设备采样、跨设备和完整服务性能的限制见 [Backend](../../../docs/architecture/backends.md)。kernel 正确性不等于完整模型质量或生产吞吐。

## 自动 tile 选择

自动调优目前覆盖标量 GEMV、驻留 NVFP4 GEMM、按通道缩放的驻留 FP8 GEMM 和原生 gated norm
的编译器 occupancy。设备能力与 dtype、布局、对齐、workspace 和显存预算先限定合法候选，再用实测择优；不按 GPU 型号或模型名分派。
Attention 拆分、图宽度和 GEMV/GEMM crossover 等策略尚未全部接入同一调优表。

机器本地缓存默认是 `~/.cache/infer-cuda/tuning.json`，遵循 `XDG_CACHE_HOME`，可用
`INFER_CUDA_TUNING_TABLE` 指定路径。GEMV 先查本地测量，再查仓库内置 `baselines/tuning.json`，
两者都按运行设备 identity（`名称|架构|显存MiB`）和 `(dtype, rows, columns)` 匹配。
缓存支持多个设备；旧的单设备 `device/entries` 格式读取时迁移，没有设备 identity 的条目不会复用。

驻留 GEMM 在同一个文件中使用独立的 typed section，与 GEMV 共用写入锁和原子替换，不混用策略。
其 key 包含真实硬件事实、CUDA API 版本、NVRM driver release/build、量化与数值合同、实际
`M/N/K`、激活模式及编译器/kernel fingerprint；FP8 Lt 还隔离库版本、算法配置和显式数值 profile。
无法取得可靠 driver build 时不跨进程复用驻留策略。性能搜索用首份真实权重代表同形状；
其他权重和 scale 使用缓存配置前，仍须通过各自完整输出的 finite/bitwise 检查。相同且仍存活的
权重 owner 可复用数值验收，形状命中不能替代新权重验收。

自动调优默认开启。服务 CLI 使用独立准备进程生成缓存，然后由干净的服务进程复用配置，
避免仅为搜索加载的设备模块减少服务 KV 容量。服务进程仍按自身真实权重校验缓存。准备进程
失败或超时（10 分钟）时退回已有有效缓存或保守配置；`--no-autotune` 跳过准备。缓存目录
不可写及 SDK 直接加载时仍在当前进程调优，因此需要计入调优驻留开销。两阶段启动多加载
一次模型，启动成本与持续推理性能分别报告。

NVFP4 和 FP8 共享每次模型加载的 30 秒新搜索软预算，target、draft 与后续
program 共用这份预算；greedy、norm、recurrent、attention 另有各自的 30 秒软预算。GEMM 预算
从首次新搜索开始，逐候选检查；已开始的 JIT 或测量不能强制中断，因此不保证整体启动耗时。
仅为实际将捕获的图宽度准备策略。搜索预算耗尽或证据未完成时使用保守配置，不把未完成的
baseline 写成已测缓存结果。缓存身份解析和已缓存配置的当前真实权重完整 finite/bitwise 验收
不受新搜索预算限制，避免搜索使已有已验证配置失效；身份不匹配或数值验收失败仍回退。

搜索按固定 AB 顺序成对测量，独立确认按 BA 反序成对测量，并使用新的 capture。
两轮都分别测冷 L2 和暖 L2，非 baseline 配置须在四组比较中均达到至少 2% 收益，并保持原量化与
数值合同；计时包含激活量化和 GEMM replay。单个候选不支持、数值不同或测量失败时淘汰该候选，
失败原因写入日志，基线失败则保守回退。kernel 测量门禁通过不等于已证明服务 TTFT、TPOT 或吞吐提升。

`--no-autotune` 禁止新的性能搜索，仍可复用有效缓存；驻留配置仍需当前真实权重的数值验收，
未命中有效缓存或数值验收失败时使用保守配置。硬件相关的 arena 预算、解码分片、投影和量化 tile 由 `tuning/`
统一选择，生产路径不再提供环境变量覆盖器；宽 prompt 的 FP8 投影默认使用设备库查询到的
cuBLASLt 算法。验证图批处理、SwiGLU 融合、chunked GDN 和 FP8 draft 投影默认启用。

窄图的 FP8 自动搜索保持原生 K 累积顺序的 tile；内部测试可通过 `INFER_CUDA_FP8_LT_AUTOTUNE=1`
（或 `true`）将 cuBLASLt 加入数值对照候选。宽 prompt 默认委托 cuBLASLt，SDK 不承诺它与原生
kernel 的 K 累积顺序相同。
四种激活 pattern 的全部输出通过 finite/bitwise 检查，只证明这些验收输入，不能证明所有输入等价。
这些 bitwise 检查属于内部数值诊断；生产默认路径的质量按任务准确率验收，
`INFER_CUDA_VENDOR_FP8` 仅用于独立测试诊断。

## 验收入口

| 入口 | 用途 |
|---|---|
| `make test-cuda-hosted` | 严格 Clippy、CPU 侧测试、Rustdoc 与示例构建；托管 CI 可运行，不需要 GPU |
| `make test-cuda` | 上述全部，再加 GPU kernel 检查 |
| `make check-attention` | 独立 F64 oracle 与 Candle 数值/性能门禁 |
| `cuda-resident-check` / `cuda-provider-check` | 驻留执行图、加载与 provider 回归 |
| `cuda-model-smoke` | 完整模型诊断与 MTP 对照 |
| `cuda-mma-probe` / `cuda-mlp-check --pdl` | lowering 与可选 PDL 能力探针 |

示例先通过 `cargo build --locked --release -p infer-backend-cuda --features cuda --example <name>` 构建，再通过 `tools/bench/safe-run.sh` 执行。数值与性能测量规范见 [CUDA 性能指南](../../../docs/guides/cuda-performance.md)。

## 职责

模型包负责权重格式、scale、shape 和执行图；`LinearStrategy` 提供合法候选；设备层管理 CUDA 生命周期与 typed buffer；kernel 负责计算，executor 负责提交、完成、状态和回滚。所有权与 checkpoint 遵守 `BackendProvider` / `KvCacheManager` 契约，prefill GEMM 与 decode GEMV 独立调优。

### 非阻塞执行

CUDA compute 默认采用提交/完成拆分，无需配置开关。prefill、首次 slot 绑定、
普通 decode、混合批次、私有 program 及 MTP draft/verify/rollback 都由有所有权的
future 推进。`submit` 入队首段工作后返回，`poll` 在等待设备时返回 `None`；
最终 stream fence 完成且 ticket 被消费后才发布结果、history 与 slot 游标。

在途 ticket 未消费前禁止 reset、release、再次 submit 与 prefix/control 操作；
丢弃 ticket 不会使 backend 空闲。graph、DMA source、pinned destination 和物理 pool
保持到设备完成；无法确认完成时隔离 backend 并保留 owner，禁止提前复用。
加载、分配与 graph capture 属于加载或资源准入，不在 compute flight 中执行。

当前每个 backend 仍只有一个 compute flight，未实现 ADR-0004 的深度 2 队列或
设备端 token/KV 延迟回填。同步诊断 API 通过同一 async 实现的兼容包装运行，
provider 不调用该包装。吞吐、TTFT/TPOT 和内存成本按性能 baseline 合同单独验收。

共享 slot 的状态不会因其他请求准入而重建私有验证图；私有验证只使用已报价的
存储，并在 reset 解绑恢复计账后准备。缺少 resident home、需要私有 fallback 的
请求从准入报价起覆盖私有验证存储，不能仅因当前 slot 空闲而少计；这可能使此前
漏计显存的请求更早等待资源。延迟创建共享池要求至少两个健康、Logits
readout 且容量适配的状态；Full readout 和超长私有请求不触发无用池捕获。

内部 execution profile 的 `phase_wall_us` 包含异步挂起和设备等待。固定批次诊断的
`submit_wall_us`、`poll_wall_us` 是调用期间经过的时间，包含线程被调度出去的时间，
均不代表线程 CPU 用时；历史证据保留其原始字段与对应源码版本。

设备验收包含 `async_prefill_first_binding_decode_and_mixed_readouts_match_serial`、
`async_mtp_pooled_and_private_match_serial`、
`async_invalid_mtp_sampling_fences_before_failure` 和
`async_abandoned_ticket_keeps_state_owned_until_backend_drop`；通过 `INFER_TEST_MODEL`
指定对应的 2B 或 MTP2 实机 checkpoint，缺少模型或设备会失败。
