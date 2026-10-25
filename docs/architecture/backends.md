# Backend 能力与扩展

正式设备架构为 NVIDIA CUDA 与 Metal；CPU 标量参考算子位于 `testing/cpu`，只用于测试与数值对照，不注册 backend。具体路径的限制见下文。

## 能力与目录

`DeviceCapabilities.backend` 使用带 payload 的 `DeviceBackend`：`Cuda(NvidiaCapabilities)` 或 `Metal(MetalCapabilities)`，没有平行的 `nvidia: Option` 身份字段；`CapabilityRequirements.backend` 同样携带后端专属要求，跨 backend 匹配会被拒绝。

| 分组 | 内容 |
|---|---|
| Common | device、dtype、memory 与公共诊断 |
| CUDA | compute capability / SM、Tensor Core、warp、graph、TMA / cluster、pinned / IPC / NVLink / GPUDirect |
| Metal | SIMD、storage 与 Apple GPU 专属要求 |
| Backend API / SPI | compile、allocation、submit / poll、资源确认、recipe 与生命周期 |

能力定义放在 IR hardware 对应子模块，执行器、loader 与 device FFI 放在 `crates/backend/<name>`，CLI 的选择 adapter 放在 `crates/service/cli/src/backend`。新增 backend 的目录与注册约定见 [Backend 分组说明](../../crates/backend/README.md)。

## 选择与隔离

发布版提供 `--backend auto|cuda|metal`：`auto` 按 CUDA → Metal 选择已实现且可用的设备，无可用设备时返回 Unsupported。CUDA 需要显式启用 `cuda` feature，并通过设备与模型能力检查。产品不提供 CPU 执行 backend。

`ModelIr` / `DataflowGraph` 描述共有语义，`KernelRegistration` / `ExecutionProgram` 绑定 backend。编译与执行都会校验能力、精度、kernel 与 program target；更换 backend 需要重新编译，checkpoint 不能直接跨 backend 恢复。

Runtime 通过静态 BackendProvider 协调资源与 ticket。具体设备 buffer / handle 留在 backend，公共 trace / probe / stats 留在 IR；Metal 不依赖 CPU executor。

## CUDA resident 准入

CUDA capture、projection workspace 与新建序列报价共用 ProgramGeometry：verification 和 wide/narrow prompt 各自记录实际捕获行数。省略宽 prompt 图的 pooled 程序只计入窄图的 activation arena；external/fusion 配置缓冲仍按实际分配上界计费。prefill 需要 verification fallback 时保留相应报价。

cached shell 按 capacity/readout 与实际 take 路径匹配。转入 slot pool 会销毁私有 verification，但不会自动销毁原宽 prompt 图；复用报价保留旧 charge 的下界，并覆盖请求 private recapture 的新建报价。公共 state_reservation_bytes_for 与实际 reserve 使用同一算法，空闲 slot 不免除 prefill 私有程序的成本。

图几何使用加载时由设备能力/tuning 得出的宽度，不按设备名称选择。报价是已知分配加 driver headroom 的保守上界，不能称作实测增量显存。实测收益与并发延迟取舍见[图几何准入实验](../archive/research/cuda-pooled-admission-geometry-2026-10-12.md)。

## Metal 执行契约

Metal 在加载期上传 F32 / BF16 / F16 权重、编译 MSL pipeline，预绑定节点并按 lifetime 分配 scratch。计算与状态使用 F32；每个 compute flight 独占共享 scratch，深度为 1。ticket 绑定 executor owner，完成后才能读取 shared output。

shared storage 用于 CPU / device 可见数据，private storage 用于设备私有数据；共享地址仍须遵守同步与 reader 生命周期。raw pointer 与系统 selector 限制在 device FFI 边界，同步规则见 [Apple 文档](https://developer.apple.com/documentation/metal/resource-synchronization)。

模型算子与 retention 见[模型执行](../guides/model-execution.md)，KV / COW / prefix 与物理 checkpoint 见 [KV Manager](kv-manager.md)。GPU profile 记录 command buffer 时间，OpTrace 记录 CPU encoding，二者都不能当作逐 kernel 的 GPU counter。

## 新增 backend

1. 在对应分组定义能力 / 要求与 loader，公共结构不增加厂商专属字段。
2. 实现编译、权重上传、状态报价 / recipe、资源确认、submit / poll 与 fence / reader 契约。
3. 注册匹配目标的 kernel，并通过独立数值、状态 invariance、取消 / 排空与跨 owner 拒绝测试。
4. 接入 CLI catalog、硬件门禁、profile 与原始性能报告；未经过实机执行的能力保持明确的 Unsupported。

## CUDA 与支持限制

CUDA provider 已接入 CLI/HTTP，执行设备驻留图、状态/图复用、受限连续解码批处理与贪心 MTP。provider 当前同步执行；批处理受 slot、KV 容量和请求兼容性约束，不能视作任意模型或批次均可融合。模型提供 topology、draft 和精度策略；设备侧选择 kernel 与融合，调度不依赖具体实现。

| 范围 | 当前限制 |
|---|---|
| Attention | 通用 dense F32、padded head ≤256；paged/量化 KV、MLA 与稀疏/线性 attention 不属于该 provider 的覆盖范围。Candle 性能门禁未放行，见 [测量指南](../guides/cuda-performance.md) |
| CUDA 执行 | 同步 provider；尚无通用异步多 compute flight、全设备采样或完整生产 SLO 验收。cuTile 运行时 JIT 需要相应工具链，非完整 AOT 制品 |
| 精度与设备 | 按能力及 `PrecisionPolicy` 选择；Hopper 的 NVFP4→BF16 转换不代表 H200 实机已验收。`test-cuda`（别名 `check-cuda`）含 Blackwell FP4 检查，不能作为所有 CUDA 设备统一通过的证明 |
| 模型与模态 | 内置 provider 与算子覆盖有限，不保证任意 HF 包可运行。图像与视频使用专用示例（多图与视频走融合 media prefill），Engine/Scheduler 多模态调度、音频及完整 DeepStack 路径未覆盖 |
| 状态与扩展 | 物理分配/淘汰 owner 仍在 backend；跨 GPU、远程 PD、offload 和动态 C ABI/WASM 加载不属于当前支持范围 |
| 性能 | CPU 分配门禁只覆盖指定路径，不代表全进程零分配；单模型/单设备实验不能代表通用 P99 或吞吐达标 |

模型扩展见 [新增模型](../guides/adding-a-model.md)，CUDA 工具链与执行入口见 [CUDA README](../../crates/backend/cuda/README.md)。服务协议限制见 [OpenAI API](../guides/openai-api.md)，诊断能力见 [Agent](agent.md)。
