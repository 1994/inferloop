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

前缀缓存的匹配、预算、LRU 和统计统一在 `PrefixCache<V, M>`。backend 通过 `PrefixPolicy`
选择索引，并实现 `PrefixReuse<V>` 获取 lease 或恢复设备快照；无需复制公共流程。
CUDA 的连续/FP8/target/draft 快照与 Metal 的共享页保留各自布局和完成协议，见 [KV 合约](kv-manager.md)。

backend 可通过 ExecutionProfileInspection.concurrency 发布加载期的显存界限、
计算/带宽拐点和实际选定宽度。runtime 先保留完成协议容量作为请求上限，再在创建
scheduler workspace 与主机完成缓冲前应用 backend 建议和较小的显式上限。
CUDA 仅在完整 pool capture 通过预算校验后发布该选择；未知计算吞吐不伪造拐点，
仍保留显存与协议约束。未报告建议的 backend 不具有自动并发最优性保证。

## CUDA resident 准入

CUDA capture、projection workspace 与新建序列报价共用 ProgramGeometry：verification 和 wide/narrow prompt 各自记录实际捕获行数。省略宽 prompt 图的 pooled 程序只计入窄图的 activation arena；external/fusion 配置缓冲仍按实际分配上界计费。prefill 需要 verification fallback 时保留相应报价。

cached shell 按 capacity/readout 与实际 take 路径匹配。转入 slot pool 会销毁私有 verification，但不会自动销毁原宽 prompt 图；复用报价保留旧 charge 的下界，并覆盖请求 private recapture 的新建报价。公共 state_reservation_bytes_for 与实际 reserve 使用同一算法，空闲 slot 不免除 prefill 私有程序的成本。

图几何使用加载时由设备能力/tuning 得出的宽度，不按设备名称选择。报价是已知分配加 driver headroom 的保守上界，不能称作实测增量显存。实测收益与并发延迟取舍见[图几何准入实验](../archive/research/cuda-pooled-admission-geometry-2026-10-12.md)。

## CUDA 执行契约

CUDA compute 默认采用非阻塞 submit/poll，无需环境变量。prefill、首次 slot 绑定、普通 decode、私有程序以及 MTP draft/verify/rollback 共用一个有所有权的执行 future；每次 poll 仅推进已经就绪的阶段，设备完成通知唤醒既有 device-owner 线程。加载、分配与 graph capture 在加载或资源准入阶段完成，不进入 compute flight。

一个 backend 同时拥有一个 compute flight。ticket 绑定 owner 与不复用的 generation，重复或跨 owner 消费会失败。完成结果、游标与资源视图在最终 stream fence 成功且 ticket 被消费后发布；ticket 丢弃、取消和 CPU 返回均不释放仍在使用的设备或 pinned 缓冲。无法确认 fence 时保留所有物理 owner，并使 backend 进入故障隔离，禁止复用状态。

readback 使用有界、可复用的 pinned 缓冲；MTP host hidden 上传复用预分配的 pinned staging，不逐 token 分配设备上传张量。图回放、D2D 绑定、draft catch-up 和状态恢复保持同一 stream 的顺序。独立诊断 API 的同步兼容包装不参与 provider 的 compute 路径。上述完成协议不代表多个 compute flight 的重叠执行或生产 SLO 达标，性能结论以配对 release 测量为准。

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

CUDA provider 已接入 CLI/HTTP，执行设备驻留图、状态/图复用、受限连续解码批处理与贪心 MTP。批处理受 slot、KV 容量和请求兼容性约束，不能视作任意模型或批次均可融合。模型提供 topology、draft 和精度策略；设备侧选择 kernel 与融合，调度不依赖具体实现。

| 范围 | 当前限制 |
|---|---|
| Attention | 通用 dense F32、padded head ≤256；paged/量化 KV、MLA 与稀疏/线性 attention 不属于该 provider 的覆盖范围。Candle 性能门禁未放行，见 [测量指南](../guides/cuda-performance.md) |
| CUDA 执行 | compute 全路径默认采用非阻塞 submit/poll；尚无通用异步多 compute flight、全设备采样或完整生产 SLO 验收。cuTile 运行时 JIT 需要相应工具链，非完整 AOT 制品 |
| 精度与设备 | 按能力及 `PrecisionPolicy` 选择；Hopper 的 NVFP4→BF16 转换不代表 H200 实机已验收。`test-cuda`（别名 `check-cuda`）含 Blackwell FP4 检查，不能作为所有 CUDA 设备统一通过的证明 |
| 模型与模态 | 内置 provider 与算子覆盖有限，不保证任意 HF 包可运行。Engine/Scheduler 已支持预处理的图像/视频 media plan 与融合 media prefill；音频、混合 image/video prompt 及完整 DeepStack 路径未覆盖 |
| 状态与扩展 | 物理分配/淘汰 owner 仍在 backend；跨 GPU、远程 PD、offload 和动态 C ABI/WASM 加载不属于当前支持范围 |
| 性能 | CPU 分配门禁只覆盖指定路径，不代表全进程零分配；单模型/单设备实验不能代表通用 P99 或吞吐达标 |

模型扩展见 [新增模型](../guides/adding-a-model.md)，CUDA 工具链与执行入口见 [CUDA README](../../crates/backend/cuda/README.md)。服务协议限制见 [OpenAI API](../guides/openai-api.md)，诊断能力见 [Agent](agent.md)。


CUDA 的请求长度接纳在异步 reservation 中选择实体 serving 计划。`serve` 在监听前
用 `CudaBackend::with_pool_captured` 按零需求报价捕获共享池，首个请求即绑定槽位；
调优 worker 与一次性命令不预先捕获，此时启动期只报告完整报价支持的并发上限，
`pooled_prefill_width` 缺失，单请求复用私有图，实际并发重试后才建立共享池并报告 worker
宽度和实体槽数。普通资源 quote 不捕获图，活动状态、slot lease 或设备 flight 未完成时
不重建池。已保留的请求 KV 页覆盖实际 prompt + 最大输出长度，物理空闲页与逻辑
上下文容量分别统计；超出共享池容量的私有请求保留独立报价和回收路径。


线程 owner 在资源控制变更后获取执行配置，并随已有不可变 snapshot 发布。
decode 只共享配置 Arc，不逐 token 克隆诊断字符串；资源确认返回前，配置读回与当前
backend 保持一致。实际物理页已经满足请求时复用池，避免更短的单请求触发扩大图。
