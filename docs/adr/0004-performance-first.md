# ADR-0004：性能优先，先做执行重叠与 AOT 覆盖

- 状态：采用
- 日期：2026-10-25

## 背景

决策提出时，CUDA 有两个结构性性能限制：

1. **执行不重叠。** `CudaBackend::execute` 在 `submit` 内执行到结算，`poll` 只消费已完成的 ticket；runtime 的设备 compute depth 为 1。GPU 执行期间，CPU 不能准备下一批请求、处理输出或规划后续批次，launch、拷贝和调度间隙无法被隐藏。
2. **kernel 覆盖率不足。** 通用 cuTile 算子通过 `tileiras` 在加载/首次使用时 JIT；FA4、oxide 等只有部分 AOT 制品。结果是部署依赖 JIT 工具链，冷启动有编译成本，产物也不是完全自包含。

vLLM v0.31 的生成路径默认开启异步调度：`EngineCore` 用 `execute_model(non_block=True)` 返回 future，并以 `max_concurrent_batches=2` 的 batch queue 在当前批执行时调度下一批；`AsyncScheduler` 用占位 token 处理尚未返回的输出。kernel 侧则是混合策略：热点 GEMM/attention 多来自预编译库或扩展，Triton/`torch.compile` 只对部分融合做首次编译并缓存，CUDA Graph 降低 launch 开销。

这说明两件事：执行重叠是调度器、KV/状态生命周期和 executor 的共同改造，不是把 CUDA 调用丢到后台线程；kernel 也不需要追求“全 JIT”或“全 AOT”，应按稳定 shape 预编译、保留 JIT fallback。

## 决策

下一阶段以**性能**为首要目标，在数值、质量、状态和 baseline 合同不变的前提下，优先推进：

1. **执行重叠（P0）。** 将 CUDA provider 改为非阻塞 `submit` / 完成 `poll`；runtime 增加深度 2 的 batch queue，使当前批执行时能够准备下一批。scheduler 必须支持延迟回填的 token/KV 状态，不能假设上一批输出已经可见。
2. **AOT 覆盖（P1）。** 对稳定 shape 和目标架构预编译 cuTile/CUDA C++ 制品，随包分发；JIT 只作为新 shape、新架构和未覆盖算子的 fallback，并保留可复现的制品身份。
3. **暂缓扩张。** 新增模型家族、额外 backend 和非性能相关能力让位于上述两项；只有性能工作确实依赖时才提前接入。

顺序上先建立可归因的 GPU 空泡/启动测量，再做执行重叠；AOT 的制品整理可并行准备，但不能跳过同一套性能与质量门。AOT 的优先级低于执行重叠，因为它主要影响启动和部署，而执行重叠影响稳态吞吐与尾延迟。

## 非目标

- 不改变 target 数值制度、接受判定或质量门槛来换性能。
- 不用无界队列、后台线程包装阻塞调用或增加 stream 数量冒充异步。
- 不照搬 vLLM 的 Python 进程模型；只采用其 future + batch queue + 占位回填的契约。
- 不在本决策内重写 runtime/SPI 或推进 DFlash2、组合推测等独立路线。

## 实施状态

截至 main `6e0859c`（PR #47，包含 PR #57），CUDA compute 已默认采用非阻塞
submit/poll，覆盖 prefill、decode、私有程序及 MTP draft/verify/rollback。设备完成通知、
固定 readback 和资源释放随飞行 owner 管理，CPU 返回不代表设备完成。

这完成了 P0 的 provider 非阻塞部分。runtime 仍只有一个 compute flight，深度 2 的 batch queue、
scheduler 占位 token/KV 延迟回填和额外在途状态预算仍待实现；不能把单 flight 非阻塞等同于
完整的双批执行重叠。现有 cuTile 路径仍依赖运行时 JIT，完整 AOT 覆盖也尚未交付。

本阶段的质量与设备验收见[默认服务收尾记录](../archive/research/artifacts/default-serving-closeout-2026-10-25.json)，
当前能力边界见[backend 架构](../architecture/backends.md)。具体负载的复测不替代本 ADR
要求的逐场景性能与资源验收。

## 后果

- 异步执行需要同时修改 executor、scheduler、KV/状态提交与取消/错误路径；风险高于单算子优化，必须先有正确性对照和回退路径。
- batch queue 会占用额外的在途状态和资源预算，准入必须计入第二批，不能只看单批容量。
- AOT 增加按架构构建、制品校验和 CI 成本；未覆盖的 shape 仍需 JIT，部署文档必须说明边界。
- 模型/后端覆盖面短期不扩张；已排期的正确性、测量和性能基线工作继续作为前置。

## 验收

- 每个候选使用已签发的 baseline profile 和配对测量，分别报告 GPU 空泡、TTFT、TPOT、吞吐与峰值显存；不接受单次运行或不同配置的倍率。
- 执行重叠必须通过 token、状态、KV、取消、错误毒化和多轮输出对照；质量门不因异步而放宽。
- AOT 覆盖必须报告冷启动/JIT 时间、制品哈希、目标架构和 JIT fallback 命中；未覆盖路径不能静默降级。
- 性能收益不足以抵消复杂度时，保留同步路径作为默认或回退，并记录否决证据。

## 参考

- [vLLM EngineCore](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/engine/core.py)：batch queue 与 `step` / `step_with_batch_queue`
- [vLLM SchedulerConfig](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/scheduler.py)：`async_scheduling` 默认与并发批数
- [vLLM Executor](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/executor/abstract.py)：`execute_model(non_block=True)` future
- [vLLM AsyncScheduler](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/core/sched/async_scheduler.py)：占位 token 与延迟回填
- [Backend 能力边界](../architecture/backends.md)
- [CUDA 性能方案](../plans/performance/cuda.md)
- [性能基线与 vLLM 对齐方案](../plans/performance/baseline.md)
