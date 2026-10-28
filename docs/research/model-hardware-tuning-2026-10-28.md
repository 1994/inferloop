# 模型与硬件联合调优研究

审查基点：`7335864ca16c53a511a8c4781da0b0cb7bd1858b`（2026-10-28，America/Los_Angeles），在 45 的隔离检出进行。本文是代码和模型配置审查，不包含本轮 GPU 正确性或性能测量，也不证明其他模型已可运行。

## 结论

需要联合考虑模型图、实际请求形状和设备，但不需要按模型名称再建立一套独立调参系统。复用现有 CUDA typed tuning：先由 provider/图声明语义、精度和形状，backend 根据设备能力、资源报价和 kernel 合同筛合法候选，再对无法可靠推导的性能选择实测；scheduler 继续负责请求策略。相同合同的实际形状可以跨模型复用性能证据，前提是该 chooser 的身份和数值验证足够完整。不同权重的数值资格不能仅凭形状继承。

固定值分三类：格式、ABI 和数值合同先维持约束；容量由实际图及设备报价推导；性能阈值或 tile 是搜索候选或保守回退，是否改变必须测量。公式能保证合法或估算容量，不自动证明速度最优。新模型没有候选资格时，要走已有合法路径或明确报不支持；不能把未经验证的专用 kernel 改成接受任意尺寸。

## 主线现状

[现有调优规则](../guides/cuda-performance.md#调优规则)已包含形状与设备。驻留 NVFP4/FP8 GEMM 的 key 包括设备、驱动、kernel 身份、M/N/K、布局和数值合同；每份实际权重仍须通过完整有限值及逐位验收。attention、recurrent、norm 等也有自己的选择和预算。图捕获前完成选择，避免逐 token 搜索。[#62](https://github.com/1994/inferloop/issues/62) 的九项工作已由 #78–#85 合入：包括搜索预算、投影识别、SwiGLU、pooled dense、并发和 KV 准入。因而旧草案 #66/#68/#70 不能代表当前主线；各项合入证据也不意味着所有设备和模型已经验收。

以下差异仍值得处理，按风险及可验证性排序：

| 项 | 代码事实与风险 | 建议 |
|---|---|---|
| 旧 projection GEMV 缓存 | [`select_tiling`](../../crates/backend/cuda/src/tuning.rs) 先按设备 identity、dtype、rows、columns 命中本地或 shipped 表即返回；[`DeviceProfile::identity`](../../crates/backend/cuda/src/device.rs) 是名称、架构、显存大小，条目不含 driver、编译器或 kernel fingerprint，也没有这一路径的当前权重验收。此路径仍由 [`loading/bindings.rs`](../../crates/backend/cuda/src/loading/bindings.rs) 使用。存在旧代码/驱动后继续沿用性能选择的风险；代码审查本身不证明结果错误。 | 先写缓存版本/身份变更和迁移回归，明确各 dtype 的数值合同及实际权重验收成本；为此 chooser 设计失效和保守回退，再单独测冷启动及服务。不要假定所有 typed section 已有同一身份合同。 |
| BF16 dense 委派 | [`tuning/prefill.rs`](../../crates/backend/cuda/src/tuning/prefill.rs) 的 `8 * 1024 * 1024` 阈值只用 N×K，M 只要求非零，不随设备或 workload 调整。 | 先采集真实图中投影及 M 的频率，比较原路径和 vendor 路径在小/大 M、实际权重、冷/暖条件下的完整成本；有稳定交叉点后纳入现有 chooser。不能仅根据 FLOP 或模型名替换阈值。 |
| prompt 宽度 | [`loading/mod.rs`](../../crates/backend/cuda/src/loading/mod.rs) 按实际预算、图与若干宽度档选择；这保证资源约束，但不是所有模型/负载的实测最优。 | 按实际图分别报价候选的 workspace、保留图和 KV；在短、长、混合及并发负载比较 TTFT、TPOT、吞吐、尾延迟、显存与启动。跨请求调整须保持现有空闲边界与资源生命周期。 |
| attention 及 split | [`tuning/attention/decode.rs`](../../crates/backend/cuda/src/tuning/attention/decode.rs) 已使用 SM、头数及槽数，但短窗口分支还使用 1024 阈值；pooled 分组/分片仍有固定边界。 | 测上下文、头数、槽数及 workspace 的交互；先独立参考、尾块、mask 和状态验证，再将有证据的合法候选接入现有 tuning。 |
| 固定 tile 与 split-K | [`constants.rs`](../../crates/backend/cuda/src/constants.rs) 的 prompt 行 tile 64/128 有部分作为 GEMM chooser 候选或回退；`PREFILL_SPLIT_K=4` 有具体调用范围，不能假设所有 GEMM 都使用。现有 [split-K 筛查](../guides/tuning-new-hardware-and-models.md#已测-split-k-实验的边界)显示不同投影方向不一致。 | 按调用路径、实际 shape 和数值合同评估；复用已有搜索，避免给所有常量机械增加搜索维度。数值顺序改变须按质量合同验收。 |
| 专用 attention 和状态 | [`attention_mma/catalog.rs`](../../crates/backend/cuda/src/resident/attention_mma/catalog.rs) 的 FA4 形状守卫包含 24/4/256；不匹配时 [`capture_attention.rs`](../../crates/backend/cuda/src/resident/capture_attention.rs) 可走 legacy 路径，但这不是对任意模型的设备资格。Conv 状态在 [`program.rs`](../../crates/backend/cuda/src/resident/program.rs) 限定 kernel 4，其他形状返回不支持。 | 明确区分性能回退与功能缺失。扩展形状需新制品/通用实现及独立数值、状态和设备测试；不可只放宽守卫。KV block 32、NVFP4 group 16、FP8 scale block 128 同时关联布局/编码合同，不能单点改成自适应值。 |

## 45 上模型形状调查

2026-10-28 在 45 只读取本机模型 `config.json`，设备查询为 RTX 5090、32607 MiB、driver 615.78.08。下面的值是配置声明，不是本轮模型加载或设备执行结果；没有读取或提交权重。不同 dtype、模型图及权重也会改变实际候选。

| 本机模型 | 文本 hidden / layers | query / KV heads / head dim | intermediate | vocab | 涵义 |
|---|---:|---:|---:|---:|---|
| Qwen3.8-27B-NVFP4 | 5120 / 64 | 24 / 4 / 256 | 17408 | 248320 | 匹配上述 FA4 头部形状；还需满足架构、lane、窗口及精度守卫。 |
| Qwen3-VL-2B | 2048 / 28 | 16 / 8 / 128 | 6144 | 151936 | 不匹配该 FA4 形状，不能以 27B 的专用路径性能推断其表现。 |
| Qwen3-0.6B | 1024 / 28 | 16 / 8 / 128 | 3072 | 151936 | 同样不匹配；是否由当前 provider 和 CUDA 路径完整支持须另行验证。 |

现有[基线契约](../plans/performance/baseline.md)已列 27B 与 VL-2B；文件存在不等于两模型本轮已通过，0.6B 也不是自动新增的发布资格。上表说明测试形状差异真实存在，不能证明固定阈值导致了性能回归。

## 建议验证顺序

1. 在固定 main 和候选源码、构建与缓存身份下，先记录加载日志、图中唯一 dtype/layout/M/N/K、attention 几何、各 chooser 命中/未命中及 fallback、实际并发和 KV 报价；核对模型权重及 provider 确实可加载。以独立缓存测冷启动和复用，记录预算耗尽与准备时间，不把缺失或回退算作已调优。
2. 先处理旧 GEMV 缓存合同，再以 dense 委派阈值为第一个有明确对照的性能研究：使用 27B 与基线契约的 VL-2B，覆盖实际可执行的短/长 prompt 和 decode/verify；0.6B 先作为资格探针，仅在 provider 和设备验证通过后纳入。按模型分别报告，不能只统计合并平均值。
3. 做独立数值参考、完整输出、尾部和实际权重检查；测同机 release 的冷/暖配对与反向确认，记录启动、workspace、显存和逐形状结果。未测到的形状、无效证据、超时及缓存不匹配均保留合法基线。
4. 对可能影响服务的选择，按[性能基线](../plans/performance/baseline.md)与仓库迭代/发布门分别验证 TTFT、TPOT、吞吐、尾延迟、资源容量、取消回收和完整质量；质量/数值顺序变化遵守对应门禁。性能失败或证据不足不启用默认路径，不凭算子收益宣布服务提速。

本研究未改变运行时代码、默认值或依赖，因此没有执行 GPU 性能和正确性测试；这里列的是后续实验计划，不是已完成的性能结论。
