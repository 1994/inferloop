# CUDA 推测解码与投影性能优化方案

状态：设计方案。基于 2026 年 10 月 9 日的 `7c79d99`。

整体排期与任务状态见 [路线图](../README.md) 的 B1、C1；状态重放与 DFlash2 分别由 R1、D2 交接。本文维护本项设计与验收。

2026 年 10 月 17 日的[长上下文性能交接](../../research/cuda-vllm-performance-handoff-2026-10-17.md)补充当前代码与实测边界：PR #10 已合并，后续 vector 分支通过 native 五组性能门；最新 32K profile 的 attention 占 71.8%，GDN 占 3.8%。当前执行顺序与独立验收按路线图 C1Q、C1A、C1M、C1V、C1H、C1D；下文 recurrent 回滚和推测组合继续作为各自前置满足后的设计储备。具体追平阶段、候选与验收门见 [vLLM 长上下文性能追平实施方案](vllm-parity.md)。

CUDA 后续工作分为两条线：一条修正测量与资源管理，降低推测解码的状态和传输成本；另一条继续按真实形状优化投影与 prompt 图。先维持现有 MTP 的正确性和并发性能，再让 DFlash2 通过显存准入，最后比较组合算法与最佳单算法。

KV cache 的分页重构（PagedAttention 的分层与落地）单独立项，见 [KV cache 分页重构方案](kv-cache-paged-attention.md)。

共同接口见 [SPI 方案](../speculation/spi.md)，模型接入见 [DFlash2 方案](../speculation/dflash2.md)，叠加语义见 [组合方案](../speculation/composition.md)。已有性能实验继续由 [性能实验记录](../../archive/research/cuda-performance-experiments.md) 维护，本方案定义后续工作和验收。

[Infernix 调研](../../research/infernix-performance.md) 给出了已接线的 ReplaySSM、草稿量化、shortlist head 和 prefix cache 参考。采用其中的执行机制时保持本项目 target 权重与数值身份；target 重新量化另做制品和质量验收。

## 先建立可靠基线

先落实 [近期提交审查](../../reviews/cuda-commits-2026-10-09.md) 的三项修正：cuBLAS handle 与实际 device/context/stream owner 绑定；FP8 workspace key 包含量化模式和 scale layout；vendor GEMM 在 capture 前完成图外预热。优先级、触发条件、源码证据与验证要求由审查记录维护。

这些修正作为扩大优化范围前的基线条件。多实例执行、混合 FP8 模式和新进程基准通过后，再扩大投影及图捕获范围。

补齐 CUDA 的 `BackendProvider::completion_timing`，让 runtime 已有的成本观察能使用真实设备时间。分别记录 CPU encoding、CPU wall、GPU command 时间；草稿、验证、接受判定、状态恢复、追赶和传输使用对应 stream 的事件测量。观测路径有预算，不能为每个算子添加主机同步。

报告还必须记录实际 prefill width、verify width、slot 数、量化制度、池化成功率和回退原因。调度成本模型使用完整固定图开销与实际接受长度，不能只按候选 token 数估算周期成本。

[vLLM 对比测试审查](../../reviews/vllm-benchmark-methodology-2026-10-09.md) 记录现有方法缺口，修复与共同验收由 [性能基线方案](baseline.md) 维护。历史服务矩阵不作为当前已验收 baseline。

## 性能实验前置条件

执行 [性能基线方案](baseline.md)，使用已验收的 baseline ID 后再开展新的性能优化实验。release 构建、当前 native/vLLM 配置对齐、比较身份、工作量、计时/统计、矩阵和证据留存均由该方案维护；本方案只定义 CUDA 的执行机制、资源需求与专属验收。

## 先降低 recurrent 回滚的显存成本

分层与公共契约见 [Recurrent 状态重放方案](../speculation/state-replay.md)。CUDA 的 Record/Fold 放在 backend 物理状态执行与 recurrent 算子层，由 serial/pooled verifier 共享；MTP 和 DFlash2 不分别实现 target 回滚。

当前 [slot verification](../../../crates/backend/cuda/src/resident/slot_verify.rs) 为每个 slot 的中间 lane 保存完整 Conv/LinearAttention 状态。仓库对 27B 的记录给出约 151 MB/序列的 F32 Delta 状态主项，快照数量为 `slots × (verify_width − 1)`。

| 配置 | Delta 快照主项估算 |
|---|---:|
| 4 slots，verify 3 | 约 1.21 GB |
| 4 slots，verify 4 | 约 1.81 GB |
| 4 slots，verify 8 | 约 4.23 GB |

现有记录中 BF16 DFlash2 草稿权重约 3.85 GB；与 verify 8 的快照合计约 8.08 GB，即 7.53 GiB。对比该记录约 4.8 GiB 的剩余预算，两项合计已经超出，还未计入额外 slot 状态、feature、arena 和 graph。这里的 MB/GB 为十进制，GiB 为二进制；原始模型与设备记录见 [性能实验记录](../../archive/research/cuda-performance-experiments.md)。

### 基点加接受前缀重放

以 committed recurrent state 作为本轮只读基点，verification 不覆盖它，只写重建所需的 Conv 输入及 GDN 原始转移记录。接受结果出来后，从基点只重放 seed 与接受前缀的 recurrent 更新。复用 verification 已计算的中间输入，避免重新执行整个 target 的投影与 attention。

优先研究 Infernix 的 raw-input Record/Fold：保存实际 cast 边界上的 raw key/value、gate/beta 原始位和 conv 投影输入；Record 与 Fold 使用同一有限精度状态转移。不要只保存另一种代数公式的摘要，也不要在回滚时按接受宽度重新投影，否则量化与舍入可能变化。

主项从每 slot 的 `verify_width − 1` 份 Delta 中间快照改为一份 committed 基点加短记录。4 slots 的 Delta 基点约 0.60 GB；若直接借用现有 committed state，它不构成额外分配，需复制或 COW 时则另计。按本项目现有 F32 输入、27B 几何估算，4 slots、8 列的 raw GDN/conv records 约 108.6 MiB，即 0.114 GB；仅比较回滚主项，预计减少约 3.5–4.1 GB，仍需计入其余状态和执行资源。推导、未计入项目及速度边界见 [Infernix 调研](../../research/infernix-performance.md)。参考实现的 BF16 key/value 与 conv records 约 54.6 MiB，不能将这个数直接用作本项目报价。

保持原有逐 token 的归一化、归约、运算和 store 顺序。除独立数值 oracle 外，直接比较 Fold 与同一物理 verification block 的相应 recurrent/conv 状态前缀，覆盖零提交、首 token 拒绝、部分接受、全接受、EOS、拒绝尾部改写和多轮输出。只读 Record 方案全接受也需要 Fold；若增加保留完整结果的快路径，单独定义 destination 与提交协议并测量收益。KV 的有效长度、页引用与 feature 范围同步截断。

### 稀疏快照作为备选

每隔若干 lane 保存一个完整 recurrent checkpoint，恢复到最近检查点后仅重放余下接受输入。它在显存和回滚时间之间提供折中，应与基点重放、现有逐 lane 快照按相同接受长度分布比较。

选择依据是完整周期时间、峰值显存和池化成功率。基点重放先作为候选实现，测量通过后才成为默认；状态降精度或缩小 Delta 表示会改变数值制度，另行立项。

## 让候选与特征留在设备端

当前草稿与验证路径将 logits、hidden 读回主机再进行接受判断。以 4 slots、每 slot 3 条 verify lane、词表 248320 为例，仅 F32 logits 就约 11.4 MiB/轮。这是传输规模估算，不等于已确认同等规模的延迟收益。

第一步在设备端实现与现有 sampler 一致的 greedy 判定和接受前缀 scan；仅回传最终 token、接受长度、停止信息与必要 metadata。target hidden 和 DFlash2 taps 保持在设备 buffer 内，供 draft 追赶与下一轮使用。

第二步融合可兼容的接受判断、cursor 更新和状态恢复，减少阶段间同步。是否能融合到同一 graph，取决于有效长度与状态协议，不以接口拆分要求额外的 CPU 往返。

已有记录指出 host readback 并非当时总延迟的主要瓶颈，因此先用阶段计时确认占比。设备驻留同时服务于 DFlash2 的特征生命周期和显存管理；性能报告单独给出传输、CPU sampling 与 GPU 判定的变化。

## 完整预算与草稿量化

准入报价同时纳入 target 权重、draft 权重、target/draft KV、recurrent state、回滚策略、feature 历史、prompt/verify arena、graph 与 workspace，以及设备保留余量。预算随实际并发和宽度变化，不能只检查模型权重大小。

加入 DFlash2 会挤占原先可用于宽 prompt 图的内存，因此 prefill width、verify width 与 slot 数需要联合选择。输出实际选择及原因；OOM 后回收或重试必须区分 driver 可用内存和 memory pool 可回收储备，不能把静默关闭池化当成正常结果。

草稿先用 BF16 建立参考，再评估 backend 已支持且经验证的 FP8/NVFP4 存储和计算。草稿量化可能降低接受长度；最终比较包含反量化、量化 workspace、selector 和额外验证成本。target 的精度制度保持可比。

新增独立 draft head 的 shortlist 实验：按代表性训练语料频率选择 token 行并保留特殊 token，绑定映射与精度身份，target verification 继续使用完整 head。将完整 head、仅 shortlist、仅低精度和两者组合分别对照，覆盖罕见标识符、多语言和工具调用；随机模式导出实际 `q`。Q8/Q4 是参考实现的存储选择，本项目未支持的 codec/kernel 需另报价，不能借 dtype 名称假定可直接接入。

## 投影与 prompt 图继续按形状选择

| 方向 | 下一步工作 | 判断依据 |
|---|---|---|
| NVFP4 vendor GEMM | 将所需权重 scale 重排放在加载期，适配 activation scale layout，做形状级选择 | 把在线量化、layout、workspace 和 capture 成本纳入比较 |
| BF16 投影 | 修复 context/stream 归属后，继续比较宽 prompt 形状 | F32 activation 转 BF16 会改变数值，独立检查误差与 token 差异 |
| FP8 投影 | 在量化制度和 workspace 正确后加入 vendor 对照 | 分 channel/block 制度、dtype 和实际形状报告 |
| 模型调用形态 | 比较 GDN/conv、gate-up/SwiGLU、head/top-k 的组合路径 | cuTile、vendor 或原生算子对照均包含 cast、quantization、layout 和 epilogue |
| Prompt 图宽度 | 保留窄图与宽图阶梯，加入 draft 后重新报价 | 测量尾块利用率、arena、TTFT 与并发 TPOT |
| Chunked recurrent | 保持量化路径的独立实验开关 | 先解决数值差异与 short/hot_long 回退，再讨论默认启用 |

算子比较同时覆盖冷 L2 与真实 serving 的缓存状态。只在某个真实形状和精度制度下获胜的路径，进入相应选择表；不把宽 prompt 的结果推广到单 token decode，也不把 microbenchmark 的加速倍数当成服务加速。

同序列的草稿和 target 存在数据依赖。跨 stream 重叠只研究具有明确独立性的工作，并通过 owner、event 和 buffer 生命周期表达依赖；不把增加 stream 数量作为默认提速手段。

PDL 作为 backend 的独立候选能力：允许后继准备工作重叠，读取依赖前执行明确等待；流式 kernel 的触发点按带宽争用与完整周期选择。先确认工具链和 backend 能力，再进行无 PDL、分阶段 PDL 与融合的对照，不能只按 launch 数量决定采用。

## 冷 prefill 与 prefix 复用分别优化

Agent TTFT 同时取决于实际重算 token 数、prefill 计算与缓存恢复。增加内容寻址 KV blocks 与稀疏 recurrent checkpoints 的复用研究，并同时管理 draft features；分别报告缓存命中、实际 prefill token、host restore 成本及无命中吞吐。参考机制与测量边界见 [Infernix 调研](../../research/infernix-performance.md)。

扩大 prompt chunk 时，对 attention wave、key splitting、partial/merge workspace 和 recurrent prefill 一起规划；精确 checkpoint 边界与短尾块单独处理。1024/4096 级别作为新的实验范围，不直接替代现有图阶梯。服务验收包含 mixed prefill/decode 的 P95 ITL，不能只用 long TTFT 决定宽度。

## 验收矩阵

正确性先比较同一 target 权重与精度下的普通 decode、现有 MTP、迁移后 MTP 和 DFlash2。greedy 检查最终 token、拒绝后多步输出、游标、KV/recurrent state 与特征覆盖。随机模式另做有效分布和统计验收。

共同服务矩阵、release、配置对齐、配对统计与证据留存按 [性能基线方案](baseline.md) 执行。CUDA 额外扫描 prompt 边界 63、64、65、127、128、129、255、256、257 token；验证宽度覆盖实际 MTP 深度、DFlash2 原生 block 和短前缀验证。专属报告记录峰值显存、实际新发布 token/周期、完整周期各阶段时间、图宽度与池化回退。

跨模块排期与启动条件由 [路线图](../README.md) 维护，性能目标与判定由基线方案维护。每项实验只改变一个可归因的策略，先证明相对同配置基线的稳定改善，再判断与 vLLM 的差距。
