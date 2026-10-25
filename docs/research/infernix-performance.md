# Infernix 性能实现调研

状态：调研与方案，未修改本项目实现。调研日期为 2026 年 10 月 9 日，参考本地 `/Users/r/temp/infernix` 的 `fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de`；本项目对照基点为 `7c79d99`。

本文区分源码确认的机制、参考仓库公布的测量，以及对本项目的迁移建议。没有在本机重新运行 CUDA 或模型 benchmark，参考数据不代表本项目已经取得相同收益。

## 结论

Infernix 的速度来自多项工作的组合：针对 RTX 5090 的量化和算子分派、低成本推测解码、紧凑的 recurrent 回滚记录、大块 prefill，以及适合多轮 agent 工作负载的 hybrid prefix cache。不同指标的收益来源不同，不能用一个 kernel 或模型量化解释全部结果。

它确实改变了部分模型的权重表示、activation 精度和草稿组件。检查到的主流转换配方没有通过删 target 层或缩小 target hidden size 提速：27B 保留原架构，使用混合 NVFP4/FP8；Flash-Next 的 Dense8 版本重新量化部分 BF16 dense 权重。另加的低精度 shortlist head 只服务草稿，target 验证保留完整词表。

对本项目最直接的投资是 ReplaySSM 与草稿量化，它们针对 DFlash2 的显存阻塞；随后是 GPU 接受判定、draft shortlist head 和实际模型形状的融合算子。Agent TTFT 还需要单独优化 prefix 复用。MTP 与 DFlash2 同轮叠加并不是这份实现已经证明有效的路径。

## 先明确比较对象

参考仓库提供了三类不同的结果，须分别理解：

| 比较 | 已公布结果 | 可以支持的判断与限制 |
|---|---|---|
| 27B，Infernix 对 NInfer，使用同一官方 `.ninfer` artifact，DFlash2，agent replay | 平均 TTFT 5.7→2.7 s；缓存比例 83.7%→90.4%；实际 prefill 982K→569K token；无命中 prefill 6127→8254 tok/s；单请求 decode 199→235 tok/s | 权重相同，支持缓存和执行改进。Infernix 还启用了 n-gram，不能将差异全部归因于 GPU kernel |
| Flash-Next，Infernix Dense8 对 Strata 的 Unsloth GGUF | 约 8K/128K/250K context 下 TTFT 1.9/17.9/34.2 s，对照 12.3/103.3/158.1 s；decode 约 139/136/153 tok/s，对照 86/77/96 | 同机器的产品配置对比，权重量化、draft 和 offload 路径不同，不能作为同权重引擎 A/B |
| 同一 Infernix，Flash-Next Dense8 对保留 NVIDIA target 权重的版本 | 保留权重版单请求 decode 约慢 18%；Dense8 每 token dense 权重读取约 5.1 GB，对照 8.6 GB | 直接支持 dense 权重量化降低读取量并释放 expert-cache 空间；这部分收益伴随 target 数值变化 |

27B agent replay 的平均 wall time 为 16.9→12.5 min，但三个 seed 的生成长度不同：一个 seed 的输出多 31%、wall 慢 4%，另外两个输出少 24–30%、wall 快 37–41%。因此不能把平均 wall 改善 25% 当作固定输出长度的速度改善。以上来自参考仓库 [README](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/README.md)。

其 9 月底的 27B serving 表又是另一批实验：prefix reuse 关闭、prefill chunk 1024、FP8 KV、不同配置和提交。表中的单请求 phase、完整 corpus 与 steady decode 使用不同时间分母；内部 server TTFT 不含 ingress 排队和 HTTP 传输。不能把该表、10 月 agent replay 与本项目短 prompt 比值直接拼接。[Serving 方法](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/docs/performance/methodology.md)

## 对模型做了哪些改变

### Target 权重按投影角色选择表示

27B 的配方不是统一把所有权重降到 4 位：

| 配方 | Target 的主要处理 |
|---|---|
| `qwen3_8_27b_nvfp4` | 从量化源直接导入 codes/scales；0–55 层 MLP 使用 NVFP4，56–63 层 MLP、attention/GDN 投影及完整 output head 使用 row-scaled FP8；embedding 从 BF16 转 FP8 |
| `qwen3_8_27b_nvfp4_nvidia` | 导入 NVIDIA 的 NVFP4 MLP、FP8 attention/GDN；embedding 转 FP8；源 NVFP4 output head 由于当前 runtime 的 vocabulary projection 支持限制，先解码再转 FP8 |
| `qwen3_8_flash_next_nvfp4` | Target 与 vision 保留 NVIDIA 存储的权重值；NVFP4 experts 做物理重排，FP8 n-gram table 写入独立 volume；MTP 草稿仍会重新量化 |
| `qwen3_8_flash_next_nvfp4_dense8` | 在上一配方上，将 GDN 大投影、attention output、shared experts、hyper-connection mixers、PLE 投影和 `lm_head` 转为 `q8_g32_fp16`；router、部分 attention 输入和控制权重保留原表示 |

这些选择由离线 converter 决定，runtime 绑定实际存储格式和各输入的 activation policy。`AllowA4`、`AllowA8` 是允许使用低精度 activation 的条件，实际路径还随形状选择。储存格式、计算精度和模型架构是三个维度，不能由 `.ninfer` 容器或“NVFP4 模型”名称推断。[转换源码](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/tools/convert/official_recipes.py)

README 还列出 Quasar QAT 和 Uncensored 的权重来源。它们是不同的模型制品选择，不能将其效果归为同一 checkpoint 上的执行优化。本项目若做转换，应保存来源、每个张量的处理和精度身份，分别建立保留权重与重新量化的基线。

### 草稿更小，并增加独立 shortlist head

转换器将 MTP/DFlash 系列的大部分草稿投影转为 Q8，同时保留 router、动态卷积控制和 selector 等指定参数。NVIDIA 27B 配方进一步将 DFlash2 的 MLP gate/up 转为 NVFP4，并约束为 A16 activation 路径。这是草稿资源和接受长度的取舍，不能照搬成 target 的量化规则。

`tools/convert/proposal.py` 从完整 head 中抽出 131072 个常用 token 的行，包含特殊 token，另存为 `q4_g64_fp16` 和 token-ID 映射。对 27B 的 248320 词表，这减少了草稿 head 的行数；同时低位宽降低权重读取。原完整 target head 保留，不是永久删除罕见 token。

DFlash2 的候选 unary top-16 可从该 head 计算，并将 shortlist ID 映射回完整词表，再进入 selector。MTP/DFlash 的 deterministic proposal 和 DFlash2 的 sparse conditional proposal 具有不同的实际 `q`；随机接受必须使用各自实际提议分布。[Proposal 源码](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/tools/convert/proposal.py)、[DFlash 数学与状态](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/docs/maintainer/dflash.md)

草稿量化与 shortlist 可以改变接受率和生成成本。在接受算法正确的条件下，输出分布由本轮计算出的 target 分布决定；这不保证它与 BF16 模型、另一 verification 宽度或逐 token 路径逐位相同。

### Dense8 的质量证据有范围

Flash-Next Dense8 卡片给出 2557 个 teacher-forced 位置上的 PPL 4.654，保留权重版本为 4.666，两者在该样本的误差范围内接近。它支持继续研究该转换，不能证明所有任务无损。27B 对 NInfer 的约 104 万 token PPL 对比是另一套、同 artifact 的引擎对照，也不能替 Dense8 量化做验收。[Dense8 模型卡](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/model-cards/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix/README.md)

## 推测解码为什么成本低

### ReplaySSM 替代逐候选完整状态快照

本项目的通用性与层次设计见 [Recurrent 状态重放方案](../plans/speculation/state-replay.md)：输入记录思想可跨 recurrent 算子使用，具体格式和转移由 backend 实现，公共层统一接受前缀提交协议。

Target verification 从 committed recurrent checkpoint 开始，计算候选输出，同时记录驱动状态转移的原始 key/value、gate/beta 和 causal conv 输入。verification 不修改 committed checkpoint。接受长度确定后，只 Fold seed 与接受前缀的原始记录，得到下一轮状态；拒绝尾部不进入 Fold。

缓存输入再重建状态的思想也见 [ReplaySSM 原作者说明](https://tridao.me/blog/2026/replayssm/)。其中还讨论 output-only decode 和延迟 flush；本节聚焦 Infernix 的短窗口 Record/Fold，不将文章在 B300 上的结果转用为 RTX 5090 或本项目的收益。

这是已接线的实现：`target_verify_forward` 设置 `RecordForReplay`；`recurrent.cuh` 中 record 和 fold 复用有限精度状态转移；`GdnReplayFoldPlan` 绑定所有层的记录、source/destination slots 与 commit columns。它不是用一个数学上等价的压缩公式近似恢复状态。[验证源码](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/src/models/qwen3_5/program/speculative/target_verification.cpp)、[递推源码](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/src/ops/linear_attention/gated_delta_net/recurrent.cuh)

参考实现对 27B 给出的容量为：

| 项目 | 每序列大小 |
|---|---:|
| 完整 FP32 recurrent image | 144 MiB |
| BF16 conv history | 2.8125 MiB |
| 每个候选位置的 raw GDN record | 0.767578 MiB |
| 每个位置的 conv record | 0.9375 MiB |
| 每个位置的完整 raw record | 1.705078 MiB |

因此 4 lanes、8 列的 raw records 约为 `4 × 8 × 1.705078 = 54.6 MiB`，另需现有 committed 状态、可能的 destination 状态及其余执行资源。约 86 倍是“每位置 snapshot 与 raw record”的容量比，不能写成整个引擎节省 86 倍或提速 86 倍。[ReplaySSM 说明](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/docs/maintainer/replayssm-gdn.md)

我们当前的 verify 8、4 slots 完整 Delta 中间快照估算约 4.23 GB。两种结构的差距足以将 ReplaySSM 放在 DFlash2 前置工作中。迁移时保留本项目实际的 cast 边界，不能为得到上述容量先将 F32 中间输入擅自转 BF16；记录格式和总资源报价应按本项目源码重新计算。

验收同时要求独立 FP32/FP64 oracle 的数学正确性，以及 Fold 与同一物理 verify block 的相应状态前缀一致。直接比较 recurrent state 和 conv history，覆盖零提交、部分接受、全接受、拒绝尾部改写和长链多轮。只比较最终文本无法证明恢复正确。

### 接受判断与大部分数据留在 GPU

Target acceptance 在 GPU 上完成 greedy 或 sparse rejection；hidden 选择、feature 和 tree compaction 同样使用设备操作。Host 读取较小的 egress，包括接受数量和发布 token，避免每轮将完整 vocabulary logits 和 hidden 传回主机。

它仍存在 host 工作：选择 graph family、准备 ingress、管理发布与资源，约束生成还读取少量 draft token 来构造 masks。普通或推测 decode 使用按 batch/宽度捕获的 graph family，推测路径包含 Forward/Finish 阶段；不能把该实现描述成整轮没有 CPU 交互。对本项目的目标是去除大数据 readback 和不必要同步，并测量阶段成本。

### DFlash2 树形验证与 copy proposal

源码中一个 Engine 选择一个 `SpeculativeBackend`，MTP、DFlash、DFlash2 互斥。它已有的组合是神经草稿加 n-gram copy：有匹配的 row 使用 copy proposal，无匹配 row 保留神经 proposal，并选择够用的验证宽度。Copy 在随机模式下是 one-hot 提议，DFlash2 sparse frame 也需要覆盖为相应的候选与 `q`。

DFlash2 可以单独从 candidate lattice 构造树，由 target 的 ancestor masks 验证，并把接受路径的 KV、ReplaySSM records、hidden 和 features 压到连续前缀。MTP 和 n-gram round 保持链式验证。检查到的入口没有 MTP 与 DFlash2 同轮联合提议。[Backend 配置](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/src/models/qwen3_5/program/planning/startup.cpp)、[Copy proposal](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/docs/ngram.md)

树形宽度不是越大越快。其四个 seed 的短 context 测量中，C=1、16 列较 8 列链式约快 8.6%，C=2、12 列约快 3.0%，C=4、9 列约慢 0.9%。长 context 成本测量与接受收益估算也显示固定宽树可能转为回退；后续 kernel 调整又改变了最佳宽度。参考实现用 batch/context/width 下的实际时间与新发布 token 数选择树宽。[树形验证与测量](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/docs/maintainer/tree-verification.md)

本项目先完成线性 verifier 和独立 DFlash2。Copy provider 可以作为成本较低的组合实验，但不替代 MTP 与 DFlash2 组合契约；树形验证另需 target 的分支状态和 compaction 能力。

## Prefill 与 CUDA 算子做了什么

### 按完整模型调用形态优化

实现直接使用 CUDA C++ 和 Tensor Core 指令，针对有限模型形状选择路径。例如 27B 的 NVFP4 gate/up 加 SwiGLU：单 token 和 2–4 token 使用 A16 fused 路径，允许 A4 时中等宽度使用 fused A4，256 列起使用 TMA 与 tiled scale 布局。QKV/GDN 输入、conv、residual 和 linear top-k 也有专门的组合实现。

这说明 kernel 对照应覆盖模型调用需要的 cast、activation quantization、双投影和 epilogue。只比较裸 GEMM，可能漏掉实际读取或写出中间张量的成本。它的这些路径随 policy、形状分派，不能从一个阈值推导所有模型都采用同一精度。[NVFP4 SwiGLU 分派](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp)

本项目已有 cuTile 的行数特化；下一步应对真实调用形态比较 cuTile、vendor 和有限 CUDA 原生算子。仅凭语言差异不能判断谁更快，迁移也不要求重写整个 backend。

### 大块 prefill 与 attention 成本模型

参考配置支持 1024/4096 token 的 prefill chunk，并在对应 attention 路径上将完整 chunk 向下对齐到 SM wave。长 prefix 和短尾块另有成本；attention 的 key splitting 计入 partial 写出、merge 与 workspace，不只最小化 wave 数。split workspace 和 KV 共用显存预算。

其 INT8、FP8 和 NVFP4 KV 的 prompt attention 使用不同数学和存储路径，包含 Tensor Core QK、按 tile 解码 V 和在线 softmax。部分 P×V 还采用更低精度：当前 Auto 对 NVFP4/K8V4 允许 E4M3 P×V，INT8 保留 FP16 P×V；依据来自按 context 分桶的 KL、top-1 和 PPL 对照。不能把“INT8 prompt attention”解释成所有乘法和累加都为 INT8。

本项目的 64/128/256 固定 prompt 图与它的大块 prefill 组织不同。扩大到 4096 会涉及 recurrent prefill、attention、arena、尾块和调度占用，不是改一个宽度常量。需要同时看 TTFT、混合负载 P95 ITL 和资源容量。[Prefill 宽度源码](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/src/models/qwen3_5/program/planning/startup.cpp)

### Kernel 重叠与融合保留失败证据

它采用 PDL，让有数据依赖的 CUDA kernels 在明确同步点前后重叠部分准备工作。流式 kernel 在主循环结束后才触发后继，消费依赖数据前等待；eager 路径保持 stream 顺序。源码表达的是执行依赖，不是让未完成的输入被提前读取。[PDL 源码](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/src/core/pdl.cuh)

NVIDIA 的 PDL 契约要求消费 kernel 在读取依赖前确认结果可见，提前调度也不保证实际并发。因此本项目需要同时表达 launch、等待与 graph dependency，并确认设备和工具链支持。[CUDA 官方文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)

实验记录中的首版 entry-time trigger 反而使完整 decode round 慢约 9–10%；后改为按 kernel 类型选择触发点，约快 2.3–2.5%。另一个 RMSNorm→SwiGLU→down 的更大融合在 PDL 已启用时慢约 0.7%，未采用。这两条证据说明“更多融合”或“更早重叠”并不自动提速，带宽争用和串行路径仍决定结果。[实验与失败记录](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/RESEARCH_NOTES.md)

## Agent TTFT 的主要系统工作

Hybrid prefix cache 以内容寻址的 64-token KV blocks 加稀疏 recurrent snapshots 复用前缀。缓存分 GPU 与 pinned host 两层，host slab 同时容纳 KV 和 snapshot；精确边界才切 prefill，灵活 snapshot 尽量落在 chunk 边界。恢复使用专门 stream，并表达层级依赖；还支持持久化和在途共享前缀等待复用。

这是服务端复用策略的变化，不改变 target 层结构。27B replay 中实际 prefill 量下降 42%，因此 TTFT 收益包含少算 token；无命中 prefill 同时提高 35%，说明也有计算路径收益。两个来源分别记录，不能将缓存命中算成 kernel 吞吐。[Hybrid cache 设计与实现状态](https://github.com/Wallawalla47/Infernix/blob/fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de/docs/maintainer/hybrid-prefix-cache-spec.md)

本项目的 hybrid prefix 需要同时恢复 KV、Conv/Delta 状态和相应 draft feature 覆盖。先定义内容身份、可恢复 checkpoint 边界与取消后的发布规则，再比较缓存冷/热、host restore 和无复用路径。只缓存 attention KV 无法正确恢复这类模型。

## 预期收益与精度边界

值得先迁移的是保持本项目 target 权重与数值规则的资源、执行和复用机制。当前基线已经使用 NVFP4/FP8，不能再把参考实现相对 BF16 或另一量化制品的优势计算为本项目的新增收益。现阶段最有把握的是回滚显存容量改善；DFlash2 相对当前最佳 MTP 的速度、CUDA 融合的完整周期收益都需要新的同配置测量。

| 工作 | 对本项目的预期收益 | 精度与质量边界 |
|---|---|---|
| ReplaySSM，保留当前 F32 记录边界 | 4 slots、verify 8 的回滚主项估算可减少约 3.5–4.1 GB；缓解宽验证导致池化关闭的问题 | 不靠压低状态精度；Fold 必须复现同一物理 verify 的接受前缀状态。仅有数学等价不够 |
| 草稿量化与 shortlist | 降低 draft 权重、head 读取及草稿成本，帮助 DFlash2 通过准入；净加速取决于接受长度损失 | 改变草稿 `q`，不直接改变 target 权重；准确的 target 验证与接受校正是输出质量契约的前提 |
| GPU 接受、融合与 PDL | 减少传输、同步和中间读写；参考实现的 PDL 完整 round 改善约 2.3–2.5%，本项目待测 | 可以保持既有算术；融合若改变 cast、归约或运算顺序，必须重新验证数值 |
| Hybrid prefix 与 prefill | 对重复前缀的 agent 请求优先改善 TTFT；参考同 artifact 对照的 TTFT 下降约 52%、无命中 prefill 吞吐提高约 35% | 正确缓存复用不要求重新量化；改变 chunk 或计算路径仍可能产生浮点差异 |
| Target Dense8、低精度 activation/P×V | 参考 Flash-Next Dense8 decode 约比保留权重版快 22%；本项目模型与基线不同，收益不能转用 | 直接改变 target 数值和分布，需要独立制品、质量对照与显式选择 |

参考百分比的实验范围和来源分别见前面的比较表、PDL 实验记录及对应模型卡；它们不是本项目的性能承诺。

### 按本项目 F32 路径重新核算回滚容量

当前 [recurrent kernel](../../crates/backend/cuda/src/resident/recurrent.rs) 的 conv 输入、GDN key/value、alpha/beta 和状态是 F32；[slot checkpoints](../../crates/backend/cuda/src/resident/slot_verify.rs) 也实际分配 F32。不能因模型元数据将 conv state 标作 BF16，就用 BF16 字节数给当前 backend 报价。

按记录中的 27B 几何：48 个 GDN 层、16 个 key head、48 个 value head、维度 128、conv channels 10240。若保留 F32 raw key/value、两个每 value-head 的门控输入和 conv 投影输入，每层每位置的记录为：

```text
raw key/value： (16 + 48) × 128 × 4 = 32768 bytes
门控输入：      2 × 48 × 4          =   384 bytes
conv 输入：     10240 × 4           = 40960 bytes
48 层合计：     74112 × 48          = 3.392578 MiB / 位置
4 slots × 8 列：                       108.5625 MiB ≈ 0.114 GB
```

原 Delta 中间快照主项约 4.23 GB。若借用现有 committed state，替换为上述 records 的容量差约 4.11 GB；若需要再增加四份 Delta 基点，另计约 0.60 GB，容量差约 3.51 GB。这是结构估算，尚未分配或测量；未计 conv 基点、额外 destination、feature、KV、arena、graph、workspace、对齐及设备保留余量。最终以完整资源报价为准，不能据此保证 BF16 DFlash2 一定装得下。

容量收益与速度收益需要分别验收。Record/Fold 在 verification 后增加接受前缀的状态更新，但避免逐位置保存完整状态；具体是否更快取决于状态带宽、并行方式和接受长度。它更大的服务收益可能来自恢复池化、避免多序列退回串行，而不是单请求的 recurrent kernel 本身变快。

### 用阶段成本约束加速预期

本项目 [已有阶段记录](../archive/research/cuda-performance-experiments.md) 中，27B、batch4、MTP 深度 2 的 target verify 约 27 ms，其中 linear 19.9 ms、Delta 4.2 ms、conv 2.3 ms，另外两轮 draft 各约 1.6 ms。这是历史配置的测量，不是 DFlash2 的新基线。

即使把 Delta 的 4.2 ms 降为 2.1 ms，verify 延迟也只下降约 `2.1 / 27 = 7.8%`；计入两轮 draft 后，已知 GPU 阶段时间下降上限约 `2.1 / 30.2 = 7.0%`，还未扣除新 Fold 与其余成本。同样，若 linear 耗时下降 20%，该 verify 约省 4.0 ms、延迟下降约 14.7%，已知 GPU 阶段下降上限约 13.2%。这些是固定其余成本、固定接受长度的条件推算，不是落地后的实测结果。

因此单项收益不能相加，kernel 加速也不能直接转为 tok/s。推测解码应比较完整周期时间除以实际发布的新 token 数，DFlash2 或组合模式都必须对照当前最佳 MTP。Prefix 复用另外报告重算 token 减少量与 restore 成本；参考 42% 的 prefill token 减少也不会自动成为 42% 的总请求延迟改善。

### 降低数值精度，不等于已经证明质量下降

重新量化 target 或改变 attention 乘法精度会改变模型实际计算出的分布，可能损伤代码、数学、多语言、长上下文和工具调用，也可能在有限样本中表现接近。Dense8 的小样本 PPL 接近支持继续实验，不能给出所有任务无损的结论。PPL、KL、top-1 和任务成功率分别衡量不同问题；输出文本不同也不能单独证明质量下降。

只量化草稿时，target 仍用完整词表计算最终分布 `p`。Greedy 必须逐位置接受 target 同意的 token，并由 target 更正首个拒绝位置；随机模式必须使用实际条件提议分布 `q` 完成接受和残差采样，包括 shortlist 外 token 的正确处理。在这些条件下，草稿更差主要降低接受率，不应额外引入提议偏差。[推测解码原论文](https://arxiv.org/abs/2211.17192) 证明的是准确接受算法下的分布保持，不是任意实现或量化 target 相对 BF16 的无损性。

该分布契约以 target 计算一致为前提。批量验证宽度、prefill 路径或数值制度若使 target logits 改变，不能只凭“有 target 验证”承诺与逐 token 基线逐位一致；随机模式保持分布也不等于固定 seed 的文本相同。ReplaySSM 的状态逐位验收则限定为同一物理 verify block 的接受前缀。

第一阶段保持当前 target 权重、计算 dtype、量化规则和采样规则，先验证 ReplaySSM、草稿资源优化和设备驻留。Target 重新量化、低精度 P×V 与改变数值规则的融合另开实验，先完成按任务和 context 分桶的质量对照，再决定是否启用。

## 迁移方案入口

执行顺序统一见 [路线图](../plans/README.md)，实验条件与收益判定执行 [性能基线方案](../plans/performance/baseline.md)。本调研提供机制、测量边界与收益估算；Target 重新量化另设独立制品与质量门禁，与保持权重身份的引擎 A/B 分别验收。

实施细节维护在 [CUDA 优化方案](../plans/performance/cuda.md)、[DFlash2 接入方案](../plans/speculation/dflash2.md)、[SPI 方案](../plans/speculation/spi.md) 和 [组合方案](../plans/speculation/composition.md)。现有源码问题见 [代码审查记录](../reviews/cuda-commits-2026-10-09.md)。
