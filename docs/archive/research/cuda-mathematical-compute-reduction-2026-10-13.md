# 从数学结构减少推理计算（2026-10-13）

继续对标 vLLM 的性能目标。本轮检查当前模型的真实 tensor shape、现有执行图和原始论文，
筛选可以少算的算法。完成 CPU 静态审计，没有改运行时默认策略，也没有新的 GPU 或服务加速结论。
研究基点为 main `e7acd2226bf2df3c3235ad98f54c4af839edfaba`。

## 当前计算量在哪里

冻结 Qwen3.8-27B-NVFP4 的 Safetensors header：hidden=5120、intermediate=17408、
64 层中 48 层 GDN、16 层 full attention，词表 248320。
最后一层为 full attention。Attention 的 q projection 包含 gate，实际 shape 是 `[12288,5120]`，不能按 6144 少计一半。
以下从全部 497 个 target projection 的实际 shape 求和，排除 embedding lookup 和 MTP。

| 投影类别 | 矩阵数 | 每 token MAC | 投影 MAC 占比 |
|---|---:|---:|---:|
| MLP gate/up/down | 192 | 17,112,760,320 | 66.789% |
| GDN 输入/输出投影 | 240 | 5,560,074,240 | 21.700% |
| Full attention 输入/输出投影 | 64 | 1,677,721,600 | 6.548% |
| LM head | 1 | 1,271,398,400 | 4.962% |
| 合计 | 497 | 25,621,954,560 | 100% |

MAC 是矩阵乘的数学工作量，按两次 FLOP 计约 51.244 GFLOP/token。
这不是实际 graph 指令数或耗时占比：没有计入量化、补偿乘积、padding、状态更新、softmax、
norm 和 cache 操作；prefill 和 decode 的有效行数也不同。
[已有图级诊断](cuda-parallel-wy-quality-and-decode-cost-2026-10-13.md)中 linear segment
约占四活跃槽图时间的 79.2%，包含量化，不等于 MLP 占 79.2%。

静态稀疏审计预先固定层 0/16/32/48 的 gate/up/down 共 12 个完整 NVFP4 张量，
检查 1,069,547,520 个编码权重。零值（包括负零）共 74,215,313，比例 6.9389%；
66,846,720 个 block-16 中没有全零块。这只证明所检查权重缺少可直接跳过的全零块，
不推断未检查层、动态 activation 稀疏率、2:4 稀疏性或矩阵秩。

## 1. 按真正需要的行裁剪终端计算

[decoder recipe](../../../crates/model/recipes/src/decoder.rs)中，最后一层的 KV 在 attention
处更新；MLP 位于它之后。普通 MTP=0、无需 Full hidden 的 prefill 只消费最后一行 logits。
因此可从输出需求反向传播：最后层保留全部有效行的 K/V 投影、RoPE 和 KV 写入，
只计算最后行的 Q/gate、attention readout、O projection、MLP、最终 norm 和词表输出。
此前 chunk 若既不读 logits 也不读 hidden，最后层完成 KV 后即可停止。

这是数据依赖消除，权重和有效 token 的公式不变。它不允许跳过前 63 层的其他行，
这些行仍为下层 K/V 和 recurrent 状态提供输入。MTP priming、Full readout、其他图内消费者
要求哪些 hidden 行，就必须保留哪些行；不能只凭层号和名称进行裁剪。
实现应根据 IR 的消费者、state effects 和 readout 要求规划行集合。

该模型每个被省掉的非最后行约少算 1,633,157,120 MAC，占表中投影预算的 6.374%。
其中 LM head 占 4.962%，最后层 Q/O/MLP 合计只占约 1.412%。
还可能省掉该 attention 层的非最后 query readout，但不省全部 K/V 工作；
这是数学预算，没有按比例预测 TTFT。

[之前单行 logits 实验](cuda-performance-experiments.md#167-单行-logits-的服务对照吞吐与首-token-中位数出现权衡)
已有正确性证据，却出现服务中位数回退并撤回；同批数据还显示驻留/排队和整组吞吐变化。
新的实验必须先保持旧 arena/admission，仅改变需要执行的行，单独对照计算效果，
随后再评估存储收缩。单行 dispatch 必须和原有效行保持同一量化与累加口径，不能直接用
另一个 GEMV 的浮点结果替代。这是首个无需改模型即可落实的计算消除方向。

首个实现先只做无 hidden/logits 读出的中间 chunk：从所有 `node.states` 写入反向保留完整
输入依赖闭包，省去末层 O/MLP 等 12 个节点，每行在已有 LM head skip 之外少算
298,844,160 MAC。仍完整执行最后 attention 的 Q/readout，不把它与 KV append 分离，
也不改变最终 chunk 的行数。两个对照程序保留相同 arena 和额外图，避免 admission 干扰。

[内部规划器](../../../crates/backend/cuda/src/resident/prefill_demand.rs)仅编译进 CUDA 单元测试，
没有生产环境变量或 CLI 开关。[完整模型检查](../../../crates/backend/cuda/tests/unit/resident_prefill_demand_model.rs)
在默认 scalar recurrent、MTP=0、冻结权重/tokens/tuning 下，对宽度 64/256 和长度
51/64/255/256/511 比较全部 F32 状态、FP8 KV、完整最终 logits、每个长度后 8 步完整
hidden/logits 与 greedy 选择；均逐位一致，hidden readout 的完整图回退也通过。
CUDA clippy 和 71 项常规测试通过。配对耗时入口已准备，尚无性能收益结论；
原始 256 题质量、服务负载与 MTP 实测未完成，不采用生产默认。
证据位于 GPU 主机 `state-prefill-demand-2026-10-13/check-v4.log` 与 `compile-check-v5.log`。

## 2. GDN 用内积读出，延后展开状态

最接近“换一个公式，省掉中间计算”的例子是：

```text
(u kᵀ) q = u (kᵀ q)
```

左边先生成一个 128×128 的外积再读出，右边先算一个长度 128 的内积，再缩放向量。
单个更新项的中间矩阵可以消失；它不是整个 GDN 或模型自动加速 128 倍。
[ReplaySSM 原始说明](https://dao-lab.ai/blog/2026/replayssm/)也区分 state reconstruction
和 output-only 路径，并明确 GDN 的 correction 依赖此前状态。

以 value×key 方向写 GDN：

```text
u_t = beta_t (v_t - a_t S_(t-1) k_t)
S_t = a_t S_(t-1) + u_t k_tᵀ
S_t q_t = G_t S_0 q_t + Σ_j c_(j,t) u_j (k_jᵀ q_t)
```

近期只保存 correction `u_j`、key 和 decay，窗口满或遇到 prefix/迁移边界才物化状态。
不能把原始 v 缓存当成 u；checkpoint 的两个 matrix-vector readout 及 flush 成本仍存在。
全模型 GDN 的完整 F32 state 为 144 MiB/sequence，延后写回主要减少状态流量。
历史内积增加计算，故需核算完整周期；此处并不保证总 FLOP 一定降低。

本项目[已有 F32 原型](cuda-replayssm-gdn-2026-10-12.md)通过独立 F64 的原容差。
外部 48 层普通 decode 状态链的最佳筛选结果为 batch1 1.218 倍、batch4 1.325 倍；
窗口加大也会变慢。baseline 是外部 scalar kernel，不能当成 native 优势或端到端收益。
下一步应对 native 测完整周期，且验证 commit/rollback、prefix materialization、inactive slot、
卷积和 KV 状态，再跑原质量与完整 token-stream 断言。

[GDN 论文](https://arxiv.org/abs/2412.06464)的 WY/chunkwise 表示主要减少串行依赖，
并不保证减少乘加。当前已有 WY；新并行准备方案的质量失败也已记录，不能当作新发现再次打开。

## 3. MLP 低秩分解：最大的少算空间，需要模型质量验收

把矩阵 `W` 近似为 `A B`，先算 `z=B x` 再算 `y=A z`：

```text
原计算：N K MAC
秩 r 分解：r (N+K) MAC
本模型单个 MLP 投影：N K = 17408 × 5120
少算的条件：r < 3956.36
r = 2048：MAC 降低 48.235%
```

若三个投影、全部 64 层都能采用这一秩，表中的总 projection MAC 理论上降低约 32.216%。
这只是设定秩后的预算，不证明权重有该有效秩、质量合格或服务提速。
完整保留 rank=5120 的分解反而增加 MAC；截断改变 checkpoint，不能声明逐位等价。
SwiGLU 的非线性位于 gate/up 与 down 之间，也不能把三个投影直接合成一个矩阵。

[NVFP4 格式](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference)
每 16 个 4-bit 权重另有一个 FP8 scale，主要存储约 0.5625 B/weight。
若 rank2048 因子存 BF16，虽然乘加少约 48%，权重字节约是原矩阵的 **1.841 倍**，
小 batch decode 可能因此更慢。因子也量化会引入新的误差，不能把两个因素分别验证后直接相加。

[SVD-LLM](https://arxiv.org/abs/2403.07378)提供 activation-aware whitening 和逐层补偿方法。
可作为另一个模型制品实验：先校准目标 workload 的 activation 分布，估计 rank/error 曲线，
选择因子精度与存储预算，再完整验证模型质量。保留现有质量门槛；不能凭保留 99% 奇异值能量
认定任务质量或生成序列等价。未经校准，不优先用普通 SVD 直接改生产权重。

## 4. 稀疏和查表：先区分结构，再决定适用后端

只有实际为零的输入/权重，或已证明不影响输出的项，才能直接省略。
当前所查 MLP 没有全零 block-16；SwiGLU 的“小值”不能直接当成零。
[Turbo Sparse](https://arxiv.org/abs/2406.05955)依靠改变 activation 并继续训练来获取高稀疏性，
属于模型改造。结构化剪枝、跳层、KV 丢弃和跨 token 近似复用也需要自己的质量验收。

[LUT-GEMM](https://arxiv.org/abs/2206.09557)和
[T-MAC](https://arxiv.org/abs/2407.00088)用低比特组合查表代替乘法，是另一类数学计算替换。
但当前 MLP 已经使用 NVFP4 W4A4 的硬件 `mmaf_scaled`，不是先完整反量化再 F32 点积；
T-MAC 的 CPU 收益不能移植为 CUDA 收益。E2M1 的非均匀值、block-16 scale、表生成、
缓存/寄存器和新累加顺序都须计入成本。CUDA 优先级低于终端裁剪和已有 Replay 原型；
CPU/缺乏低比特矩阵指令的设备可以另行筛选。

只需要 greedy top1 时，也可研究带严格界的 maximum-inner-product search，跳过已证明不会
获胜的词表行。例如部分点积 `p` 的剩余项可用 Cauchy–Schwarz 上界 `p+||w_rest||||x_rest||`。
必须包含量化与浮点累加误差、tie-breaking，并在无法排除时回到完整点积；ANN 或高概率界
不能当成精确替代。全 logits、随机采样和 logits processors 的需求另行判断。
[Exact MIPS 研究](https://arxiv.org/abs/1706.01449)还展示规则矩阵乘可能胜过复杂剪枝。
LM head 只占上述投影预算约 4.96%，因此先测界是否足够紧，不先建设大型索引。

## 实验顺序与验收

1. 做 state-aware 的终端行需求分析，先冻结 arena/admission，比较实际少算的图；
   独立数学参考和原有效行的 bitwise 检查通过后再进入固定上下文、完整质量与服务验证。
2. 对既有 F32 GDN readout 原型测 native 完整周期，窗口由设备与 workload 的真实成本选择；
   状态边界和质量通过前只保留内部测试入口，不向生产用户提供算法开关。
3. 为 MLP 低秩/结构化稀疏建立独立模型制品与校准方案，先验证质量和字节预算；
   不能把它当作保持原 checkpoint 算术的内核优化。

Prefix 缓存已有实现，只有实际命中才减少 prefill 重算；零命中 workload 无新增收益。
[精确推测解码](https://arxiv.org/abs/2211.17192)减少 target 的串行轮次、复用权重读取，
并不必然减少总 FLOP。已有 MTP 需要按 `(draft+verify+commit)/实际发布 token 数` 衡量完整周期，
不能用理论接受率或单个验证 GEMM 作收益；greedy 保留 target token 断言，随机模式需精确校正。

所有新运行时改动仍先通过完整 CI，再进行正式服务性能对照；保持 F32 状态、独立参考、
原误差阈值、原 256 题质量门槛和完整输出断言，不因公式在实数上相等就免除浮点验收。

## 可复算证据

冻结审计目录为 `/home/r/benchmarks/inference.rs/math-reduction-2026-10-13/`。
`static-audit.py` SHA256：`4021b4ca99dbc3fcf90c976b3f59f5f2e05b9a3b492f26a1e749c139b8b895ea`；
`static-audit.json` SHA256：`3aeeeef386f34f1d60a2aad597acfcb485de51b49c911b007e7f2a881bb8711d`。
JSON 保留全部 projection shape、样本 tensor 字节哈希、逐码 histogram 和 block 零值计数。
config SHA256：`1b3c71868d1299e52df6fc907deb202d5132b1ef0f72aae0ef6d15185dd53a5c`；
index SHA256：`429430e1b9e65b2cb98eff8cd10a06e70a09cee89c48487a3914684aeb6df57f`。

CPU 审计使用 `safe-run.sh --memory-gib 4`，protected unit `inference-benchmark-2301038`，
退出 0，runtime 2.657 s，memory peak 504.8 MiB，swap=0；没有启动 GPU 模型或计时。
