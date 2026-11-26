# ReplaySSM：gated-delta 层缓存输入而非状态（CUDA 设计）

状态：设计方案，基于 `3f868b1c`（2026-11-15），未修改实现。分层与回滚协议见
[Recurrent 状态重放与回滚分层方案](../speculation/state-replay.md)（R1）；本文是其中
“backend 选择物理回滚策略”在 CUDA gated-delta（GDN）上的一个具体策略，不改变公共契约。

来源：Ze-Wei Liou 等，[ReplaySSM: Cache SSM Inputs, Not State](https://arxiv.org/abs/2610.13778v1)
（附录 B）、[作者博客](https://tridao.me/blog/2026/replayssm/)、
[参考实现](https://github.com/Johnny-Liou/ReplaySSM) 和
[FlashInfer GDN 内核](https://github.com/flashinfer-ai/flashinfer/pull/4081)。
论文在 Qwen3.5 GDN 上报告，batch 256、L = 16 时，标准解码的 SSM 内核加速 1.46–1.68×，
端到端加速 1.20–1.27×，batch 1 的端到端吞吐为基线的 0.96–0.99×。
Qwen3.5-122B 的推测解码在 B300、batch 512、窗口 4 下为传统推测解码吞吐的 2.10×。
本项目的 27B 小批次收益单独验证。

## 1. 为什么做：本仓库的现状与预期收益

27B（Qwen3.8 NVFP4）有 48 个 GDN 层，几何为 16 个 key head × 128、48 个 value head × 128。
每层每序列的状态是 48 × 128 × 128 = 786,432 个元素；默认 BF16 状态（#255）为 1.5 MiB，
48 层合计 **72 MiB/序列**。

当前路径（`resident/slot_recurrent.rs`、`slot_commit.rs`）已经是“无快照验证”：验证 lane
读状态但不写，接受数确定后由 commit 图重放接受前缀并**写回整份状态**。因此每个推测周期
仅计一次 checkpoint 读写即约 144 MiB/序列，验证 lane 之间仍串行执行 delta 递推。

实测（RTX 5090，DFlash2，4 序列 × 8 候选，节点级设备计时，见
`/home/r/benchmarks/inference.rs/verify-lanes-2026-11-15/`）：验证图 20.8 ms 中 `delta`
节点合计 **3.4 ms**（48 层，≈70 µs/层），是投影之后最大的单项。

按每层每 slot 读取一次 BF16 检查点 S₀（1.5 MiB）和最多 16 条已提交记录（259 KiB）估算，
4 个 slot 合计约 337 MiB。按 1.79 TB/s 折算，理想读取耗时约 0.2 ms。额外读取、
候选记录写入、三角求解和 flush 分开计入实测。若验证的 delta 从 3.4 ms 降到 1.0 ms，
仅这一项相当于 27.5 ms 步长的约 8.7%。第 7 节验收完整周期收益。

### 1.1 实测修正（2026-11-16，P1 原型）

上面的收益估算把验证 `delta` 的 3.4 ms 当成带宽问题，实测否定了这个前提。

- 现有 `delta_verify_lanes` 已经是“每 slot 读一次状态”：状态 tile 留在寄存器里，8 个候选在其上
  串行更新，不保存逐候选快照。3 slot × 8 候选时单层约 40–56 µs，而状态读取的带宽下限只有约
  3.5 µs。耗时来自每个候选 3 次依赖的跨线程归约（8 × 3 = 24 次），不是显存流量。
- P1 的 chunked 验证原型在分支 `feat/replayssm-verify`（未合入）：对串行内核在 1e-5 容差内一致，
  覆盖 F32/BF16 状态、32/64 列切分和 padding lane。用 `(I + A)⁻¹ = (I − A)(I + A²)(I + A⁴)(I + A⁸)`
  消掉候选间的串行步，按 key 维 32 行分块压住寄存器后，最佳约 65 µs（128 列一块、occupancy 1），
  仍慢于串行。原因是固定开销：key/query 乘积、16 × 16 求逆和 F32 精度拆分共约 60 次 16 行的
  tensor core 乘积，在 cuTile 中每次的固定延迟累加后超过 24 次归约。
- 第一版原型把整份 `[16, 128]` key/query tile 及其拆分同时放在寄存器里，溢出导致 capture 超出池
  预算，池被重新规划为 1 个 slot，端到端从 372 降到 92 tok/s。任何新内核在接入前都必须先在单层
  计时上胜过现有内核，并核对 capture 后的池宽度。

结论：在本仓库，ReplaySSM 在推测验证上的主要收益（单次读状态、无快照）已经由现有内核拿到。
剩余收益在**状态写回**：每步存回整份状态。标准解码在大并发时最明显，batch 64 时 27B 每步的
GDN 状态读写约 9 GB（72 MiB × 2 × 64），按带宽约 5 ms；缓存输入可以省掉其中的存回部分。

## 2. 算法（GDN）

记号：每个 value head 的状态 S ∈ ℝ^{d_v × d_k}；一步输入 (q, k, v, g, β)，α = e^g。
基线一步：S ← αS；u ← β(v − S k)；S ← S + u kᵀ；y ← S q。

关键：**缓存 u（修正后的写入值）而不是 v**。u_t 依赖 S_{t−1}，缓存 v 会重新引入串行依赖；
缓存 u 后 S_t = α_t S_{t−1} + u_t k_tᵀ，状态展开成“衰减的检查点 + 加权外积和”。

每 slot 每 GDN 层保存检查点 S₀ 和环形缓冲 B = {(u_j, k_j, g_j)}。
h 是已提交长度，L 是已提交上限。T 是本轮实际有效验证位置数，含 seed。填充位置跳过。
令 G_j 为自检查点起的累计 gate，Ā = e^{G_h}，w_j = e^{G_h − G_j}。

- 历史读出（不物化状态）：对任意向量 x，S_h x = Ā (S₀ x) + Σ_j w_j u_j (k_jᵀ x)。
- 标准解码（论文附录 B.2 的输出形式）：hq = S_h q，hk = S_h k；u = β(v − α hk)；
  y = α hq + u (k·q)；把 (u, k, g) 追加进缓冲。
- 推测验证（论文附录 B.3，chunked delta rule）：T 个有效位置，G_s 从本轮 seed 开始累加。
  1. hq_s = S_h q_s，hk_s = S_h k_s（对 T 个候选是一次 GEMM：S₀ 只读一次）。
  2. A_{s,s'} = β_s e^{G_s − G_{s'}} (k_s·k_{s'})，s' < s（严格下三角）；
     R_s = β_s (v_s − e^{G_s} hk_s)。
  3. U = (I + A)⁻¹ R（一个 T × T 三角求解，T ≤ 9）。
  4. y_s = e^{G_s} hq_s + Σ_{s' ≤ s} e^{G_s − G_{s'}} (k_{s'}·q_s) U_{s'}。
  5. 将本轮记录写到已提交前缀之后的 T 个槽位，保持检查点和已提交游标。
- 提交：由现有事务校验提供消费长度 c。正常提交 c = 1 + a（seed 加接受的 a 个草稿），
  0 ≤ a < T。首个草稿被拒仍提交 seed。取消在提交前丢弃整段临时记录，保持原检查点与游标。
- Flush：成功提交时，若 h + T > L，先把旧的 h 条已提交记录汇总进检查点
  S₀ ← Ā S₀ + Σ_j w_j u_j k_jᵀ，再把环起点前移 h、令 h ← 0。随后统一执行 h ← h + c。
  先汇总旧记录使待提交记录在原址成为新前缀。提交后始终有 h ≤ L，前提是 T ≤ T_max ≤ L。
  未消费的尾部在该事务最后一个设备读者完成后可覆盖。

实数运算下公式与逐步递推等价。三角求解、历史重建、u/k 的缓存转换与检查点 flush
会改变有限精度下的舍入位置。F32 路径用 F32 记录作为数值参考，BF16 记录配合
`RecurrentProfile::Bf16` 在现有 BF16 容差下独立验收。

## 3. 数据布局与容量

验证计划固定物理容量 C = L + T_max。前 L 个逻辑位置留给已提交记录，额外 T_max 个位置
容纳尚未提交的一轮。所有索引以 C 取模，每个 slot 使用当前事务的实际 T。

| 项 | 形状（每 slot 每 GDN 层） | 精度 | 27B，L = 16、T_max = 9 |
|---|---|---|---|
| 检查点 S₀ | `[H_v, d_v, d_k]` | `RecurrentProfile` | 1.5 MiB（BF16） |
| u 环 | `[C, H_v, d_v]` | BF16 或 F32 | 300 KiB（BF16） |
| k 环 | `[C, H_k, d_k]` | BF16 或 F32（已归一化） | 100 KiB（BF16） |
| g 环 | `[C, H_v]` | F32 | 4.69 KiB |
| 环指针 | `start, committed`（i32） | — | 每 slot 一组，所有层共用 |

BF16 记录约增加 19 MiB/序列（48 层），相对于当前无快照路径是新增驻留成本。
池报价（`SlotPool::rollback_bytes` / `attention_bytes` 同级）按实际 dtype 与布局计入全部记录、
游标和临时 workspace。宽窄图共享同一物理环和游标，临时存储按实际峰值报价。

**L 不做环境变量**（AGENTS：不发布调参开关）：从“缓冲读取 h·(H_v d_v + H_k d_k)·2 字节”与
“flush 写回 H_v d_v d_k·字节数 / 平均 flush 间隔”的成本模型推导，加载时可用一次测量修正；
以论文使用的 16 为初值，结合实际消费长度、flush 频率和池容量确定 L。

环形缓冲用 `(start + i) mod C` 索引。flush 写回检查点，记录保留在原地址。
所有层先使用同一组旧游标，最后由一次池级提交更新游标。

## 4. 内核

全部放在 `infer-cuda-kernels` 的 recurrent 模块，契约在 `contract/` 声明，tile 实现；AOT 返回
显式 `unsupported`（与 lanes attention 相同）。

1. `replay_verify`：一次 launch 覆盖池内所有 slot 的 T 个候选。每个 CTA 负责
   (slot, value head 组)：读 S₀ 一次，算 hq/hk（S₀ 与 `[d_k, 2T]` 的乘积，tensor core），
   缓冲项的 k_jᵀ[q|k] 是 `[h, 2T]` 的小 GEMM，A 与三角求解在寄存器/共享内存完成（T ≤ 9），
   输出 y 并写候选 (U, k, g)。key head 被 3 个 value head 共享，k·q 在组内算一次；
   每个共享 key 记录也由单一写者写入。
2. `replay_decode`：T = 1 的特化（标准解码与 serial 路径）。
3. `replay_flush`：按每 slot 的 flush 标志（设备端数据，不是编译期常量）把已提交条目汇总进 S₀。
   提交图里固定发射，未到阈值的 slot 早退。单独发射时额外读取检查点的成本计入 flush。
4. 提交：沿用 `commit_verify_many` 的接受计数上传。按同一消费长度完成 GDN 游标、conv1d、
   KV 与 features 的提交。沿用现有设备完成边界发布结果，记录随事务和 slot lease 保留到
   最后一个设备读写完成。提交失败且检查点可能被部分写入时，隔离该状态并禁止复用。

conv1d 状态（短窗口）保持现有处理；它很小，不在本方案收益范围内。

## 5. 接入点

- 下沉：`recurrent_lowering.rs` 增加 `Recurrence` 的 replay lowering，按能力（GDN 几何、
  `T ≤ MAX_VERIFICATION_WIDTH`、tile 可用）在绑定时选择。自动选择可以继续使用已支持的
  现有策略，显式请求 ReplaySSM 而能力不足时返回明确错误。
- `slot_recurrent.rs` / `slot_commit.rs`：验证改为 `replay_verify`；commit 图去掉状态写回，
  改为消费计数上传、按需 flush、最后更新游标。DFlash2 的宽验证使用同一 replay 内核，
  无需依赖 `slot_recurrent::LANES = 3` 的窄 GDN 路径。
- `SlotStates` / 池分配：每 slot 增加 u/k/g 环与指针；`loading/budget.rs` 的池报价加入该项。
- Prefill：新 slot 以空环启动，prompt 图继续使用现有 chunked recurrent 路径。
  在已有 replay 状态上续做 prefill 时，先物化已提交状态，再继续递推。
- 在 checkpoint/导出（`export`、`export_unpaged`）、prefix 缓存的 recurrent 快照、
  slot 迁移及读取 target recurrent 状态的草稿 ingest 之前，先物化已提交状态并等待设备完成。
  恢复从空环启动。单纯读取 paged KV 的 `kv_views` 无需物化 GDN。
- 设备计时：内核天然落在 `OpClass::Recurrence`，无需额外埋点；flush 单独成节点，便于看摊薄成本。

## 6. 数值与正确性

- 参考：`examples/recurrent-delta/golden.json` 与 `crates/engine/state/tests/recurrent_reference.rs`
  校验已登记的参考制品，逐步递推使用 `crates/engine/workloads/src/recurrent.rs` 的独立参考。
  对同一输入逐项比较每个有效位置的输出、任意接受前缀的状态、flush 后状态与后续输出。
  F32 和 BF16 分别沿用各自的 recurrent 容差，覆盖非零基点、不同 gate 与长序列重复 flush。
- 设备测试显式运行：T = 3、8、9；仅 seed、部分及全接受；取消零提交；h + T 等于及超过 L；
  环回绕、变宽、填充、非活跃 slot、各 slot 不同步 flush、宽窄图切换、取消后复用和导出恢复。
  同时对齐 GDN、conv、KV 与有效输出的消费位置。
- 质量：沿用 `make check-iteration` 的冻结题集、64 题基线和阈值。有限精度策略的启用以
  数值门禁和任务质量为条件。P1 继续使用既有状态提交路径。

## 7. 性能验收

按[性能基线方案](baseline.md)固定设备、权重、dtype、调优表、图宽和请求形状。迭代使用
`make check-iteration` 和针对改动路径的同机 release A/B。逐配置比较 DFlash2、MTP
（各 4 并发、256 token）与无推测的 C16 及单请求短解码，先固定接受长度比较递推成本，
再比较实际接受率下的端到端收益。
记录 `/native/v1/device/timelines` 的 verify、commit、flush 时间与节点级 `Recurrence`，
以及每配置的吞吐、TTFT、TPOT、尾延迟、峰值显存和最大可用并发。

## 8. 分阶段

**结论（2026-11-16）：暂不实施。** 按 1.1 节的实测，在当前模型与硬件上逐项估算收益：

| 收益来源 | 本仓库现状 | 预期收益 |
|---|---|---|
| 验证只读一次状态、无逐候选快照 | 现有 `delta_verify_lanes` 已经做到 | 0（chunked 原型更慢） |
| 提交不写回整份状态 | commit 阶段只占步长 0.4–1.4% | < 1% |
| 省掉快照显存、提高并发 | 现有路径没有快照 | 0 |
| 标准解码每步不写回状态 | 27B 在 5090 上池宽上限 8；batch 8 时状态读写约 1.15 GB（≈0.65 ms），权重读取约 17.8 GB（≈10 ms），状态约占 5%，ReplaySSM 省掉其中约一半写回、再扣除缓冲读取 | ≈2–3% |

论文的 1.2–1.9× 来自三个本仓库不具备的条件：batch 128–512、激活参数约 10B 的 MoE
（每步权重读取小，状态流量占主导）、逐候选保存状态快照的推测基线。

**重新评估的条件**（任一满足即重开）：接入激活参数小的混合 MoE 模型；或硬件显存允许
batch ≥ 64；或设备计时显示最大并发下 `OpClass::Recurrence` 超过步长约 15%。重开时按下列顺序，
每步都要先在单层内核计时上胜过现有内核，并核对 capture 后池宽度不变：

1. **标准解码**：环形缓冲 + `replay_decode`（T = 1，只读检查点、追加 `(u, k, g)`）+ `replay_flush`。
2. **推测提交**：commit 从写状态改为前移指针，复用环与 flush；补齐所有读状态消费者的 flush
   （导出、prefix 缓存、`kv_views`）。
3. **推测验证**：只有出现单层胜过 `delta_verify_lanes` 的实现才接入；原型在
   `feat/replayssm-verify`。
4. **L 的推导**：成本模型 + 加载时测量，替换初值 16。

## 9. 风险与待定

- 三角求解在 T = 9 时的寄存器压力与精度（论文用 WY/chunk 形式；必要时 F32 求解）。
- BF16 的 u/k 舍入与检查点舍入会进入后续状态，flush 不会消除已引入的误差。按上下文长度
  和 flush 次数观察误差；必要时重新评估记录精度与其显存、性能成本。
- 与 prefix 缓存共享：只能共享 flush 后的检查点；共享尾部仍按 copy-on-write 规则。
- 论文的 vLLM 实现是参考而非依赖；不引入新的运行时或第三方 crate。
