# ReplaySSM：gated-delta 层缓存输入而非状态（CUDA 设计）

状态：设计方案，基于 `3f868b1c`（2026-11-15），未修改实现。分层与回滚协议见
[Recurrent 状态重放与回滚分层方案](../speculation/state-replay.md)（R1）；本文是其中
“backend 选择物理回滚策略”在 CUDA gated-delta（GDN）上的一个具体策略，不改变公共契约。

来源：Tri Dao 等，*ReplaySSM: Cache SSM Inputs, Not State*（arXiv 2610.13778；
[博客](https://tridao.me/blog/2026/replayssm/)；代码 <https://github.com/Johnny-Liou/ReplaySSM>，
vLLM 实现见 vllm-project/vllm#48018，FlashInfer 内核见 flashinfer-ai/flashinfer#3975、#4081）。
论文在 Qwen3.5（GDN）上报告：标准解码 SSM 内核 1.43–1.64×、端到端 1.20–1.27×（batch 256，
buffer 16）；推测解码（窗口 4，MTP）端到端 1.87–1.96× 于 vLLM 标准解码，并发上限 3.0–3.3×。
batch 1 标准解码无收益。

## 1. 为什么做：本仓库的现状与预期收益

27B（Qwen3.8 NVFP4）有 48 个 GDN 层，几何为 16 个 key head × 128、48 个 value head × 128。
每层每序列的状态是 48 × 128 × 128 = 786,432 个元素；默认 BF16 状态（#255）为 1.5 MiB，
48 层合计 **72 MiB/序列**。

当前路径（`resident/slot_recurrent.rs`、`slot_commit.rs`）已经是“无快照验证”：验证 lane
读状态但不写，接受数确定后由 commit 图重放接受前缀并**写回整份状态**。因此每个推测周期
每层每序列至少一次状态读写（≈144 MiB/序列），且验证 lane 之间是串行 delta 递推。

实测（RTX 5090，DFlash2，4 序列 × 8 候选，节点级设备计时，见
`/home/r/benchmarks/inference.rs/verify-lanes-2026-11-15/`）：验证图 20.8 ms 中 `delta`
节点合计 **3.4 ms**（48 层，≈70 µs/层），是投影之后最大的单项。

ReplaySSM 下验证每层每 slot 只读一次检查点 S₀（1.5 MiB）加缓冲（L = 16 时 ≈ 0.26 MiB），
4 个 slot 约 336 MiB/步，按 1.79 TB/s 的下限约 0.2 ms；状态写回只在 flush 时发生并被摊薄。
**估计**可把 3.4 ms 降到 1 ms 以内（另加 commit 图的状态写回被去掉），约占 DFlash2 步长
27.5 ms 的 8–10%。这是估计，验收以第 7 节的实测为准。

## 2. 算法（GDN）

记号：每个 value head 的状态 S ∈ ℝ^{d_v × d_k}；一步输入 (q, k, v, g, β)，α = e^g。
基线一步：S ← αS；u ← β(v − S k)；S ← S + u kᵀ；y ← S q。

关键：**缓存 u（修正后的写入值）而不是 v**。u_t 依赖 S_{t−1}，缓存 v 会重新引入串行依赖；
缓存 u 后 S_t = α_t S_{t−1} + u_t k_tᵀ，状态展开成“衰减的检查点 + 加权外积和”。

每 slot 每 GDN 层保存：检查点 S₀，以及环形缓冲 B = {(u_j, k_j, g_j)}，长度 h ≤ 容量 L。
令 G_j 为自检查点起的累计 gate，Ā = e^{G_h}，w_j = e^{G_h − G_j}。则

- 历史读出（不物化状态）：对任意向量 x，S_h x = Ā (S₀ x) + Σ_j w_j u_j (k_jᵀ x)。
- 标准解码（论文 Algorithm 6 的输出形式）：hq = S_h q，hk = S_h k；u = β(v − α hk)；
  y = α hq + u (k·q)；把 (u, k, g) 追加进缓冲。
- 推测验证（论文 Algorithm 8，chunked delta rule）：T 个候选，候选内累计 gate G_s。
  1. hq_s = S_h q_s，hk_s = S_h k_s（对 T 个候选是一次 GEMM：S₀ 只读一次）。
  2. A_{s,s'} = β_s e^{G_s − G_{s'}} (k_s·k_{s'})，s' < s（严格下三角）；
     R_s = β_s (v_s − e^{G_s} hk_s)。
  3. U = (I + A)⁻¹ R（一个 T × T 三角求解，T ≤ 9）。
  4. y_s = e^{G_s} hq_s + Σ_{s' ≤ s} e^{G_s − G_{s'}} (k_{s'}·q_s) U_{s'}。
  5. 把 (U_s, k_s, g_s) 写入缓冲的候选区（尚未提交）。
- 提交：接受 a 个候选 = 缓冲长度 h ← h + a。被拒候选只是不前移指针，**无状态恢复**。
- Flush：当 h + 2T > L 时（提前一个窗口，保证下一步至少有 T 个空位；论文 5.3 节的理由），
  把**已提交**条目汇总进检查点：S₀ ← Ā S₀ + Σ_j w_j u_j k_jᵀ，清空已提交部分；
  当前步的候选不汇总（可能被拒）。

数学上与基线等价（浮点误差内）；u、k 以 BF16 缓存会引入一次舍入，检查点精度沿用
`RecurrentProfile`（F32 参考 / BF16 默认）。

## 3. 数据布局与容量

| 项 | 形状（每 slot 每 GDN 层） | 精度 | 27B，L = 16 |
|---|---|---|---|
| 检查点 S₀ | `[H_v, d_v, d_k]` | `RecurrentProfile` | 1.5 MiB（BF16） |
| u 环 | `[L, H_v, d_v]` | BF16 | 192 KiB |
| k 环 | `[L, H_k, d_k]` | BF16（已归一化） | 64 KiB |
| g 环 | `[L, H_v]` | F32 | 3 KiB |
| 环指针 | `start, committed`（i32） | — | 每 slot 一组，所有层共用 |

额外容量约 12 MiB/序列（48 层），远小于现有每候选快照方案；进入池的资源报价
（`SlotPool::rollback_bytes`/`attention_bytes` 同级的新项），不能当作免费。

**L 不做环境变量**（AGENTS：不发布调参开关）：从“缓冲读取 h·(H_v d_v + H_k d_k)·2 字节”与
“flush 写回 H_v d_v d_k·字节数 / 平均 flush 间隔”的成本模型推导，加载时可用一次测量修正；
论文对 Qwen3.5 测得 16 最优，作为初值与上限检查。

环形缓冲用 `(start + i) mod L` 在内核里索引，提交与 flush 只改指针，避免按 slot 的数据相关拷贝
（CUDA Graph 下也不需要为每行发射拷贝内核）。

## 4. 内核

全部放在 `infer-cuda-kernels` 的 recurrent 模块，契约在 `contract/` 声明，tile 实现；AOT 返回
显式 `unsupported`（与 lanes attention 相同）。

1. `replay_verify`：一次 launch 覆盖池内所有 slot 的 T 个候选。每个 CTA 负责
   (slot, value head 组)：读 S₀ 一次，算 hq/hk（S₀ 与 `[d_k, 2T]` 的乘积，tensor core），
   缓冲项的 k_jᵀ[q|k] 是 `[h, 2T]` 的小 GEMM，A 与三角求解在寄存器/共享内存完成（T ≤ 9），
   输出 y 并写候选 (U, k, g)。key head 被 3 个 value head 共享，k·q 在组内算一次。
2. `replay_decode`：T = 1 的特化（标准解码与 serial 路径）。
3. `replay_flush`：按每 slot 的 flush 标志（设备端数据，不是编译期常量）把已提交条目汇总进 S₀。
   图里固定发射，未到 flush 的 slot 早退；批内不同 slot 在不同步 flush。
4. 提交：沿用现有“主机已知接受数、上传计数”的提交方式（`commit_verify_many`），只是由写状态
   变为写指针；不引入新的 host 同步点。若以后把接受判定放到设备端，再加设备端指针前移内核。

conv1d 状态（短窗口）保持现有处理；它很小，不在本方案收益范围内。

## 5. 接入点

- 下沉：`recurrent_lowering.rs` 增加 `Recurrence` 的 replay lowering，按能力（GDN 几何、
  `T ≤ MAX_VERIFICATION_WIDTH`、tile 可用）选择，不按模型名分支；不满足时返回显式错误或
  保持现有路径，不能静默回退到另一种数值。
- `slot_recurrent.rs` / `slot_commit.rs`：验证改为 `replay_verify`；commit 图去掉状态写回，
  改为指针上传 + `replay_flush`。`slot_recurrent::LANES = 3` 的限制随之消失（DFlash2 的 8 个
  候选走同一内核）。
- `SlotStates` / 池分配：每 slot 增加 u/k/g 环与指针；`loading/budget.rs` 的池报价加入该项。
- Prefill：prompt 图直接写检查点（缓冲为空），保持现有 chunked recurrent 路径；prefill 结束时
  指针归零。
- 读状态的消费者在读之前必须 flush：checkpoint/导出（`export`、`export_unpaged`）、prefix 缓存
  的 recurrent 快照、`kv_views`、草稿 ingest 若读取 target 状态。恢复时缓冲为空。
  这是正确性要求：任何把 S₀ 当作“当前状态”的地方都要换成“flush 后的 S₀”。
- 设备计时：内核天然落在 `OpClass::Recurrence`，无需额外埋点；flush 单独成节点，便于看摊薄成本。

## 6. 数值与正确性

- 参考：`examples/recurrent-delta/golden.json` 与 `crates/engine/state/tests/unit/recurrent_reference.rs`
  的 f64 参考（R1 已固化）。新增：对同一输入序列，“基线逐步递推”与“检查点 + 缓冲 + chunked 验证”
  的输出与 flush 后状态在 F32 下逐项接近，BF16 下按现有 recurrent BF16 容差。
- 设备测试（ignored，按进程单跑）：T = 3 与 8；全接受、全拒绝、部分接受；flush 恰在边界、
  提前一个窗口的条件；环回绕；非活跃 lane；不同 slot 不同步 flush；导出/恢复前 flush。
- 质量：64 题 GSM（与 #267 相同脚本）不劣于基线；不要求 token 一致（见数值契约决定）。

## 7. 性能验收

同机同会话 A/B，release，27B NVFP4：DFlash2 与 MTP（各 4 并发、256 token、3 次），以及
无推测的 C16 解码（论文称标准解码在大 batch 才有收益，需确认小 batch 无回退）。
证据：`/native/v1/device/timelines` 的阶段时间、节点级 `Recurrence` 时间、tok/s；逐配置报告，
不能用总体加速掩盖某一配置的回退。显存：报告池报价变化与最大并发。

## 8. 分阶段

1. **P1 只换验证**：`replay_verify` 在缓冲为空（h = 0）时等价于现有验证，commit 仍写状态。
   先验证 chunked 三角求解的数值与 8 候选收益（应已拿到大部分 `delta` 时间的下降）。
2. **P2 缓冲提交与 flush**：commit 改指针，加入 `replay_flush` 与环；补齐所有读状态消费者的 flush。
3. **P3 标准解码**：`replay_decode` 用于非推测的池解码；按 C16 结果决定是否默认。
4. **P4 L 的推导**：成本模型 + 加载时测量，替换初值 16。

## 9. 风险与待定

- 三角求解在 T = 9 时的寄存器压力与精度（论文用 WY/chunk 形式；必要时 F32 求解）。
- 缓冲 BF16 的 u 舍入在长序列上的累计误差：flush 把它汇总进检查点，误差不随步数无界增长，
  但需要长上下文样本确认（与 #257 的长题一起排）。
- 与 prefix 缓存共享：只能共享 flush 后的检查点；共享尾部仍按 copy-on-write 规则。
- 论文的 vLLM 实现是参考而非依赖；不引入新的运行时或第三方 crate。
