# CUDA 性能实验记录

**定位**：保留既有 A/B、失败实验、源码调查和已落地改动。本文包含历史假设与早期待办，当前任务状态和执行顺序统一见 [路线图](../../plans/README.md)，设计与验收见对应详细方案；正式性能实验统一执行 [性能基线方案](../../plans/performance/baseline.md)。文中 `RM§x` 来自已删除的 Serving 历史记录，仅保留作原始笔记编号。

**历史探索目标**：第一阶段 8 格全矩阵 ≤1.10x；第二阶段 TPOT 类指标稳定领先 vLLM 20–30%。这些是目标，不能作为已取得或可直接承诺的收益。

当前拟议的 recurrent 回滚方案采用保留实际 F32 输入的 Record/Fold，尚未实现；资源与精度边界见 [状态重放方案](../../plans/speculation/state-replay.md)、[Infernix 调研](../../research/infernix-performance.md)。下文的早期 BF16 记录容量、草稿压缩目标及探索性建议以这些方案的完整报价与验收为准。

**置信度标记**：【确证】= 代码或实测确认；【推断】= 有依据的推理；【待测】= 依赖尚
未做的实验。推算收益一律以绝对锚点给出，方便下一轮 bench 后用实测替换。

## 〇、三个共识前提

后续所有方案建立在这三条上：

1. **三个瓶颈同一条主线：「一遍权重服务多少 token」**。27B long TTFT 是 prompt 图权重
   读 4 遍（RM§二.1）；batch4 TTFT 是准入串行 + 每请求各一次全宽 replay；27B batch4
   TPOT 疑似每步 replay 数随并发上涨。修法都指向"权重一遍过、token 成批过"。
2. **调度预算按 replay 次数、不按 token 数**。后端实测 prefill 单次 replay 成本与
   chunk 内 token 数无关（固定宽度图，RM§四 与 `executor/execution.rs:112-141` 互证）
   → 切碎 prefill 是纯亏损；保护 ITL 的旋钮是"本步是否发 prefill"，不是"发多少
   token"。这条与经典 continuous batching 的成本假设相反，是调度侧一切改动的起点。
3. **测量纪律沿用 RM§六**，新增三个一等指标（见 §五）：每图带宽 %、每 token replay
   数、MTP 接受长度。

## 一、追平（≤1.10x）：三项修法

### ① 27B batch4 TPOT 1.897 —— 先测量，假设是 MTP 路径

**机制事实**【确证】：batch decode 走共享 slot 图、一次 replay 推 ≤4 条 lane
（`resident/batch.rs:66`，权重一遍过）；MTP 开启时 decode 走 `slot_speculate` =
propose + verify + catch_up **三次 replay**（`executor/execution.rs:481-550`）；单发
/非池化 decode 回退到每请求私有图串行（`execution.rs:314,348-380`）；
`CB_DECODE_SLOTS = 4`（`constants.rs:101`）。

**假设**【待测】：batch4 恰好顶满 slot 池，每步 replay 数 >1（MTP 三倍化或溢出串行），
每步权重遍数随并发上涨 —— 与"batch1 TPOT 1.14–1.27、batch4 反而 1.897"自洽。

**实验（不改代码）**：

1. `INFER_CUDA_EXECUTION_PROFILE` 跑 27B batch4 稳态段，数每 step replay 次数与 graph
   名。≠1 则根因坐实。
2. 同负载 `--num-speculative-tokens 0` vs 开启，对比 TPOT 与接受长度。

**修法（按成本排序，确认后执行）**：

- **自适应 MTP**：active lanes ≥ 阈值或接受长度 < 阈值时跳过 draft/verify，走纯 slot
  decode。vLLM/SGLang 高并发下同样关闭投机解码。【推断：改动在 executor/调度交界，
  中等】
- 若 verify 确认未跨 lane 成批：verify lane 化。
- 顺带削减每 replay 的同步 readback（`batch.rs:807`）。

**预期**【待测，条件性】：27B batch4 TPOT 1.897 → ~1.2（回归 batch1 水平）；2B
batch4 TPOT 1.383 → ~1.0。

**实测（两个实验都做了，假设被否）**：`INFER_CUDA_PREFILL_PROFILE` 数出的每步
replay 是 2×`slot_decode`（合批 draft propose，各 ~1.6 ms）+ 1×`slot_verify`
（target 12-lane 前向，~27 ms）；`slot_verify` 里 `linear` 19.9 ms、`delta` 4.2 ms、
`conv` 2.3 ms。同负载 `--num-speculative-tokens 0` 反而更慢：TPOT 24.3 ms vs 开启
20.2 ms。**所以 batch4 TPOT 的根因不是"投机三倍化"，关掉投机是净亏损**；瓶颈是
target 前向本身（杠杆 A 的带宽 + `delta` 延迟），MTP 已经在帮忙（每步 ~1.9 token/lane）。
自适应 MTP 作为"高并发让位"机制仍可留作长尾保护，但不能当作 TPOT 的修法。

### ② batch4 TTFT 2.45–2.63x —— 快修 A/B + 结构修 C

**机制链**【确证，来自调度层逐行调查】：

- 执行严格单飞（`runtime/src/engine/mod.rs:123`）；worker 在 step 飞行中不服务控制
  命令（`runner/worker.rs:254`）→ 准入被量化到 step 边界。
- 准入 = 2–3 轮异步工单（quote → Reserve → Prefix，
  `pipeline/admission.rs:361-413`、`pipeline/scheduling/resources.rs:33-74`）；冷态
  Reserve 现场捕私有 CUDA 图（`executor/state.rs:100-108`，且要求 `idle()`，
  `state.rs:45`）。
- mixed step = 多个 task 各自串行 replay（`execution.rs:112-141`），无跨 task 融合；
  prefill chunk 跑固定宽度图、成本与 token 数无关；MTP 时另有 `prime_draft`
  （`execution.rs:727-729`）。27ms ≈ 8.4ms prefill replay + draft/verify 开销 +
  readback/host 间隙【构成确证，精确分解待 profile】。
- CudaBackend 未实现 `completion_timing`（`foundation/spi/src/backend.rs:142-144`
  默认 None）→ 成本模型永不校准（`scheduler/src/cost.rs:272-274` fallback 1µs/token）
  → `gpu_budget_us` 是死代码。

**快修 A：准入去串行化**【推断：低风险】

- 压缩工单往返（quote+Reserve 合并；无 prefix 可命中的新请求跳过 Prefix 往返）。
- 冷捕图改为**加载期按容量预热池**（池复用路径已存在，`state.rs:70-98`）。
- worker 单飞模型不动，只降往返次数。
- 附带收益：RM 中 27B long TTFT 的 ~55ms 前后开销、2B hot_long TTFT 1.873 同源自降。

**快修 B：实现 CudaBackend 的 `completion_timing`**【确证是前置】

使能项：不产生直接毫秒收益，但它校准分角色 replay 成本、`gpu_budget_us` 才生效，
后续一切按时间预算的调度调参都依赖它。

**结构修 C：lane 化 prefill（跨序列一次 replay）**

修正 RM§二.2 的预期：调度层合并 execution **不够** —— 后端对非 decode task 仍逐个串行
replay，4×8.4=33.6ms 赢不了 vLLM 的 19.5ms。要赢需跨序列 lane 化 prefill 图（注意力
按 lane 屏蔽；slot decode 图已证明该机制可行）。配套：步级 token 预算 ≥ Σ pending
prefill token（现默认 64，`runtime/src/config.rs:22`，装不下 4×48）。

**预期**（锚点：2B 单次 48-token prefill 8.4ms；native batch4 TTFT ≈47ms vs vLLM
≈19.5ms）：

- 仅 A：2.63x → ~1.3–2.0x（省准入排队，设备时间仍 4 次串行 replay）【推断】。
- A+C：→ ~0.5–0.8x（4×48 token 一次 replay ≈ 8.4–12ms）【推断，依赖 C 成立】。
- 27B batch4 同机制，量级需一次 profile 补数据。

**TPOT 代价门槛**：阶段分离/合并须实测 batch4/hot_long 稳态 TPOT，回退 ≤5% 才合入
（RM§二.2 的遗留要求）。

**已落地（比 A/C 更前置的一步）：prompt chunk 原子化**。profile 显示真正的浪费不在
准入，而在调度把 prompt chunk 切碎：`packing.rs` 的 `candidate()` 把 prefill quantum
再截到 `fair_quantum_tokens`（8），`probe()` 还会递减重试，于是 4 个并发 52-token
prompt 被切成 11/21/31/41 的碎片，而**每个碎片仍然付一次固定宽度图的完整 replay**
（`INFER_CUDA_EXECUTION_PROFILE`：base 120 次 prefill chunk 里有 8 次纯碎片）。
改成 `candidate()` 对 prefill 直接给整块、本轮预算装不下就整轮推迟（不在 token 预算上
切片）；`probe()` 仍保留自 `quantum` 向下的递减搜索，所以设备预算（`gpu_budget_us`）
拒绝整块时，单 token 的 overrun 逃生口仍然有效、不会出现空 StepPlan；预算小于一个
chunk 时也回退旧的 fair-quantum 行为，两种退化路径都不会饿死 prompt。实测：

| 用例 | wall | TTFT | TPOT |
|---|---|---|---|
| 27B batch4 | 1.342 → **1.313** | 2.620 → **2.114**（中位 241→194 ms） | 2.148 → 2.086 |
| 2B batch4 | 1.655 → **1.628** | 2.506 → **2.023**（中位 49→39 ms） | 1.549 → 1.584 |

5 轮交错 A/B（15 个 wall 样本、60 个 TTFT/TPOT 样本），token 序列逐字节一致；profile 里
prefill chunk 全部是整块（51/52/63/64），碎片 replay 消失。单请求用例（short/long/
hot_long）行为**可证等价**：预算 64 = 一个 chunk，切与不切发出的 token 数相同，所以只有
并发 prompt 的场景变化。推迟的 prompt 由既有 aging（20 ms）兜底，不会饿死。

### ③ 27B long TTFT 5.86x —— 64 行特化 GEMM + 宽度分档连锁

**阻塞的绕法**【确证，重读 `workspace.rs` 后的设计】：不需要给 Workspace 加任何可变
字段。`record()` 已从 output 张量推出 `rows`（`resident/nvfp4_gemm/workspace.rs:
109-110`），buffers 本就按 `(rows, columns)` 做 key（`:17`）。tile 选择做成 record
时的纯函数：

```
rows ≤ 16 → 现有 [16,64] kernel（decode 路径不动）
rows > 16 → 新 [64,64] 特化 kernel（prompt 64 行 → 1 个 row block → 权重读 1 遍）
```

同时绕开 RM§3.4 两次失败：clamp 两档不会跌破 kernel 16 行硬下限（避免 +394.6% 的
mmaf 退化）；按 rows 分派没有共享可变状态（避免"最后写入者赢"）。实现 = 复制
`packed` kernel 改输出 tile（`nvfp4_gemm.rs:90-116`，约 20 行）+ `workspace.rs:126`
按 rows 选 partition 常量与 kernel。FP8 兄弟 kernel `matmul`（`fp8_gemm.rs:34-59`，经
`workspace.rs` 的 `record_fp8` 用同一常量）有相同 4x 重读结构，同 PR 镜像修。前置补丁
`docs/patches/quant-gemm-row-tile-m.patch` 已合入并删除。

**预期**【推断，锚点：prompt replay 42.88ms@505GB/s vs verify 24.71ms@947GB/s，同
权重同节点数】：单 replay 42.88 → ~24.7ms；27B long TTFT 5.86x → ~3.8x（8×24.7+55
≈ 253ms vs vLLM ≈68ms）。单独不到 ≤1.10x。

**实测（已落地）**：修法与预期机制不符。按行数分派落地后，隔离 kernel 基准
（`nvfp4_packed_tile_sweep`）显示"权重读 4 遍"确实存在（`[16,64]` → `[64,64]` 在
m=1..64 全部形状上快 1.3–1.5x），但服务侧收益小得多：27B long 的 prompt replay
46.9 → 43.1 ms（−8%），`linear` 桶 27.3 → 23.1 ms（−15%），TTFT 比值 5.91 → 5.45
（−8%）；short 1.30 → 1.22，batch4 2.53 → 2.39，wall 回退 0.3–2.8%，token 逐字节
一致。**不要把 tile 全局放宽到 `[64, 128]`**：隔离基准里它在每个行数都更快
（0.66–0.87x），服务侧却全局回退（short wall 1.14 → 1.30、batch4 TPOT 1.86 →
2.05）。隔离 GEMM 基准缺 launch 间隙与并发形状，不能替代服务矩阵。

**否掉的剩余假设**：prompt replay 的其余时间不在权重流量上 —— 同一 replay 里
`delta`（48 层线性注意力的循环状态）占 16.2 ms，`linear` 只降了 15%；12-lane 的
target verify 里 `linear` 19.9 ms、`delta` 4.2 ms、`conv` 2.3 ms。带宽只有峰值
~40–50%，属延迟遮蔽问题（杠杆 A），不是重读问题。

**连锁（③′）：prompt 图宽度分档 —— 必须重测 RM§四 被否的 128-lane 实验**。该实验是
在坏 GEMM 上做的（128 行 = 8 个 row block = 权重读 8 遍），结论已失效。③ 修好后带
宽 bound 的宽图边际成本骤降（更宽仍一遍权重）。目标形态：宽度分档的多张 prompt 图
（如 64/256，按 prompt 长度路由）。预期：27B long TTFT → ~1.5x（2×24.7+55 ≈ 105ms）；
2B long 3.26x → ~1.2x（507 token 从 4 次 replay 到 1 次 512 宽）。**27B long 要到
≤1.10x 还需 ②A 把 ~55ms 前后开销砍到 ~18ms 以下 —— 该格需要 ③+③′+②A 同时落地。**

**实测（已落地）**：不对。③′ 的收益**不**来自更宽的 bound，而来自"**按 chunk 长度路由到最窄够用的图**"。
一次 prompt chunk = 一次固定宽度 replay，成本与 chunk 内 token 数无关，所以
128 宽图对 511-token prompt 是省 4 次 replay，对 51-token prompt 是白跑 60% 的 lane。
两次实测：27B 全程 128 宽（`--max-num-batched-tokens 128`）long TTFT 5.72 → 4.98，
但 short 1.18 → 1.47、batch4 1.95 → 2.41（+24%，与 loading 注释里记的三格回退同源）。

落到 `ProgramWeights::{prefill_width, narrow_prefill_width}` + `DeviceProgram::prompt_narrow`
后：**捕获宽窄两张 prompt 图，按 chunk 长度选图**（≤ 窄宽走窄图，否则走宽图），
engine 的 chunk 仍取宽宽（`prefill_chunk_tokens = 128`）。27B 选到
`prefill_width=128, narrow=64`，2B 仍是单张 128 图（窄宽=宽宽，不额外捕获）。
实测（4 轮交错 A/B 池化，24 个 wall / 96 个 TTFT·TPOT 样本，token 逐字节一致）：

| 27B 用例 | wall | TTFT | TPOT |
|---|---|---|---|
| long | 1.914 → **1.686**（−12%） | 5.973 → **4.717**（−21%） | 1.383 → 1.321（−5%） |
| batch4 | 1.240 → 1.198（−3%） | 1.953 → 1.972 | 1.980 → **1.837**（−7%） |
| short | 1.253 → 1.202（−4%） | 1.210 → 1.200 | 1.260 → **1.201**（−5%） |
| hot_long | 0.992 → 0.992 | 0.368 → 0.366 | 1.249 → 1.244 |

2B 配置完全未变（单图），其 ±5% 摆动即噪声下限。显存：27B 两张图并存后 resident 22.9 GiB /
state_budget 4.8 GiB，与单张 64 图（23.0 / 4.7）持平，准入并发不变。
`MAX_PREFILL_LANES` 仍是 128；再宽需要同时放开 `loading/mod.rs` 的宽度白名单。

### 合成预期（全落地后，推算）

| 用例 | wall 现值 | wall 预期 | 依赖 |
|---|---:|---:|---|
| 2B short | 0.974 | ~0.97 | 不动 |
| 2B hot_long | 1.084 | ~1.0–1.05 | ②A |
| 2B long | 1.137 | ~1.0 | ③′ |
| 2B batch4 | 1.490 | ~1.0–1.2 | ②A+①（+②C 更低） |
| 27B hot_long | 0.951 | ~0.95 | 不动（① 不得误伤 batch1 MTP） |
| 27B short | 1.150 | ~1.05 | ③ |
| 27B long | 1.761 | ~1.3–1.4 | ③+③′；仍红，TPOT 1.272 是 decode 侧问题 |
| 27B batch4 | 1.201 | ~1.0–1.1 | ①+②+③ |

约 7/8 格进 ≤1.10x；27B long 残留为下一个主攻目标（需单独的 decode 侧根因轮）。
下一轮 bench（perf-r41）后用实测绝对值替换本表。

### 模型通用性（换模型是否要重改）

kernel 工作在"量化格式 × 张量形状"层面，无模型特定假设：K 是 const generic 按
columns 加载期自动编译；prefill_width/batch_width 由加载期 arena 预算推导
（`loading/mod.rs:252-279`）；③ 的分派依据 rows 同为形状驱动。

| 新模型 | 是否要动 |
|---|---|
| 同结构同 NVFP4 不同尺寸 | 零改动 |
| 同结构 BF16 | 不走此 kernel，无影响 |
| 同结构 FP8 | 镜像修 `matmul`（已含在 ③ 内） |
| 不同结构（MoE/hybrid 等） | kernel 仍通用；新 GEMM 路径（如 grouped GEMM）需单独检查同类问题 |

tile 常量绑定的是硬件（tensor core MMA 形状、带宽），换 GPU 架构才需重调。换模型
要做的只是常规复验（数值门 + bench），不是适配。

## 二、超越（<1.0x）：五个杠杆

追平后双方都是"每步一遍权重"，赢家 = 带宽利用率更高、每 token 步数更少的人。

**前置基线核查**【待测，一次配置检查】：bench 里 vLLM 是否跑同一 NVFP4 checkpoint。
若它实际跑 BF16，我们手握 4x 权重流量优势却没赢，余量比矩阵显示的更大。

- **杠杆 A：decode GEMM 带宽 53% → 75–80%**（最大、最通用）。verify 947 GB/s 是该卡
  峰值（~1.79TB/s）的 53%；Marlin 级 W4 kernel 同档卡可到 75–85%。方向：K 向
  cp.async.bulk/TMA 双缓冲、更宽 N tile 增加在途字节（现 weight tile [64,128]、每 k
  步仅 4KB，延迟遮蔽不足是 53% 的典型成因）、quantize 与 GEMM 的依赖链流水化。
  **不要走 split-K（RM§四 已否）**。预期 TPOT 全面 ×0.7 量级。【推断，工程量大但无
  未验证假设】**外部佐证**：同栈的 HF grout（cuTile Rust）在同档 5090 上 decode 达
  76% 带宽（171 tok/s × 8GB fp16），论文报告其 GEMM 达 cuBLAS 98% —— 带宽余量真实
  存在，cuTile DSL 不是天花板（arXiv:2606.15991）。**委托对照路**：quantize 本来就
  是独立 kernel（`workspace.rs:115-123` 与 `packed` 分开 record），换 cuBLASLt
  NVFP4（Blackwell 原生 block-scaled）零融合损失，Rust FFI 即可、不引入 C++；按形状
  赛跑进 `tuning.rs` 成对测量，差 <2% 用二进制、差得多留自研。regime 判断【推断】：
  decode（M≤16）带宽 bound 可追平；prefill（M=64，AI≈256 FLOP/B）在分水岭上、部分
  算力 bound，cuBLASLt 可能保有 10–20% kernel 级优势 —— 用系统级优势覆盖，不死磕。

  **实测（2026-10，RTX 5090 / CUDA 13.4，冷 L2 协议）**：`grout` 的做法是**线性层全部
  交给 cuBLAS**，自己只用 cuTile 写 attention / norm / rope / KV / argmax（`src/cublas.rs`，
  `gemm_ex` + TN + fp16 in/out；`GROUT_CUBLAS_COMPUTE16` 在 sm_120 上默认用 **fp16 累加**，
  并带 per-arch/per-shape 的 tuning record）。grout 的对比是同并发下单请求吞吐略胜 vLLM
  （B200 Qwen3-32B `request_gen_tps` 79.6–80.1 vs 78.8–79.2），**它不是 serving 吞吐引擎**，
  所以"像 vLLM"= 单请求快路径做到极致，与我们的 batch4/并发格不是同一件事。

  照抄 `杠杆 A` 的委托路做了一次成对测量（同权重布局、不重排、同一 L2 flush 协议）：

  | 形状 (m,n,k) | 我们出厂 tile | 我们最好 tile | cuBLASLt NVFP4 | 倍率 |
  |---|---|---|---|---|
  | 12, 17408, 5120 | 0.0630 ms `[16,64]` | 0.0427 `[64,128]` | **0.0369 ms** | 1.71x |
  | 64, 17408, 5120 | 0.0426 `[64,64]` | 0.0422 `[64,128]` | **0.0389 ms** | 1.10x |
  | 12, 5120, 17408 | 0.0648 `[16,64]` | 0.0524 `[16,128]` | **0.0389 ms** | 1.67x |
  | 64, 5120, 17408 | 0.0648 | ~0.053 | **0.0410 ms** | 1.58x |

  即：**窄行（decode/verify）形状赢 1.6–1.7x，prompt（M=64）只赢 1.1x**；TPOT 是最大受益
  项（verify replay 线性 19.9 → ~12 ms 量级），prefill 收益有限。热 L2 下 cuBLASLt 看着有
  3x，是 L2 假象，必须用冷 L2 协议比较。

  **落地阻塞点【确证，查 cuBLAS 文档 + 实测】**：`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`
  的 scale 不是线性 `[n, k/16]`，而是 **128×64 分块 + swizzle** 布局（"a single tile of
  scaling factors is applied to a 128x64 block"，offset 公式
  `(sf_inner + sf_outer*sf_inner_dim)*128`，起始地址 16B 对齐，且**不支持转置**）。用线性
  scale 布局实测：k=64 全对、k=512 只对一半、n=5120 时只累加了部分 K 块 —— 与文档一致。
  所以委托实现 = ①加载期把权重 block-scale 重排进该 swizzle；②`quantize` kernel 直接按
  swizzle 写激活 scale（或再加一个重排 kernel）；③`TN` + `COMPUTE_32F` + `CUDA_R_32F`
  scale type；④按形状赛跑进 `tuning.rs`。这是有确定规格的工程量，不是未验证假设。
  FP8 那条更简单：`CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F` 正好是"每行 A 缩放 ×
  每行 B 缩放"，与 `record_fp8` 的 per-token × per-channel 一一对应，无需 swizzle。
- **杠杆 B：MTP 融合成一张图 + 动态 lane**。27B 是 4-bit 权重、decode 相对优势却没
  跑赢 2B —— 最可能是三次 replay + 同步 readback 吃掉了量化红利。propose/verify/
  catch_up 一次捕获、设备侧采样与接受判定；按接受率动态调 lane 数。vLLM 的 spec
  开销是其公认弱项，图捕获引擎做这件事结构性更便宜。与 ① 是同一投资的两面：① 决
  定"何时不做"，B 决定"做的时候多便宜"。【推断】
- **杠杆 C：FP8/FP4 KV cache**。hot_long TPOT 1.187 的税在 KV 带宽；FP8 KV 减半注意
  力读流量，目标 <0.9。需过数值门。【推断】
- **杠杆 D：固定开销 55ms → ~10ms**。②A 覆盖准入部分，剩余是 readback/采样/
  detokenize 链路。short 类 TTFT 稳定 <1.0，配合 ③′ 长 prompt 压到 ~1.0。
- **杠杆 E：离线 autotune 常量**。基建已存在 —— `crates/backend/cuda/src/tuning.rs`
  （成对图测量、tile 搜索空间、5e-4 数值匹配容差、胜者须超基线 2% 才采纳、结果持久
  化）；本杠杆是把搜索空间扩展到 `QUANT_GEMM_TILE`、lane 宽度、group size 等更多常
  量，按 GPU 型号在打包期运行。典型 5–15%。IR 层目前只有 validate/plan_lifetimes
  （`foundation/ir/src/dataflow.rs`）、无融合重写 pass；融合以手工 kernel 承担。若未
  来要补，正确形态是闭集 rewrite 规则（RMSNorm+residual、SiLU gate 等），跨后端共
  享 —— 不做通用 auto-fusion 编译器（op 空间是闭集，ROI 为负）。

远期（研究性质，先记账）：prefill/decode 双流共跑（prefill 算力 bound、decode 带宽
bound，SM 分区；现 worker 严格单流）；device-side graph launch 零 host 间隙 decode。

## 三、调度侧路线（生产并发）

前提见 §〇.2：预算单位是 replay 槽位。近/中/远三程：

**近程（直接改 batch4 两格）**

1. 双预算打包：prefill/decode 预算分离；prefill 只在 ITL 余量内发射（布尔决策，
   不切碎）。改 `packing.rs:310-321` 的 min 链与 `policy.rs:186-216` 的 role_bias。
2. 准入微合批窗口：到达后等 2–5ms 攒批，配合 ②A 把"串行准入"变"一批准入"。
3. 懒捕获全部改启动预热：slot 池现在首次连续 ≥2 个 decode 才懒捕获
   （`execution.rs:173-229`），生产上是周期性毛刺源；启动时按容量预捕全部图。
4. ① 自适应 MTP。

**中程（生产并发的命门）**

5. **并发 >4 的 slot 悬崖调查**【待测】：`max_num_seqs=16` vs `CB_DECODE_SLOTS=4`，
   第 5 个并发 decode 起溢出到每请求私有图串行。RM§四 已证明盲目加宽（4→8）更慢
   （8 路 batch 4.10s→6.6s）且根因未查。两条路：查清 8-lane 回退根因【推断：更宽
   图的状态读写/readback 成本】，或分层批（多组 4-lane，请求固定分组）。
6. 前缀亲和调度：按 prefix 树对排队请求聚类，把 hot_long 0.369 的前缀优势从"碰运
   气"变"人为制造"。
7. 长度感知排序：`max_tokens` 提示或预测器做 SRPT 式排序，降 TTFT p99。
8. 饱和降级：准入控制 + 背压 + 优先级类间抢占（restore 图已有 KV 换出底子）；租户
   公平与限速（`policy.rs` 已有 tenant tiebreak）。

**远程（结构性答案）**

9. **PD 分离**（prefill/decode 实例级拆分 + KV 传输）：治 TTFT/ITL 互相干扰的终极
   方案。本后端严格单飞 + 图捕获的架构反而让它更干净（每实例只跑自己的图集，无动
   态形状问题）。②C 是通往它的踏脚石。
10. 重叠调度扩展：`pipeline/scheduling/overlap.rs:23-71` 已有 prepare_next 预规划；
    下一步让准入在 step 飞行中完成（控制通道与飞行解耦），消掉最后的 step 边界量化。

## 四、执行顺序与验证

| 阶段 | 内容 | 验证门槛 |
|---|---|---|
| 0 | §五 观测指标落地；① 的两个测量实验 | 不改代码；产出 batch4 每步 replay 数 |
| 1 | ②A + ②B + ③（kernel 特化，小改动可并行）+ §六 预占阈值与预算瀑布（消费端先只接 KV/前缀池） | 27 host 测试 + 数值门；③ 单 replay ≤28ms |
| 2 | 全矩阵重测（perf-r41），实测替换 §一 推算表 | 已赢格子（2B short/decode、27B hot_long）零回退 |
| 3 | 杠杆 A + B | TPOT 格 ×0.7；MTP 接受长度不降 |
| 4 | ②C lane prefill、③′ 宽度分档（大改动，视阶段 2 结果排期；图资产经 §六 瀑布资助） | ② 的 TPOT ≤5% 回退门槛 |
| 5 | 调度中程项、杠杆 C、E | p99 与饱和负载 soak（过载面见 [serving-stability.md](../../guides/serving-stability.md) §五.8） |
| 6 | §七 profile 自适应（auto 默认档） | 同一配置双负载矩阵复测，两侧 ≤5% 差距 |
| 7 | §八 不变量收口（不建 TP，只收口假设） | 8 条不变量逐项过 review；engine 无直接设备引用、动态决策只经 SignalSnapshot |
| 8 | §九 H200 接入与 FP8 三制度（与阶段 2–5 可并行，依赖 H200 机型到位） | mma-probe 证据落盘 + wgmma lowering 数值门 + H200 全矩阵基线 |

统一用 `tools/bench/serve-compare.py` 标准矩阵 + `INFER_CUDA_EXECUTION_PROFILE`
系列开关；任何比值遵守 RM§六（同源重测，不拼来源）。

## 五、观测基建（一切调参的前提）

进 metrics 的一等指标（现状：都没有）：

1. 每图带宽 %（kernel 工作的仪表盘）；
2. 每 token replay 数（① 与 slot 悬崖的直达指标）；
3. MTP 接受长度（① 与杠杆 B 的反馈信号）；
4. 每 step 队列深度与分角色等待时间、ITL 直方图（p50/p99）、KV/arena 占用率
   （调度调参依据）；预算瀑布与各池水位（§六）。

## 六、显存预算：预占阈值（空间换效率）

**共识**：单模型单卡是主要部署形态，显存不折算成能力就是浪费。机制已备（加载期捕获 +
`loading/budget.rs` 准入记账 + 运行期零分配），缺的是"余量 → 能力"的预算策略与一个
运维可调阈值。

**现状**【确证】：`device.rs:96-145` 的预算全是 `total_memory_bytes` 的纯函数 ——
headroom total/32、arena total/64（5090 上 512MiB/请求）、checkpoint total/16、
resident_states total/512MiB clamp [4,64]；消费端被常量封顶（`CB_DECODE_SLOTS=4`、
`CB_SLOT_TOKENS=4096`、`MAX_CAPACITY_TOKENS=32768`）。27B NVFP4 权重 ~14GB 之外，
~15GB 余量闲置：不变并发、不变上下文、不变图资产。

**参数设计**：

- `memory_utilization`（runtime.json / CLI，对齐 vLLM `--gpu-memory-utilization` 语
  义）：单实例可动用总显存比例，**全包口径**（权重 + 图 + 各池），serve 默认 0.90。
- per-model-instance 语义：多模型共卡（如 LLM + embedding）各自显式下调；加载期总
  claim 超额直接报错 —— fail at load，不 OOM at runtime。
- 除阈值外不加旋钮：瀑布优先级固定，避免 vLLM 式旋钮税（参数对齐的原则是能力平价，
  不是旋钮平价）。

**预算瀑布**（加载期顺序）：

1. claim = total × utilization；
2. − 权重（加载实测）；
3. − headroom 下限（`device.rs` 现有派生保留为 floor）；
4. − 图资产与 workspace（按实际捕获计；③′ 分档图、②C lane 图在此入账）；
5. 余量 → 状态预算：先 recurrent 池（目标并发 × 每序列固定字节），其余全部 → KV 页
   池；前缀保留**共享** KV 池（retention 策略，不独立设池）。

**常量 → 派生改造清单**：

- `MAX_CAPACITY_TOKENS=32768` → min(池 ÷ 目标并发, 显式 max_model_len)；
- `CB_SLOT_TOKENS=4096` → 随池派生；
- `CB_DECODE_SLOTS=4` 维持不动，预算只资助 slot **组**（分层批，§三.5），禁止加宽
  lane（RM§四 4→8 回退）；
- arena/checkpoint 除数进 tuning 搜索空间（杠杆 E），不再拍常量；
- `sequence_budget()`（`loading/budget.rs`）保持为准入执行器，改核对口径为 claim 后
  的池。

**27B hybrid 实账**（`examples/qwen3.8-27b/config.json`：64 层 = 16 全注意力 + 48
线性，kv_heads 4、head_dim 256）：

| 项 | 数值 |
|---|---|
| KV | 16 层 × 2 × 4 × 256 × 2B = **64 KiB/token**（FP8 减半） |
| recurrent 状态 | 48 层 × 48 vheads × 128×128 × 2B ≈ **72 MiB/序列**（与长度无关） |
| 256k 上下文 KV | bf16 = 16 GiB；FP8 = 8 GiB |
| 0.90 claim 余量（5090 32GB） | 28.8 − 权重 ~14 − 图/arena ~2.5 ≈ **12 GiB** |

结论一：**对齐 `--context-length 262144` 在 32GB 卡上强依赖杠杆 C（FP8 KV）** ——
bf16 的 16 GiB 放不进 12 GiB 余量。结论二：64 并发的 recurrent 池 = 4.6 GiB，这是
hybrid 模型的固有税，瀑布让它显性化（对应 SGLang `--mamba-full-memory-ratio` 的维
度，但由瀑布自动定，不做旋钮）。

**收益排序与门槛**：

1. KV/前缀池扩容 + 容量帽派生化 —— 现在可做，确定性收益（hot_long 前缀优势放大、
   256k 准入解锁）。指标：前缀命中率（`--enable-cache-report` 对应物）、可准入的
   并发 × 上下文。
2. 图资产扩容 —— 门禁：③ 落地 + ③′ 重测（`loading/mod.rs:246-251` 的实测陷阱：
   上次用空闲显存资助 128-lane 图，四格回退三格；当时是 4x 重读的坏 GEMM）。

**验证**：加载期打印瀑布 + `/native/v1/runtime` 暴露 + 池水位进 metrics（§五.4）；
bench 加三类：多 distinct prefix 的 hot_long 压力、256k 准入选通、并发 1→64 扫描；
现有矩阵零回退。

## 七、部署 profile：默认兼得（auto），可选钉死

**原则**："单机低延迟 vs serving 吞吐"不是用户的选择题。默认 `auto` 档下引擎按
负载信号在延迟/吞吐谱上自动滑动 —— 队列空时自动深投机、准入短路（延迟行为）；
队列涨时自动降 draft 深度、开攒批窗口、prefill 让路 ITL（吞吐行为）。
`latency` / `throughput` 两档只是把曲线端点钉死（SLO 场景用），**不是另一套代
码路径**。

**诚实的边界**：饱和点上真兼得违反物理 —— 高并发时投机 verify 与在跑请求抢算
力，必须让位（① 的自适应 MTP 就是这个让位机制）。"兼得"的定义是**每个负载点
的行为自动正确**，不是同时满足两个极点。

**信号**（全部依赖 §五 观测基建，阶段 0 前置）：队列深度、ITL 余量、MTP 接受长
度、每 token replay 数。

**执行器**（全是运行时策略，无 kernel 改动）：

1. MTP 深度/lane 动态化（① 的推广：阈值 → 连续曲线）；
2. 准入微合批窗口 0–5ms 随队列深度伸缩（§三.2 的参数化）；
3. prefill 发射布尔门（§〇.2 共识的直接应用）；
4. 深投机图/设备侧闭环的启用与挂起。

**迟滞**：所有阈值双侧化 + 最小驻留时间，防止 MTP 深度在边界抖动。

**图资产与显存**：加载期尽量捕获两模式图资产的并集；装不下时瀑布内降级顺序为
slot decode 图（必需）→ prompt 宽度分档（③′）→ 深投机变体 → 设备侧闭环；KV 池
地板优先于投机图。27B/32GB 的 ~12GiB 余量（§六 实账）大概率装得下并集；装不下
的硬件组合靠秒级冷启动做快速重载切换。

**前置依赖**：①（自适应 MTP 机制前身）、②B（`completion_timing` 成本校准）、
§六 瀑布（资产入账）、`CB_DECODE_SLOTS` 加载期派生化（§六 清单）。

**验证**：同一二进制同一 `auto` 配置连跑两类负载 —— batch1 延迟矩阵 + 并发
goodput 矩阵；两侧与各自钉死档的差距 ≤5%，否则自适应曲线不合格。profile 三档
的切换不重编译、不重捕图（并集已捕获时）。

## 八、并行化路线与防重构不变量

**定位**：数据中心主战场。吞吐的第一答案是 DP 不是 TP —— 模型单卡装得下时，
多副本 + 路由零通信开销，永远优先；TP 解决容量（装不下）或延迟（分摊权重字节）；
EP 等 MoE 落地后再谈。

| 阶段 | 内容 | 前提与工作主体 |
|---|---|---|
| 0 | DP + 路由层 | 现可做；实例互不知情，路由按前缀指纹粘会话 |
| 1 | TP=2/4 节点内（NVLink 机型） | rank 运行时壳、图内 all-reduce、加载期分片、hybrid mixer 状态分片、§七 信号全局化 |
| 2 | PP（可选） | 点对点通信最简单，但有流水气泡；吞吐场景与 TP 组合 |
| 3 | EP | 前置 = MoE 本身（grouped GEMM 单卡先行），最远 |

**kv_heads=4 → 本模型高效 TP 上限为 4**（attention 按头分片，TP>4 需复制 KV），
加载期强校验。5090 无 NVLink：TP 的开发/CI/验收需 NVLink 机型，硬件预算先行。

### 防重构不变量（现在做便宜，以后做是重构）

1. **逻辑后端**：engine 只经 `BackendProvider` 与"逻辑后端"对话，任何代码禁止
   假设 1 进程 = 1 设备；TP 时同一 trait 由 rank-group 实现（submit/poll/
   reserve 向各 rank 广播）。排查点：runtime/pipeline 与 executor 的直接设备引
   用。
2. **CollectiveOps 抽象**：图构造中的集合通信走 trait —— 单机 = no-op 直通，
   TP = NCCL（C API/bindgen，同 cuBLASLt 路径）。**单机与 TP 共用同一张图构造
   代码**，collective 节点只是实现不同。这是"不重构"的核心保险。
3. **决策点唯一**：一切动态决策（§七 的 MTP 深度、准入窗口、prefill 门）只读
   `SignalSnapshot`；单机 = 本地采样，TP = rank 归约/rank0 广播。执行器禁止读
   rank 本地状态，否则 TP 下决策发散 = collective 死锁。
4. **分片是加载期视图**：权重命名钩子（`weight_prefixes`/`anchor_slot`）之上加
   shard view（column/row 切 = 形状变换）；provider 对 TP 无感，分片计划由后端
   按 tp_degree 声明进 ExecutionProgram。
5. **KV/状态按 rank 参数化**：heads_per_rank 进 KV manager 参数而非从 config
   直取；mixer 状态分片规则进 IR 的 StateKind 声明（linear attention 按 head、
   conv 按 channel）。
6. **确定性瀑布**：§六 预算瀑布必须是（设备 identity, 模型, 配置）的纯函数 —
   TP 各 rank 独立算出同一结果、图资产集逐字节一致，否则 capture 顺序发散即死
   锁。
7. **配置与观测预留**：runtime.json 预留 parallelism 度数（dp/tp/pp）；metrics
   标签预留 rank 维度；`memory_utilization` 保持 per-instance 语义（DP = N 个
   互不知情的实例）。
8. **DP 前缀亲和钩子**：实例暴露前缀指纹/命中率 API（复用 §三.6 信号）；DP 不
   需要实例间共享任何状态。

**已知冲突点（提前记账）**：

- §七 auto profile 在 TP 下每个动态决策 = 一次全局同步，延迟模式下 TP>1 的投机
  加深收益要打同步折扣。
- 杠杆 B 的图内接受判定天然 rank 一致（logits all-gather 后同 seed 确定性），
  不冲突。
- 单 rank 故障 = 全组 hang（NCCL 超时）：[serving-stability.md](../../guides/serving-stability.md)
  需补分布式故障章。

**现在不做**：不建 NCCL、不建 rank shell、不写分片实现 —— 只收口假设。

## 九、硬件与精度矩阵：H200/FP8 数据中心线

**定位**：双硬件线 —— 5090/NVFP4（消费单机）与 H200/FP8（数据中心）。H200 是
sm_90 Hopper，**无 FP4 Tensor Core：NVFP4 在 H200 上没有硬件路径，FP8 即正确精
度**。provider 的 `PrecisionPolicy` 按设备能力选路是现成机制（"支持原生 FP4 与
支持某个高效 Tensor Core tile 是两件事"），不需要新架构。

**现状**【确证】：FP8 GEMM kernel 已存在（`nvfp4_gemm.rs:14-55` 的 `matmul`，
与 `packed` 共用常量），③ 的 4x 重读修复同 PR 镜像到它；Hopper 转换路径存在但
**未实机验收**（cuda-performance.md §能力与精度）；tuning 表按设备 identity 索
引，新硬件零代码接入、首轮自动测量。

**工作清单**：

1. **H200 硬件接入**：`cuda-mma-probe` 能力验收、证据落 `benchmarks/capabilities/`、
   `baselines/tuning.json` 加 H200 基线。CI/bench 机型预算先行（软件就绪度无法
   在无硬件时验证）。
2. **lowering 重验**：wgmma（sm_90a）与 tcgen05（sm_120）是两条 lowering 路径，
   数值门 + `inspect-cutile-cache.py` SASS 检查全量重跑；TMA 在 Hopper 是主场，
   杠杆 A 的双缓冲方向直接受益；PDL sm_90 原生支持（`mlp/pdl.rs` 无需改动）。
2a. **chunked recurrent prefill（本轮实测：奖品很大，但现在是错的）**。27B `long` 用例
   （4×511 token）的 prefill 是 **~900 ms 设备时间、不是主机开销**：`INFER_CUDA_EXECUTION_PROFILE`
   显示 `prefill_chunk` 每块 63 ms（mtp0 也 63 ms，所以与投机/readback 无关），
   `INFER_CUDA_PREFILL_PROFILE` 按算子拆开是 **delta 345.8 ms（38%）+ linear 336.0 ms（37%）**
   + embedding 93.5 + attention 51.9。delta 是逐 lane 读写整块状态：48 层 × 128 lane ×
   3.146 MB × 2 = 38.7 GB/块，38.7 GB ÷ 1.79 TB/s = **21.6 ms/块**，与实测 21.6 ms 完全吻合
   —— 它已经跑在 100% HBM 带宽上，唯一出路是少搬数据（把状态留在寄存器里跨 lane 复用）。

   仓库里正好有这条路（`recurrent_prefill`，状态在寄存器 tile 里跨 lane 迭代），但它对
   **激活量化的模型被关掉**（`input_scales`/`fp8_inputs` 非空即返回空 workspace）。本轮把门
   打开实测：**long TTFT 4.565 → 3.433（−25%）、wall 1.621 → 1.460（−10%）、batch4 TTFT
   1.971 → 1.659（−16%）**，但 **21/21 条 token 序列全部不同**。

   于是用仓库自带的全模型数值门复现：
   `INFER_TEST_MODEL=/home/r/models/Qwen3.8-27B-NVFP4 INFER_TEST_TOKENS=<json> INFER_TEST_PREFILL_WIDTH=128
   cargo test --release -p infer-backend-cuda --features cuda checkpoint_chunk_recurrence_matches_legacy_logits -- --ignored`
   → **`hidden relative_l2=0.0436, max_abs=0.4986`（门限 1e-4 / 1e-2）**。原注释写的是
   "FP32 rounding slightly"，实测是 **4.4% 的语义级偏差**，所以这是正确性门、不是精度取舍。
   二分：把卷积强制回逐 lane（`conv_prefill::record` 返回 false）后偏差**逐位相同**
   （0.04364158），说明卷积路径在该模型上根本没被选中，问题在 **chunked Delta 本身**；
   而合成测试 `chunk_delta_matches_independent_recurrence` 在同一几何（kh:vh=1:3、dim 128、
   lanes 128、count<lanes）下 2e-5 通过 —— 所以缺陷在 capture 喂给 kernel 的东西，不在
   表达式树（两侧逐算子结构一致）。**下一步**：对比 `capture_state.rs:74`（逐 lane：扁平
   1-D view，`[head]` 索引）与 `recurrent_prefill/workspace.rs:111`（chunked：显式
   `view([lanes, 2*KH+VH, D])`，`[lane, head, 0]` 索引）两处输入 view 的布局假设。

   **第 6 轮把范围收窄到"capture 的输入绑定"**。(a) cuTile JIT cache 不背锅：`l2_key`
   是对 tileiras 字节码做 SHA-256，kernel 体一改 key 就变，我这轮的 kernel 编辑都是真生效的。
   (b) 逐个排除：masked lane 写零（去掉后偏差**逐位相同**）、chunked 卷积（把 delta 换回逐 lane、
   卷积保持 chunked → **偏差恰好 0.00000000**，即卷积精确等价）、metadata 布局（chunk 的
   `[position, count, state_offset, 0]` 与逐 lane 的 `state_pos = pos + state_offset` 语义一致）、
   共享 scratch（改成每个 delta 节点独占 scratch，偏差仍逐位相同）、prefill 宽度（32/64/128
   三档偏差完全一样）。(c) **新增测试证明两个 kernel 各自都是对的**：
   `chunk_delta_matches_independent_recurrence` 补上模型几何 (kh16/vh48/dim128/lanes128) 后通过；
   新增 `per_lane_delta_matches_independent_recurrence_at_model_geometry`（此前**只有 chunked
   被测过**，shipping 的逐 lane 路径从没在这个几何上对过参考），worst state 6.9e-8、
   worst out 8.7e-9。既然两边在相同输入下都精确，那 5% 只能来自 **capture 喂进去的输入不同**
   —— 下一步是 dump chunked capture 的 `qkv/beta/alpha/state` 绑定与逐 lane 路径逐值对比。

   **同轮落地的小改动**：MTP 草稿的 priming 过去按固定 `PREFILL_LANES`(32) 切块，
   511-token prompt 变成 16 次 replay/请求（profile 里 64 条 23-node 记录 × 2.1 ms ≈ 134 ms）。
   草稿的 prompt 图其实能捕到 128 宽，现在按 `spec.program.prefill_width()` 切块：
   **long TTFT −1.7%（4 轮 long-only 池化，0.9833）、wall −0.6%，token 4 轮逐字节一致**；
   常驻显存与池化宽度不变（仍 width 4、0 个 disabled step）。收益远小于 134 ms 的原因是
   priming 与其它请求的 prefill replay 在 GPU 上重叠，只有约 0.3 ms/次落在关键路径上。

   **第 7 轮：把"4.4% 漂移"这个数字本身证伪了，并给出真实的分层误差。**
   `checkpoint_chunk_recurrence_matches_legacy_logits` 里两次 `prefill_batch` 之后直接比较
   host 向量；在比较前插一次 `reclaim_barrier()`（device synchronize）后，**单 token 用例
   全部 state 与 readout 都是 `relative_l2 = 0.00000000`** —— 原来的 0.0436/0.0529 是把
   放大后的端到端数字当成了逐层误差。加上同步后逐 state 打印，真实情况是：

   | 位置 | 误差 |
   |---|---|
   | 第一个 delta state（TensorId(25)） | **9.5e-6** |
   | 后续 delta states | 2.4e-3 … 1.7e-3 |
   | 下游 conv states | 2.1e-2 … 6.5e-2 |
   | 单 token、同几何 | **恰好 0.0**（每个 state 与 readout） |

   即：**单 lane 逐位一致，多 lane chunk 才差**，是 lane 循环里的 f32 重结合，经 64 层 +
   激活量化放大成不同的贪心续写（端到端 21/21 序列改变、long TTFT −25%）。这一轮同时确认
   chunked 卷积路径在该模型上**从未被选中**（`weights.constants.contains_key` 命中），
   所以那些"conv state 漂移"是 delta 漂移的下游后果，不是卷积的问题。

   新增的测试资产（都是这轮或上轮加的、可复用的）：
   `per_lane_delta_matches_independent_recurrence_at_model_geometry`（补上 shipping 路径
   从未被测过的模型几何参考）、`chunked_recurrence_matches_per_lane_at_model_geometry`
   （参数化几何的集成 fixture，~2 s 复现）、`chunk_delta_transpose_permutes_head_and_lane`
   （scratch→输出的置换）、以及 model_check 里的 device 同步与 state 一致性诊断。

   **结论**：门继续关着是对的，但理由从"语义级 4.4% 偏差"改成"lane 循环的 f32 重结合"。
   修法是让 chunked kernel 在 lane 循环内逐位复现逐 lane 的 store/load 序列（布局/metadata/
   scratch/transpose 都已对独立参考验证过），而不是找布局 bug。收益仍是最差格 −25% TTFT。

   **第 8 轮：先给出可用的开关与实测，再谈 e2e。** 定位到差异是**纯 kernel lowering**：
   新测试 `chunk_and_per_lane_delta_agree_exactly` 在**逐字节相同输入**上直接跑两个 kernel
   （无 capture、无 program），state `max_abs = 9.3e-9`、output `7.5e-9`（目标 0），
   0.67 s 一轮 —— 这就是修 lane 循环的快门。把 q/k/v/beta 改成与逐 lane 完全相同的扁平
   `[D]`/`[1]` 分区+索引后差异不变，所以不是载入形状，而是循环体本身的收缩/调度。

   在此前提下把 chunked 路径做成**显式 opt-in**（`INFER_CUDA_CHUNKED_RECURRENT=1`，
   默认仍是逐 lane、逐位不变），27B 两轮实测（ratio native/vLLM）：

   | 用例 | wall | TTFT | TPOT |
   |---|---|---|---|
   | long | 1.619 → **1.455**（−10.1%） | 4.513 → **3.411**（−24.4%） | 1.271 → 1.222 |
   | short | 1.144 → 1.240（+8.4%） | 1.274 → **1.086**（−14.8%） | 1.134 → 1.253（+10.5%） |
   | batch4 | 1.593 → 1.582 | 2.312 → **2.079**（−10.1%） | 1.681 → 1.651 |
   | hot_long | 0.950 → 1.009（+6.2%） | 0.369 → **0.330**（−10.6%） | 1.186 → 1.284（+8.3%） |

   即：**TTFT 全面下降 10–24%**（预填充收益），但 short/hot_long 的 TPOT/wall 反而退 ~8–10%。
   后者经 3 轮复测确认不是噪声（short wall 1.084 / TPOT 1.105、hot_long 1.062 / 1.083）。

   **一处纠正**：先前把 TPOT 回退归因于"verify 图也切到了 chunked"是错的。`Workspace::new`
   只在 `batch.rs:364` 的 `build32` 里被调用，而 `build()` 仅在 `width >= PREFILL_LANES` 时
   分派到 `build32`（`batch.rs:265`），所以 chunked 路径**本来就只作用于预填充图**；batch/slot/
   verify 图走 `Workspace::default()`（逐 lane）。因此 short/hot_long 的 wall/TPOT 变化目前
   **没有解释**，是一个待查项，而不是已知机制。

   默认保持关闭的理由因此收窄为两条：token 会变（0/21 与默认一致，属数值取舍）、以及短序列
   的 wall/TPOT 回退尚无机制解释。long 用例则是全面变好（TTFT −24%、wall −10%、TPOT −4%）。

   **第 9 轮：把"修 lane 循环"这条路彻底关掉。** 新增 `chunk_and_per_lane_delta_gap_map`
   按几何逐点测两个 kernel 的差：

   | lanes×count | state gap | output gap |
   |---|---|---|
   | **1×1** | **2.98e-8** | **7.45e-9** |
   | 2×1 / 2×2 | 2.98e-8 / 5.96e-8 | 7.45e-9 |
   | 128×1 / 128×51 | 2.98e-8 / 9.31e-9 | 7.45e-9 |

   **单 lane、单 token 就已经差**，所以不是循环carry、不是 masked lane、不是 chunk 长度。
   逐元素统计（`INFER_GAP_INDEX=1`）：1×1 时 **6144 个 output 里 4634 个、786432 个 state 里
   59412 个**在末位不同 —— 大面积末位差，即同一个表达式树在 cuTile 里被收缩/调度成不同指令。
   已验证与载入形状无关（把 q/k/v/beta 换成与逐 lane 完全相同的扁平 `[D]`/`[1]` 分区+索引，
   数字一模一样）。**当时据此推断是编译器收缩/调度，但未证明 FMA 是原因，也未排除
   归约顺序差异；不能据此断言源码无法修复。** 2026-10-11 的优化等级扫描见 16.11。
   chunked 递推因此作为显式数值选择保留
   （`INFER_CUDA_CHUNKED_RECURRENT=1`），默认仍是逐 lane。

   收益已实测并记录：long TTFT −24%、wall −10%、TPOT −4%，各用例 TTFT 全面 −10~24%；
   short/hot_long 的 wall/TPOT 回退 8–10%（机制仍未解释）。

   **第 11 轮：委托到底值不值，先量再接线。** `resident/prefill_gemm/bench.rs` 把
   "cast + cuBLAS" 与现在的 prompt GEMM 放在同一张图里对打（2B 的真实形状，30 次重放，
   含必需的一次热身）：

   | 形状 (m×n×k) | 我们的 kernel | cast + cuBLAS | 比值 |
   |---|---|---|---|
   | 128×2048×2048 | 18.1 µs | 5.2 + 15.5 = 20.7 µs | **0.88×（我们更快）** |
   | 128×12288×2048 | 95.9 µs | 5.2 + 41.5 = 46.7 µs | **2.05×** |
   | 128×2048×8192 | 55.5 µs | 6.0 + 33.8 = 39.8 µs | **1.40×** |

   结论是**按形状分档**：宽 N（MLP 的 gate/up）委托赢 1.4–2×，方阵（attention 投影）打平，
   差的那一项恰好被"先把 f32 激活收窄成 bf16"这一步吃掉 —— 该步在 tile=1024 时 5.2 µs，
   tile=8192 时 8.1 µs，是纯粹的带宽/占用问题。因此接线规则应当**按权重规模分档**
   （≈ ≥8M 元素才委托），或者更稳的做法是在热身阶段对每个形状真跑一次赛跑再缓存胜者。

   注意 cuBLAS 要求两个操作数同类型，所以激活必须先转 bf16；同一层里 q/k/v 共享一份
   normalized hidden、MLP 的 gate/up 也共享，所以**每层转一次**可把 5.2 µs 摊到 2–3 个投影上，
   这是接线时的正确形态。

   **第 12 轮：接线完成并端到端实测。** 委托按第 11 轮的分档规则接进
   `prefill_projection::record`（`Support::record_dense`，权重 ≥8M 元素才委托），
   workspace 照 `nvfp4` 的模式挂在 `BatchBuilder`/`DeviceProgram` 上，激活按元素数惰性分配
   bf16 scratch。开关 `INFER_CUBLAS_PROJECTIONS=1`，默认关闭。

   接线中发现一个**必须遵守的约束**：cuBLAS 首次遇到某个 configuration 会做 host 侧工作
   （选 workspace/算法），捕获流会拒绝，所以 `DeviceProgram::new` 里先用 `warm_delegated`
   把每个 (width, n, k) 在设备流上跑一次再捕获图。否则请求直接失败：
   `operation not permitted when stream is capturing`。

   2B 两轮交错实测（ratio native/vLLM）：

   | 用例 | wall | TTFT | TPOT |
   |---|---|---|---|
   | short | 0.972 → 0.975 | 1.015 → **0.914**（−10.0%） | 0.973 → 0.983 |
   | long | 1.137 → 1.111 | 3.190 → **2.925**（−8.3%） | 0.983 → 0.979 |
   | batch4 | 1.430 → 1.428 | 2.151 → **1.921**（−10.7%） | 1.354 → 1.374 |
   | hot_long | 1.093 → 1.090 | 1.882 → **1.814**（−3.6%） | 1.031 → 1.032 |

   **TTFT 全面下降 3.6–10.7%**，wall/TPOT 基本不动（委托只作用于 MLP 的宽投影，attention 的
   方阵按测量被排除）。代价是数值：**21 条序列中 16 条与默认逐位相同，5 条不同**（cuBLAS 的
   累加顺序不同）。因此与 chunked 递推一样保持 opt-in，默认路径不变。

   **需要用户裁决的一点**：目前"性能改动必须 token 逐位一致"是仓库自己设的验收线（见
   `docs`），但两个已验证的加速（chunked 递推 −24% long TTFT、委托 −4~11% TTFT）都过不了
   这条线。若改为"logit 级相对误差 + 抽样一致性"这类数值门禁，它们就能成为默认路径。

   **第 13 轮：一个不改数值的加速（27B long TTFT −4.3%）。** 先给两个模型做了分桶剖析
   （2B：linear 53%、attention 32%；解码每 token 要读 4 GB bf16 权重，实测 3.1 ms/token
   对带宽下限 2.2 ms，vLLM 也在同一水平，没有余量）。真正可下手的是**提示图宽度**：

   仓库里已有的记录（`loading/mod.rs`）本来就写着"128-lane 提示图与两个 64-lane chunk
   **数值完全一致**，并把八次 replay 变成四次（long TTFT −13%）"。因为 key/value 一律过 KV
   cache、prompt GEMM 逐元素只在 K 上累加，所以**加宽 chunk 是数值中性的**。于是把
   `MAX_PREFILL_LANES=128` 之上加了一级 `WIDE_PREFILL_LANES=256`：

   - 27B（量化 recurrent）：`prefill_width=256, narrow=64`（arena 361/503 MiB），
     **long TTFT 0.3061 → 0.2929（−4.3%）、wall −1.0%**，两轮 **3/3 序列逐位相同**。
   - 2B（dense）：256 实测**无收益**（long TTFT 0.999、short 1.007），因此 dense 模型的
     阶梯仍止于 128，只有量化 recurrent 模型才拿这一级——这也解释了为什么原先的注释里
     只有 64→128 有收益。

   宽度选择逻辑抽成了 `select_prompt_width`，并写明"arena 是真正的约束，这个 cap 只是阶梯
   上限"。这轮没有触碰任何数值路径，默认矩阵因此净赚。

2b. **DFlash2 草稿模型（已调研 + 已下载，未实现）**：`z-lab/Qwen3.8-27B-DFlash2`
   （1.924B / 81 个 BF16 张量 / 3.849 GB，已下载到 `/home/r/models/Qwen3.8-27B-DFlash2`
   并校验张量可读）。结构：`fc.weight [5120,25600]` 把 **5 个目标层
   （`target_layer_ids [5,19,33,47,61]`）的 hidden 拼接**投回 5120；5 层 Qwen3 骨干
   （32 heads / 8 kv heads、head_dim 128、滑动窗口 2048、**块内非因果**）；每层两组
   "双抽头动态卷积" `attention_conv`/`mlp_conv`（`base_kernel [2,2,5120]` +
   `kernel_projection [1280,5120]`，即 conv_kernel_size 2 × conv_group_size 16）；
   `candidate_selector`（`hidden_projection [256,5120]` + 两个 `[248320,256]` 码本）在选择
   器里从 top-16 候选中追踪一条连贯路径。它**没有自己的 embedding / lm_head**，复用目标
   模型的。`block_size 8` 意味着**每条序列 8 条 verify lane**。

   **前置阻塞已定位（实测）**：我们现在的 verify 宽度上界不是算力而是**显存**。MTP 深度
   扫描（27B，2 轮，ratio native/vLLM-mtp2）显示 depth 2 最优（short wall 1.140 /
   TPOT 1.133），depth 3 起 batch4 崩塌（wall 1.110 → 2.679、TPOT 1.775 → 3.715），
   而 short 几乎不动——这是**池化被静默关闭**：`INFER_CUDA_EXECUTION_PROFILE` 记录
   `pool_creation_failed`，depth 3 时 width 4 报 `CUDA driver error: out of memory`，
   width 3/2 报 `Capacity: resident F32 state exceeds available device memory minus 1 GiB
   headroom`，于是 205 个 step 全部 `pool_disabled`，四条序列退回串行。27B 常驻已 22.9 GiB
   / 预算 27.7 GiB，恢复池化所需的 checkpoints 是按 lane 逐份的 F32 状态快照。

   **显存账（实测 + 解析）**：27B 是 64 层 = 48 层 `linear_attention` + 16 层
   `full_attention`，Delta 状态每层 48 个 value head × 128 × 128 × **F32** = 3.15 MB，
   48 层合计 **151 MB/序列**。池化回滚快照（`slot_verify.rs:51 slot_checkpoints`）
   只快照 `Conv | LinearAttention`，数量 = `slots × (verify-1)`：
   verify 3 → 4×2×151 MB = **1.21 GB**，verify 4 → **1.81 GB**，
   **verify 8（DFlash2）→ 4×7×151 MB = 4.23 GB**。而 27B 常驻后只剩
   27.7 − 22.9 ≈ **4.8 GiB**，还要容纳 4 份 slot 状态、arena、图和（DFlash2 的）
   3.85 GB 草稿权重。**8 lane 的 verify 仅回滚快照一项就超过全部余量** —— 这就是
   depth 3 起池化被关掉、以及 DFlash2 在本机不可行的真正原因；KV 不是主项
   （16 层 full attention，capacity 128 时约 16 MB，且有 `kv_scales` 时根本不在此分配）。

   **本轮实测（含一个已修的真 bug）**：一次失败的池化尝试**不会把内存还回来** —— 失败前
   `free 2476 MiB / pool 6 MiB`，失败后 `free 44 MiB / pool 2438 MiB`。内存进了 device
   memory pool，而 `cuMemGetInfo` 把 reserve 未用的块算作已用，于是 width 3/2 的重试在
   "free 只剩 44 MiB"上做准入判断，必然失败，池化被永久关闭。同时 `reclaim_cached_for_pool`
   只认 `Capacity`，而 driver OOM 是 `Backend`，所以 OOM 之后一次回收都没做。

   落地：回收条件扩到 `Capacity | Backend`；新增 `CudaDevice::pool_reclaimable_bytes()`
   （pool reserved − used）供诊断；两处拒绝信息带上 MiB；池化被放弃时 warn 出最后一次失败
   原因。**试过但撤回的两条路**：(i) 把 `available_memory_bytes = free + pool 可回收` 用于
   准入 —— 语义正确，但实测变成"准入通过、真分配 driver OOM"，depth 3 时好时坏，故撤回
   以免把干净的 Capacity 拒绝变成硬失败；(ii) `set_release_threshold(0)` —— 驱动是惰性
   trim，`cuMemGetInfo` 仍然显示 ~0 空闲，无效。

   **结论（本机硬约束）**：verify 4 的 4-slot 池需要约 2.9 GB 而当时只有 2.5 GB，回滚快照
   就是主项（`slots × (verify−1) × 151 MB`）；即使勉强降到 width 2，4 条序列只有 2 条能
   入池，实测 depth 3 的 batch4 仍输给 depth 2（TPOT 3.64 vs 1.80，wall 2.685 vs 1.093）。
   **DFlash2 的 verify 8 需要 4×7×151 MB = 4.23 GB 快照，在这台 31.4 GB 卡上不可行** ——
   要它成立，必须先让每个 lane 的状态快照不再是 151 MB F32（接受后重建 / 降精度 / 缩小
   Delta 状态本身），而不是继续调池化参数。基准配置 mtp2 完全不受影响：token 逐字节一致，
   short/long/hot_long 比值变化 <1%。

   因此 DFlash2 的顺序是：**(1) 降 verify 的 per-lane 状态成本**（接受后重建设为默认）；
   (2) 独立 checkpoint 加载（草稿是另一个 package）+ 草稿量化（3.85 GB → ~1 GB）；
   (3) DFlash2 IR：fc 融合、5 层双抽头卷积、块内非因果注意力、目标层 hidden 抽头、
   草稿侧 KV cache；(4) 选择器（码本打分 + 路径遍历）；(5) 块草稿 + 8-lane verify 集成与
   实测。**(1) 之前，(3)-(5) 做完也无法在本机跑起来。**

3. **FP8 精度制度补齐**：128×128 block-scaled（DeepSeek 系与 Flash-Next MTP
   expert 的事实标准）、per-tensor（PLE）、per-channel 三制度进 loader + kernel；
   **FP8 KV（杠杆 C 在本线提前）** —— 141GB 卡 + FP8 KV 是长上下文数据中心标
   配。

   **进展：block-128×128 已落地并实测**（`Qwen3-VL-2B-Instruct-FP8`，vLLM 原生
   `fp8` scheme：`weight_block_size [128,128]`、`weight_scale_inv`、动态 per-token
   量化）。loader 侧 `normalize_fp8_scheme` 把 vLLM scheme 翻译成内部形状，
   `WeightEncoding::Fp8Block` 绑定 `weight_scale_inv`，并把每个通道块的 scale 沿
   行展开，kernel 因此与 per-channel 路径同形；kernel 侧
   `fp8_gemm::quantize_block`（每 128 列一个 scale）+ `matmul_block`（每个 K step
   的部分积先乘 `xs[行,k块] * ws[通道块,k块]` 再累加——缩放操作数会二次过 FP8 舍
   入）。数值门 `fp8_block_mma_matches_independent_block_quantization` 对独立构造
   的 per-128-block 参考（f64、真值 scale）在 (12,65,512)/(64,129,1024)/(12,256,5120)
   三形状通过。2B FP8 矩阵（native/vLLM，2 轮）：short wall 1.239 / TTFT **0.749**
   / TPOT 1.299；long 1.496 / 3.085 / 1.312；batch4 1.666 / 1.251 / **1.750**；
   hot_long 1.457 / 2.093 / 1.393。**历史记录：token 与 vLLM 仅 3/21 逐字节一致，存在权重处理差异**：vLLM 在
   sm_120 上选 DeepGemmFp8BlockScaledMMKernel 且 `is_deep_gemm_e8m0_used()` 为真，
   `requant_weight_ue8m0_inplace` 会**用 checkpoint 的 fp32 scale 反量化、再用
   `per_block_cast_to_fp8(..., use_ue8m0=True)` 重新量化并把新 fp8 权重与 2 的幂
   scale 原地写回**（`fp8_utils.py:881-945`）。在本 checkpoint 上抽 12 个投影实测：
   权重相对 Frobenius 变化均值 **2.67%**、最大 2.73%，单权重变化中位数 **2.19%**
   —— 即 vLLM 评估的是"同一份权重的另一种量化"，多一次 e4m3 舍入 + 2 的幂
   scale 约束；native 的加载路径保留 checkpoint 的权重与 scale。
   **2026 年 10 月 9 日审查补充**：保留权重不能证明整模型算术误差为零，抽样权重变化也不能证明所有 token 差异均由再量化导致。本轮未取得原始报告复核历史 dispatch；仍需固定上下文的独立 logits/state 对照与任务质量验收，不能仅据该差异排除实现问题。具体边界见 [对比方法审查](../../reviews/vllm-benchmark-methodology-2026-10-09.md)。
4. **委托赛跑扩到 FP8**：cuBLASLt FP8 在 Hopper 极成熟，杠杆 A 的成对测量在
   H200 上预期更多形状判给 cuBLASLt —— 这不丢脸，是赛跑机制按设计工作。
5. **§六 瀑布按 141GB 重算**：27B FP8 权重 ~27GB，KV/并发余量与 5090 完全不同
   量级；瀑布是纯函数，重算即得，但验收矩阵要补 H200 行。

**竞争判断**【诚实】：H200+FP8 是 vLLM 最强主场（FlashInfer/FP8 路径全部战
验），不要指望复刻 5090 线的领先故事。本线目标是**入场券** —— 不在 H200 上就
不在数据中心。差异化指望：MTP 融合图（杠杆 B）、§七 auto profile、单二进制运
维形态，而不是 kernel 裸速。

**与 Flash-Next 的衔接**：该模型官方姿势 TP=8×B200；FP8 checkpoint（推断
~135GB+）在 H200 上单卡 141GB 是极限、TP=2 稳妥 ——  MoE 前置清单见 §八 阶段 3
与上一轮评估（grouped GEMM → 混合精度 loader → Qwen Sparse Attention → PLE →
expert offload）。

## 十、2026-10-09 B3 基线后的实测归因

本节记录首批服务基线（`2b-mtp0-serving-v1`、`27b-mtp0-serving-v1`、`27b-mtp2-serving-v1`）
签发时的实测归因，用于替换此前的推算锚点。比率方向统一为 **native / vLLM**。

### 服务层结论【确证】

| profile | short | long | batch4 | hot_long | long TTFT | batch4 TTFT |
|---|---:|---:|---:|---:|---:|---:|
| 2b-mtp0 | 1.03–1.06 | 1.24–1.26 | 1.56–1.58 | 1.17–1.22 | 3.4–3.5 | 1.8–2.3 |
| 27b-mtp0 | 1.07–1.12 | 1.28–1.29 | 1.35–1.41 | 1.09–1.13 | 3.9–4.1 | 1.7–2.0 |
| 27b-mtp2 | 1.26–1.31 | 1.70–1.73 | 3.50–3.56 | 1.05–1.13 | 4.2–4.5 | 7.3–8.1 |

两个独立配对单位结论一致；单位间最差格漂移 12.0–18.6%，接近阈值的格目前不可分辨。

启动时间不属于以上任何数字：vLLM 13.8–39.6 s，native 3.3–5.1 s。vLLM 运行时的整机负载与
风扇主要由这段启动（CUDA graph capture、inductor 编译、按 0.88 分配 KV）造成。

### 硬件归因【确证】

同一次测量内（active window，仅工作负载运行期间）：

| profile | native GPU 利用率 | vLLM GPU 利用率 | native 进程 CPU | vLLM 进程 CPU |
|---|---:|---:|---:|---:|
| 2b-mtp0 | 20.0–20.5% | 10.3–20.0% | 91–92% | 42–44% |
| 27b-mtp0 | 62.3–63.6% | 56.4–65.9% | 98–99% | 26% |
| 27b-mtp2 | 54.8–55.0% | 34.9–41.6% | 98% | 30–38% |

**native 的 GPU 利用率不低于 vLLM**，因此"GPU 没吃满"不是差距来源；差距是每步的效率与
主机侧成本。native 同时占满一个 CPU 核，而 vLLM 只用 26–44%。

### 逐算子设备时间（27B、mtp 0）【确证】

`INFER_CUDA_PREFILL_PROFILE` + `INFER_CUDA_PROFILE_GRAPH_ONLY=1`（仅图边界事件，避免逐节点
事件自身的开销），同一 profile 的完整矩阵：

| 图 | 每次 replay | 设备时间占比 | 主要算子 |
|---|---:|---|---|
| `slot_decode` | 18.69 ms | 5514 ms | linear 17.3 / delta 1.0 / norm 0.8 ms |
| `prefill` | 142.95 ms | 2573 ms | delta 58.3(41%) / linear 41.3(29%) / attention 37.3(26%) |
| `prefill_last` | 51.21 ms | 1434 ms | linear 24.5(45%) / delta 19.3(36%) / attention 5.4 ms |

`INFER_CUDA_EXECUTION_PROFILE` 同一轮给出的主机侧单次往返：`prefill_step` 16.09 ms × 761 次，
`prefill_chunk` 107.95 ms × 38 次。该数值是**包含等待设备的同步往返**，因此不是叠加项——
它也说明解码路径上主机回读**没有**成为关键路径。

并发确实被合并：181 次调用带 4 条 decode lane（batch4 真批处理），其余单条来自单请求用例。

### 与内存 roofline 的对比【确证】

27B NVFP4 权重 21.81 GiB（23.42 GB）；RTX 5090 峰值带宽 1.79 TB/s。

- 读一遍权重的下限：**13.07 ms/token**。
- native `slot_decode` 实测 18.69 ms → **1.43× 下限**。
- 服务层 long 用例：native ≈20.8 ms/token，vLLM ≈17.3 ms/token（1.32× 下限）。

结论：解码已经接近带宽墙，两者相差约 1.2–1.3×，**不是数量级差异**；真正的倍差在
prompt/prefill 与并发准入上（long TTFT 3.9–4.5x、batch4 1.35–3.56x）。

### 优先顺序【推断】

1. **Delta 的 prefill 占用率**（`recurrent_prefill::delta`，见 §一 与本节 prefill 表）：
   每个 CTA 为每个 value head 串行处理 LANES 个 token，每次迭代读写整个 D×D=64 KB 状态，
   网格只有 48 个 CTA（170 SM）。把 value 维切成 S 块是**数值精确**的（列之间互不依赖，
   归约只发生在 key 维），网格变成 48×S，这是当前 prefill 41% 占比的直接杠杆。
2. **每 token 的 replay 数**（§〇 结论 1）：prefill 单次 replay 成本与 chunk 内 token 数无关，
   因此"权重一遍过、token 成批过"仍然成立。
3. 解码侧：把 greedy 采样搬到设备端、只回传 token，可去掉每步 993 KB 的整词表 D2H 与
   一次 `to_vec` 主机拷贝；但它不是当前关键路径，收益应低于上面两项，排在后面。

复现命令（必须在受保护范围内）：

```sh
bash tools/bench/safe-run.sh env \
    INFER_CUDA_PREFILL_PROFILE=/tmp/ops.jsonl INFER_CUDA_PROFILE_GRAPH_ONLY=1 \
    INFER_CUDA_EXECUTION_PROFILE=/tmp/steps.jsonl \
    python3 tools/bench/serve-compare.py --engine native \
    --model /home/r/models/Qwen3.8-27B-NVFP4 --executable target/release/infer \
    --inputs artifacts/workloads/Qwen3.8-27B-NVFP4-inputs.json --output-dir artifacts/b3 \
    --mtp 0 --tokens 64 --gpu-memory-utilization 0.88 --max-model-len 65536 --max-num-seqs 16 \
    --run-id <新 ID> --profile-id 27b-mtp0
```

### 十.1 值维分块的 A/B 与真正的瓶颈【确证】

按 §十 的推断 1，实现了 per-lane Delta 的 value 维分块：状态列互不依赖（归约只在 key 维），
把 `state`/`out` 按 `[1, D, D/S]`、`[1, 1, D/S]` 分区即可，网格从每 value head 一块变成 S 块。
切换用 `INFER_CUDA_RECURRENT_VALUE_SPLIT`，默认 1（保持既有行为）。

**数值**（`per_lane_delta_matches_independent_recurrence_at_model_geometry`，kh=16、vh=48、D=128
的 f64 独立参考）：split 1 最差 state 6.90e-8 / out 8.67e-9；split 4 为 6.71e-8 / 1.04e-8；
split 4 相对 split 1 的差为 state 2.98e-8 / out 7.45e-9，与既有 chunked-vs-per-lane 的
9.31e-9 / 7.45e-9 同量级，属于 cuTile lowering 的末位差，不是数值口径变化。

**性能：两个核都没有收益。** 27B、mtp 0，同一 profile 的完整矩阵（图边界事件）：

| 内核 | split | `prefill` 每次 replay | `prefill_last` | `slot_decode` |
|---|---:|---:|---:|---:|
| per-lane | 1 | 143.60 ms | 51.51 ms | 18.71 ms |
| per-lane | 4 | 147.36 ms | 52.74 ms | 18.76 ms |
| chunked | 1 | 143.98 ms | 51.46 ms | 18.70 ms |
| chunked | 4 | 147.64 ms | 52.86 ms | 18.68 ms |

网格从 48 块变到 192 块（170 SM）却不变快，说明这个核**不是**占用率受限。逐节点画像还显示
27B 的 prefill 图每层只有 **一个** delta 节点（1154 节点 / 64 层 ≈ 18 个/层，delta 占 1 个），
即实际走的是 chunked 核，因此第一次只改 per-lane 核时测到的"无收益"同时意味着改错了核；
补上 chunked 核后仍是 +2.5%，两条路径一起否定了占用率假设。

**真正的瓶颈是 prompt 图的节点数和每节点开销。** `INFER_CUDA_PREFILL_PROFILE` 的逐节点输出：
32-lane 的 prefill 图有 **1154 个节点**，单次 replay 119.3 ms，即 **每节点 103 µs**；其中 delta 占
61.28 ms。按 64 层换算，per-lane 的 prompt token 每层约 **70 µs**，而 vLLM 的 511-token prompt
TTFT 约 150 ms、即每层每 token 约 5 µs——**差在每层每 token 的固定开销，约 15×**，不在带宽。

**下一次要测的是 replay 次数，而不是核内并行度。** `prefill` 图在整轮矩阵里只 replay 18 次
（`prefill_last` 28 次），即一个 prompt 基本由 1–2 次 replay 处理完；`long` 的 TTFT 248 ms 对应
511 token，约 0.49 ms/token，而 vLLM 同负载约 63 ms、即 0.12 ms/token。所以下一步先量清楚
「每个 prompt 长度用了多少次 replay、每次 replay 覆盖多少 token、以及单次 replay 里各节点占比」，
再决定是合并窄算子、加宽 tile，还是减少 replay 次数；在这些数据之前不再改核。

§十 推断 1 作为优化方向被这次 A/B 否定。分块代码保留：split=1 时是原有路径，数值已验证
（chunked 在 4 组几何上 split 4 相对单块最差 2.98e-8 state / 1.49e-8 out，per-lane 为
2.98e-8 / 7.45e-9，都在 f64 参考的 2e-5 之内），后续若要换 lowering 或换 tile 宽度可直接复用。

### 十.2 prompt 路径的逐算子定位【确证】

按 §十.1 的结论，把每个 case 单独成 workload 跑一遍（这样 replay 能归属到 case），27B、mtp 0、
release 制品，图边界事件。权重一遍的物理下限是 **13.07 ms/replay**（21.81 GiB / 1.79 TB/s）。

| case | 图 | 每次 replay | replay 次数 | 相对下限 |
|---|---|---:|---:|---:|
| short | `prefill_last` | 36.4 ms | 4 | 2.8× |
| long | `prefill` | 112.8 ms | 4 | 8.6× |
| long | `prefill_last` | 119.8 ms | 4 | 9.2× |
| batch4 | `prefill_last` | 36.1 ms | 16 | 2.8× |
| batch4 | `slot_decode` | 18.8 ms | 281 | **1.44×** |
| hot_long | `prefill` | 153.3 ms | 14 | 11.7× |

**解码已经贴住带宽墙（1.44×），prompt 路径差 2.8–11.7×**，这与服务层"只有 decode 的格子在 1.1×
附近、含 prefill 的格子 1.3–4.5×"完全一致。

`long`（511 token，8 次 replay、9236 个节点、951.4 ms 设备时间，每节点 103 µs）的算子构成：

| 算子 | 设备时间 | 占比 | 节点数 | 每节点 |
|---|---:|---:|---:|---:|
| **delta** | 462.3 ms | **48.6%** | 384 | **1204 µs** |
| **linear** | 335.6 ms | **35.3%** | 3972 | 84.5 µs |
| attention | 71.1 ms | 7.5% | 128 | 556 µs |
| gated_norm | 24.0 ms | 2.5% | 384 | 62.4 µs |
| norm | 16.0 ms | 1.7% | 1288 | 12.4 µs |
| conv | 11.0 ms | 1.2% | 384 | 28.7 µs |
| 其余（add/multiply/silu/rope） | 26.5 ms | 2.8% | 2432 | ≤17 µs |

两个可归因的靶子：

1. **delta：48.6%，每节点 1.2 ms。** chunked 核每块循环 LANES 条 lane，状态已在寄存器里跨 lane 复用，
   每 lane 的全局流量只有 q/k/v（各 D 个 f32）；按 128 lane 折算每 lane 约 9.4 µs，而每 lane 的状态
   流量下限只有 0.07 µs。§十.1 已证明加块无效，所以下一步要么降低每 lane 的依赖链延迟（例如把
   `q/k` 的加载与上一层重叠、把 reduce 换成 warp 级实现），要么换算法；在拿到核内计数前不动它。
2. **linear：35.3%，每节点 84.5 µs**，而 M=128、K=N=5120 的单次投影权重 52 MB、下限约 29 µs，
   即偏离 2.9×。这是按形状选 tile 的问题，仓库已有 `prefill_gemm/bench_check.rs` 可直接量。

另外，整图**每节点 103 µs** 这个量级本身说明：prompt 路径的成本主要由"节点数 × 每节点固定开销"
决定，1154 节点/次 replay 的规模下，任何节点级浪费都会被放大 1000 次；这也是为什么先把每节点的
占比量清楚比继续调并行度更重要。

## 十一、2026-10-10 Stage A/B：并发准入计价与同步消除

§十 的归因把 batch4 3.50–3.56x 指向"并发准入"，本节记录当天据此落地的两批改动与实测。
比率方向统一为 **native / vLLM**；vLLM 侧在**同机同驱动**（615.78.08）上重测
（`stage-b2-mtp2-vllm-g1`），因为 10-09 签发基线是 615.71.09，硬件身份已不匹配。

### 11.1 症结：池化序列被按"私有验证图"计价【确证】

`CB_DECODE_SLOTS=4` 的槽位池本该让 4 条并发序列共享一张验证图，但准入计价对所有序列
一视同仁地收了"私有 verify 图 + 私有 wide prompt 图"的账。结果是：**第 4 条请求进不了
准入，落回串行私有路径**，而它又与 3 条池化序列抢同一块显存，于是"每步只跑 1 条序列
（blocked=3）"——batch4 的 7.3–8.1x TTFT 就来自这里，不是内核慢。

### 11.2 Stage A：按实际用途计价

| 位置 | 改动 |
|---|---|
| `executor/state.rs` | 新增 `private_verification_for`：只有当池已满（`pooled >= pool.width()`）或形状不适合池化时才按私有图计价 |
| `loading/budget.rs` | `sequence_budget(capacity, readout, private_verification)`：池化序列不再为 verify 回滚快照/批工作区付账 |
| `loading/mod.rs`·`resident/program.rs` | 新增 `sequence_pooled`/`new_pooled`/`graph_policy`/`forfeit_verification`：池化程序不捕获私有 verify 图；显存紧张时也不捕获 wide prompt 图 |
| `resident/capture.rs` | `record_program(graph, skip_logits)`：prefill 图把 logits 读出留给它的 `_last` 孪生图 |
| `executor/execution.rs` | `warm_slot_pool`：装载期就捕获槽位池，第一条并发请求即可批处理（代价是启动 +≈2.7 s） |
| `constants.rs` | `CB_SLOT_TOKENS` 4096 → 2048：池是装载期捕获的，它的 fp8 KV 与并发序列争物理显存；4096 行时池 + 4 条序列超卡 |

### 11.3 Stage B1：把"浪费的同步"换成流序

单流执行器里，host 只在真正要读数据时才该同步。原先 catch_up 的"零拷贝 readback"、
commit 的部分接受 restore 都是纯同步浪费：

| 位置 | 改动 |
|---|---|
| `device.rs` | `copy_h2d_pinned`：从 pinned 暂存区入队的异步 H2D，不在这里同步 |
| `resident/batch.rs` | `stage_lanes`（校验+metadata staging 抽出）、`replay_detached`（state-only 图 `async_on`，不保留 future）、`barrier()` |
| `resident/slot_batch.rs` | 每 lane 的 pinned 上传环（`EXTERNAL_UPLOAD_RING = MAX_VERIFICATION_WIDTH + 1`，无同步链内不得覆写）+ `SlotPool` 的 `Drop` barrier |
| `resident/slot_batch/draft.rs` | `upload_external`/`upload_lanes`/`run_external_detached`：catch_up 走无同步链 |
| `resident/slot_verify.rs` | commit 的 restore 改 `async_on`；`barrier()` 供池teardown 排空 |
| `executor/drafting.rs` | catch_up 改用 `run_external_detached` |

### 11.4 实测（2026-10-10 23:14，`mine-mtp2-native-g1`）

| case | ttft | tpot | wall |
|---|---:|---:|---:|
| short | 1.85 | 1.58 | 1.60 |
| long | 6.10 | 1.44 | 1.92 |
| batch4 | 2.64 | 2.07 | 2.10 |
| hot_long | 0.42 | 1.28 | 1.05 |

- **Stage A 命中目标**：batch4 TTFT 668 → 236 ms（7.3–8.1x → 2.64x），batch4 wall
  3.50x → 2.04–2.10x；4 条序列同时驻留。
- **Stage B1 收益很小**：TPOT 只降 1–3%（batch4 18.96 → 18.34 ms），落在单位间漂移内。
  说明 10-09 记的那些 host 同步原本就不在关键路径上，§十.2 的"主机回读没有成为关键路径"
  在这里第二次被证实。
- 三次复跑（21:33 / 21:46 / 22:49 / 23:14）互为复现，单格漂移 ≤4%。

### 11.5 未决项一：native 绝对 TPOT 比 10-09 基线慢 4–25%【待归因】

同一 profile、同一命令行、同一模型，10-09 签发基线（rev `2bda883`）与今天：

| case | tpot 10-09 | tpot 今天 | Δ | vLLM 10-09 | vLLM 今天 |
|---|---:|---:|---:|---:|---:|
| short | 10.85 | 13.5 | +25% | 8.66 | 8.59 |
| long | 12.32 | 13.0 | +6% | 9.07 | 9.10 |
| batch4 | 16.66 | 19.0 | +14% | 9.38 | 9.19 |
| hot_long | 11.42 | 11.9 | +4% | 7.85 | 9.26 |

排除项：SM 时钟（2833 vs 2841 MHz）、功耗（205 vs 208 W）、进程 CPU（97.8% vs 97.4%，
同为一核占满）、`config_readback` 的调度参数逐字相同；`2bda883..HEAD` 之间**没有任何**
改动 CUDA 解码路径的提交（只有测试搬迁、文档、`fdbcb89` 的 workspace 依赖声明——Cargo.lock
无 diff、一次 refuted 实验的 Delta value-split 且默认 split=1）。

因此嫌疑集中在**环境**：驱动 615.71.09 → 615.78.08，或 cuTile 运行期 JIT 的产物变化——
vLLM 侧绝对数字不动支持这一点（它的 kernel 是预编译的，我们是 JIT/lowering 出来的）。
在解释清楚之前，**跨驱动的绝对数字不可比**，只有同驱动的成对比率能用；签发新基线前必须
先复现/推翻这条。

### 11.6 池化程序放弃 wide prompt 图的代价，以及为什么换不回来【确证，已否】

`graph_policy` 为了 4 条池化序列能同时驻留，让池化程序只保留 narrow prompt 图
（`wide_prompt = private_verification || narrow_prefill_width < PREFILL_LANES`）。
单请求也走池化程序，于是 511-token 的 `long` 被切成更多次 replay：long TTFT
317.6 → 396 ms（+25%），比值 4.2–4.5 → 6.10。

**把 wide 图还给池化程序的两个变体都实测否掉了**（同 profile、同 vLLM 参照，
`exp-wideprompt-native-g1`、`exp-firstwide-native-g1`）：

| 变体 | long TTFT | batch4 TTFT | batch4 wall |
|---|---:|---:|---:|
| narrow only（现状） | 398 ms | 243 ms | 1.48 s |
| 4 条池化都给 wide | **311 ms** | 671 ms | 2.32 s |
| 只有第一条池化驻留给 wide | **311 ms** | 671 ms | 2.31 s |

- 给全部池化程序 wide：long 快 22%，但 batch4 退回串行准入（TTFT 243 → 671 ms，
  比值 2.64 → 7.27），因为 wide arena 是每序列最大的分配，4 份加池就超卡。
- **只给第一条驻留：结果一样坏。** 冷启动 server 上只发 batch4（无历史 shell）测得
  热态 group wall 1.52 → 2.15 s，第 4 条 lane 的 TTFT 1.26 s，defer `Preparation` 15 次：
  一份 wide arena 已经足以让第 4 条排不进来。shell 复用池按 `(capacity, readout)` 取，
  还会把前序用例留下的 wide shell 再发给并发请求，加重这一点。
- 结论：**wide prompt 图与"该请求是否独自在跑"绑定，而这件事在准入时刻不可知**
  （`reserve` 只拿到 capacity/readout，池是空的还是即将来 3 条看不出区别）。
  在 prefill 能与并发共享同一份 arena（或按批次大小惰性升级程序）之前，
  池化序列只能走 narrow 图；这条记为已知代价，不再尝试调 `graph_policy`。

**第三次变体：池化程序只捕一张 prompt 图，但捕 wide 那张**（`exp-wideonly-native-g1`）。
前两次失败都是"一个程序捕两张图"，于是把约束改写成"一个程序一张图"再测：

| case | 指标 | narrow（现状） | wide-only | 比值 narrow → wide-only |
|---|---|---:|---:|---|
| long | TTFT | 394 ms | **314 ms** | 6.04 → **4.82** |
| short | TTFT | 56 ms | 99 ms | 1.81 → **3.22** |
| batch4 | TTFT | 240 ms | 286 ms | 2.60 → **3.10** |
| batch4 | wall | 1454 ms | 1439 ms | 2.06 → 2.03 |
| hot_long | wall | 802 ms | 804 ms | 1.00 → 1.01 |

**batch4 这次没崩**（1.44 s，与 narrow 相同），所以前两次的瓶颈确实是"每个池化程序捕几张图"
而不是图有多宽 —— 4×1 张能同时驻留，4×2（或 1×2 + 3×1）不能。但 wide-only 把 `short`
的 51-token prompt 也推上 256-lane 图：TTFT 56 → 99 ms（**+78%**）。三个 TTFT 格子的比值之和
从 1.81+6.04+2.60=10.45 变成 3.22+4.82+3.10=11.14，**净亏**，因此回滚。

**这次量出了 prompt 图的两个机制【确证】**（后面要动 prefill 的人必须知道）：

1. **prompt 图单次 replay 的成本随 lane 数近似线性**：128 → 256 lane 让 51 token 的 prefill
   从 56 ms 涨到 99 ms（≈1.8x）。所以"更宽的图"只在 prompt 长到值得用它换 replay 次数时才划算
   —— 511 token 是 2×256 便宜于 4×128（394 → 314 ms），51 token 反过来。
   这正是 `prefers_narrow_prompt` 按 chunk 长度选图的依据。
2. **池化程序的显存瓶颈是"捕了几张 prompt 图"，不是"图有多宽"**：1 张（无论 128 还是 256）
   都能让 4 条并发驻留，2 张就不行。

因此"既要 short 快又要 long 快"的正路**不是加第二张图**，而是让**一张图同时服务两个区间**
（例如让 256-lane 图对掩码行不再付满代价），或让 prefill 与并发共享同一份 arena。
在那之前 `graph_policy` 维持 narrow-only。

### 11.7 decode 侧的两条负结果：tile 已经是优解，逐节点画像不能再往下挖

`slot_verify`（4 lane、MTP2、12 行）是 decode 的最大单项：图边界事件中位
**26.68 ms**，逐节点一次 replay 的构成（node_count 1155，与 total 一致）：

| 算子 | 每 replay | 节点数 | 每节点 |
|---|---:|---:|---:|
| **linear** | 19.76 ms | 497 | 36.8 µs（中位） |
| delta | 3.89 ms | 48 | 81 µs |
| conv | 2.34 ms | 48 | 49 µs |
| attention | 1.38 ms | 16 | 86 µs |
| norm+add+multiply+rope+silu+split+sigmoid+gated_norm | ~2.7 ms | 529 | ≤7 µs |
| lm_head（组 64 的第 2 个 linear） | 0.78 ms | 1 | 783 µs |

**tile 扫描【确证，已否】**：给 `quant_gemm_tile` 加了一个 capture 期的 env 覆盖
（`INFER_CUDA_QUANT_TILE=rows,cols`，只覆盖 decode 宽度），用同一套 `slot_verify`
图时间扫了列 tile：

| 列 tile | slot_verify 中位 |
|---|---:|
| **64（现状）** | **24.74 ms** |
| 32 | 25.05 ms |
| 128 | 25.06 ms |
| 256 | 26.68 ms |

`[16,64]` 已经是优解；加宽（更少 CTA）和收窄（更多 CTA 但激活重读翻倍）都更差。
覆盖代码已回滚——decode GEMM 的问题不在形状选择。

**逐节点画像的 9 个离群点不是形状病理【确证】**：64 层里 9 层各有一个 linear 节点
200–320 µs，而同样的形状在其它层只有 20–53 µs。但它们

- 出现在**不同的投影位置**（组内下标 11 / 6 / 19 / 14 都有），
- 间隔**非常规律**（约每 96 个节点一次，即每 5–6 层），
- 且 `segments` 之和恰好等于 `total_ms`（没有未计入的间隙）。

形状相同、位置随机、间隔规律 —— 这是**采集期的干扰被摊到某一节点**，不是可修的核。
结论：per-node 画像的**总量**可信，**单个离群节点**不可信，不要照着它改 kernel；
要判断某个形状慢不慢，用隔离 bench + 服务矩阵两把尺子。

### 11.8 下一步（按证据排序）

1. **B2b cuBLASLt NVFP4 委托**（§二 杠杆 A，规格已确证）：这是 decode 唯一还没试过的
   大杠杆。11.7 已经把"换 tile/换形状"这条排除掉，剩下的就是**同一个形状换更好的核**：
   隔离测量窄行形状 cuBLASLt 赢 1.6–1.7x，而我们的 GEMM 在服务里只有 ~50% 带宽。
   两个必须带着的教训：① `[64,128]` 隔离赢 1.47x、服务端反而慢 1.7%
   （`slot_verify` 26.69 → 27.43 ms），所以验收只认服务矩阵；② decode 单个 GEMM 只有
   ~40 个 CTA，填不满 170 SM（`PROMPT_GEMM_TILE_ROWS` 注释），任何减少 CTA 的方向先输。
2. **B4 设备端 greedy argmax**：draft 路径 host 侧 5.5 ms/tick 对设备 2.8 ms/tick，
   差额主要是 8×993 KB 的 logits D2H 与 host 串行 argmax。门控条件
   `temperature == 0 && presence == 0 && repetition == 1.0`，tie-break 必须与 host
   `greedy()` 逐位一致（首个最大值、total order、非有限值报错）。
3. **B3 Row 派发跨 lane 批量化**：`delta` 3.9 + `conv` 2.3 + norm/rope/add ≈ 2.7 ms 的
   12 路串行延迟链。

**测量前先确认二进制**：同一 worktree 里有另一条工作线在构建，`make local-build` 的产物
`target/release/infer` 被 CPU-only 构建覆盖过（24.8 MB → 8.3 MB，随后服务报
`no supported GPU backend available`）。跑基准前 `stat -c%s target/release/infer`
应 >20 MB，否则先重新 `make local-build`。

复现命令（native 侧，vLLM 侧复用 `stage-b2-mtp2-vllm-g1`）：

```sh
bash tools/bench/safe-run.sh python3 tools/bench/serve-compare.py --engine native \
  --model /home/r/models/Qwen3.8-27B-NVFP4 --executable target/release/infer \
  --inputs artifacts/workloads/Qwen3.8-27B-NVFP4-inputs.json --output-dir artifacts/stage-b-20261010 \
  --mtp 2 --tokens 64 --gpu-memory-utilization 0.88 --max-model-len 32768 --max-num-seqs 16 \
  --run-id <新 ID> --profile-id 27b-mtp2 --checklist benchmarks/profiles/27b-mtp2.json
python3 tools/bench/compare-results.py \
  artifacts/stage-b-20261010/Qwen3.8-27B-NVFP4-mtp2-vllm-stage-b2-mtp2-vllm-g1.json \
  artifacts/stage-b-20261010/Qwen3.8-27B-NVFP4-mtp2-native-<新 ID>.json
```

## 十二、prefill 逐算子浪费归因（2026-10-10）

触发这个问题的是服务层的一个规律：**prompt 越长，比值越差**（short 1.81 → long 6.04），
而 native 的 GPU 利用率**高于** vLLM（53.5% vs 41.6%）。利用率更高却更慢，说明差距不是
"GPU 没吃饱"，而是**每个有用 token 上烧掉的周期更多**。本节把 prefill 的每次 replay
拆到算子级，并用"有用 FLOPs / 有用字节"两个下界去卡每个算子。

### 12.1 复现口径

`INFER_CUDA_PREFILL_PROFILE`（逐节点，不加 `GRAPH_ONLY`），单条 `long`（511 token）请求，
release 制品。profile 每条记录带 `tokens` 与 `position`，因此可以直接读出**每个 prompt
graph replay 实际吃了多少 token**：

| graph | tokens | position | node_count | total_ms |
|---|---:|---:|---:|---:|
| prefill | 64 | 256 | 1154 | 45.4 |
| prefill | 64 | 320 | 1154 | 45.8 |
| prefill | 64 | 384 | 1154 | 46.4 |
| prefill_last | 63 | 448 | 1155 | 47.5 |

→ **511 token 被切成 8 次 replay，每次 64 token**（8 × 46 ms ≈ 368 ms，与服务层
long TTFT 394 ms 一致；vLLM 同负载 65 ms）。

### 12.2 每次 replay（64 token，46.1 ms）的构成与下界

权重一遍 = 15.79 GB（48 层线性注意力 249 MB + 16 层全注意力 239 MB，FP4 按 0.5 B/参数、
FP8 按 1 B/参数）→ 带宽下界 **8.82 ms**；GEMM 有用 FLOPs = 2·64·24.3 G = **3.11 TFLOP**。

| 算子 | ms | 占比 | 节点 | µs/节点 | µs/token | 有用下界 | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| **linear** | 22.46 | 48.7% | 496 | 45.3 | 351 | 8.82 ms（带宽） | 2.5×；实测 138 TFLOP/s = FP8 峰值 ~16%、704 GB/s = 峰值带宽 ~39%，**先撞发射/算力，不是带宽** |
| **delta** | 16.00 | 34.7% | 48 | **333** | **250** | ~0.1 ms（FLOPs）/ 0.17 ms（字节） | **~150×**，实测 ~0.5 TFLOP/s = fp32 峰值 ~0.5% |
| **attention** | 3.25 | 7.1% | 16 | 203 | 51 | ~0.02 ms | **~160×** |
| norm | 1.17 | 2.5% | 161 | 7.3 | 18 | ~0.2 ms | 5×，且 161 个独立节点 |
| gated_norm | 0.94 | 2.0% | 48 | 19.5 | 15 | ~0.06 ms | 15× |
| conv | 0.67 | 1.5% | 48 | 14.0 | 10 | ~0.05 ms | 13× |
| add / multiply / silu / rope / split / sigmoid / embedding | 1.70 | 3.7% | 321 | 4–97 | 17 | ~0.5 ms | 3–5× |

**每个有用 token 的 prefill 设备时间是 720 µs**，其中 GEMM 351 µs、**非 GEMM 369 µs（51%）**。
vLLM 整条 prefill 的每 token 预算只有 ~128 µs —— 也就是说**光是我们的 delta 一项
（250 µs/token）就超过 vLLM 做完整个 token 的预算**。这才是"native 计算量更大"的实处。

### 12.3 无效计算在哪：delta 与 attention 是同一个病

`recurrent_prefill::delta`（`crates/backend/cuda/src/resident/recurrent_prefill.rs:16`）是
**逐 token 串行扫描**：网格 = VH × SW = 48 × 1 = **48 个 CTA（170 SM）**，每个 CTA 把 chunk
的 LANES 个 token 一个个循环过去，每个 token 做两次"跨线程归约 key 维 D=128"：

- 算法本身每 token 每 head 需要 3×D² ≈ 49 K FLOP（状态衰减、rank-1 更新、读出），
  ×48 head ×48 层 = 113 MFLOP/token。**这部分 FLOP 是不可避免的**；
- 但每个 token 要付两次跨线程归约（`reduce_sum(..., 0i32)`）的同步/共享内存代价，
  64 个 token 就是 128 次归约/CTA，而只有 48 个 CTA 去遮这段延迟。

§十.1 已经把"值维分块（SW=4 → 192 CTA）"实测否掉了：**加 CTA 没用**，因为限制不是占用率，
而是每个 token 的归约依赖链。**跨 chunk 宽度的对照也支持这一点**：同一次 profile 里
decode 的 delta 节点是 53.7 µs / 12 lane（4.5 µs/token/层，走 per-lane 核），prefill 是
333 µs / 64 lane（5.2 µs/token/层，走 chunked 核）—— **两者每 token 成本几乎相同**，
说明 chunked 核并没有把逐 token 的开销摊掉，只是把状态留在了寄存器里。

所以正路是**把逐 token 的串行扫描换成 chunk 级矩阵乘**（WY / chunked linear attention：
intra-chunk 用 (C×C)×D 的张量核矩阵乘，状态更新用 D×(C)×(C×D) 一次算完），把"每 token
两次归约"变成"每 chunk 两次归约"。

`attention`（3.25 ms / 16 节点 / 203 µs）是同一类：C=64、KV≤320 的 prefill attention
有用 FLOPs 只有 ~8 GFLOP/replay，本该是几个张量核矩阵乘 + softmax，实测 ~160× 于下界。

### 12.4 另外两条

1. **chunk 宽度与图宽度脱节**：`select_prompt_width` 对 quantized-recurrent 模型给出的
   `narrow_prefill_width = 64`、`prefill_width = 256`；而 Stage A 之后池化程序
   `graph_policy` 只捕 narrow 那张 → 池化序列的 prompt 图 **只有 64 lane**，
   于是 chunk = 64。私有程序捕 256 → chunk = 256，511 token 只要 2 次 replay。
   这就是 §11.6 里 long TTFT 317 → 396 ms 的**全部原因**：replay 次数 2 → 8。
   但把 256 lane 图给池化程序又有 §11.6 的代价（51-token prompt 从 56 → 99 ms，
   因为掩码 lane 也要付钱）。**修完 12.3 之后这条才值得再谈**：非 GEMM 的每 token
   成本塌下去以后，chunk 宽度的收益才会体现为权重流量（8 遍 → 2 遍，126 GB → 32 GB）。
2. **linear 是"发射受限"而非带宽受限**：138 TFLOP/s（FP8 峰值 ~16%）而字节只用到
   704 GB/s（39%）。所以 tile 形状（§11.7 已扫完，`[16,64]` 是优解）不是出路，
   出路是同形状换更高效的核 —— §二 杠杆 A 的 cuBLASLt 委托。

### 12.5 结论与动作顺序

| 序 | 动作 | 依据 | 预期 |
|---|---|---|---|
| 1 | **delta 改 chunk 级矩阵乘**（WY 形式） | 34.7% 的 prefill 设备时间，150× 于下界，且 §十.1 已排除占用率 | replay 46 → ~31 ms；long TTFT 394 → ~280 ms；decode 也减 ~2.6 ms/replay |
| 2 | **prefill attention 同批处理** | 7.1%，160× 于下界 | replay 再 −3 ms |
| 3 | 窄算子融合（norm+add 等 537 节点/4.5 ms） | 9.7%，3–15× 于下界 | replay −2 ms 量级 |
| 4 | GEMM 委托 cuBLASLt（§二 杠杆 A） | linear 48.7%，138 TFLOP/s = 16% 峰值 | 隔离 1.6–1.7×，须服务矩阵验收 |
| 5 | 再谈 chunk 宽度/单图双区间（§11.6） | 依赖 1 落地 | 权重流量 4× |

抓取命令（单条 long、逐节点画像）：

```sh
pkill -f '^target/release/infer'
INFER_CUDA_PREFILL_PROFILE=/tmp/gp.jsonl target/release/infer /home/r/models/Qwen3.8-27B-NVFP4 \
  --listen 127.0.0.1:38091 --num-speculative-tokens 2 --gpu-memory-utilization 0.88 \
  --max-model-len 32768 --max-num-seqs 16 &
python3 /tmp/repro-case.py http://127.0.0.1:38091 long   # 先跑一遍预热，再删 profile 重跑
```

### 12.6 已否掉的低成本尝试：delta 的代数重排【确证，已回滚】

§12.3 的机制假设是"每 token 两次跨线程归约"。最便宜的验证是不动并行度、只重排代数：
输出可以写成

```
sum_d (decay*old + k⊗diff)[d,s] * q[d]
  = decay * sum_d old[d,s]*q[d] + diff[s] * (k·q)
```

这样两次大归约都只读 `old`，互相独立，链从"归约→更新→归约"变成"归约(并行)→更新"，
并且不再需要为了求和而物化 `updated` 整块 tile。实现后跑现成的配对基准
（`resident_recurrent_prefill_benchmark_tests::recurrent_chunk_performance_gate`，
`do_bench_paired` + 清 L2，chunked 核 vs 逐 lane 核）：

| 几何 | 原式 new_ms | 重排后 new_ms | 变化 |
|---|---:|---:|---:|
| (2, 4, 32, 32) | 0.0431 | 0.0451 | **+4.5%** |
| (16, 48, 128, 32) | 0.0720 | 0.0789 | **+9.6%** |
| (16, 48, 128, 128) | 0.2529 | 0.2766 | **+9.4%** |

**慢了 4.5–9.6%**：cuTile 对原式的 elementwise+归约已经融得很好，重排反而多了一次
`k·q` 归约和若干 broadcast。数值上重排是安全的（chunked-vs-per-lane 输出差
7.45e-9 → 1.12e-8，仍比断言 1e-7 小一个量级），但**没有收益，已回滚**。

结论：delta 的每 token 算术已经是最优写法，成本来自**结构**（逐 token 串行扫描 +
每 token 归约 + 48 CTA 的网格），所以只有算法级改写（WY / chunked 矩阵乘）或换并行
分解能救它；**不要再去调这个核的算式**。这也解释了为什么 §十.1 的值维分块同样无效。

顺带修好了这个基准本身：它自 `0ba23c5` 加入 `SW` 泛型后就没再传第 5 个泛型，
在 HEAD 上是**编译不过**的（`not enough generic arguments to instantiate const parameter SW`），
所以"chunked 核对逐 lane 核"的护栏一直没在跑。现在它能在 release 下直接量
（`cargo test --release -p infer-backend-cuda --features cuda -- --ignored recurrent_chunk_performance_gate --nocapture`，
约 2 秒），是后续改 delta 的快速 A/B 工具。

### 12.7 未解释但最便宜的线索：delta 在图内比隔离慢 2.6×【**已更正，见 12.9**：这不是缺口，是比错了核】

同一颗核、同一几何（KH=16、VH=48、D=128、SW=D），两个口径对不上：

| 口径 | LANES | 核时间 | 每 token | 来源 |
|---|---:|---:|---:|---|
| 隔离配对基准（清 L2） | 64 | 0.1322 ms | **2.07 µs** | `recurrent_chunk_performance_gate` (16,48,128,64) |
| 隔离配对基准（清 L2） | 128 | 0.2527 ms | 1.97 µs | 同上 (16,48,128,128) |
| 服务图内逐节点事件 | 64 | 333 µs | **5.2 µs** | prefill 每次 replay |

同一宽度（64）下差 **2.5×**。图内那 46 ms 的记账是自洽的（各节点之和 = 图总时间），仪器本身只解释约 7%
（同一 slot_verify：GRAPH_ONLY 26.68 ms vs 逐节点 28.5 ms），所以这不是测量口径问题。

如果这条差距能收掉，**不需要改算法**：replay 46 ms 里 delta 占 16 ms，压到隔离水平即
~6.5 ms，prefill 直接少 ~10 ms/replay（22%），long TTFT 394 → ~310 ms。

顺带：`recurrent_chunk_performance_gate` 现在也量 64 lane（模型 prompt 图实际捕的宽度），
省得再插值；它在 (16,48,128,64) 上是 chunked 0.132 ms vs 逐 lane 0.334 ms。

下一步的判别实验很便宜（改 `resident_recurrent_prefill_benchmark_tests.rs`，约 2 秒一轮）：
在候选图里插一个与服务器同量级的大 GEMM（例如 [64, 17408, 5120] 的 FP4/FP8 投影），
让 delta 在它之后跑。若 2.6× 复现，就是邻接 kernel 造成的 L2/带宽干扰或图内调度缺口，
方向转成"保护 delta 的工作集"或"把 delta 与邻居合并"；若不复现，则差距来自服务侧的
别的因素（例如同一 stream 上的依赖链、图节点间的发射间隙），要另找口径。

### 12.8 结论

- **prefill 的差距不在"GPU 没吃饱"**：利用率比 vLLM 高，每 token 设备时间却是 vLLM 的
  5.6×（720 µs vs 128 µs），其中非 GEMM 占 51%。
- **两个真靶子**：`delta`（34.7%，150× 于下界）与 `linear`（48.7%，138 TFLOP/s = FP8 峰值 16%）。
  `attention` 7.1% 同类但量级小一半。
- **已排除的低成本路线**（不要再试）：值维分块（§十.1）、tile 形状（§11.7、§12.7 相关）、
  delta 代数重排（§12.6）。
- **剩下的路只有三条**：① delta 的算法级改写（WY / chunked 矩阵乘）；② GEMM 委托
  （FP8 那条不需要 scale swizzle，覆盖 15.79 GB 里的 7.22 GB）；③ 先查 §12.7 的 2.6× 缺口。

### 12.9 更正 12.7：那 2.5× 不是缺口，是比错了核；真正的发现是快的那颗核默认关着

§12.7 把"隔离 2.07 µs/token vs 图内 5.2 µs/token"当成未解释的服务侧开销。**这个结论是错的，
原因是我拿 chunked 核的隔离时间去比服务里实际跑的核。** 量化检查：

- 服务里跑的是**逐 lane** 核。`recurrent_prefill::Workspace::new`
  （`crates/backend/cuda/src/resident/recurrent_prefill/workspace.rs:41`）对量化 checkpoint
  **默认返回空 workspace**，`record()` 随即返回 false，落到 `capture_state.rs` 的
  `TensorOp::Delta` 分支 —— 那是 `recurrent::delta`（per-lane），**每个 token 一次 launch**，
  64 个 token 的 launch 都记在同一个 graph node 下（profile 的边界是 node，不是 kernel）。
- 同一个配对基准里，per-lane 那一侧的隔离成本是 (16,48,128,64) **0.334 ms / 64 次 launch
  = 5.2 µs/token**，也就是 **333 µs / 64-token 节点 —— 与服务图内实测的 333 µs 完全一致**。

所以没有缺口：5.2 µs/token 就是逐 lane 核的价格，chunked 核（0.132 ms / 64 token =
2.07 µs/token）**比服务里实际跑的核快 2.5×**，只是它默认被关着。

### 12.10 把 chunked 递推打开：官方矩阵实测（**基线口径见 12.13，那里的同刻基线才是准的**）

`INFER_CUDA_CHUNKED_RECURRENT=1` 下，同一个 `long`、同一 profile、逐节点画像：

| 算子 | 逐 lane（默认） | chunked | 变化 |
|---|---:|---:|---:|
| **delta** | 16.00 ms（333 µs/节点） | **5.77 ms（120 µs/节点）** | **−64%** |
| linear | 22.46 | 21.26 | −5% |
| attention | 3.25 | 3.19 | −2% |
| 其余（norm/conv/gated_norm/…） | 4.36 | 4.19 | −4% |
| **每次 replay 合计** | **46.07 ms** | **34.41 ms** | **−25%** |
| 每 token | 720 µs | 538 µs | −25% |

即 chunked 核的隔离收益（120 µs vs 333 µs）**原样兑现到图内**，不需要额外的诊断。
官方矩阵（`chunked-mtp2-native-g1` vs `final-mtp2-native-g1`，同一 vLLM 参照）：

| case | 指标 | 逐 lane | chunked | Δ | vs vLLM（chunked） |
|---|---|---:|---:|---:|---:|
| short | TTFT | 55.9 ms | 45.3 ms | **−19.0%** | 1.469 |
| short | wall | 898 ms | 922 ms | +2.6% | 1.614 |
| short | TPOT | 13.36 ms | 13.92 ms | +4.1% | 1.620 |
| long | TTFT | 394.0 ms | 314.4 ms | **−20.2%** | 4.822 |
| long | wall | 1194 ms | 1118 ms | **−6.4%** | 1.747 |
| long | TPOT | 12.78 ms | 12.74 ms | −0.3% | 1.401 |
| batch4 | TTFT | 240.0 ms | 206.7 ms | **−13.9%** | 2.237 |
| batch4 | wall | 1454 ms | 1412 ms | **−2.9%** | 1.996 |
| batch4 | TPOT | 18.53 ms | 18.28 ms | −1.4% | 1.988 |
| hot_long | TTFT | 81.7 ms | 72.9 ms | **−10.8%** | 0.337 |
| hot_long | wall | 802 ms | 862 ms | +7.4% | 1.079 |
| hot_long | TPOT | 11.39 ms | 12.54 ms | +10.2% | 1.354 |

三个 TTFT 比值之和 1.81+6.04+2.60 = **10.45 → 8.53**（short/long/batch4），四个 wall 比值之和
6.50 → 6.44（基本持平）。

### 12.11 short/hot_long 的 TPOT 回退机制：是 MTP 接受率，不是核成本【确证】

§（前面第 8 轮）把"short/hot_long 的 wall/TPOT 回退 8–10%"记为**待查**。本轮从
`stream_events` 的 token 到达间隔反推每步产出（同一步内接受的 token 到达间隔 <2 ms），
再算**每步解码成本**：

| case | 每步解码 | 步数 | TPOT |
|---|---:|---:|---:|
| short 逐 lane → chunked | 30.15 → **29.43 ms** | 29 → **33** | 13.87 → 15.41 ms |
| long 逐 lane → chunked | 30.00 → **30.32 ms** | 28 → **25** | 13.32 → 12.02 ms |
| hot_long 逐 lane → chunked | 29.88 → **30.41 ms** | 24 → **26** | 11.37 → 12.54 ms |

**单流三条用例的每步解码成本差 ≤2%（在噪声内），变的是步数。** 步数变化来自 token 变了：
两条路的输出 token 只有 31/64（short）、38/64（long）、**1/64（hot_long）** 相同，
内容不同 → 草稿/目标的吻合率不同 → 每个产出 token 需要的 verify replay 次数不同。
`batch4` 因为 4 条 lane 的到达事件交织，间隔反推不可靠（10.09 → 11.50），但它的 TPOT 几乎没动。

结论：chunked 预填充**对解码没有每步成本**（与"chunked 只作用于预填充图"的代码事实一致），
short/hot_long 的回退是**内容相关的接受率二阶效应**，不是可以调掉的核开销。

**因此默认开关的取舍应该重新陈述**：TTFT 收益（−11~−20%）是确定的；wall/TPOT 的 ±10% 是
内容相关的、方向不可预测的（同一个模型上 long 的接受率反而升高、TPOT −10%）。
也就是说，关掉它的代价是一个**数值口径选择**（输出差 1 ULP 起），而不是性能。

### 12.12 结论（修订版）

- 差距不在"GPU 没吃饱"：native 利用率更高，每 token 设备时间是 vLLM 的 5.6×，其中非 GEMM 过半。
- **最大的单项已经写好并且在跑得通的形态下实测过：chunked 递推核**（delta −64%，
  每次 replay 46.07 → 34.41 ms，TTFT 全面 −11~−20%）。它默认关闭是**数值口径**决定
  （cuTile lowering 的 1 ULP 差 → 贪心续写不同），不是性能。
- 打开之后剩下的靶子：`linear` 占 replay 的 **61.8%**（138 TFLOP/s = FP8 峰值 ~16%，
  704 GB/s = 39% 带宽）；`attention` 9.3%、200 µs/节点，与 delta 同类。
  **§12.14 的委托否定结论已被 16.10 的真实 API/计算证据修正**：FP8 scalar raw GEMM
  加原顺序 epilogue，以及 NVFP4 无损 scale 重排，均不必重新量化 checkpoint。
- 已彻底关闭的低成本路线：值维分块（§十.1）、tile 形状（§11.7）、delta 代数重排（§12.6）、
  以及"图内比隔离慢"这条伪线索（§12.9）。

### 12.13 同刻基线、跨小时漂移，以及 mtp0 的复核（本轮关键修正）

12.10 的表用的是 **23:14** 的 per-lane 基线，而 chunked 跑在 **00:0x**。补一条同刻基线后
发现机器的解码吞吐会**跨小时漂移**：

| mtp0 用例 | 22:03 的 per-lane 基线 → 现在重跑 | 漂移 |
|---|---:|---:|
| short TPOT | 16.39 → 17.83 ms | **+8.8%** |
| long TPOT | 16.44 → 18.04 ms | **+9.7%** |
| batch4 TPOT | 21.14 → 23.15 ms | **+9.5%** |
| hot_long TPOT | 16.90 → 18.53 ms | **+9.6%** |

也就是说先前的"mtp0 上 chunked 让 TPOT 全面 +10%"**是漂移，不是代码**：拿同刻基线比，
chunked 在 mtp0 上 TPOT **不动**（−0.0 ~ −2.1%）。此后所有 A/B 都必须用**同刻或交错**的
基线，跨小时的绝对数字不可比（与 §11.5 是同一类坑）。

**mtp0（无投机）同刻对比**：

| case | TTFT | TPOT | wall |
|---|---:|---:|---:|
| short | **−14.0%** | −0.1% | −0.7% |
| long | **−31.8%** | −0.0% | −6.1% |
| batch4 | **−16.6%** | −2.1% | −2.1% |
| hot_long | **−14.9%** | −0.5% | −1.3% |

**四项全面变好，没有任何回退。** 这是 §12.11 机制的独立验证：关掉投机以后，token 内容
不再影响解码成本，于是 chunked 只剩下预填充的收益，回退随之消失。

**mtp2（投机开）同刻对比**（两轮 chunked 一致）：

| case | TTFT | TPOT | wall |
|---|---:|---:|---:|
| short | **−13.2%** | +7.9% | +6.5% |
| long | **−20.2%** | −3.0% | **−8.0%** |
| batch4 | **−14.3%** | −6.2% | **−10.4%** |
| hot_long | **−11.3%** | +8.9% | +7.0% |

所以最终口径是：

- **TTFT 在所有 profile、所有用例上稳定下降**（mtp0 −14~−32%，mtp2 −11~−20%），这是
  确定的预填充收益（replay 46.07 → 34.41 ms，delta −64%）。
- **TPOT/wall 只有在开了投机时才会出现 ±9% 的摆动**，方向由 token 内容（→ MTP 接受率）
  决定，不是核成本；mtp0 上它们是不动的。
- 因此开关的取舍是一个**输出数值口径**决定，而不是性能决定；性能这一侧的代价基本不存在。

**适用范围**：chunked 递推只对**含 gated-delta 层**的量化 checkpoint 有意义。本仓两个 profile 里，
27B（`Qwen3_5ForConditionalGeneration`，48 层线性注意力 + 16 层全注意力、NVFP4/FP8）适用；
2B（`Qwen3VLForConditionalGeneration`，标准全注意力、无 delta 层）**根本没有这条路径**
——`Workspace::new` 只会为 `TensorOp::Delta` 建 buffer，没有 Delta 节点就是空 workspace，
`record()` 直接返回 false。所以这条开关对 2B 是 no-op，两个 profile 之间没有回退风险。

### 12.14 早期委托模式探测【历史结论已修正，见 16.10】

本节的 OUTER_VEC 模式失败记录仍有效，但旧探针遗漏 SCALAR_32F，不能据此排除所有
FP8 委托路径。NVFP4 scale 的无损布局重排也不等于重新量化。以下否定结论属于当时的
推断，2026-10-11 已由真实计算探针推翻；服务收益仍待验证。

§二 杠杆 A 记着一条推断：FP4 需要 128×64 的 scale swizzle，但 **"FP8 那条更简单：
`CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F` 正好是每行 A 缩放 × 每行 B 缩放，与
`record_fp8` 的 per-token × per-channel 一一对应，无需 swizzle"**。本轮把这条用真实
API 打了一遍（CUDA 13.x 头文件 + `libcublasLt.so.13`，RTX 5090 / sm_120 / 驱动 615），
**结论是这条推断不成立**。

**先确认规格（来源：nv 的 `cublasLt.h`，不是转述）**：

| 枚举 | 值 | 语义 |
|---|---:|---|
| `CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F` | **3** | "vectors of CUDA_R_32F … expected to have **M and N elements** respectively；A 的第 i 个与 B 的第 j 个相乘" |
| `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` | 2 | e4m3 数据 + UE8M0 每 32 元素块缩放（mxfp8） |
| `CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F` | 4 | f32 每 128 元素块缩放 |
| `_DESC_A/B_SCALE_POINTER` / `_MODE` | 17 / 18 / 31 / 32 | |

映射是**对得上**的：我们的 `out[m,n] = act[m,k]·w[n,k]^T`，cublas 侧 M_cb = 输出通道、
N_cb = token，所以 A-slot（权重）的 scale 长 N_cb 个 = 每个输出通道一个
（正是 checkpoint 里的 `weight_scale [n, 1]` BF16），B-slot（激活）长 M_cb 个 = 每个 token
一个（正是 `quantize` 的 `qs`）。**即 27B 的 FP8 投影（in_proj_qkv/z、out_proj、q/k/v/o，
共 7.22 GB）本来就是"每通道 × 每 token"的口径。**

**但这条 mode 在本机拿不到算法**（`cublasLtMatmulAlgoGetHeuristic` 的返回）：

| mode | FP8(e4m3) 数据 | BF16 数据 |
|---|---|---|
| **3 = OUTER_VEC_32F** | **status 15 = NOT_SUPPORTED**（256³、12×10240×5120、64×17408×5120 全一样） | status 7 = INVALID_VALUE |
| 2 = VEC32_UE8M0 | status 0，**有算法**（上述所有形状） | — |
| 4 = VEC128_32F | status 15 | status 7 |
| 1 = VEC16_UE4M3 | status 7 | — |

（TN 与 NN、带/不带 workspace 上限都试过；mode 3 从未返回过算法。）

**当时误将 OUTER_VEC 视为唯一无需重新量化的模式，因而过度排除了委托路径。**
以下描述的是当时探测到的 FP8 mode 2（mxfp8）代价，不涵盖后续验证的 scalar 路径：

1. 权重必须**重新量化**成 e4m3 + **UE8M0（2 的幂）每 32 元素 K 块**的 scale，而 checkpoint
   存的是每通道 BF16 scale —— 这是换量化口径，不是换 GEMM；
2. 激活侧也要新增一个产出同布局的 quantize kernel；
3. mxfp8 的 A/B scale 按 nv 自己的参考实现（`quack/bench/cublaslt_quant_out.py`）是
   `(rm, rk, 32, 4, 4)` **分块/重排**布局，不是扁平向量 —— "无需 swizzle" 在 mode 2 上
   同样不成立（本轮的 heuristic 接受扁平张量，但没有验证核实际按哪种布局读，**未验证**）。

**当时的“B2b 路线关闭”结论已撤回（见 16.10）。** 上述失败只适用于探测过的模式；
必须区分 checkpoint 数值重编码与可逆的 scale 字节重排。要拿这部分收益，前提是先做一次**离线重新量化**（把权重存成
mxfp8 或 NVFP4 的 cublasLt 布局），那是模型打包层的工程，不是推理核的调优。在这之前，
decode 的 `linear` 只能靠自研核效率（138 TFLOP/s = FP8 峰值 ~16%）。

复现（本机，venv 里有 torch；脚本在 /tmp，未入库）：

```sh
artifacts/vllm-compare/bin/python /tmp/lt_ws.py    # mode 3 vs mode 2 的算法可得性
```

### 12.15 attention：resident 路是纯 SIMT，而仓库里已经有一颗张量核 SDPA【确证 + 可复用】

§十.2 把 attention 记成"7.5%、556 µs/节点"，但没有说清它贵在哪。本轮把它量清楚了。

**规模**（chunked 预填充，每次 replay 34.41 ms）：attention **3.19 ms / 16 节点 = 199 µs/节点**，
占 9.3%；decode（`slot_verify` 28.5 ms）里是 1.29 ms / 16 节点 = 80 µs/节点，占 4.5%。

**它随活跃 KV 长度线性增长**（同一次 profile，16 层的节点时间彼此几乎相同）：

| position（KV 长度） | 256 | 320 | 384 | 448 |
|---|---:|---:|---:|---:|
| 每节点 | 164 µs | 190 µs | 217 µs | 225 µs |

约 **0.32 µs / KV token / 层**，所以不是固定开销，是逐 KV 的成本。

**贵在哪（读码确认）**：`resident/attention_prefill.rs::decode`
（decode 侧的 `attention_decode.rs` 同构）是**纯 SIMT f32**——
`convert_tile` 把 fp8 的 K/V 转成 f32，然后 `query * key` + `reduce_sum(…, 1i32)` 算分数、
softmax 后再 `reduce_sum` 做 PV。**整个文件没有一处 `mmaf`**。也就是说：

- 每个 32-token KV block 都要做**跨线程归约**（score 的 key 维归约、block max、PV 归约），
  和 §12.3 的 delta 是同一种病；
- 网格是**每个 (lane, query_head) 一个 CTA**（预填充 64×24 = 1536 个），GQA 的 6 个 query head
  各自把同一个 KV head 再读一遍（组内不复用）；
- 实测 ≈ **3 TFLOP/s**，而 fp32 SIMT 峰值约 122 TFLOP/s（170 SM × 128 lane × 2 × 2.8 GHz）
  —— 核内还有 ~40× 的余量，且这部分算力本来可以用张量核（fp8 输入已在缓存里）。

**可复用的现成资产【新信息】**：仓库里**已经有一颗张量核 SDPA**，只是没接给 LLM：

- `crates/backend/cuda/src/attention/kernels.rs`（`sdpa::attention`，含 `mmaf`，模板参数带
  `GROUP`（GQA）、`ONLINE`（online softmax）、`MASK`（含 `WINDOW_MASK = 2`）、Q/K/PIPE/DV 分块）、
  `attention/decode.rs`（单 query 变体）、`attention/plan.rs`（`DenseAttentionPlan`，
  "caller-owned buffers and stream"，用 `infer_kernel_api::attention::AttentionDescriptor` 描述）；
- 它有自己的**数值闸门与基准**：`tests/unit/attention_gate_check.rs`（`INFER_ATTENTION_GATE`，
  `make check-attention`）、`tools/attention/*`（fixture 生成 + Candle 对照 `infer-attention-candidate`）、
  `tools/bench/attention.sh`、以及 `attention/benchmark_check.rs` 的计时输出；
- **但它只被 vision 用**（`crates/backend/cuda/src/vision/program.rs:574/855`），
  resident 的 LLM 路走的是上面那颗 SIMT 核。

**所以 attention 这条不是"从零写 flash attention"**，而是"把已有的张量核 SDPA 接进 resident 路"。
要处理的差异是明确的四条：① 输入是 **fp8 KV cache**（SDPA 入口是 `Tensor<f32>`，需要转换或加一个
fp8 变体）；② KV 布局是 `[kv_heads, capacity, head_dim]` + 每个 chunk 的 append；
③ 每 lane 位置不同（metadata 的 base/count/offset）；④ 滑动窗口与因果掩码语义要对齐。
预计收益上限：attention 3.19 ms → 亚毫秒量级（预填充约 −8%），decode 同向。

### 12.16 把现成的张量核 SDPA 拿到 LLM 几何上实跑：通过数值闸门，且快 3.1–3.5×【确证】

§12.15 提出"resident 的 attention 换张量核"。本轮**没有改一行仓库代码**，直接用仓库已有的
attention 闸门（`INFER_ATTENTION_GATE` + `tests/unit/attention_gate_check.rs`）在**27B 的
attention 几何**上跑了现成的 `DenseAttentionPlan`：在 /tmp 造了一份同格式的 fixture
（3 个 case，按 `tools/attention/attention_fixtures.py` 的 schema 与 `packed()` 布局），
`cargo test --release ... -- --ignored attention_native_gate`。

| case（q × KV × heads × kv_heads × D，causal） | 张量核 SDPA | resident SIMT 核 | 倍数 | F64 oracle 误差 |
|---|---:|---:|---:|---:|
| 64 × 448 × 24 × 4 × 256（最后一个 chunk） | **72.4 µs** | 225.3 µs | **3.11×** | 5.65e-7（参考幅值 0.085） |
| 64 × 320 × 24 × 4 × 256 | **53.6 µs** | 190.5 µs | **3.55×** | 7.30e-7（参考幅值 0.144） |
| 3 × 511 × 24 × 4 × 256（decode 形） | 88.3 µs | 80 µs / 12 lane | 0.9×（**不划算，见下**） | 4.68e-7 |

三次都 `test result: ok`，即**数值闸门通过**：D=256、GQA 6:1、causal、448 KV 这些我们的真实
几何，现成 SDPA 都支持（闸门的 tile 选择给的是 `[32,32,0]`）。

**收益**：预填充 attention 3.19 ms/replay → 约 1.0 ms，**省 ~2.2 ms = chunked replay 的 6.4%**
（per-lane replay 46.07 ms 上同样是这 2.2 ms）。注意这只算了 attention 本身：真接进 resident 路
还要把 fp8 KV 转成 SDPA 入口要求的 f32（每层 918 KB fp8 → 3.7 MB f32，×16 层 ≈ 74 MB/replay
≈ 41 µs），或者给 SDPA 加一个 fp8 变体；即使加上这笔，净收益仍在 3× 量级。

**decode 形的注意点【未验证的边界】**：3 个 query 时 SDPA 反而慢（88 µs vs 80 µs），因为
`DenseAttentionPlan` 的 tile 是 32——3 行只填 9%，其余 29 行是掩码。decode 要走的是
`attention/decode.rs` 的单 query 变体（"Single-query SDPA avoids computing padded query rows"），
本轮的 fixture 没有覆盖它，所以 **decode 侧能不能同样提速还没量**，别当成已知结论。

**结论**：attention 这条路的可行性已经用真实几何 + 数值闸门 + 计时确认，剩余工作是集成而非
算法：① fp8 KV 输入（转换或 fp8 变体）；② `[kv_heads, capacity, head_dim]` 布局 + 每 chunk 的
append；③ 每 lane 位置：闸门与 `DenseAttentionPlan` 的 causal 掩码只接受**一个** `query_start`
（`plan.rs:216 => [1, 1, offset(query_start), 0, 0]`），而 resident 的 SIMT 核是按 lane 算
`position = base + lane + offset` 的（`attention_prefill.rs:82`）。**单序列 chunk 下两者等价**
（把 `query_start` 取成 `base + offset`，lane 内单调），但**多序列打包到一个 chunk 时 lane 位置
不是单调的，现有 SDPA 掩码表达不了**。所以第一步应限定"单序列 chunk"，多序列要么先按序列拆开，
要么给 SDPA 加一个逐 lane 位置数组的掩码变体——这是一个需要先确认的集成边界，不是细节；
④ 滑动窗口/因果语义对齐；⑤ decode 侧改用单 query 变体并复量。

复现（fixture 脚本在 /tmp，未入库；格式与仓库的 `tools/attention` 一致）：

```sh
artifacts/vllm-compare/bin/python /tmp/llm_attn_fixture.py --out /tmp/llm-attn/fixtures.safetensors
INFER_ATTENTION_GATE=/tmp/llm-attn/fixtures.safetensors \
  cargo test --release --locked -p infer-backend-cuda --features cuda --lib -- \
  --ignored attention_native_gate --nocapture
```

### 12.17 B3（decode 逐 lane 录制）的机制与规模：Row 派发占 verify replay 的 21.7%【确证】

§二 杠杆 B3 只写了一行"`slot_verify` 里 `Dispatch32::Row` 的逐 lane 录制改成跨 lane 批量
kernel"。本轮把它变成有数字、有边界、有拦路石的任务。

**机制（读码确证）**：`batch.rs:175-229` 捕获槽位图时按节点分派：

- `TensorOp::Linear` → `batch_projection::record_slots`，**一条 batched kernel**；
- 其余走 `dispatch32(node, weights)`（`batch.rs:655`）：
  - **`Batched`**（一次 `record()`，一条 kernel）：`Silu`/`Sigmoid`/`Add`/`Multiply`（输入不是常量）、
    `Norm`/`Split`（input0 不是常量）、`GatedNorm`；
  - **`Row`**（`for lane in 0..width { record_row(node, lane) }`，**每个节点 `width` 条 kernel**）：
    其余全部 —— 在本模型上就是 **`Conv`、`Delta`、`Attention`**。

**规模**（同一次 profile，`slot_verify` 中位 26.25 ms，1155 节点）：

| 算子 | ms | 节点 | µs/节点 | 派发 |
|---|---:|---:|---:|---|
| linear | 18.10 | 497 | 36.4 | batched |
| **delta** | **2.32** | 48 | **48.4** | **Row ×4** |
| **conv** | **2.26** | 48 | **47.0** | **Row ×4** |
| **attention** | **1.11** | 16 | **69.7** | **Row ×4** |
| norm | 0.85 | 161 | 5.3 | batched |
| rope | 0.51 | 32 | 15.8 | batched |
| add / multiply / silu / sigmoid / split / gated_norm | 1.52 | 352 | 3.9–6.2 | batched |

**Row 三项合计 5.69 ms = verify replay 的 21.7%。**

**对照说明固定开销有多大**：batched 的那些节点每次 3.9–6.2 µs；而 conv 每个节点 47 µs 是
**4 条 kernel**（≈11.8 µs/条），delta 48.4 µs（≈12.1 µs/条）。conv4 一条的实际工作量只有
80 个 CTA × 4-tap × 128 通道（微秒级），所以这 11 µs 里绝大部分是**逐条 launch 的固定开销**。
预填充侧同样两个算子只有 13.5 µs/节点（conv 走 `conv_prefill` 的 2 条 kernel、delta 走
chunked 核一条），正是"批起来"之后的成本。

**拦路石（必须先解决，否则整条 B3 会做歪）**：预填充那两个 batched 核**不能直接搬到 decode**。
它们扫的是**同一份状态**上的一段 *时间*（chunk 内 LANES 个 token 共享一组 h0/h1/h2 或一个
D×D state）；而 decode 要批的是**4 个互不相干的槽位状态**（`lane_states[lane]` 各自一套）。
所以 B3 需要的是**带 slot 维的新 kernel 变体 + 池化状态布局**（把每槽的 conv/delta 状态收进
一个可索引张量），不是"调用另一个已有函数"。

**预估收益**：若 conv/delta 各变成"每节点一条 kernel"（~10–15 µs），可省约 **3 ms ≈ decode
tick 的 11%**（batch4 tpot 19.03 → ~16.9 ms，比值 2.02 → ~1.80）；attention 的 16 个节点另有
空间（§12.16 已量到张量核 SDPA 更快，但那是预填充形；decode 形要用单 query 变体，未量）。

### 12.18 attention 换张量核的落地方案（设计已定，代码下一轮写）

§12.16 用仓库现成的 `DenseAttentionPlan` 量到 3.1–3.5×，但**不能直接接**：`plan.rs` 要求
调用方提供 **f32** 的 K/V 缓冲，且 `plan.shapes()` 的尺寸在建 plan 时固定；我们的 KV 是
**fp8 且按容量分配**（池 2048 槽、私有程序可到 32768），转成 f32 是 134 MB/层，不可行。
所以正确形态不是"用 plan"，而是**把已验证的内层搬到 resident 核里**：保留 fp8 KV cache、
保留运行时的 KV 上界，只把 QK^T / PV 换成张量核。下面是把 `attention/kernels.rs::sdpa::attention`
改造成 resident 变体的完整映射（已逐项核对过现有核与 SDPA 的语义）。

**操作数映射（全部可在不改布局的前提下表达）**

| SDPA 角色 | resident 张量 | 布局 | 取法 |
|---|---|---|---|
| `query` | arena 里的 q | 现为 `[lanes*HEADS, D]`，可 view 成 `[lanes, HEADS, D]` | `q.partition([QT,1,D]).load([tile, head, 0])` → reshape `[QT,D]` |
| `key` / `value` | fp8 KV cache | `[kv_heads, capacity, D]` | `keys.partition([1,KB,D]).load([kv_head, block, 0])` → reshape `[KB,D]`（**与现有 SIMT 核同一句**） |
| `out` | arena 输出 | 现为 `[lanes*HEADS, D]`，view 成 `[lanes, HEADS, D]` | `out.partition([QT,1,D]).store(.., [tile, head, 0])` |
| `scale` | `1/√D` | 现有核已算 | 沿用 |
| `k_scale`/`v_scale` | 现有核已有 | — | `convert_tile` 之后再乘（与 SIMT 核一致） |

**分块与网格**：`QT`（每个 tile 的 lane 数，建议 32 或 64）× `KB`（KV 块，32/64）；
网格 = `(lanes/QT) × HEADS`。即"同一 head、连续 lane"为一组 —— 这正是 §12.16 里已经实测过的
分块（闸门跑 `queries=64` 时 grid 就是 2×24，72 µs），**所以分块本身不需要再验证**。
`QT` 越大越省 K/V 重复读（每个 KV head 被 GROUP=6 个 head 各读一遍，与 QT 无关）。

**掩码**：tile 内第 i 行的位置是 `base + lane0 + i + offset`，同一 tile 内单调 ⇒ 可以套用
SDPA 的 `visibility` 形状检查：`MASK=1`（causal）+ 运行时 `query_start = base + lane0 + offset`。
§12.16 的 fixture 正是这个用法（`query_start=384`、`kv=448`、`tokens=64`、causal），已过闸门。
滑动窗口用 `MASK=2` 同理。

**唯一未决的语义：`lane >= count` 的非活跃行**（已核到具体后果，写核时必须一起做）：

- SIMT 核是 `if lane < count && position >= 0 { 计算 } else { out.store(0.0f32) }`
  （`attention_prefill.rs:83/145`）—— 非活跃行**显式写 0**；
- `append` 只写 `for lane in 0..count`（`attention_prefill.rs:30`），所以非活跃行在 KV cache 里
  是**陈值**；
- tile 化的核没有 per-row early-out。若只加 `valid &= (row < count)`（row = tile 内 lane 序号 +
  lane0），该行的分数会全被掩成 `MASKED`，于是 `next = MASKED`、`exp(MASKED - MASKED) = 1`，
  得到 `acc = Σ v`、`row_sum = KB` —— **有限但错误的值**（不是 NaN）。要**和 SIMT 路逐位一致**
  （下游按 metadata 忽略这些行，但显存里的值会被后续 kernel 读到），还需要在结尾对非活跃行
  `select` 成 0。
- 好消息：attention 的归约只发生在 **KV 维**，没有跨行归约，所以非活跃行的值不会污染活跃行；
  必须处理只是为了避免写出"有限但无意义"或（若 row_sum 为 0）NaN 的值。

**代码骨架**（`attention_prefill.rs` 新增一个 `decode_tiled` entry，不动现有 `decode`）：

```rust
#[cutile::entry()]
fn decode_tiled<E: ElementType, const D: i32, const DV: i32, const GROUP: i32,
                const QT: i32, const KB: i32>(           // 网格 (lanes/QT) x HEADS
    out: &mut Tensor<f32, { [-1, -1, DV] }>,             // [lanes, HEADS, DV]
    q:   &Tensor<f32, { [-1, -1, D] }>,                  // [lanes, HEADS, D]
    keys:   &Tensor<E, { [-1, -1, D] }>,                 // [kv_heads, capacity, D]
    values: &Tensor<E, { [-1, -1, DV] }>,
    metadata: &Tensor<i32, { [-1] }>,
    window: i32, k_scale: f32, v_scale: f32,
) {
    let pid = get_tile_block_id();
    let head = pid.1; let kv_head = head / GROUP;
    // base/count/offset 从 metadata 读（与现有核逐字相同）
    let qt: Tile<f32, {[QT, D]}> = q.partition(shape![QT,1,D]).load([pid.0, head, 0]).reshape(shape![QT,D]);
    let mut acc: Tile<f32, {[QT, DV]}> = constant(0.0, shape![QT, DV]);
    let mut row_max: Tile<f32, {[QT]}> = constant(MASKED, shape![QT]);
    let mut row_sum: Tile<f32, {[QT]}> = constant(0.0, shape![QT]);
    let kp = keys.partition(shape![1, KB, D]);
    let vp = values.partition(shape![1, KB, DV]);
    let last = (base + offset + pid.0*QT + QT - 1) / KB + 1;      // 运行时上界
    for block in 0..last {
        let k: Tile<f32,{[KB,D]}> = convert_tile(kp.load([kv_head, block, 0]).reshape(shape![KB,D])) * k_scale.broadcast(shape![KB,D]);
        let v: Tile<f32,{[KB,DV]}> = convert_tile(vp.load([kv_head, block, 0]).reshape(shape![KB,DV])) * v_scale.broadcast(shape![KB,DV]);
        let raw = precise_mma(qt, k.transpose()) * scale.broadcast(shape![QT,KB]);
        let valid = /* causal + window + (row_lane < count) */;
        let s = select(valid, raw, constant(MASKED, shape![QT,KB]));
        let next = max_tile(row_max, reduce_max(s, 1i32));
        let p = exp(s - next.reshape(shape![QT,1]).broadcast(shape![QT,KB]));
        acc = acc * exp(row_max - next).reshape(shape![QT,1]).broadcast(shape![QT,DV])
            + precise_mma(p, v);
        row_sum = row_sum * exp(row_max - next) + reduce_sum(p, 1i32);
        row_max = next;
    }
    out.partition(shape![QT,1,DV]).store((acc / row_sum.reshape(shape![QT,1]).broadcast(shape![QT,DV])).reshape(shape![QT,1,DV]), [pid.0, head, 0]);
}
```

`precise_mma` / `MASKED` 从 `attention/kernels.rs` 抄一份即可（它是模块内的普通 `fn`，
用 `mmaf` 做 bf16 补偿乘；`attention_prefill.rs` 里现在一次 `mmaf` 都没有）。

**接线与验收**：`record_prefill_attention` 里按 `INFER_CUDA_TILED_ATTENTION` 选择新核（默认惰性）、
q/out 以 `[lanes, HEADS, D]` 视图记录；验收顺序是 ① `make local-build` + 现有单测/GPU 测试，
② `examples/provider_check`（它是**示例**不是测试，不需要动 test inventory，正好当数值校验器），
③ 官方矩阵与默认路径比 token（`token_mismatches` 应回到同一量级）。

预期：预填充 attention 3.19 ms → ~1.0 ms（replay −6.4%）；decode 形要另做单 query 变体（未量）。

### 12.19 试写张量核 attention 的结果：卡在 grid 推导，需要 KV cache 改成 token-major【实测失败】

按 §12.18 的方案真的写了一版 `decode_tiled`（87 行，含 `precise_mma`/`MASKED` 的拷贝）并挂在
`INFER_CUDA_TILED_ATTENTION` 后面，**编译、clippy、单测、GPU 测试全过，服务也能跑起来**
（不崩、输出有限），但和默认路径对不上：

| case | rep | 默认 64 token vs tiled 的前缀一致率 |
|---|---|---|
| short | 0/1/2 | 0/64、1/64、0/64 |
| long | 0/1/2 | 1/64、0/25（提前 EOS，只出 24 token）、0/64 |
| batch4 | 0/1/2 | 1/64、0/64、0/64 |
| hot_long | 0/1/2 | 1/64、1/64、1/64 |

**0–2% 一致率意味着数值是错的，不是浮点重排**（重排只会偶发翻转，不会是 0%）。
batch4 wall 也从 1.44 s 涨到 2.10–2.25 s。已**整段回滚**，工作区回到提交状态。

**原因（由 shape 约定推出，未做编译器级验证）**：cuTile 的 grid 不是显式给的，而是从各参数的
partition 形状推出来的；SDPA 之所以能用 `pid.1` 当 head 索引，是因为它**四个参数全是 2-D**：
`query:[-1,-1]` 与 `key:[-1,-1]` 都按 `[Q,D]`/`[K,D]` 切，于是 `key` 的 axis-1 分块数正好是
`kv_heads`（`kv_heads*D / D`），与 query 的 head 轴对齐。而我们的 KV cache 是 **head-major**
`[kv_heads, capacity, D]`，只能按 `[1, KB, D]` 切 ⇒ 分块数是 `(kv_heads, capacity/KB, 1)`，
axis-1 是 **capacity 方向**（32768/32 = 1024），与 query 的 24 个 head 根本对不上；
无论编译器取逐轴 max 还是 min，`pid.1` 都不可能是 head 索引 ⇒ 每个 CTA 去读错误的 KV 区域。
这正好解释"有限、不崩、但完全不对"。

**下一步的正解是把 KV cache 改成 token-major `[capacity, kv_heads*D]`**（SDPA 的约定），
这样 K 的 `[KB, D]` partition 就是 `(capacity/KB, kv_heads)`，`pid.1` 恢复成 kv head。
代价是这条链上的写入/读取要一起改：`append` 的 store 索引、SIMT 的 `decode`（prefill+decode
两处）、池化的 KV 分配/快照与 prefix cache。**这属于"为了换成张量核要先动 KV 布局"**，
而不是换一个核 —— 这个判断是本次实测换来的，比 §12.18 的估计更硬。

复现：把 12.18 的 `decode_tiled` 写回 `attention_prefill.rs`、在 `record_prefill_attention`
里按 `INFER_CUDA_TILED_ATTENTION` 接线，然后跑官方矩阵比 token（本次 run-id
`tiled-mtp2-native-g2`，报告在 artifacts 里，未入库）。

### 12.20 chunked 递推现在是正式开关：`--chunked-recurrent`（默认仍是逐位精确的逐 lane 路）

§12.10–12.13 量到的收益此前只能靠环境变量 `INFER_CUDA_CHUNKED_RECURRENT` 打开，操作方既看不到
也没法在部署描述里表达。本轮把它接成**正式的 CLI 选项**（`crates/service/cli/src/arguments.rs`），
一路 `Selection` → `LoadOptions` → `ProgramWeights.chunked_recurrent` → `recurrent_prefill::Workspace`；
环境变量仍然有效（两者取或），**默认不变**（量化 checkpoint 仍走逐位精确的逐 lane 路）。

同刻基线对照（同一窗口、同一 vLLM 参照，server 用 `--extra --chunked-recurrent` 起）：

| case | 指标 | flag off | `--chunked-recurrent` | Δ | 环境变量（早先一次） |
|---|---|---:|---:|---:|---:|
| short | TTFT | 53.0 ms | 44.7 ms | **−15.7%** | 46.9 ms |
| short | TPOT | 13.11 | 14.16 | +8.0% | 14.45 |
| short | wall | 880 | 936 | +6.4% | 957 |
| long | TTFT | 399.6 | 319.5 | **−20.1%** | 317.0 |
| long | TPOT | 13.07 | 12.60 | −3.6% | 12.57 |
| long | wall | 1227 | 1114 | **−9.2%** | 1110 |
| batch4 | TTFT | 190.0 | 162.2 | **−14.6%** | 204.1 |
| batch4 | TPOT | 17.97 | 16.30 | −9.3% | 17.65 |
| hot_long | TTFT | 85.7 | 72.0 | **−16.0%** | 75.7 |
| hot_long | TPOT | 11.45 | 12.42 | +8.4% | 12.48 |

**CLI 路与环境变量路逐格一致**（long TTFT −20.1% vs −20.2%、short −15.7% vs −13.2%、hot_long wall
857 vs 863 ms、TPOT 14.16 vs 14.45），说明接线确实走到了 `Workspace::new`。TTFT 全用例下降
14.6–20.1%，与 §12.13 的结论一致；short/hot_long 的 TPOT +8% 仍是 §12.11 的 MTP 接受率二阶效应
（mtp0 上为零，已实测）。

本轮这一窗口里 batch4 的 wall 绝对值偏高（2200 ms 级，平时 ~1500 ms），说明机器上还有别的负载；
但配对内的 TTFT/TPOT 差值与早先窗口吻合，所以结论以配对比值为准（§12.13 的教训）。

### 12.21 prompt GEMM tile 也扫过了：现默认 [64,64] 就是最优（§11.7 只覆盖了 decode 宽度）

§11.7 的 tile 扫描是靠 `INFER_CUDA_QUANT_TILE` 覆盖器做的，而那个覆盖器只对 **decode 宽度**
（`rows <= QUANT_GEMM_TILE[0]`）生效，**预填充用的 [PROMPT_GEMM_TILE_ROWS, 64] 从没在交付配置里
扫过**（constants.rs 只记了"全局加宽到 [64,128] 端到端变差"，那是 Stage A 之前、非 chunked 的旧配置）。
本轮把覆盖器改成对**所有宽度**生效，在**当前配置**（`--chunked-recurrent`）下重扫了一次。

一次 run 一个 tile，读 profile 的各图总量（中位，ms）：

| tile | prefill | prefill_last | slot_verify |
|---|---:|---:|---:|
| 64×32 | 9.66 | 31.74 | 42.48 |
| **64×64（现默认）** | **8.63** | **31.26** | 30.52 |
| 64×128 | 8.83 | 34.03 | 30.18 |
| 64×256 | 10.15 | 46.76 | 42.32 |

服务侧同向（long 三次重复的 wall 中位）：64×64 = 1.235 s、64×128 = 1.249 s、64×32 = 1.604 s、
64×256 = 1.641 s。

**结论：[64,64] 已经是最优，不改。** 32 和 256 差 20–35%，128 在 prefill_last 上反而差 9%
（只在 slot_verify 上小胜 1%，属噪声）。所以**预填充 GEMM 只跑到 39–41% 带宽不是 tile 问题**，
`constants.rs` 里"加宽 N 端到端变差"的旧记录在当前配置下依然成立——剩下唯一的路就是那条已经写在
§二的 kernel 级改动（K 向 `cp.async.bulk`/TMA 双缓冲、把每 k 步的在飞字节从 4 KB 提上去），
那是重写 kernel，不是调参。

覆盖器保留在 `constants.rs`（`INFER_CUDA_QUANT_TILE=rows,cols`，进程内读一次，默认不生效），
这样几何或 kernel 变了以后这条结论可以重测；复现见上表，脚本在 /tmp/tile-sweep.sh（未入库）。

### 12.22 SDPA 的 TMA 流水线搬到 GEMM 上：无复现收益，已回滚【实测否证】

§二 的杠杆 A 说预填充 GEMM 的瓶颈是"每 k 步在飞字节不足（weight tile [64,128] 仅 4 KB）"，
方向是 K 向 `cp.async.bulk`/TMA 双缓冲。本轮去试了**现成的机制**：cuTile 的
`load_pipelined::<LATENCY>`（`_core.rs:1205`）就是开 TMA 并带延迟提示，**全仓只有
`attention/kernels.rs` 的 SDPA K 循环用了它（3 处），三个 GEMM 核一次都没用**。

做法：给 `nvfp4_gemm::kernels::packed`（生产 W4A4 路）和 `fp8_gemm::kernels::matmul{,_block}`
加一个 `PIPE` 泛型，`PIPE > 0` 时照抄 SDPA 的 `load_pipelined::<4>`，否则保持原 `.load()`；
开关走 `INFER_CUDA_GEMM_PIPE`（默认 0）。**数值上完全中性**：同一 matrix 的 12 次 trial
（short/long/batch4/hot_long 各 3 次）token **逐位相同**（TMA 只换搬运方式）。

**结果：没有可复现收益。**

| A/B | prefill（draft，23 节点） | prefill_last（target，1155 节点） | slot_decode | slot_verify |
|---|---:|---:|---:|---:|
| 第一次 FP4 only：0 → 1 | 9.81 → **8.66（−11.7%）** | 31.22 → 31.24 | 1.62 → 1.58 | 25.49 → 25.67 |
| 第二次 FP4+FP8：0 → 1 | 10.38 → 10.33 | 31.15 → 31.32 | 1.62 → 1.60 | 25.57 → 25.83 |

**第一次那个 −11.7% 没有复现**（第二次是 10.38 vs 10.33），是**同一配置的 run-to-run 波动**，
不是开关效应——这正是 §12.13 记过的坑，只不过这次发生在"图总量"这个看起来更干净的指标上。
第二次的差异是混合的（−1.2% ~ +1.0%），服务侧也混合（short TTFT −1.5%、long +2.4%、
batch4 −0.8%，hot_long 那次 PIPE=0 的 run 被干扰到 TTFT 1.5 s，属异常样本）。

**结论**：SDPA 的 TMA 配方**不能**照搬到这两个 GEMM 上——它们在 K 维的 tile（512/256/128）
和在飞字节本来就比 attention 的 K 块大得多，再加 4 级 TMA 只增加共享内存/同步开销。
已**整段回滚**（三个文件），工作区回到提交状态。杠杆 A 若还有收益，得从"重排 K 循环结构 /
提高 tile 内 K 跨度"入手，而不是贴一个现成的 load 变体。**这个否证的价值在于：把"用现成的
TMA load 就能提带宽"这条捷径关掉了。**

## 十三、KV cache 管理的重构（为什么、目标形态、落地顺序）

### 13.1 现状：固定槽位 + 每序列按最大长度预留 + 前缀整状态拷贝

- **分配**（`resident/fp8_cache.rs::allocate`、`program.rs::allocate_states`）：每个
  `AttentionKv` 状态一次性分配 `capacity × columns × 2`（K/V 各一）的**扁平 fp8 缓冲**，
  `capacity` 是**每序列最大 token 数**，与实际长度无关。
- **布局**：head-major `[kv_heads, capacity, head_dim]`；核里是
  `keys.partition([1,K,D]).load([kv_head, block, 0])`——**位置即下标，没有间接层**。
- **管理**（`resident/slot_batch.rs::SlotPool`）：`slots = width` 个槽位、`free` 栈、
  每槽一套 `States`+`Fp8Caches`、`copy_plan`（KV 搬 `rows` 行 / conv·delta 整块）。
- **准入**（`executor/execution.rs:378-392`）：并发上限 = 槽位数；领不到槽的序列这一步不跑。
- **前缀复用**（`executor/execution.rs:13-55`）：`history.len()` 是 `PREFIX_GRANULARITY = 256`
  的整数倍时**快照整个 program 状态**，复用时 `bind` 把 `rows` 行 KV **逐 head 拷贝**进新槽。

**为什么长上下文差——根因不是"没分页"，是"固定形状的 CUDA 图"**：图在捕获时把张量形状钉死，
于是每序列的 KV 必须按**最大长度**预留才能进图。**block table 正是把"图形状"与"序列长度"
解耦的那个机制**；这也解释了为什么 `max_model_len` 只能取"装得下"的值。

### 13.2 数字（27B fp8，capacity 32768）

| 项 | 值 | 出处 |
|---|---:|---|
| KV 每 token | **32 KB** | 16 个全注意力层 × 4 kv_heads × 256 head_dim × 2(K+V) × 1 B |
| 每序列 KV 预留 | **≈1.07 GB** | 32768 × 32 KB，**与实际长度无关** |
| 递推状态（delta，48 层） | **≈151 MB/序列** | 48 层 × 48 value_heads × 128 × 128 × 4 B（几何见 §十） |
| 复用拷贝量（3584 前缀） | **≈114 MB/次** | 3584 × 32 KB，逐 head 设备内拷贝 |
| 前缀快照粒度 | 256 token，**整状态** | `PREFIX_GRANULARITY`，KV+递推全量 |

即：**KV 预留随 `max_model_len` 线性膨胀，而一个 8k 上下文的真实需求只有 262 MB** —— 32 倍
浪费；这直接把 agentic 场景（长共享前缀 + 高并发）的并发数压在个位数。

### 13.3 目标形态：PagedAttention + 块级前缀共享（vLLM 的两件套）

agentic 负载的特征决定了选项：**长且高度重复的前缀**（system prompt、工具 schema、few-shot）、
**多轮**（前缀只增长）、**高并发**。对应的最优管理是：

1. **块池 + 每序列 block table**：KV 切成固定 token 数的块，全局按需分配；序列只持有自己块 id 的
   表。图形状不再依赖序列长度 ⇒ 不再需要按最大长度预留。
2. **块级内容哈希 + 引用计数 + COW**（Automatic Prefix Caching）：相同前缀的块直接共享，
   **零拷贝**；写共享块时才复制。替代现在"256 token 整状态快照 + 全量拷贝"。
3. **块级逐出/换出**（后面再谈）：长尾前缀分页到主机内存。

### 13.4 本模型的硬约束：递推状态不可分页、不可共享（必须先说清楚）

64 层里 **48 层是 gated-delta（+conv）递推**，它带一个**每序列**状态（≈151 MB），
**不能像 KV 那样分块/共享**——vLLM 的 PagedAttention 假设纯注意力，直接照搬会漏掉这一半。
因此正确形态是：

- **16 个全注意力层的 KV 走块池 + 共享**（这一半能分页、能零拷贝共享）；
- **48 层递推状态保持每序列**，但要在**与共享块相同的边界**上做检查点，这样"共享前缀"要同时恢复
  KV 块和递推检查点；检查点应改成**按需生成 + 复用不拷贝**（现在是每次复用都全量拷 151 MB）。

这个边界也决定了并发的上限：即便 KV 完全分页，≈6 GiB 可用状态内存 ÷ (151 MB + KV) 仍是硬顶。

### 13.5 落地顺序（每步默认路径不变、新路径挂开关、测试全绿）

1. **块池 + block table 间接寻址**：`append`、prefill attention、decode attention 三处核改成
   `(arena, block_table)` 取数；`bind`/预算跟着改；先用**恒等映射**（序列 i 的块 j → 物理
   `i*blocks_per_seq + j`）验证数值与 token 完全一致。这一步不改变内存占用，只是把间接层建起来。
2. **真实块分配器**：空闲块链 + 按需分配 ⇒ **去掉每序列最大长度预留**（这一步才拿到内存/并发收益）。
3. **块级前缀哈希共享 + COW** ⇒ 复用零拷贝。
4. **准入/计价改成块数**：`state_reservation_bytes`、engine 的 `capacity` 语义。
5. **块内布局改 token-major** ⇒ 接上 §12.16 已验证的张量核 SDPA（3.1–3.5×，decode 另做单 query 变体）。

验收：每步都跑官方矩阵（token 必须与改动前一致，除第 5 步外），并把图总量/并发数/显存占用记进本文件。

### 13.6 更正 §12.19/§12.22 的前提：grid 来自**主机侧 partition**，KV 不需要改 token-major

写重构第①步前先把 round-9 那次失败的真正原因查清楚了，结论**推翻我之前的判断**：

**机制（读 cuTile 源码确证）**：
- 每个 `Partition` 有自己的 `grid()`（`tensor.rs:457`，`partition_launch_grid(shape, partition_shape)`，
  **partition 的轴 k → grid 轴 k**）。
- 内核的 launch grid 由**主机侧传入的 partition 参数**决定；**整张传进去的张量（`&keys`）不贡献
  流式轴**——它是 `MappedLaunchPartition` 里的 **OWNED** 轴：`validate()`（`tensor.rs:359-400`）
  只把 `map_shape != OWNED` 的轴计入 `streamed_tiles`，OWNED 轴"不遍历、不贡献 tile 数"。
- 所以内核里对整张张量的 `partition(...).load([运行时标量...])` 是**设备侧视图**，与 grid 无关。

**round-9 失败的原因**：我把 `output` 看成 `[lanes*query_heads, head_dim]` 再按 `[QT, head_dim]` 切
⇒ grid = `(lanes*heads/QT, 1, 1)`，**轴 1 恒为 1**，于是 `pid.1` 恒等于 0，
`kp.load([pid.1 / GROUP, block, 0])` **所有 CTA 都去读 kv_head 0** ⇒ 有限、不崩、完全不对。
（当时我误判成"head-major 布局表达不了"，见 §12.19/§12.22 的结论。）

**正确写法**（`plan.rs` 里 SDPA 的绑法）：把 q/out 看成 **`[lanes, heads*D]`** 再按 `[QT, D]` 切
⇒ grid = `(lanes/QT, heads, 1)`，`pid.1` 就是 head；KV 整张传入、设备侧按
`[1, KB, D]` 切并用 `pid.1 / GROUP` 索引——**和现有 SIMT 核完全一样的索引方式**。

**两个推论**（都改写计划）：
1. **张量核 SDPA 不需要先改 KV 布局**：KV 是整张传入 + 设备侧索引，head-major 完全可用。
   §13.5 的第 ⑤ 步（"块内改 token-major 才能接 SDPA"）**前提不成立**，SDPA 可以更早接进来。
2. **分页第①步可行且无 grid 风险**：把块池整张传入，用 block table 里的**运行时标量**做
   `partition([1, BT, D]).load([block_id, 0, 0])`——与 SIMT 核现在的 `load([head, block, 0])` 同构。
   必要时还能用 cuTile 的显式 `.grid((x,y,z))`（`compile_api.rs:215`，全仓尚未使用）钉死 launch 形状。

顺带更正 §12.22 的措辞：attention 那条路的拦路石从来不是布局，而是**主机侧绑定时 head 轴被压平**。

### 13.7 按 §13.6 的绑法重做张量核 attention：**长上下文快 1.58×**（短块慢 12%）

§13.6 找到真正的原因后（主机侧绑定时 head 轴被压平），把 `decode_tiled` 按正确绑法重新接上：
`q`/`out` 看成 `[lanes, heads*head_dim]`、按 `[QT=32, head_dim]` 切 ⇒ grid = `(lanes/QT, heads)`，
`pid.1` 就是 head；KV 整张传入、设备侧按 `[1, KB=32, D]` 索引 `[pid.1/GROUP, block, 0]`。
开关 `INFER_CUDA_TILED_ATTENTION`，默认关。

**内核这次能跑通**（不再像 round-9 那样"有限但全错"），同一次 matrix、同一台机器，按位置分桶看
`prefill_last` 里 attention 节点的耗时：

| 位置桶 | 记录数 | SIMT（默认） | tiled | 比值 |
|---|---:|---:|---:|---:|
| 0–1023 | 384 | 51.2 µs | 57.3 µs | **1.12×（慢）** |
| 3072+ | 64 | **1587 µs** | **1004 µs** | **0.63×（快 1.58×）** |

其余算子两轮逐项相同（linear 38.9 µs、delta 98.3 µs、conv 10.2 µs、norm 6.1 µs），即**只有 attention 变了**。

**解读**：
- 短块慢 12% 是因为 QT=32 让 grid 只剩 `(lanes/QT)×heads = 2×24 = 48` 个 CTA，而 SIMT 核是
  每 (lane, head) 一个 CTA（64×24 = 1536 个）——短块下并行度输给 SIMT（但绝对量很小，51 vs 57 µs）。
- **长块快 1.58×** 正是长上下文要的那一半：KV 越长，张量核把 QKᵀ/PV 从 SIMT f32 换掉的收益越大，
  而 SIMT 核的成本线性涨（225 µs@448 → 1587 µs@3072+）。
- 下一步（提高短块并行度）很明确：`QT` 降到 16 或把 KV 循环也切到 grid 上（split-K），
  让 grid 从 48 涨到数百个 CTA。

**数值验收仍未完成，而且不能再用 token 比**：§12.11 已经证明两条 lowering 只要差 ~1e-8，
贪心 token 就可能大面积分叉（当时 64 个 token 只对上 1 个）。本轮 token 一致率 long 62/64、38/64、
37/64，short/batch4 0–5/64——**这既不能证明对、也不能证明错**，必须换成**数值比较器**
（比对 logits / attention 输出，容差按 §12.16 闸门的 1e-4 相对量级）。这一步是下一轮第一件事；
在拿到数值结论前，该路径保持**默认关闭**，不进入交付配置。

### 13.8 张量核 attention **通过数值验收**：token 分叉是贪心放大，不是 bug

§13.7 留下的唯一问题是"数值对不对"。做法是把它接到**仓库已有的数值测试**上，而不是新增测试：
`tests/unit/resident_attention_prefill.rs` 的 `check_case()` 本来就是「跑 resident 核 → 和
`reference()` 主机参考逐元素比」的夹具，我把 `decode_tiled` 加进同一个 case（同一份 q/KV/metadata、
同一个参考），容差用仓里给同款补偿 bf16 mma 定的相对量级（attention 闸门用的是 `1e-4 × scale`）：

```
test resident::attention_prefill::tests::chunk_attention_matches_reference_with_tails_windows_and_draft_offset ... ok
```

覆盖的是 tails / window / draft-offset / 非活跃行这些真正会出错的边界；SIMT 那条仍按 2e-5 卡死。
**结论：绑定修对之后，张量核那条在数值上是成立的**，1e-4 相对量级内与 f32 SIMT 路一致。

由此也能解释 §13.7 里 matrix 上 token 一致率很差（short/batch4 0–5/64）：**那是贪心放大的必然结果**，
不是内核错了——§12.11 早就量过，两条 lowering 只要差 ~1e-8，64 个 token 只对上 1 个。
所以：
- 该路径的**正确性**已经用数值比较器确认（相对量级 1e-4）；
- 它的**输出口径**与 f32 SIMT 路不同（同 chunked 递推的处境），因此**保持默认关闭**，
  要和 `--chunked-recurrent` 一样由使用方显式选择；
- 它在长上下文快 1.58×（§13.7），短块慢 12%（grid 只有 48 个 CTA）——推广前应先修短块
  （QT 降到 16 或对 KV 循环做 split-K，把 grid 抬到数百）。

### 13.9 张量核 attention 的 QT 扫描：**QT=16 在两个区间都赢，长上下文 6.2×**

§13.7 记下"短块慢 12%"并猜是 grid 太小（48 个 CTA 对 SIMT 的 1536 个）。扫了一下 QT：

| 位置桶 | SIMT（默认） | QT=32 | **QT=16** |
|---|---:|---:|---:|
| 0–1023 | 51.2 µs | 57.3 µs（1.12× 慢） | **36.9 µs（1.39× 快）** |
| 3072+ | 1587.2 µs | 1004.2 µs（1.58× 快） | **256.0 µs（6.2× 快）** |

QT=16 的**幅度**说明真正的原因不是"grid 不够大"那么简单：QT=32 时每 CTA 的累加器是
`32 × 256 × 4 B = 32 KB` f32，远超寄存器预算（SM 共 256 KB），必然溢出/占不满 occupancy；
QT=16 减半到 16 KB，同时 CTA 数从 48 涨到 96。两个因素叠起来，长块就被拉开到 6×。
所以 `TILED_QUERY_TILE` 从 32 改成 **16**。

数值上两个宽度都过：`check_case()` 里现在对 `qt ∈ {16, 32}` 各跑一遍，都用同一个主机参考、
同一个 1e-4 相对容差，`chunk_attention_matches_reference_with_tails_windows_and_draft_offset` 通过。

**结论**：长上下文那半边现在有 **6.2×**（attention 节点 1587 → 256 µs），短块也从"慢 12%"变成"快 1.39×"——
即张量核 attention 在**全区间**都不输 SIMT。它仍是**默认关闭**的可选项（输出末位不同，口径同
`--chunked-recurrent`），但现在已经没有"短块回退"这个理由挡着它了。

### 13.11 第①步**已撤回**：官方 matrix 抓到了小夹具抓不到的 bug

§13.10 记的块表改动（提交 `b3ddcd8`）在**单测/GPU 测试全绿**的情况下，被**欠着的那次官方 matrix 对照**
当场打回：

```
crates/backend/cuda/src/resident/attention_prefill.rs:56: tile block: [0,0,0] …
partition access out of bounds: dim 0, block index >= ceil(?/1) or index < 0
```

即 `append` 里 `table[position / BT]` 越界——**表长与内核实际用到的位置对不上**：真机 profile 的
capacity（32768 → 1024 块）下必然触发，而单测/GPU 测试的夹具 capacity 只有 128 左右，位置永远落在
少数几块里，**这个不一致在小夹具里根本不可能暴露**。表长是"由 capacity 派生"的结构，夹具的 capacity
与交付配置差两个数量级，所以测试绿不等于这一步是对的。

处置：`git revert b3ddcd8`（`b94bf1a`），回到未引入块表的状态；随后复验 fmt/clippy/27 单测/4 GPU 测试
全绿，官方 matrix 重新正常出 4 个 case 的完整结果。

**两条记录下来的教训**：
1. **一切"按 capacity 派生"的结构（表长、块数、预算）必须用交付级 capacity 验证**，小夹具只能验证
   语义不能验证尺寸。这类改动以后第一件事就是跑 matrix，再谈细节。
2. 撤回本身说明"每步都跑官方 matrix"这条纪律是有效的——它挡下了一个测试全绿但真机会越界的改动。

第①步的正确做法（下一轮）：先把**表长与它服务的 KV 缓冲容量**在同一个地方派生（同一个 `capacity`
来源），再让 append/attention 都用它；并且先在**私有程序路径**用交付 capacity 复现一次
（`--max-model-len 32768` 的最小请求即可触发），确认步长一致后再接读路径。

### 13.12 撤回的真正原因查到了：**那次调用的位置本来就超出了 arena**（疑似既有隐患）

上一轮只写了"表长与内核寻址的 capacity 对不上"。本轮把表装回去、在 `record_attention` 两个入口
打印 `(capacity, KV 张量形状, 表长)`，用真机 profile 复现，拿到失败现场前三行的数据：

```
KV_TABLE capacity=128 keys_shape=[131072]      values_shape=[131072]      table_shape=[4]
KV_TABLE capacity=128 keys_shape=[4, 128, 256] values_shape=[4, 128, 256] table_shape=[4]
attention_prefill.rs:56: partition access out of bounds: dim 0, block index >= ceil(?/1) or index < 0
```

**表长是自洽的**：capacity 128 → 4 块 ✓（池化那条是 capacity 2048 → 64 块 ✓）。真正越界的是
**表的下标** `position / BT`——该调用寻址的位置 **≥ 128**，而它服务的 KV 缓冲只有 128 行。

**这意味着什么**：改动前那个内核在同一调用里**直接**用同一个 `position` 写 KV（`[1, CAP, D]`
分区的 `[0, position, 0]`）。也就是说，**这个位置在旧代码里同样越过了 arena**——只是
`partition_mut(..).store(..)` 没有像 `load` 那样把越界报出来，于是**静默写到了缓冲之外**。
换句话说：我加的这次 `load` 把一个**既有的、静默的越界**变成了显式报错。

这解释了为什么单测/GPU 测试永远看不到它：它们的夹具里没有"喂给容量 128 的程序一个 ≥128 的位置"
这种组合。

**下一步（必须先做，否则第①步做不成）**：在**主机侧**把 `metadata::update` 写入的
`(base, count, offset)` 与该 program 的 capacity 对一次账，凡是 `base + count + offset > capacity`
就直接报错——这既是分页的前置（block table 绝不能越界索引），也能判定"这是既有 bug 还是我引入的"。
在没有这个护栏之前，不再往核里加表。

### 13.13 给 prompt 路补上与 slot 路同款的位置护栏：**交付级 workload 上零违规**

§13.12 说下一步是"主机侧把 (base, count, offset) 与 capacity 对账"。做完之后的结论**否定了一条假设**：

- `SlotDecodeGraph::stage_lanes` 早就有 `state_pos >= capacity → 报错`（`batch.rs:906`），
  但 **prompt/batch 路没有这一条**。补上两处：
  - 每 lane：`state_pos < -1 || state_pos >= capacity` 报错（`-1` 是"非活跃"哨兵，其余负数才会
    让内核去索引到负行）；
  - 整块：`base + tokens.len() - 1 + state_offset >= capacity` 报错。
- **官方 matrix（`--chunked-recurrent`，4 个 case × 4 次）16 次 trial 全部完成、护栏零命中**。

所以：**交付级 workload 上，主机侧写入 metadata 的位置从来没有越界**——第 18 轮块表那次
"table[position/BT] 越界" **不是**"主机给了越界位置"造成的。这条假设被排除，剩下的是我自己的接线问题
（表长/表实例与内核寻址不匹配，或内核侧算出的下标有问题）。下一次必须**在内核侧**把
`position/BT` 与实际表长打出来，或干脆把表长从**捕获时那张 KV 张量的形状**派生（而不是从另一个
capacity 参数），让"表长与 arena 深度"只可能有一个来源。

顺带记一个自己踩的坑：护栏第一版我把下界写成 `position < 1`，于是 **base=0 的正常请求**被拒
（日志：`prompt state rows end at 50 (base 0, 51 tokens, offset 0), past the 128-token KV arena`）。
**护栏自己的语义也要用正常路径验一遍**——又是 matrix 把它抓出来的。

### 13.14 块表越界再收窄一步：**喂给那次 append 的 metadata 不是护栏校验的那张**

把块表重新接上，并加了一道**捕获期一致性检查**（`Capture::check_table`：表宽必须等于
`capacity / KV_BLOCK_TOKENS`，否则直接报错）。结果：

- 一致性检查 **0 命中**（表宽与 arena 深度处处相符）；
- 但 `attention_prefill.rs:56`（= 表加载 `blocks.load([position / BT])`）**仍然越界**，而且发生在
  **request=1**（最短的 52-token 提示）；
- 我的 prompt 护栏（校验 `prefill_info` 的 `base + count - 1 + offset < capacity`）**没有命中**。

把这三条放在一起只能推出一个结论：**那次 append 绑定的 metadata 不是 `prefill_info`**。
`record_prefill_attention` 有两类绑定：Batched 走 `prefill_info`（`[base, count, offset, 0]`），
Row 走**每 lane 的 metadata**（`[rope_pos, token, state_pos, 0]`）——后者的字段语义完全不同
（field[1] 是 token、field[2] 是 state 行号），而 append 核正是按 `[base, count, offset]` 去读的。
我的护栏校验的是 `prefill_info`，所以对 Row 绑定这一路**恰好不生效**。

**下一步（很具体）**：在 `Capture::record_prefill_attention` 里打印这次调用绑定的到底是哪张
metadata（`self.mode` + `self.metadata` 与 `prefill_info` 的指针/shape），并在 Row 绑定下把
field[0..3] 的值 dump 出来。先确认"Row 预填充也会记录 append"这个假设——如果是，那 Row 绑定
本身就带着一个既有的字段错位问题（与块表无关），也就不奇怪为什么只有加了表之后才炸。

### 13.15 `BatchGraph::run` 少写了一个 sidecar：**真实的不对称，但在交付 workload 上行为中性**

顺着 §13.14 的线索把"哪张 metadata"查清楚了，结果是一处**真实的代码不对称**：

- prompt 图（`capture()` 里 `mode: CaptureMode::Prefill`）绑定的是 **`prefill_info`**，
  而 append 与递推核都按 `[base, count, offset]` 读它；
- **`run32`（宽 prompt，width ≥ PREFILL_LANES）会写它**（而且位置经过 §13.13 的护栏）；
- **`run`（窄 prompt，width < PREFILL_LANES —— 正是池化 + `--chunked-recurrent` 走的那条）
  只写每 lane 的 metadata，从不写 `prefill_info`** ⇒ 该图重放时读到的是**上一次写入的旧 base**。

已修（`BatchGraph::run` 在 `prefill.is_some()` 时按 `run32` 的方式更新 `prefill_info`，并同样过护栏）。

**但必须如实说**：修完之后，**官方 matrix 12 次 trial 的 token 与修之前逐位相同**（64/64），
单测/GPU 测试也全绿。也就是说在交付 workload 上这处旧值**没有影响到输出**——所以它**不是**
第 18 轮块表越界的原因。（很可能是窄 prompt 图最终走的是 Row 分支、绑每 lane metadata，
`prefill_info` 对它并不生效；那也解释了为什么改了没有行为差异。）

保留这个改动的理由只有一条，而且是诚实的：**"其中一个入口写、另一个入口不写"本身就是隐患**，
下一次谁改动 prompt 图的绑定就会踩上它；而它已经验证过不改变任何输出。

另外，本轮也**排除了一整类解释**：既然"KV 位置错"的那条路径改动前后输出逐位相同，
说明这条链上的写位置差异**不会进入最终结果**——块表越界的原因还得在别处找（下次要在核里
把 `BT`、表长、`position` 三个数**写进一个调试缓冲读回来**，不再靠推）。

### 13.16 第①步**落地**：块表（恒等映射）通过官方 matrix，根因是**泛型顺序错位**

五轮追下来的根因，居然是**我自己在生产调用点把泛型顺序写反了**：

- 内核声明：`append<E, D, CAP, QUANT, BT>` —— **QUANT 在前，BT 在后**；
- 生产调用点传的是：`[dtype, head_dim, capacity, KV_BLOCK_TOKENS, quant]` —— **后两个换了位**。

于是内核读到 **`BT = 1`**（那个 quant 标志），`position / BT` 变成了 `position` 而不是
`position / 32` ⇒ 表下标一路冲到 `position`（最大 2047）而表只有 `capacity/32`（池化 64）个元素
⇒ **必然越界**，而且发生在最短的 request=1 上（位置 4 就超过 4 元表）——与五轮来观察到的现象完全一致。

**为什么测试一直是绿的**：单测里泛型是**显式按正确顺序**写的
（`["f32", DIM, CAPACITY, "0", KV_BLOCK_TOKENS]`），所以它验证的是**内核本身**，而错的是**生产调用点**。
cuTile 的泛型是一串**位置参数**（这里是 5 个 `String`），编译器**无从检查**，必须靠"调用点与内核声明一致"
这条纪律——而这正是最容易被漏掉的地方。

**修复后（提交见下）的验证**：官方 matrix（`--chunked-recurrent`，4 case × 4 trial）
**越界 0 次**，且 **12 次 trial 的 token 与加块表之前逐位相同**——这正是恒等映射应有的结果，
也说明这次改动**没有改变任何行为**，纯结构。

**五轮排查的教训**（值得单独记）：
1. 位置参数型接口（尤其多枚同类型参数）出错时**现象会指向完全错误的方向**——我先后把原因归到
   "表长算错""调用本身越界""主机位置越界""陈旧 sidecar"，全是错的，真相是一个参数顺序。
2. 每次"缩小范围"都该问一句：**有没有一条最便宜的检查能直接证伪当前假设？** 这次只要在核里
   或调用点把 `BT` 打出来就够了，我却先做了四轮推理。
3. 夹具与生产调用点**各写一份泛型/参数**时，夹具的正确不能代表生产正确。

### 13.17 读路径也全部走表：整个 KV 访问链（写 + 四个读）现在都过 block table

第①步的读半边补齐。现在**每一处 KV 访问都先用 `table[逻辑块]` 换成物理块**：

| 访问点 | 文件 / 内核 | 说明 |
|---|---|---|
| 预填充写 | `attention_prefill::append` | §13.16 已通 |
| 预填充读（SIMT） | `attention_prefill::decode` | `kp/vp.load([head, table[b], 0])` |
| 预填充读（张量核，开关后） | `attention_prefill::decode_tiled` | 同上（表分区改名 `table_blocks`，避免与循环上界 `blocks` 撞名） |
| decode 读（SIMT） | `resident/attention::decode` | `load([pid.0/GROUP, table[b], 0])` |
| decode 读（split-KV） | `attention_decode::partial` | 表经 `attention_decode::Workspace::record` 透传 |

**这次先做了泛型顺序审计**（§13.16 的教训）：六个内核的声明顺序与调用点逐一对齐后才构建——
`append<E,D,CAP,QUANT,BT>`、`decode<E,D,GROUP,HEADS,BT>`、`decode_tiled<E,D,GROUP,QT,KB,BT>`、
`attention::decode<E,D,GROUP,BT>`、`partial<E,D,GROUP,PARTS,BT>`，全部把 `BT` 放在**末位**，
调用点也都在末位。

**实测**：官方 matrix（`--chunked-recurrent`，4 case × 4 trial）**越界 0 次**，**12/12 trial token 逐位
相同**（与加块表之前对比）。恒等映射下必须如此，所以这一步同样是**纯结构、零行为变化**。

至此**间接层完整**：块池（现在还是"每序列一块恒等映射"）→ 每序列表 → 五处核 → 全部按物理块取数。
下一步第②步就能把表的内容换成**真分配器的输出**，而核、绑定、预算都不用再动。

### 13.18 并发被一个**常量**卡住，不是被显存卡住——这正是第②步要解的

查池化构建时发现：`create_slot_pool`（`execution.rs:258-264`）把池宽 clamp 到
**[2, `CB_DECODE_SLOTS`]**，而 `CB_DECODE_SLOTS = 4`（`constants.rs:157`）——也就是说
**并发上限是策略常量，不是显存**；它只是在这个宽度装不下时才逐级降到 2。

实测（8 个**相同**的 52-token 并发请求，同机同配置）：

| 配置 | wall | TTFT min / med / **max** |
|---|---:|---|
| 现默认（池宽 ≤4） | 6.53 s | 2282 / 2446 / **5008 ms** |
| `CB_DECODE_SLOTS = 16` | **5.16 s（−21%）** | 1835 / 1998 / **3836 ms** |

现默认那行的 **max ≈ 2× med** 就是"**分两批、每批 4 个**"的签名 ✓；提到 16 之后 N=8 明显变好，
但 N=16（wall 8.77 s、med TTFT 4367 ms）**还是两批**——因为池在显存不够时会自动降到一个较小的宽度
（设计如此），说明真正的天花板是**每槽的状态占用**。

**但这不能就这么落地**：16 槽配置下官方 matrix 的第一个请求被**准入拒绝**：

```
admission StateBytes: required=1118322192, available=958402560  code=Capacity
```

即**池按槽预留 + 请求再按序列全上下文（32768 → 1.07 GB）预留**，两者抢同一份预算。
所以已回退到 4（matrix 恢复 16/16 trial ✓）。

**这条给重构提供了量化的靶子**：单靠"把池开宽"就能拿到 N=8 wall −21%；再往上就必须要
**一个共享块池 + 按块计价**（第②、④步）——把"每槽预留 + 每序列全上下文预留"这份**双重预留**
去掉，并发和上下文长度才能各自增长。换句话说：**当前设计里"并发"和"上下文长度"是在互相挤兑的**。

### 13.19 补课：读路径改动漏掉了**直接调核的测试**

第 24 轮把表接进 `attention_decode::partial` 与 `attention::decode` 之后，我跑了
`cargo test`（当时显示了 "27 passed"），但那一次**没有真正重编测试目标**；等第 25 轮再跑时
测试目标直接编译失败——`tests/unit/resident_attention_decode.rs` 与
`tests/unit/resident_attention_prefill.rs` 里有**直接调用这些核**的地方（`partial`、`decode`、
`decode_tiled`），所以核签名加参数后它们必须一起改。

已修：三个测试调用点各补一张恒等块表与 `BT` 泛型；现在
`cargo check --all-targets` ✅、27 单测 ✅、4 个 GPU 硬件测试 ✅、两个 `attention_decode` 测试
（含 `split_kv_matches_independent_causal_window_reference` 这个数值参考测试）✅。

**教训（与 §13.16 同源）**：改内核签名时，除了生产调用点，还要找**所有直接调核的测试与示例**；
并且"测试通过"必须确认那次运行**真的重编了测试目标**（`cargo test` 如果被缓存/SKIP 过，
输出的 `test result` 可能来自旧的二进制）。这次是靠 `--all-targets` 才发现的。

### 13.20 长上下文根本进不了池：`CB_SLOT_TOKENS = 2048` 是个门槛常量

读准入路径时发现并实测的一条结构性事实：

- 池化序列必须满足 `state.capacity <= CB_SLOT_TOKENS`（`execution.rs:188`），而
  `CB_SLOT_TOKENS = 2048`（`constants.rs:162`，注释还写着它当初是为了"32 GB 卡上第四个请求别被饿死"
  才调下来的）；
- 因此**交付 profile（`max_model_len 32768`）下超过 2048 token 的请求根本不可能进池**，一律走**私有
  program**，并按其**全上下文**预留 KV（32768 → 1.07 GB/序列）⇒ 长上下文的并发完全由显存决定。

实测（4 个并发的 **2560-token** 提示，`--max-model-len 32768`）：

| `CB_SLOT_TOKENS` | wall | TTFT min / med / max | `gpu_budget` 递延 |
|---|---:|---|---:|
| **2048（原默认）** | 10.59 s | 3228 / 7491 / 8679 ms | **167** |
| **8192** | **9.21 s（−13%）** | 3166 / **5809（−22%）** / 7998 | **113（−32%）** |

`gpu_budget`（= 显存预算）正是这类请求被推迟的原因；把池的槽容量提到 8192 之后，2560-token 的会话
能进池（拿到前缀复用/暖槽），显存压力与延迟都下来了。

**官方 matrix 在 8192 下照常 16/16 trial 通过**（token 与 2048 配置逐位对比见下），所以这个常量
**可以落地**。安全性来自池构造器本身：宽度装不下时会**逐级降宽**（`create_slot_pool` 从上限往 2 试），
所以调大槽容量最坏只是池变窄，不会失败。

**但它是过渡手段**：真正的解法是**共享块池**——序列只持有它实际用到的块，`CB_SLOT_TOKENS`
这种"每槽按容量预留"的门槛就该消失（第②步）。这条实测同时给出了第②步的验收方式：
同样 4×2560 并发下 `gpu_budget` 递延应继续下降，且**不再需要靠调大常量**。

### 13.21 第②步第一块（共享 arena）：设计定稿，写到最后两处未落地，已回退

**设计**（行为等价，先把结构立起来）：池的每个 KV 状态从"每槽一份 arena"变成**一份共享 arena**
`[kv_heads, width × capacity, head_dim]`，每槽的表从"恒等"改成**带块偏移**：
槽 `i` 的逻辑块 `b` → arena 块 `i × blocks_per_slot + b`。因为偏移恰好等于原来每槽 arena 的起点，
**寻址与今天逐字节相同**，所以这一步理应是"纯结构、token 不变"。

**触点（这次全部枚举清了）**：
1. `fp8_cache::allocate_shared(device, graph, capacity, slots, scales)`（arena 行数 = `capacity × slots`），
   原 `allocate` 保留给私有 program；
2. `SlotPool.fp8: Vec<Fp8Caches>` → **`Fp8Caches`（一份）**；
3. 新增 `slot_tables(device, capacity, width)` 生成带偏移的表（把 `new` 里那段挪出来，顺带解决
   clippy 的 122 行超限）；
4. `build_slots` / `build_slot_verify` / `capture_slot_verify` 的 `lane_fp8: &mut [Fp8Caches]`
   → 一份 `&mut Fp8Caches`（每个 lane 复用同一份）；
5. `copy_kv_fp8` / `copy_kv_tensor` 需要 `KvSpan { capacity, offset, on_dst }`——**卡点就在这里**：
   池化导入（bind）是"共享侧在**目的**"，而 **draft 导出（`slot_batch/draft.rs`）是"共享侧在源"**，
   角色相反，所以不能只给"目的侧"加偏移；
6. `state_bytes`：共享 arena 只计一次。

**为什么回退**：我在两处调用点（bind、draft 导出）用锚点替换 `KvSpan` 时**没匹配上**（缩进不同），
`make local-build` 报错，而我这一轮的预算已经用尽。按目标约束"不得让工作区处于编译不过的状态"，
整块回退（`git checkout` 五个文件）。回退后复核：`cargo check --all-targets` **0 错误**、clippy
`-D warnings` ✅、release 构建 ✅、27 单测 ✅、4 个 GPU 硬件测试 ✅。

**下一轮直接落地**（顺序已定）：先把 `KvSpan` 的两处调用点改完再编译，然后再做**arena 尺寸由预算决定**
（`total_blocks < width × capacity`）+ **分配不到块就递延**（复用 `DeferReason::StateCapacity` 既有机制）
——那一步才是收益兑现点（§13.20 已给出验收：4×2560 并发的 `gpu_budget` 递延应继续下降）。

### 13.22 共享 arena 第二次尝试：编译通过、但 **draft 路径设备级出错**，已回退

这次按 §13.21 的清单一次改完（`fp8_cache::allocate_shared`、`SlotPool.fp8` 变一份、`slot_tables`
带偏移、三个 builder 共用一份、`KvSpan`、`state_bytes` 计一次），而且 **`cargo check` +
`clippy -D warnings` 一次通过**（上一轮那两处失配这次用精确锚点 + "先断言再写入"解决了）。

但验证不通过，失败点很集中：

| 验证项 | 结果 |
|---|---|
| `cargo check --all-targets` / clippy | ✅ 0 错误 |
| 27 单测 / `attention_decode` 数值测试 | ✅ |
| `resident::slot_verify`（池化 verify） | ✅ 2/2 |
| **`resident::slot_batch` 的两个 draft 测试** | ❌ 2/2：`DriverError(719, "unspecified launch failure")`（设备级越界） |
| 官方 matrix token | ❌ short/batch4 分叉（1–36/64），long/hot_long 一致 |

**根因（下次直接修）**：`draft.rs` 里那次 `copy_kv_fp8` 的**第一个参数是目的地**
（`target.fp8_states_mut()`，目标序列的私有 arena），**第二个才是池的 arena**——所以原来的
`slot_capacity` 描述的是「**目的地**的每头行数」，不是池 arena 的。我按"偏移侧=源"去改
（`on_dst: false`），于是**源侧 stride 用了池 arena 的行数、目的侧用了自己的**，draft 导出越界。

**正确的做法**：不要再编码"哪一侧是共享侧"这种隐式约定，让 `KvSpan` **显式描述两侧**
（`dst: {capacity, offset}`、`src: {capacity, offset}`）——bind 时 `dst=pool, src=private`，
draft 导出时 `dst=private, src=pool`，两侧都自带 (capacity, offset)，歧义就消失了。

回退后复核：`--all-targets` **0 错误** ✅、clippy ✅、27 单测 ✅、4 个 GPU 硬件测试 ✅、fmt ✅、
release 构建 ✅、工作区干净。

### 13.23 共享 arena 第三次尝试：`KvSpan` 改成**双侧行偏移**后编译通过，但 draft 路径仍有设备越界

按 §13.22 的结论改：`KvSpan { dst: usize, src: usize }` **只带两侧的行偏移**，容量由张量自身
的 extent 决定（校验也据此），于是"哪一侧是共享侧"的歧义彻底消失。`cargo check` /
`clippy -D warnings` / `--all-targets` 全部一次通过，非 draft 的测试与 `slot_verify` 也过。

但 draft 仍然越界，而且这次**定位到了具体内核**（`CUDA_LAUNCH_BLOCKING=1`）：

```
crates/backend/cuda/src/resident/attention.rs:113: tile block: [0, 0, 0], position: []:
  partition access out of bounds: dim 1, block index >= ceil(?/32) or index < 0
```

`attention.rs:113` 正是 **decode 注意力的 KV 读**（`kp.load([pid.0 / GROUP, physical, 0])`），
"dim 1" 就是 arena 的块维：**表给出的物理块号超过了该图所用 arena 的块数**。

检查过的、应该一致的地方：池的 arena 用 `allocate_shared(capacity, width)`（= `capacity*width` 行
= `capacity*width/32` 块），表用 `slot_tables(capacity, width)`（条目最大 `width*blocks - 1`）——两者
自洽；私有 program 的表是 `identity_table(capacity)`，其 arena 也是同一 capacity——也自洽。

所以**还有第三处**"表与其 arena 不同源"的地方（draft 路径专属），下一轮先用最便宜的检查钉死它：
在**每次捕获**时打印 `(figure 的 arena 块数, 该表的 shape 与最大条目)`——这两数一比就知道是谁错配，
不用再推理。（当前已回退；回退后 clippy ✅、27 单测 ✅、4 GPU 硬件测试 ✅。）

### 13.24 三轮失败的共同根因：`capacity` 同时承担了**两个角色**

回头看 §13.22/§13.23 的失败，并对照代码里 `self.capacity` 的每一处用法，根因其实一句话：

> **共享 arena 之后，"每槽的寻址窗口"和"arena 的行数"不再是同一个数，但代码里它们共用一个
> `capacity`。**

具体地，`Capture::capacity`（来自 `BatchBuilder.capacity`）被同时用于：

1. **KV 张量的形状/分区**：`keys.reshape(&[kv_heads, self.capacity, head_dim])`、
   `(&mut keys).partition([1, self.capacity, head_dim])`、以及 append 核的 `CAP` 泛型
   —— 这些要的是 **arena 的行数**（共享后 = `capacity × width`）；
2. **每槽的语义窗口**：槽的游标、逻辑块数、`SlotCopy` 的判据 —— 这些要的是**每槽**的 capacity。

共享 arena 后我把表和 arena 都换成了"共享空间"，却让 (1) 仍拿每槽 capacity ⇒
**内核看到的分区范围比表给出的物理块号小** ⇒ `attention.rs:113` 那个
"dim 1, block index >= ceil(?/32)" 正是这么来的（而 reshapes 在部分路径上也因此不自洽）。

**结论：第②步的正确前置是先把这两个角色拆开**（例如 `Capture` 增加 `arena_capacity`，
只用于 (1)；`capacity` 保留 (2) 的语义），而不是先动 arena。这也解释了为什么三轮都是
"编译通过、验证挂"——我一直在改数据的**归属**，却没改**范围**。

**已落地的、不受此影响的东西保持不变**：第①步块表（恒等映射、行为逐位一致）、
`CB_SLOT_TOKENS = 8192`（−13% wall / −22% 中位 TTFT）、`--chunked-recurrent`、张量核 attention。

### 13.25 `CB_SLOT_TOKENS` 的实际上限就是 **8192**：再往上会打穿准入

顺着 §13.20 把池槽容量再推到 16384 试了一次：

| 项 | 8192（保留） | 16384（已回退） |
|---|---|---|
| **官方 matrix** | **16/16 trial 通过** | ❌ **0 trial**：出现准入拒绝 |
| 4×2560 并发 | wall 9.21 s，`gpu_budget` 113 | wall 9.36 s，`gpu_budget` 102 |
| 2×8000 并发 | — | wall 15.2 s，TTFT 7807 / **13904** ms（两批） |

原因与第 25 轮那次 16 槽一模一样的**双重预留**：池按"槽 × 容量"预留（4×16384×32KB ≈ 2 GB），
而每个超过池容量的请求还要按**全上下文**再预留 1.12 GB（32768 时），两者抢同一份预算 ⇒
matrix 的第一个请求就被拒。

**所以 8192 是当前计价方式下的实际上限**（这也是为什么长上下文要真正受益，必须先做第④步
"按块计价"——把双重预留去掉，而不是继续调大这个常量）。16384 已回退，8192 配置复验
官方 matrix **16/16** 通过。

### 13.26 拆开 `capacity` 的两个角色之后：编译通过、draft 仍失败

按 §13.24 做了拆分：`Capture`/`BatchBuilder` 增加 `arena_capacity`（只用于 KV 张量的
reshape/分区/append 的 `CAP` 泛型），`capacity` 保留每槽语义；池传 `capacity * width`，
私有 program 传 `capacity`。**编译、clippy、27 单测全过**，但 draft 两个测试**依然失败**：

```

```

⇒ 说明"表与 arena 不同源"的第三处**不在 capacity 的角色上**。下一轮改用 §13.23 里定下的
最便宜的检查：在**每次捕获**时打印该图的 `(arena 行数/块数, 表的 shape 与最大条目)`，
两数一比即可定位（不再靠推理）。已回退：clippy ✅、27 单测 ✅。

### 13.27 第一次**测量**（而不是推理）：表的**形状**是对的，问题在表的**取值**或归属

按 §13.23 的办法，在共享 arena + `arena_capacity` 拆分的基础上，于**每次捕获**打印：

```
KV capture: arena_rows=128 table_len=2 table_reach_rows=64 per_slot=64
KV capture: arena_rows=256 table_len=2 table_reach_rows=64 per_slot=64
```

读法：`per_slot=64`（每槽 2 个 32 行的块）、`arena_rows = capacity × width`（128 = 2×64、
256 = 4×64）**都自洽**；`table_len=2` 是**每槽**的表长（也自洽）。

**但这次打印量错了量**：我打的是表的**长度**，而越界报错说的是**表的取值**（物理块号）——
`table_reach_rows` 用 `len × BT` 算是 64，其实表里的值才是关键（槽 1 的表是 `[2,3]`，
而 arena 有 4 个块，也仍在范围内）。所以这次测量**排除了"表长度/arena 行数不匹配"**，
把嫌疑收窄到**表的取值**或**某个图拿到了别人的表**。

**下一轮的正确打印**（一行）：把表的**最大值**打出来——`table` 是设备张量，需要一次
`to_host` 回读（诊断期可以接受），打印 `(arena_blocks, max(table), table_len, owner)`。
`max(table) × BT + rows ≤ arena_rows` 一成立/不成立，就知道是谁错配。

（本轮改动已回退：`--all-targets` 0 错误、clippy ✅、4 个 GPU 硬件测试 ✅。）

### 13.28 把表的**取值**也量了：形状和取值**都自洽** ⇒ "表/arena 错配"这一族假设被排除

这次把表的取值直接打出来（表是我们自己构造的，不需要设备回读）：

```
KV slot table: slot=0 len=2 base=0 max=1
KV slot table: slot=1 len=2 base=2 max=3      <- 2 槽池：arena 128 行 = 4 块，max=3 在范围内
KV slot table: slot=0..3 len=2 base=0/2/4/6 max=1/3/5/7   <- 4 槽池：arena 256 行 = 8 块，max=7 在范围内
KV capture: arena_rows=128 table_len=2 per_slot=64
KV capture: arena_rows=256 table_len=2 per_slot=64
```

对照关系全部成立：`max(table) × 32 + 32 ≤ arena_rows` ✓（3×32+32=128 ✓、7×32+32=256 ✓）。
**所以故障不是"表的形状"也不是"表的取值"与 arena 不匹配**——这一族假设（我在 §13.23–13.27 反复推的）
被实测排除。

**还剩下的嫌疑**（下一轮直接量）：
1. **`identity_table`**（私有/draft program 的表）——这次**没有**插桩，而它是 draft 路径独有的；
2. **运行期**传进去的 `position`/窗口（`attention.rs:113` 的物理块号是由 `position/32` 索引表得到的）——
   若某个 lane 拿到了**别的槽的位置**，`blocks.load([k])` 就会越过**表的长度**（dim 0），
   而报错说的是 dim 1（arena），所以更可能是"表里的值被当成行号"以外的第三种情形；
3. 因此下一步改为在**回放时**打印 `(lane, position, block 索引, 表项)`（host 侧，一次一行），
   直接看**那一次**读取的输入。

（改动已回退：`--all-targets` 0 错误、clippy ✅、4 个 GPU 硬件测试 ✅。补丁仍在 `/tmp/shared-arena.patch`。）

### 13.29 第④步第一刀：池内序列不再**重复预留**——正确、已验证，但对交付 profile **不生效**

**改动**（`provider.rs::state_reservation_bytes_for`）：若请求能被池的某个槽容纳
（`capacity <= pool.capacity()`）且池还有空槽，则它只被收取**固定 headroom**，而不是按自己的
capacity 再预留一遍状态——因为**池在构建时已经为 `width` 个槽付过账**（这也正是 §13.18/13.25 里
"池变宽就把长上下文请求挤掉"的机制）。判断条件与 `execution.rs:161` 现有的"能否进池"规则完全一致。
新增 `SlotPool::has_free_slot()`。

**验证**：官方 matrix **16/16 trial 通过**（与改动前同样通过）、27 单测 ✅、4 个 GPU 硬件测试 ✅。
也就是说这条改动**安全但不改变交付路径的行为**。

**为什么对交付 profile 不生效**（这轮最有价值的发现）：matrix 里每个请求的 `plan.reserved_tokens`
是 **32768**（模型的最大上下文），而请求被收取的正是 `32768 × 32KB ≈ 1.07 GB`。既然
`32768 > pool.capacity()`（我们能负担的槽容量最大 8192，推到 16384 会让 *池构建* 本身失败），
**这条减免永远不会命中交付 profile 的请求** ⇒ 16k 配置在减免之下**依然 0 trial**（实测）。

**结论（指向第④步的正确做法）**：真正的瓶颈不是"池不够宽"，而是**引擎按模型最大上下文给每个请求
计划/计价**（`plan.reserved_tokens = max_model_len`）。只要这点不变，任何能负担的池都无法覆盖它，
减免也永远不会命中。⇒ 第④步的正确一刀是**让计划/预留反映请求实际需要的长度**（prompt +
max_new_tokens），而不是模型上限；那时"按块计价"才真正把并发与上下文长度解耦。

### 13.30 更正 §13.25 与 §13.29：16384 的失败是**准备超时**，不是准入；报价的大头是**激活 arena**，不是 KV

拿日志把两件事对上了，两处旧结论都要改：

**（1）`CB_SLOT_TOKENS = 16384` 的失败原因不是"准入拒绝"**（§13.25 写错了）。`pricing16k` 的日志里
**没有任何** `admission ... StateBytes` 行，而是：

```
INFO  CUDA slot pool captured at load width=4 capacity=16384
INFO  request accepted request=1 workload=Generate { max_new_tokens: 64 }
WARN  request finished unsuccessfully request=1
      reason=Failed("resource preparation acknowledgement timed out") e2e_us=30000101
```

即：池在加载时**构建成功**（16384 也建起来了）、请求**被接受**，然后在**准备阶段超过 30 s 超时** ✗。
8192 的同位置日志显示一切在 **~0.3 s** 内完成（`pool captured 02.42 → listening 02.61 → accepted 02.70`）✓。

⇒ 真正的墙是**准备（preparation）随容量增长**，而不是显存。这与我更早测到的
`deferred_reasons[preparation]` 才是一致的（第 25/26 轮我就见过 preparation 递延，却把它归到槽位/显存上）。

**（2）`state_reservation_bytes` 的大头不是 KV**（§13.29 的"1.07 GB = 32768 × 32 KB"是错的）。
`budget.rs::program` 的项依次是：`attention_scratch` + **`ActivationArena::required_bytes(graph, lanes)`**
+ 递推状态（按 `prompt` 通道）+ ... + KV 页（`capacity × columns × 2`）。以交付配置算：
`capacity=128` 时 KV 页只有 **~0.26 MB**，而**激活 arena（按 lanes 计的整图激活）才是 1.12 GB 量级的大头** ✓。
所以那条 1.12 GB 的准入拒绝是**激活预算**造成的，与"每序列按最大长度预留 KV"无关。

⇒ 这也解释了为什么 §13.29 那条"池内减免"对交付 profile **不生效**：它减的是 KV/状态那一份，
而真正压垮准入的是**固定激活预算**——减免它需要的是"池已提供共享激活 arena"这个事实被计价承认，
而不是把 KV 归零。

**下一步的正确杠杆（都有了实测支撑）**：
- 长上下文的墙是**准备开销随容量增长**（16384 就 >30 s）⇒ 要么让**池覆盖该容量**从而不走私有准备，
  要么让准备/预填充按**实际 prompt 长度**而不是预留容量来做；
- 准入的墙是**激活预算**（固定且大）⇒ 池内请求应当只被收取"池未覆盖的那部分"。

### 13.31 把"准备"和"16 槽"都量到底：真正的双重预留在**激活/工作区 scratch**，不在 KV

**（1）16384 的挂起是 MTP 特有的。** 同一台机器、同一常量：

| 配置 | 结果 |
|---|---|
| `CB_SLOT_TOKENS=16384`，**不开 MTP** | ✅ 正常：wall 3.0 s、TTFT 1.93 s |
| `CB_SLOT_TOKENS=16384`，**开 MTP**（`--num-speculative-tokens 2`） | ❌ **挂起**：30 s 后 `resource preparation acknowledgement timed out`，TTFT 0 |

日志里池子在两种情况下都**构建成功**（`CUDA slot pool captured at load width=4 capacity=16384`），
差别只在 MTP ⇒ 挂起出在 **draft/验证准备**这一段，与容量叠加时才触发（8192 时正常）。

**（2）§13.29 的池内计价落地之后，16 槽配置**（`CB_DECODE_SLOTS=16`）**依然失败**，但这次日志给出了
真正的原因，而不是"准入拒绝"：

```
WARN CUDA slot pool prewarm failed; will retry on the first batchable step capacity=8192
     last_failure=width 2: Capacity: resident F32 state exceeds available device memory
{"code":"Capacity","message":"dataflow scratch exceeds program workspace budget"}
```

第一条：**池的 prewarm 连 width=2 都建不起来**（`resident F32 state exceeds available device memory`）；
第二条：某个请求的程序**编译**就失败——`compilation.rs:109` 把 dataflow 的 scratch 与
`workspace_limit`（设备的 arena 预算）比，**scratch 超了**。

**（3）结论：真正的"双重预留"是激活/工作区 scratch，不是 KV。**
池在构建时为自己的激活 arena 预留一份 workspace，**每个私有请求再为自己的程序预留一份**；
池一宽（16 槽）或容量一大（16384），两边的 scratch 抢同一份预算 ⇒ prewarm 失败 / 程序编译失败。
这与 §13.30 量到的"1.12 GB 报价由激活 arena 主导"完全一致 ✓，也解释了为什么 §13.29 减免 KV
那一份**减错了项**。

**⇒ 第④步的正确切法**：池内请求的程序应当以**池已持有的共享 arena 预算**去编译
（`crates/model/compiler` 的 `workspace_limit` 来源：`device.arena_budget_bytes()`），而不是各自新开一份。
这条比"把 KV 计价改成块数"更贴近实测瓶颈。

（`CB_DECODE_SLOTS` 已回退到 4、`CB_SLOT_TOKENS` 保持 8192；matrix 复验 16/16 通过。）

### 13.32 池宽的天花板是**每序列的递推状态**，不是 KV，也不是激活 arena

想用"给池留出 arena 预算"来松开池宽（在 `select_prompt_width` 前扣掉池的
`ActivationArena::required_bytes(graph, CB_DECODE_SLOTS × batch_width)`），实测**没有奏效**：

```
INFO prompt graph width selected prefill_width=256 narrow_prefill_width=64   ← 仍然选了最宽档
WARN CUDA slot pool prewarm failed ... capacity=8192
     last_failure=width 2: Capacity: resident F32 state exceeds available device memory
```

两个读数很有信息量：

1. 扣掉池的 arena 之后**仍然够**选 256 档 ⇒ 激活 arena 那一项**不是** 16 槽场景的瓶颈（我 §13.31 的
   推断只对了一半）；
2. prewarm 在 **width=2** 就失败，报的是 **`resident F32 state`**，不是 arena、不是 scratch——
   也就是**每序列的递推状态**（48 层 gated-delta/conv，本模型 ~151 MB/序列）本身放不下：
   加载时 `resident=22.9 GiB state_budget=4.8 GiB`，16 槽 × ~170 MB ≈ 2.7 GB 还要与
   prewarm 重试（16→15→…→2，每次都要分配再失败）的峰值叠加。

**结论（对目标很关键）**：**KV 分页并不能单独提高并发**——因为 48 层递推状态**必须每序列一份**
（这正是本目标里写明的约束），它才是池宽/并发的硬上限。这也解释了我一路测到的现象：
- 第 25 轮"16 槽被拒"、第 36 轮"prewarm 失败"，都发生在**状态**这一侧；
- `CB_SLOT_TOKENS`（8192）之所以能落地并带来 −13%/−22%，是因为它让**更多请求能进池**（省掉重复准备），
  而不是因为省了 KV 显存。

⇒ 想真正提高并发，需要的是**减少每序列递推状态的占用**（量化/共享 checkpoint）或**降低池的槽状态开销**，
而不是继续在 KV 或 arena 上做文章。（`CB_DECODE_SLOTS` 已回退到 4，matrix 16/16 通过。）

### 13.33 每槽状态的确切账目：**262 MiB KV + 149 MiB f32 递推状态 = 405 MiB**，于是并发 ≈ 12

给池的构造加了临时插桩（打印每槽状态分解与总字节），在 `capacity=8192` 下实测：

```
KV state budget: capacity=8192 f32_elements=39223296 f32_mib=149 kv_bytes=268435456 slot_mib=405
KV state part: TensorId(1050) shape=[48,128,128] mib=3   （48 层各若干份，共 149 MiB）
KV state budget: capacity=8192 f32_elements=16777216 f32_mib=64 kv_bytes=0 slot_mib=64  ← draft 池的一槽
加载时：resident=22.9 GiB  state_budget=4.8 GiB  max_resident_states=62  prefill_width=256
```

- **KV 是 256 MiB/槽**，而且这个数**正好等于** 16 个全注意力层 × 8192 行 × 4 个 KV 头 × 256 × 2(K+V) × 1 B
  = 268435456 B ✓ —— 也就是**没有浪费**（我一度以为是 8–16 倍超配，实为 16 层）；
- **f32 递推状态 149 MiB/槽**（48 层 gated-delta/conv，以 `[48,128,128]` 的若干张量为主）；
- ⇒ **每槽 405 MiB**，而引擎给的 `state_budget = 4.8 GiB` ⇒ **最多约 12 槽** ✓✓ —— 这正好解释了
  §13.18/13.31/13.32 里"16 槽 prewarm 失败"的算术。

**这把整个调查收口成一张清晰的权衡表**：

| 想要 | 代价 | 现状 |
|---|---|---|
| 槽容量大（长会话能进池） | 每槽 KV 线性增长（8192 → 256 MiB） | `CB_SLOT_TOKENS=8192` ✓ |
| 槽数多（并发高） | 每槽 ×(KV + 149 MiB 状态) | `CB_DECODE_SLOTS=4`（4×405=1.6 GiB ✓；16×405=6.3 GiB > 4.8 ✗） |

**而这个"容量 ↔ 并发"的互斥正是第②/③步（块池 + 按需分配 + 块级共享）要消除的**：序列只该持有它
**实际用到**的块，而不是每槽预留满容量。所以本目标的第②步不是"锦上添花"，而是**唯一**能同时要
长上下文与高并发的机制 ✓。

**顺手发现的一个独立问题（未修，记录）**：开了 MTP 之后，`(2048, 16 槽)` 与 `(16384, 4 槽)` 两种
池几何都会在**准备阶段挂起 30 s（0 token）**，而 `(8192, 4)` 正常。看起来与**池化推测图的捕获**
在特定 (capacity, width, verify) 组合下有关，值得单独排查。

**另一个单点测量（需复现）**：`(2048, 16)` 配置下 N=8 并发 wall **4.99 s**（第 25 轮同口径基线 6.53 s，
−24%），TTFT max 3597 vs 5008 ms；但日志显示该池实际只捕获到 **width=4**（内存不足自动降宽），
所以这个改善**不是**来自更宽的池，机制待复现确认，不当作结论。

## 十四、KV cache 管理重构：交付状态与结论（2026-10-10）

### 14.1 已落地（默认路径，均有实测与门禁）

| # | 改动 | 实测 | 出处 |
|---|---|---|---|
| 1 | **① block table 间接寻址**：`append`(预填充写) + 预填充读(SIMT/张量核) + decode 读(SIMT/split-KV) 五处核全部经 `table[逻辑块] → 物理块` 取数 | 恒等映射下官方 matrix **12/12 trial token 逐位相同**、越界 0 次 | §13.16、§13.17 |
| 2 | **`CB_SLOT_TOKENS` 2048 → 8192**（池槽容量，让更长会话能进池） | 4×2560 并发：wall 10.59 → **9.21 s（−13%）**、TTFT 中位 7491 → **5808 ms（−22%）**、`gpu_budget` 递延 167 → **113（−32%）**；matrix 16/16 且 token 逐位一致 | §13.20 |
| 3 | **池内计价**（`state_reservation_bytes_for`）：占用池槽的请求不再为自己那份状态二次预留 | matrix 16/16（改动前后一致）、27 单测、4 GPU 测试全绿 | §13.29 |
| 4 | `--chunked-recurrent`（递推层分块预填充，开关默认关） | TTFT：short **−15.7%**、long **−20.1%**、batch4 **−14.6%**、hot_long **−16.0%** | §12.20 |
| 5 | `INFER_CUDA_TILED_ATTENTION`（张量核 attention，默认关） | 位置 ≥3072：**256.0 µs vs SIMT 1587.2 µs = 6.2×**；短桶 QT=16 也赢 | §13.9 |

### 14.2 未落地（②–⑤）：卡在哪、以及"不要再走"的路

**② 共享块池 / 真实分配器：未落地。** 设计定稿并两次实现（§13.21 触点清单、§13.22–13.23
`KvSpan` 双侧偏移、§13.24 拆 `capacity` 两个角色），**编译与 clippy 全绿但 draft 测试设备越界**：

```
attention.rs:113: partition access out of bounds: dim 1, block index >= ceil(?/32)
```

排查过程本身有价值（都已记档）：表**形状**自洽（§13.27）、表**取值**也在范围内（§13.28），
因此"表/arena 错配"整族假设被实测排除；故障**只在 draft 路径**、且 `capacity` 角色拆分
（`arena_capacity`）之后依然存在（§13.26）⇒ 下一处该查的是 **draft 私有 program 的
`identity_table` 与它的 arena 归属**，以及**回放时**那次读的 `(lane, position, block, 表项)`。
补丁保存在 `/tmp/shared-arena.patch`（含已知的三处 `program.rs` 修补）。

**④ 计价改为块数：只落了第一刀。** §13.29 的池内减免是**正确但不生效**的——因为
`reserved_tokens` 本来就是"prompt + max_new_tokens"（§13.30 更正），而 1.12 GB 的报价大头是
**固定激活 arena**，不是 KV ⇒ 减 KV 减错了项。

**当时试过的路径**：cuBLASLt 部分模式失败（“必须离线重量化”的结论已由 16.10 修正）、GEMM tile 扫描、
SDPA `load_pipelined::<4>` TMA、delta 代数重写、value-dim 拆分、per-node outlier 理论、
"池变宽"路线（§13.18/13.25/13.31/13.32 四次否证）、prefill rung 预算（§13.32）。

### 14.3 硬边界（实测账目，决定了并发上限）

| 量 | 值 | 来源 |
|---|---|---|
| 每槽 KV（capacity 8192，fp8） | **262 MiB** = 16 层 × 8192 × 4 头 × 256 × 2 × 1 B | §13.33 |
| 每槽 f32 递推状态（48 层 delta/conv） | **149 MiB** | §13.33 |
| **每槽合计** | **405 MiB** | §13.33 |
| 引擎 `state_budget` | **4.8 GiB** | 加载日志 |
| ⇒ **并发上限** | **≈12 槽**（16×405 MiB = 6.3 GiB > 4.8 GiB） | §13.33 |

**容量 ↔ 并发是互斥的**：槽容量大→长会话能进池但每槽 KV 线性增长；槽数多→并发高但每槽 ×405 MiB。
**消除这个互斥正是第②/③步（块池 + 按需分配 + 块级共享）**——序列只持有实际用到的块。
所以第②步不是优化，而是长上下文与高并发共同依赖的机制。

**另一个独立缺陷（未修）**：开 MTP 后，`(2048, 16 槽)` 与 `(16384, 4 槽)` 两种池几何会在
准备阶段**挂起 30 s（0 token）**，而 `(8192, 4)` 正常 ⇒ 指向池化推测图捕获在特定
(capacity, width, verify) 组合下的问题。日志证据：`reason=Failed("resource preparation
acknowledgement timed out")`（§13.30/§13.31）。

### 14.4 推荐默认

- 保持 `CB_SLOT_TOKENS = 8192`、`CB_DECODE_SLOTS = 4`（12 槽账目下的安全落地值）；
- 长 prompt/多轮场景开 `--chunked-recurrent`（TTFT −15~20%）；
- 长上下文（≥3072）且可接受末位差异时开 `INFER_CUDA_TILED_ATTENTION`（6.2×）；
- `--max-model-len` 越大，单请求状态越贵（KV 按 token 线性）；`reserved_tokens` 与
  `CB_SLOT_TOKENS` 的关系决定一个请求**能否进池**（能进池就省掉重复准备）。

### 14.5 目标完成度（诚实结论）

| 目标步骤 | 状态 |
|---|---|
| ① 块池 + block table 间接寻址（恒等映射验证） | ✅ **已完成并验证**（五处核 + bind + 预算；12/12 token 一致） |
| ② 真实块分配器（按需分配、无每序列预留） | ❌ 未落地（编译过、draft 越界未定位到最后一处） |
| ③ 块级前缀哈希共享 + COW | ❌ 未开始 |
| ④ 准入/计价改为块数 | ⚠️ 部分（池内减免已落地；`reserved_tokens`/激活 arena 侧未改） |
| ⑤ token-major 布局 + 张量核 SDPA | ❌ 未开始（张量核 attention 已单独立项并实测 6.2×） |

**因此本目标未完成**；已完成的是：间接层（①）、两项可测的默认改动（2/3 行）、
以及把"为什么长上下文/高并发上不去"从猜测变成了**实测账目**（§13.18–13.33）。

## 十五、P0a/P1/P2：CUDA 闭环 SPI 分页契约（2026-10-11）

按 [KV 分页方案](../../plans/performance/kv-cache-paged-attention.md) §五 的 P0a/P1/P2 落地，
本节记录实现口径与全部实测。改动已提交：`220699c`（P0a 工具与 manifest 修复）、
`87ca6d0`（P1/P2 实现）；
二进制指纹：before `b5058da5…`、after `e4d7f398…`（`target/release/infer`）。

### 15.1 P0a：冻结 manifest 的 role 方向缺陷【已修】

`freeze-baseline.py` 的 `freeze()` 原来按命令行顺序写 `reports[].role`，与 gate 数值的
实际方向无关 ⇒ 三个冻结基线（`2b-mtp0`/`27b-mtp0`/`27b-mtp2-serving-v1`）的
`baseline`/`candidate` 角色全反（kv-cache-paged-attention.md §2.1）。修法：`freeze()`
改用 `gate.orientation()` 判定方向后按角色序写 `reports[]`/`engines`，`source`/`hardware`
跟随被测侧，gate 块新增 `ratio_orientation`；三个 manifest 的 role 交换为
`baseline=vllm / candidate=native`（manifest 本身不入文件哈希校验，报告文件未动）。
验证：`test_freeze_baseline`/`test_verify_baseline`/`test_compare_results` 79 测试全绿；
`verify-baseline.py` 对三个基线全部 "numbers recomputed"（`compare()` 按 engine identity
重定向，与文件内顺序无关）；`./tools/check/gate.sh tools` 全绿。

### 15.2 P1：CUDA `kv_cache()`【已落地并验收】

实现口径（`crates/backend/cuda/src/executor/provider.rs`，细节以代码注释为准）：

- 块 = `KV_BLOCK_TOKENS = 32` token；`bytes_per_block` 从目标图的 `AttentionKv` 状态实算
  （27B fp8 = **1 MiB/块**，16 层 × 32 × 1024 列 × 2 × 1 B）。
- `total_blocks = max(池块数, MAX_CAPACITY_TOKENS/32 = 1024)`——下限保护私有路径
  （池降宽/缺席时长请求仍能被私有预留服务）。`active_blocks` = 在租槽的 Σceil(历史/32)；
  `available_blocks` = 池块数 − active（在租槽尾块是增长的来源，必须算入，否则 4 槽全租时
  所有请求被误 defer）；`cached_blocks` = 前缀缓存内容块当量（仅观测）；`pool_bytes` 仅 KV。
- 新增 `SlotLedger`（allocations/releases）与 `PrefixCache.evictions`/`cached_tokens()`；
  池不存在（prewarm 失败）→ 返回 `None`，行为与改动前完全一致。
- **陷阱（已修）**：`crates/service/cli/src/support/serving.rs` 在有 kv_cache 时把
  `num_gpu_blocks = total_blocks` 灌进 engine 逻辑页池；若不同调 `block_size`
  （Metal 16→16 无操作，CUDA 16→32），逻辑池容量会从 4096×16=65536 token 悄悄缩到
  1024×16=16384，长上下文被拒。调和后为 1024×32=32768 = `max_model_len` = 物理池容量。

验收证据：`/native/v1/runtime` 的 `config_readback.kv_cache` 由 **`null`**（冻结基线记录的
缺口的原样）变为 `{block_size:32, total_blocks:1024, free_blocks:1024, bytes_per_block:1048576,
pool_bytes:1073741824, …}`；`/native/v1/observability` 新增真实
`available_kv_blocks`/`active_kv_blocks`/`cached_kv_blocks`/`kv_pool_bytes`。数值自洽性抽查：
4×2560 并发（4 条相同 prompt）跑完后 `cached_kv_blocks=320` = 4×⌈2560/32⌉，
allocations/releases 配对、active 归零。

### 15.3 P2：`state_page_growth()`【已落地；验收口径按实测修正】

口径：在租序列 `allocated_pages = ceil(history/32)`（跨块边界 growth=1）；私有序列
`allocated_pages = ceil(capacity/32)`、`required_pages` 恒 0（满容量预付，物理上从不新分配，
如实上报）；未知 state → `Error::invalid`（与 Metal 一致）。`cost_query.page_growth`
不再恒 `None`。

**计划写的验收"`deferred_reasons` 中 `state_capacity`/`gpu_budget` 在 4×2560 与 2×8000
下显著下降"在实测中不成立，且原因已查清**（同机同刻 A/B，`--chunked-recurrent`、
mtp2、0.88、32768/16）：

| 负载 | before deferred 增量 | after deferred 增量 | wall before → after |
|---|---|---|---|
| 4×2560（冷） | TokenBudget +24、GpuBudget +72、Preparation +8 | **逐项相同**（+23/+72/+8） | 11.96 s → 13.17 s（噪声内） |
| 4×2560（热） | Preparation +8 | TokenBudget +23、GpuBudget +72、Preparation +8 | 11.01 s → 10.76 s（噪声内） |
| 2×8000 | Preparation +4 | Preparation +4 | 19.12 s → 19.03 s |
| 5×8000（越池） | Preparation +9，ok=4 **fail=1** | Preparation +9，ok=4 **fail=1** | 40.4 s → 38.7 s |

`state_capacity` 在全部负载、两侧构建上恒为 0；`gpu_budget` +72 两侧一致。原因：这两组
负载的递延来自**每步 token 预算**（`max_num_batched_tokens` 切片）与**执行预算**，
KV/状态计价从来不是它们的瓶颈；池装得下时按块计价无从体现。P2 的交付物因此是
"调度器每步拿到真实增长数"这一机制本身（可证、可查），而不是这两组负载的递延变化；
该验收条款对这两组负载不适用，已在方案 §五 如实改写。

**越池边界（5×8000 = 40000 token > 4 槽池）**：第 5 条请求在两侧构建上**同样**以
`Failed("resource preparation acknowledgement timed out")`（30 s、0 token）失败——
与 §14.3 末尾/方案 §九.1 记录的准备挂起同一签名。即这是**既有准入竞态**（5 条请求在彼此
成为 active 之前全部通过准入，第 5 条的私有预留撞上已租满的池），**不是 P1/P2 回归**；
matrix 4 并发以内的负载覆盖不到它。修法方向：准入计入"准备中"的在途预留（单独立项）。

### 15.4 官方矩阵 A/B（行为保持的核心证据）

`artifacts/kv-paging-p1p2-20261011/`，run-id `kv-before` / `kv-after`，
`--checklist benchmarks/profiles/27b-mtp2.json`、`--extra --chunked-recurrent`：

- **16/16 trial 完成、28/28 请求流 token 逐位一致**（含 warmup 流），0 处 mismatch；
  两侧 checklist 均 compliant、`completed=true`。
- `hot_long` 前缀复用两侧均 3584 token/repeat；组 wall（before → after，3 repeat）：
  short 1.034/0.849/0.958 → 1.048/0.864/0.958；long 1.067/1.093/1.166 → 1.075/1.113/1.163；
  batch4 2.468/2.384/2.178 → 2.338/2.351/2.178；hot_long 0.914/0.904/0.900 → 0.913/0.914/0.912。
  全部在 run 间噪声内，**无性能回退也没有虚报收益**（P1/P2 是只读上报，行为不变是预期）。
- 环境备注：测量窗口内同机有另一工作线的 cargo 构建在跑（CPU 竞争），绝对值仅作参考；
  token 一致性不受其影响。

### 15.5 门禁与剩余项

- `cargo clippy --locked -p infer-backend-cuda --features cuda -- -D warnings`、
  `cargo check --all-targets`、`cargo test`（28 单测，含新增
  `kv_block_bytes_counts_attention_states_in_storage_bytes`）、GPU 测试
  `resident::slot_batch`/`slot_verify` 4/4、release `--lib attention_decode` 2/2，全绿。
- **P3 的 `CB_DECODE_SLOTS=16` prewarm 验收物理不可达**：16×405 MiB ≈ 6.3 GiB > 4.8 GiB
  状态预算（§14.3），只有 P5 的共享 arena 能解开；方案 §五 已注明。P3 剩余价值
  （私有路径报价块化）未做。
- **P4/P5/P6 未做**；P4 参照实现应看 `crates/engine/state/src/kv.rs` 的
  `KvCacheManager`，§八 的"共享 arena + 偏移表"路线维持否证。
- **P0 的 ≥5 单元成对基线阻塞**：本机无 vLLM（`import vllm` 不可用），需先恢复环境。


## 十六、2026-10-11 当前 main 对比与下一阶段的优化边界

本轮从合并后的 `c5aec9e6c1d663ff1e19b8f485293e13679a09f4` 出发，先阅读实验记录、核对实际
调用路径，再重新运行同机 native/vLLM。此前的 GEMM 改动已合入 main，不能继续拿合并前的
绝对时间作为当前缺口。45 上 `artifacts/vllm-compare/bin/python` 可以导入 **vLLM 0.31.0**；
因此 KV 计划中“无法导入 vLLM”的历史阻塞不再成立。

### 16.1 当前差距：诊断配对，不是正式签发

RTX 5090 仅作为验证设备；驱动 `615.78.08`。27B NVFP4、MTP 0、同一份 token workload、
每请求生成 64 token；GPU memory utilization 0.88、max model length 65536、max sequences 16。
两个引擎依次新启动，每个 case 排除一次 warmup、测量三次。**只有一个独立配对单位**，
不足基线计划要求的五个，以下数字用于确定优化顺序，不用于宣布性能达标。

下表为请求样本中位数，时间单位 ms，比率为 native/vLLM；batch4 的请求时间不代表整组 makespan。

| case | 输入 token | native TTFT | vLLM TTFT | TTFT 比率 | native TPOT | vLLM TPOT | TPOT 比率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 51 | 42.307 | 51.829 | 0.816 | 16.451 | 14.607 | 1.126 |
| long | 511 | 237.159 | 63.706 | 3.723 | 16.451 | 14.652 | 1.123 |
| batch4 | 52 × 4 | 172.224 | 94.005 | 1.832 | 21.505 | 16.537 | 1.300 |
| hot_long | 3647 | 71.434 | 70.672 | 1.011 | 16.932 | 14.623 | 1.158 |

short/long/batch4 的测量 trial 均报告 prefix reuse = 0。hot_long 每次 native 重用 3584 token、
vLLM 重用 3136 token，剩余计算量不同，**该格不能作为同等预填充工作的速度对照**。
三个进程内 repeat 也不能当成三个独立配对。当前尚未覆盖 2B、MTP 2、持续及混合负载的新五配对矩阵。

原始报告、telemetry、workload 和制品身份保存在：
`/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/current-pair/`。
冻结的 native release 制品 SHA256 为
`20330634ff509035e0851d87ee414a7cde7b572fc8c5853e5baeff5fb34d2e07`，workload SHA256 为
`2d88b924fa4ae3e83a6763e0c2eca986791808cf21ed0e0fcbafbe26b05e6a15`。

### 16.2 从代码确认的词表输出浪费

`resident/batch.rs::capture32` 的 last 图仍把 logits 的 Linear 节点交给
`prefill_projection::record`，后者按 `arena.lanes()` 对所有 prompt 行做投影；
`BatchGraph::run32` 却只回读 `tokens.len() - 1` 那一行。`ActivationArena::plan` 又把每个 slot
都按完整 prompt 宽度分配，包括这个终端词表输出。`program_capacity` 已明确禁止 logits
被图内节点消费，因此这个输出的中间行没有后续图内消费者。

这提供了一项不依赖 GPU 型号的独立优化：用设备端 metadata 选取最后一个有效 hidden 行，
保留既有投影的权重格式、输入舍入和 K 累加顺序，只生成一行 logits；prompt 的 hidden 行仍
完整保留给 MTP priming/Full readout。相应的 arena、admission 和 reclaimable bytes 必须一起
改，不能只缩分配后继续沿用旧的宽度除法或旧的可回收字节数。

以 248320 词表、F32 logits 为例，256 行约 242.5 MiB，单行约 0.95 MiB。这个存储浪费
同时限制更宽 prompt 图和并发驻留；减少它后，再根据设备预算验证 512-token 图和池化请求的
宽图。**这些是代码确认的浪费和待测方案，尚没有服务侧收益测量，不能提前写成加速结论。**

### 16.3 GDN 的矩阵路线：更快的求解仍欠正确性验收

新实验将 gated-delta 的逐 token rank-1 递推改写为 32-token 的因果线性系统，并用补偿 BF16
tensor-core 乘积与 F32 累加计算状态和输出。先实现串行三角求解，再利用严格下三角矩阵
`A^32 = 0`，通过有限乘积 `(I-A)(I+A^2)(I+A^4)(I+A^8)(I+A^16)` 求逆。
这改变浮点计算顺序，不能视为与既有递推逐位等价。

隔离 CUDA-event 测量如下，时间单位 µs。矩阵求逆版在同一 GPU 测量窗口前后各跑一次，
中间重跑已有 chunked 核；串行三角版来自此前的诊断窗口，没有独立签发交错服务收益。
每行的算子测量内部还与逐 lane 图交错，包含 L2 清理和预分配 graph replay。

| 几何 (KH,VH,D,rows) | 逐 lane | 已有 chunked | 串行三角矩阵版 | 矩阵求逆版两次 |
|---|---:|---:|---:|---:|
| (16,48,128,32) | 172.8 | 72.736 | 61.056 | 58.176 / 58.240 |
| (16,48,128,64) | 335.1 | 131.696 | 146.704 | 112.672 / 112.384 |
| (16,48,128,128) | 662.8 | 252.576 | 276.992 | 206.112 / 205.824 |

串行三角版在 64/128 行比已有 chunked 慢，应排除。矩阵求逆版在这些几何上有约 15–20%
的进一步收益，但**没有进入生产 dispatch**：三乘积版在 D=32、29 个有效 token 的跨分块
输出一致性检查中出现 `1.3411045e-7`，超过该夹具的 `1e-7` 阈值；四乘积、8 worker warps 的
后续版本仍出现 `1.8719584e-7`。失败不能通过放宽既有测试或只报告对逐 lane 的倍数来隐藏。
串行三角版完成了独立 F64 参考的全套几何、tail、offset 测试；矩阵求逆版的测试因上述检查
提前停止，不能声称完成了同样的数值覆盖。

原始日志、候选补丁和测试制品 SHA256 保存在上述持久目录的
`wy-experiment/`、`wy-neumann/`、`wy-worker8/`、`wy-precise4/`。实验源码归档后，工作树已恢复
既有递推实现及其测试。继续这一方向时，需先独立证明矩阵算术误差、跨 chunk 状态和真实
checkpoint 的固定上下文 logits，再验收任务质量及 MTP 接受率；不能把内核倍数直接乘到 TTFT 上。

### 16.4 张量核 attention 的块映射漏洞

检查 `attention_prefill::decode_tiled` 发现：循环已查询 `physical` block，但 key 读取仍使用
逻辑 `block`；value 使用 `physical`。identity table 会掩盖该错误，非连续布局则导致 K/V
来自不同位置。本轮将 key 也按 `physical` 读取，并扩展现有 GPU fixture：将整个已有 KV
历史和新 append 区间一起重排，再用非 identity table 对照独立参考，覆盖两种 query tile、
tail、window 和 draft offset。该回归验证通过。默认 dispatch 的数值策略未因这个修复改变。

### 16.5 后续优先顺序与验收

1. 先减少终端词表投影的无用行，并修正存储计价；验证尾块最后行、Full readout 和 MTP priming。
2. 用实测预算和 replay 数评估更宽 prompt 图及池化 prompt workspace，避免再次把短 prompt
   强塞进宽图。保留 63/64/65、127/128/129、255/256/257 等边界与并发容量覆盖。
3. GDN 矩阵算法单独通过数值和质量验收；张量核 attention 单独测真实模型、长短上下文及块表，
   再评估组合收益。不能仅打开已有开关就宣称解决了浮点口径和 MTP 接受率问题。
4. 针对 batch4/MTP 的逐 lane 状态算子 dispatch 和快照流量重新画像，优先减少每步固定工作。
   CPU 忙、GPU 利用率或旧版 roofline 数字都不能替代当前路径的设备计时。
5. 候选通过必要 correctness 检查后，CI 与性能测试并行开展；正式结论仍需要两模型及支持的
   MTP profile、全部 case 的至少五个独立交错配对，以及吞吐、尾延迟和任务质量。每格相对
   vLLM 的目标是 1.10x 以内；本轮未达到该目标。


### 16.6 终端 logits 单行化：候选实现与正确性（默认路径已撤回）

`BatchGraph`候选  的  宽 prompt 捕获为符合条件的终端 Dense/FP8 Linear 输出分配一行 logits。
先按设备端 `count - 1` 收集最后一个有效输入行，再保留旧 prompt 投影的 tile 几何、输入舍入/
动态量化与 K 累加顺序。27B 的输出头实际是动态 per-token FP8，已覆盖这条生产路径；没有
改用 decode 的 split-K 算法。旧 FP4 head、非 Linear 输出、被消费的 logits、hidden/logits
同一 tensor 和 vendor 投影模式保留原有存储与投影路径。

单行结果使用独立 slot，避免把曾与 logits 复用的前序 activation 也缩成一行。hidden 仍按完整
prompt 宽度保存，Full readout 和 MTP priming 的行语义不变。admission 按每张 prompt arena
及 gather scratch 的实际尺寸计价，回收额度也按实际保留字节计算，含重新建立/丢弃验证图的路径。
自动宽度选择使用同一价格，但本改动没有增加新的宽度档位。

45 上 release CUDA 验证：默认 lib tests **29 passed / 43 ignored**；独立存储测试覆盖旧 slot
复用、零宽度和溢出；GPU 测试覆盖尾 K/N 与最后行边界（1/31/32/33/63/64/65/127/128/129/255/256），
与旧 full-width BF16 MMA 的有效行逐位相同。既有宽 prompt 的尾块/标量续接测试通过。真实
27B checkpoint 使用 511 个固定 token、128 行 prompt 图，对照旧 full-width 捕获，所有
hidden、最后行 logits、验证结果、回滚后的连续 decode **逐位相同**。生产 CUDA lib Clippy 通过；
全 tests Clippy 仍存在本轮以前的 fixture lint，不能表述为所有硬件门禁已绿。

候选源码补丁、测试制品身份和原始日志保存于
`/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/last-logits/`。
以上仅证明候选正确性与实际取消了中间行。下节服务测量发现回退后，main 已恢复原有 arena、计价与完整行投影。


### 16.7 单行 logits 的服务对照：吞吐与首 token 中位数出现权衡

测试使用 45 上冻结的 c5aec9e baseline 和独立候选二进制，两组新进程配对按 AB/BA 交错；
每个进程包含一次 warmup 与三次测量，输出 64 token，模型内存比例 0.88、max-num-seqs 16。
这只有两个独立单位，是诊断数据，不满足正式签发要求的五组，也不是新的 vLLM 对比。
源码身份由各自工作树记录，冻结二进制 SHA256 独立记录。误标源码的早期试跑保留但排除。

以下为各组 request TTFT 中位数变化，负值表示候选更快；batch4 使用各请求指标，
不能当成整组 makespan。M32 候选沿用原 prompt tile；M16 只缩小 FP8 最后行 tile。

| 模型 / MTP / case | M32 第一组 | M32 第二组 | M16 第一组 | M16 第二组 |
|---|---:|---:|---:|---:|
| 2B / 0 / batch4 | -17.2% | -12.4% | 未测 | 未测 |
| 27B / 0 / long (511) | +1.5% | -0.4% | 未测 | 未测 |
| 27B / 0 / short | +4.4% | +13.9% | 未测 | 未测 |
| 27B / 2 / batch4 | +34.6% | +24.4% | +31.5% | +32.6% |
| 27B / 2 / long (511) | -1.3% | +0.2% | +0.7% | +2.0% |

M16 的 27B/MTP2 batch4 TPOT 同时增加 3.0% / 8.2%，request wall 增加 2.1% / 3.0%。
缩小 tile 没消除首 token 中位数回退，未满足每格延迟的默认签发要求：单行路径已从 main 撤回，
保留之前 GEMM 的已验证优化和独立 KV block 寻址修复。所有候选源码、原始测量及测试日志
仍在持久制品目录，不需要保留实验分支。

2B/MTP0、27B/MTP2 的 M32 两组和 M16 的 27B/MTP2 两组，每组 21 个测量请求的输出 token
均与 baseline 相同。27B/MTP0 的 M32 第一组全部相同，第二组有 6 个 batch4 请求不同；
baseline 两次自对照也有 2 个 batch4 请求差异，非 batch 请求稳定。固定上下文的 hidden/logits、
验证/回滚/续接逐位检查通过，但这不能替代并发调度下的质量验收，也不能把差异归因于新算术。
M16 的 64/128 行真实 27B checkpoint 检查都逐位通过。

已知存储收益仍成立：248320 词表的 256 行 F32 logits 从约 242.5 MiB 降至约 0.95 MiB；
但这些服务指标不能只按 request 中位数判断：下一节补齐原报告的 group makespan 与图级计时。
仍需拆分最后行投影本身和 arena/admission 变化，不能用缩 tile 假设替代测量。

原始数据：`last-logits/paired-qualified/`（M32）、`last-logits/m16/paired/`（M16）。
冻结 baseline SHA256：`20330634ff509035e0851d87ee414a7cde7b572fc8c5853e5baeff5fb34d2e07`；
M32：`9b8b8e05a504e4bcb363fb6669aafa91f92f0ca5f822a689c1f0909d2971a984`；
M16：`0c86b8287dfb365f2dab03d1e8a85d9e8ba99d5f2a172a4c2331ef1a0674db6c`。


### 16.8 同一组原始数据的吞吐/排队收益与图级诊断

16.7 的首 token 中位数回退不表示候选的所有服务指标都变差。使用同一报告已记录的
`group_makespan_seconds`（整组时间，不是 request wall 中位数）重新统计 27B/MTP2/batch4：

| 候选 / 独立单位 | baseline 整组中位数 ms | candidate 整组中位数 ms | 整组时间变化 |
|---|---:|---:|---:|
| M32 / 1 | 2329.74 | 1534.44 | -34.1% |
| M32 / 2 | 2257.79 | 1686.59 | -25.3% |
| M16 / 1 | 2243.40 | 1469.86 | -34.5% |
| M16 / 2 | 2135.67 | 1426.89 | -33.2% |

每组 12 个测量请求中，M32 的最大 TTFT 从 1499.02/1415.73 ms 降至 292.63/257.05 ms，
平均 TTFT 从 454.92/453.53 ms 降至 209.05/202.55 ms；M16 的最大 TTFT 从
1472.43/1374.77 ms 降至 252.65/253.34 ms。小样本最大值不能冒充 p95/p99 签发。
baseline 每个 batch4 repeat 都有一个请求等待约 1.2–1.5 s，候选消除了这一长等待；
其余三个请求的等待分布同时改变，造成中位数反而更慢。因此需要同时验收吞吐、首 token
中位数、尾延迟与质量，不能只筛选其中有利的指标。默认撤回针对既定的逐格延迟要求，
并不否认上述实测吞吐/排队收益；优化驻留与调度仍是值得继续的方向。

另跑一组带 `INFER_CUDA_PREFILL_PROFILE` / `INFER_CUDA_PROFILE_GRAPH_ONLY=1` 的
M32 同权重对照，仅作诊断，不纳入未插桩的性能签发。图内设备计时中，52-token 的
`prefill_last` 中位数为 baseline 36.029 ms、候选 36.287 ms；单请求 3-token 验证为
23.215/23.235 ms，3 请求合并的 9-token 验证为 24.393/24.389 ms。
baseline 没有 4-token slot decode 或 12-token slot verify，候选分别出现 164/82 次；
候选 12-token 验证中位数为 24.773 ms。这些 count 是整个插桩运行的 replay 数，
包含 warmup 与全部 case，不能冒充独立配对单位或只对应某一个 batch repeat。

图内相同工作量的时间接近，而合并执行的规模从至多 3 请求扩展到 4 请求，支持将收益
优先归因于驻留/调度行为变化，而不是最后行 GEMM 的设备时间加速。这仍是证据支持的
推断；要区分 arena 容量、admission 计价与调度政策的因果贡献，需要下一轮分项对照。
图级诊断原始文件在 `last-logits/profile-diagnostic/`，两端冻结二进制与源码身份沿用
16.7 的 baseline/M32 候选，插桩运行的整组时间约 2002.44→1363.56 ms 只作辅助观察。


### 16.9 MTP 外部 hidden 上传合并：数值通过，服务收益不足

在干净基线 40f3b76 与 detached 候选 a227677 上，将逐 lane 的 host upload 改为一次连续
upload，再在同一 stream 上绑定各 lane。回归覆盖连续 priming、KV/rollback、草稿 offset、
无效输入与不同宽度；目标 hidden/logits/KV 的逐位检查通过。CUDA 默认测试 28 passed，
生产 CUDA Clippy 通过。候选未进入默认路径。

27B/MTP2 四场景运行顺序为 A1/B1/B2/A2，每个进程排除一次热身、测量三次。第一对在
构建期间相隔 418.344 s，只能作漂移诊断；第二对间隔 8.640 s，其 TTFT 变化为 short
+2.52%、long -0.02%、batch4 -0.64%、hot_long +1.20%，TPOT 变化约 -0.04% 到 -0.46%。
所有 21 个测量请求的 greedy token 在两对中均一致；报告没有 MTP accepted/proposed 计数。
这些数据不支持默认采纳，也未满足至少五个独立 fresh-process 配对单位的签发要求。

原始数据位于 45 上
`/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/mtp-upload/`，摘要为
`summary.json`；冻结基线 SHA256 为
`cfd3e17af78383f40adb5bf698ad0934e226c6c5e952ff371ac68f38ca18b02d`，
候选为 `03213597b7f6618c0828f84fa2f26d732d5113862f39e39a368c223d6dd46e8c`。

### 16.10 cuBLASLt raw FP8 与原 NVFP4 布局：真实计算可行，服务收益未签发

复查旧探针发现其只测试 mode 3/4/1/2，遗漏 mode 0（SCALAR_32F）。保持原 FP8 E4M3
bytes，使用 scalar 1 做 F32 raw GEMM，再按原顺序执行
`(acc * activation_scale[row]) * weight_scale[column]`，不改变 activation quantizer 或
checkpoint scale。NVFP4 的 packed E2M1 bytes 和 block16 E4M3 scales 也能保持数值编码，
仅对 scale 进行可逆的布局重排。它们不适用于当前 FP8 block128 路径。

CUDA 13.4 / cuBLASLt 130800、验证卡 sm120 上，typed-header heuristic 探针的 FP8
F32 输出 28/28、NVFP4 F32 输出 14/14 找到算法；随后独立 CUDA 程序的 32 个真实 matmul
用例均通过数值 guard。小用例对全部输出使用 F64 oracle；大用例只对固定 35–37 个输出做
oracle，并对全部输出检查 finite，不能冒充大矩阵逐元素证明。FP4 GPU scale swizzle 与 CPU
映射字节相同，抽样值匹配；FP8 保持两次分别舍入的 scale 乘法。合成输入的通过不等于
真实 checkpoint 或整模型质量验收。

CUDA graph 计时包含 FP8 epilogue / NVFP4 activation-scale swizzle，尚不包含原 activation
quantizer。cold 每次在计时前刷掉约 4 倍设备 L2；30 次 replay 是同一进程内测量。
例如 M64 的 N/K=17408/5120、5120/17408、5120/5120，cold 总时间 FP8 为
69.632/73.728/28.672 us，NVFP4 为 38.912/38.944/16.384 us。
这些是 vendor 单路结果，不能作为 native GEMM 或服务加速比。

因此重新打开委托调查，但默认 dispatch 保持不变。下一门禁需包含原 quantizer 的同输入
native/vendor 对照、真实 checkpoint 输出，以及独立服务配对；若算术顺序改变，须声明
新的数值 profile 并按预先确定的质量门禁验收。生产 dispatch 应按设备能力/shape/可用算法
选择和回退，不能绑定某个显卡型号。

证据为同目录 `lt-heuristic.jsonl`、`lt-matmul.jsonl` 和对应日志。
真实计算探针源码 SHA256：
`582c0199e29eb7ba2152f2dd6643d9c0a476bdf9e05e0e266008c740bf874f20`。
模式、默认 scalar 与布局规范参见
[cuBLAS 13.4](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulmatrixscale-t)。

### 16.11 GDN 编译优化等级排查与资源准备指标语义

保留 legacy kernel 的 O3，chunk 路分别用 opt_level 0/1/2/3 编译。在独立工作树中运行
`chunk_delta_opt_level_sweep`：每级覆盖 16 个 case（零计数、tail/full、offset、split 与
跨 chunk 持续状态），四个等级各汇总 2,829,844 个不一致 F32 元素，所有已记录 case 指标不因等级变化；日志没有完整 tensor hash，不能断言跨级完整数组
逐位相同。差异计数累加各快照，空 chunk 可能再次计入已有 state 漂移。诊断测试总体通过只表示扫描完成，不能视为 exact 数值门禁通过。编译优化等级包含在
cuTile 的缓存身份中；没有把四组运行混为同一缓存。原始数据在同根目录
`gdn-opt-level/sweep.log`，补丁指纹
`1cdee5a3c1b4b32979627ffb4a9119e693f6e918326eedf54bd43bb588a5c3b3`。

这进一步否定“仅降编译等级即可修复”的方向，未证明差异来自 FMA。Tile IR 的浮点归约
允许实现选择求值顺序；下一诊断需要逐阶段定位 norm、prediction、update 和 readout，
再用未插桩结果验证诊断没有改变被测行为。
[Tile IR reduce 规范](https://docs.nvidia.com/cuda/tile-ir/13.4/sections/operations.html#cuda-tile-reduce)

另复核 `metrics.resource_preparation`：其 wall_time 从请求首次 park 到最终
ResourceAcknowledged，包含容量拒绝重试、等前面请求释放资源、控制队列和 poll 延迟，
不等于 CUDA backend 的实际 prepare 时间。batch4 的八次 count 通常来自四个 Reserve 与
四个 Prefix 最终 ack，失败重试不另产生最终 ack。因此 2.1–2.3 s 的合计值支持存在入场/
资源等待，但不能直接判定 GPU arena 捕获耗时为 2.1–2.3 s。40f3b76 与 3a188f4 的
这一指标实现相同。

默认 256 宽与显式 64 宽的同二进制 fresh-process 诊断已完成，但仅一个独立对照单位。
64 宽让 batch4 TTFT 中位数 593.25→171.93 ms，同时 TPOT 14.61→16.32 ms，整组时间
2122.92→2098.78 ms；请求从两早两迟变为三早一迟。long TTFT 365.02→396.08 ms。
hot_long 的 prefix reuse 从 3584→1536，TTFT 74.61→2163.24 ms，实际重算工作量不等，
不能把它签发为相同热前缀工作量的 kernel 加速对照。该联合策略改变 graph/chunk/runtime
token budget，当前不适合全局默认。原始数据与 argv 在
`mtp-upload/stage64-plan/`，source 为干净 40f3b76，二进制沿用 16.9 基线。


### 16.12 第一性原理复核：容量、执行路径与长输入（2026-10-11）

此前 D83→0dd 的五个 fresh-process AB/BA 配对没有证明稳定的整体提升。停止扩大
large-GEMM 参数搜索，先用真实 CUDA trace、容量计价和同输入 vLLM 参考定位差距。
以下新实验都是诊断，不构成五配对正式验收，也不宣称已经逼近 vLLM。

**decode 的限制不只是 GEMM。** 0dd 的短请求 Nsight 窗口中，CUDA kernel/copy 活跃
覆盖约 82.4%，三类矩阵 kernel 总时长约占窗口 58.4%。按每轮权重只读取一次估算，
主干 FP8/NVFP4 约达到 1.46/1.50 TB/s；这是流量模型，不是实测 DRAM bandwidth，
不能推导所有 GEMM 或 prefill 都接近极限。96 个 `[48,5120]` BF16 小投影仅占主干
线性权重约 0.25%，却占主干 kernel 时间约 8.3%，每次仅两个 CTA，值得单独优化。
单活跃请求的验证仍复制全部四个物理槽位的两份 checkpoint：27 轮产生约
33.9GB D2D payload、39.4ms copy 时间。`cuStreamSynchronize` 大部分时间与 GPU 活跃
重叠，不能把整个同步 API 时间算作 CPU 开销。

trace 的 batch 尾部报告 CUDA events 丢失，不能把未记录区间当作 GPU 空闲。
证据：`first-principles/nsys-0dd5899/critical-path-analysis.json`、原始 nsys/sqlite。
根目录均为 `/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/`。

**容量候选验证了权重复用方向，也暴露延迟取舍。** 在 detached 0dd 工作树恢复
安全 terminal-head 末行计算，保留全部 hidden 与原量化/MMA 顺序，并按构建策略统一预算报价与
graph widths；尚未覆盖重用 private 宽图时的实际保留计价。候选不改变 1GiB 设备余量或 graph slack。
44 个 CPU 测试通过；合成 tail 测试通过；真实 27B、511-token prompt 的全部 hidden、
末行 logits、verification/rollback 和后续 decode 逐位一致。

同二进制输入、冻结调参缓存、MTP2、64 输出 token 的两个独立 AB/BA 配对：

| batch4 指标 | pair0 | pair1 |
| --- | ---: | ---: |
| 整组完成时间变化 | -35.94% | -31.11% |
| 请求完成时间中位数变化 | -19.54% | -19.01% |
| TTFT 中位数变化 | -64.80% | -65.62% |
| TPOT 中位数变化 | +17.68% | +16.90% |

全部 112 个请求输出 token/长度/结束原因及工作量控制相同；hot reuse 均为 3584。
当前基线每批两早两迟，候选四条更早驻留；早首字请求的第1→2 token 间隔约
75→161ms，符合其他 prefill 阻塞的假设，但报告不足以精确分摊 GPU/host 各阶段。
固定 poolwidth 不等于固定活跃工作量，不能据此直接称 GEMM 回归。
旧轮基线三早一迟与当前两早两迟的具体容量边界原因尚未观测；实际配置、程序、
输入与缓存一致，启动阶段显存观测差约 162MiB，缺少 reserve 时 pool used/reserved
与 driver-free 数据。不能把跨时段驻留数当作同一稳定基线。

候选尚未合入；prompt-head 的 hardware tuning 覆盖也需要补齐。当前保留原 prompt
native baseline，不能宣称 compact 实际 shape 的最优参数已自动选择。
证据：`first-principles/compact-capacity-validation/identity.json`（source tree
`7e946da9e5912767f554e5f7e0f7ac647448f5b1`）及
`first-principles/compact-capacity-paired/{plan,diagnostic-summary}.json`。

**vLLM 的真实差距已重新测量。** 0.31.0/MTP2 在 65536/.88 配置启动失败：需要
2.75GiB KV、只有 2.56GiB。另以 32768 对齐 native 实际平台容量运行，保留 native CLI
声明 65536 的差异。所有实际输入不变，最大 prompt 3647、输出 64。
`--prefix-match-unit 32` 保留默认 MTP 尾部重算规则，三个 hot trial 的实际复用均为
3584，未禁用安全 block-drop。以下是一次 fresh vLLM 服务与两个 native 配对的诊断
中位数，不是跨引擎质量或正式性能验收：

| 场景 | 0dd TTFT / TPOT(ms) | vLLM TTFT / TPOT(ms) |
| --- | ---: | ---: |
| short | 51.93 / 12.23 | 33.21 / 7.98 |
| long(511 tokens) | 361.56 / 11.87 | 85.70 / 7.99 |
| hot_long(复用3584) | 78.55 / 11.16 | 31.59 / 7.71 |

batch4 整组耗时约 0dd 2.180s、compact 1.448s、vLLM 0.645s，仍有明显差距。
vLLM short 每轮 accepted 1.462，native 已完成 trace 为 1.449；batch 为 1.260 对
1.238。native 未记录 proposed，不能把 accepted/2 叫接受率；这些不同服务的数字
只说明 MTP 已正常工作，不支持把首要方向改成新的推测算法。
证据：`first-principles/vllm-mtp2-cap32768/`、`reference-diagnostic-summary.json`。

**后续收敛在执行路径。** 容量/混合调度改善整体吞吐；长输入的 GDN 与低效预填充
解释不了的部分需要真实分段测量；减少无效 checkpoint 与 GPU→CPU logits 搬运/扫描
可改善 decode。默认 GDN 对每 token 读写完整 state，511 tokens 的逻辑访问约154GB、
约2.46万次调用，但 node-major 同层工作集可能命中 L2，不能称全是 DRAM 流量。
已有寄存器 chunk 方案约快2.5倍，却未通过 exact 门禁；O0–O3 扫描未解决差异，FMA
原因未被证明。下一步定位首个不同计算阶段，不重复优化等级扫描。

同 compact 程序、相同公开声明的prefill_width256的短场景调度诊断（当时未采样live capture宽度），仅改 token budget256→64：
batch4 TTFT 中位数 218.18→172.15ms，整组 1416.45→1412.47ms，最大 TTFT
221.90→292.71ms，TPOT 17.38→17.86ms；全部输出一致。它只转移延迟，不适合
直接设成全局默认，更不能推广到 long/hot。证据：`first-principles/selector-diagnostic/`。
硬件相关选择继续归属现有 tuning 与能力检测，验证卡型号不能进入 dispatch 决策。


### 16.13 融合 pooled checkpoint 与真实 prompt 图宽（2026-10-11）

**checkpoint 已形成可落地收益。** 在原 Conv/Delta 的 state writer 内，把已经计算的
新状态同时写入回滚快照，消除另一遍 state read 与独立 D2D copy。仅 pooled verify
需要快照的候选使用融合核；最后一个物理候选仍用原核，inactive lane 跳过两种 state
store。checkpoint shape、restore memcpy、接受/回滚与 graph/resource 销毁顺序保持。
原 `recurrent.rs` 保留为独立数值参考。没有增加设备型号判断或硬件参数：Conv 与 Delta
沿用原 tile 与 `value_block`，已有 GEMM tuning 继续生效。

生产提交 `3a149e3019a42e75a42850c36b4b26e3905702f3` 与测试冻结源码树
`686b805f87f78d8c5d11f21741626296b1c66144` 完全对应。冻结 CLI SHA256 为
`a29b423fa1ea7557d5c3e47220978c664358263e1c41c1dec831ac53c371d5e6`；基线沿用
0dd 二进制，其与4e基底仅文档不同。43 CPU 测试与4个显式 GPU 回归通过；新测试
比较全部 output/state/checkpoint 的 finite 与 `to_bits`，覆盖真实16/48/128几何、
split1/4、非零状态、tail、inactive、部分接受及 slot rebind 后继续验证。
格式、policy/layout/inventory 和生产 CUDA Clippy 通过。额外启用 CUDA test Clippy
仍有24个旧测试 lint 错误，本次新增代码的错误已修复；不能称完整 `check-cuda` 通过。

27B NVFP4、MTP2、64输出 token，共6个独立 AB/BA 配对、12个新服务进程，336个
含 warmup 的请求输出、输入、长度和结束原因均一致。实际 hot reuse 都为3584；
程序、缓存、输入及运行环境身份完整校验。每个服务内3次测量只是一组独立单位。

pair2 必须保留为混杂组：候选启动前新增约295MiB的 lockscreen 图形进程，加载日志
device/state budget 从27.7/4.8GiB变成27.4/4.5GiB。batch4 整组中位数
2159.89→3346.04ms（+54.92%），max TTFT +105.29%，不能用 TPOT 改善掩盖。
遥测 Scheduled/Submitted 从约1.9 work/step变成1.0；后续 lockscreen 保持时，两边
都为1 work/step且更慢。缺少具体 Capacity 拒绝的 needed/free 记录，不能声称已证明
唯一根因。未停止任何用户/桌面进程。

在额外 pair5 之前声明并一致应用资格规则：对照两边的加载 device/state budget 显示
与外部桌面 GPU 进程集合。0/1/3/4/5 共五个匹配配对，以下是独立 pair 变化的中位数，
负值为耗时降低。每场景的 TPOT 和整组耗时五对均改善；不能称统计显著性证明：

| 场景 | TPOT 变化 | 整组耗时变化 | TTFT 变化 |
| --- | ---: | ---: | ---: |
| short | -8.43% | -7.63% | +1.43% |
| long | -8.19% | -5.74% | -0.16% |
| batch4 | -6.94% | -7.46% | -6.65% |
| hot_long | -7.59% | -6.98% | -3.72% |

batch4 max TTFT 的 pair 变化中位数为-7.24%，五对均改善；TPOT 改善范围5.89–10.90%，
整组4.50–7.85%。short TTFT变化范围-6.31%到+4.21%，所以没有稳定的 prefill 提升。
预算日志只保留0.1GiB显示精度，进程集合只在采样时观测，匹配不证明精确byte或全过程
背景相同。`aggregate-all6.json` 保留全部六对，包括异常组；条件匹配摘要与规则在
`background-qualification.json`/`background-qualification-plan.json`。完整身份、GPU
回归与原报告位于同根目录 `first-principles/checkpoint-fused-{validation,paired}/`。
这只覆盖该设备/模型/四个有限场景，没有补齐27B MTP0、2B、持续/混合负载或跨引擎
质量矩阵，不构成 vLLM 接近目标完成证明。

**长 prompt 的实际执行路径已经观测。** 冻结0dd、511-token long 请求的节点事件
诊断中，target 实际为7×64+63，draft为3×128+127，二者分阶段交错但每段 draft
只消费已完成的 target 前缀。三次测量 target graph 时间中位数329.01ms，其中
Linear160.07ms、Delta115.44ms、Attention20.11ms、Conv4.67ms；draft另28.19ms。
事件插桩会改变 wall_time，以上只作节点归因，不能替换正式约362ms的TTFT。
证据在 `first-principles/prefill-node-0dd5899/` 的 `nodes.jsonl`、`node-analysis.json`
和 `width-cause.json`。

生成请求是 Prefill role，scheduler上限256，`forward_chunk_tokens=64` 并非此处根因。
pooled `graph_policy` 为减少每序列 arena 成本，只捕获narrow64；`Sequence::run` 按
实际 resident `prefill_width()` 再分块。LoadedModel的公开256不是 live capture 宽度。
同一原二进制可用显式 `--max-num-batched-tokens 256` 诊断：关闭自动宽度选择后
narrow=0，pooled会保留wide256。这只用于long因果测试，未改变默认；后续自动选择
需要进入已有 tuning，并同时按实际保留图计预算、验证短/混合请求尾延迟。

compact7e 的 `graph_policy` 未改，且 private shell 转 pooled 时可能仍保留旧wide图；
其报价按desired policy计算，存在未观测的 underquote 路径。候选未合入，后续必须按
actual retained captures计价或安全丢弃未报价图。已有 `chunk_and_per_lane_delta_agree_exactly`
实际仍使用1e-7容差，不能拿它作为默认exact准入；寄存器chunk需先定位首个不同计算
阶段，再用完整 state/output 位模式验证。继续扩大低层 GEMM tile 搜索不足以解决长输入差距。


显式256的单次节点诊断已执行，真实target为256+255，draft仍128+128+128+127。
warmup与3次测量的全部256个输出token、长度、EOS及结束原因均与原自动宽度参考相同。
原runner返回1，因为比较错误地包含了`finished.measurement`的计时字段；服务返回0，
实际环境及缓存检查通过。未重写原success状态或重跑覆盖证据，独立稳定字段复核在
`first-principles/prefill-node-explicit256-0dd5899/{cpu-qualification,width-validation}.json`。
带事件记录的target节点中位数329.01→237.22ms，Linear160.07→83.79ms，Delta
115.44→114.54ms，draft28.19→30.45ms。它证实少读取几轮权重的方向值得跟进，
但仅两个不同时段的插桩服务，不能当作正式端到端速度验收。尚未改变默认图宽，仍需
无插桩独立配对、完整数值门禁与短/热/混合负载，并将自动选择接入现有硬件tuning。


### 16.14 图宽的无插桩反例与真实模型数值门禁（2026-10-11）

**全局扩大到256不是默认候选。** 使用16.13已合入的同一融合CLI、相同冻结缓存和
输入，完成AB与BA两个fresh-process对照。baseline为`--no-autotune`；candidate再向
原生CLI传`--max-num-batched-tokens 256`，关闭自动选择。两臂公开prefill_width仍为
256；根据已验证源码路径，pooled实际应分别捕获64与256。这次无插桩服务没有新增
live capture观测，不能把公开配置读回冒充实时图宽测量。

每臂28请求（7次warmup、21次测量），共112请求的全部token、长度、EOS、结束原因
及工作量控制一致，hot reuse为3584。每服务内3次重复不是3个独立配对单位。测试前
冻结资格规则：采样时的外部GPU进程集合相同，加载device/model/state预算0.1GiB显示
区间相同。pair0的state预算4.7/4.8不匹配，原数据保留作混杂诊断；pair1匹配，正式
合格配对仅1/5，不足以签发收益。以下负值表示耗时降低：

| 场景 | pair0 TTFT / 整组变化 | pair1 TTFT / 整组变化 |
| --- | ---: | ---: |
| short | +82.21% / +3.89% | +93.19% / +5.79% |
| long(511) | -19.47% / -7.72% | -18.64% / -7.82% |
| batch4 | +154.10% / +74.05% | +139.82% / +71.90% |
| hot_long | +58.42% / +6.31% | +52.57% / +3.28% |

两对batch4中baseline都为Scheduled353/Submitted188（每step约1.878请求），candidate
为353/353（1.000）。共享slot pool宽度仍为4；这是实际合并工作减少的观察，不证明
pool变成1，也不证明具体admission拒绝原因。资源准备合计包含等待与重试，不可当作
后端capture的实际CPU时间。全局256的反例重复出现，因此停止为它补齐推广测试；
不能用long改善掩盖short/hot及batch4回退。以后新默认候选的收益仍需至少五个合格
独立配对和混合负载门禁。

仅target完整prompt arena从64宽94,789,632 bytes增至256宽379,158,528 bytes，每个
resident增加284,368,896 bytes。当前按配置宽度报价仍覆盖自动64，因此不是已经证实
的低报；在compact按实际捕获降低报价后，复用旧private宽图却未重建将形成低报风险。
后续应同时处理终端head的单行存储、窄/宽图选择、完整保留资源预算及缓存几何匹配。
图宽与kernel参数均归属已有typed tuning，按设备能力、shape、数值合同和实际驻留需求
选择。当前workspace跳过prompt head rows>=32的测量，不能宣称compact head已经复用
M64/M256的measured Choice，更不能查询M1缓存替代其合同。

原始报告、环境、计划与资格规则在`first-principles/prefill-width-fused-paired/`；新增
`diagnostic-summary.json` SHA256为
`824b5cb1042af4d27512f84f6a5e0baf259a93c0664e25a78e74c035433deeef`。
CLI仍为16.13的a29b二进制，未改变默认或生产算术。

**64/256在该真实输入上通过完整数值门禁。** 新增ignored CUDA回归，共享一次27B
NVFP4权重上传，但分别分配状态、KV与arena，捕获真实64/256 prompt图，容量1024，
verification宽3、target MTP0、autotune与chunked recurrent均关闭。511-token冻结输入
在256与511共同边界比较每个有效hidden行、末行完整vocab logits，以及所有F32状态和
完整FP8 K/V数组（包含未使用尾部）。随后比较3行verification的全部hidden/logits及
状态，commit1后的全状态，以及8步固定同输入decode的hidden/logits/状态。

累计879,233,536次元素比较、3,223字段；nonfinite、逐位差异与max absolute均为0。
这里包含多次状态快照，不是8.79亿个互不重复元素。冻结缓存原样保留。prepared GEMM
配置按两臂记录，但明确不是replay trace；新读取接口全部`cfg(test)`，生产行为不变。
门禁只证明这一个冻结输入及target配置，未覆盖draft/pooled完整MTP、所有prompt长度、
其他设备或跨引擎质量，也不替代compact head的独立数值对照。

冻结测试源码树为`7c7180c0a74afaf5c43dda6e7fdc601424e2ea97`，测试二进制SHA256为
`376e2694a73a011af9afe738e04febf684167f6354b3a1632011bc54c3e93195`。证据在
`first-principles/prefill-width-numeric-validation-42f4ea2/`的source/build/binary与GPU日志。
第一次复制制品缺执行权限，exit126且未启动模型，保留原失败；修正权限后第二次实际
GPU运行exit0。43个默认CPU测试通过、52个GPU/环境测试ignored；格式、policy、layout、
inventory及生产CUDA Clippy通过。CUDA test Clippy仍在相同旧位置报24个既有lint，
新测试没有新增错误；这不等于完整check-cuda通过。


### 16.15 GDN算法路径复核：逐token递推与代数分块（2026-10-11）

从45上冻结成功服务日志及安装源码复核：vLLM 0.31.0的日志第34行明确选择FlashInfer
GDN prefill（FlashInfer 0.7.0.post1）。调用链为Qwen GDN attention的真实prefill入口，
最终`chunk_gated_delta_rule(..., backend="flashinfer")`，未显式传use_cp。
这证实高级backend选择，没有kernel trace证明每次最终CP或非CP分派。

原生量化checkpoint默认在recurrent_prefill workspace构建时返回empty，capture随后
对每lane执行原`recurrent::delta`，每token读写F32状态。可选carry-chunk把状态留在一个
kernel内再逐token递推，减少launch/state访存，但没有消除块内token时间轴上的递推。
FlashInfer非CP SM120 kernel在64-token块内计算KKT、三角inverse、QKT与MMA，块间
仍carry。CP另做各块T/M/N预计算、较粗的串行边界fixup及各块输出；并非完整并行scan。
这是算法层面的区别，不能用继续扫描GEMM tile或cuTile编译优化等级来替代。

API的use_cp默认auto。按已记录验证设备170SM及模型48状态heads，单序列满足源码中
GDDR SM12的`parallel_work * 3 < SM_count`示例条件，两序列起不满足。这只是条件推断，
并非运行分派证据；HTTP batch4也不等于单次kernel有4条序列。原生后续选择仍应走
设备能力与现有tuning，不复制设备名称分类或绑定验证卡型号。

从实数代数看，Delta递推可写成对上一状态的仿射变换，块内组合后允许矩阵运算并行。
浮点结合顺序改变则不保证逐位相同。vLLM的QKV/输出为BF16，FlashInfer inverse使用F16，
累积/状态默认F32；原生默认为F32 QKV/输出/状态。因此FlashInfer不是当前exact合同下
可直接启用的候选。下一项算法实验须事先选定数值profile与真实任务质量门禁，同时保留
完整状态/输出误差和长前缀继续decode证据；同token只能证明所测greedy输出，不能单独
证明任务质量。未通过exact或任务质量门禁前，不更改默认算术。
现有compare工具的quality.available固定为false，27B profile的quality只是占位；旧pilot
仅4条GSM8K有答案标签且没有评分/阈值，不能作为新算术的任务质量准入证据。

本次只有CPU源码阅读，没有新GPU trace、耗时或算法加速结论。完整证据（18份日志、
版本与源码的SHA256及原文行）在
`first-principles/gdn-vllm-source-review.json`，SHA256为
`59140ad1f32afe58db4d8e5b3b66a7c9048dc9fe3aea532d757de3a5aa3e318a`。
所引7份native源码逐字节核对已提交42f4ea2，不引用主工作区的其他未提交修改。
