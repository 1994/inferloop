# GGUF grouped-int 性能：问题记录与修复方案

状态：P1、P2 已修复（本 PR）；P3–P6 为方案。格式与 kernel 家族的结构见
[grouped-int kernel family](../engineering/grouped-int-kernels.md)。

测量环境：RTX 5090，Qwen3-VL-2B 由 BF16 HF 权重经 `convert_hf_to_gguf.py` + `llama-quantize`
得到各格式；inferloop 与 llama-server（`-ngl 99 -c 4096 -np 1`）用同一 HTTP 接口、同一文件。
脚本与原始数据：`/home/r/benchmarks/inference.rs/gguf-prefill-2026-11-16/`（`latency.py`、
`batch_requests.py`、`collect.py`、`*.jsonl`）。prompt 时延为 `max_tokens=1` 的客户端墙钟中位数
（3 次，预热 1 次，每次唯一前缀，不命中 prefix cache）；TPOT 为 `max_tokens=129` 与 `max_tokens=1`
中位数之差除以 128。

## 基线（main `3301aafc`）

| 格式 | 引擎 | prompt 33 | prompt 513 | prompt 2049 | TPOT |
|---|---|---:|---:|---:|---:|
| BF16 | inferloop | 9.4 ms | 43 ms | 313 ms | 3.29 ms |
| BF16 | llama.cpp | 63 ms | 82 ms | 311 ms | 3.20 ms |
| Q4_0 | inferloop | 443 ms | 746 ms | 3119 ms | 3.24 ms |
| Q4_0 | llama.cpp | 9.3 ms | 26 ms | 89 ms | 1.78 ms |
| Q4_K_M | inferloop | 483 ms | 874 ms | 3635 ms | 3.61 ms |
| Q4_K_M | llama.cpp | 9.0 ms | 28 ms | 92 ms | 1.89 ms |
| Q5_K_M | inferloop | 687 ms | 1643 ms | 6706 ms | 4.74 ms |
| Q5_K_M | llama.cpp | 9.2 ms | 29 ms | 94 ms | 1.98 ms |
| Q6_K | inferloop | 658 ms | 1456 ms | 6398 ms | 4.99 ms |
| Q6_K | llama.cpp | 9.0 ms | 30 ms | 102 ms | 2.09 ms |
| Q8_0 | inferloop | 629 ms | 1977 ms | 6709 ms | 5.08 ms |
| Q8_0 | llama.cpp | 9.6 ms | 28 ms | 92 ms | 2.27 ms |

并发：Q4_K_M 4 / 8 个并发请求（各 128 token）都只有 170 tok/s，即完全串行。

## 问题与根因

| # | 问题 | 根因 |
|---|---|---|
| P1 | prefill 比 BF16 慢 10–20× | prompt 以 32/128/… 行为一块，`grouped_int_rows` 对每行发一次 GEMV，每个投影把整份权重读 `rows` 次 |
| P2 | 每请求固定 ~30 ms，且并发请求串行 | grouped-int 在 `Mode::Slots`/`Mode::Rows` 返回 `Unsupported`，pooled serving 构造失败，退回每请求重建 device program（`slotted=false`，约 20 ms/次）的串行路径；日志里这个原因被泛化成 `Capacity: could not construct pooled CUDA storage` |
| P3 | 单请求 decode：4-bit 不比 BF16 快，5/6/8-bit 比 BF16 慢 45–55%，比 llama.cpp 慢 1.8–2.2× | GEMV 每 CTA 16 输出行、沿 K 串行；2048 输出只有 128 个 CTA，不到 5090 一个 wave，单算子只到 470–600 GB/s（26–34% roofline） |
| P4 | 5/6/8-bit 设备字节数大于文件 | code 一字节一个、term 用 f32：Q5_K 5.5→10 bits/值，Q6_K 6.56→10，Q4_K 4.5→6（见 grouped-int 方案的存储代价表） |
| P5 | 长 prompt 仍是 llama.cpp 的 4–5× | 本 PR 的 rows kernel 是 BF16 MMA，`[32,16]` 输出块，每块重解码 `[256,16]` 权重；2048 行时约 0.6 µs/行/投影，受解码与小 N 块限制 |
| P6 | 2–12 行（并发 decode）时 rows kernel 比满 32 行还慢 | 不足 32 行的块走越界谓词化的 load/store，4 行 75 µs，32 行 31 µs（Q4_0 gate） |

## 本 PR 的修复

**P1 —— rows kernel。** `nibble::{gemm, gemm_zero}`、`byte::{gemm, gemm_zero}`：每个 CTA 负责
`[32 行, 16 输出]`，每个 span 把 code 和 term 解码成一次 `[256, 16]` BF16 权重块，与 `[32, 256]` BF16
激活做 `mmaf`，F32 累加 —— 与 NVFP4/FP8 prompt GEMM 相同的数值契约。布局、term 下标和 packing
都沿用 GEMV 的 tile-major 布局，不增加新的设备布局；行块高度是 const generic `ROWS`，取
`GROUPED_INT_GEMM_ROWS = 32`（64/128 在 128–2048 行上慢 1.4–3×，16 更慢）。

**P2 —— pooled rows。** lowering 让 `Mode::Rows`/`Mode::Slots` 走 `RowsKernel::GroupedInt`；
`grouped_int_rows` 对 1 行仍用 GEMV（单请求 decode 不变），多行用 rows kernel，末块谓词化。
`create_slot_pool` 改为返回最窄宽度的真实错误，日志不再把 `Unsupported` 报成 `Capacity`。

### 结果（同一环境，`infer-new` vs 基线）

kernel（`cuda-gguf-gemv <file> model.layers.0.mlp.gate_proj.weight <rows>`，6144×2048，50 次 replay）：
对 BF16 舍入后的参考，最大相对误差 ≤ 5e-7（8 种格式，rows = 2/3/12/32/33/512）。

| 格式 | 1 行 GEMV | 32 行 | 512 行 | 512 行逐行 GEMV（旧，按 1 行时间估算） |
|---|---:|---:|---:|---:|
| Q4_0 | 15.0 µs | 30.9 µs | 314 µs | ≈ 7.7 ms |
| Q4_K | 15.7 µs | 37.7 µs | 364 µs | ≈ 8.0 ms |
| Q5_K | 30.1 µs | 34.9 µs | 359 µs | ≈ 15.4 ms |
| Q6_K | 26.0 µs | 34.5 µs | 364 µs | ≈ 13.3 ms |
| Q8_0 | 29.5 µs | 37.5 µs | 322 µs | ≈ 15.1 ms |

serving：

| 格式 | prompt 33 | prompt 513 | prompt 2049 | TPOT |
|---|---:|---:|---:|---:|
| BF16 | 9.4 → 8.8 ms | 43 → 43 ms | 313 → 312 ms | 3.29 → 3.30 ms |
| Q4_0 | 443 → 18.4 ms | 746 → 71 ms | 3119 → 423 ms | 3.24 → 3.20 ms |
| Q4_K_M | 483 → 20.4 ms | 874 → 79 ms | 3635 → 452 ms | 3.61 → 3.58 ms |
| Q5_K_M | 687 → 21.6 ms | 1643 → 87 ms | 6706 → 480 ms | 4.74 → 4.75 ms |
| Q6_K | 658 → 21.7 ms | 1456 → 87 ms | 6398 → 494 ms | 4.99 → 5.14 ms |
| Q8_0 | 629 → 19.0 ms | 1977 → 86 ms | 6709 → 452 ms | 5.08 → 5.08 ms |

上表 Q6_K TPOT 的 +3% 是单轮噪声：交替 3 轮复测（32 token prompt，各 5 次中位数）基线
5.30/5.12/5.54 ms，新 5.08/5.11/5.11 ms；Q4_K_M 基线 3.85/3.59/3.57 ms，新 3.60/3.58/3.59 ms。
单请求 decode 现在走 slotted program（1 行仍是 GEMV），没有观察到回退。

并发（Q4_K_M，各 128 token）：4 并发 170 → 335 tok/s，8 并发 170 → 812 tok/s。

数值：
- 单请求 decode 仍是 F32 GEMV；4 并发时 4/4 个输出与串行相同。
- prompt 与多行 decode 改为 BF16 激活 × BF16 权重。8 并发时 4/8 个输出在近似并列的 token 处
  与串行分叉（如人名 “…o”/“…lyn”），文本连贯。
- 与 llama.cpp 的贪心一致性（8 prompt × 64 token，平均共同前缀词数）不降：Q4_K_M 31.5 → 34.0、
  Q4_1 17.5 → 19.8、Q8_0 20.5 → 22.6，其余格式持平；与各自 BF16 的一致性 Q8_0 23.0 → 21.0、
  Q5_0 9.9 → 11.2，其余持平。

## 后续方案

**P3 decode GEMV 占用率（优先）。** 目标：单请求 TPOT 4-bit ≤ 2.0 ms、5/6/8-bit ≤ 2.5 ms。
1. split-K：每个 16 行 tile 的 span 分给 `S` 个 CTA，`S` 由输出 tile 数与 SM 数推导
   （`S = ceil(2 · SMs / tiles)`，上限为 span 数），部分和写到 workspace，由一个 reduce 或 atomic 合并；
   workspace 在 program 构建时按最大 `outputs · S` 预留，不在 decode 中分配。
2. 2048 列的 nibble GEMV 现在每次迭代解 4 span（1024 值），K=2048 时只有 2 次迭代；split-K 后
   每 CTA 1 次迭代，延迟主要来自 load，应把一次迭代的 load 拆成两段以重叠。
3. 验收：`cuda-gguf-gemv` 1 行 ≥ 1.2 TB/s（gate/down 两种形状），serving TPOT 与 llama.cpp 对照，
   并确认 BF16/NVFP4 路径不受影响。

**P4 设备字节数。**
1. term 由 f32 改为 f16：K-quant 的 `d · sc` 在 f16 下有舍入（`d` 是 f16、`sc` 是 6-bit），
   需要先测与 llama.cpp 的一致性；legacy 格式的 `d` 本身就是 f16，无损。所有格式省 1–2 bits/值。
2. 5-bit：nibble 平面 + 1-bit 高位平面（`CodePacking::NibbleHigh`），一个 packer 分支与两个 kernel
   变体；Q6_K 用 nibble + 2-bit 平面。格式层不变。
3. 与 P3 一起做：P3 提高到带宽受限后，字节数才直接决定 TPOT。

**P5 长 prompt。** llama.cpp 的 MMQ 把激活量化为 q8_1、用 int8 MMA；我们是 BF16 MMA + 每块重解码。
1. 先测 `[32, 64]` 输出块（一次解 4 个 16 行 tile，复用激活块）是否比 `[32, 16]` 快；
2. 再评估 int8 激活（per-32 块 scale）+ int8 MMA：属于数值契约变化，需单独的精度对照；
3. 验收：2049 token prompt ≤ 150 ms（Q4_K_M），并给出每投影的设备时间分解。

**P6 小批 rows。** 2–12 行的越界谓词化开销：方案一是 slot arena 的激活按 32 行对齐分配（只读的多余行
为 0，输出多余行不读），kernel 不再走越界路径；方案二是 rows ≤ 4 时逐行 GEMV。先测方案一的显存代价
（每层激活 `32 · hidden · 4 B`，可忽略）。

**P2 的剩余项。** pooled serving 之前对 grouped-int 每个宽度重试一次才失败；现在已能构造，但
失败路径仍逐宽度重试，与错误码无关。若以后出现非容量错误（`Unsupported`），应直接停止重试。
