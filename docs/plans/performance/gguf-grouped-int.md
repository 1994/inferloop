# GGUF grouped-int 性能：问题记录与修复方案

状态：P1、P2 已修复；P3 前两步（word-load GEMV、tile 按行拆给多个 CTA）已修复；P4 第一步（5/6-bit nibble + 高位平面）、P6 第一步（行块 GEMV）已实现；P3 余项、P4 余项、P5、P6 余项为方案。格式与 kernel 家族的结构见
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

## P3 第一步：word-load GEMV

nsys（`--cuda-graph-trace=node`）分解 Q4_0 单请求 decode：GEMV 占每 token 2.46 ms（gate/up 0.93、
q/o/down 0.63、lm_head 0.59、k/v 0.31），其余 kernel 合计约 0.5 ms。`cuda-gguf-gemv` 原来重复同一份权重、
命中 96 MB L2，测到的时间比 serving 短（gate 10.5 vs 16.6 µs）；现在轮换足够多的权重副本（≥ 256 MB），
每次 launch 都从 DRAM 读，与 serving 一致。

逐项消融（冷 L2，Q4_0 gate）：只 load+convert+reduce 的下限就有 10.2 µs（≈ 620 GB/s），把 code 当
`u32` 读后降到 6.0 µs —— u8 tile 的 load 本身是瓶颈。占用率提示、每步 reduce 改为最后一次 reduce、
split-K 都只有小幅收益；先按组求和再乘 scale（中间轴 reduce）、用 `extract` 拆激活在 cuTile 里反而更慢。

修复：两种 packing 的 GEMV 都把 code 读成 `u32`；字节位 `b` 各用一个累加器（行 `4w + b`），code 通过
`2^23` 尾数技巧精确转成浮点，不做整数转换；K 循环结束后才 reduce；每步 2 个 span（span 数为奇数时 1 个）；
nibble 的 bias 格式按组减 `bias · Σx`，不逐元素减。数值仍是 F32 GEMV，8 种格式对 F32 参考的最大相对误差
≤ 2.4e-5。

kernel（冷 L2，µs，旧 → 新）：

| 格式 | gate 6144×2048 | k 1024×2048 | q 2048×2048 | down 2048×6144 | lm_head 151936×2048 |
|---|---:|---:|---:|---:|---:|
| Q4_0 | 16.4 → 12.1 | 6.1 → 4.3 | 6.3 → 4.9 | 11.6 → 8.8 | 600 → 261（Q6_K） |
| Q4_K_M | 17.1 → 14.5 | 6.3 → 5.0 | 6.6 → 5.7 | 20.7 → 12.9（Q6_K） | 601 → 261 |
| Q5_K_M | 28.2 → 16.9 | 10.0 → 5.2 | 10.1 → 6.6 | 20.7 → 13.1 | 600 → 261 |
| Q6_K | 32.7 → 15.8 | 11.7 → 4.9 | 12.0 → 6.1 | 20.7 → 13.1 | 596 → 262 |
| Q8_0 | 31.8 → 15.7 | 11.3 → 4.8 | 11.4 → 5.8 | 24.0 → 12.2 | 580 → 262 |

lm_head 现在 ≈ 1.5 TB/s；层内 GEMV 仍只有 0.6–0.9 TB/s。

serving（同一环境与脚本，32 token prompt 的 TPOT，`gemv.jsonl`）：

| 格式 | 上一步 | 本步 | llama.cpp |
|---|---:|---:|---:|
| BF16 | 3.30 ms | 3.35 ms | 3.20 ms |
| Q4_0 | 3.20 ms | 2.47 ms | 1.78 ms |
| Q4_K_M | 3.58 ms | 2.74 ms | 1.89 ms |
| Q5_K_M | 4.75 ms | 2.94 ms | 1.98 ms |
| Q6_K | 5.14 ms | 2.82 ms | 2.09 ms |
| Q8_0 | 5.08 ms | 2.78 ms | 2.27 ms |

BF16 路径代码未变，+1.5% 在单轮噪声内（上一步 A/B 的同格式波动 ±4%）。prompt 时延不变（P5）。

## P3 第二步：tile 按行拆给多个 CTA

16 行 tile 的每个字节串是 4 个 code word（各 4 行），所以一个 CTA 可以只取其中 `W` 个 word，
不需要跨 CTA reduce。GEMV 改为用 2-D 视图读 code 和 term：K ≤ 4096 时 `W = 2`（64/128/384 tile 的投影
grid 翻倍），更长的 K 上 nibble 保持整 tile；byte 在长 K 上每步 4 个 span。两种 packing 都加两 CTA/SM 的
占用率提示；nibble bias 格式按组累加 `bias · Σx`，最后减一次。

kernel（冷 L2，µs，第一步 → 第二步）：

| 格式 | gate | k | q | down | lm_head |
|---|---:|---:|---:|---:|---:|
| Q4_0 | 12.1 → 10.1 | 4.3 → 3.3 | 4.9 → 4.7 | 8.8 → 9.7 | 261 → 247 |
| Q4_K_M | 14.5 → 10.9 | 5.0 → 3.5 | 5.7 → 5.0 | 12.9 → 13.8 | 261 → 246 |
| Q5_K_M | 16.9 → 16.2 | 5.2 → 4.2 | 6.6 → 6.6 | 13.1 → 13.7 | 261 → 247 |
| Q6_K | 15.8 → 15.0 | 4.9 → 3.9 | 6.1 → 6.1 | 13.1 → 13.7 | 262 → 246 |
| Q8_0 | 15.7 → 14.0 | 4.8 → 3.8 | 5.8 → 6.0 | 12.2 → 13.0 | 262 → 225 |

6144 列的 down 在所有格式上慢 0.6–1.1 µs（2-D 视图下整 tile 比原 1-D 读法慢）；每层合计快 2.7–10.8 µs。

serving TPOT（32 token prompt，`split.jsonl`）：

| 格式 | 第一步 | 第二步 | llama.cpp |
|---|---:|---:|---:|
| BF16 | 3.35 ms | 3.28 ms | 3.20 ms |
| Q4_0 | 2.47 ms | 2.21 ms | 1.78 ms |
| Q4_K_M | 2.74 ms | 2.50 ms | 1.89 ms |
| Q5_K_M | 2.94 ms | 2.85 ms | 1.98 ms |
| Q6_K | 2.82 ms | 2.74 ms | 2.09 ms |
| Q8_0 | 2.78 ms | 2.64 ms | 2.27 ms |

## P4 第一步：5/6-bit nibble + 高位平面

Q5_0/Q5_1/Q5_K/Q6_K 的低 4 位沿用 nibble 平面（与 4-bit 同一布局），其上 1/2 位放进整矩阵 nibble 之后的高位平面
（`CodePacking::NibbleHigh`）：每 span `[32·H, 4]` 个 `u32`，word `(j, w)` 的第 `4H·k + H·b + i` 位是行
`4w + b`、值 `32H·k + j` 的第 `4 + i` 位。GEMV（`high::gemv{,_zero}`）按 byte kernel 的结构读 term，四个行字节
放在最后一维、只用一个 `[VALUES, W, 4]` 累加器，code 为 `nibble | 高位 << 4` 再走 `2^23` 尾数技巧；每步 4 span、
`W = 2`、2 CTA/SM（扫描过 1/2/8 span、`W` 1/4、占用率 1/3/4）。设备字节：Q5_0 9→6、Q5_1/Q5_K 10→7、Q6_K 10→8
bits/值。数值不变（Q5_0 与参考逐位相同，K-quant 最大相对误差与原先同为 1e-6 量级）。

kernel（冷 L2，µs，P3 第二步 → 本步）：

| 张量 | gate | k | q | down | lm_head |
|---|---:|---:|---:|---:|---:|
| Q5_K（Q5_K_M） | 16.0 → 13.3 | 4.3 → 4.4 | 6.6 → 5.8 | — | — |
| Q6_K | 15.1 → 13.1 | 3.9 → 4.4 | 6.1 → 5.9 | 13.7 → 12.8 | 246 → 210 |

1024 行的 k/v 慢 0.1–0.5 µs（64 个 tile，延迟受限，解码指令更多）。

serving（`p4.jsonl`，对照 main `e4daccf7`，同一二进制构建方式；ms）：

| 格式 | TPOT | prompt 513 | prompt 2049 |
|---|---:|---:|---:|
| Q4_0（对照） | 2.19 → 2.21 | 69.0 → 67.7 | 439 → 434 |
| Q4_K_M | 2.54 → 2.42 | 83.2 → 76.8 | 468 → 468 |
| Q5_0 | 2.72 → 2.52 | 72.5 → 85.7 | 451 → 503 |
| Q5_K_M | 2.88 → 2.66 | 82.8 → 88.4 | 491 → 516 |
| Q6_K | 2.76 → 2.62 | 81.7 → 88.2 | 505 → 535 |
| Q8_0（对照） | 2.65 → 2.66 | 84.9 → 85.0 | 464 → 463 |

取舍：TPOT 快 4–8%、权重显存少 20–33%，prompt 慢 5–18%（Q5_0 最多）。rows kernel 每个 32 行块按 span 重解一次
高位平面，解码受限（100 行 Q6_K gate 116 → 127 µs）；试过与 GEMV 同样的 word 解码，GEMM 反而慢 50%，已放弃。
prompt 路径由 P5（int8 MMA、激活块复用）整体重写，届时解码只做一次。

## P6 第一步：行块 GEMV

1–8 行（MTP 验证的 3 行、C2–C8 的批量 decode）原先走 32 行的 BF16 rows GEMM：3 行补成 32 行，每个 CTA 只算
16 个输出，验证图的 linear 耗时是 decode 的 6 倍。现在 `nibble::gemv_rows{,_zero}`、`high::gemv{,_zero}`、
`byte::gemv{,_zero}` 带行块维 `R`：每个 CTA 对 `R` 行只解一次码字，把一个 word 的四个输出行沿最后一维解成一个
`[R, VALUES, W, 4]` 累加器，每行在 F32 下按单行顺序累加（`(code - offset) · factor` 先算，再乘该行激活）。
`R` 取行数的下一个 2 的幂、上限 4（`GROUPED_INT_GEMV_ROWS`），最多 8 行（两个行块），多于 8 行仍走 GEMM；
每步 span 数按 `R` 等比缩小，使每 CTA 的累加器数与单行相同。单行 nibble 保留原先每字节一个累加器的
`gemv{,_zero}`：共享布局在 `Q4_K_M` 单请求 decode 上慢 15%（0.681 → 0.784 s）；byte 共享布局不慢（Q8_0
0.464 → 0.458 s），直接替换。设备测试 `grouped_int_rows` 对 8 种格式 × 1/2/3/4/5/8/9 行与 f64 标量参考比对
（GEMV 相对误差 ≤ 2e-6，9 行 GEMM 按 BF16 容差）。

serving（RTX 5090，`fixed-work-serving.py` 每请求 256 token，每臂两轮新进程交替，中位数；
`/home/r/benchmarks/inference.rs/gguf-mtp-2026-11-21/ab4/`、`ab3/`）：

| 模型 | 场景 | 基线 | 本 PR |
|---|---|---:|---:|
| Qwen3.5-2B `Q4_K_M` | C1 decode | 0.686 s | 0.677 s |
| Qwen3.5-2B `Q4_K_M` | C1 MTP 深度 2 | 1.341 s | 0.790 s |
| Qwen3.5-2B `Q4_K_M` | C4 decode | 3.154 s | 1.672 s |
| Qwen3.5-2B `Q4_K_M` | C8 decode | 2.263 s | 2.254 s |
| Qwen3.5-0.8B `Q8_0` | C1 decode | 0.464 s | 0.458 s |
| Qwen3.5-0.8B `Q8_0` | C4 decode | 1.885 s | 0.921 s |

C4 批量 decode 快 1.9–2.0 倍，C1 decode 与 C8 不变（C8 的 8 行走两个行块，与 GEMM 打平）。批量行改为 F32
累加后与单行 decode 的数值一致性更好，部分批量请求的 token 摘要因此与基线不同（基线是 BF16 GEMM）。
扫描过行块上限 2 与 4：上限 2 的 MTP 验证略快（0.754 vs 0.783 s），C4 慢（1.742 vs 1.619 s），取 4。

剩余问题：GEMV 本身受指令而非带宽限制（单行 2048 列投影约 600 GB/s），每多一行就多一份完整的乘加，
3 行验证仍约为 1 行 decode 的 3 倍（Q6_K `lm_head` 0.34 → 1.17 ms）。所以 C1 时 MTP 深度 2（0.790 s）
仍慢于不开 MTP（0.677 s），`--num-speculative-tokens auto` 对 grouped-int 目标暂不启用 MTP，显式深度
照常加载。把行维放到最后一维（每线程持有全部行）测过，耗时不变。

## 后续方案

**P3 余项（优先）。** 目标：单请求 TPOT 4-bit ≤ 2.0 ms、5/6/8-bit ≤ 2.5 ms。剩余差距 0.4–0.9 ms：5/6-bit 的设备字节数（P4 第一步后 Q5_K 7、Q6_K 8 bits/值 vs 文件 5.5/6.56）、非 GEMV kernel（norm、rope、attention 的 split/merge，约 0.5 ms，与 BF16 路径共有）以及 kernel 间空隙（约 0.2 ms）。
1. split-K：每个 16 行 tile 的 span 分给 `S` 个 CTA，`S` 由输出 tile 数与 SM 数推导
   （`S = ceil(2 · SMs / tiles)`，上限为 span 数），部分和写到 workspace，由一个 reduce 或 atomic 合并；
   workspace 在 program 构建时按最大 `outputs · S` 预留，不在 decode 中分配。
2. 2048 列的 nibble GEMV 现在每次迭代解 4 span（1024 值），K=2048 时只有 2 次迭代；split-K 后
   每 CTA 1 次迭代，延迟主要来自 load，应把一次迭代的 load 拆成两段以重叠。
3. 验收：`cuda-gguf-gemv` 1 行 ≥ 1.2 TB/s（gate/down 两种形状），serving TPOT 与 llama.cpp 对照，
   并确认 BF16/NVFP4 路径不受影响。

**P4 余项。**
1. term 由 f32 改为 f16：K-quant 的 `d · sc` 在 f16 下有舍入（`d` 是 f16、`sc` 是 6-bit），
   需要先测与 llama.cpp 的一致性；legacy 格式的 `d` 本身就是 f16，无损。所有格式省 1–2 bits/值。
2. 1024 行投影（k/v）：GEMV 在 64 个 tile 上延迟受限，P3 的 split-K 同样适用。
3. 高位平面的 prompt 解码代价随 P5 一起解决。

**P5 长 prompt。** llama.cpp 的 MMQ 把激活量化为 q8_1、用 int8 MMA；我们是 BF16 MMA + 每块重解码。
1. 先测 `[32, 64]` 输出块（一次解 4 个 16 行 tile，复用激活块）是否比 `[32, 16]` 快；
2. 再评估 int8 激活（per-32 块 scale）+ int8 MMA：属于数值契约变化，需单独的精度对照；
3. 验收：2049 token prompt ≤ 150 ms（Q4_K_M），并给出每投影的设备时间分解。

**P6 余项：让验证行摊薄。** 行块 GEMV 只省了重复解码，没省每行的乘加。llama.cpp 的 MMVQ 把激活量化为
q8_1，用 `dp4a`/int8 MMA 一次处理 4 个权重、最多 8 行；对应地：
1. 1–8 行的 int8 激活（每 32 值一个 scale）+ 整数点积 kernel，属于数值契约变化，需与 f64 参考和
   llama.cpp 输出分别对照；
2. 验收：3 行验证 ≤ 1.5 倍单行 decode，C1 MTP 深度 2 快于不开 MTP，届时去掉 `auto` 的 grouped-int 例外；
3. 9–32 行仍走 GEMM，由 P5 一起重写。

**P2 的剩余项。** pooled serving 之前对 grouped-int 每个宽度重试一次才失败；现在已能构造，但
失败路径仍逐宽度重试，与错误码无关。若以后出现非容量错误（`Unsupported`），应直接停止重试。
