# vLLM 长上下文性能追平实施方案

状态：实施方案。基于 2026 年 10 月 17 日[长上下文性能交接](../../research/cuda-vllm-performance-handoff-2026-10-17.md)与 vector 分支（性能代码基点 `b43c51d`，树同 `bcc742b`）。

本方案把交接的 C1Q/C1A/C1M/C1V/C1H/C1D 六条子任务展开为可执行的阶段、候选、命令与验收门。执行边界（隔离 checkout、safe-run-v3、全局锁、空闲门、NCU 约定、证据一次性写入）全部沿用交接文档，不重复定义。

## 1. 追平的可核验定义

"追平 vLLM"按[性能基线方案](baseline.md)的合同落地为三条同时成立的判定，不接受单项截图：

1. **性能**：预注册的 32K/64K C1/C4 匹配矩阵上，native/vLLM 配对 95% 成本比上界 ≤ 1.10（沿用 profile 既有 `vllm_max_ratio: 1.1`）;C4 以整组 makespan 与整组实际输出吞吐计，不用单请求吞吐相加。
2. **质量**:1,024 题严格 scorer 下，候选相对当前 HEAD 与固定 vLLM（wheel `0.31.1rc1.dev289+g3fe917abb`）的配对正确率变化，整体单侧 95% Tango score 下界（按源题聚类修正）不低于 suite 声明的非劣界，且每个 variant 的点估计变化在界内（[`score-task-quality.py`](../../guides/cuda-performance.md#完整模型任务质量复算)）。不要求输出逐 token 相同；位精确候选的逐 token 对照只作为数值正确性检查。
3. **无原生回退**：原 20 项性能门（主比较 CI95 上界 ≤0.98，次比较 ≤1.05，臂内漂移 ≤5%,820 对完整流一致）对每一个合入候选成立。

## 2. 差距的定量分解

固定制品：native 冻结构建（SHA256 `730c2348…`）与 vLLM 固定 wheel；负载 32,384 输入 token、256 输出、MTP2、0.88 显存利用率、32,768 窗口；设备 RTX 5090 SM120,170 SM,32 GiB,CUDA 13.4.92。

| 构成 | native 中位数 | 占比 | vLLM 对应路径（已核验） |
|---|---:|---:|---|
| target attention | 18.8841 s | 70.9% | FlashInfer BF16 prefill 张量核 + FP8 KV |
| draft attention | 0.2469 s | 0.9% | — |
| linear | 4.8169 s | 18.1% | Cutlass NVFP4 块缩放 GEMM |
| GDN delta | 1.0158 s | 3.8% | FlashInfer GDN |
| embedding | 0.6153 s | 2.3% | — |
| **GPU 算子合计** | ≈25.6 s | — | — |
| **32K C1 TTFT（未插桩）** | **28.0951 s** | — | **4.6480 s（旧诊断，非配对验收）** |

GPU 算子合计与 TTFT 之间还有 ≈2.5 s 的主机/调度/回读间隙，需 nsys 归因（§5.6)。

模型几何（`/home/r/models/Qwen3.8-27B-NVFP4/config.json`):64 层，`full_attention_interval=4` → **16 个全注意力层**,48 个 GDN 层；注意力 24 q-head / 4 kv-head / head_dim 256；词表 248,320;MTP 1 层。

按此几何，32K 因果 prefill 的注意力有效 FLOPs ≈ 16 层 × 4 × 24 × 256 × (32,384²/2) ≈ **2.06×10¹⁴ FLOP**。当前内核 19.13 s → 有效 ≈10.8 TFLOPS，约为 5090 FP32 CUDA core 峰值（104.8 TFLOPS）的 **10.3%**。NCU（`vector-hardware-counters-v2/review-v1.json`)：指令瓶颈，LSU 54.1%、FMA-heavy 50.8%、DRAM 峰值仅 0.664%;80 寄存器、零 spill、2 CTA/SM。

## 3. 关键算术结论（决定方案形状）

对 SM120 必须先把三条数值路线的账算清（Phase 0 用实测固化，此处为设计依据）:

| 路线 | 5090 峰值级 | 分解开销 | 有效上限量级 | 数值身份 |
|---|---:|---:|---:|---|
| F32 CUDA core（当前合同） | 104.8 TFLOPS | 1× | 40–60 TFLOPS | 位精确可保持 |
| TF32/BF16 分解仿真 FP32(3xTF32 / BF16x9) | 209 / 419 TFLOPS | 3× / 9× | ≈70 / ≈47 TFLOPS 峰值 | 乘积误差 ≤2⁻²¹/2⁻³⁰，但归约重排 → 非位精确 |
| 原生 BF16/FP8 张量核 | 419/838 TFLOPS | 1× | 150–250 TFLOPS | 降低 target 精度，**现行合同禁止**;BF16 实测 95/1024 |

两个推论：

1. **本卡上"不降精度的张量核仿真"对 F32 CUDA core 没有量级优势**(3xTF32 峰值仅 ≈2× FP32，实际效率相抵）。attention 追平 vLLM 的 8–15 倍余量只存在于原生 BF16/FP8——那是合同与质量门双重禁止区（BF16 FA4 95/1024、 tiled 6-product 已否决）。**在现行数值合同下,32K C1 TTFT 的可达下限约 8–10 s（纯位精确优化），与 vLLM 4.65 s 仍差约 2×。** 完全追平需要 §8 的政策决策包，本方案先把合同内收益全部兑现。
2. **H/B 数据中心卡上结论反转**(FP32 ≈67、BF16 ≈989 TFLOPS,15× 比例 > 9× 分解成本）:FP32 仿真张量核 attention 是 C1H 的设计预留，SM120 候选不为它让路，见 §7。

C4(4×32K,makespan 129.46 s vs vLLM 旧诊断吞吐折算 ≈31 s）除同一算子瓶颈外还有结构性差距：超过 `CB_SLOT_TOKENS=8192` 的序列走**私有串行路径**(`execution.rs:426 run_serial`),4 条 32K 各自持有 target+draft 双程序（报价各 3,099,791,228 B)，不进 slot 池合批；vLLM 是连续合批。C1M 先量实际驻留再决定合批形态（§5.4)。

## 4. 阶段总览与依赖

```
Phase 0  C1A-0 可削减成本量化（GPU 窗口,2–3 天）          ┐
Phase 1  C1A-1 位精确结构候选（无质量风险,先行）          │ attention 主线(串行,一次一个候选)
Phase 2  C1A-2 精度保持仿真探针→(条件触发)MMA 候选        ┘
Phase 3  C1L   linear 完整成本（attention 落地后启动）
Phase 4  C1M   真实显存采集→驻留/合批（CPU 审计可即并行,GPU 窗口排队）
Phase 5  C1V   稳定 vLLM 长矩阵（与 Phase 1 并行,用独立 GPU 窗口）
Phase 6  C1Q   质量模式一致性（CPU 归因先行,可即并行）
支撑项   tuning cache 出厂 / embedding 审计 / 主机间隙归因 / 冷启动回退消除
决策点   D0（数值合同政策包,Phase 2 探针证据齐后提交）
收尾     C1D 分支验收合并;C1H 待 H/B 实机
```

串行规则：同一设备的构建与 GPU 测量走原全局锁（`benchmark-safe.lock`),safe-run-v3 `--memory-gib 50` 一次一个重任务；CPU 侧的审计、证明、脚本编写全部并行预置。每个候选一次只改一个可归因策略，按"数值 → 边界 → canary → 资源 → 自动 tuning → 20 项服务门 → 质量 1024"顺序过门，任何一门失败即冻结记录，不返场重测刷过。

## 5. 分阶段实施

### Phase 0 — C1A-0：可削减成本量化（交接强制第一步）

目标：把 attention 19.13 s 拆到指令级来源，给出三条路线的实测 roofline，再选定 Phase 1 候选参数。产出决策记录，不产内核。

1. **PC 级归因（纯 CPU，先做）**：展开 `vector-hardware-counters-v2/run-v1/` 既有 NCU `metrics/source` CSV，把 65.64 亿 FMUL、35.66 亿 FADD、31.98 亿 HADD2.F32 与 LSU 指令映射回 `kernels/vector-attention/attention-sm120.cu` 源码行，按 QK 点积 / softmax（含 `canonical_exp` 多项式）/ PV / staging 装载 / 屏障与循环开销五类汇总占比。已知的源码级嫌疑（逐一量化，不预设主次）:
   - staging 用 **2 字节** `ld.global.u16` 装载 K/V（每线程每块 ≈22 次全局装载）;
   - 同步 staging：每 32-key 块两次 `__syncthreads()`，无 cp.async、无双缓冲，全局延迟不进计算阴影；
   - PV 对 `probabilities`/`shared_values` 的逐标量 shared 读取（每 warp 每块数百次 LDS);
   - 1.5 wave(512 CTA / 340 并发槽）的尾量化与长链串行（每 CTA 顺序走最多 ≈1,024 块）。
2. **roofline fixture(GPU 窗口）**：复用 `vector-hardware-counters-v2` 的 hash 钉死 fixture 模式（`cuModuleLoadDataEx` + 确定性种子输入 + 3 预热 + 逐位复跑），分别测：(a) 现状内核在历史 {0, 8192, 32512} 的每块迭代周期数；(b) 仅移除 staging 同步（计算不变）的上界；(c) 仅向量化装载的上界。得到"位精确饮食"的可兑现上限估计。
3. **决策记录**：新目录 `exact-shared-attention-2026-10-16/vector-pipeline-diet-v0/`（或新研究根，按证据规则一次性写入），固化三类占比、可削减周期预算、Phase 1 候选的目标函数（每主比较 ≥2% 的原门不变，瞄准远大于 2% 的结构性收益）。

出口判据：决策记录给出"staging/发射/读取饮食预计合计 ≥1.5×"或如实记录不足并转向。

### Phase 1 — C1A-1：位精确结构候选（默认主线，无质量风险）

约束：**逐位输出不变**——不改 `canonical_exp`、归约树、shuffle 顺序、`--fmad=false --ftz=false --prec-div=true`、FP8 ±480 码保留、非活跃行精确 0.0、score 缩放 `0.0625f` 的应用点。只改数据搬运与调度。

候选改造点（按 Phase 0 归因排序合并为**一个**可归因候选，不拆成多次刷门）:

1. staging 向量化：K/V 全局装载改为 16B `v4.u32`（每线程每块 ≈3 次），寄存器内 widen 后按现 `key_layout` STS（保留 2 路 STS 冲突与否由数据决定——已否决的 inverse-gather 方案**不复活**，只允许在向量化后顺带消除）;
2. 寄存器/双缓冲流水线：块 N+1 的全局装载在块 N 计算期间发射（预算 80+22 寄存器会跌破 2 CTA/SM，须用 Phase 0 的占用数据在"分两微批装载 / 仅 K 预取 / 降 rows-per-CTA"之间选定，占用门 ≥2 CTA/SM 不变）;
3. PV 读取节食：`probabilities[warp][32]` 每 j 循环重复读改为按块预取寄存器（或调整 i/j 循环序）,`shared_values` 改向量 LDS；消费端数据布局不变，保证 consumer LDS 仍无冲突；
4. 发射与尾块：CTA 到（lane, head) 的映射反序，使最长因果链先进入 wave;`#pragma unroll 1` 循环在寄存器预算允许处放开；
5. 打包 F32 审计：确认 SM120 SASS 的 `FMUL2/FFMA2.F32` 可用性（HADD2.F32 已在指令混合中出现），对归约树中天然成对的独立加/乘显式 `float2` 化——仅在编译器未自动生成且逐位语义可证时采用。

验收序列（全部沿用既有设施，不新建门）:

- **数值**:CPU 穷举布局证明（同 `layout-CPU-proof-v1.json` 模式：双射、对齐、bank 无冲突）→ 36 通用场景 + 9 真实输入逐位（`validate-numerical-v1.py` 模式，含 ±256 B `0xA5` canary 与资源门：寄存器、local=0、shared、占用按新内核单独登记 decision-rules)→ 仓内 `tuning/attention/capture.rs` 对 canonical RustSimt 的 `exact_match` 与 F64 <1e-4。
- **边界**:prompt 63/64/65/127/128/129/255/256/257 扫描 + capacity/lane/position/window 矩阵 + `tests/unit/resident_attention_prefill.rs` 全部既有用例。
- **自动 tuning**：新增 `tuning/attention/<tier>.rs`，历史 {0, 8192, 32512} × capacity 32768，冷/暖 AB/BA,`MIN_CONFIRM_SPEEDUP=1.02`，短历史回退 ≤1.05;`provenance.json` + `verify-artifact.py` 接入 `tools/check/gate.sh`;tuning 表 SCHEMA 更新。
- **服务**：服务资格（原 28 流 + 四条 32K 流一致）→ 五组 AB/BA 20 项门 + 独立复算（`vector-performance-run-v1.py`/`analyze-v1.py` 模式，新证据身份）。
- **质量**:1,024 题完整输出逐 token 相同（位精确候选预期平凡通过，仍完整跑）。

估计（以 Phase 0 实测修正）:attention 19.13 → 9–12 s;32K C1 TTFT → 18–21 s。达不到主比较门则冻结并记录，不进服务门。

### Phase 2 — C1A-2：精度保持仿真探针 →（条件触发）MMA 候选

动机：把 Phase 1 之后仍存的 ≈2× 距离能否在"不降低 target 精度"内继续压缩，变成一个**先廉价证伪**的问题，避免直接建内核赌质量门。

1. **数值探针（GPU 窗口，1–2 天）**：用真实输入回放设施（`gpu-real-attention-capture-v1` 的 9 份张量 + `real-attention-replay-v1` 模式）在主机/设备上仿真三种分解的 attention 输出，不改产品内核：
   - 2xFP16(Q/V 概率按 hi/lo 分裂，K 由 FP8 widen 后在 FP16 中精确，2 个乘积，F32 累加）;
   - 3xTF32;BF16x9（对照）。
   报告：对 canonical F64 oracle 的最大误差分布，并与当前内核自身对 oracle 的误差同图比较（证明"不低于现行精度")。
2. **质量预判（GPU 窗口）**：取探针中误差最小者，用注意力输出扰动注入方式跑 256 题（quality 设施的单 shard，原 scorer/预算不变），统计完整流一致率。判定阈值预先写入计划：**仅当 256/256 一致**才值得建内核（任何不一致都预示 1024 规模必然失败——tiled/BF16 已有证据)。
3. **条件触发的 MMA 候选**：探针通过才实现张量核内核（mma.sync 流水线、cp.async 多级、Q 驻留寄存器、保持 FP8 KV 布局与 metadata 合同），经 `mma` 类准入（F64 ≤1e-2）后走完整服务 + 1,024 质量门。探针失败则整个方向记录否决（与 K 搬运否决同格式），产出转入 §8 决策包。

SM120 上 3xTF32/BF16x9 的有效上限预计与 Phase 1 后的 CUDA core 内核接近（§3)，故本阶段主要价值是**证伪/证实 2xFP16 与政策决策的证据**，不应预期合同内还有第二个大倍数。

### Phase 3 — C1L:linear 完整成本（attention 落地后启动；交接规则：linear 成为主要剩余成本再排）

现状：NVFP4 W4A4(16 元素 E4M3 块缩放）,`resident/nvfp4_gemm.rs`(cuTile 融合量化 + `mmaf_scaled` [16,64])+ `prefill_gemm.rs`(prompt GEMM、FP4 unpack、split-K=4)+ `vendor_fp8.rs`(cuBLASLt)。vLLM 用 Cutlass NVFP4。

1. 先用 `INFER_CUDA_PREFILL_PROFILE` 与 nsys 把 4.82 s 拆到 GEMM 形状（prefill 各宽度 / decode GEMV / MTP 草稿），确认量化、unpack、epilogue 的占比；确认 linear 的数值准入门类（exact 位门还是 profile 误差门——`tuning/resident_gemm.rs` 合同先读清）。
2. 候选（单归因，按归因选定其一）:cuTile 内核 tiling/占用饮食；或 vendor NVFP4 块缩放 GEMM(cuBLASLt/CUTLASS 在 SM120 的 FP4 张量核）。数值走对应门类 + 质量 1024。
3. 验收同 Phase 1 序列。目标 linear → ≤2.2 s。

GDN(3.8%）与 embedding(2.3%）不插队：GDN 仅在 linear 之后仍高于 embedding 才按同模式立候选；embedding 0.62 s 先做归因（32K gather 不应有此量级，疑似与读出/首次 logits 相关），确认构成后再决定。

### Phase 4 — C1M：真实显存与多会话驻留（C4 追平的前提）

1. **编译运行既有采集工具**（交接指定的第一步）:`source-current-memory-components-v1/.../cuda-resident-memory-components.rs`。已确认编译阻塞一行：第 56 行 `query.synchronize()` → `core.synchronize?`(cuda-core 0.4 `Device::synchronize`)；修正后按证据规则**重签 `plan-v1.json` 摘要**，然后 build-v1/run-v1 流程（fmt/layout/policy/test-inventory → clippy `-D warnings` → release 构建 → `…/cuda-resident-memory-components /home/r/models/Qwen3.8-27B-NVFP4 0.88`,50 GiB 保护单元 + 空闲门）。预期只装得下 1 对 target+draft（旧归因显示 2 对下界已超旧预算 232 MB)，如实记录。
2. **拿到四张实际账**:target/draft 实际池增量（对报价 3,099,791,228 B)、base activation、图/capture 的驱动级成本（当前完全未计量）、dataflow/lifetime(`target-dataflow.json`)。
3. **再决定实现**（不许先承诺 4 条或 16 条 32K 驻留）:
   - arena/workspace：按实测 life-time 收敛 `ActivationArena` 计划尺寸与 256 MiB `graph_headroom` 报价的虚耗；
   - C4 合批：当前 >8,192 token 序列走私有串行路径。两条路线按实测选择：(a) 提高 `CB_SLOT_TOKENS`/槽宽使 32K 进 slot 池（需报价兑现）；或 (b) 页级 KV——内核已是 32-token block table 间接寻址（`constants.rs:49` 注释即为此预留），把 Metal 侧已有的 `engine/state/src/kv.rs` 引用计数/COW 机制接到 CUDA，替换身份 block table，使多条 32K 共享批量 decode 图；prefix 快照从整态改为块粒度。
   - 验收：报价兑现证明、状态独立、取消/回滚/prefix 正确，报告实际驻留条数与完整 C4 makespan/整组吞吐（五组配对）。
4. 注意既有成果不重复：闲置不兼容 slot 池回收（≈3.22 GB）已生效；FA4 的 +131 MiB 报价来源已解释，两者都不再立项。

### Phase 5 — C1V：稳定 vLLM 长矩阵（与 Phase 1 并行，独立 GPU 窗口）

1. **先解释旧 C4 漂移 11.13%**（超 5% 门）：候选假设按序排查——GPU 时钟/热（`hardware-monitor.py` 遥测逐 trial 比对）、vLLM 侧 prefix 缓存与 chunked prefill(`max_num_batched_tokens=2048`）在不同进程顺序下的路径差异、MTP 接受率方差、主机 cgroup 邻居。只允许用固定制品与实际配置的证据下结论，禁止为重跑取好漂移值。
2. 实质修复后**预注册**匹配矩阵：32K/64K × C1/C4,vLLM 臂按 adapter-v6 的后端证据针（FlashInfer GDN / Cutlass NVFP4 / FLASHINFER BF16 / XQA FP8 KV）逐项核验；64K 注意旧配置边界已记录在案，不推广。
3. 全部样本、质量、冷启动与独立复算齐备后，才签发正式跨引擎倍率（替代"6.04 倍"诊断比值）。

### Phase 6 — C1Q：模式一致性与发布质量（CPU 归因先行）

1. 对 vLLM 的 loss 14 / gain 18(2.1291% 失败）：按 `cross-mode-first-divergence-audit-v1` 的模式，定位首个**通用**数值或状态差异（候选面：RoPE 部分旋转 0.25 与 mrope section [11,11,10] 的边界、attention 输出门 swish 的实现序、FP8 KV 量化钳制 ±448 与 scale 制度、logits 读出精度）。定位到类别前不动任何题目相关参数；修复按原模式合同验收冻结二进制。
2. MTP2 对 MTP0(256 题，完整流相同 50)：同模式归因（S5/S10 对照）已排除新路径，剩余差异按状态提交/回滚合同定位；不用归因替代原门。
3. 最新 vector 二进制补做完整 1,024 题质量验收（当前只有 820 性能流证据）。

### 支撑项（随主线顺带交付）

- **tuning cache 出厂**：分支 `baselines/tuning.json` 的 attention 条目为空导致首次加载现场搜索，是冷启动 +3.6046 s 的主因。用离线 tuning 运行生成 SM120 条目随仓库出厂（指纹绑定制品哈希，变更自动失效），消除冷启动回退。
- **主机间隙归因**:nsys(`candidate-vector-shared-binding-v2.py` 的会话模式）拆 TTFT 28.0951 − GPU 25.6 ≈ 2.5 s 的构成（调度、tokenize、读出、MTP 同步），单列候选，不与算子候选混在同一 A/B。
- 64K 输入集（`64k-mtp0` 配置已存在）在 32K 矩阵稳定后扩展。

## 6. 验收矩阵汇总

| 阶段 | 数值 | 边界/canary | 资源 | tuning | 服务（五组 20 门） | 质量 |
|---|---|---|---|---|---|---|
| P1 attention 位精确 | 36+9 逐位 + exact_match + F64<1e-4 | 63…257 + 0xA5 | regs/local/shared/占用登记 | ≥1.02 冷暖 AB/BA，历史 {0,8K,32512} | 主 ≤0.98 / 次 ≤1.05 / 漂移 ≤5% / 820 流 | 1024/1024 逐 token |
| P2 attention 仿真 | 探针误差 ≤ 现行 + F64 对照 | 同上 | 同上 | 同上 | 同上 | **256/256 预判 → 1024/1024** |
| P3 linear | 按门类（exact 或 F64 误差门） | 形状矩阵 | 同上 | 同上 | 同上 | 1024/1024 |
| P4 驻留/合批 | 状态逐位（KV/recurrent/prefix) | 取消/回滚/驱逐 | 报价兑现 + 物理驻留条数 | — | C4 makespan/整组吞吐 | 不回归 |
| P5 跨引擎矩阵 | — | — | — | — | 预注册 32K/64K C1/C4，漂移 ≤5% | 沿用 |
| P6 质量 | 首个通用差异定位记录 | — | — | — | — | 1024 题配对正确率非劣（整体置信下界 + 各 variant 点估计） |

## 7. C1H 与 C1D

- **C1H**:H/B 无实机，六个 AOT 变体（SM90a WGMMA/TMA、SM100a/SM103a 单双 CTA tcgen05、SM120）仅离线核验，不承诺性能。设计预留：数据中心卡 FP32:BF16 = 1:15,FP32 仿真（BF16x9）在该比例下优于 CUDA core——Phase 2 的探针误差数据同时作为 H/B 候选的输入，但 SM120 候选不为此改变形状。设备到位后按交接逐架构验 ABI/数值/资源/自动选择/回退/完整服务。
- **C1D**：每个合入 PR 自带对应质量、服务、CI 与持久证据；PR #10 的"对主干零退化"个别批准不继承。Phase 1/3/4 各自独立成 PR;Phase 2 探针无论成败都先以研究记录归档。

## 8. 决策点 D0：数值合同政策包（Phase 2 证据齐后提交）

若探针确认合同内路线在 SM120 的 attention 下限显著高于 vLLM 的 BF16+FP8 路径（预期如此）,35K C1 完全追平只剩一个政策问题：是否允许 target attention 在定义了**新的质量合同**后使用原生 BF16/FP8 张量核。提交给决策者的证据包包括：

1. Phase 0/1 的实测 roofline 与已达到的位精确收益（合同内上限的硬证据）;
2. Phase 2 探针的误差分布与 256 题预判；
3. vLLM 自身即 BF16 attention + FP8 KV 的事实与其质量对照（vLLM 962/1024 vs native 966/1024，跨引擎门当前 2.1291% 失败是 C1Q 的既有问题，不因性能工作改变）;
4. 可选形态：默认保持 F32 精确、按 profile 选择启用（不动逐位门与 decode 合同）、或不启用只记录差距原因。本方案不预选答案，也不在任何门未过时默认开启。

## 9. 否决档案（禁止重复）

| 方向 | 处理 |
|---|---|
| 单独 inverse-gather K 生产者（去 2 路 STS 冲突） | 真实输入仅 0.5–0.6%，原门否决；不复活、不拆分重测 |
| QT16、tiled/WY/prepared、BF16 FA4 target | 质量失败保持；不换名重做、不重新默认开启 |
| 相邻量化复用、设备 greedy、旧池回收、KV 上报 | 已是默认路径，核对即可 |
| 优先削 GDN launch | 当前 3.8%,10 月 13 日旧画像作废 |
| 大 M FP8 Lt | linear 成主成本后再排；不动 decode 逐位门 |
| Record/Fold、异步 provider、PDL | 各自 SPI/状态合同齐备前不替代 attention 主线 |

## 10. 里程碑（诊断目标，以实测修正，不构成验收）

| 里程碑 | 内容 | 32K C1 TTFT 估计 | C4 makespan 估计 |
|---|---|---:|---:|
| M0 | Phase 0 决策记录 + tuning cache 出厂 | 28.1 → ≈27 s（冷启动不计） | 129.5 s |
| M1 | Phase 1 位精确 attention 过全部门 | ≈18–21 s | ≈95–110 s |
| M2 | Phase 3 linear + 主机间隙候选 | ≈15–17 s | — |
| M3 | Phase 4 驻留/合批 + Phase 5 稳定矩阵 | — | 实测后给目标，先修结构性串行 |
| M4 | D0 政策决策（若批准原生张量核） | 合同内终点 ≈8–10 s；政策变更后 ≈5–7 s | 同步重估 |

合同内（不降低 target 精度）的终点预计距 vLLM 约 1.7–2.2×；§1 的 ≤1.10 追平判定依赖 D0 的决策结果，本方案对此如实记录，不用诊断比值冒充验收结论。

## 11. 立即可启动清单（不占用 GPU 窗口）

1. Phase 0 第 1 步：既有 NCU source CSV 的 PC→源码行归因脚本与五类汇总（纯 CPU)。
2. C1M 第 1 步：在隔离 checkout 应用一行 `core.synchronize` 修正，重签 plan 摘要，排队 GPU 窗口。
3. C1Q 第 1 步：loss14/gain18 的首个通用差异归因（CPU，既有 cross-mode 审计脚本族）。
4. tuning cache 出厂：离线 tuning 运行 → `baselines/tuning.json` SM120 条目（CPU 编排 + 一个 GPU 窗口）。
5. Phase 2 探针的仿真脚本（基于既有 9 份真实输入张量，纯 CPU/主机侧实现，GPU 窗口只做确认运行）。
6. 本方案在路线图 README 的登记与 Phase 0 决策记录的目录身份（按证据规则一次性写入）。
