# CUDA 长上下文性能结论与任务交接

交接日期：2026 年 10 月 17 日，Asia/Shanghai。实验主要于 10 月 16 日完成。目标仍是长上下文延迟和并发吞吐对标 vLLM；任务状态统一见[路线图](../plans/README.md)的 C1 子任务。

**当前主线是 target attention。** 新 profile 中 attention 占 GPU 算子时间 71.8%，GDN 已降到 3.8%。PR #10 已合入主干；后续 vector 分支通过 native 性能门，但完整发布质量和稳定跨引擎对照未完成。

## 版本和交付边界

| 项目 | 固定身份与范围 |
|---|---|
| 已合并主干 | [PR #10](https://github.com/1994/inference.rs/pull/10)，merge 92578d138c8d064442a75bbab0e1645b0da828e3；CI 10/10、原 20 项性能门、1,024 题对原主干完整输出逐 token 一致 |
| 后续优化分支 | codex/fa4-architecture-prefill；性能代码基点 b43c51d517af6fbbee734f0529e5cbd87500b801，包含 FA4 架构接入、MTP proposal 和 VectorSharedKv。交接当日（2026-10-17）尚未合入 main；2026-10-18 已由 merge `e86e05b` 合入 main，本行保留交接时点并注明该后续状态 |
| 测量代码映射 | b43c51d 仅合并 main 历史，整个 tree 与 bcc742b35308d9adf279cdbf2ed5860cd378ce79 相同，没有借历史合并重测 |
| 已测 release | SHA256 730c2348f1948f769f498bdee5c6d674660d80a8f19321f5ebacd598a3f159d8；冻结构建 build-vector-shared-integration-v2/result.json |

PR #10 使用用户明确批准的“对主干零退化”合并政策。该次批准仅覆盖 PR #10，既没有将原 vLLM 质量失败改成通过，也没有批准后续整个分支直接合并。

## 已验证的服务收益

设备为 RTX 5090 SM120，170 SM，32 GiB 级显存，CUDA 13.4.92，驱动 615.78.08。固定模型为 Qwen3.8-27B-NVFP4。32K 对照使用相同的 32,384 输入 token、256 输出 token、MTP2、0.88 显存利用率、32,768 benchmark 窗口。

以下是**vector 候选对 S10 native 父版本**的正式结果，两者都不是 vLLM，S10 也不等同于 PR #10。

| 指标 | S10 中位数 | vector 中位数 | 五组配对结论 |
|---|---:|---:|---|
| 32K 单请求 TTFT | 30.6729 s | 28.0951 s | 延迟降低 8.41%，95% 区间 8.34%–8.48% |
| 32K 四请求组 makespan | 139.7757 s | 129.4601 s | 总完成时间降低 7.38% |
| 32K 四请求整组吞吐 | 7.3260 token/s | 7.9098 token/s | 吞吐提高 7.97% |
| 冷启动至服务就绪 | 24.0282 s | 27.6328 s | 中位数增加 3.6046 s，包含自动调优成本 |

AB、BA、AB、BA、AB 共五组、10 个新模型进程，每个进程保留原四种负载的一次预热和 21 次测量，以及 32K C1/C4 各一次预热和测量。没有过滤、替换或追加样本。

20 项性能门全部通过，820 对完整输入及输出流相同，两臂内部稳定，独立脚本已复算全部原始报告。C1 TTFT、C4 makespan 的配对 95% cost-ratio 上界为 0.9165894、0.9270609，低于原 0.98 门；其他指标上界和各臂漂移分别满足原 1.05、5% 门。

四请求吞吐为“整组实际输出 token / 整组 makespan”，不证明四条 32K 状态同时驻留或 prefill 合批。不同轮次的百分比不能直接相乘当作累计实测收益。

## 与 vLLM 的差距

固定 wheel 为 0.31.1rc1.dev289+g3fe917abb，源提交 3fe917abb13160d60bea5330399f5c09ca282348。实际路径已核验：FlashInfer GDN、Cutlass NVFP4、FLASHINFER BF16 prefill、FP8 KV/XQA。native target 保留 F32 数值合同，不能声称两者计算精度完全相同。

- 最新 native 正式 32K C1 TTFT 为 28.0951 s，旧 vLLM 诊断为 4.6480 s，相除约 **6.04 倍**。vLLM 数字来自一个新进程的一次预热和三次测量；这是跨轮次诊断比值，不是新的配对验收结论。
- 旧 vLLM 32K C4 吞吐中位数为 32.7615 token/s，但三次测量最大相对漂移约 11.13%，超过 5% 门。当前没有合格的稳定 C4 跨引擎倍率。
- 64K 历史对照还有上下文窗口配置边界，不能推广成当前分支的匹配结果。

## 当前瓶颈与硬件依据

新 profile 使用一次预热、三次测量，四次完整输出与固定参考一致；预热不计入下表。**这是 GPU event 算子时间，不是未插桩 TTFT。**

| 算子 | 中位数 | GPU 算子时间占比 |
|---|---:|---:|
| attention | 19.1310 s | 71.80% |
| linear | 4.8169 s | 18.08% |
| GDN delta | 1.0158 s | 3.81% |
| embedding | 0.6153 s | 2.31% |

target attention 为 18.8841 s，draft attention 仅 0.2469 s。10 月 13 日“GDN 占约一半”的旧画像已经不能代表当前代码。

同输入 NCU 单次诊断显示：VectorSharedKv 的 LSU pipe 使用率为 54.10%，FMA heavy 为 50.84%，DRAM 峰值使用率仅 0.664%；80 寄存器、零 spill、资源允许每 SM 两个 CTA。相对旧 SharedKv，共享加载指令降到 67.46%。当前形状不能按 DRAM 带宽瓶颈处理；候选必须解释实际指令、片上访问和依赖成本能削减多少。

当前仍有约 65.64 亿 FMUL、35.66 亿 FADD、31.98 亿 HADD2.F32 动态 warp 指令。不能把存在低效指令或 bank conflict 直接等同于主要延迟原因。

### K 搬运候选已经否决

当前布局存在 2 路 K 共享写入 bank conflict；NCU 的 66,879,488 个额外 shared wavefront 全部位于三个 STS 位置，消费者 LDS 没有这类额外 wavefront。

独立候选改为连续共享写入、逆布局 global K gather。CPU 穷举证明共享数据相同；36 个通用场景和 9 份真实输入逐位一致，仍为 80 寄存器、零 spill、每 SM 两个 CTA。代价是每 warp 的 global K 32-byte sectors 从 2 增到 4。

第一轮固定 55 场景中，真实输入 aggregate cost ratio 约 0.9938–0.9949，只快约 **0.5%–0.6%**；部分 synthetic checkpoint 的 warm 数字约快 5%，没有转化为真实输入和冷缓存的合格收益。该候选未满足每个主比较至少 2% 的原门，已经停止，不能接入生产或重新测到通过。

## 质量边界

官方任务答案是 ground truth；vLLM 是配对比较对象，不是 token oracle。

| 结果 | 已证明的范围 |
|---|---|
| PR #10 对原主干 1,024 题 | 完整输出逐 token 相同，native 均正确 966 题，vLLM 正确 962 题 |
| 原主干及 PR #10 对 vLLM | loss 14、gain 18，单侧 95% 上界 2.1291% 高于原 2% 门，仍失败 |
| BF16 FA4 target 的 1,024 题 | 总正确数也为 966，但完整输出仅相同 95 题，任务控制门失败 |
| S10 MTP2 对原 MTP0 的首批 256 题 | 244 对 246，loss 3、gain 1，完整流相同 50；原门失败，后续 768 题未继续 |
| 同批 S5 MTP2 对 S10 MTP2 的归因 | 256 条完整流相同，仅能说明差异不是 S10 新路径造成，不能替代原质量门 |
| 最新 vector 二进制 | 820 条性能流一致，尚无该二进制的完整 1,024 题任务质量验收 |

原 generation budget、EOS、严格 scorer、题目和统计门保持不变。不能靠同模式归因、总准确率相同、扩样或重抽题消除原失败。模式一致性应从通用数值、状态提交和回滚合同定位，不对留出题答案调参。

## 并发和显存事实

- 闲置且不兼容的 slot pool 回收已经生效，最新日志记录回收约 3.22 GB，不能再作为新修复。
- 当前 32K target 加 draft 的 private quote 为 3,099,791,228 bytes，比旧报价增加 137,363,464 bytes，约 131 MiB，来源是一组 program 级 FA4 F32 draft BF16 K/V、Q/O staging 和 metadata，不是每层重复分配。
- 旧 S4 实际 private pair pool 增量为 2,683,231,932 bytes，不能作为当前分支实际分配量；非 pool 的图、代码和 capture 成本也必须计入。
- 当前分项采集工具在 current-memory-components-v1，源码在 source-current-memory-components-v1/crates/backend/cuda/examples/cuda-resident-memory-components.rs，**尚未编译运行**。它会采集 target、draft、base activation 和实际 dataflow/lifetime。
- 通用 inspect-package CLI 不支持该 NVFP4 precision provider，本次纯 CPU 尝试失败已保留；后续使用已准备的 LoadedModel 诊断。服务槽数不证明物理驻留数，先量实际资源再决定 arena、工作区复用或块级 prefix/COW。

## 可独立拆解的任务

| 编号 | 工作范围与输入 | 必须交付的验收结果 |
|---|---|---|
| C1Q | 模式一致性与发布质量：固定 scorer、原 MTP0 控制、失败 MTP2 输出及 matched MTP2 归因 | 首个真实数值或状态差异的通用定位和修复；按原模式合同验收发布二进制；保留失败，不能用同模式归因代替原门 |
| C1A | Target attention 完整成本：本次 profile、NCU、9 份真实输入、canonical F32 oracle | 先量可削减成本，再做一个可归因候选；原数值、边界、canary、资源门通过后，自动 SPI/tuning、原 20 项五组服务门及相应质量门全部通过 |
| C1M | 真实显存与多会话驻留：分项采集源码、quote、engine/KvPages 合同 | 先测实际 target/draft、arena、workspace、graph，再决定实现；报价兑现、状态独立、取消/回滚/prefix 正确，报告实际驻留与完整 C4 makespan/吞吐 |
| C1V | 稳定 vLLM 长矩阵：冻结制品、强 wheel、相同输入和实际配置 | 先解释旧 C4 波动或配置边界，实质修复后预注册匹配 32K/64K C1/C4 矩阵；保留全部样本、质量、冷启动与独立复算，合格后才签发正式倍率 |
| C1H | H/B 实机验收：现有六份 AOT 与架构 tuning 接线 | 设备到位后逐架构验 ABI、数值、owner、资源、自动选择和完整服务；当前无 H/B 机器，离线编译不等于性能收益 |
| C1D | 分支交付与合并：对应质量、服务、CI 和持久证据 | 形成可 review 的 PR 并处理反馈；PR #10 已完成，后续分支不能继承其个别合并批准 |

C1A 的具体下一版 kernel 尚未选定。CPU 审计可以独立拆解，同一机器的重构建及 GPU 测量串行。C1M 不能先承诺 4 条或 16 条 32K 驻留，C1V 不能只为取得好漂移值重跑，C1D 不能把文档或分支推送当成优化完成。

H/B 代码已离线编译六个变体：SM90a WGMMA/TMA、SM100a 与 SM103a 的单 CTA/双 CTA tcgen05/TMEM/TMA，以及 SM120。尚无 H100/H200、B200/B300 实机数值或性能结论。

## 不要重复的工作

| 方向 | 处理 |
|---|---|
| QT16、被否决的 tiled/WY/prepared、BF16 FA4 target | 保留质量失败，不能重新默认开启或换名重做 |
| 单独 inverse-gather K 生产者 | 本轮原门已否决，停止该候选 |
| 相邻量化复用、设备 greedy、旧池回收、KV 上报 | 核对已有默认路径，不重复立项 |
| 继续优先削 GDN launch | 当前只占 3.8%，不依据旧画像插队 |
| 大 M FP8 Lt | linear 成为主要剩余成本后再排；独立质量 profile，不改 decode 逐位门 |
| Record/Fold、异步 provider、PDL | 依赖各自 SPI/状态与完整周期合同，不能替代当前 target attention 主线 |

## 执行边界

证据主机为 r@192.168.1.45，隔离 checkout 均在 /home/r/benchmarks/inference.rs。主工作区 /home/r/code/inference.rs 保持干净，本地 /Users/r/code/inference.rs 镜像保持只读。

重任务使用既有 host-memory50-2026-10-15/safe-run-v3.sh --memory-gib 50：主机 50 GiB 上限、45 GiB high、swap 0、pids 256、OOM group kill，启动至少 8 GiB MemAvailable；50 GiB 不是 GPU 显存预算。一次一个重任务，使用原全局锁。

GPU 启动保留连续 30 个间隔至少一秒、不超过 2% 的原空闲门，不干预桌面进程、驱动或锁频。NCU 保留原仅 CAP_PERFMON 方案和 --clock-control none。既有独占输出目录不能直接重跑覆盖，制品和脚本变化必须有新身份。

合格优化经 SPI/tuning 自动生效，不加让用户判断好坏的开关；benchmark 显式窗口不变成产品上下文上限；不通过降低 target 精度、放大生成预算或更换 scorer 获取通过。本文不保存凭证。

## 原始证据入口

统一根目录为 /home/r/benchmarks/inference.rs/exact-shared-attention-2026-10-16，以下均相对此根目录。读取时核对源码、二进制、配置和全部原始样本绑定，不能只看 completed 字段。

| 结论 | 入口 |
|---|---|
| 合并和历史映射 | PR10-merge-completion-v1.json；main-vector-integration-v1/publication-main-history-v1.json |
| 冻结构建与 GPU 验证 | build-vector-shared-integration-v2/result.json；gpu-vector-shared-v2/result.json |
| 正式五组与独立复算 | vector-performance-v1/analysis-v1.json；vector-performance-v1/independent-final-audit-v1.json；原始报告在 vector-performance-v1/runs/ |
| 最新算子归因 | vector-operator-profile-v1/review-v1.json；原始 event 和请求范围在 runs/vector-exact-target-32k-operators-v1/ |
| NCU 与原始 PC | vector-hardware-counters-v2/review-v1.json；其 run-v1/ 下的 NCU、metrics/source CSV |
| K 搬运否决 | vector-shared-producer-bank-v1/one-shot-decision-v1.json；其 numeric-v1/、timing-v1/ 与 layout-CPU-proof-v1.json |
| 原质量门 | bitexact-performance-quality-contract-audit-v1.json；quality-fa4-confirmation-v2/comparison-v3.json |
| MTP 跨模式失败与同模式归因 | quality-proposal-confirmation-v2/cross-mode-first-divergence-audit-v1.json；quality-MTP2-attribution-v1/matched-control-comparison-v2.json |
| 旧 vLLM 诊断 | runs/fresh-vllm32k-mtp2-c1-v1/result.json；runs/fresh-vllm32k-mtp2-c4-v1/result.json |
| H/B 离线范围 | FA4-AOT-integration-v1/H-B-instruction-integrity-audit-v1.json |
| 显存与未执行工具 | current-memory-quote-attribution-v1.json；current-memory-components-v1/plan-v1.json；其 metadata-v1/result.json 为失败的 CPU CLI 尝试 |

关键 SHA256：正式性能汇总 6d6cacb0a19100900a6ab6d8b858b70dba0a417a09657e1ddfbaad6ba35ef9c7；独立复算 85d104ae55cd9fc7bd7de9b919f3b4544ede183e3e8df6b0db7dfe54f8f6a885；算子归因 b85b52d3fd2b6bd5a3950716ba7eb56a096741ffa446d4b975b5e04fac07b594；硬件计数器 1104ebdfc6568c181335dddd62fcda4376df2dcf436159d4f0c80e8c5829152a；K 搬运否决 f473e19e0a170753bdff81bda7631777cc1370d0dea09d4e56aa24f87fa97796。

后续执行先选择一个 C1 子任务，复核对应证据和现状。无需重复已经完成的五组 native 复测；正式证据还需按共同基线合同持久存档，本交接不替代 baseline ID。
