# 默认非阻塞 CUDA 执行评审

当前结论：默认非阻塞计算的方向符合设备驻留与减少 host 等待的路线，但性能验收仍不可判定，PR 保持 draft，不能合入。MTP2 batch4 TTFT 在十个进程对后的 95% 区间仍跨过预定 +10% 上限，且部分 ratio 漂移超出预定稳定性要求。不能用 2B 收益抵消这一项。

本次延续已合并的 PR #56（main `4746d89a24260e799dcd437cdb2f72e6d8070f84`），全部后续实现位于 PR #57。compute 默认使用非阻塞 submit/poll，没有启用开关。覆盖 prefill、首次 slot 绑定、普通 decode、混合 readout、private program 与 MTP draft/verify/rollback。只有一个 compute flight；这不是 ADR-0004 的双 flight 队列或全设备 token/KV 延迟回填。

## 所有权与执行成本

挂起的 future 拥有整个 executor，保留图、device/pinned buffer、scratch 与状态。ticket 校验 owner/generation，最终 stream fence 成功且 ticket 被消费后才发布输出、游标和资源视图。完成唤醒既有 device owner；未增加 runtime 或逐请求线程。ticket drop 不释放飞行资源，未知完成进入隔离。独立诊断 API 的同步兼容包装不经过 provider compute；load/capture/resource admission 与稳态计算分别处理。

review 发现并修正了隐含同步 ToHostVec、每次 hidden 上传分配、F32 KV 全容量复制、逐 head D2D 提交、超大 logits staging 超限，以及飞行期间只读 program 查询兼容问题。KV 按已覆盖行复制，recurrent state 仍整体复制；head-major prefix 用一条 cuMemcpy3DAsync，并测试不同容量和未触碰尾部。没有改变数值算术或放宽容差。

非阻塞所有权本身不保证服务加速。最终 fence 必须覆盖 detached draft catch-up、worker store 和 rollback；pinned staging 增加保留的 host memory。性能判定逐模型、逐场景保留慢项。

## 比较身份与统计

实测设备为 45 上的 RTX 5090 32 GiB。baseline 源码 `4746d89a24260e799dcd437cdb2f72e6d8070f84`，candidate 测量源码 `212ebac6b23b5387ef34fb9547d311a4f14f1bf1`。两边使用 rustc 1.99.0、相同 workspace release（thin LTO、codegen-units=1）、相同 CUDA features、模型制品、输入、采样、EOS、有效资源配置和冻结 tuning；无并行构建/benchmark。

每个新进程内每 case 一次 warmup、三次测量，进程对 AB/BA 交替。进程对是独立单位，不能把请求/token 当独立复测。按预先声明的配对 ratio 中位数做 cluster bootstrap；N≤5 精确枚举，扩样使用固定 seed 的 100000 次抽样。95% 区间逐项报告，不能视为联合 SLO。延迟区间上界 ≤1.10 才通过该项，最小有用收益为 5%。MTP2 batch4 TTFT 五组不可判定，依预设规则扩至十组；保留全部慢样本。

完整输出 token、count、finish reason、逐轮 cache reuse 和生效配置要求配对一致；自然 EOS 不补 token。原 27B hot_long 立即 EOS，TPOT 不可用，另登记 private-decode-only 长 prompt 补测。该补测将 newline padding 放在原问题之前，保留问题结尾，单独 warmup 填 cache；不将其与原 EOS profile 混算。

部分进程对 ratio 漂移超过预设 10%。因此这些数据不签发新稳定 baseline，也不替换历史 serving baseline；收益幅度和放行状态必须按这一限制解释。

原始报告、日志、遥测与输入持久保留在 `45:/home/r/benchmarks/inferloop/cuda-nonblocking-v1`，不依赖 checkout 的 artifacts 暂存。[机器可读汇总](artifacts/cuda-nonblocking-2026-10-25.json)记录源码、binary/input/report/telemetry hashes、每个独立单位和不可用项；[复现说明](artifacts/cuda-nonblocking-2026-10-25/README.md)附冻结输入、profile 和脚本。

## 逐场景实测

共 30 个独立进程对；baseline 与 candidate 各 600 个完整逻辑请求，其中各 150 个 warmup、450 个测量请求。所有配对的完整 token、finish reason、生效配置与逐轮 cache reuse 一致。下表是配对 ratio 中位数的变化，方括号为 95% 区间；正数表示更慢。逐项绝对秒值、全部配对值及原始哈希在机器汇总中。

| 模型 / 场景 | TTFT 变化 [95% CI] | TPOT 变化 [95% CI] | group makespan 变化 [95% CI] |
|---|---:|---:|---:|
| 2B MTP0 / short | -22.5% [-23.4%, -8.7%] | -0.7% [-1.9%, +0.1%] | -1.0% [-2.4%, -0.2%] |
| 2B MTP0 / long | -26.1% [-27.9%, -20.7%] | -1.0% [-1.3%, +1.0%] | -3.6% [-4.3%, -2.1%] |
| 2B MTP0 / batch4 | -25.5% [-32.2%, -18.9%] | -0.5% [-2.1%, +0.0%] | -3.2% [-5.4%, -2.4%] |
| 2B MTP0 / hot_long | -0.4% [-0.9%, +0.2%] | -2.1% [-2.4%, -0.5%] | -0.4% [-1.1%, -0.1%] |
| 27B MTP0 / short | +6.4% [-1.5%, +9.1%] | +0.1% [+0.1%, +0.4%] | +0.2% [+0.0%, +0.3%] |
| 27B MTP0 / long | -0.9% [-3.1%, +0.7%] | -0.0% [-1.6%, +0.3%] | -0.1% [-1.8%, +0.4%] |
| 27B MTP0 / batch4 | -0.4% [-2.0%, +0.7%] | +0.1% [-3.0%, +0.2%] | -1.3% [-3.3%, +0.4%] |
| 27B MTP0 / hot_long | +0.0% [-1.8%, +0.5%] | 不可用（EOS） | +0.0% [-1.8%, +0.5%] |
| 27B MTP2 / short | +3.7% [-1.0%, +7.2%] | +1.3% [+0.7%, +2.6%] | +1.3% [+0.8%, +2.3%] |
| 27B MTP2 / long | +0.2% [-0.6%, +1.6%] | +1.5% [+1.3%, +2.1%] | +1.6% [+1.1%, +2.2%] |
| 27B MTP2 / batch4 | +2.1% [+0.9%, +12.0%] **不可判定** | +1.2% [-0.1%, +2.9%] | +1.6% [+0.1%, +3.5%] |
| 27B MTP2 / hot_long | -0.3% [-0.6%, +0.2%] | 不可用（EOS） | -0.3% [-0.6%, +0.2%] |
| 27B MTP0 private / hot_long | +0.1% [-0.8%, +0.9%] | -0.4% [-1.6%, +0.6%] | -0.2% [-1.1%, +0.8%] |
| 27B MTP2 private / hot_long | +0.1% [-0.7%, +1.1%] | -0.4% [-2.0%, +0.4%] | -0.0% [-1.2%, +1.2%] |

2B 的 short/long/batch4 TTFT 显示超过预定 5% 的有用收益，区间排除无变化；这不构成全模型普遍加速。27B MTP0 short TTFT 中位数慢 6.4%；27B MTP2 的部分 TPOT/group makespan 也略慢，均逐格保留。关键阻塞为 27B MTP2 batch4 TTFT：十组后的中位数 +2.06%，95% 区间 [+0.95%, +11.98%]，仍不可证明满足 +10% 上限。

诊断计时单独保留，不混入正式样本。target prefill/verify 阶段接近基线；candidate 某次 warmup 出现 `prefill → decode+2prefill → 3decode+prefill`，baseline 该次为 `prefill → decode+3prefill`。这是准入/分组对首 token 延迟敏感的证据，尚不能将全部差异归因到某个 kernel 或声称已修复其影响。

batch4 的有限样本 TTFT 尾部如下。每个请求只作为描述性观测，不被当作独立进程对；这些 p95/max 没有生产尾延迟保证。

| 模型 | 每边测量请求 | baseline observed p95 / max ms | candidate observed p95 / max ms |
|---|---:|---:|---:|
| 2B MTP0 | 60 | 73.93 / 77.78 | 57.45 / 60.17 |
| 27B MTP0 | 60 | 120.47 / 123.46 | 121.22 / 135.23 |
| 27B MTP2 | 120 | 176.38 / 180.88 | 181.43 / 210.51 |

## 启动与内存成本

下表分别取每个进程 startup 和采样 server-tree RSS peak 的跨进程中位数，baseline → candidate。RSS 是采样峰值，可能漏掉更短的瞬时尖峰；cgroup 峰值还包含 harness 和启动。GPU 峰值为包含桌面进程的设备级观测，全部原始值列在机器汇总。CPU 采样包含请求间隔，仅作资源观测。

| 模型 / profile | startup 秒 | sampled RSS peak MiB |
|---|---:|---:|
| 2B MTP0 | 5.413 → 5.417 | 934.6 → 1002.7 |
| 27B MTP0 | 20.429 → 20.729 | 1299.2 → 1427.7 |
| 27B MTP2 | 29.586 → 29.141 | 1565.9 → 1687.9 |
| 27B MTP0 private | 20.129 → 19.832 | 1325.6 → 1450.9 |
| 27B MTP2 private | 28.539 → 28.238 | 1553.7 → 1687.5 |

pinned staging 有明确保留成本：同一 readback 批次限制 64 MiB（含配对整数 staging），大 geometry 使用 compact arena，图/slot 切换不逐步分配 pinned 内存。这个限制不是整个进程的 pinned 总量。配置回读保持相同图宽度与并发约束，已测 batch4 的四个请求完整完成；未测最大配置 16 请求的满载峰值。

## 源码边界与设备验证

全部 serving 计时均绑定 candidate `212ebac`，baseline `4746d89`。其后的 `f6527dd` 修复错误路径：未知 fence/driver completion 时，`poll` 必须返回错误、消费 ticket，并保留飞行 owner 与故障隔离。新增测试先在旧实现上失败（实际为 `Ok(None)`），没有注入真实设备故障破坏后续实验。成功计算的设备操作序列未改，但未在 f652 重跑整套 serving 矩阵，不能把本文解释为最终 head 已通过性能验收。

历史完整 `make test-cuda` 在 `b793428` 上收集 308、通过 143、ignored 165；额外显式运行 5 个 pinned readback 与 4 个 2B/27B owner fixture 测试，全部通过。`212ebac` 的 F32 worker prefix、FP8 不等容量 prefix/tail，以及 prefill/first binding/decode/mixed readout 三项真机 release 回归通过。独立的 resident/slot 数值与状态套件也保留在原始日志中。GPU capture 测试使用串行 runner；早期并行 capture 失败保留，不能算作通过。

f652 的 `cargo fmt --all --check`、`make check-rust` 和 `make check-tools` 通过。每次交付仍须检查 PR 当前 head 的 CI 和设备校验，不能用本文的历史结果替代最新检查。CUDA hosted 的编译、lint 与未执行/ignored GPU 测试，也不能当作设备执行或吞吐证明。测试清单已登记 848 项。

## 放行限制

当前四个基础场景加 private decode 补测不能建立持续/混合到达率的生产 SLO。owner 的混合 prefill/decode 状态与完整输出已由设备测试覆盖；持续负载吞吐、P99、跨 GPU 与配置上限 16 序列的满载内存仍未由这份有限矩阵证明。下一步需要在保持完整输出/资源合同的前提下，降低准入与执行分组对 MTP2 batch4 TTFT 的影响，并在预先登记的稳定设备窗口重建可放行的独立对照。不得事后放宽 1.10 门槛、删除慢样本或签发不稳定 baseline。
