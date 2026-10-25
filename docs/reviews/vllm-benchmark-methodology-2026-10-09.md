# vLLM 对比测试方法审查

审查日期：2026 年 10 月 9 日。源码基点：`7c79d99`。本轮检查测试脚本、服务配置与历史记录，只修改文档。后续验收统一进入 [路线图](../plans/README.md) 的 P0 · B2/B3，具体合同见 [性能基线方案](../plans/performance/baseline.md)。本文保留问题与验证证据，不另定基线规则。

## 结论

当前方法适合开发阶段发现明显回退，尚不足以证明“当前版本与 vLLM 的精确性能差距”。已有记录和源码支持 prefill 图重放、并发准入与池化资源存在优化空间；历史矩阵的具体倍率、几个百分点的胜负、生产吞吐与尾延迟仍需重新测量。

测试没有明显的“少生成 token 所以更快”漏洞：它使用同一组 token 输入、检查实际输出长度，并要求热缓存出现复用。主要缺口在配置与数值身份、接口工作量、样本与负载范围、原始证据保存。偏差方向不统一，不能给历史数字统一乘一个修正系数。

| 要回答的问题 | 现有证据能支持到哪里 |
|---|---|
| native 是否有真实瓶颈 | 历史阶段记录与当前调用路径支持瓶颈方向；本轮未复测耗时 |
| 当前是否比 vLLM 慢 5.86 倍或 1.897 倍 | 不能确认。这是历史特定用例、特定指标的比值，原始报告未在当前工作区找到 |
| TPOT 比值是否等于吞吐比值 | 不等于。现有 TPOT 是每请求平均出 token 间隔，吞吐需按共同测量窗口统计 |
| 相同模型路径是否表示相同数值计算 | 不表示。权重处理、activation、KV、recurrent state 与算术规则需要分别确认 |
| 是否已验证生产服务与 MTP 普遍收益 | 未验证。现有负载是四种有限请求组，缺少持续压力、混合负载与代表性任务 |

## 做对的部分

[输入生成](../../tools/bench/serve-workloads.py) 第 44–82 行由同一个 tokenizer 生成 token 数组，两边直接使用这些数组；[报告](../../tools/bench/serve-compare.py) 第 375 行保存 workload 哈希。这避免两套 chat template 或重新分词造成输入长度不同。非热缓存用例在正文开头加入变化的请求标识，也降低了完整 prompt 意外命中的机会。

[请求采集](../../tools/bench/serve-compare.py) 第 187–265 行使用流式首个 token 到达时间，保存 token IDs、完成原因和到达时间；native 的 unsuccessful measurement 会失败。[门禁](../../tools/bench/compare-results.py) 第 15–80 行拒绝未完成报告、缺失样本、无效延迟、不同实际输入/输出长度、重复 slot，以及没有任何热前缀复用的报告。预热与正式样本分开，每个出现的用例至少有三轮正式测量。

这些检查应保留。它们证明部分工作量已对齐，不能替代下面的配置、质量和统计检查。

## P1：报告身份不足，门禁会接受不同模型包与资源预算

位置：[serve-compare.py](../../tools/bench/serve-compare.py) 第 362–379 行；[compare-results.py](../../tools/bench/compare-results.py) 第 40–51 行。

门禁只对齐模型路径、workload 哈希、输出上限、temperature、MTP 深度、cache 标记和 EOS。`model_package_sha256` 虽被保存，但由调用者手填，默认为 `None`，且不参与校验；`gpu_memory_utilization` 被保存，也不参与校验。引擎版本可缺失，GPU/driver、实际计算与 KV dtype、权重再量化、执行环境和最终生效配置没有形成可校验的报告身份。

本轮用现有 fixture 做内存内探针：模型包哈希分别为 A/B，或显存比例分别为 0.88/0.50，只要延迟与其余字段相同，门禁都会返回 `passed=True`。这说明门禁通过不能证明比较条件相同；不说明历史实验一定用错条件。

修正要求：自动记录实际模型制品指纹、源码/二进制、每个引擎自己的固定版本、GPU UUID/型号、driver/工具链与影响执行的配置。严格对照要求共同制品与资源条件对齐；产品对照允许已声明的数值或策略差异，但必须附质量结果。两个引擎的版本与二进制当然不同，要求分别固定和可追溯，不要求哈希彼此相等。

## P1：调度限制只传给 vLLM，不能把结果归因于 CUDA kernel

位置：[serve-compare.py](../../tools/bench/serve-compare.py) 第 78–127 行；[runtime 默认值](../../crates/engine/runtime/src/config.rs) 第 20–24、90–92 行；[加载配置](../../crates/service/cli/src/backend/cuda.rs) 第 41–49 行；[服务配置派生](../../crates/service/cli/src/support/serving.rs) 第 29–43 行。

脚本默认给 vLLM 传入 `max_model_len=8192`、`max_num_seqs=16`、`max_num_batched_tokens=2048`。native 命令只传 MTP 深度与显存比例，不传这些限制。native 当前默认序列数也为 16，但 token budget 从 64 开始，prefill 图宽度自动选择，随后按实际图宽度提高 runtime budget。`--extra` 或配置文件还可以改变这些条件。脚本没有回读并核验最终限制。

因此，长 prompt 与 batch4 比值包含 chunk 大小、准入和容量策略的影响。这些可以构成实际服务性能差距，但不能表述为“相同计算下我们的 CUDA 算子慢这么多”。同为 0.88 的显存比例也不表示两边的 KV、state、回滚、graph 和 workspace 分配相同。

修正要求：分别报告实际上下文容量、scheduler token budget、prefill/verify 宽度、可驻留序列数、KV/state dtype、池化回退与预算分解。对照约束按含义定义，不强行要求不同引擎的同名参数同值。区分共同数值/资源条件的诊断对照，以及同卡、同质量要求下各自合理配置的产品对照。

## P2：关闭 prefix cache 的参数不生效，热缓存检查也过弱

位置：[serve-compare.py](../../tools/bench/serve-compare.py) 第 118–119、171–184、372、393–397、437–438 行。

`--no-prefix-cache` 仅让 vLLM 命令不再追加 enable 参数，没有发送关闭参数；native 命令没有处理该选项。报告却始终写 `prefix_cache_enabled=True`，热用例仍要求命中。本轮探针确认 `prefix_cache=False` 时没有显式 disable。vLLM `v0.31.0` 的 `CacheConfig` 默认开启 prefix caching，省略 enable 不能关闭它；这仅核实当前参考版本的行为，历史所用版本仍需从原始环境核实。[vLLM cache 配置源码](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/cache.py#L142)

默认开启缓存的历史实验不因此自动失效。问题是当前脚本无法可靠表达冷缓存对照，而且仅要求全部正式 hot_long 轮次的累计复用量大于零：一轮命中、其余轮重算也能通过，少量共同前缀命中也不能证明长 prompt 已充分复用。非热用例没有检查实际 miss/recompute。

修正要求：开关在两边真实生效，报告记录实际状态；每请求或每轮记录缓存复用量和实际重算 token。分别测无复用、固定长前缀复用、真实多轮增量。热缓存产品对照允许实现能力差异，差异需要解释；冷计算诊断必须确认实际重算量相同。

## P2：两边接口承担不同工作，客户端延迟不能直接代表引擎耗时

位置：[serve-compare.py](../../tools/bench/serve-compare.py) 第 187–243 行；[native SSE](../../crates/service/frontdoor/src/http/native.rs) 第 78–130 行。

native 测 `/native/v1/stream`，输入与输出均为 token ID，不生成响应文本。vLLM 测 `/v1/completions`，开启 `return_token_ids` 后仍返回生成文本；`v0.31.0` 的该选项还会在第一个 chunk 返回 prompt token IDs。长输入因此会增加首个响应的序列化、传输与解析工作。具体成本未在本轮量化，不能按它推算倍率修正。[vLLM 请求协议](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/openai/completion/protocol.py#L169)、[流式响应实现](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/openai/completion/serving.py#L360)

两边使用同一客户端并不能消除服务端工作量差异。native 每 token 一个事件与 vLLM 每 chunk 多个 token 也有不同的事件开销，偏差方向不能一概判断。当前 native OpenAI 接口还拒绝 `stream=true` 且只接收字符串 prompt，不能仅改路由就复用现有流式对照。[当前接口限制](../../crates/service/frontdoor/src/http/openai/request.rs) 第 50–55、76–79 行。

修正要求：诊断侧测相同 token 工作量并记录服务端 queue/prefill/decode/输出处理和 GPU 周期；用户侧测两边确实支持的等价文本接口，核对最终 token 输入与采样设置。在等价文本 streaming 能力具备前，现有结果标为“native token SSE / vLLM OpenAI text SSE 对照”，不宣称 OpenAI 服务或纯 CUDA 性能已对齐。

## P2：TPOT 公式基本合理，但文档把它当吞吐与逐 token 延迟

位置：[serve-compare.py](../../tools/bench/serve-compare.py) 第 241–263 行；[compare-results.py](../../tools/bench/compare-results.py) 第 81–90 行；原历史服务矩阵已删除，不能再作为复核依据。

现有公式为 `(最后可见 token 到达时间 − 首个可见 token 到达时间) / (可见 token 数 − 1)`。vLLM 官方 benchmark 也按每请求计算 TPOT；其 completions 客户端使用最后有效输出时刻结束 token 延迟。MTP 一次返回多个 token，并不使这个平均值天然错误。[指标说明](https://docs.vllm.ai/en/latest/benchmarking/cli/#understanding-the-latency-metrics)、[v0.31.0 benchmark 客户端](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/benchmarks/lib/endpoint_request_func.py#L236)

但同一 chunk 中的 token 被写成相同到达时刻，这不是各 token 的 GPU 完成时间，也不能把展开数组的零间隔用于 P95/P99 ITL。正式采集应另外保存原始事件的时间与 token 数，ITL 统计事件间隔，TPOT 保留请求级摊销定义。

本轮模拟两个 chunk：100 ms 收到 3 个 token，200 ms 再收到 3 个，脚本得到 TPOT 20 ms/token；真正的两次输出之间间隔为 100 ms。这个结果符合平均摊销含义，并不表示用户每 20 ms 收到一个 token。若全部 token 在一次输出中，TPOT 为零，当前正数检查会拒绝合法完成结果；零值或单 token 请求要按指标可用性处理，不能一律判定测试失败。

门禁对 batch4 的 12 个请求 TPOT 取中位数，吞吐则应按整个共同窗口的 `总输出 token / 总耗时` 计算，两者不能互相取倒数。例如历史 27B batch4 的 TPOT 比值为 1.897，而相同输出工作量下的 group wall 比值为 1.201；即使全部历史数字有效，也不能据 TPOT 宣称总吞吐差 1.897 倍。TTFT 也会通过排队、prefill 和占用影响整体服务容量。

此外，报告每请求的 wall 计到流结束，group wall 包含线程池创建/收尾，而 TPOT 截止最后可见 token，三个终点不同。应分别命名和记录 `time_to_last_token`、request completion wall、group makespan，避免混用。

## P2：三轮小样本与四种请求组不足以验证小幅收益或生产容量

位置：[serve-workloads.py](../../tools/bench/serve-workloads.py) 第 20–41 行；[serve-compare.py](../../tools/bench/serve-compare.py) 第 293–322、424–426 行；[compare-results.py](../../tools/bench/compare-results.py) 第 77–98 行。

默认只有一次预热与三次正式测量。两个引擎各启动一次并跑完所有用例，脚本没有配对交替的外层驱动、预热稳定性判断或置信区间。仓库后续部分 A/B 记录有交错测量，应保留其单独证据，不能反推所有历史矩阵都已交错。几个百分点的差异可能受到进程、时钟、热状态与后台负载影响；中位数本身不给出不确定性。

batch4 通过逐个提交线程发出请求，没有共同开始屏障或实际发出/到达偏斜记录，尤其难把首轮 TTFT 离散完全归因于服务端调度。每组完成后还等待 1.1 秒，属于有限突发请求组，不是持续并发或固定到达率的容量测试。

四个用例均来自同一段英文正文，主要覆盖短输入、重复正文、四个短请求、一个完全重复的热 prompt。历史 long 为 511 token，不能代表 8K/32K 等长上下文。MTP 接受率受任务影响，重复性英文负载不能代表代码、多语言、工具调用与不同输出长度；没有 mixed prefill/decode、排队增长、P95/P99 与 SLO goodput。

另有完整性缺口：workload loader 不校验每个用例的预期 concurrency，门禁也不要求 batch4 必须四个 slot。本轮将 fixture 用例改为 batch4、保留 concurrency=1，仍通过。允许局部 case 文件是合理功能，但报告必须声明期望矩阵并校验其中的 slot 数，否则“完整矩阵”可能漏测。

修正要求：三轮保留为 smoke；正式性能判断执行 [统一计时与统计要求](../plans/performance/baseline.md#计时与统计不能产生假收益)。补齐共同发出屏障、期望矩阵检查、代表性任务及持续/混合负载，避免以有限突发组推断生产容量。

## P2：性能通过不代表数值或质量通过，历史 FP8 结论过强

位置：[compare-results.py](../../tools/bench/compare-results.py) 第 75、101–105 行；[FP8 历史记录](../archive/research/cuda-performance-experiments.md) 的 block-scaled FP8 实验。

默认仅统计 token mismatch，不使性能门禁失败。两边输出相同长度、不同内容可以通过；对质量已独立验收的产品对照，这可以接受，对“保持同一 target 数值”的改动则不够。不同生成轨迹也会改变 MTP 接受长度和计算量。

历史记录给出 2B FP8 仅 3/21 序列与 vLLM 完全一致，并记录 UE8M0 再量化造成抽样投影权重约 2.67% 的相对 Frobenius 变化。当前 vLLM 参考源码确实存在先按原 scale 反量化、再以 power-of-two scale 重新量化的函数；是否用于历史实测必须由那个版本与实际 dispatch 证明。该函数能解释数值不等价的一项来源，不能证明所有输出差异均来自它，也不能证明本实现的整模型计算“误差 0”或质量不降。[vLLM 再量化源码](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L1012)

修正要求：分别记录 checkpoint 身份与执行中的数值身份。保留独立算子参考，再补固定上下文下的整模型 logits/state 对照；同引擎的普通 decode/MTP 比较最终输出与状态。跨引擎若无法逐位对齐，使用相同 teacher-forced token 轨迹做诊断，并独立按任务验证质量。性能、数值正确性和任务质量各自给结论，不能用任一门禁替代其余两项。

## 原始证据与本轮验证

当前工作区没有 `artifacts/perf-r40/`，其他 serving/vLLM 对比原始报告也未找到；`benchmarks/baselines/` 保存的是投影微基准，不是替代性的 vLLM 服务矩阵。历史表格与阶段耗时在本轮只能作为仓库已有记录引用，不能逐请求核查配置、完成原因和样本。没有在 CUDA 设备上重跑服务。

[serve-compare.py](../../tools/bench/serve-compare.py) 第 359–383 行按 `模型目录名-MTP深度-engine` 命名并直接写入报告与 server log，同一 output-dir 的后续运行会覆盖旧证据。应使用唯一 run ID，拒绝覆盖，并保存关联的 workload、服务器日志、环境、遥测和判定结果。已有 safe-run 与 hardware-monitor 可继续作为运行保护和遥测入口，但 serving 脚本不会自动调用它们，报告也不验证它们是否存在。

本轮执行现有两个 Python 测试模块，共 **15 项通过**；内存内探针复现不同模型哈希/预算被接受、单 slot batch4 被接受、cache=False 未发关闭参数、native token budget 未转发，以及流式突发的 TPOT 语义。没有新增测试或修改脚本。当前统一门禁 [gate.sh](../../tools/check/gate.sh) 第 9 行只收集 `test_compare_results.py`，漏收 `test_serve_compare.py`；该问题随 [测试组织方案](../plans/engineering/tests.md) 收敛。

## 修复与验收入口

上述修复统一进入 [路线图](../plans/README.md) 阶段 0，按 [性能基线方案](../plans/performance/baseline.md) 的交付顺序验收。该方案维护对照身份、release 与配置对齐、完整矩阵、统计判断和证据留存，覆盖正确性诊断、计算调度诊断、MTP 增益与服务产品对照。

先修比较身份与报告语义，再重跑完整矩阵；只有新报告能用于“当前差距”。历史 profile 可继续指导选取实验，但 CUDA kernel、调度、接口与数值成本需要按各自证据归因。
