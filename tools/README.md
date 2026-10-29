# 工具导航

统一入口是仓库根目录的 `make check`，各分项见[质量门禁](../docs/guides/quality-gates.md)。运行暂存写入 `artifacts/`，不进入产品运行依赖；已验收基线按 [性能基线方案](../docs/plans/performance/baseline.md) 持久保存。

| 目录 | 入口 | 用途 |
|---|---|---|
| `build` | [backend_cfg.rs](build/backend_cfg.rs) | CLI 与 CUDA crate 共用的后端编译能力条件 |
| `cuda` | [kernels.py](cuda/kernels.py) | CUDA AOT 制品统一清单、完整性校验和固定工具链重建，见 [Kernel 工程](../crates/backend/cuda/kernels/README.md) |
| `check` | [gate.sh](check/gate.sh) | 本地与 CI 共用的 fail-fast 门禁 |
| `check` | [layout.py](check/layout.py)、[policy.py](check/policy.py) | 目录与依赖边界、文档链接、lint/feature 与闲置依赖规则 |
| `check` | [clean-artifacts.py](check/clean-artifacts.py) | 删除可重建输出，保留文档引用的证据与模型资产 |
| `check` | [metal.py](check/metal.py) | 真实 Metal 的 CLI、Agent、HTTP 与 KV 验收 |
| `validation` | [smoke-host.py](validation/smoke-host.py) | 所选 backend 的 CLI、Agent、HTTP/SSE 与观测检查 |
| `validation` | [smoke-paged-kv.py](validation/smoke-paged-kv.py) | 页池压力、抢占、checkpoint、golden 与所有权排空 |
| `validation` | [cpu-open-loop.py](validation/cpu-open-loop.py) | 固定到达 HTTP 负载、控制延迟与显式容量拒绝 |
| `validation` | [openai-decisions.py](validation/openai-decisions.py) | `/v1/decisions` 已知答案准确率、协议形状、批量问题与生成并存 |
| `fixtures` | [export-qwen-golden.py](fixtures/export-qwen-golden.py)、[export-text-golden.py](fixtures/export-text-golden.py) | 独立参考的模型与文本 golden 导出 |
| `fixtures` | [export-moe-golden.py](fixtures/export-moe-golden.py)、[export-moe-operators.py](fixtures/export-moe-operators.py) | [MoE 单卡验证](../docs/guides/moe-validation.md)的模型与 FP8 独立参考 |
| `fixtures` | [requirements.txt](fixtures/requirements.txt) | 参考导出环境的固定依赖 |
| `bench/containers` | [README.md](bench/containers/README.md) | 排序索引、摘要查找与实际 PrefixCache 的候选容器对比，独立依赖图 |
| `bench/cpu` | [Cargo.toml](bench/cpu/Cargo.toml) | 隔离的 release 分配计数与协议测量，通过 `make check-cpu` 运行 |
| `bench` | [cuda-baseline.sh](bench/cuda-baseline.sh) | CUDA 投影基线采集，用法见 [CUDA 性能指南](../docs/guides/cuda-performance.md) |
| `bench` / `attention` | [attention.sh](bench/attention.sh) | 原生 attention 与 Candle 的独立数值、性能门禁 |
| `bench` | [check-cuda-service.py](bench/check-cuda-service.py)、[mtp-ab.py](bench/mtp-ab.py) | CUDA 服务与 MTP A/B 验证；参数见 `--help` |
| `bench` | [serve-workloads.py](bench/serve-workloads.py)、[serve-cases.py](bench/serve-cases.py) | 生成共同 token 输入与选取局部场景；每个用例声明并发 slot，采集端据此校验完整矩阵 |
| `bench` | [experiment-checklist.py](bench/experiment-checklist.py) | 实验清单：`--init` 由 workload 与模型制品生成 profile，`--check` 校验一次运行，`--compare` 输出两边配置对照表 |
| `bench` | [serve-compare.py](bench/serve-compare.py) | 服务采集：记录 release 构建身份、模型制品指纹、生效配置回读、逐请求计时与逐轮 GPU/进程遥测；每次运行用唯一 `--run-id` 且拒绝覆盖。必须在 [safe-run.sh](bench/safe-run.sh) 内运行，否则拒绝采集 |
| `bench` | [freeze-baseline.py](bench/freeze-baseline.py) | 把已过门禁的配对冻结成不可覆盖的基线目录，可用 `--verify` 复核哈希 |
| `bench` | [compare-results.py](bench/compare-results.py) | 报告门禁：先校验身份、矩阵、缓存复用与计时可用性，再分别给出性能与数值结论；debug 构建、未生效开关和不完整矩阵判为无效而非通过 |
| `bench` | [process-pair-gate.py](bench/process-pair-gate.py) | 按 profile 核验固定数量（至少五对）与预声明顺序的 fresh native 进程对，重算主指标改善与守护退化的配对区间 |
| `bench` | [fixed-work-serving.py](bench/fixed-work-serving.py)、[graph-profile-summary.py](bench/graph-profile-summary.py) | 单进程固定输出量计时（零参数或指定参数启动，记录 token 摘要证明输出一致）与逐节点 GPU 事件汇总，用法见[新硬件与新模型的调优流程](../docs/guides/tuning-new-hardware-and-models.md) |
| `bench` | [hardware-monitor.py](bench/hardware-monitor.py) | 保存 NVIDIA/主机遥测，供独立核查实验环境 |
| `bench` | [safe-run.sh](bench/safe-run.sh) | GPU 验证任务的内存限额与互斥保护 |
| `vision` | [图像指南](../docs/guides/vision.md) | 官方参考导出、预处理、视觉塔与端到端 parity |
| `package` | [package.py](package/package.py) | 根 build.rs 契约、Zig 编译、归档与验收，见 [打包指南](../docs/guides/packaging.md) |

性能采集必须在内存限额与互斥保护内进行，遥测工具也会拒绝其他环境：

```sh
bash tools/bench/safe-run.sh python3 tools/bench/serve-compare.py --engine native ... \
    --checklist benchmarks/profiles/27b-mtp0.json --run-id <唯一 ID>
```

服务的性能门禁是 **native/vLLM ≤ 1.10×**（`compare-results.py` 的 ratio 方向为「被测引擎 / 参考引擎」，
跨引擎时必须显式给出参考侧；同引擎配对按候选相对基线解释）。

服务检查依赖已构建的 CLI，参数通过各脚本的 `--help` 查看。参考模型导出示例见[微型模型说明](../examples/qwen-hybrid-tiny/README.md)；CPU 基准的计数范围见 [CPU 性能测量](../docs/guides/cpu-performance.md)。

```sh
python3 tools/check/clean-artifacts.py --dry-run   # 预览将被删除的生成输出
make clean-artifacts                              # 实际删除
```

清理会移除历史验收输出、调试日志与可重建的 golden Python 环境，但保留文档链接的证据及其 summary 日志、顶层模型文件与模型 / tokenizer 包。参考导出环境按样例说明重新创建；普通 Rust / Metal 验证直接使用已提交的 golden。

正式基线不能依赖清理脚本对文档链接的保留策略；清理前须确认已按基线方案保存完整证据与内容哈希。
