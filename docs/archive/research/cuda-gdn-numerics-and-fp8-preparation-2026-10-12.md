# GDN 分块数值与 FP8 重复准备成本（2026-10-12）

重复 FP8 输入准备是可消除的小成本；当前 GDN 分块仍未通过全模型数值门槛。不能把孤立 kernel 的末位差异当成整个量化模型的误差范围。

## 版本与条件

45 直接执行 `git pull --ff-only upstream main`，vLLM 工作区干净，当前官方 main 是 `f26d9f432ced7b1087d02d29bffed2249df1a18e`。实际安装/历史实测仍为 vLLM 0.31.0，本轮没有登记最新 main 的性能结果。该提交的 cu130/cu129 精确 wheel 元数据均返回 404；源码分析与运行版本继续分开记录。

native 生产源码基点为 `0f2003fa3220f8100490bba0cea4515f58256ac2`，其 quality CI 已全部成功。验证机为 45 的 RTX 5090；本轮修复涉及测试对照、CUDA 示例/测试的严格 lint 整理和注释，没有按 GPU 名称选择参数。48 层 GDN、hidden 5120，checkpoint 混合 FP8/NVFP4/BF16，activation/recurrent state 保持 F32。

构建、微基准和模型诊断均使用 safe-run，内存上限分别为 8/16/24 GiB，swap=0，并发构建=1。GPU 只有保留的 KWin 进程，没有停止外来进程。模型对照使用冻结 tuning cache `d212f19361deeb8adf7cd321a268080d047920bcc5fa8973a251abeb404bcb7f`，运行后未改变。

## 重复 FP8 量化的成本

最新 vLLM 的 [GDN projection](https://github.com/vllm-project/vllm/blob/f26d9f432ced7b1087d02d29bffed2249df1a18e/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L651)合并 Q/K/V/Z 和 B/A。native QKV/Z 是独立 Linear，共享逻辑输入但各自记录 FP8 quantize。

探针使用生产 FP8 quantize kernel，K=5120，比较 96 次与 48 次量化的 CUDA graph。两边 packed code 与 scale 相同，每种 shape 测 warm/cold 两种条件。没有 GEMM 或服务请求，因此差值只估计可删除节点的成本，不能登记为端到端加速。

| M | warm 少 48 次量化 | cold 少 48 次量化 |
|---|---:|---:|
| 1 | 0.275120 ms | 0.275456 ms |
| 12 | 0.278720 ms | 0.278752 ms |
| 64 | 0.280832 ms | 0.280736 ms |
| 256 | 0.572672 ms | 0.572992 ms |

M=3、4 也测得约 0.277/0.276 ms。历史 graph-only 诊断的 target verify 中位数约 20.84 ms，0.28 ms 约为其 1.3%，只用于判断投入顺序，不是当前生产 TPOT 的收益预测。删除重复量化不能单独覆盖跨引擎差距；projection 合并还需另测 pack、GEMM 和输出切分成本。

## 全模型检查的两个对照缺口

[checkpoint model check](../../../crates/backend/cuda/src/resident/recurrent_prefill/model_check_tests.rs)原先没有显式设置 chunked_recurrent。量化模型默认使用 scalar Delta，不设置环境 opt-in 时，名为 chunk 的检查可能比较两个 scalar 程序，产生假阳性。

LEGACY_CAPTURE 还同时关闭卷积分块，原检查没有单独隔离 Delta。修复后明确启用候选，并用测试专用 LEGACY_DELTA_CAPTURE 保持卷积、权重、宽度和 GEMM policy 相同。其他 legacy 卷积检查继续使用原开关；生产默认路径与数值门槛不变。

## 保持卷积相同后，Delta 分块仍有明显模型误差

固定输入为既有 511-token prompt，prefill 宽度 64，之后做宽度 3 验证、commit 1 和 16 个固定 token 的继续 decode。直接比较同权重程序的状态、hidden 和 logits，避免生成路径变化混入算法耗时。

第一层卷积的三个状态数组逐位相同；第一层 Delta 状态 relative L2 为 1.5636e-7，最大绝对差 7.6294e-6，428355/786432 个元素位模式不同。首个分块返回的第一行 hidden relative L2 为 4.3642%，第一个 logits relative L2 为 26.2506%。误差最早出现在 Delta；量化和后续层可能放大它，尚未单独证明放大的具体机制，也未证明初始差异一定来自 FMA 合并。

8 个 prompt 末位、3 个验证、16 个继续 decode，共 27 个 logits 向量中，1 个 argmax 改变；reference-to-candidate KL 最大 0.78569，total variation 最大 0.45155。这不是任务质量集，但已经不支持以“只是末位误差”放行或放宽现有阈值。

同路径 sanity control 使用同一个诊断 executable、两个 scalar Delta 程序和相同卷积：192 个状态数组没有位差，557 个 hidden/logits 比较的数值误差为 0，KL 为 0。排除了两个独立程序本身必然漂移的解释。诊断完成不等于数值验收通过；生产 opt-in/default 均未扩大。

## 取舍与下一步

保留精确重复工作消除，区分 serial 和 pooled 后测推测验证中的 checkpoint 与 launch 成本，以及 QKV/Z、B/A 融合收益。pooled 的 record_row_checkpoint 已融合状态 writer 和快照；serial batch 仍记录独立 memcpy，不能把已有 pooled 融合当成新候选。checkpoint 融合须证明 padding、状态更新、验证/提交/回滚和逐位输出一致，不能删除回滚契约。

长 prefill 的真正分块 GDN 仍有更大收益空间，但必须定位求值差异并通过独立参考及全模型检查。新硬件参数、kernel 选择和 workspace/graph 几何继续接入 tuning，按设备能力、shape、精度和预算选择；本次测量数字不成为硬编码阈值。完整最新 vLLM main 服务矩阵仍是剩余工作，目标没有完成。

## 修复验证

严格 CUDA all-targets Clippy、格式、目录/策略/test inventory 检查通过；常规 CUDA 单元检查 47 项通过，受影响 GPU 检查 14 项通过。修正后的全模型检查在没有 chunk 环境开关时明确启用候选，仍以原阈值拒绝 4.36% 的首行 hidden 误差；这是预期拒绝的回归验证，不是分块质量验收通过。

## 原始证据

45 上本轮目录：

- `/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles/fp8-input-preparation-cost-v1`：探针源码、编译身份、12 个 GPU 测量、原始日志、精确 main wheel 核验、上一提交 CI。
- `/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles/gdn-full-model-numerics-v1`：原检查失败，diagnostic 下修正后的逐层状态/分布误差，以及 legacy-control 同路径对照。隔离诊断只记录误差，没有改变生产 gate。

- FP8 原始 GPU 日志 SHA256：`b8256321cc16359f7481179723debe33b3b91f994b7f6fb3ea830c3ac4f21acc`。
- 仅 Delta 对照日志 SHA256：`52e8c0f03c3b5ed4b9d9558628dbffb5f73b77eac8e17bd5b3a85b930b41a46b`。
- 同路径对照日志 SHA256：`dc65371841d18633425b31aa564fb5d0cf173ecc49c493aa6320219f1425cf1c`。
