# GGUF 后续工作

状态：**跟踪项**。PR #266 已合入 native GGUF 文本、视觉、多模态和 Q4_K/Q6_K GEMV；本文件记录尚未进入生产路径的遗留项，后续 PR 按项收口。

## 1. int8 MMQ 接入 prefill

已完成：

- `q4k_int8_two_n64`：两层 int8（主项 + int8 残差），64 output tile；
- 真实 `gate_proj` 全 6144 输出最大绝对误差 `1.18e-4`；
- 合成 6144×2048 带宽 333 GB/s、0.047 ms/launch；
- 单 group、code 解包、int8 MMA 均有独立校验。

未完成：

- 激活量化仍在测试 host 侧，需 device 端主项/残差量化 kernel；
- 尚未接 `RowsKernel`/`Projection::rows`，prefill 仍走逐行 grouped-int GEMV；
- 需要完整 grid 的端到端 16-token prefill 与 llama.cpp pp16 对比；
- Q6_K 尚无 int8 MMQ。

## 2. deepstack 文本注入

**已完成（#282）**：deepstack 特征经 `TensorOp::MediaFeature` 加到前 3 个 decoder 层的输出上，safetensors 与 GGUF 共用同一条路径。与 HF Qwen3-VL-2B（含 deepstack）对照：特征相对误差 3.9e-4，logits 最大绝对误差 0.10，greedy token 一致。

未完成：GGUF 包没有 preprocessor 配置，多模态端到端还跑不起来（#283），GGUF 的 deepstack 因此只验证了导入和图结构。

## 3. 质量与性能收口

- safetensors 包的多模态输出已与 HF 做数值配对（#282）；GGUF 多模态先要补 preprocessor（#283），之后再做配对；
- 当前 tokenizer、preprocessor、mmproj 映射有真实制品验证，但没有全量图片集；
- 需要把 native 文本、视觉、多模态的 TTFT/TPOT 与 llama.cpp/参考包分项记录；
- Q4_K GEMV 仍低于带宽 roofline，prompt GEMV 仍是逐行路径。

## 4. 发布与兼容

- GGUF 入口由 CLI/model 的 `gguf` feature 控制；
- 已通过 `make check-rust`、CUDA hosted、macOS、MSRV、security、package 和 quality-gate；
- 后续修改需要继续跑对应 CUDA device 测试和真实模型冒烟。
