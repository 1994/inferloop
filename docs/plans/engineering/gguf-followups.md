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

GGUF 端到端也已验证（#283）：BF16 GGUF + F16 mmproj 的 greedy token 与 HF 参考一致，HTTP 图片请求的输出与 safetensors 包逐字相同。之前多模态"跑不起来"的原因是：`.gguf` 以软链接部署时，`PackageSource` 把路径规范化到链接目标所在目录，找不到旁边的 mmproj，部署被当成纯文本；而加载器按给定路径找，能绑定视觉塔。两者现在都按给定路径查找 mmproj。

## 3. 质量与性能收口

- safetensors 包的多模态输出已与 HF 做数值配对（#282）；GGUF 多模态（BF16）与 HF 的 greedy token 一致（#283）；
- 当前 tokenizer、preprocessor、mmproj 映射有真实制品验证，但没有全量图片集；
- 需要把 native 文本、视觉、多模态的 TTFT/TPOT 与 llama.cpp/参考包分项记录；
- Q4_K GEMV 仍低于带宽 roofline，prompt GEMV 仍是逐行路径。

## 4. 架构适配与 MTP

**已完成**：GGUF 按 `general.architecture` 选适配器，见 [GGUF 架构适配层](gguf-architectures.md)。Qwen3.5（`qwen35`）
的尾部 `nextn` block 作为 MTP 头按规范名绑定，BF16 GGUF 的 MTP 加速与 safetensors 相同。

未完成：

- `qwen35` 的 tiled value head（key heads ≠ value heads，27B）；
- grouped-int 目标的 MTP 验证仍慢于单行 decode（性能文档 P6 余项）；
- `DFlash2` 的 GGUF 草稿（`general.architecture = dflash`）。

## 5. 发布与兼容

- GGUF 入口由 CLI/model 的 `gguf` feature 控制；
- 已通过 `make check-rust`、CUDA hosted、macOS、MSRV、security、package 和 quality-gate；
- 后续修改需要继续跑对应 CUDA device 测试和真实模型冒烟。
