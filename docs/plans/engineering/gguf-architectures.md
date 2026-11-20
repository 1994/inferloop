# GGUF 架构适配层

状态：**已实现**（`qwen3vl`、`qwen35`、`dflash`）。GGUF 只是另一种存储格式：适配层把一个 `general.architecture`
翻译成普通的 HF 配置和规范张量名，之后的 provider、recipe、编译、backend、MTP 与 serving 与
safetensors 完全相同。新增一个架构是填一个适配器，不改加载、绑定或执行路径。

## 分层

| 归属 | 内容 | 位置 |
|---|---|---|
| 平台 | 容器解析、grouped-int 编码、tokenizer（按 `tokenizer.ggml.pre` 选切分规则）、mmproj 发现 | `infer-format`、`storage::gguf_tokenizer`、`GgufPackage` |
| 平台 | llama.cpp 共有的 block 字段名（`attn_q`、`ffn_gate`……） | `gguf_arch::common_block_field` |
| 平台 | 尾部 `nextn` block 拆成草稿头：`{arch}.nextn_predict_layers` 个尾 block 不属于主干 | `gguf_arch::LayerPlan` |
| 平台 | 值变换的逆运算（`1 + w` 的 norm、`-exp(A_log)`），在解码 float 张量时做 | `gguf_arch::ValueTransform` |
| 平台 | 每个存储张量必须有去处；未知张量、重复名、对 grouped-int 张量做值变换都显式报错 | `GgufPackage::open` |
| 平台 | 草稿头按规范名绑定（`imported.speculation.prefix`），与格式无关 | `QuantizedPackage::bind_gguf` |
| 适配器 | 合成 HF 配置；本架构特有的张量名、变换与形状；架构级校验 | `impl GgufArchitecture` |

适配器的全部接口：

```rust
pub trait GgufArchitecture: Sync {
    fn name(&self) -> &'static str;                       // general.architecture
    fn config(&self, text: &Metadata, tensors: &[TensorSpec], vision: Option<&Metadata>)
        -> Result<Vec<u8>>;                               // HF config.json
    fn tensor(&self, tensor: &TensorSpec, layers: &LayerPlan) -> Option<MappedTensor>;
    fn validate(&self, text: &Metadata, tensors: &[TensorSpec]) -> Result<()> { Ok(()) }
    fn accepts_vision(&self) -> bool { false }
}
```

`MappedTensor` 是规范名 + `ValueTransform` + 可选的规范形状（例如 `ssm_conv1d` 存为 `[C, K]`，HF 是
`[C, 1, K]`）。未登记的架构报 `Unsupported` 并列出已知架构。

## 新增一个架构

1. 在 `providers/<family>/gguf/` 写适配器，登记到 `gguf_arch::ARCHITECTURES`；
2. `config`：从 `{arch}.*` 元数据合成 provider 已能导入的 HF 配置（层数用 `LayerPlan::backbone`，MTP 层数用
   `LayerPlan::draft`；`output.weight` 缺失即 tie）；
3. `tensor`：先查平台的 `common_block_field`，再映射本架构特有字段，并写明转换器改了什么（看 llama.cpp
   `convert_hf_to_gguf.py` 的 `modify_tensors`）；
4. `validate`：转换器做了加载时无法还原的变换（例如对量化张量重排列）时显式 `Unsupported`；
5. 验证：BF16 GGUF 的每个张量（解码 + 逆变换）与 safetensors 同名张量逐一比对，greedy 输出与 safetensors 包一致，
   再跑量化文件和显式 MTP。

## Qwen3.5（`qwen35`）

转换器的四处改动由适配器还原：除 gated-delta 的 `ssm_norm` 外所有 norm（含 MTP 头的三个）存为 `1 + w`；
`ssm_a = -exp(A_log)`；`ssm_conv1d` 去掉单例轴；MTP 层是尾部 `nextn` block（`eh_proj`、`enorm`、`hnorm`、
`shared_head_norm` 对应 `mtp.fc`、`mtp.pre_fc_norm_embedding`、`mtp.pre_fc_norm_hidden`、`mtp.norm`）。
线性注意力几何来自 `qwen35.ssm.*`（`group_count` = key heads，`time_step_rank` = value heads，`state_size` =
key dim，`inner_size / time_step_rank` = value dim）。

已知限制：

- key heads ≠ value heads（27B 是 16/48）时转换器把 value head 重排成 tiled 顺序，连量化的 `ssm_out` 列也一起
  重排，加载时无法还原，显式 `Unsupported`；需要 delta kernel 直接读该顺序；
- `nextn_predict_layers > 1` 显式 `Unsupported`。

验证（RTX 5090，`/home/r/benchmarks/inference.rs/gguf-mtp-2026-11-21/`）：

| 包 | 3 道 greedy 问答 | MTP 深度 2 |
|---|---|---|
| Qwen3.5-2B safetensors | 391 / Paris / 2,3,5,7,11 | 1.62× |
| Qwen3.5-2B BF16 GGUF | 同上，token 摘要与 safetensors 相同 | 1.61×，摘要不变 |
| Qwen3.5-2B `Q4_K_M` | 391 / Paris / 2,3,5,7,11 | 摘要不变；比不开 MTP 慢，见 [GGUF grouped-int 性能](../performance/gguf-grouped-int.md) P6 |
| Qwen3.5-0.8B `Q8_0` | 无新增代码直接加载；491（小模型自身答错）/ Paris / 2,3,5,7,11 | 显式深度 2 输出与不开 MTP 相同 |

BF16 GGUF 与 safetensors 的全部 335 个张量逐一比对，最大相对误差 0。

## DFlash2 草稿（`dflash`）

`--draft-model` 指向 `.gguf` 文件时，`attach_dflash2` 用 `dflash` 适配器读它，之后的配置校验、权重清单校验、
`fc` 按 tap 切块和 tap 图都与 safetensors 草稿共用。文件里的 `dflash.target_layers` 从嵌入输出起算，比 HF 的
`target_layer_ids` 大 1；文件不记录目标深度，`num_target_layers` 由挂载的目标提供；llama.cpp 把 chat 结束符记为
EOS，与 HF `eos_token_id` 不同，所以草稿不声明 EOS，由目标决定。草稿的 float 张量像 GGUF 目标一样常驻 host
（BF16 1.9B 草稿约 3.8 GB，另加 `fc` 切块 0.26 GB）；量化草稿显式 `Unsupported`，因为草稿的 CUDA 投影只接 float/FP8。

验证（RTX 5090，`/home/r/benchmarks/inference.rs/gguf-dflash2-2026-11-21/`）：`z-lab/Qwen3.8-27B-DFlash2-GGUF` 的
BF16 文件 81 个张量与 safetensors 草稿逐一相同；以 Qwen3.8-27B-NVFP4 为目标、9 个请求各 256 token，GGUF 草稿与
safetensors 草稿的 greedy token 摘要相同，耗时 1.075 s 对 1.074 s。
