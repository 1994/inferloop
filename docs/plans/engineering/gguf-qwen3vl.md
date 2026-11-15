# Qwen3-VL GGUF 接入

状态：**配置、张量映射与形状/类型校验已实现**（`infer-models` 的 `gguf` feature）。真实制品 310/310 文本张量、316/316 视觉张量均有映射并通过校验。**不走离线转换**：下一步直接接原生 GGUF 包源、Q4_K/Q6_K 块加载和 GGML kernel；vision 执行与 deepstack 仍未实现。

## 验证制品

ModelScope `Qwen/Qwen3-VL-2B-Instruct-GGUF`：

| 文件 | 大小 | 架构 | 张量 | 类型 |
|---|---:|---|---:|---|
| `Qwen3VL-2B-Instruct-Q4_K_M.gguf` | 1.11 GB | `qwen3vl` | 310 | Q4_K 168、Q6_K 29、F32 113 |
| `mmproj-Qwen3VL-2B-Instruct-F16.gguf` | 819 MB | `clip` | 316 | F16 106、F32 210 |

文本模型：28 层、hidden 2048、16/8 heads、key/value length 128、FFN 6144、vocab 151936、context 262144、mRoPE sections `[24,20,20,0]`、`n_deepstack_layers=3`。

视觉塔：24 层、hidden 1024、FFN 4096、16 heads、image 768、patch 16、spatial merge 2、projector `qwen3vl_merger`、GELU、learned position grid 48×48（2304）、deepstack indexes `[5,11,17]`。

## 文本映射

| GGUF | HF / 现有模型 |
|---|---|
| `token_embd.weight` | `model.embed_tokens.weight`（tied） |
| `output_norm.weight` | `model.norm.weight` |
| `blk.N.attn_norm.weight` | `model.layers.N.input_layernorm.weight` |
| `blk.N.attn_q/k/v.weight` | `model.layers.N.self_attn.q/k/v_proj.weight` |
| `blk.N.attn_q_norm/k_norm.weight` | `q_norm`/`k_norm`（head_dim 128） |
| `blk.N.attn_output.weight` | `o_proj.weight` |
| `blk.N.ffn_norm.weight` | `post_attention_layernorm.weight` |
| `blk.N.ffn_gate/up/down.weight` | `mlp.gate/up/down_proj.weight` |

`ggml` 的 dims 反转后与 HF 的 `[out, in]` 一致；不需要转置权重，只需要 name 映射。Q4_K_M 的 `token_embd` 是 Q6_K，`attn_v`/`ffn_down` 是 Q6_K，其余投影 Q4_K。

## 视觉映射

现有 `VisionSlot` 约定使用 `model.visual.` 前缀和 HF 名字；mmproj 的张量名需要转换：

| mmproj | 现有 vision slot |
|---|---|
| `v.blk.N.attn_qkv.weight/bias` | `blocks.N.attn.qkv.weight/bias` |
| `v.blk.N.attn_out.weight/bias` | `blocks.N.attn.proj.weight/bias` |
| `v.blk.N.ffn_up.weight/bias` | `blocks.N.mlp.linear_fc1.weight/bias` |
| `v.blk.N.ffn_down.weight/bias` | `blocks.N.mlp.linear_fc2.weight/bias` |
| `v.blk.N.ln1/ln2.*` | `blocks.N.norm1/norm2.*` |
| `v.post_ln.*` | `merger.norm.*` |
| `mm.0.*` | `merger.linear_fc1.*` |
| `mm.2.*` | `merger.linear_fc2.*` |
| `v.position_embd.weight [1024,2304]` | `pos_embed.weight`，需转成 `[2304,1024]` |
| `v.patch_embd.weight` + `.weight.1` | Conv3D 的两个 temporal slice |
| `v.deepstack.{5,11,17}.{norm,fc1,fc2}` | 尚无 slot |

## 三个实现缺口

1. **Patch embed 的 temporal 维被拆开。** HF 是 Conv3D `[hidden, in, 2, 16, 16]`，GGUF 存成两个 4D Conv2D `[hidden, in, 16, 16]`（`weight`、`weight.1`）。适配层要按 `t=0/1` 合并成 5D，或让 vision kernel 接受两个 slice 并求和。不能只读 `weight`，否则会少一半时间信息。
2. **Deepstack 没有进入现有 inventory。** 现有 `slots()` 只生成 12×depth+9 个张量，mmproj 多出的 19 个张量主要是 3 个 deepstack merger（每个 norm+fc1+fc2 共 6 个）和第二个 patch slice。文本侧还需要在解码前 3 层把对应 deepstack 输出加到图像 token 上；目前 `ModalityEncoder` 和文本层都没有这个数据流。
3. **F16 视觉权重与量化文本权重共存。** 文本 Q4_K/Q6_K 走 GGML kernel；视觉塔是 F16/F32，继续走现有 BF16/F16 dense 路径。两者在同一个虚拟包内需要不同的 weight source 和 kernel family，不能把 vision 权重也送进 Q4_K GEMV。

## 接入顺序

1. **配置合成**：从 `qwen3vl.*` 和 `clip.*` metadata 生成现有 `ImportedModel`/`ModalityEncoder`，文本字段与 HF `config.json` 对齐；`image_token_id` 从 tokenizer 字典解析（当前制品为 151655）。
2. **文本-only 路径**：`qwen3vl` 的 name map + Q4_K/Q6_K 权重，先跑通 greedy 文本生成，和 HF 参考对齐。
3. **视觉塔**：mmproj name map，patch embed 双 slice 合并，learned pos grid 插值；用 `qwen3vl_vision_reference.py` 或 HF 前向对 hidden/merger 输出做逐张量对照。
4. **Deepstack**：为 `ModalityEncoder` 增加 deepstack merger 描述，vision 输出多组特征；文本前 3 层按 image mask 注入。没有这一步只能算“能看文本”，不是完整 Qwen3-VL。
5. **多模态端到端**：图像预处理（768、mean/std 0.5）、prompt 展开、image token 对齐、greedy 输出；先做单图，再做视频/多图。

## 验收

- 两个 GGUF 都能被 `infer-gguf` 解析；张量清单与本文表一致。
- 文本层与 HF 的 logits 在定义容差内一致；Q4_K 路径用参考解码校验。
- 视觉塔逐阶段（patch、每层、merger）与参考一致；deepstack 三组特征逐元素一致。
- 端到端单图 prompt 的 greedy token 与 HF/llama.cpp 对齐；显存和首 token 延迟单独报告。
- 在这些通过前，不把 VL 路径加入默认 CLI/服务入口。
