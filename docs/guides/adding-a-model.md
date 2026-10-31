# 新增模型（ModelProvider 接入指南）

本文说明如何在不修改核心分派的前提下接入一个新的模型家族。对应 vLLM 的 [Adding a Model](https://docs.vllm.ai/en/v0.6.5/models/adding_model.html)：vLLM 用 `@register_model` 把模型类登记进 `ModelRegistry`，再按 `config.architectures` 解析；这里用 `infer_spi::ModelProvider` 加 `infer_models::ModelRegistry` 实现同一件事，差别是 Rust 的注册必须在编译期由组合根显式完成。

相关文件：

| 内容 | 位置 |
|---|---|
| 扩展契约 | [crates/foundation/spi/src/model.rs](../../crates/foundation/spi/src/model.rs) |
| 注册表 | [crates/model/package/src/providers/registry.rs](../../crates/model/package/src/providers/registry.rs) |
| 参考实现（Qwen） | [crates/model/package/src/providers/qwen.rs](../../crates/model/package/src/providers/qwen.rs) |
| 包与权重绑定 | [crates/model/package/src/storage/package.rs](../../crates/model/package/src/storage/package.rs) |

## 1. 契约

`ModelProvider` 需要提供以下信息：

| 方法 | 作用 |
|---|---|
| `metadata()` | provider 身份、名字与 SPI 版本；注册时校验 |
| `architectures()` | 认领的 HF `architectures` 或 `model_type` 字符串 |
| `import(id, config)` | 把 `config.json` 导入为 `ImportedModel`——家族配置与能力声明 |
| `graph(model)` | 生成模型的执行图：算子顺序、连接、权重槽与状态访问 |
| `draft_graph(model)` | 可选的草稿模型及其执行图；默认明确返回 unsupported |
| 权重命名 | `weight_prefixes()` / `anchor_slot()` / `weight_source()`，默认值适用于标准 `model.` 前缀的 decoder |

`ImportedModel` 的字段就是定制清单，默认值即"无定制"：

| 字段 | 表达什么 | 谁消费 |
|---|---|---|
| `model: ModelIr` | 语义：mixer、state、位置编码、模态列表 | compiler、所有 backend |
| `draft_layers` | 草稿层数（0 关闭推测） | 加载期校验 |
| `requirements: CapabilityRequirements` | 需要的设备能力（compute dtype、CUDA graphs/TMA/clusters） | `DeviceCapabilities::require` |
| `precision: PrecisionPolicy` | 存储/计算策略，如 `Preferred([nvfp4, bf16])` | 加载器按 `supports_compute` 解析 |
| `speculation: Option<DraftHead>` | 草稿头前缀、层数、融合投影与 norm 槽位 | CUDA draft 绑定 |
| `modalities: Vec<ModalityPlan>` | 非文本模态与占位 token | 上报与门禁 |

provider 必须 `Send + Sync`：注册表是进程级共享的。`graph` 未实现时加载失败，不会自动使用默认 decoder。模型配方不依赖后端；编译器只接收已生成的图，runtime 通过 `BackendProvider::execution_graph` 取得绑定图。后端需要支持图中的算子和状态语义，否则在加载或编译期明确拒绝。

## 2. 实现 provider

以一个新家族 `acme` 为例：

```rust
use infer_core::{Error, ModelId, ProviderId, Result};
use infer_ir::{BackboneKind, FeedForward, Head, Mixer, ModelIr, Modality, PositionSpec};
use infer_spi::{ImportedModel, ModelProvider, ProviderMetadata, SPI_VERSION};
use serde::Deserialize;

pub struct AcmeProvider;

#[derive(Deserialize)]
struct AcmeConfig {
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
}

impl ModelProvider for AcmeProvider {
    fn graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        // 仅在该家族确实采用这份 decoder 配方时复用；其他拓扑在本 provider 中构造。
        infer_model_recipes::decoder::lower(model)
    }

    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            id: ProviderId::new(2).unwrap_or(ProviderId::ONE),
            name: "acme-hf-config".into(),
            spi_version: SPI_VERSION,
        }
    }

    fn architectures(&self) -> &'static [&'static str] {
        // HF `architectures` 与 `model_type` 都要列，配置里出现哪一个都能命中。
        &["acme", "AcmeForCausalLM"]
    }

    fn import(&self, id: ModelId, config: &[u8]) -> Result<ImportedModel> {
        let config: AcmeConfig =
            serde_json::from_slice(config).map_err(|e| Error::invalid(e.to_string()))?;
        let hidden = config.hidden_size;
        let head_dim = 128;
        let mixers = (0..config.num_hidden_layers)
            .map(|_| Mixer::Attention {
                query_heads: hidden / head_dim,
                kv_heads: hidden / head_dim,
                head_dim,
                sliding_window: None,
                output_gate: false,
                qk_norm: true,
            })
            .collect();
        let model = ModelIr {
            id,
            backbone: BackboneKind::Decoder,
            vocab_size: config.vocab_size,
            hidden_size: hidden,
            max_sequence: config.max_position_embeddings,
            mixers,
            feed_forward: FeedForward::Dense {
                intermediate: 4 * hidden,
            },
            position: PositionSpec {
                rope_theta: 10_000.0,
                rotary_fraction: 1.0,
                multimodal_sections: vec![],
                interleaved: false,
            },
            norm_epsilon: config.rms_norm_eps,
            norm_weight_offset: 0.0,
            heads: vec![Head::LanguageModel],
            modalities: vec![Modality::Text],
            state: vec![],
            tied_embeddings: false,
        };
        // 维度、范围与 owner 由 IR 统一校验，不要在这里另写一套。
        model.validate()?;
        Ok(ImportedModel::new(model, 0))
    }
}
```

Qwen 的 `QwenProvider` 是完整参考：它解析 `text_config`、hybrid `layer_types`、RoPE 与 vision，并覆盖 `weight_prefixes()`（`model.language_model.`）和 `weight_source()`（`lm_head.weight` 不在骨干前缀下）。

## 3. 权重命名

通用包层不假设任何 HF 命名：

- `weight_prefixes()`：候选前缀，从最具体到最宽；`anchor_slot()`（默认 `embed_tokens.weight`）在哪个前缀下存在，就用哪个前缀。
- `weight_source(slot, prefix)`：把图里的 canonical slot 变成 HF 张量名；默认 `format!("{prefix}{slot}")`。

共享 head、`lm_head` 根位置、`language_model` 嵌套都在这里表达，而不是回到 `bind_weights`。草稿头（MTP）不在命名钩子里，而是 `ImportedModel.speculation` 的 `prefix` + `FusionPlan{projection, norms}`：CUDA 侧据此拼接槽位，不再硬编码 `mtp.fc.weight`。

## 3.5 一份声明：以 omni 家族为例

接入一个 omni 模型时，唯一要改的地方就是它的 `import`：

```rust
fn import(&self, id: ModelId, config: &[u8]) -> Result<ImportedModel> {
    let config: OmniConfig = serde_json::from_slice(config)?;
    let mut imported = ImportedModel::new(omni_ir(id, &config)?, config.mtp_layers);
    imported.requirements = CapabilityRequirements {
        compute_dtypes: vec![DType::Bf16],
        ..Default::default()
    };
    // Blackwell 用原生 FP4，其他卡解码成 BF16；模型只声明偏好，不认架构名。
    imported.precision = PrecisionPolicy::Preferred(vec![
        PrecisionPlan::nvfp4_block(),
        PrecisionPlan::bf16(),
    ]);
    imported.speculation = (config.mtp_layers > 0).then(|| DraftHead {
        prefix: "mtp.".into(),
        layers: config.mtp_layers,
        fusion: Some(FusionPlan {
            projection: "fc.weight".into(),
            norms: ["pre_fc_norm_embedding.weight".into(), "pre_fc_norm_hidden.weight".into()],
        }),
    });
    // 有 audio_token_id 就多一个模态，引擎不需要认识"omni"这个词。
    imported.modalities = modality_placeholders(&config);
    Ok(imported)
}
```

`infer-models` 的注册表测试 `an_omni_family_converges_every_customization_into_one_import` 就是这条路径：注册 → 按 `architectures` 解析 → 断言模态、草稿头、能力需求与精度解析，全程没有改核心代码。

## 4. 注册

注册在组合根完成。内置列表是 `ModelRegistry::builtin()`：

```rust
use std::sync::Arc;
use infer_core::ModelId;
use infer_models::{ModelPackage, ModelRegistry, QuantizedPackage};

let mut registry = ModelRegistry::builtin();
registry.register(Arc::new(AcmeProvider))?;
let package = ModelPackage::open_with(&registry, root, ModelId::ONE)?;
let quantized = QuantizedPackage::open_with(&registry, root, ModelId::ONE)?;
```

`ModelPackage::open` / `QuantizedPackage::open` 走 `default_registry()`，因此新增家族需要走 `open_with`（或把 provider 并入内置列表）。注册失败的两类原因：

- `InvalidInput`：重复认领同一架构，或 provider 声明了空架构列表；
- `Unsupported`：配置里没有任何已注册架构，报错会列出解析到的架构名并指向 `infer_spi::ModelProvider`。

## 5. 验证

1. **导入单测**（不需要权重）：用 `examples/` 下的 `config.json` 调 `import()`，断言层数、`hidden_size`、`state` 与 `modalities`。
2. **注册表测试**：在 [registry.rs](../../crates/model/package/src/providers/registry.rs) 的测试模块里覆盖认领、嵌套 `text_config`、重复注册与未知架构。
3. **元数据预检**：
   ```sh
   cargo run -p infer-cli -- inspect-model --config path/to/config.json --index path/to/model.safetensors.index.json
   cargo run -p infer-cli -- inspect-package --package path/to/package
   ```
   两个命令现在都先经注册表解析，未注册的家族在读取权重之前就会失败。
4. **真机加载**：按 [模型执行](model-execution.md) 与对应 backend 的验证流程运行；CUDA 端使用 `cuda-loaded-model-check` / `cuda-provider-check`，见 [CUDA 后端说明](../../crates/backend/cuda/README.md)。

## 6. 边界

- 只认领你能导入的架构；`architectures()` 是静态字符串列表，没有通配。重叠由注册期拒绝，而不是运行期随机选择。
- 配置字段一律按可选处理：导出器常把未设置字段写成 `null`（注册表的 hints 解析已如此）。
- provider 不拥有设备内存与 kernel；它只产出 IR、声明与命名映射。设备分配仍在 backend，编译仍在 `infer-compiler`。
- `modalities` 既用于上报与门禁，也驱动真实执行：CUDA 后端按声明绑定视觉塔（`LoadedModel::vision`），图像经预处理、塔编码与占位符合并进入 resident 路径。未声明的模态、缺失的占位符、占位符与编码数量/位置不匹配、编码宽度不符都会返回 `Unsupported`/`InvalidInput`，不会静默按纯文本运行；视频复用同一视觉塔，由 `VideoProcessor` 完成抽帧与时间 patch；音频编码器仍是缺口。
- 动态加载（C ABI/WASM）与从独立 crate 分发模型属于后续工作，见 [ADR-0003](../adr/0003-model-providers.md)、[能力边界](../architecture/backends.md)。
