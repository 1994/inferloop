use super::weight_roles;
use infer_core::{ModelId, Result};
use infer_ir::TensorStorage;
use infer_models::QwenProvider;
use infer_spi::ModelProvider;

#[test]
fn tied_embedding_is_also_an_output_projection() -> Result<()> {
    let mut config: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../examples/qwen-hybrid-tiny/config.json"
    ))
    .map_err(|error| infer_core::Error::invalid(error.to_string()))?;
    config["tie_word_embeddings"] = true.into();
    let provider = QwenProvider;
    let model = provider.import(
        ModelId::ONE,
        &serde_json::to_vec(&config)
            .map_err(|error| infer_core::Error::invalid(error.to_string()))?,
    )?;
    let graph = provider.graph(&model.model)?;
    let embedding = graph
        .tensors
        .iter()
        .find(|tensor| matches!(&tensor.storage, TensorStorage::Weight { slot } if slot == "embed_tokens.weight"))
        .ok_or_else(|| infer_core::Error::invalid("embedding missing from graph"))?;
    assert_eq!(
        weight_roles(&graph, embedding.id, "embed_tokens.weight"),
        (true, true)
    );
    Ok(())
}
