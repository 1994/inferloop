use super::*;
use crate::providers::registry::default_registry;
use infer_ir::{TensorOp, TensorStorage};

fn config(extra: &str) -> Vec<u8> {
    format!(
        r#"{{"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":512,
        "vocab_size":1024,"num_hidden_layers":2,"num_attention_heads":4,
        "num_key_value_heads":2,"intermediate_size":1536,"max_position_embeddings":4096{extra}}}"#
    )
    .into_bytes()
}

#[test]
fn a_llama_configuration_lowers_to_a_decoder_without_query_key_norms() -> Result<()> {
    let bytes = config(r#","rope_theta":500000.0,"tie_word_embeddings":true"#);
    let provider = default_registry().resolve(&bytes)?;
    assert_eq!(provider.metadata().name, "llama-hf-config");
    let imported = provider.import(ModelId::ONE, &bytes)?;
    let model = &imported.model;
    assert_eq!(model.mixers.len(), 2);
    assert!(model.mixers.iter().all(|mixer| *mixer
        == Mixer::Attention {
            query_heads: 4,
            kv_heads: 2,
            head_dim: 128,
            sliding_window: None,
            output_gate: false,
            qk_norm: false,
        }));
    assert!(model.tied_embeddings);
    assert!((model.position.rope_theta - 500_000.0).abs() < f64::EPSILON);
    assert!(imported.speculation.is_none());
    let graph = provider.graph(model)?;
    let attention = graph
        .nodes
        .iter()
        .filter(|node| matches!(node.op, TensorOp::Attention { .. }))
        .count();
    assert_eq!(attention, 2);
    // Without query/key norms the only norms are the per-layer pre-norms and the final one.
    let norms = graph
        .nodes
        .iter()
        .filter(|node| matches!(node.op, TensorOp::Norm { .. }))
        .count();
    assert_eq!(norms, 2 * 2 + 1);
    assert!(graph.tensors.iter().any(|spec| matches!(
        spec.storage,
        TensorStorage::State {
            kind: StateKind::AttentionKv,
            ..
        }
    )));
    Ok(())
}

#[test]
fn heads_default_to_the_hidden_split_without_grouped_query_attention() -> Result<()> {
    let bytes = config("").into_iter().collect::<Vec<_>>();
    let text = String::from_utf8(bytes).map_err(|e| Error::invalid(e.to_string()))?;
    let bytes = text.replace(r#""num_key_value_heads":2,"#, "").into_bytes();
    let imported = LlamaProvider.import(ModelId::ONE, &bytes)?;
    assert!(matches!(
        imported.model.mixers[0],
        Mixer::Attention {
            kv_heads: 4,
            head_dim: 128,
            ..
        }
    ));
    Ok(())
}

#[test]
fn unmodeled_llama_semantics_are_refused_at_import() {
    for extra in [
        r#","rope_scaling":{"rope_type":"llama3","factor":8.0}"#,
        r#","attention_bias":true"#,
        r#","mlp_bias":true"#,
        r#","hidden_act":"gelu""#,
    ] {
        let error = LlamaProvider
            .import(ModelId::ONE, &config(extra))
            .unwrap_err();
        assert_eq!(error.code, infer_core::ErrorCode::Unsupported, "{extra}");
    }
}

#[test]
fn an_untied_head_is_bound_at_the_package_root() -> Result<()> {
    let bytes = config(r#","tie_word_embeddings":false"#);
    let imported = LlamaProvider.import(ModelId::ONE, &bytes)?;
    let graph = LlamaProvider.graph(&imported.model)?;
    let sources: Vec<String> = graph
        .tensors
        .iter()
        .filter_map(|spec| match &spec.storage {
            TensorStorage::Weight { slot } => Some(LlamaProvider.weight_source(slot, "model.")),
            _ => None,
        })
        .collect();
    assert!(sources.iter().any(|name| name == "lm_head.weight"));
    assert!(
        sources
            .iter()
            .any(|name| name == "model.embed_tokens.weight")
    );
    assert!(!sources.iter().any(|name| name == "model.lm_head.weight"));
    Ok(())
}
