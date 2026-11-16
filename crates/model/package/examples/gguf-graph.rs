//! Dump the native GGUF text graph for layer-boundary analysis.
use infer_core::ModelId;
use infer_models::{GgufPackage, default_registry};
use std::env::args;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let path = args().nth(1).ok_or("usage: gguf-graph <gguf>")?;
    let package = GgufPackage::open_auto(path)?;
    let graph = {
        let text = package.config();
        let provider = default_registry().resolve(text)?;
        let imported = provider.import(ModelId::ONE, text)?;
        provider.graph(&imported.model)?
    };
    for node in &graph.nodes {
        let tensor = graph
            .tensors
            .iter()
            .find(|tensor| tensor.id == node.outputs[0])
            .ok_or("missing tensor")?;
        let storage = match &tensor.storage {
            infer_ir::TensorStorage::Weight { slot } => format!("weight:{slot}"),
            infer_ir::TensorStorage::Activation => "activation".into(),
            infer_ir::TensorStorage::State { layer, .. } => format!("state:{layer}"),
        };
        println!(
            "id={} op={:?} inputs={:?} storage={}",
            node.id.get(),
            node.op,
            node.inputs.iter().map(|id| id.get()).collect::<Vec<_>>(),
            storage
        );
    }
    Ok(())
}
