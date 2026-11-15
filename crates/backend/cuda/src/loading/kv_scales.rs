//! Qwen-specific source names are resolved at load time, outside device execution.
use infer_core::{Error, Result};
use infer_ir::TensorOp;
use infer_models::QuantizedPackage;
use std::collections::BTreeMap;

/// Bind checkpoint-calibrated static KV scales to graph state IDs.
/// # Errors
/// Rejects missing, non-scalar, nonfinite or nonpositive scales.
pub fn load(package: &mut QuantizedPackage) -> Result<BTreeMap<infer_core::TensorId, [f32; 2]>> {
    let mut result = BTreeMap::new();
    for node in package.graph.nodes.clone() {
        if !matches!(node.op, TensorOp::Attention { .. }) {
            continue;
        }
        let layer = node
            .layer
            .ok_or_else(|| Error::invalid("attention layer binding"))?;
        let query = package
            .weights
            .get(&format!("layers.{layer}.self_attn.q_proj.weight"))
            .ok_or_else(|| Error::invalid("query binding"))?;
        let prefix = query
            .data()
            .name
            .rsplit_once("q_proj.")
            .ok_or_else(|| Error::invalid("query source name"))?
            .0
            .to_owned();
        let sources = [
            package.source(&format!("{prefix}k_scale"))?,
            package.source(&format!("{prefix}v_scale"))?,
        ];
        let mut scales = [0.0; 2];
        for (i, source) in sources.iter().enumerate() {
            let values = super::floats(package, source)?;
            if values.len() != 1 || !values[0].is_finite() || values[0] <= 0.0 {
                return Err(Error::invalid("invalid static KV scale"));
            }
            scales[i] = values[0];
        }
        result.insert(node.states[0], scales);
    }
    Ok(result)
}
