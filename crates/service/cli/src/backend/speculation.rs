//! Choose the CLI draft depth from the provider's recipe and complete checkpoint bindings.
use super::SpeculativeTokens;
use infer_ir::{FeedForward, TensorStorage};
use infer_models::QuantizedPackage;

/// An explicit depth keeps the loader's existing validation. Automatic mode must not make a
/// previously loadable target require an absent, partial or unsupported optional head.
pub fn resolve(
    requested: SpeculativeTokens,
    prefill_cap: Option<usize>,
    package: &QuantizedPackage,
) -> usize {
    match requested {
        SpeculativeTokens::Depth(depth) => depth,
        SpeculativeTokens::Auto => {
            if matches!(prefill_cap, Some(1 | 2)) || !complete_head(package) {
                0
            } else {
                2
            }
        }
    }
}

fn complete_head(package: &QuantizedPackage) -> bool {
    let Some(plan) = &package.imported.speculation else {
        return false;
    };
    if plan.layers != 1
        || plan.prefix.is_empty()
        || matches!(package.imported.model.feed_forward, FeedForward::Moe { .. })
    {
        return false;
    }
    let Some(fusion) = &plan.fusion else {
        return false;
    };
    let hidden = package.imported.model.hidden_size;
    let Some(fused_hidden) = hidden.checked_mul(2) else {
        return false;
    };
    for (slot, shape) in [
        (fusion.projection.as_str(), vec![hidden, fused_hidden]),
        (fusion.norms[0].as_str(), vec![hidden]),
        (fusion.norms[1].as_str(), vec![hidden]),
    ] {
        if package
            .mtp
            .get(&format!("{}{slot}", plan.prefix))
            .is_none_or(|weight| weight.shape() != shape)
        {
            return false;
        }
    }
    let Ok((_, graph)) = package.provider.draft_graph(&package.imported.model) else {
        return false;
    };
    graph.tensors.iter().all(|tensor| {
        let TensorStorage::Weight { slot } = &tensor.storage else {
            return true;
        };
        let weight = if matches!(slot.as_str(), "embed_tokens.weight" | "lm_head.weight") {
            package.weights.get(slot)
        } else {
            package.mtp.get(&format!("{}{slot}", plan.prefix))
        };
        weight.is_some_and(|weight| weight.shape() == tensor.shape)
    })
}

#[cfg(test)]
mod tests;
