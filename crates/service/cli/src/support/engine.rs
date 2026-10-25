//! Engine CLI support.
use super::read_json;
use crate::backend;
use crate::backend::SelectedBackend;
use infer_core::{Error, Result};
use infer_ir::PrecisionPlan;
use infer_runtime::{Engine, RuntimeConfig};
use infer_spi::{BackendProvider as _, SchedulingRecommendation};
use std::path::Path;

pub fn selected_engine(
    config: RuntimeConfig,
    model: Option<&Path>,
    package: Option<&Path>,
    memory_mib: u64,
    choice: backend::Selection,
) -> Result<Engine<SelectedBackend>> {
    configured_engine(Some(config), model, package, memory_mib, choice)
}

pub fn configured_engine(
    config: Option<RuntimeConfig>,
    model: Option<&Path>,
    package: Option<&Path>,
    memory_mib: u64,
    choice: backend::Selection,
) -> Result<Engine<SelectedBackend>> {
    let automatic = config.is_none();
    let mut config = config.unwrap_or_default();
    config.block_size = choice.block_size.unwrap_or(config.block_size);
    if let Some(tokens) = choice.max_num_batched_tokens {
        config.max_num_batched_tokens = tokens;
    }
    if let Some(tokens) = choice.max_model_len {
        config.max_model_len = Some(tokens);
    }
    if let Some(tokens) = choice.max_output_tokens {
        config.max_output_tokens = Some(tokens);
    }
    let explicit_sequences = choice.max_num_seqs.is_some();
    config.max_num_seqs_explicit = explicit_sequences;
    if let Some(sequences) = choice.max_num_seqs {
        config.max_num_seqs = sequences;
    }
    if let Some(name) = choice.served_model_name.clone() {
        config.served_model_name = Some(name);
    }
    // The refined selection is the only one the backend sees, and building it here consumes the
    // caller's value on every path instead of leaving it borrowed in host-only builds.
    let choice = backend::Selection {
        block_size: Some(config.block_size),
        ..choice
    };
    if let Some(package) = package {
        let backend = config
            .cpu
            .placement
            .device
            .scope(|| backend::load(package, memory_mib, &choice))?;
        let ir = backend.model_ir().clone();
        let registry = backend.registry()?;
        if automatic {
            let recommendation = backend.scheduling_recommendation()?;
            apply_recommendation(&mut config, explicit_sequences, recommendation);
        }
        if automatic {
            super::serving::derive_config(&mut config, &backend, &ir, &registry)?;
        }
        let mut engine = Engine::new(backend, ir, PrecisionPlan::f32(), &registry, config)?;
        if package.join("readouts.safetensors").exists() {
            let provider = infer_workloads::ProjectionWorkloads::open(
                package,
                engine.model().hidden_size,
                engine.model().vocab_size,
            )?;
            engine = engine.with_workloads(provider)?;
        }
        Ok(engine)
    } else {
        let _ = model;
        Err(Error::invalid("a device backend requires --package"))
    }
}
fn apply_recommendation(
    config: &mut RuntimeConfig,
    explicit: bool,
    recommendation: Option<SchedulingRecommendation>,
) {
    if explicit {
        return;
    }
    let Some(recommendation) = recommendation else {
        return;
    };
    config.max_num_seqs = recommendation.max_sequences;
    config.max_num_seqs_explicit = false;
    tracing::info!(
        target: "infer::load",
        max_num_seqs = recommendation.max_sequences,
        memory_bound = recommendation.memory_bound,
        knee_bound = ?recommendation.knee_bound,
        "automatic scheduling bound selected"
    );
}

pub fn config(path: Option<&Path>) -> Result<RuntimeConfig> {
    path.map(read_json)
        .transpose()
        .map(Option::unwrap_or_default)
}

#[cfg(test)]
#[path = "../../tests/unit/support_engine.rs"]
mod tests;
