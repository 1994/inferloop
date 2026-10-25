//! Agent CLI support.
use super::selected_engine;
use crate::backend;
use crate::constants::MIB_U64;
use infer_agent::transport::serve;
use infer_core::{Error, Result};
use infer_runtime::RuntimeConfig;
use std::{io, path::Path};

pub fn agent(
    package: Option<&Path>,
    memory_mib: u64,
    probe_memory_mib: u64,
    backend_choice: backend::Selection,
) -> Result<()> {
    let mut engine = selected_engine(
        RuntimeConfig::default(),
        None,
        package,
        memory_mib,
        backend_choice,
    )?;
    engine.backend_mut().enable_layer_probes(
        probe_memory_mib
            .checked_mul(MIB_U64)
            .ok_or_else(|| Error::invalid("probe budget overflow"))?,
    )?;
    let mut service = infer_agent::AgentService::new(engine, backend::catalog())?;
    serve(
        &mut service,
        &mut io::stdin().lock(),
        &mut io::stdout().lock(),
    )
}
