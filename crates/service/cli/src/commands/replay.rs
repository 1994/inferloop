//! Replay commands.
use super::{ReplayOptions, backend, config, print, read_json, run_to_idle, selected_engine};
use infer_core::{Error, Result};
use infer_runtime::{Engine, ReplayAction, RuntimeSnapshot};

pub fn replay(options: ReplayOptions, backend_choice: backend::Selection) -> Result<()> {
    let ReplayOptions {
        journal,
        snapshot,
        config: config_path,
        model,
        package,
        host_memory_mib,
    } = options;
    let mut engine = if let Some(path) = snapshot {
        let snapshot: RuntimeSnapshot = read_json(&path)?;
        let engine = selected_engine(
            snapshot.config.clone(),
            model.as_deref(),
            package.as_deref(),
            host_memory_mib,
            backend_choice,
        )?;
        let registry = engine.backend().registry()?;
        Engine::restore_with_workloads(
            engine.backend().fresh()?,
            &registry,
            snapshot,
            engine.fork_workloads()?,
        )?
    } else {
        let mut engine = selected_engine(
            config(config_path.as_deref())?,
            model.as_deref(),
            package.as_deref(),
            host_memory_mib,
            backend_choice,
        )?;
        let actions: Vec<ReplayAction> = read_json(
            journal
                .as_deref()
                .ok_or_else(|| Error::invariant("clap requires journal or snapshot"))?,
        )?;
        engine.replay(&actions)?;
        engine
    };
    run_to_idle(&mut engine)?;
    let snapshot = engine.snapshot()?;
    print(
        &serde_json::json!({"runtime":engine.inspect(),"results":snapshot.requests.values().filter_map(|r|r.completed.as_ref()).collect::<Vec<_>>(),"decisions":engine.decisions()}),
    )?;

    Ok(())
}
