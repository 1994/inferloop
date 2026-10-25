//! Inference commands.
use super::{
    RunOptions, backend, config, print, read_json, results, run_requests, selected_engine,
    write_json,
};
use infer_core::Result;
use infer_ir::CanonicalRequest;

pub fn run(options: RunOptions, backend_choice: backend::Selection) -> Result<()> {
    let RunOptions {
        requests,
        config: config_path,
        model,
        package,
        host_memory_mib,
        op_trace,
        journal,
        snapshot,
        events,
    } = options;
    let input: Vec<CanonicalRequest> = read_json(&requests)?;
    let ids = input.iter().map(|r| r.id).collect::<Vec<_>>();
    let mut engine = selected_engine(
        config(config_path.as_deref())?,
        model.as_deref(),
        package.as_deref(),
        host_memory_mib,
        backend_choice,
    )?;
    let trace = run_requests(&mut engine, input)?;
    if let Some(path) = journal {
        write_json(&path, &engine.journal()?)?;
    }
    if let Some(path) = snapshot {
        write_json(&path, &engine.snapshot()?)?;
    }
    if let Some(path) = events {
        write_json(&path, &trace)?;
    }
    if let Some(path) = op_trace {
        write_json(&path, &engine.backend().trace())?;
    }
    print(
        &serde_json::json!({"runtime":engine.inspect(),"execution":engine.backend().inspection(),"results":results(&engine,&ids)?}),
    )?;

    Ok(())
}
