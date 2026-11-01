use super::{Model, mtp::Mtp};
use infer_backend_cuda::{
    strategy::LinearTiling,
    tuning::{projection_key, tune},
};
use infer_core::{Error, Result};
use serde_json::Value;
use std::fs::metadata;
use std::fs::read;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

pub fn run(model: &Model, draft: Option<&Mtp>) -> Result<()> {
    let mut seen = BTreeSet::new();
    let mut results = Vec::new();
    let mut weights: Vec<_> = model
        .projections
        .values()
        .map(super::weights::Projection::resident)
        .collect();
    if let Some(draft) = draft {
        weights.extend(
            draft
                .model
                .projections
                .values()
                .map(super::weights::Projection::resident),
        );
        weights.push(draft.fc.resident());
    }
    for weight in weights {
        let (key, _, _) = projection_key(&weight)?;
        if seen.insert(key.clone()) {
            eprintln!("tuning {key}");
            results.push(tune(&model.device, &weight)?);
        }
    }
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "schema": 1, "results": results,
        "measurement": "paired CUDA events around graph replay, cold L2; actual model weights and synthetic activations",
    })).map_err(|e| Error::invalid(e.to_string()))?);
    Ok(())
}

pub fn load(path: &Path, model: &Model) -> Result<BTreeMap<String, LinearTiling>> {
    if metadata(path)
        .map_err(|e| Error::invalid(e.to_string()))?
        .len()
        > 16 * 1024 * 1024
    {
        return Err(Error::invalid("tuning report exceeds 16 MiB"));
    }
    let report: Value =
        serde_json::from_slice(&read(path).map_err(|e| Error::invalid(e.to_string()))?)
            .map_err(|e| Error::invalid(e.to_string()))?;
    if report["schema"] != 1 {
        return Err(Error::invalid("unsupported tuning schema"));
    }
    let rows = report["results"]
        .as_array()
        .ok_or_else(|| Error::invalid("missing tuning results"))?;
    let target = serde_json::to_value(model.device.target().name())
        .map_err(|e| Error::invalid(e.to_string()))?;
    let gpu = model.device.name()?;
    let mut tuning = BTreeMap::new();
    for row in rows {
        if row["target"] != target || row["gpu"].as_str() != Some(gpu.as_str()) {
            return Err(Error::invalid("tuning hardware does not match current GPU"));
        }
        let dimension = |key: &str| {
            row["selected"][key]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| Error::invalid("invalid tuning tile"))
        };
        let key = row["key"]
            .as_str()
            .ok_or_else(|| Error::invalid("invalid tuning key"))?;
        if tuning
            .insert(
                key.to_owned(),
                LinearTiling::new(dimension("rows")?, dimension("columns")?)?,
            )
            .is_some()
        {
            return Err(Error::invalid("duplicate tuning key"));
        }
    }
    Ok(tuning)
}
