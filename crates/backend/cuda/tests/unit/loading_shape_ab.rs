use super::*;
use crate::tuning::resident_gemm::{ResidentTuningSession, select};
use crate::tuning::resolve_table;
use infer_ir::TensorOp;
use std::env::var;
use std::error::Error as StdError;
use std::path::PathBuf;
use std::result::Result as StdResult;

#[test]
#[ignore = "opt-in projection-only shape A/B with real checkpoint and CUDA; run inside safe-run"]
fn prefill_projection_shape_ab() -> StdResult<(), Box<dyn StdError>> {
    let model = var("INFER_TEST_MODEL")?;
    let table = PathBuf::from(var("INFER_CUDA_TUNING_TABLE")?);
    if table.exists() {
        return Err(format!("refusing to overwrite tuning evidence: {table:?}").into());
    }
    let device = CudaDevice::new(0)?;
    let loaded = LoadedModel::open(
        device,
        model,
        ModelId::new(1)?,
        LoadOptions {
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    for shape in [[1024, 5120, 17408], [1024, 17408, 5120]] {
        let node = loaded
            .graph
            .nodes
            .iter()
            .find(|node| {
                node.op == TensorOp::Linear
                    && loaded.graph.tensors.iter().any(|tensor| {
                        tensor.id == node.outputs[0] && tensor.elements() == Ok(shape[1])
                    })
                    && loaded.graph.tensors.iter().any(|tensor| {
                        tensor.id == node.inputs[0] && tensor.elements() == Ok(shape[2])
                    })
            })
            .ok_or_else(|| Error::invalid("projection shape not found"))?;
        let weight = loaded
            .weights
            .projections
            .get(&node.inputs[1])
            .ok_or_else(|| Error::invalid("projection weight not found"))?;
        let scale = *loaded
            .weights
            .input_scales
            .get(&node.inputs[1])
            .ok_or_else(|| Error::invalid("projection scale not found"))?;
        let session = ResidentTuningSession::for_hardware_validation();
        let selected = select(loaded.device(), weight, scale, shape, true, &session);
        let table_data = resolve_table(Some(&table))?
            .ok_or_else(|| Error::invariant("missing projection tuning table"))?;
        let entry = table_data
            .resident_gemm
            .iter()
            .find(|entry| {
                entry.key.m == shape[0] && entry.key.n == shape[1] && entry.key.k == shape[2]
            })
            .ok_or_else(|| Error::invariant("missing projection tuning entry"))?;
        assert_eq!(entry.selected, selected);
        assert!(entry.bitwise_verified);
        assert!(!entry.trials.is_empty());
        println!("{}", serde_json::to_string(entry)?);
    }
    Ok(())
}
