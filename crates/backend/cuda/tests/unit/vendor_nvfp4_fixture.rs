use crate::device::CudaDevice;
use crate::loading::LoadedModel;
use crate::mlp::ProjectionWeight;
use infer_core::Error;
use infer_ir::TensorOp;
use std::collections::BTreeSet;
use std::error::Error as StdError;
const OUTPUTS: usize = 5120;
const COLUMNS: usize = 17408;
type TestResult<T = ()> = Result<T, Box<dyn StdError>>;
pub struct Layer {
    pub(crate) weight: ProjectionWeight,
    pub(crate) scale: f32,
}
pub fn layers(_device: &CudaDevice, loaded: &LoadedModel) -> TestResult<Vec<Layer>> {
    let mut seen = BTreeSet::new();
    let mut layers = Vec::new();
    for node in &loaded.graph().nodes {
        if node.op != TensorOp::Linear || !seen.insert(node.inputs[1]) {
            continue;
        }
        let Some(weight @ ProjectionWeight::Fp4(codes, _, _)) =
            loaded.weights.projections.get(&node.inputs[1])
        else {
            continue;
        };
        if codes.shape() != [i32::try_from(OUTPUTS)?, i32::try_from(COLUMNS / 2)?] {
            continue;
        }
        let scale = *loaded
            .weights
            .input_scales
            .get(&node.inputs[1])
            .ok_or_else(|| Error::invalid("FP4 activation scale missing"))?;
        layers.push(Layer {
            weight: weight.clone(),
            scale,
        });
    }
    if layers.len() != 56 {
        return Err("this checkpoint screen requires all 56 target down projections".into());
    }
    Ok(layers)
}
