//! Cold-start roofline for the actual loaded linear graph, including mixed storage/compute.
//! Memory funding remains the complete pooled serving quote, not a weights-only estimate.
use super::LoadedModel;
use crate::device::device_error;
use crate::mlp::ProjectionWeight;
use crate::target::CudaTarget;
use crate::tuning::pool_dense;
use cutile::prelude::{DType, Tensor};
use infer_core::{Error, Result, TensorId};
use infer_ir::TensorOp;
use std::collections::BTreeMap;

/// Clock attributes are in kilohertz.
const HZ_PER_KHZ: u128 = 1000;
/// Multiply and accumulate count as two operations.
const FLOPS_PER_MAC: u128 = 2;
/// Dense SM120 FP4/F32-accumulate operations per SM per clock.
const SM120_FP4: u128 = 4096;
/// Dense SM120 FP8/F32-accumulate operations per SM per clock.
const SM120_FP8: u128 = 1024;
/// SM120 BF16/F32-accumulate rate; the F32-input path records two BF16 products.
const SM120_BF16: u128 = 512;
/// Dense Hopper BF16/F32-accumulate operations per SM per clock.
const SM90_BF16: u128 = 4096;
/// Dense Hopper FP8/F32-accumulate operations per SM per clock.
const SM90_FP8: u128 = 8192;
/// Both supported SM families have 128 scalar F32 lanes, each executing an FMA.
const SCALAR_F32: u128 = 256;
/// A global FP4 scale is one F32 word.
const GLOBAL_SCALE_BYTES: u128 = 4;
/// The existing two-term F32 input decomposition uses two BF16 MMA products.
const DENSE_PRODUCTS: u128 = 2;

/// Architectural rates, not a table indexed by GPU marketing names. Dense/sparse and
/// accumulator precision must not be conflated. Sources: NVIDIA RTX Blackwell whitepaper
/// Table 3 and NVIDIA Hopper Architecture In-Depth, Tensor Core architecture/table 3.
#[derive(Clone, Copy)]
struct Rates {
    normalization: u128,
    fp4: Option<u128>,
    fp8: u128,
    dense: u128,
}
impl Rates {
    const fn for_target(target: &CudaTarget) -> Option<Self> {
        match target {
            CudaTarget::BlackwellSm120 => Some(Self {
                normalization: SM120_FP4,
                fp4: Some(SM120_FP4),
                fp8: SM120_FP8,
                dense: SM120_BF16 / DENSE_PRODUCTS,
            }),
            CudaTarget::HopperSm90 => Some(Self {
                normalization: SM90_FP8,
                fp4: None,
                fp8: SM90_FP8,
                dense: SM90_BF16 / DENSE_PRODUCTS,
            }),
            CudaTarget::Other(_) => None,
        }
    }
    fn compute(self, mode: Compute) -> Result<u128> {
        match mode {
            Compute::Fp4 => self
                .fp4
                .ok_or_else(|| Error::unsupported("native FP4 roofline on this architecture")),
            Compute::Fp8 => Ok(self.fp8),
            Compute::Dense => Ok(self.dense),
            Compute::Scalar => Ok(self.dense.min(SCALAR_F32)),
        }
    }
}
#[derive(Clone, Copy)]
enum Compute {
    Fp4,
    Fp8,
    Dense,
    Scalar,
}
fn extent(elements: usize) -> Result<u128> {
    u64::try_from(elements)
        .map(u128::from)
        .map_err(device_error)
}
fn mul(a: u128, b: u128) -> Result<u128> {
    a.checked_mul(b)
        .ok_or_else(|| Error::invalid("concurrency roofline overflow"))
}
fn add(a: u128, b: u128) -> Result<u128> {
    a.checked_add(b)
        .ok_or_else(|| Error::invalid("concurrency roofline overflow"))
}
fn tensor_bytes<T: DType>(tensor: &Tensor<T>) -> Result<u128> {
    mul(extent(tensor.size())?, size_of::<T>() as u128)
}
fn bytes(weight: &ProjectionWeight) -> Result<u128> {
    match weight {
        ProjectionWeight::Dense(w) => tensor_bytes(w),
        ProjectionWeight::Fp8(w, scales) | ProjectionWeight::Fp8Block(w, scales) => {
            add(tensor_bytes(w)?, tensor_bytes(scales)?)
        }
        ProjectionWeight::Fp4(w, scales, _) => add(
            add(tensor_bytes(w)?, tensor_bytes(scales)?)?,
            GLOBAL_SCALE_BYTES,
        ),
    }
}
fn elements(sizes: &BTreeMap<TensorId, usize>, id: TensorId) -> Result<usize> {
    sizes
        .get(&id)
        .copied()
        .ok_or_else(|| Error::invariant("roofline graph tensor"))
}

/// Harmonic mixed-precision roofline: bytes/BW divided by sum(2*parameters/peak)
/// and target tokens per sequence per replay. Equal-precision graphs reduce to the
/// documented (peak/BW)*(bytes/parameters)/(2*tokens) formula.
fn bound(
    memory_bytes: u128,
    normalized_flops: u128,
    chip_peak: u128,
    bandwidth: u128,
    tokens: usize,
) -> Result<Option<usize>> {
    if normalized_flops == 0 {
        return Ok(None);
    }
    if memory_bytes == 0 || chip_peak == 0 || bandwidth == 0 || tokens == 0 {
        return Err(Error::invalid(
            "roofline requires positive device/model facts",
        ));
    }
    let numerator = mul(memory_bytes, chip_peak)?;
    let denominator = mul(mul(bandwidth, normalized_flops)?, extent(tokens)?)?;
    Ok(Some(
        usize::try_from(numerator.div_ceil(denominator))
            .unwrap_or(usize::MAX)
            .max(1),
    ))
}
impl LoadedModel {
    fn roofline_mode(
        &self,
        id: TensorId,
        weight: &ProjectionWeight,
        n: usize,
        k: usize,
    ) -> Compute {
        match weight {
            ProjectionWeight::Fp4(..) if self.weights.input_scales.contains_key(&id) => {
                Compute::Fp4
            }
            ProjectionWeight::Fp8(..) | ProjectionWeight::Fp8Block(..)
                if self.weights.fp8_inputs.contains(&id) =>
            {
                Compute::Fp8
            }
            _ if pool_dense::supported([1, n, k]) => Compute::Scalar,
            _ => Compute::Dense,
        }
    }
    pub(crate) fn concurrency_roofline(&self) -> Result<Option<usize>> {
        if self.graph.nodes.iter().any(|node| {
            matches!(
                node.op,
                TensorOp::MoeRoute { .. }
                    | TensorOp::ExpertLinear { .. }
                    | TensorOp::MoeCombine { .. }
            )
        }) {
            tracing::info!(target:"infer::load","routed-expert roofline unavailable; concurrency uses memory bounds");
            return Ok(None);
        }
        let Some(rates) = Rates::for_target(self.device.target()) else {
            tracing::info!(target:"infer::load",architecture=%self.profile.architecture,
                "compute roofline unavailable; concurrency uses memory and protocol bounds");
            return Ok(None);
        };
        let clock = u128::from(self.device.core_clock_khz()?);
        let chip_peak = mul(
            mul(u128::from(self.profile.multiprocessors), clock)?,
            mul(HZ_PER_KHZ, rates.normalization)?,
        )?;
        let sizes = self
            .graph
            .tensors
            .iter()
            .map(|spec| Ok((spec.id, spec.elements()?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let mut memory_bytes = 0;
        let mut normalized_flops = 0;
        for node in &self.graph.nodes {
            if node.op != TensorOp::Linear || self.weights.constants.contains_key(&node.inputs[0]) {
                continue;
            }
            let weight = self
                .weights
                .projections
                .get(&node.inputs[1])
                .ok_or_else(|| Error::invariant("roofline projection weight"))?;
            let n = elements(&sizes, node.outputs[0])?;
            let k = elements(&sizes, node.inputs[0])?;
            let rate = rates.compute(self.roofline_mode(node.inputs[1], weight, n, k))?;
            memory_bytes = add(memory_bytes, bytes(weight)?)?;
            let flops = mul(mul(extent(n)?, extent(k)?)?, FLOPS_PER_MAC)?;
            normalized_flops = add(normalized_flops, mul(flops, rates.normalization / rate)?)?;
        }
        let result = bound(
            memory_bytes,
            normalized_flops,
            chip_peak,
            u128::from(self.profile.memory_bandwidth_bytes_per_second()),
            self.weights.batch_width.max(1),
        )?;
        tracing::info!(target:"infer::load",roofline_bound=?result,
            weight_bytes=%memory_bytes,normalized_flops=%normalized_flops,clock_khz=%clock,
            target_rows=self.weights.batch_width.max(1),"default concurrency compute/bandwidth bound");
        Ok(result)
    }
}
#[cfg(test)]
#[path = "../../tests/unit/loading_concurrency.rs"]
mod tests;
