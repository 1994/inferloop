//! Cold-start roofline for the actual loaded linear graph, including mixed storage/compute.
//! Memory funding remains the complete pooled serving quote, not a weights-only estimate.
use super::LoadedModel;
use crate::device::device_error;
use crate::mlp::ProjectionWeight;
use crate::tuning::pool_dense;
use cutile::prelude::{DType, Tensor};
use infer_core::{Error, Result, TensorId};
use infer_cuda_kernels::contract::target::TensorRates;
use infer_ir::TensorOp;
use std::collections::BTreeMap;

/// Clock attributes are in kilohertz.
const HZ_PER_KHZ: u128 = 1000;
/// Multiply and accumulate count as two operations.
const FLOPS_PER_MAC: u128 = 2;
/// Both supported SM families have 128 scalar F32 lanes, each executing an FMA.
const SCALAR_F32: u128 = 256;
/// A global FP4 scale is one F32 word.
const GLOBAL_SCALE_BYTES: u128 = 4;

/// Tensor core rates of the device, read by this graph's precisions.
#[derive(Clone, Copy)]
struct Rates(TensorRates);
impl Rates {
    fn compute(self, mode: Compute) -> Result<u128> {
        match mode {
            Compute::Fp4 => self
                .0
                .fp4
                .ok_or_else(|| Error::unsupported("native FP4 roofline on this architecture")),
            Compute::Fp8 => Ok(self.0.fp8),
            Compute::Dense => Ok(self.0.dense),
            Compute::Scalar => Ok(self.0.dense.min(SCALAR_F32)),
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
        use cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE as CLOCK_RATE;
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
        let Some(rates) = self.device.target().tensor_rates().map(Rates) else {
            tracing::info!(target:"infer::load",architecture=%self.profile.architecture,
                "compute roofline unavailable; concurrency uses memory and protocol bounds");
            return Ok(None);
        };
        let clock = u128::from(self.device.attribute(CLOCK_RATE)?);
        let chip_peak = mul(
            mul(u128::from(self.profile.multiprocessors), clock)?,
            mul(HZ_PER_KHZ, rates.0.normalization)?,
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
            normalized_flops = add(normalized_flops, mul(flops, rates.0.normalization / rate)?)?;
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
