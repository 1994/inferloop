//! Capture the same quantizer and fused recorder used by prompt graphs.
use super::super::{
    GemmConfig, ResidentGemmKey, SAMPLE_CENTER, SAMPLE_DIVISOR, SAMPLE_PERIOD, SAMPLE_STRIDE,
    driver_build, kernel_environment,
};
use super::evidence::{candidates, compatible};
use crate::constants::{NVFP4_GROUP_SIZE, NVFP4_QUANT_CODES_TILE, NVFP4_QUANT_SCALES_TILE};
use crate::device::{CudaDevice, device_error};
use crate::mlp::ProjectionWeight;
use crate::resident::nvfp4_gemm::kernels;
use crate::tuning::projection_budget::ProjectionShape;
use crate::tuning::resident_gemm::QUANTIZATION_ROWS;
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::prelude::*;
use infer_core::{Error, Result};
use std::collections::BTreeMap;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex, OnceLock};

/// Changes to fused arithmetic, quantization or confirmation invalidate prior measurements.
const SCHEMA: &str = "cutile0.4-swiglu-v1:ab-search:ba-confirm:k256:f32:block16:full-codes-scales";
/// Largest finite, nonnegative E4M3 scale byte.
const MAX_FINITE_SCALE: u8 = 0x7e;
/// Separate arithmetic family from standalone W4A4 projections.
const MODE: &str = "nvfp4-w4a4-swiglu-block16";
type Operands<'a> = (&'a Tensor<f4e2m1fnx2>, &'a Tensor<f8e4m3fn>);
type Outputs<'a> = (&'a mut Tensor<f4e2m1fnx2>, &'a mut Tensor<f8e4m3fn>);

pub fn record(
    scope: &Scope,
    config: GemmConfig,
    (gate, up): (&ProjectionWeight, &ProjectionWeight),
    (input_scale, down_scale): (f32, f32),
    (q, qs): Operands<'_>,
    (codes, scales): Outputs<'_>,
    columns: usize,
) -> StdResult<(), DeviceError> {
    let (
        ProjectionWeight::Fp4(gate, gate_scales, gate_global),
        ProjectionWeight::Fp4(up, up_scales, up_global),
    ) = (gate, up)
    else {
        return Err(DeviceError::Internal("fused SwiGLU weights".into()));
    };
    let [m, n] = config.tile();
    scope.record(
        kernels::packed_swiglu(
            codes.partition([m, n / 2]),
            scales.partition([m, n / NVFP4_GROUP_SIZE]),
            q,
            qs,
            gate,
            gate_scales,
            up,
            up_scales,
            1.0 / (gate_global * input_scale),
            1.0 / (up_global * input_scale),
            down_scale,
        )
        .generics(
            [columns, m, n, n / 2, n / NVFP4_GROUP_SIZE]
                .iter()
                .map(ToString::to_string)
                .collect(),
        ),
    )?;
    Ok(())
}

pub(super) struct Captured {
    pub(super) graph: CudaGraph<()>,
    codes: Tensor<f4e2m1fnx2>,
    scales: Tensor<f8e4m3fn>,
    _input: Arc<Tensor<f32>>,
    _q: Tensor<f4e2m1fnx2>,
    _qs: Tensor<f8e4m3fn>,
}
impl Captured {
    pub(super) fn new(
        device: &CudaDevice,
        weights: (&ProjectionWeight, &ProjectionWeight),
        scales: (f32, f32),
        shape: ProjectionShape,
        config: GemmConfig,
    ) -> Result<Self> {
        let [m, n, k] = shape;
        weights.0.validate(n, k)?;
        weights.1.validate(n, k)?;
        if !compatible(config, shape)
            || !m.is_multiple_of(QUANTIZATION_ROWS)
            || <[f32; 2]>::from(scales)
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(Error::invalid("SwiGLU tuning geometry or scales"));
        }
        let length = m
            .checked_mul(k)
            .ok_or_else(|| Error::invalid("SwiGLU input extent"))?;
        let values = (0..length)
            .map(|i| {
                (f32::from(
                    u16::try_from((i % SAMPLE_PERIOD) * SAMPLE_STRIDE % SAMPLE_PERIOD)
                        .unwrap_or_default(),
                ) - SAMPLE_CENTER)
                    / SAMPLE_DIVISOR
            })
            .collect();
        let input = device.upload(values, &[m, k])?;
        let mut q = api::zeros::<f4e2m1fnx2>(&[m, k / 2])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut qs = api::zeros::<f8e4m3fn>(&[m, k / NVFP4_GROUP_SIZE])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut codes = api::zeros::<f4e2m1fnx2>(&[m, n / 2])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut out_scales = api::zeros::<f8e4m3fn>(&[m, n / NVFP4_GROUP_SIZE])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let graph = CudaGraph::scope(&device.stream, |scope| {
            scope.record(
                kernels::quantize_rows(
                    (&mut q).partition([QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1]]),
                    (&mut qs).partition([QUANTIZATION_ROWS, NVFP4_QUANT_SCALES_TILE[1]]),
                    &input,
                    scales.0,
                )
                .generics(vec![k.to_string()]),
            )?;
            record(
                scope,
                config,
                weights,
                scales,
                (&q, &qs),
                (&mut codes, &mut out_scales),
                k,
            )
        })
        .map_err(device_error)?;
        Ok(Self {
            graph,
            codes,
            scales: out_scales,
            _input: input,
            _q: q,
            _qs: qs,
        })
    }
    pub(super) fn run(&self, device: &CudaDevice) -> Result<()> {
        self.graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        Ok(())
    }
    pub(super) fn bytes(&self, device: &CudaDevice) -> Result<(Vec<u8>, Vec<u8>)> {
        let codes = device
            .read_borrowed(&self.codes)?
            .into_iter()
            .map(f4e2m1fnx2::to_bits)
            .collect();
        let scales: Vec<_> = device
            .read_borrowed(&self.scales)?
            .into_iter()
            .map(|v| v.0)
            .collect();
        if scales.is_empty() || scales.iter().any(|v| *v > MAX_FINITE_SCALE) {
            return Err(Error::invariant(
                "SwiGLU tuning produced nonfinite or negative scales",
            ));
        }
        Ok((codes, scales))
    }
}
pub(super) fn verify(actual: &(Vec<u8>, Vec<u8>), expected: &(Vec<u8>, Vec<u8>)) -> Result<()> {
    if actual.0.is_empty() || actual.1.is_empty() || actual != expected {
        return Err(Error::invariant(
            "SwiGLU tuning changed complete codes or scales",
        ));
    }
    Ok(())
}
pub(super) fn key(device: &CudaDevice, shape: ProjectionShape) -> Result<ResidentGemmKey> {
    let mut key = ResidentGemmKey::new(
        device.profile()?,
        CudaDevice::driver_version()?,
        shape,
        driver_build(),
        fingerprint(device, shape)?,
    )?;
    let compile = CompileOptions::default();
    key.kernel_schema = format!(
        "{SCHEMA}:jit-opt{:?}:debug{}",
        compile.opt_level, compile.device_debug
    );
    key.activation_mode = MODE.into();
    Ok(key)
}
fn fingerprint(device: &CudaDevice, shape: ProjectionShape) -> Result<String> {
    type Memo = Mutex<BTreeMap<(String, ProjectionShape), String>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let environment = kernel_environment(device)?;
    let memo = MEMO.get_or_init(|| Mutex::new(BTreeMap::new()));
    let cached = memo
        .lock()
        .map_err(|_| Error::invariant("SwiGLU identity cache poisoned"))?
        .get(&(environment.clone(), shape))
        .cloned();
    if let Some(value) = cached {
        return Ok(value);
    }
    let [m, n, k] = shape;
    let mut q = api::meta::<f4e2m1fnx2>(&[m, k / 2])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut qs = api::meta::<f8e4m3fn>(&[m, k / NVFP4_GROUP_SIZE])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let weight = api::meta::<f4e2m1fnx2>(&[n, k / 2])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let weight_scales = api::meta::<f8e4m3fn>(&[n, k / NVFP4_GROUP_SIZE])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut codes = api::meta::<f4e2m1fnx2>(&[m, n / 2])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut scales = api::meta::<f8e4m3fn>(&[m, n / NVFP4_GROUP_SIZE])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut identities = vec![quantizer_key(device, shape, &mut q, &mut qs)?];
    for config in candidates(shape) {
        let [tm, tn] = config.tile();
        identities.push(
            kernels::packed_swiglu(
                (&mut codes).partition([tm, tn / 2]),
                (&mut scales).partition([tm, tn / NVFP4_GROUP_SIZE]),
                &q,
                &qs,
                &weight,
                &weight_scales,
                &weight,
                &weight_scales,
                1.0,
                1.0,
                1.0,
            )
            .generics(
                [k, tm, tn, tn / 2, tn / NVFP4_GROUP_SIZE]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            )
            .l2_cache_key_on(&device.stream)
            .map_err(device_error)?,
        );
    }
    let fingerprint = serde_json::to_string(&identities).map_err(device_error)?;
    memo.lock()
        .map_err(|_| Error::invariant("SwiGLU identity cache poisoned"))?
        .insert((environment, shape), fingerprint.clone());
    Ok(fingerprint)
}

fn quantizer_key(
    device: &CudaDevice,
    shape: ProjectionShape,
    q: &mut Tensor<f4e2m1fnx2>,
    qs: &mut Tensor<f8e4m3fn>,
) -> Result<String> {
    let [m, _, k] = shape;
    let input = api::meta::<f32>(&[m, k])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    kernels::quantize_rows(
        q.partition([QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1]]),
        qs.partition([QUANTIZATION_ROWS, NVFP4_QUANT_SCALES_TILE[1]]),
        &input,
        1.0,
    )
    .generics(vec![k.to_string()])
    .l2_cache_key_on(&device.stream)
    .map_err(device_error)
}
