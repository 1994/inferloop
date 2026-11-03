//! Capture the same quantizer and fused recorder used by prompt graphs.
use super::super::{
    GemmConfig, ResidentGemmKey, SAMPLE_CENTER, SAMPLE_DIVISOR, SAMPLE_PERIOD, SAMPLE_STRIDE,
    kernel_environment,
};
use super::evidence::{candidates, compatible};
use crate::constants::{NVFP4_GROUP_SIZE, NVFP4_QUANT_CODES_TILE, NVFP4_QUANT_SCALES_TILE};
use crate::device::{CudaDevice, device_error};
use crate::mlp::ProjectionWeight;
use crate::resident::tensor_buffer as buffer;
use crate::tuning::preparation::driver_build;
use crate::tuning::projection_budget::ProjectionShape;
use crate::tuning::resident_gemm::QUANTIZATION_ROWS;
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{
    Fp4QuantKind, Fp4QuantPlan, GemmLoop, GemmPlan, Kernels, Recording,
};
use infer_cuda_kernels::tile::identity;
use infer_cuda_kernels::tile::runtime::Graph;
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
    (contract, scope): (&Kernels, &Scope),
    config: GemmConfig,
    (gate, up): (&ProjectionWeight, &ProjectionWeight),
    (input_scale, down_scale): (f32, f32),
    (q, qs): Operands<'_>,
    (codes, scales): Outputs<'_>,
    columns: usize,
) -> StdResult<(), DeviceError> {
    let (ProjectionWeight::Fp4(_, _, gate_global), ProjectionWeight::Fp4(_, _, up_global)) =
        (gate, up)
    else {
        return Err(DeviceError::Internal("fused SwiGLU weights".into()));
    };
    let shape = |axis: usize| {
        usize::try_from(codes.shape()[axis]).map_err(|e| DeviceError::Internal(e.to_string()))
    };
    contract
        .projection
        .gemm_swiglu(
            &Recording::scope(scope),
            &GemmPlan {
                main_loop: GemmLoop::Plain,
                rows: shape(0)?,
                outputs: shape(1)? * 2,
                columns,
                tile: config.tile(),
            },
            (buffer(&*codes)?, buffer(&*scales)?),
            (buffer(q)?, buffer(qs)?),
            (gate.weight_ref()?, up.weight_ref()?),
            (
                1.0 / (gate_global * input_scale),
                1.0 / (up_global * input_scale),
                down_scale,
            ),
        )
        .map_err(|e| DeviceError::Internal(e.to_string()))
}

pub(super) struct Captured {
    pub(super) graph: Graph,
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
        let q = api::zeros::<f4e2m1fnx2>(&[m, k / 2])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let qs = api::zeros::<f8e4m3fn>(&[m, k / NVFP4_GROUP_SIZE])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut codes = api::zeros::<f4e2m1fnx2>(&[m, n / 2])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut out_scales = api::zeros::<f8e4m3fn>(&[m, n / NVFP4_GROUP_SIZE])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let contract = device.kernels();
        let graph = Graph::scope(&device.stream, |scope| {
            contract
                .projection
                .quantize_fp4(
                    &Recording::scope(scope),
                    &Fp4QuantPlan {
                        kind: Fp4QuantKind::RowGroups {
                            rows: QUANTIZATION_ROWS,
                        },
                        rows: m,
                        columns: k,
                        codes_tile: [QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1]],
                        scales_tile: [QUANTIZATION_ROWS, NVFP4_QUANT_SCALES_TILE[1]],
                    },
                    buffer(&q)?,
                    buffer(&qs)?,
                    buffer(input.as_ref())?,
                    scales.0,
                )
                .map_err(|e| DeviceError::Internal(e.to_string()))?;
            record(
                (&contract, scope),
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
    let tiles: Vec<_> = candidates(shape).map(GemmConfig::tile).collect();
    let identities = identity::swiglu_candidates(
        &device.stream,
        shape,
        identity::SwigluSearch {
            group: NVFP4_GROUP_SIZE,
            codes_tile: [QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1]],
            scales_tile: [QUANTIZATION_ROWS, NVFP4_QUANT_SCALES_TILE[1]],
            tiles: &tiles,
        },
    )?;
    let fingerprint = serde_json::to_string(&identities).map_err(device_error)?;
    memo.lock()
        .map_err(|_| Error::invariant("SwiGLU identity cache poisoned"))?
        .insert((environment, shape), fingerprint.clone());
    Ok(fingerprint)
}
