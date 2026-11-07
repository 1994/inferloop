//! Capture the same quantizer and fused recorder used by prompt graphs.
use super::super::{
    GemmConfig, ResidentGemmKey, SAMPLE_CENTER, SAMPLE_DIVISOR, SAMPLE_PERIOD, SAMPLE_STRIDE,
    kernel_environment,
};
use super::evidence::{candidates, compatible};
use crate::constants::{NVFP4_GROUP_SIZE, NVFP4_QUANT_CODES_TILE, NVFP4_QUANT_SCALES_TILE};
use crate::device::{CudaDevice, device_error};
use crate::mlp::ProjectionWeight;
use crate::tuning::preparation::driver_build;
use crate::tuning::projection_budget::ProjectionShape;
use crate::tuning::resident_gemm::QUANTIZATION_ROWS;
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{
    Buffer, Fp4Pair, Fp4QuantKind, Fp4QuantPlan, Fp8, GemmLoop, GemmPlan, Kernels, Recording,
};
use infer_cuda_kernels::tile::identity;
use infer_cuda_kernels::tile::jit::JitOptions;
use infer_cuda_kernels::tile::runtime::{Allocation, Graph};
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

/// Changes to fused arithmetic, quantization or confirmation invalidate prior measurements.
const SCHEMA: &str = "cutile0.4-swiglu-v1:ab-search:ba-confirm:k256:f32:block16:full-codes-scales";
/// Largest finite, nonnegative E4M3 scale byte.
const MAX_FINITE_SCALE: u8 = 0x7e;
/// Separate arithmetic family from standalone W4A4 projections.
const MODE: &str = "nvfp4-w4a4-swiglu-block16";
type Operands<'a> = (Buffer<'a, Fp4Pair>, Buffer<'a, Fp8>);

/// Record the fused gate/up GEMM that writes the down projection's codes and scales from the
/// quantized input; `extents` is `[rows, hidden, columns]`.
/// # Errors
/// Rejects non-FP4 weights and a rejected launch.
pub fn record(
    (contract, scope): (&Kernels, &Recording<'_>),
    config: GemmConfig,
    (gate, up): (&ProjectionWeight, &ProjectionWeight),
    (input_scale, down_scale): (f32, f32),
    quantized: Operands<'_>,
    outputs: Operands<'_>,
    [rows, hidden, columns]: ProjectionShape,
) -> Result<()> {
    let (ProjectionWeight::Fp4(_, _, gate_global), ProjectionWeight::Fp4(_, _, up_global)) =
        (gate, up)
    else {
        return Err(device_error("fused SwiGLU weights"));
    };
    contract
        .projection
        .gemm_swiglu(
            scope,
            &GemmPlan {
                main_loop: GemmLoop::Plain,
                rows,
                outputs: hidden,
                columns,
                tile: config.tile(),
            },
            outputs,
            quantized,
            (gate.weight_ref()?, up.weight_ref()?),
            (
                1.0 / (gate_global * input_scale),
                1.0 / (up_global * input_scale),
                down_scale,
            ),
        )
        .map_err(|e| device_error(e.to_string()))
}

pub(super) struct Captured {
    pub(super) graph: Graph,
    codes: Allocation<f4e2m1fnx2>,
    scales: Allocation<f8e4m3fn>,
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
        // The graph retains every allocation it records, so only the outputs stay here.
        let runtime = device.runtime();
        let input = runtime.upload::<f32>(values)?;
        let q = runtime.zeros::<f4e2m1fnx2>(m * (k / 2))?;
        let qs = runtime.zeros::<f8e4m3fn>(m * (k / NVFP4_GROUP_SIZE))?;
        let codes = runtime.zeros::<f4e2m1fnx2>(m * (n / 2))?;
        let out_scales = runtime.zeros::<f8e4m3fn>(m * (n / NVFP4_GROUP_SIZE))?;
        let contract = device.kernels();
        let graph = device.tile().capture(|scope| {
            contract
                .projection
                .quantize_fp4(
                    scope,
                    &Fp4QuantPlan {
                        kind: Fp4QuantKind::RowGroups {
                            rows: QUANTIZATION_ROWS,
                        },
                        rows: m,
                        columns: k,
                        codes_tile: [QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1]],
                        scales_tile: [QUANTIZATION_ROWS, NVFP4_QUANT_SCALES_TILE[1]],
                    },
                    q.operand(),
                    qs.operand(),
                    input.operand(),
                    scales.0,
                )
                .map_err(|e| device_error(e.to_string()))?;
            record(
                (&contract, scope),
                config,
                weights,
                scales,
                (q.operand(), qs.operand()),
                (codes.operand(), out_scales.operand()),
                [m, n, k],
            )
        })?;
        Ok(Self {
            graph,
            codes,
            scales: out_scales,
        })
    }
    pub(super) fn run(&self, device: &CudaDevice) -> Result<()> {
        device.tile().sync(self.graph.launch())?;
        Ok(())
    }
    pub(super) fn bytes(&self, device: &CudaDevice) -> Result<(Vec<u8>, Vec<u8>)> {
        let runtime = device.runtime();
        let codes = runtime
            .read(&self.codes)?
            .into_iter()
            .map(f4e2m1fnx2::to_bits)
            .collect();
        let scales: Vec<_> = runtime
            .read(&self.scales)?
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
    let compile = JitOptions::default();
    key.kernel_schema = format!(
        "{SCHEMA}:jit-opt{:?}:debug{}",
        compile.opt_level(),
        compile.device_debug()
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
