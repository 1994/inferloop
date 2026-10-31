//! Own every temporary until the captured graph is destroyed.
use super::{Choice, Key};
use crate::device::{CudaDevice, device_error};
use crate::strategy::LinearTiling;
use crate::tuning::preparation::driver_build;
use crate::tuning::projection_budget::ProjectionShape;
use crate::tuning::resident_gemm::{ResidentGemmKey, kernel_environment};
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::{half::bf16, prelude::*};
use infer_core::{Error, Result};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Identity covers the original and exact row-tiled reduction kernels and paired protocol.
const SCHEMA: &str =
    "pool-dense-v1:private-k-order:row1-row4:legacy-mma:ab-ba:cold-warm:full-finite";
/// Native row-major activation and weight arithmetic.
const MODE: &str = "f32-bf16-pooled-dense";
/// Bounded deterministic activation pattern.
const PERIOD: usize = 131;
/// Coprime stride visits every activation in the pattern.
const STRIDE: usize = 17;
/// Signed center of the activation sample.
const CENTER: f32 = 65.0;
/// Normalize the activation sample.
const DIVISOR: f32 = 37.0;

pub(super) struct Captured {
    pub(super) graph: CudaGraph<()>,
    output: Tensor<f32>,
    _input: Arc<Tensor<f32>>,
    _weight: Arc<Tensor<bf16>>,
}
impl Captured {
    pub(super) fn new(
        device: &CudaDevice,
        weight: &Arc<Tensor<bf16>>,
        shape: ProjectionShape,
        tile: LinearTiling,
        choice: Choice,
    ) -> Result<Self> {
        let [m, n, k] = shape;
        let [wn, wk] = weight.shape() else {
            return Err(Error::invalid("pooled dense weight rank"));
        };
        if !(super::searchable(shape) || choice.prompt_family() && super::prompt_searchable(shape))
            || usize::try_from(*wn).map_err(device_error)? != n
            || usize::try_from(*wk).map_err(device_error)? != k
        {
            return Err(Error::invalid("pooled dense weight geometry"));
        }
        let length = m
            .checked_mul(k)
            .ok_or_else(|| Error::invalid("pooled dense input extent"))?;
        let values = (0..length)
            .map(|i| {
                (f32::from(u16::try_from((i % PERIOD) * STRIDE % PERIOD).unwrap_or_default())
                    - CENTER)
                    / DIVISOR
            })
            .collect();
        let input = device.upload(values, &[m, k])?;
        let mut output = api::zeros::<f32>(&[m, n])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let view = input.view(&[m, k]).map_err(device_error)?;
        let graph = CudaGraph::scope(&device.stream, |scope| {
            choice.record(scope, weight, &view, &mut output, k, tile)
        })
        .map_err(device_error)?;
        Ok(Self {
            graph,
            output,
            _input: input,
            _weight: Arc::clone(weight),
        })
    }
    pub(super) fn run(&self, device: &CudaDevice) -> Result<()> {
        self.graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        Ok(())
    }
    pub(super) fn output(&self, device: &CudaDevice) -> Result<Vec<f32>> {
        let values = device.read_borrowed(&self.output)?;
        if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
            return Err(Error::invariant(
                "pooled dense produced nonfinite or empty output",
            ));
        }
        Ok(values)
    }
}
pub(super) fn verify(actual: &[f32], expected: &[f32]) -> Result<()> {
    if actual.is_empty()
        || actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(a, b)| !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits())
    {
        return Err(Error::invariant(
            "pooled dense changed private reduction bits",
        ));
    }
    Ok(())
}
pub(super) fn key(
    device: &CudaDevice,
    shape: ProjectionShape,
    tile: LinearTiling,
    base: Choice,
) -> Result<Key> {
    type Memo = Mutex<BTreeMap<(String, ProjectionShape, usize, usize, bool), String>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let environment = kernel_environment(device)?;
    let memo = MEMO.get_or_init(|| Mutex::new(BTreeMap::new()));
    let cache = (
        environment,
        shape,
        tile.rows(),
        tile.columns(),
        base.prompt_family(),
    );
    let cached = memo
        .lock()
        .map_err(|_| Error::invariant("pooled dense identity cache poisoned"))?
        .get(&cache)
        .cloned();
    let fingerprint = if let Some(value) = cached {
        value
    } else {
        let value = if base.prompt_family() {
            prompt_fingerprint(device, shape)?
        } else {
            fingerprint(device, shape, tile)?
        };
        memo.lock()
            .map_err(|_| Error::invariant("pooled dense identity cache poisoned"))?
            .insert(cache, value.clone());
        value
    };
    let mut native = ResidentGemmKey::new(
        device.profile()?,
        CudaDevice::driver_version()?,
        shape,
        driver_build(),
        fingerprint,
    )?;
    let compile = CompileOptions::default();
    native.kernel_schema = format!(
        "{SCHEMA}:jit-opt{:?}:debug{}",
        compile.opt_level, compile.device_debug
    );
    native.activation_mode = MODE.into();
    Ok(Key {
        native,
        tile: [tile.rows(), tile.columns()],
        baseline: base,
    })
}
fn prompt_fingerprint(device: &CudaDevice, shape: ProjectionShape) -> Result<String> {
    use crate::resident::prefill_gemm::gemm;
    let [m, n, k] = shape;
    let input = api::meta::<f32>(&[m, k])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let weight = api::meta::<bf16>(&[n, k])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut output = api::meta::<f32>(&[m, n])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut identities = Vec::new();
    for choice in [
        Choice::PromptBase,
        Choice::PromptM64N128,
        Choice::PromptM32N256,
        Choice::PromptM128N64,
    ] {
        let [rows, outputs] = choice.prompt_tile();
        identities.push(
            gemm::dense_tile((&mut output).partition([rows, outputs]), &input, &weight)
                .generics(vec![rows.to_string(), outputs.to_string(), k.to_string()])
                .l2_cache_key_on(&device.stream)
                .map_err(device_error)?,
        );
    }
    serde_json::to_string(&identities).map_err(device_error)
}
fn fingerprint(device: &CudaDevice, shape: ProjectionShape, tile: LinearTiling) -> Result<String> {
    use crate::constants::SMALL_GEMM_TILE;
    use crate::resident::{linear_batch::batched, pool_gemv, small_gemm_kernels};
    let [m, n, k] = shape;
    let input = api::meta::<f32>(&[m, k])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let weight = api::meta::<bf16>(&[n, k])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut output = api::meta::<f32>(&[m, n])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let generics = vec![
        bf16::DTYPE.as_str().into(),
        tile.rows().to_string(),
        tile.columns().to_string(),
        k.to_string(),
    ];
    let gemv =
        pool_gemv::kernels::dense((&mut output).partition([1, tile.rows()]), &input, &weight)
            .generics(generics.clone())
            .l2_cache_key_on(&device.stream)
            .map_err(device_error)?;
    let shared = batched::dense(
        (&mut output).partition([super::SHARED_ROWS, tile.rows()]),
        &input,
        &weight,
    )
    .generics(generics)
    .l2_cache_key_on(&device.stream)
    .map_err(device_error)?;
    let native =
        small_gemm_kernels::dense((&mut output).partition(SMALL_GEMM_TILE), &input, &weight)
            .generics(vec![bf16::DTYPE.as_str().into(), k.to_string()])
            .l2_cache_key_on(&device.stream)
            .map_err(device_error)?;
    let mut identities = vec![gemv, shared, native];
    // Wider tiles are candidates only outside the qualified family, whose identity stays frozen.
    if !super::supported(shape) {
        for width in [super::NATIVE_WIDE_64, super::NATIVE_WIDE_128] {
            identities.push(
                small_gemm_kernels::dense_wide(
                    (&mut output).partition([super::NATIVE_TILE_ROWS, width]),
                    &input,
                    &weight,
                )
                .generics(vec![
                    bf16::DTYPE.as_str().into(),
                    width.to_string(),
                    k.to_string(),
                ])
                .l2_cache_key_on(&device.stream)
                .map_err(device_error)?,
            );
        }
    }
    serde_json::to_string(&identities).map_err(device_error)
}

/// Reconstruct a validated tile from persisted dimensions.
pub(super) fn tile(key: &Key) -> Result<LinearTiling> {
    LinearTiling::new(key.tile[0], key.tile[1])
}
