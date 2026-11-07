//! Own every temporary until the captured graph is destroyed.
use super::{Choice, Key};
use crate::constants::SMALL_GEMM_TILE;
use crate::device::{CudaDevice, device_error};
use crate::mlp::Matrix;
use crate::resident::buffers::RowMajor;
use crate::strategy::LinearTiling;
use crate::tuning::preparation::driver_build;
use crate::tuning::projection_budget::ProjectionShape;
use crate::tuning::resident_gemm::{ResidentGemmKey, kernel_environment};
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::identity;
use infer_cuda_kernels::tile::jit::JitOptions;
use infer_cuda_kernels::tile::runtime::bf16;
use infer_cuda_kernels::tile::runtime::{Allocation, Graph};
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

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
    pub(super) graph: Graph,
    output: Allocation<f32>,
    _input: Allocation<f32>,
    _weight: Matrix<bf16>,
}
impl Captured {
    pub(super) fn new(
        device: &CudaDevice,
        weight: &Matrix<bf16>,
        shape: ProjectionShape,
        tile: LinearTiling,
        choice: Choice,
    ) -> Result<Self> {
        let [m, n, k] = shape;
        if !(super::searchable(shape) || choice.prompt_family() && super::prompt_searchable(shape))
            || weight.shape() != [n, k]
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
        let runtime = device.runtime();
        let input = runtime.upload(values)?;
        let output = runtime.zeros(m * n)?;
        let kernels = device.kernels();
        let graph = device.tile().capture(|scope| {
            choice.record(
                (&kernels, scope),
                weight,
                input.operand(),
                RowMajor::new(output.operand(), m, n)?,
                k,
                tile,
            )
        })?;
        Ok(Self {
            graph,
            output,
            _input: input,
            _weight: weight.clone(),
        })
    }
    pub(super) fn run(&self, device: &CudaDevice) -> Result<()> {
        device.tile().sync(self.graph.launch())?;
        Ok(())
    }
    pub(super) fn output(&self, device: &CudaDevice) -> Result<Vec<f32>> {
        let values = device.runtime().read(&self.output)?;
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
    let compile = JitOptions::default();
    native.kernel_schema = format!(
        "{SCHEMA}:jit-opt{:?}:debug{}",
        compile.opt_level(),
        compile.device_debug()
    );
    native.activation_mode = MODE.into();
    Ok(Key {
        native,
        tile: [tile.rows(), tile.columns()],
        baseline: base,
    })
}
fn prompt_fingerprint(device: &CudaDevice, shape: ProjectionShape) -> Result<String> {
    let tiles = [
        Choice::PromptBase,
        Choice::PromptM64N128,
        Choice::PromptM32N256,
        Choice::PromptM128N64,
    ]
    .map(Choice::prompt_tile);
    let identities = identity::dense_prompt_candidates(device.tile(), shape, &tiles)?;
    serde_json::to_string(&identities).map_err(device_error)
}
fn fingerprint(device: &CudaDevice, shape: ProjectionShape, tile: LinearTiling) -> Result<String> {
    let wide: &[usize] = if super::supported(shape) {
        &[]
    } else {
        &[super::NATIVE_WIDE_64, super::NATIVE_WIDE_128]
    };
    let identities = identity::dense_candidates(
        device.tile(),
        shape,
        identity::DenseSearch {
            tile_rows: tile.rows(),
            tile_columns: tile.columns(),
            shared_rows: super::SHARED_ROWS,
            native_tile: SMALL_GEMM_TILE,
            wide_rows: super::NATIVE_TILE_ROWS,
            wide,
        },
    )?;
    serde_json::to_string(&identities).map_err(device_error)
}

/// Reconstruct a validated tile from persisted dimensions.
pub(super) fn tile(key: &Key) -> Result<LinearTiling> {
    LinearTiling::new(key.tile[0], key.tile[1])
}
