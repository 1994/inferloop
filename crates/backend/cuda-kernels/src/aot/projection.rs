//! AOT implementation of the projection family; kernels live in `projection/*.cu`.
#![expect(
    unsafe_code,
    reason = "Passes buffers that already carry the contract's guarantee to the driver"
)]

use super::artifact::Manifest;
use super::driver::{Device, Function};
use crate::contract::buffer::{Bf16, Buffer, Fp4Pair, Fp8};
use crate::contract::kernels::{Projection, WeightRef};
use crate::contract::plan::{
    DownPlan, Fp4QuantPlan, Fp8QuantPlan, GemmPlan, LastRowsPlan, PoolGemvPlan, RowsPlan,
    THREE_ROWS,
};
use crate::contract::recording::Recording;
use infer_core::{Error, ErrorCode, Result};
use std::ffi::{CStr, c_void};
use std::fs;
use std::path::Path;

/// Manifest name of the only entry this implementation launches.
const ENTRY_NAME: &str = "pool_gemv_bf16";
/// Device symbol of [`ENTRY_NAME`].
const ENTRY: &CStr = c"pool_gemv_bf16";
/// One warp per output element.
const WARP_THREADS: u32 = 32;

/// Projection kernels loaded from an AOT artifact set.
pub struct AotProjection {
    function: Function,
}

impl AotProjection {
    /// Load the projection entry from `root`, for the architecture `device` reports.
    /// # Errors
    /// Fails when the manifest is missing, invalid or built for another architecture, a digest
    /// differs, or the cubin/module/function cannot be loaded.
    pub fn load(device: &Device, root: &Path) -> Result<Self> {
        let bytes = fs::read(root.join("manifest.json")).map_err(|error| {
            Error::new(
                ErrorCode::Backend,
                format!("AOT manifest {}: {error}", root.display()),
            )
        })?;
        let manifest = Manifest::parse(&bytes)?;
        manifest.select(device.arch()?)?;
        manifest.verify(root)?;
        let entry = manifest
            .entry(ENTRY_NAME)
            .ok_or_else(|| Error::unsupported("AOT projection entry missing"))?;
        let cubin = fs::read(root.join(&entry.cubin)).map_err(|error| {
            Error::new(
                ErrorCode::Backend,
                format!("AOT cubin {}: {error}", entry.cubin),
            )
        })?;
        let module = device.load_module(&cubin)?;
        Ok(Self {
            function: module.function(ENTRY)?,
        })
    }
}

impl Projection for AotProjection {
    fn dense(
        &self,
        _recording: &Recording<'_>,
        _plan: &DenseRowsPlan,
        _out: Buffer<'_, f32>,
        _x: Buffer<'_, f32>,
        _w: Buffer<'_, Bf16>,
    ) -> Result<()> {
        Err(not_compiled("dense"))
    }

    fn dense_pair(
        &self,
        _recording: &Recording<'_>,
        _plan: &PromptPlan,
        _outs: (Buffer<'_, f32>, Buffer<'_, f32>),
        _x: Buffer<'_, f32>,
        _weights: (Buffer<'_, Bf16>, Buffer<'_, Bf16>),
    ) -> Result<()> {
        Err(not_compiled("dense_pair"))
    }

    fn prompt_fp8(
        &self,
        _recording: &Recording<'_>,
        _plan: &PromptPlan,
        _out: Buffer<'_, f32>,
        _x: Buffer<'_, f32>,
        _weight: WeightRef<'_>,
    ) -> Result<()> {
        Err(not_compiled("prompt_fp8"))
    }

    fn prompt_fp4_split(
        &self,
        _recording: &Recording<'_>,
        _plan: &SplitGemmPlan,
        _out: Buffer<'_, f32>,
        _partial: Buffer<'_, f32>,
        _x: Buffer<'_, f32>,
        _weight: WeightRef<'_>,
    ) -> Result<()> {
        Err(not_compiled("prompt_fp4_split"))
    }

    fn quantize_fp4(
        &self,
        _recording: &Recording<'_>,
        _plan: &Fp4QuantPlan,
        _codes: Buffer<'_, Fp4Pair>,
        _scales: Buffer<'_, Fp8>,
        _input: Buffer<'_, f32>,
        _scale: f32,
    ) -> Result<()> {
        Err(not_compiled("quantize_fp4"))
    }

    fn quantize_swiglu(
        &self,
        _recording: &Recording<'_>,
        _plan: &Fp4QuantPlan,
        _codes: Buffer<'_, Fp4Pair>,
        _scales: Buffer<'_, Fp8>,
        _inputs: (Buffer<'_, f32>, Buffer<'_, f32>),
        _scale: f32,
    ) -> Result<()> {
        Err(not_compiled("quantize_swiglu"))
    }

    fn gemm_fp4(
        &self,
        _recording: &Recording<'_>,
        _plan: &GemmPlan,
        _out: Buffer<'_, f32>,
        _codes: (Buffer<'_, Fp4Pair>, Buffer<'_, Fp8>),
        _weight: WeightRef<'_>,
        _inverse_scale: f32,
    ) -> Result<()> {
        Err(not_compiled("gemm_fp4"))
    }

    fn gemm_swiglu(
        &self,
        _recording: &Recording<'_>,
        _plan: &GemmPlan,
        _out: (Buffer<'_, Fp4Pair>, Buffer<'_, Fp8>),
        _codes: (Buffer<'_, Fp4Pair>, Buffer<'_, Fp8>),
        _weights: (WeightRef<'_>, WeightRef<'_>),
        _scales: (f32, f32, f32),
    ) -> Result<()> {
        Err(not_compiled("gemm_swiglu"))
    }

    fn quantize_fp8(
        &self,
        _recording: &Recording<'_>,
        _plan: &Fp8QuantPlan,
        _codes: Buffer<'_, Fp8>,
        _scales: Buffer<'_, f32>,
        _input: Buffer<'_, f32>,
    ) -> Result<()> {
        Err(not_compiled("quantize_fp8"))
    }

    fn gemm_fp8(
        &self,
        _recording: &Recording<'_>,
        _plan: &GemmPlan,
        _out: Buffer<'_, f32>,
        _codes: (Buffer<'_, Fp8>, Buffer<'_, f32>),
        _weight: WeightRef<'_>,
        _block_scales: bool,
    ) -> Result<()> {
        Err(not_compiled("gemm_fp8"))
    }

    fn rows(
        &self,
        _recording: &Recording<'_>,
        _plan: &RowsPlan,
        _out: Buffer<'_, f32>,
        _x: Buffer<'_, f32>,
        _weight: WeightRef<'_>,
    ) -> Result<()> {
        Err(not_compiled("rows"))
    }

    fn row(
        &self,
        _recording: &Recording<'_>,
        _plan: &DownPlan,
        _out: Buffer<'_, f32>,
        _x: Buffer<'_, f32>,
        _weight: WeightRef<'_>,
    ) -> Result<()> {
        Err(not_compiled("row"))
    }

    fn pack_row(
        &self,
        _recording: &Recording<'_>,
        _shape: (usize, usize),
        _out: Buffer<'_, f32>,
        _input: Buffer<'_, f32>,
        _row: usize,
    ) -> Result<()> {
        Err(not_compiled("pack_row"))
    }

    fn unpack_row(
        &self,
        _recording: &Recording<'_>,
        _shape: (usize, usize),
        _out: Buffer<'_, f32>,
        _input: Buffer<'_, f32>,
        _row: usize,
    ) -> Result<()> {
        Err(not_compiled("unpack_row"))
    }

    fn pack_three(
        &self,
        _recording: &Recording<'_>,
        _columns: usize,
        _out: Buffer<'_, f32>,
        _inputs: [Buffer<'_, f32>; THREE_ROWS],
    ) -> Result<()> {
        Err(not_compiled("pack_three"))
    }

    fn unpack_three(
        &self,
        _recording: &Recording<'_>,
        _columns: usize,
        _outs: [Buffer<'_, f32>; THREE_ROWS],
        _input: Buffer<'_, f32>,
    ) -> Result<()> {
        Err(not_compiled("unpack_three"))
    }

    fn last_rows(
        &self,
        _recording: &Recording<'_>,
        _plan: &LastRowsPlan,
        _out: Buffer<'_, f32>,
        _rows: Buffer<'_, f32>,
        _metadata: Buffer<'_, i32>,
    ) -> Result<()> {
        Err(not_compiled("last_rows"))
    }

    fn pool_gemv_bf16(
        &self,
        recording: &Recording<'_>,
        plan: &PoolGemvPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        w: Buffer<'_, Bf16>,
    ) -> Result<()> {
        let out = out.require("projection output", plan.rows * plan.outputs)?;
        let x = x.require("projection input", plan.rows * plan.columns)?;
        let w = w.require("projection weight", plan.outputs * plan.columns)?;
        let rows = u32::try_from(plan.rows).map_err(|_| geometry("rows"))?;
        let elements = u32::try_from(plan.outputs).map_err(|_| geometry("outputs"))?;
        let mut outputs = i32::try_from(plan.outputs).map_err(|_| geometry("outputs"))?;
        let mut columns = i32::try_from(plan.columns).map_err(|_| geometry("columns"))?;
        let (mut x_pointer, mut w_pointer, mut out_pointer) =
            (x.address(), w.address(), out.address());
        let mut parameters = [
            (&raw mut x_pointer).cast::<c_void>(),
            (&raw mut w_pointer).cast::<c_void>(),
            (&raw mut out_pointer).cast::<c_void>(),
            (&raw mut columns).cast::<c_void>(),
            (&raw mut outputs).cast::<c_void>(),
        ];
        // SAFETY: the five parameters match `pool_gemv_bf16` and live through this call; the
        // caller upholds the device-address contract of `Projection`.
        unsafe {
            self.function.launch(
                recording,
                (elements, rows, 1),
                (WARP_THREADS, 1, 1),
                &mut parameters,
            )
        }
    }
}

/// This family's AOT artifact holds only [`ENTRY_NAME`] so far.
fn not_compiled(entry: &str) -> Error {
    Error::unsupported(format!("projection {entry} has no AOT kernel"))
}

fn geometry(field: &str) -> Error {
    Error::invalid(format!("projection {field} exceeds i32"))
}
