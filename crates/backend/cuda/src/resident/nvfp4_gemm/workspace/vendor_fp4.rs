//! Capture-local byte views of the already funded, unused fused product.
use super::Workspace;
use crate::constants::{NVFP4_QUANT_CODES_TILE, NVFP4_QUANT_SCALES_TILE};
use crate::device::{CudaDevice, vendor_fp8::nvfp4};
use crate::mlp::ProjectionWeight;
use crate::resident::arena::{Fp4Output, Fp4Scratch};
use crate::resident::capture::error;
use crate::resident::nvfp4_gemm::kernels;
use crate::tuning::resident_gemm::QUANTIZATION_ROWS;
use crate::tuning::resident_gemm::{GemmConfig, WeightKey};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;
use infer_cuda_kernels::tile::runtime::Graph;

impl Workspace {
    #[expect(
        unsafe_code,
        reason = "The exclusive, locked prompt arena outlives captures; synchronous preparation completes all alias use"
    )]
    pub(crate) fn prepare_vendor_down(
        &mut self,
        device: &CudaDevice,
        weight: &ProjectionWeight,
        scale: f32,
        input: &mut Tensor<f32>,
        output: &mut Tensor<f32>,
        (rows, columns): (usize, usize),
    ) -> Result<(), DeviceError> {
        let ProjectionWeight::Fp4(w, _, _) = weight else {
            return Ok(());
        };
        let outputs = usize::try_from(w.shape()[0]).map_err(error)?;
        if !outputs.is_multiple_of(nvfp4::SCALE_ROWS) || !self.vendor_fp4.enabled() {
            return Ok(());
        }
        // SAFETY: The fused plan never materializes this product; the locked arena retains it
        // beyond all captures, and all probe/pack accesses are ordered on its owning stream.
        let Some(mut scratch) = unsafe { Fp4Scratch::new(input, rows, columns) }.map_err(error)?
        else {
            return Ok(());
        };
        // SAFETY: Preparation keeps the exclusive output owner live until all temporary graphs
        // complete and are destroyed before this function returns.
        let mut output = unsafe { Fp4Output::new(output, rows, outputs) }.map_err(error)?;
        let native = self
            .strategies
            .get(&(
                WeightKey::new(weight, scale).map_err(error)?,
                rows,
                outputs,
                columns,
            ))
            .copied()
            .unwrap_or_else(|| GemmConfig::baseline(rows));
        let (q, qs) = self
            .buffers
            .get_mut(&(rows, columns))
            .ok_or_else(|| error("FP4 vendor quantization"))?;
        let candidate = {
            let mut op = nvfp4::Operands {
                weight,
                scale,
                q,
                qs,
                scratch: scratch.bytes(),
                output: output.tensor(),
            };
            if self.vendor_fp4.known(&op)?.is_some() {
                return Ok(());
            }
            self.vendor_fp4.candidate(&mut op, native)?
        };
        let Some(candidate) = candidate else {
            return Ok(());
        };
        if candidate.needs_probe() {
            initialize(device, scratch.floats(), (q, qs), scale, columns)?;
        }
        self.vendor_fp4.prepare(
            nvfp4::Operands {
                weight,
                scale,
                q,
                qs,
                scratch: scratch.bytes(),
                output: output.tensor(),
            },
            candidate,
        )?;
        Ok(())
    }

    #[expect(
        unsafe_code,
        reason = "The fused plan proves a sole-consumer product; its source arena outlives every same-stream replay"
    )]
    pub(crate) fn record_vendor_down(
        &self,
        scope: &Scope,
        weight: &ProjectionWeight,
        scale: f32,
        input: &mut Tensor<f32>,
        output: &mut Tensor<f32>,
        columns: usize,
    ) -> Result<bool, DeviceError> {
        if !self.vendor_fp4.enabled() {
            return Ok(false);
        }
        let rows = usize::try_from(output.shape()[0]).map_err(error)?;
        // SAFETY: The fused plan never materializes this product; the locked arena retains it
        // beyond all captures, and all probe/pack accesses are ordered on its owning stream.
        let Some(mut scratch) = unsafe { Fp4Scratch::new(input, rows, columns) }.map_err(error)?
        else {
            return Ok(false);
        };
        let (q, qs) = self
            .buffers
            .get(&(rows, columns))
            .ok_or_else(|| error("FP4 vendor quantization"))?;
        self.vendor_fp4.record(
            scope,
            nvfp4::Operands {
                weight,
                scale,
                q,
                qs,
                scratch: scratch.bytes(),
                output,
            },
        )
    }
}
fn initialize(
    device: &CudaDevice,
    input: &mut Tensor<f32>,
    (q, qs): (&mut Tensor<f4e2m1fnx2>, &mut Tensor<f8e4m3fn>),
    scale: f32,
    columns: usize,
) -> Result<(), DeviceError> {
    let graph = Graph::scope(&device.stream, |scope| {
        scope.record(
            nvfp4::probe_input(
                (&mut *input).partition([QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1] * 2]),
            )
            .generics(vec![columns.to_string()]),
        )?;
        scope.record(
            kernels::quantize_rows(
                (&mut *q).partition([QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1]]),
                (&mut *qs).partition([QUANTIZATION_ROWS, NVFP4_QUANT_SCALES_TILE[1]]),
                &*input,
                scale,
            )
            .generics(vec![columns.to_string()]),
        )?;
        Ok(())
    })?;
    graph.launch().sync_on(&device.stream)?;
    Ok(())
}
