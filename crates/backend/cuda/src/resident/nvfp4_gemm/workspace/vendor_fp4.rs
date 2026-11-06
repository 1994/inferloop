//! Capture-local byte views of the already funded, unused fused product.
use super::Workspace;
use crate::constants::{NVFP4_QUANT_CODES_TILE, NVFP4_QUANT_SCALES_TILE};
use crate::device::{CudaDevice, vendor_fp8::nvfp4};
use crate::mlp::ProjectionWeight;
use crate::resident::ActivationArena;
use crate::resident::arena::Fp4Scratch;
use crate::resident::buffers::RowMajor;
use crate::resident::capture::error;
use crate::tuning::resident_gemm::QUANTIZATION_ROWS;
use crate::tuning::resident_gemm::{GemmConfig, WeightKey};
use cutile::prelude::*;
use infer_cuda_kernels::contract::Buffer;
use infer_cuda_kernels::contract::Recording;
use infer_cuda_kernels::tile::projection::fp4_probe::{self, ProbeGeometry};

impl Workspace {
    /// Screen and warm the vendor FP4 down projection that reads arena slot `input`, an F32
    /// product the fused plan never materializes, into the `[rows, outputs]` `output`.
    pub(crate) fn prepare_vendor_down(
        &mut self,
        device: &CudaDevice,
        weight: &ProjectionWeight,
        scale: f32,
        (arena, input): (&ActivationArena, usize),
        output: Buffer<'_, f32>,
        (rows, columns): (usize, usize),
    ) -> Result<(), DeviceError> {
        let ProjectionWeight::Fp4(w, _, _) = weight else {
            return Ok(());
        };
        let outputs = w.rows();
        if !outputs.is_multiple_of(nvfp4::SCALE_ROWS) || !self.vendor_fp4.enabled() {
            return Ok(());
        }
        // Probe and pack accesses to the unused product are ordered on its owning stream.
        let Some(scratch) = Fp4Scratch::new(arena, input, rows, columns).map_err(error)? else {
            return Ok(());
        };
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
            .get(&(rows, columns))
            .ok_or_else(|| error("FP4 vendor quantization"))?;
        let extents = [rows, outputs, columns];
        let candidate = {
            let op = nvfp4::Operands {
                weight,
                scale,
                q: q.operand(),
                qs: qs.operand(),
                scratch: scratch.bytes(),
                output,
                extents,
            };
            if self.vendor_fp4.known(&op)?.is_some() {
                return Ok(());
            }
            self.vendor_fp4.candidate(&op, native)?
        };
        let Some(candidate) = candidate else {
            return Ok(());
        };
        if candidate.needs_probe() {
            fp4_probe::initialize(
                &device.runtime(),
                scratch.floats(),
                (q.buffer(), qs.buffer()),
                scale,
                ProbeGeometry {
                    rows: QUANTIZATION_ROWS,
                    columns,
                    code_tile: NVFP4_QUANT_CODES_TILE[1],
                    scale_tile: NVFP4_QUANT_SCALES_TILE[1],
                },
            )
            .map_err(error)?;
        }
        self.vendor_fp4.prepare(
            &nvfp4::Operands {
                weight,
                scale,
                q: q.operand(),
                qs: qs.operand(),
                scratch: scratch.bytes(),
                output,
                extents,
            },
            candidate,
        )?;
        Ok(())
    }

    /// Record the vendor FP4 down projection over arena slot `input`, a sole-consumer product
    /// the fused plan never materializes, into the `[rows, outputs]` `output`; `false` when no
    /// vendor plan covers it.
    pub(crate) fn record_vendor_down(
        &self,
        scope: &Recording<'_>,
        weight: &ProjectionWeight,
        scale: f32,
        (arena, input): (&ActivationArena, usize),
        output: RowMajor<'_, f32>,
        columns: usize,
    ) -> Result<bool, DeviceError> {
        if !self.vendor_fp4.enabled() {
            return Ok(false);
        }
        let rows = output.rows;
        // Probe and pack accesses to the unused product are ordered on its owning stream.
        let Some(scratch) = Fp4Scratch::new(arena, input, rows, columns).map_err(error)? else {
            return Ok(false);
        };
        let (q, qs) = self
            .buffers
            .get(&(rows, columns))
            .ok_or_else(|| error("FP4 vendor quantization"))?;
        self.vendor_fp4.record(
            scope,
            &nvfp4::Operands {
                weight,
                scale,
                q: q.operand(),
                qs: qs.operand(),
                scratch: scratch.bytes(),
                output: output.buffer,
                extents: [rows, output.columns, columns],
            },
        )
    }
}
