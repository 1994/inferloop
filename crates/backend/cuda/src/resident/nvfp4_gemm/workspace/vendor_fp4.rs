//! Capture-local byte views of the already funded, unused fused product.
use super::Workspace;
use crate::constants::{NVFP4_QUANT_CODES_TILE, NVFP4_QUANT_SCALES_TILE};
use crate::device::{CudaDevice, vendor_fp8::nvfp4};
use crate::mlp::ProjectionWeight;
use crate::resident::arena::{Fp4Output, Fp4Scratch};
use crate::resident::capture::error;
use crate::tuning::resident_gemm::QUANTIZATION_ROWS;
use crate::tuning::resident_gemm::{GemmConfig, WeightKey};
use cutile::prelude::*;
use infer_cuda_kernels::tile::projection::fp4_probe::{self, ProbeGeometry};

impl Workspace {
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
        let outputs = w.rows();
        if !outputs.is_multiple_of(nvfp4::SCALE_ROWS) || !self.vendor_fp4.enabled() {
            return Ok(());
        }
        // The fused plan never materializes this product, so its memory is free scratch.
        let Some(mut scratch) = Fp4Scratch::new(input, rows, columns).map_err(error)? else {
            return Ok(());
        };
        let mut output = Fp4Output::new(output, rows, outputs).map_err(error)?;
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
            let op = nvfp4::Operands {
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
            self.vendor_fp4.candidate(&op, native)?
        };
        let Some(candidate) = candidate else {
            return Ok(());
        };
        if candidate.needs_probe() {
            fp4_probe::initialize(
                &device.stream,
                scratch.floats(),
                (q, qs),
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
                q,
                qs,
                scratch: scratch.bytes(),
                output: output.tensor(),
            },
            candidate,
        )?;
        Ok(())
    }

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
        // The fused plan never materializes this product, so its memory is free scratch.
        let Some(mut scratch) = Fp4Scratch::new(input, rows, columns).map_err(error)? else {
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
                q,
                qs,
                scratch: scratch.bytes(),
                output,
            },
        )
    }
}
