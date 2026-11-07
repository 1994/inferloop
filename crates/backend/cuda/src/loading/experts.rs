//! Bind expert banks once; kernels read only the selected matrices.
use super::floats;
use crate::constants::GIB;
use crate::mlp::Matrix;
use crate::{
    constants::{F32_BYTES, FP8_BLOCK_COLUMNS},
    device::CudaDevice,
    mlp::ProjectionWeight,
    resident::moe::ExpertBank,
};
use cuda_core::f8e4m3fn;
use infer_core::{Error, ErrorCode, Result, TensorId};
use infer_cuda_kernels::tile::runtime::bf16;
use infer_ir::{TensorOp, TensorStorage};
use infer_models::{DeviceWeight, QuantizedPackage, WeightEncoding};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn load(
    device: &CudaDevice,
    package: &mut QuantizedPackage,
) -> Result<(BTreeMap<TensorId, ExpertBank>, BTreeSet<TensorId>)> {
    let specs: BTreeMap<_, _> = package
        .graph
        .tensors
        .iter()
        .map(|spec| (spec.id, spec.clone()))
        .collect();
    let nodes = package.graph.nodes.clone();
    let mut banks = BTreeMap::new();
    let mut bank_ids = BTreeSet::new();
    for node in nodes {
        let TensorOp::ExpertLinear {
            experts,
            input,
            output,
            ..
        } = node.op
        else {
            continue;
        };
        let mut parts = Vec::new();
        for id in &node.inputs[2..] {
            let TensorStorage::Weight { slot } = &specs[id].storage else {
                return Err(Error::invalid("expert weight storage"));
            };
            parts.push(
                package
                    .weights
                    .get(slot)
                    .cloned()
                    .ok_or_else(|| Error::invalid("expert weight binding"))?,
            );
            bank_ids.insert(*id);
        }
        if parts.len() != experts || experts == 0 {
            return Err(Error::invalid("expert bank count"));
        }
        let first = &parts[0];
        if parts.iter().any(|part| {
            part.encoding != first.encoding
                || part.fp8_token_input != first.fp8_token_input
                || part.shape != [output, input]
        }) {
            return Err(Error::unsupported("mixed expert bank encodings"));
        }
        let quantized_inputs = first.fp8_token_input;
        let rows = experts
            .checked_mul(output)
            .ok_or_else(|| Error::invalid("expert bank overflow"))?;
        admit(device, first.encoding, rows, input)?;
        let projection = match first.encoding {
            WeightEncoding::Float => {
                let mut data = Vec::new();
                for part in &parts {
                    data.extend(floats(package, &part.data)?.into_iter().map(bf16::from_f32));
                }
                ProjectionWeight::Dense(Matrix::upload(device, data, &[rows, input])?)
            }
            WeightEncoding::Fp8Channel | WeightEncoding::Fp8Block => {
                fp8(device, package, &parts, rows, input, output)?
            }
            WeightEncoding::Nvfp4 => {
                return Err(Error::unsupported(
                    "NVFP4 expert activation quantization requires an explicit numerical profile",
                ));
            }
        };
        let bank = ExpertBank {
            projection,
            experts,
            rows: output,
            columns: input,
            quantized_inputs,
        };
        bank.validate(experts, output, input)?;
        banks.insert(node.inputs[2], bank);
    }
    Ok((banks, bank_ids))
}

fn admit(device: &CudaDevice, encoding: WeightEncoding, rows: usize, columns: usize) -> Result<()> {
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| Error::invalid("expert bank size overflow"))?;
    let element_bytes = if encoding == WeightEncoding::Float {
        2
    } else {
        1
    };
    let scales = match encoding {
        WeightEncoding::Fp8Channel => rows,
        WeightEncoding::Fp8Block => rows
            .checked_mul(columns / FP8_BLOCK_COLUMNS)
            .ok_or_else(|| Error::invalid("expert scale size overflow"))?,
        _ => 0,
    };
    let bytes = elements
        .checked_mul(element_bytes)
        .and_then(|n| scales.checked_mul(F32_BYTES).and_then(|s| n.checked_add(s)))
        .ok_or_else(|| Error::invalid("expert storage size overflow"))?;
    let available = device
        .memory_info()?
        .0
        .saturating_sub(device.profile()?.device_headroom_bytes());
    if bytes as u64 > available {
        return Err(Error::new(
            ErrorCode::Capacity,
            format!("expert bank needs {bytes} bytes, available {available} after device headroom"),
        ));
    }
    Ok(())
}

fn fp8(
    device: &CudaDevice,
    package: &mut QuantizedPackage,
    parts: &[DeviceWeight],
    rows: usize,
    input: usize,
    output: usize,
) -> Result<ProjectionWeight> {
    let encoding = parts[0].encoding;
    let mut data = Vec::new();
    let mut scaling = Vec::new();
    for part in parts {
        data.extend(
            package
                .read_view(&part.data, 2 * GIB as u64)?
                .iter()
                .copied()
                .map(f8e4m3fn),
        );
        let scales = floats(
            package,
            part.scale
                .as_ref()
                .ok_or_else(|| Error::invalid("expert FP8 scale"))?,
        )?;
        if scales.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err(Error::invalid(
                "expert FP8 scales must be finite and positive",
            ));
        }
        if encoding == WeightEncoding::Fp8Block {
            let [block_rows, block_columns] = part
                .weight_block
                .ok_or_else(|| Error::invalid("expert block scale"))?;
            if block_columns != FP8_BLOCK_COLUMNS
                || block_rows == 0
                || !input.is_multiple_of(block_columns)
            {
                return Err(Error::unsupported(
                    "expert FP8 block columns must divide input width in blocks of 128",
                ));
            }
            let blocks = input / block_columns;
            for row in 0..output {
                let start = (row / block_rows) * blocks;
                scaling.extend_from_slice(
                    scales
                        .get(start..start + blocks)
                        .ok_or_else(|| Error::invalid("expert FP8 scale bounds"))?,
                );
            }
        } else {
            if scales.len() != output {
                return Err(Error::invalid("expert FP8 channel scales"));
            }
            scaling.extend(scales);
        }
    }
    let data = Matrix::upload(device, data, &[rows, input])?;
    if encoding == WeightEncoding::Fp8Channel {
        Ok(ProjectionWeight::Fp8(
            data,
            device.runtime().upload(scaling)?,
        ))
    } else {
        Ok(ProjectionWeight::Fp8Block(
            data,
            Matrix::upload(device, scaling, &[rows, input / FP8_BLOCK_COLUMNS])?,
        ))
    }
}
