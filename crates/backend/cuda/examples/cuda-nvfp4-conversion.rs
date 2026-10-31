//! Exercise Hopper's weight conversion on an actual model matrix without loading the full model.
#[cfg(cuda_backend)]
use infer_backend_cuda::nvfp4::to_bf16;
#[cfg(cuda_backend)]
use std::env::args;
#[cfg(cuda_backend)]
use std::error::Error as StdError;
#[cfg(cuda_backend)]
fn main() -> Result<(), Box<dyn StdError>> {
    use cutile::half::bf16;
    use infer_backend_cuda::{device::CudaDevice, strategy::LinearTiling};
    use infer_core::{Error, ModelId};
    use infer_models::{QuantizedPackage, WeightEncoding, convert_float_bytes};
    let root = args()
        .nth(1)
        .ok_or_else(|| Error::invalid("model path is required"))?;
    let mut package = QuantizedPackage::open(&root, ModelId::ONE)?;
    let (name, weight) = package
        .weights
        .iter()
        .filter(|(_, w)| w.encoding == WeightEncoding::Nvfp4)
        .max_by_key(|(_, w)| w.data.bytes)
        .map(|(name, w)| (name.clone(), w.clone()))
        .ok_or_else(|| Error::invalid("model has no NVFP4 projection"))?;
    let [rows, columns] = weight.shape.as_slice() else {
        return Err(Error::invalid("matrix shape").into());
    };
    let source = weight
        .scale
        .as_ref()
        .ok_or_else(|| Error::invalid("scale"))?;
    let global_source = weight
        .global_scale
        .as_ref()
        .ok_or_else(|| Error::invalid("global scale"))?;
    let packed = package.read(&weight.data, 1024 * 1024 * 1024)?;
    let scales = package.read(source, 1024 * 1024 * 1024)?;
    let global = package.read(global_source, 16)?;
    let global = convert_float_bytes(&global, global_source.dtype)?;
    let global: [u8; 4] = global.as_slice().try_into()?;
    let converted = to_bf16(
        &packed,
        &scales,
        f32::from_le_bytes(global),
        *rows,
        *columns,
        1024 * 1024 * 1024,
    )?;
    let device = CudaDevice::new(0)?;
    let input = (0..*columns)
        .map(|i| u16::try_from(i % 251).map(|v| (f32::from(v) * 0.17).sin() * 0.2))
        .collect::<Result<Vec<_>, _>>()?;
    let matrix = device.upload(
        converted.iter().copied().map(bf16::from_bits).collect(),
        &weight.shape,
    )?;
    let vector = device.upload(input.clone(), &[*columns])?;
    let actual =
        device.read(&device.matvec_tiled(vector, matrix, LinearTiling::new(16, 256)?)?)?;
    let error = compare(&converted, &input, &actual)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "passed": true, "device": device.name()?, "target": device.target().name(),
            "model": root, "projection": name, "shape": weight.shape,
            "packed_bytes": packed.len(), "converted_bytes": converted.len() * 2,
            "maximum_absolute_error_vs_bf16_f64_reference": error,
            "scope": "actual model weight conversion and BF16 GPU projection; not H200 execution or full model quality"
        }))?
    );
    Ok(())
}

#[cfg(cuda_backend)]
fn compare(weights: &[u16], input: &[f32], actual: &[f32]) -> infer_core::Result<f64> {
    if weights.len() / input.len() != actual.len() {
        return Err(infer_core::Error::invariant("output shape"));
    }
    let mut maximum = 0.0f64;
    for (row, value) in weights.chunks_exact(input.len()).zip(actual) {
        let expected: f64 = row
            .iter()
            .zip(input)
            .map(|(w, x)| f64::from(f32::from_bits(u32::from(*w) << 16)) * f64::from(*x))
            .sum();
        let difference = (f64::from(*value) - expected).abs();
        if !value.is_finite() || difference > 1e-4 * expected.abs().max(1.0) {
            return Err(infer_core::Error::invariant(
                "converted BF16 projection mismatch",
            ));
        }
        maximum = maximum.max(difference);
    }
    Ok(maximum)
}

#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA conversion check requires Linux and an NVIDIA device");
    exit(1);
}
