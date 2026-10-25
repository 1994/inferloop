//! Vision-encoder checks on real hardware: weight binding plus per-stage golden alignment.
//!
//! Every stage is compared against the official `transformers` implementation's output for the
//! same input, so a wrong op is localized instead of surfacing as a wrong final embedding.
#[cfg(cuda_backend)]
use cutile::jit_cache::enable_default;
#[cfg(cuda_backend)]
use infer_backend_cuda::device::CudaDevice as DeviceCudaDevice;
#[cfg(cuda_backend)]
use infer_backend_cuda::vision::VisionWeights as VisionVisionWeights;
#[cfg(cuda_backend)]
use std::env::args as env_args;
#[cfg(cuda_backend)]
use std::env::var;
#[cfg(cuda_backend)]
use std::env::var_os;
#[cfg(cuda_backend)]
use std::error::Error as StdError;
#[cfg(cuda_backend)]
use std::fs::read;
#[cfg(cuda_backend)]
fn main() -> Result<(), Box<dyn StdError>> {
    use cutile::half::bf16;
    use infer_backend_cuda::{device::CudaDevice, vision::VisionWeights};
    use infer_core::{Error, ModelId};
    use infer_models::{QuantizedPackage, vision as inventory};

    let mut args = env_args().skip(1);
    let root = args
        .next()
        .ok_or_else(|| Error::invalid("model package path required"))?;
    let golden = args
        .next()
        .or_else(|| var("INFER_VISION_GOLDEN").ok())
        .ok_or_else(|| Error::invalid("golden safetensors path required"))?;
    let case_index: usize = match args.next() {
        Some(value) => value
            .parse()
            .map_err(|error| Error::invalid(format!("golden case index: {error}")))?,
        None => 0,
    };
    enable_default()?;
    let device = CudaDevice::new(0)?;
    let mut package = QuantizedPackage::open(&root, ModelId::ONE)?;
    let encoder = package
        .imported
        .modalities
        .iter()
        .find_map(|plan| plan.encoder.clone())
        .ok_or_else(|| Error::unsupported("package declares no vision encoder"))?;

    let weights = VisionWeights::load(&device, &mut package, &encoder)?;
    let declared = inventory::slots(&encoder)?.len();
    if weights.len() != declared {
        return Err(Error::invariant("vision binding count disagrees with the inventory").into());
    }

    // Binding spot check: one device tensor must equal the checkpoint bytes exactly.
    let slot = "patch_embed.proj.bias";
    let source = package.source(&format!("{}{slot}", encoder.prefix))?;
    let bytes = package.read(&source, 512 * 1024 * 1024)?;
    let expected: Vec<bf16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16::from_bits(u16::from_le_bytes(*pair)))
        .collect();
    if expected != device.read(weights.get(slot)?)? {
        return Err(Error::invariant("vision weight upload mismatch").into());
    }

    let case = golden_case(&golden, case_index)?;
    if var_os("INFER_VISION_TRACE_BLOCKS").is_some() {
        trace_blocks(&device, &weights, &encoder, &case, &golden, case_index)?;
    }
    let errors = run_stages(&device, &weights, &encoder, &case, &root)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "passed": errors.worst().is_finite() && errors.worst() <= TOLERANCE,
            "package": root,
            "golden": golden,
            "bound_tensors": weights.len(),
            "patches": case.patches,
            "grid": [case.grid.0, case.grid.1, case.grid.2],
            "hidden_size": encoder.hidden_size,
            "heads": encoder.heads,
            "patch_embed_max_abs": errors.patch_embed.0,
            "patch_embed_max_rel": errors.patch_embed.1,
            "post_pos_max_abs": errors.post_pos.0,
            "post_pos_max_rel": errors.post_pos.1,
            "norm1_max_abs": errors.norm1.0,
            "norm1_max_rel": errors.norm1.1,
            "qkv_max_abs": errors.qkv.0,
            "qkv_max_rel": errors.qkv.1,
            "attention_max_abs": errors.attention.0,
            "attention_max_rel": errors.attention.1,
            "attention_prefix_max_rel": errors.attention_prefix.1,
            "attention_tail_max_rel": errors.attention_tail.1,
            "block_max_abs": errors.block.0,
            "block_max_rel": errors.block.1,
            "tower_max_abs": errors.tower.0,
            "tower_max_rel": errors.tower.1,
            "merger_max_abs": errors.merger.0,
            "merger_max_rel": errors.merger.1,
            "merger_from_reference_max_abs": errors.merger_reference.0,
            "merger_from_reference_max_rel": errors.merger_reference.1,
            "tower_rms": errors.tower_rms,
            "tolerance_rel": TOLERANCE,
            "attention_mode": errors.attention_mode,
        }))?
    );
    let worst = errors.worst();
    if !worst.is_finite() || worst > TOLERANCE {
        return Err(Error::invariant(format!("vision stages deviate by {worst} relative")).into());
    }
    Ok(())
}

/// Compare each block with the same reference input, separately from accumulated drift.
#[cfg(cuda_backend)]
fn trace_blocks(
    device: &DeviceCudaDevice,
    weights: &VisionVisionWeights,
    encoder: &infer_spi::ModalityEncoder,
    case: &GoldenCase,
    golden: &str,
    index: usize,
) -> infer_core::Result<()> {
    use infer_backend_cuda::vision;
    let mut file = infer_models::SafetensorsFile::open(golden)?;
    let mut reference_input = case.post_pos.clone();
    let mut accumulated = reference_input.clone();
    for layer in 0..encoder.depth {
        let reference = file
            .read_f32(&format!("c{index}/blocks/{layer}"), 1 << 30)?
            .data;
        let isolated = vision::block(
            device,
            weights,
            encoder,
            layer,
            &reference_input,
            case.grid,
            case.patches,
        )?;
        accumulated = vision::block(
            device,
            weights,
            encoder,
            layer,
            &accumulated,
            case.grid,
            case.patches,
        )?;
        eprintln!(
            "block {layer}: isolated={:?} accumulated={:?}",
            max_error(&isolated, &reference),
            max_error(&accumulated, &reference)
        );
        reference_input = reference;
    }
    Ok(())
}

/// Relative tolerance of every stage against the F32 reference.
///
/// Hidden-state magnitudes grow through the tower, so agreement is judged against each
/// reference's own scale; the absolute maxima are reported alongside.
#[cfg(cuda_backend)]
const TOLERANCE: f32 = 1e-2;

/// Maximum absolute and reference-relative error of one stage.
#[cfg(cuda_backend)]
type StageError = (f32, f32);

/// Per-stage errors of the stages that are wired so far.
#[cfg(cuda_backend)]
struct StageErrors {
    patch_embed: StageError,
    post_pos: StageError,
    norm1: StageError,
    qkv: StageError,
    attention: StageError,
    attention_prefix: StageError,
    attention_tail: StageError,
    attention_mode: &'static str,
    block: StageError,
    tower: StageError,
    merger: StageError,
    merger_reference: StageError,
    tower_rms: f32,
}

#[cfg(cuda_backend)]
impl StageErrors {
    fn worst(&self) -> f32 {
        self.stages()
            .iter()
            .fold(0f32, |worst, stage| worst.max(stage.1))
    }

    const fn stages(&self) -> [StageError; 9] {
        [
            self.patch_embed,
            self.post_pos,
            self.norm1,
            self.qkv,
            self.attention,
            self.block,
            self.tower,
            self.merger,
            self.merger_reference,
        ]
    }
}

/// Run patch embed, position add, `LayerNorm`, `QKV` and attention against the reference.
#[cfg(cuda_backend)]
fn run_stages(
    device: &DeviceCudaDevice,
    weights: &VisionVisionWeights,
    encoder: &infer_spi::ModalityEncoder,
    case: &GoldenCase,
    root: &str,
) -> infer_core::Result<StageErrors> {
    use infer_backend_cuda::vision;
    use infer_models::vision as inventory;

    let taps = inventory::position_taps(case.grid, encoder)?;
    let positions = inventory::gather_positions(&position_table(root, encoder)?, encoder, &taps)?;
    let embedded = vision::patch_embed(device, encoder, weights, &case.pixels, &[], case.patches)?;
    let posed = vision::patch_embed(
        device,
        encoder,
        weights,
        &case.pixels,
        &positions,
        case.patches,
    )?;
    let normalised = vision::layernorm(
        device,
        weights,
        encoder,
        "blocks.0.norm1.weight",
        "blocks.0.norm1.bias",
        &posed,
        case.patches,
    )?;
    let qkv = vision::project(
        device,
        weights,
        "blocks.0.attn.qkv.weight",
        "blocks.0.attn.qkv.bias",
        &normalised,
        case.patches,
        encoder.hidden_size * 3,
    )?;
    let mode = vision::AttentionMode::from_environment();
    let attended = vision::attention_with(
        device,
        weights,
        encoder,
        0,
        &normalised,
        case.grid,
        case.patches,
        mode,
    )?;
    let block = vision::block(device, weights, encoder, 0, &posed, case.grid, case.patches)?;
    let tower = vision::tower(device, weights, encoder, &posed, case.grid, case.patches)?;
    let pooled = vision::merger(device, weights, encoder, &tower, case.patches)?;
    // Same merger fed the reference states: separates merger error from tower error amplification.
    let pooled_from_reference =
        vision::merger(device, weights, encoder, &case.last_hidden, case.patches)?;
    Ok(StageErrors {
        patch_embed: max_error(&embedded, &case.patch_embed),
        post_pos: max_error(&posed, &case.post_pos),
        norm1: max_error(&normalised, &case.norm1),
        qkv: max_error(&qkv, &case.qkv),
        attention: max_error(&attended, &case.attn_proj),
        attention_mode: match mode {
            vision::AttentionMode::Online => "Online",
            vision::AttentionMode::Exact => "Exact",
        },
        attention_prefix: block_error(&attended, &case.attn_proj, encoder.hidden_size, 0),
        attention_tail: block_error(&attended, &case.attn_proj, encoder.hidden_size, 1),
        block: max_error(&block, &case.block),
        tower: max_error(&tower, &case.last_hidden),
        merger: max_error(&pooled, &case.pooler),
        merger_reference: max_error(&pooled_from_reference, &case.pooler),
        tower_rms: rms_error(&tower, &case.last_hidden),
    })
}

/// Golden case-0 inputs and the stage references.
#[cfg(cuda_backend)]
struct GoldenCase {
    pixels: Vec<f32>,
    patch_embed: Vec<f32>,
    post_pos: Vec<f32>,
    norm1: Vec<f32>,
    qkv: Vec<f32>,
    attn_proj: Vec<f32>,
    block: Vec<f32>,
    last_hidden: Vec<f32>,
    pooler: Vec<f32>,
    patches: usize,
    grid: (usize, usize, usize),
}

/// Load golden case 0.
#[cfg(cuda_backend)]
fn golden_case(golden: &str, index: usize) -> infer_core::Result<GoldenCase> {
    use infer_core::Error;
    use infer_models::SafetensorsFile;
    let mut file = SafetensorsFile::open(golden)?;
    let prefix = format!("c{index}");
    let pixels = file.read_f32(&format!("{prefix}/pixel_values"), 1 << 30)?;
    let patch_embed = file.read_f32(&format!("{prefix}/patch_embed"), 1 << 30)?;
    let post_pos = file.read_f32(&format!("{prefix}/post_pos"), 1 << 30)?;
    let norm1 = file.read_f32(&format!("{prefix}/block0_norm1"), 1 << 30)?;
    let qkv = file.read_f32(&format!("{prefix}/block0_qkv"), 1 << 30)?;
    let attn_proj = file.read_f32(&format!("{prefix}/block0_attn_proj"), 1 << 30)?;
    let block = file.read_f32(&format!("{prefix}/block0"), 1 << 30)?;
    let last_hidden = file.read_f32(&format!("{prefix}/last_hidden_state"), 1 << 30)?;
    let pooler = file.read_f32(&format!("{prefix}/pooler_output"), 1 << 30)?;
    let grid = file.read_bytes(&format!("{prefix}/grid_thw"), 1 << 12)?;
    let grid: Vec<i64> = grid
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| i64::from_le_bytes(*chunk))
        .collect();
    let grid = (
        usize::try_from(grid[0]).map_err(|_| Error::invalid("golden grid"))?,
        usize::try_from(grid[1]).map_err(|_| Error::invalid("golden grid"))?,
        usize::try_from(grid[2]).map_err(|_| Error::invalid("golden grid"))?,
    );
    let patches = pixels
        .shape
        .first()
        .copied()
        .ok_or_else(|| Error::invalid("golden pixel shape"))?;
    Ok(GoldenCase {
        pixels: pixels.data,
        patch_embed: patch_embed.data,
        post_pos: post_pos.data,
        norm1: norm1.data,
        qkv: qkv.data,
        attn_proj: attn_proj.data,
        block: block.data,
        last_hidden: last_hidden.data,
        pooler: pooler.data,
        patches,
        grid,
    })
}

/// Learned position table of the checkpoint, as F32.
#[cfg(cuda_backend)]
fn position_table(
    root: &str,
    encoder: &infer_spi::ModalityEncoder,
) -> infer_core::Result<Vec<f32>> {
    use infer_core::Error;
    use infer_models::{SafetensorsFile, SafetensorsIndex};
    use std::path::Path;
    let root = Path::new(root);
    // Sharded packages carry an index; single-file packages do not.
    let index = root.join("model.safetensors.index.json");
    let name = format!("{}pos_embed.weight", encoder.prefix);
    let shard = if index.exists() {
        let index = SafetensorsIndex::parse(
            &read(&index).map_err(|error| Error::invalid(error.to_string()))?,
        )?;
        index
            .weight_map
            .get(&name)
            .ok_or_else(|| Error::invalid("missing position table"))?
            .clone()
    } else {
        "model.safetensors".to_owned()
    };
    let mut file = SafetensorsFile::open(root.join(shard))?;
    Ok(file.read_f32(&name, 1 << 30)?.data)
}

/// Relative error over the 32-token row block `index` of a `[rows, width]` activation.
#[cfg(cuda_backend)]
fn block_error(actual: &[f32], reference: &[f32], width: usize, index: usize) -> StageError {
    let (start, end) = (index * 32 * width, (index + 1) * 32 * width);
    if end > actual.len().min(reference.len()) {
        return (0.0, 0.0);
    }
    max_error(&actual[start..end], &reference[start..end])
}

/// Root-mean-square error, which separates broad drift from a few outliers.
#[cfg(cuda_backend)]
fn rms_error(actual: &[f32], reference: &[f32]) -> f32 {
    let sum: f32 = actual
        .iter()
        .zip(reference.iter())
        .map(|(value, gold)| (value - gold) * (value - gold))
        .sum();
    let count = actual.iter().zip(reference).fold(0.0f32, |n, _| n + 1.0);
    (sum / count.max(1.0)).sqrt()
}

/// Absolute and reference-relative maximum error.
#[cfg(cuda_backend)]
fn max_error(actual: &[f32], reference: &[f32]) -> StageError {
    if actual.len() != reference.len() || actual.is_empty() {
        return (f32::INFINITY, f32::INFINITY);
    }
    let mut max_abs = 0f32;
    for (value, gold) in actual.iter().zip(reference.iter()) {
        if !value.is_finite() || !gold.is_finite() {
            return (f32::INFINITY, f32::INFINITY);
        }
        max_abs = max_abs.max((value - gold).abs());
    }
    let scale = reference
        .iter()
        .fold(0f32, |scale, value| scale.max(value.abs()));
    (
        max_abs,
        if scale > 0.0 {
            max_abs / scale
        } else {
            max_abs
        },
    )
}

#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA vision checks require Linux");
    exit(1);
}

#[cfg(all(test, cuda_backend))]
mod tests {
    use super::{max_error, rms_error};

    #[test]
    fn rms_uses_every_element_in_large_tensors() {
        let actual = vec![1.0; 131_072];
        let reference = vec![0.0; actual.len()];
        assert!((rms_error(&actual, &reference) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn invalid_comparisons_cannot_pass_the_gate() {
        assert!(max_error(&[1.0], &[]).1.is_infinite());
        assert!(max_error(&[f32::NAN], &[1.0]).1.is_infinite());
        assert!(max_error(&[1.0], &[f32::INFINITY]).1.is_infinite());
        assert!((max_error(&[1.0], &[0.0]).1 - 1.0).abs() < f32::EPSILON);
    }
}
