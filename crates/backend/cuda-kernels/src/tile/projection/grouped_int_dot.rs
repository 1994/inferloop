//! Grouped-int projections of two to eight rows through the hand-written int8 dot-product kernel.
//!
//! The kernel (`kernels/grouped-int-dot/grouped_int_dot.cu` of the CUDA backend, embedded as
//! nvcc PTX) quantizes the activations to int8 per 32 values and multiplies them by the codes with
//! `dp4a`, so an extra row costs integer dot products instead of an F32 multiply-add per weight;
//! see the kernel source for the layout and numeric contract. It loads once per device; a device
//! whose driver cannot load the PTX keeps the cuTile row GEMV.
#![expect(
    unsafe_code,
    reason = "Launches a loaded PTX kernel over buffers whose extents are checked here"
)]
use super::contract::{RowsShape, split_codes, validate_grouped_int};
use crate::contract::buffer::{Buffer, Owner};
use crate::contract::grouped_int::{CodePacking, GROUPED_INT_TILE_ROWS, GroupedIntWeight};
use crate::contract::recording::{Foreign, Recording};
use crate::tile::backend_error;
use crate::tile::runtime::{Argument, KernelLaunch};
use cuda_core::sys::{
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as MULTIPROCESSORS,
    cuDeviceGetAttribute, cudaError_enum_CUDA_SUCCESS as CUDA_SUCCESS,
};
use cuda_core::{Device, Function};
use infer_core::{Error, Result};
use infer_ir::{CodeOffset, GROUPED_INT_SPAN, GroupedIntFormat};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

/// The kernel's PTX, built by `tools/cuda/kernels.py` from the registered source.
const PTX: &str =
    include_str!("../../../../cuda/kernels/grouped-int-dot/grouped-int-dot-sm120.ptx");
/// Threads of one CTA: eight warps.
const THREADS: u32 = 256;
/// Output tiles one wide CTA covers, sharing each quantized span; a narrow CTA covers one tile
/// with eight reduction splits instead.
const WIDE_TILES: usize = 4;
/// Activation rows of each compiled variant; a launch takes the smallest that holds its rows.
const ROW_VARIANTS: [usize; 3] = [2, 4, 8];
/// Alignment of the activation and term rows the kernel reads as `float4`.
const VECTOR_BYTES: u64 = 16;
/// Entry name prefixes, by packing and offset kind; the variants append `_r{rows}`.
const ENTRIES: [(&str, Kind); 6] = [
    ("grouped_int_dot_nibble_zero", Kind::NibbleZero),
    ("grouped_int_dot_nibble_bias", Kind::NibbleBias),
    ("grouped_int_dot_high1_zero", Kind::HighOneZero),
    ("grouped_int_dot_high1_bias", Kind::HighOneBias),
    ("grouped_int_dot_high2_zero", Kind::HighTwoZero),
    ("grouped_int_dot_byte_zero", Kind::ByteZero),
];

/// The packing and offset kind of one compiled entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    NibbleZero,
    NibbleBias,
    HighOneZero,
    HighOneBias,
    HighTwoZero,
    ByteZero,
}

impl Kind {
    /// The entry for `format`, when one is compiled.
    const fn of(format: GroupedIntFormat) -> Option<Self> {
        let bias = matches!(format.offset(), CodeOffset::Bias);
        match (CodePacking::of(format), bias) {
            (CodePacking::Nibble, false) => Some(Self::NibbleZero),
            (CodePacking::Nibble, true) => Some(Self::NibbleBias),
            (CodePacking::NibbleHigh { high_bits: 1 }, false) => Some(Self::HighOneZero),
            (CodePacking::NibbleHigh { high_bits: 1 }, true) => Some(Self::HighOneBias),
            // Two high bits exist only as `Q6_K`, whose sixteen-value groups the entry assumes.
            (CodePacking::NibbleHigh { high_bits: 2 }, false)
                if format.group_values() == HIGH_TWO_GROUP =>
            {
                Some(Self::HighTwoZero)
            }
            (CodePacking::Byte, false) => Some(Self::ByteZero),
            _ => None,
        }
    }
}

/// Values per term group the two-high-bit entry reads.
const HIGH_TWO_GROUP: usize = 16;

/// How many tiles one CTA of an entry covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Width {
    /// [`WIDE_TILES`] tiles over two reduction halves.
    Wide,
    /// One tile over eight reduction splits.
    Narrow,
}

impl Width {
    /// The entry-name suffix of this width.
    const fn suffix(self) -> &'static str {
        match self {
            Self::Wide => "",
            Self::Narrow => "_n",
        }
    }

    /// Output tiles one CTA covers.
    const fn tiles(self) -> usize {
        match self {
            Self::Wide => WIDE_TILES,
            Self::Narrow => 1,
        }
    }

    /// The width for `tiles` output tiles: wide CTAs share each quantized span among four tiles,
    /// unless fewer than one of them per two multiprocessors would run. On RTX 5090 (170 SMs, three
    /// rows) the 2048x6144 `Q6_K` down projection took 33 us in 32 wide CTAs and 18 us narrow,
    /// while the 6144x2048 `Q4_K` gate took 11.6 us in 96 wide CTAs and 12.9 us narrow.
    const fn of(tiles: usize, multiprocessors: usize) -> Self {
        if tiles.div_ceil(WIDE_TILES) * 2 >= multiprocessors {
            Self::Wide
        } else {
            Self::Narrow
        }
    }
}

/// The loaded entries of one device, keyed by kind, row variant and width, and its
/// multiprocessor count.
struct Entries {
    functions: BTreeMap<(Kind, usize, Width), Arc<Function>>,
    multiprocessors: usize,
}

/// Entries per device ordinal; `None` when the device cannot load the PTX.
fn entries(ordinal: usize) -> Result<Option<Arc<Entries>>> {
    static LOADED: OnceLock<Mutex<BTreeMap<usize, Option<Arc<Entries>>>>> = OnceLock::new();
    let cache = LOADED.get_or_init(Mutex::default);
    let poisoned = || Error::invariant("grouped-int dot module cache poisoned");
    let cached = cache.lock().map_err(|_| poisoned())?.get(&ordinal).cloned();
    if let Some(entries) = cached {
        return Ok(entries);
    }
    // Loading runs outside the lock; a concurrent first use keeps whichever module lands first.
    let entries = load(ordinal)?;
    Ok(cache
        .lock()
        .map_err(|_| poisoned())?
        .entry(ordinal)
        .or_insert(entries)
        .clone())
}

/// Load every entry on `ordinal`; a module the driver refuses means the device lacks the kernel.
fn load(ordinal: usize) -> Result<Option<Arc<Entries>>> {
    let device = Device::new(ordinal).map_err(backend_error)?;
    let Ok(module) = device.load_module_from_ptx_src(PTX) else {
        return Ok(None);
    };
    let mut functions = BTreeMap::new();
    for (name, kind) in ENTRIES {
        for rows in ROW_VARIANTS {
            for width in [Width::Wide, Width::Narrow] {
                let function = module
                    .load_function(&format!("{name}_r{rows}{}", width.suffix()))
                    .map_err(backend_error)?;
                functions.insert((kind, rows, width), Arc::new(function));
            }
        }
    }
    Ok(Some(Arc::new(Entries {
        functions,
        multiprocessors: multiprocessors(&device)?,
    })))
}

/// The multiprocessor count of `device`.
fn multiprocessors(device: &Device) -> Result<usize> {
    let mut value = 0;
    // SAFETY: `device` is live and `value` is a live output.
    let status =
        unsafe { cuDeviceGetAttribute(&raw mut value, MULTIPROCESSORS, device.cu_device()) };
    if status != CUDA_SUCCESS {
        return Err(Error::invariant(format!(
            "multiprocessor count query failed: {status}"
        )));
    }
    usize::try_from(value).map_err(|_| Error::invalid("negative multiprocessor count"))
}

/// Whether the dot-product kernel takes `rows` rows of `format` on `ordinal`.
/// # Errors
/// Returns the driver's failure to open the device or to find an entry of a loaded module.
pub(super) fn available(ordinal: usize, format: GroupedIntFormat, rows: usize) -> Result<bool> {
    let largest = ROW_VARIANTS[ROW_VARIANTS.len() - 1];
    if rows < ROW_VARIANTS[0] || rows > largest || Kind::of(format).is_none() {
        return Ok(false);
    }
    Ok(entries(ordinal)?.is_some())
}

/// One launch of a loaded entry, enqueued on the stream the recording hands it.
struct Launch {
    function: Arc<Function>,
    grid: u32,
    arguments: Vec<Argument>,
}

impl Foreign for Launch {
    fn enqueue(self, stream: *mut c_void) -> Result<()> {
        // SAFETY: `project` built `arguments` in the entry's parameter order and types, over
        // buffers it checked against the kernel's extents and the recording retains.
        let launch = unsafe {
            KernelLaunch::new(
                self.function,
                stream as usize,
                self.grid,
                THREADS,
                self.arguments,
            )
        };
        launch.enqueue_on(stream)
    }
}

/// Record the `[rows, outputs]` projection of `x` by `weight` into `out`.
/// # Errors
/// Rejects a row count or format without an entry, extents that disagree with the shape or
/// exceed the kernel's `i32` indices, misaligned rows, or a device without the kernel.
pub(super) fn project(
    ordinal: usize,
    recording: &Recording<'_>,
    shape: RowsShape,
    out: Buffer<'_, f32>,
    x: Buffer<'_, f32>,
    weight: GroupedIntWeight<'_>,
) -> Result<()> {
    let RowsShape {
        rows,
        outputs,
        columns,
    } = shape;
    let format = weight.format;
    let kind = Kind::of(format)
        .ok_or_else(|| Error::unsupported(format!("{} int8 dot kernel", format.label())))?;
    let variant = ROW_VARIANTS
        .into_iter()
        .find(|&variant| (ROW_VARIANTS[0]..=variant).contains(&rows))
        .ok_or_else(|| Error::unsupported(format!("int8 dot kernel over {rows} rows")))?;
    let entries =
        entries(ordinal)?.ok_or_else(|| Error::unsupported("int8 dot kernel on this device"))?;
    let tiles = outputs / GROUPED_INT_TILE_ROWS;
    let width = Width::of(tiles, entries.multiprocessors);
    let function = entries
        .functions
        .get(&(kind, variant, width))
        .ok_or_else(|| Error::invariant("int8 dot entry missing from its module"))?;
    validate_grouped_int(&weight, outputs, columns, 1)?;
    let (codes, high) = split_codes(&weight, outputs * columns)?;
    if out.len() != rows * outputs
        || x.len() != rows * columns
        || !columns.is_multiple_of(GROUPED_INT_SPAN)
        || !outputs.is_multiple_of(GROUPED_INT_TILE_ROWS)
        || ![x.address(), weight.factors.address()]
            .into_iter()
            .chain(weight.biases.map(Buffer::address))
            .all(|address| address.is_multiple_of(VECTOR_BYTES))
    {
        return Err(Error::unsupported(format!(
            "{} int8 dot kernel operands",
            format.label()
        )));
    }
    let index = |value: usize| i32::try_from(value).map_err(|_| Error::invalid("int8 dot extent"));
    let grid = u32::try_from(tiles.div_ceil(width.tiles()))
        .map_err(|_| Error::invalid("int8 dot grid"))?;
    // The entry indexes every operand with `i32` products of these extents.
    index(rows * columns.max(outputs))?;
    let zero = match format.offset() {
        CodeOffset::Zero(zero) => i32::from(zero),
        CodeOffset::Bias => 0,
    };
    let address = |buffer: Option<u64>| Argument::Address(buffer.unwrap_or(0));
    let arguments = vec![
        Argument::buffer(out),
        Argument::buffer(x),
        Argument::buffer(codes),
        address(high.map(Buffer::address)),
        Argument::buffer(weight.factors),
        address(weight.biases.map(Buffer::address)),
        Argument::I32(index(rows)?),
        Argument::I32(index(outputs)?),
        Argument::I32(index(columns)?),
        Argument::I32(zero),
    ];
    recording.retain(out);
    recording.retain(x);
    recording.retain(weight.codes);
    recording.retain(weight.factors);
    if let Some(biases) = weight.biases {
        recording.retain(biases);
    }
    let keep: Vec<Owner> = vec![Arc::clone(function) as Owner];
    recording.foreign(
        Launch {
            function: Arc::clone(function),
            grid,
            arguments,
        },
        keep,
    )
}
