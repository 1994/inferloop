use super::kernels;
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::METADATA_FIELDS;
use crate::device::CudaDevice;
use crate::resident::kernels::aux::prefill_rope;
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_cuda_kernels::tile::runtime::Graph;
use std::error::Error;

const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const DIM: usize = 32;
const CAPACITY: usize = 128;
const LANES: usize = 32;

fn data(size: usize, phase: u16) -> Vec<f32> {
    (0..size)
        .map(|i| {
            (f32::from((u16::try_from(i % 97).expect("bounded") * 7 + phase) % 31) - 15.0) / 16.0
        })
        .collect()
}

fn reference(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    head: usize,
    position: usize,
    window: usize,
) -> Vec<f64> {
    let start = if window == 0 {
        0
    } else {
        (position + 1).saturating_sub(window)
    };
    let offset = head / (HEADS / KV_HEADS) * CAPACITY * DIM;
    let mut scores: Vec<f64> = (start..=position)
        .map(|p| {
            query
                .iter()
                .zip(&keys[offset + p * DIM..offset + (p + 1) * DIM])
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>()
                / f64::from(u32::try_from(DIM).expect("dimension")).sqrt()
        })
        .collect();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for score in &mut scores {
        *score = (*score - maximum).exp();
    }
    let total: f64 = scores.iter().sum();
    (0..DIM)
        .map(|d| {
            scores
                .iter()
                .enumerate()
                .map(|(i, s)| s * f64::from(values[offset + (start + i) * DIM + d]))
                .sum::<f64>()
                / total
        })
        .collect()
}

#[test]
#[ignore = "requires CUDA; run inside safe-run with --test-threads 1"]
fn chunk_attention_matches_reference_with_tails_windows_and_draft_offset()
-> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (start, count, offset, window) in [
        (0, 32, 0, 0),
        (3, 17, 0, 7),
        (31, 32, 0, 0),
        (0, 17, -1, 0),
        (32, 1, -1, 7),
        (3, 117, 0, 0),
        (1, 117, -1, 7),
    ] {
        check_case(&device, (start, count, offset, window))?;
    }
    Ok(())
}

fn check_case(device: &CudaDevice, case: (i32, i32, i32, i32)) -> Result<(), Box<dyn Error>> {
    let (start, count, offset, window) = case;
    let lanes = if count > 32 { 128 } else { LANES };
    let query = data(lanes * HEADS * DIM, 1);
    let key = data(lanes * KV_HEADS * DIM, 3);
    let value = data(lanes * KV_HEADS * DIM, 11);
    let mut expected_keys = data(KV_HEADS * CAPACITY * DIM, 13);
    let mut expected_values = data(KV_HEADS * CAPACITY * DIM, 17);
    let mut keys = api::copy_host_vec_to_device(&Arc::new(expected_keys.clone()))
        .sync_on(&device.stream)?
        .reshape(&[KV_HEADS, CAPACITY, DIM])?;
    let mut values = api::copy_host_vec_to_device(&Arc::new(expected_values.clone()))
        .sync_on(&device.stream)?
        .reshape(&[KV_HEADS, CAPACITY, DIM])?;
    let query_device = device.upload(query.clone(), &[lanes * HEADS, DIM])?;
    let key_device = device.upload(key.clone(), &[lanes, KV_HEADS, DIM])?;
    let value_device = device.upload(value.clone(), &[lanes, KV_HEADS, DIM])?;
    let metadata = device.upload(vec![start, count, offset, 0], &[4])?;
    let blocks = CAPACITY / KV_BLOCK_TOKENS;
    let identity: Vec<i32> = (0..blocks)
        .map(|b| i32::try_from(b).expect("blocks fit"))
        .collect();
    let table = device.upload(identity, &[blocks])?;
    kernels::append(
        (&mut keys).partition([1, CAPACITY, DIM]),
        (&mut values).partition([1, CAPACITY, DIM]),
        &key_device,
        &value_device,
        &metadata,
        &table,
        1.0,
        1.0,
    )
    .generics(vec![
        "f32".into(),
        DIM.to_string(),
        CAPACITY.to_string(),
        "0".into(),
        KV_BLOCK_TOKENS.to_string(),
        "1".into(),
    ])
    .sync_on(&device.stream)?;
    // A permuted table must move the appended rows with it, which is what proves the kernel
    // resolves positions through the table instead of assuming identity.
    let swapped: Vec<i32> = (0..blocks)
        .map(|b| i32::try_from(b ^ 1).expect("blocks fit"))
        .collect();
    let swapped_table = check_permuted_append(
        device,
        &key_device,
        &value_device,
        &metadata,
        &key,
        &swapped,
        case,
    )?;
    let mut output = api::zeros::<f32>(&[lanes * HEADS, DIM]).sync_on(&device.stream)?;
    kernels::decode(
        (&mut output).partition([1, DIM]),
        &query_device,
        &keys,
        &values,
        &metadata,
        &table,
        window,
        1.0,
        1.0,
    )
    .generics(vec![
        "f32".into(),
        DIM.to_string(),
        (HEADS / KV_HEADS).to_string(),
        HEADS.to_string(),
        KV_BLOCK_TOKENS.to_string(),
    ])
    .sync_on(&device.stream)?;
    for lane in 0..usize::try_from(count)? {
        let position = start + i32::try_from(lane)? + offset;
        if position < 0 {
            continue;
        }
        let position = usize::try_from(position)?;
        for head in 0..KV_HEADS {
            let src = (lane * KV_HEADS + head) * DIM;
            let dst = (head * CAPACITY + position) * DIM;
            expected_keys[dst..dst + DIM].copy_from_slice(&key[src..src + DIM]);
            expected_values[dst..dst + DIM].copy_from_slice(&value[src..src + DIM]);
        }
    }
    // Same inputs and same reference through the tensor-core twin. q and out are bound as
    // `[lanes, HEADS*DIM]` partitioned `[QT, DIM]` so the launch grid is (lanes/QT, HEADS) and
    // grid axis 1 is the head; binding them as `[lanes*HEADS, DIM]` collapses that axis, pid.1
    // becomes zero and every CTA reads KV head 0. Its compensated BF16 mma is held to the
    // relative tolerance the attention gate uses for the same product.
    // Both query-tile widths the capture path can pick: 32 rows per CTA is register bound
    // (32x256 f32 accumulators) while 16 is the mma minimum and doubles the CTA count.
    // Relocate the entire cache, including history before this chunk. A tensor-core query
    // must resolve both K and V through the same non-identity table.
    check_tiled_decode(
        device,
        &query_device,
        (&keys, &values, &table, &swapped_table),
        &metadata,
        (&query, &expected_keys, &expected_values),
        &swapped,
        (lanes, case),
    )?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA; run inside safe-run with --test-threads 1"]
fn chunk_rotary_uses_each_token_position_and_preserves_unrotated_tail() -> Result<(), Box<dyn Error>>
{
    const WIDTH: usize = 128;
    const HALF: usize = 32;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let input = data(LANES * HEADS * WIDTH, 5);
    let frequencies: Vec<_> = (0..HALF)
        .map(|i| 0.01 / f32::from(u16::try_from(i + 1).expect("rotary width")))
        .collect();
    let input_device = device.upload(input.clone(), &[LANES * HEADS, WIDTH])?;
    let frequency_device = device.upload(frequencies.clone(), &[HALF])?;
    let metadata = device.upload(vec![37_i32, 17, 0, 0], &[4])?;
    let mut output = api::zeros::<f32>(&[LANES * HEADS, WIDTH]).sync_on(&device.stream)?;
    prefill_rope(
        (&mut output).partition([1, HALF]),
        &input_device,
        &frequency_device,
        &metadata,
    )
    .generics(vec![WIDTH.to_string(), HALF.to_string(), HEADS.to_string()])
    .sync_on(&device.stream)?;
    let actual = device.read_borrowed(&output)?;
    for row in 0..LANES * HEADS {
        let position = f32::from(u16::try_from(37 + row / HEADS)?);
        for d in 0..WIDTH {
            let base = row * WIDTH;
            let expected = if d < HALF * 2 {
                let angle = position * frequencies[d % HALF];
                let other = if d < HALF {
                    -input[base + d + HALF]
                } else {
                    input[base + d - HALF]
                };
                input[base + d].mul_add(angle.cos(), other * angle.sin())
            } else {
                input[base + d]
            };
            assert!(
                (actual[base + d] - expected).abs() < 2e-5,
                "rotary row={row} dimension={d}"
            );
        }
    }
    Ok(())
}

fn check_permuted_append(
    device: &CudaDevice,
    key_device: &Tensor<f32>,
    value_device: &Tensor<f32>,
    metadata: &Tensor<i32>,
    key: &[f32],
    swapped: &[i32],
    (start, count, offset, _): (i32, i32, i32, i32),
) -> Result<Arc<Tensor<i32>>, Box<dyn Error>> {
    let blocks = swapped.len();
    let swapped_table = device.upload(swapped.to_vec(), &[blocks])?;
    let mut moved_keys =
        api::copy_host_vec_to_device(&Arc::new(data(KV_HEADS * CAPACITY * DIM, 13)))
            .sync_on(&device.stream)?
            .reshape(&[KV_HEADS, CAPACITY, DIM])?;
    let mut moved_values =
        api::copy_host_vec_to_device(&Arc::new(data(KV_HEADS * CAPACITY * DIM, 17)))
            .sync_on(&device.stream)?
            .reshape(&[KV_HEADS, CAPACITY, DIM])?;
    kernels::append(
        (&mut moved_keys).partition([1, CAPACITY, DIM]),
        (&mut moved_values).partition([1, CAPACITY, DIM]),
        key_device,
        value_device,
        metadata,
        &swapped_table,
        1.0,
        1.0,
    )
    .generics(vec![
        "f32".into(),
        DIM.to_string(),
        CAPACITY.to_string(),
        "0".into(),
        KV_BLOCK_TOKENS.to_string(),
        "1".into(),
    ])
    .sync_on(&device.stream)?;
    let mut expected_moved = data(KV_HEADS * CAPACITY * DIM, 13);
    for lane in 0..usize::try_from(count)? {
        let position = start + i32::try_from(lane)? + offset;
        if position < 0 {
            continue;
        }
        let position = usize::try_from(position)?;
        let logical = position / KV_BLOCK_TOKENS;
        let row = usize::try_from(swapped[logical]).expect("blocks fit") * KV_BLOCK_TOKENS
            + position % KV_BLOCK_TOKENS;
        for head in 0..KV_HEADS {
            let src = (lane * KV_HEADS + head) * DIM;
            let dst = (head * CAPACITY + row) * DIM;
            expected_moved[dst..dst + DIM].copy_from_slice(&key[src..src + DIM]);
        }
    }
    assert_eq!(
        moved_keys.to_host_vec().sync_on(&device.stream)?,
        expected_moved,
        "append ignored the block table"
    );
    let _ = moved_values;
    Ok(swapped_table)
}

type CacheViews<'a> = (
    &'a Tensor<f32>,
    &'a Tensor<f32>,
    &'a Tensor<i32>,
    &'a Tensor<i32>,
);
type HostCache<'a> = (&'a [f32], &'a [f32], &'a [f32]);

fn check_tiled_decode(
    device: &CudaDevice,
    query_device: &Tensor<f32>,
    cache: CacheViews<'_>,
    metadata: &Tensor<i32>,
    reference_data: HostCache<'_>,
    swapped: &[i32],
    geometry: (usize, (i32, i32, i32, i32)),
) -> Result<(), Box<dyn Error>> {
    let (keys, values, table, swapped_table) = cache;
    let (query, expected_keys, expected_values) = reference_data;
    let (lanes, (start, count, offset, window)) = geometry;
    let mut permuted_keys = vec![0.0_f32; expected_keys.len()];
    let mut permuted_values = vec![0.0_f32; expected_values.len()];
    let block_elements = KV_BLOCK_TOKENS * DIM;
    for head in 0..KV_HEADS {
        for (logical, &physical) in swapped.iter().enumerate() {
            let source = head * CAPACITY * DIM + logical * block_elements;
            let target = head * CAPACITY * DIM + usize::try_from(physical)? * block_elements;
            permuted_keys[target..target + block_elements]
                .copy_from_slice(&expected_keys[source..source + block_elements]);
            permuted_values[target..target + block_elements]
                .copy_from_slice(&expected_values[source..source + block_elements]);
        }
    }
    let permuted_keys = device.upload(permuted_keys, &[KV_HEADS, CAPACITY, DIM])?;
    let permuted_values = device.upload(permuted_values, &[KV_HEADS, CAPACITY, DIM])?;
    for (keys, values, table) in [
        (keys, values, table),
        (&permuted_keys, &permuted_values, swapped_table),
    ] {
        for qt in [16_usize, 32] {
            let mut tiled = api::zeros::<f32>(&[lanes, HEADS * DIM]).sync_on(&device.stream)?;
            let query_tiled = query_device.view(&[lanes, HEADS * DIM])?;
            let scale = 1.0 / f32::from(u16::try_from(DIM).expect("dim fits in u16")).sqrt();
            kernels::decode_tiled(
                (&mut tiled).partition([qt, DIM]),
                &query_tiled,
                keys,
                values,
                metadata,
                table,
                window,
                1.0,
                1.0,
                scale,
            )
            .generics(vec![
                "f32".into(),
                DIM.to_string(),
                (HEADS / KV_HEADS).to_string(),
                qt.to_string(),
                LANES.to_string(),
                KV_BLOCK_TOKENS.to_string(),
                super::CausalBlockBound::FullChunk.generic(),
            ])
            .sync_on(&device.stream)?;
            let tiled_actual = tiled.to_host_vec().sync_on(&device.stream)?;
            for lane in 0..lanes {
                let position = start + i32::try_from(lane)? + offset;
                for head in 0..HEADS {
                    let base = (lane * HEADS + head) * DIM;
                    let expected = if lane < usize::try_from(count)? && position >= 0 {
                        reference(
                            &query[base..base + DIM],
                            expected_keys,
                            expected_values,
                            head,
                            usize::try_from(position)?,
                            usize::try_from(window)?,
                        )
                    } else {
                        vec![0.0; DIM]
                    };
                    for (dim, expected) in expected.into_iter().enumerate() {
                        let error = (f64::from(tiled_actual[base + dim]) - expected).abs();
                        assert!(
                            error <= 1e-4 * expected.abs().max(1.0),
                            "tiled qt={qt} start={start} count={count} offset={offset} window={window} lane={lane} head={head} dim={dim} error={error}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

// Actual Qwen attention geometry, rather than the smaller F32-only fixture above.
use cuda_core::f8e4m3fn;
use cutile::bench::{BenchOptions, do_bench_paired};
use std::time::Duration;

type NumericResult<T = ()> = Result<T, Box<dyn Error>>;
const MODEL_HEADS: usize = 24;
const MODEL_KV_HEADS: usize = 4;
const MODEL_DIM: usize = 256;
const MODEL_CAPACITY: usize = 1024;
const KEY_SCALE: f32 = 0.5;
const VALUE_SCALE: f32 = 2.0;
const REFERENCE_TOLERANCE: f64 = 1e-4;

#[derive(Clone, Copy, Debug, serde::Serialize)]
struct ChunkCase {
    lanes: usize,
    base: i32,
    count: usize,
    offset: i32,
    window: usize,
    permuted: bool,
}

#[derive(Clone, Copy, Debug)]
enum AttentionVariant {
    Simt,
    Tiled {
        qt: usize,
        bound: super::CausalBlockBound,
    },
}
impl AttentionVariant {
    fn label(self) -> String {
        match self {
            Self::Simt => "production-simt".into(),
            Self::Tiled { qt, bound } => format!("tiled-qt{qt}-{bound:?}"),
        }
    }
}

fn fp8_decode(code: u8) -> f32 {
    let magnitude = code & 127;
    let exponent = magnitude >> 3;
    let mantissa = magnitude & 7;
    let value = if exponent == 0 {
        f32::from(mantissa) / 512.0
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    if code & 128 == 0 { value } else { -value }
}

/// Independent enumeration of finite E4M3 codes, including subnormals and ties-to-even.
fn fp8_encode(value: f32) -> f8e4m3fn {
    assert!(value.is_finite());
    let magnitude = f64::from(value.abs().min(448.0));
    let mut best = 0_u8;
    let mut distance = f64::INFINITY;
    for code in 0_u8..=126 {
        let error = (f64::from(fp8_decode(code)) - magnitude).abs();
        if error < distance || (error.total_cmp(&distance).is_eq() && code.is_multiple_of(2)) {
            best = code;
            distance = error;
        }
    }
    f8e4m3fn(best | if value.is_sign_negative() { 128 } else { 0 })
}

fn model_data(size: usize, phase: u16) -> Vec<f32> {
    (0..size)
        .map(|i| (f32::from((u16::try_from(i % 251).unwrap() * 23 + phase) % 197) - 98.0) / 64.0)
        .collect()
}

fn model_codes(phase: usize) -> Vec<f8e4m3fn> {
    (0..MODEL_KV_HEADS * MODEL_CAPACITY * MODEL_DIM)
        .map(|i| {
            f8e4m3fn(
                32 + u8::try_from((i * 11 + phase) % 31).unwrap()
                    + if (i + phase).is_multiple_of(3) {
                        128
                    } else {
                        0
                    },
            )
        })
        .collect()
}

fn physical_row(position: i32, table: &[i32]) -> Option<usize> {
    let logical = usize::try_from(position).ok()?;
    let block = usize::try_from(*table.get(logical / KV_BLOCK_TOKENS)?).ok()?;
    Some(block * KV_BLOCK_TOKENS + logical % KV_BLOCK_TOKENS)
}

fn relocate(codes: &[f8e4m3fn], table: &[i32]) -> Vec<f8e4m3fn> {
    let mut moved = vec![f8e4m3fn(0); codes.len()];
    for logical in 0..MODEL_CAPACITY {
        let physical = physical_row(i32::try_from(logical).unwrap(), table).unwrap();
        for head in 0..MODEL_KV_HEADS {
            let source = (head * MODEL_CAPACITY + logical) * MODEL_DIM;
            let target = (head * MODEL_CAPACITY + physical) * MODEL_DIM;
            moved[target..target + MODEL_DIM].copy_from_slice(&codes[source..source + MODEL_DIM]);
        }
    }
    moved
}

fn append_reference(
    case: ChunkCase,
    source: (&[f32], &[f32]),
    table: &[i32],
    logical: (&mut [f8e4m3fn], &mut [f8e4m3fn]),
) {
    let (keys, values) = logical;
    for lane in 0..case.count {
        let position = case.base + i32::try_from(lane).unwrap() + case.offset;
        if physical_row(position, table).is_none() {
            assert!(position < 0);
            continue;
        }
        let position = usize::try_from(position).unwrap();
        for head in 0..MODEL_KV_HEADS {
            for dimension in 0..MODEL_DIM {
                let input = (lane * MODEL_KV_HEADS + head) * MODEL_DIM + dimension;
                let output = (head * MODEL_CAPACITY + position) * MODEL_DIM + dimension;
                keys[output] = fp8_encode(source.0[input] / KEY_SCALE);
                values[output] = fp8_encode(source.1[input] / VALUE_SCALE);
            }
        }
    }
}

/// Operand owners outlive each captured executable; history and appended rows use one table.
struct ChunkFixture {
    case: ChunkCase,
    query: Arc<Tensor<f32>>,
    keys: Arc<Tensor<f32>>,
    values: Arc<Tensor<f32>>,
    metadata: Arc<Tensor<i32>>,
    table: Arc<Tensor<i32>>,
    host_query: Vec<f32>,
    initial_keys: Vec<f8e4m3fn>,
    initial_values: Vec<f8e4m3fn>,
    expected_keys: Vec<f8e4m3fn>,
    expected_values: Vec<f8e4m3fn>,
    logical_keys: Vec<f8e4m3fn>,
    logical_values: Vec<f8e4m3fn>,
}
impl ChunkFixture {
    fn new(device: &CudaDevice, case: ChunkCase) -> NumericResult<Arc<Self>> {
        assert!(case.count <= case.lanes);
        let blocks = MODEL_CAPACITY / KV_BLOCK_TOKENS;
        let table: Vec<i32> = (0..blocks)
            .map(|b| i32::try_from(if case.permuted { b ^ 1 } else { b }).unwrap())
            .collect();
        let host_query = model_data(case.lanes * MODEL_HEADS * MODEL_DIM, 1);
        let keys = model_data(case.lanes * MODEL_KV_HEADS * MODEL_DIM, 3);
        let values = model_data(case.lanes * MODEL_KV_HEADS * MODEL_DIM, 11);
        let mut logical_keys = model_codes(13);
        let mut logical_values = model_codes(17);
        let initial_keys = relocate(&logical_keys, &table);
        let initial_values = relocate(&logical_values, &table);
        append_reference(
            case,
            (&keys, &values),
            &table,
            (&mut logical_keys, &mut logical_values),
        );
        let query = device.upload(host_query.clone(), &[case.lanes, MODEL_HEADS * MODEL_DIM])?;
        let keys = device.upload(keys, &[case.lanes, MODEL_KV_HEADS, MODEL_DIM])?;
        let values = device.upload(values, &[case.lanes, MODEL_KV_HEADS, MODEL_DIM])?;
        let metadata = device.upload(
            vec![case.base, i32::try_from(case.count)?, case.offset, 0],
            &[METADATA_FIELDS],
        )?;
        let expected_keys = relocate(&logical_keys, &table);
        let expected_values = relocate(&logical_values, &table);
        let table = device.upload(table, &[blocks])?;
        Ok(Arc::new(Self {
            case,
            query,
            keys,
            values,
            metadata,
            table,
            host_query,
            initial_keys,
            initial_values,
            expected_keys,
            expected_values,
            logical_keys,
            logical_values,
        }))
    }

    fn record_append(
        &self,
        scope: &Scope,
        keys: &mut Tensor<f8e4m3fn>,
        values: &mut Tensor<f8e4m3fn>,
    ) -> Result<(), DeviceError> {
        scope.record(
            kernels::append(
                keys.partition([1, MODEL_CAPACITY, MODEL_DIM]),
                values.partition([1, MODEL_CAPACITY, MODEL_DIM]),
                &self.keys,
                &self.values,
                &self.metadata,
                &self.table,
                KEY_SCALE,
                VALUE_SCALE,
            )
            .generics(vec![
                f8e4m3fn::DTYPE.as_str().into(),
                MODEL_DIM.to_string(),
                MODEL_CAPACITY.to_string(),
                "1".into(),
                KV_BLOCK_TOKENS.to_string(),
                "1".into(),
            ]),
        )?;
        Ok(())
    }

    fn record_compute(
        &self,
        scope: &Scope,
        output: &mut Tensor<f32>,
        cache: (&Tensor<f8e4m3fn>, &Tensor<f8e4m3fn>),
        variant: AttentionVariant,
    ) -> Result<(), DeviceError> {
        let group = MODEL_HEADS / MODEL_KV_HEADS;
        let block = KV_BLOCK_TOKENS;
        let window = i32::try_from(self.case.window).map_err(|e| tensor_error(&e.to_string()))?;
        match variant {
            AttentionVariant::Simt => {
                let query = self
                    .query
                    .view(&[self.case.lanes * MODEL_HEADS, MODEL_DIM])?;
                scope.record(
                    kernels::decode(
                        output.partition([1, MODEL_DIM]),
                        &query,
                        cache.0,
                        cache.1,
                        &self.metadata,
                        &self.table,
                        window,
                        KEY_SCALE,
                        VALUE_SCALE,
                    )
                    .generics(vec![
                        f8e4m3fn::DTYPE.as_str().into(),
                        MODEL_DIM.to_string(),
                        group.to_string(),
                        MODEL_HEADS.to_string(),
                        block.to_string(),
                    ]),
                )?;
            }
            AttentionVariant::Tiled { qt, bound } => {
                let scale = 1.0 / f32::from(u16::try_from(MODEL_DIM).unwrap()).sqrt();
                scope.record(
                    kernels::decode_tiled(
                        output.partition([qt, MODEL_DIM]),
                        &self.query,
                        cache.0,
                        cache.1,
                        &self.metadata,
                        &self.table,
                        window,
                        KEY_SCALE,
                        VALUE_SCALE,
                        scale,
                    )
                    .generics(vec![
                        f8e4m3fn::DTYPE.as_str().into(),
                        MODEL_DIM.to_string(),
                        group.to_string(),
                        qt.to_string(),
                        block.to_string(),
                        block.to_string(),
                        bound.generic(),
                    ]),
                )?;
            }
        }
        Ok(())
    }
}

/// Executables drop before every KV/output/input allocation they reference.
struct AttentionGraph {
    graph: Graph,
    output: Tensor<f32>,
    keys: Tensor<f8e4m3fn>,
    values: Tensor<f8e4m3fn>,
    fixture: Arc<ChunkFixture>,
    variant: AttentionVariant,
}
impl AttentionGraph {
    fn new(
        device: &CudaDevice,
        fixture: &Arc<ChunkFixture>,
        variant: AttentionVariant,
    ) -> NumericResult<Self> {
        let mut keys = api::copy_host_vec_to_device(&Arc::new(fixture.initial_keys.clone()))
            .sync_on(&device.stream)?
            .reshape(&[MODEL_KV_HEADS, MODEL_CAPACITY, MODEL_DIM])?;
        let mut values = api::copy_host_vec_to_device(&Arc::new(fixture.initial_values.clone()))
            .sync_on(&device.stream)?
            .reshape(&[MODEL_KV_HEADS, MODEL_CAPACITY, MODEL_DIM])?;
        let shape = if matches!(variant, AttentionVariant::Simt) {
            [fixture.case.lanes * MODEL_HEADS, MODEL_DIM]
        } else {
            [fixture.case.lanes, MODEL_HEADS * MODEL_DIM]
        };
        let mut output = api::zeros::<f32>(&shape).sync_on(&device.stream)?;
        let graph = Graph::scope(&device.stream, |scope| {
            fixture.record_append(scope, &mut keys, &mut values)?;
            fixture.record_compute(scope, &mut output, (&keys, &values), variant)?;
            Ok(())
        })?;
        Ok(Self {
            graph,
            output,
            keys,
            values,
            fixture: Arc::clone(fixture),
            variant,
        })
    }

    fn check(&self, device: &CudaDevice, expected: &[f64]) -> NumericResult<Vec<f32>> {
        check_complete_graph(
            device,
            &self.fixture,
            &self.graph,
            &self.output,
            (&self.keys, &self.values),
            self.variant,
            expected,
        )
    }
}

fn check_complete_graph(
    device: &CudaDevice,
    fixture: &ChunkFixture,
    graph: &Graph,
    output: &Tensor<f32>,
    cache: (&Tensor<f8e4m3fn>, &Tensor<f8e4m3fn>),
    variant: AttentionVariant,
    expected: &[f64],
) -> NumericResult<Vec<f32>> {
    graph.launch().sync_on(&device.stream)?;
    let keys = device.read_borrowed(cache.0)?;
    let values = device.read_borrowed(cache.1)?;
    assert_eq!(keys, fixture.expected_keys, "complete FP8 key append");
    assert_eq!(values, fixture.expected_values, "complete FP8 value append");
    let actual = device.read_borrowed(output)?;
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0_f64;
    for (index, (&value, &reference)) in actual.iter().zip(expected).enumerate() {
        assert!(value.is_finite() && reference.is_finite());
        let error = (f64::from(value) - reference).abs();
        max_abs = max_abs.max(error);
        assert!(
            error <= REFERENCE_TOLERANCE * reference.abs().max(1.0),
            "{:?} {} index={index} abs={error} reference={reference}",
            fixture.case,
            variant.label(),
        );
    }
    eprintln!(
        "{}",
        serde_json::json!({
            "attention_numeric": variant.label(), "case": fixture.case,
            "heads": MODEL_HEADS, "kv_heads": MODEL_KV_HEADS, "dim": MODEL_DIM,
            "kv_format": "e4m3", "scales": [KEY_SCALE, VALUE_SCALE],
            "elements": actual.len(), "cache_elements": keys.len() + values.len(),
            "finite": true, "max_abs": max_abs, "tolerance": REFERENCE_TOLERANCE,
            "complete_append_codes": true,
        })
    );
    Ok(actual)
}

/// Cost arms share the exact KV/output addresses; all executables drop before their operands.
struct SharedAttentionGraphs {
    graphs: Vec<(AttentionVariant, Graph)>,
    output: Tensor<f32>,
    keys: Tensor<f8e4m3fn>,
    values: Tensor<f8e4m3fn>,
    fixture: Arc<ChunkFixture>,
}
impl SharedAttentionGraphs {
    fn new(device: &CudaDevice, fixture: &Arc<ChunkFixture>) -> NumericResult<Self> {
        let mut keys = api::copy_host_vec_to_device(&Arc::new(fixture.initial_keys.clone()))
            .sync_on(&device.stream)?
            .reshape(&[MODEL_KV_HEADS, MODEL_CAPACITY, MODEL_DIM])?;
        let mut values = api::copy_host_vec_to_device(&Arc::new(fixture.initial_values.clone()))
            .sync_on(&device.stream)?
            .reshape(&[MODEL_KV_HEADS, MODEL_CAPACITY, MODEL_DIM])?;
        let mut output = api::zeros::<f32>(&[fixture.case.lanes, MODEL_HEADS * MODEL_DIM])
            .sync_on(&device.stream)?;
        let mut graphs = Vec::new();
        let mut variants = vec![AttentionVariant::Simt];
        for qt in [16, 32] {
            for bound in [
                super::CausalBlockBound::FullChunk,
                super::CausalBlockBound::QueryTile,
            ] {
                variants.push(AttentionVariant::Tiled { qt, bound });
            }
        }
        for variant in variants {
            let shape = if matches!(variant, AttentionVariant::Simt) {
                [fixture.case.lanes * MODEL_HEADS, MODEL_DIM]
            } else {
                [fixture.case.lanes, MODEL_HEADS * MODEL_DIM]
            };
            output = output.reshape(&shape)?;
            let graph = Graph::scope(&device.stream, |scope| {
                fixture.record_append(scope, &mut keys, &mut values)?;
                fixture.record_compute(scope, &mut output, (&keys, &values), variant)?;
                Ok(())
            })?;
            graphs.push((variant, graph));
        }
        let bundle = Self {
            graphs,
            output,
            keys,
            values,
            fixture: Arc::clone(fixture),
        };
        for (variant, _) in &bundle.graphs {
            eprintln!(
                "{}",
                serde_json::json!({
                    "attention_cost_capture": variant.label(), "case": fixture.case,
                    "device_pointers": bundle.addresses(), "same_allocations": true,
                    "kv_elements": bundle.keys.size() + bundle.values.size(),
                    "output_elements": bundle.output.size(),
                })
            );
        }
        Ok(bundle)
    }

    fn addresses(&self) -> serde_json::Value {
        serde_json::json!({
            "keys": self.keys.device_pointer().cu_deviceptr(),
            "values": self.values.device_pointer().cu_deviceptr(),
            "output": self.output.device_pointer().cu_deviceptr(),
            "query": self.fixture.query.device_pointer().cu_deviceptr(),
            "append_keys": self.fixture.keys.device_pointer().cu_deviceptr(),
            "append_values": self.fixture.values.device_pointer().cu_deviceptr(),
            "metadata": self.fixture.metadata.device_pointer().cu_deviceptr(),
            "table": self.fixture.table.device_pointer().cu_deviceptr(),
        })
    }

    fn check(
        &self,
        device: &CudaDevice,
        index: usize,
        expected: &[f64],
    ) -> NumericResult<Vec<f32>> {
        let (variant, graph) = &self.graphs[index];
        check_complete_graph(
            device,
            &self.fixture,
            graph,
            &self.output,
            (&self.keys, &self.values),
            *variant,
            expected,
        )
    }
}

struct F64Attention<'a> {
    fixture: &'a ChunkFixture,
    keys: Vec<f64>,
    values: Vec<f64>,
}
impl<'a> F64Attention<'a> {
    fn new(fixture: &'a ChunkFixture) -> Self {
        Self {
            fixture,
            keys: fixture
                .logical_keys
                .iter()
                .map(|code| f64::from(fp8_decode(code.0)) * f64::from(KEY_SCALE))
                .collect(),
            values: fixture
                .logical_values
                .iter()
                .map(|code| f64::from(fp8_decode(code.0)) * f64::from(VALUE_SCALE))
                .collect(),
        }
    }

    fn row(&self, lane: usize, head: usize, position: usize) -> Vec<f64> {
        let window = self.fixture.case.window;
        let first = if window == 0 {
            0
        } else {
            (position + 1).saturating_sub(window)
        };
        let cache = head / (MODEL_HEADS / MODEL_KV_HEADS) * MODEL_CAPACITY * MODEL_DIM;
        let query = (lane * MODEL_HEADS + head) * MODEL_DIM;
        let query = &self.fixture.host_query[query..query + MODEL_DIM];
        let scale = f64::from(u16::try_from(MODEL_DIM).unwrap()).sqrt();
        let mut probabilities: Vec<f64> = (first..=position)
            .map(|key| {
                query
                    .iter()
                    .zip(&self.keys[cache + key * MODEL_DIM..cache + (key + 1) * MODEL_DIM])
                    .map(|(&q, &k)| f64::from(q) * k)
                    .sum::<f64>()
                    / scale
            })
            .collect();
        let maximum = probabilities
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        for probability in &mut probabilities {
            *probability = (*probability - maximum).exp();
        }
        let sum = probabilities.iter().sum::<f64>();
        let mut result = vec![0.0_f64; MODEL_DIM];
        for (key, probability) in probabilities.into_iter().enumerate() {
            let start = cache + (first + key) * MODEL_DIM;
            for (output, &value) in result
                .iter_mut()
                .zip(&self.values[start..start + MODEL_DIM])
            {
                *output = probability.mul_add(value, *output);
            }
        }
        for value in &mut result {
            *value /= sum;
        }
        result
    }

    fn complete_output(&self) -> Vec<f64> {
        let case = self.fixture.case;
        let mut result = vec![0.0_f64; case.lanes * MODEL_HEADS * MODEL_DIM];
        for lane in 0..case.count {
            let position = case.base + i32::try_from(lane).unwrap() + case.offset;
            if position < 0 {
                continue;
            }
            let position = usize::try_from(position).unwrap();
            assert!(position < MODEL_CAPACITY);
            for head in 0..MODEL_HEADS {
                let offset = (lane * MODEL_HEADS + head) * MODEL_DIM;
                result[offset..offset + MODEL_DIM].copy_from_slice(&self.row(lane, head, position));
            }
        }
        result
    }
}

/// Oracle counts blocks containing a visible key for any actual query; it also reports skipped
/// work without treating theoretical block counts as measured performance.
fn visible_blocks(case: ChunkCase, qt: usize, tile: usize) -> usize {
    (tile * qt..(tile + 1) * qt)
        .filter(|&lane| lane < case.count)
        .filter_map(|lane| {
            usize::try_from(case.base + i32::try_from(lane).unwrap() + case.offset).ok()
        })
        .map(|position| (position + 1).div_ceil(KV_BLOCK_TOKENS))
        .max()
        .unwrap_or(0)
}

#[test]
fn causal_block_reference_covers_boundaries_and_inactive_tiles() {
    let case = ChunkCase {
        lanes: 64,
        base: 0,
        count: 64,
        offset: 0,
        window: 0,
        permuted: false,
    };
    assert_eq!(
        (0..4)
            .map(|tile| visible_blocks(case, 16, tile))
            .collect::<Vec<_>>(),
        [1, 1, 2, 2]
    );
    assert_eq!(
        (0..2)
            .map(|tile| visible_blocks(case, 32, tile))
            .collect::<Vec<_>>(),
        [1, 2]
    );
    let tail = ChunkCase { count: 17, ..case };
    assert_eq!(
        (0..4)
            .map(|tile| visible_blocks(tail, 16, tile))
            .collect::<Vec<_>>(),
        [1, 1, 0, 0]
    );
    let shifted = ChunkCase {
        base: 31,
        offset: -1,
        ..tail
    };
    assert_eq!(visible_blocks(shifted, 16, 0), 2);
    assert_eq!(visible_blocks(shifted, 16, 1), 2);
    assert_eq!(
        visible_blocks(
            ChunkCase {
                count: 0,
                base: 64,
                ..case
            },
            16,
            0
        ),
        0
    );
    assert_eq!(
        visible_blocks(
            ChunkCase {
                count: 1,
                offset: -1,
                ..case
            },
            16,
            0
        ),
        0
    );
    let negative = ChunkCase {
        count: 32,
        offset: -17,
        ..case
    };
    assert_eq!(visible_blocks(negative, 16, 0), 0);
    assert_eq!(visible_blocks(negative, 16, 1), 1);
}

#[test]
fn fp8_reference_rounds_even_and_preserves_block_table_boundaries() {
    assert_eq!(fp8_encode(0.0).0, 0);
    assert_eq!(fp8_encode(-0.0).0, 128);
    assert_eq!(fp8_encode(1.0 / 1024.0).0, 0);
    assert_eq!(fp8_encode(3.0 / 1024.0).0, 2);
    assert_eq!(fp8_encode(448.0).0, 126);
    assert_eq!(fp8_encode(-1000.0).0, 254);
    for code in 0_u8..=126 {
        assert_eq!(fp8_encode(fp8_decode(code)).0, code);
        assert_eq!(fp8_encode(fp8_decode(code | 128)).0, code | 128);
    }
    let table = [1, 0];
    assert_eq!(physical_row(-1, &table), None);
    assert_eq!(physical_row(0, &table), Some(32));
    assert_eq!(physical_row(31, &table), Some(63));
    assert_eq!(physical_row(32, &table), Some(0));
    assert_eq!(physical_row(63, &table), Some(31));
    assert_eq!(physical_row(64, &table), None);
}

#[test]
fn tiled_model_geometry_keeps_query_and_gqa_head_axes_separate() {
    for width in [64, 256, 512] {
        for qt in [16, 32] {
            let shape = [width, MODEL_HEADS * MODEL_DIM];
            let partition = [qt, MODEL_DIM];
            assert_eq!(shape[0] / partition[0], width / qt);
            assert_eq!(shape[1] / partition[1], MODEL_HEADS);
            for head in 0..MODEL_HEADS {
                assert_eq!(head / (MODEL_HEADS / MODEL_KV_HEADS), head / 6);
            }
            assert_eq!(super::CausalBlockBound::FullChunk.generic(), "0");
            assert_eq!(super::CausalBlockBound::QueryTile.generic(), "1");
        }
    }
}

fn numeric_cases() -> Vec<ChunkCase> {
    [64, 256, 512]
        .into_iter()
        .flat_map(|lanes| {
            [
                ChunkCase {
                    lanes,
                    base: 0,
                    count: lanes,
                    offset: 0,
                    window: 0,
                    permuted: false,
                },
                ChunkCase {
                    lanes,
                    base: 31,
                    count: lanes - 1,
                    offset: -1,
                    window: 17,
                    permuted: true,
                },
                ChunkCase {
                    lanes,
                    base: 0,
                    count: 1,
                    offset: -1,
                    window: 0,
                    permuted: true,
                },
                ChunkCase {
                    lanes,
                    base: 64,
                    count: 0,
                    offset: -1,
                    window: 7,
                    permuted: true,
                },
                ChunkCase {
                    lanes,
                    base: 33,
                    count: 17,
                    offset: -1,
                    window: 0,
                    permuted: true,
                },
            ]
        })
        .collect()
}

fn assert_bitwise(case: ChunkCase, qt: usize, baseline: &[f32], candidate: &[f32]) {
    assert_eq!(baseline.len(), candidate.len());
    for (index, (&left, &right)) in baseline.iter().zip(candidate).enumerate() {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "{case:?} qt={qt} index={index}"
        );
    }
    eprintln!(
        "{}",
        serde_json::json!({
            "attention_bitwise": true, "case": case, "qt": qt,
            "elements": baseline.len(), "differences": 0,
            "baseline": "full-chunk-tiled", "candidate": "per-query-tile-upper-bound",
        })
    );
}

fn check_model_case(device: &CudaDevice, case: ChunkCase) -> NumericResult {
    let fixture = ChunkFixture::new(device, case)?;
    let expected = F64Attention::new(&fixture).complete_output();
    let simt = AttentionGraph::new(device, &fixture, AttentionVariant::Simt)?;
    simt.check(device, &expected)?;
    for qt in [16, 32] {
        let baseline = AttentionGraph::new(
            device,
            &fixture,
            AttentionVariant::Tiled {
                qt,
                bound: super::CausalBlockBound::FullChunk,
            },
        )?;
        let candidate = AttentionGraph::new(
            device,
            &fixture,
            AttentionVariant::Tiled {
                qt,
                bound: super::CausalBlockBound::QueryTile,
            },
        )?;
        let full = baseline.check(device, &expected)?;
        let pruned = candidate.check(device, &expected)?;
        assert_bitwise(case, qt, &full, &pruned);
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA; smoke before full numerical gate; safe-run --test-threads 1"]
fn fp8_model_geometry_causal_bound_smoke() -> NumericResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for lanes in [64, 256, 512] {
        check_model_case(
            &device,
            ChunkCase {
                lanes,
                base: 31,
                count: lanes - 1,
                offset: -1,
                window: 17,
                permuted: true,
            },
        )?;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA; complete FP8/F64/bitwise gate; safe-run --test-threads 1"]
fn fp8_model_geometry_causal_tiles_match_full_reference_and_baseline_bits() -> NumericResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for case in numeric_cases() {
        check_model_case(&device, case)?;
    }
    Ok(())
}

fn measure_attention_pair(
    device: &CudaDevice,
    shared: &SharedAttentionGraphs,
    arms: (usize, usize),
    cold: bool,
    reverse: bool,
) -> NumericResult {
    let options = BenchOptions {
        warmup: Duration::from_millis(20),
        rep: Duration::from_millis(50),
        min_reps: 10,
        max_reps: 30,
        clear_l2: cold,
    };
    let baseline = &shared.graphs[arms.0];
    let candidate = &shared.graphs[arms.1];
    let addresses = shared.addresses();
    let run = |capture: &Graph| {
        capture
            .launch()
            .sync_on(&device.stream)
            .map_err(|e| tensor_error(&e.to_string()))
    };
    let (first, second) = if reverse {
        (&candidate.1, &baseline.1)
    } else {
        (&baseline.1, &candidate.1)
    };
    let (a, b) = do_bench_paired(&device.stream, &options, |_| run(first), |_| run(second))?;
    let (a, b) = if reverse { (b, a) } else { (a, b) };
    assert!(
        a.times_ms()
            .iter()
            .chain(b.times_ms())
            .all(|x| x.is_finite() && *x > 0.0)
    );
    assert_eq!(addresses, shared.addresses());
    eprintln!(
        "{}",
        serde_json::json!({
            "attention_cost": "append-fp8-quantization+compute", "case": shared.fixture.case,
            "baseline": baseline.0.label(), "candidate": candidate.0.label(),
            "cold_l2": cold, "order": if reverse { "BA" } else { "AB" },
            "baseline_ms": a.times_ms(), "candidate_ms": b.times_ms(),
            "baseline_median_ms": a.median_ms(), "candidate_median_ms": b.median_ms(),
            "speedup": a.median_ms() / b.median_ms(), "device_pointers": addresses,
            "same_allocations": true,
            "independent_service_units": false, "jit_in_measured_graph": false,
        })
    );
    Ok(())
}

fn cost_model_case(device: &CudaDevice, case: ChunkCase) -> NumericResult {
    let fixture = ChunkFixture::new(device, case)?;
    let expected = F64Attention::new(&fixture).complete_output();
    let shared = SharedAttentionGraphs::new(device, &fixture)?;
    shared.check(device, 0, &expected)?;
    for (qt, full, pruned) in [(16, 1, 2), (32, 3, 4)] {
        assert_bitwise(
            case,
            qt,
            &shared.check(device, full, &expected)?,
            &shared.check(device, pruned, &expected)?,
        );
        for arms in [(0, full), (0, pruned), (full, pruned)] {
            for cold in [true, false] {
                for reverse in [false, true] {
                    measure_attention_pair(device, &shared, arms, cold, reverse)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA; only after complete numerical gate; diagnostic costs, no service claim"]
fn fp8_model_geometry_append_and_compute_paired_cold_warm_ab_ba() -> NumericResult {
    if cfg!(debug_assertions) {
        return Err("attention costs require release".into());
    }
    let compile = CompileOptions::default();
    if compile.device_debug || compile.opt_level.is_some_and(|level| level != 3) {
        return Err("attention costs require CUDA O3 without debug".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (lanes, count, base) in [(64, 51, 0), (256, 255, 256), (512, 511, 0)] {
        cost_model_case(
            &device,
            ChunkCase {
                lanes,
                base,
                count,
                offset: 0,
                window: 0,
                permuted: true,
            },
        )?;
    }
    Ok(())
}
