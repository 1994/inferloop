use super::*;
use crate::resident::attention::attention;
use cutile::cutile_compiler::compile_api::KernelCompiler;
use std::error::Error;

type TestResult<T = ()> = StdResult<T, Box<dyn Error>>;

#[test]
fn pooled_attention_kernels_compile_without_a_device() -> TestResult {
    for (dim, kv, parts, group) in [(32, 2, 4, 4), (256, 4, 64, 6)] {
        let columns = 256;
        let physical = 8192;
        let fields = i32::try_from(METADATA_FIELDS)?;
        KernelCompiler::new(kernels::__module_ast_self, "kernels", "pack_control")
            .target("sm_120")
            .generics(vec![columns.to_string()])
            .strides(&[
                ("out", &[columns, 1]),
                ("t0", &[1]),
                ("t1", &[1]),
                ("t2", &[1]),
                ("t3", &[1]),
                ("parameters", &[1]),
                ("metadata", &[1]),
            ])
            .compile()?;
        KernelCompiler::new(kernels::__module_ast_self, "kernels", "append_pages")
            .target("sm_120")
            .generics(
                [dim, physical, kv, 12, fields]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            )
            .strides(&[
                ("keys", &[physical * dim, dim, 1]),
                ("values", &[physical * dim, dim, 1]),
                ("input_keys", &[kv * dim, dim, 1]),
                ("input_values", &[kv * dim, dim, 1]),
                ("control", &[columns, 1]),
            ])
            .compile()?;
        KernelCompiler::new(kernels::__module_ast_self, "kernels", "partial_slots")
            .target("sm_120")
            .generics(
                [dim, kv, parts, fields]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            )
            .strides(&[
                ("out", &[32 * dim, dim, 1]),
                ("peaks", &[32, 1]),
                ("masses", &[32, 1]),
                ("queries", &[group * dim, dim, 1]),
                ("keys", &[physical * dim, dim, 1]),
                ("values", &[physical * dim, dim, 1]),
                ("control", &[columns, 1]),
            ])
            .compile()?;
        KernelCompiler::new(kernels::__module_ast_self, "kernels", "merge_slots")
            .target("sm_120")
            .generics(
                [dim, parts, group, group * kv]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            )
            .strides(&[
                ("out", &[dim, 1]),
                ("numerator", &[parts * 32 * dim, 32 * dim, dim, 1]),
                ("peaks", &[parts * 32, 32, 1]),
                ("masses", &[parts * 32, 32, 1]),
            ])
            .compile()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn pooled_attention_preserves_output_bits_and_cache_bytes() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for slots in [2, 5, 16] {
        let mut fixture = Fixture::new(&device, slots)?;
        for (position, window) in [
            (0, 0),
            (30, 0),
            (4094, 0),
            (8189, 1024),
            (8189, 4096),
            (8189, 0),
        ] {
            fixture.check(&device, position, window)?;
        }
    }
    Ok(())
}

const HEADS: usize = 24;
const KV: usize = 4;
const DIM: usize = 256;
const CAPACITY: usize = 8192;
const PARTS: usize = 64;
const SCALES: [f32; 2] = [0.5, 2.0];

fn upload<T: DType>(device: &CudaDevice, values: Vec<T>, shape: &[usize]) -> TestResult<Tensor<T>> {
    Arc::try_unwrap(device.upload(values, shape)?).map_err(|_| "test upload still shared".into())
}

struct Fixture {
    slots: usize,
    capacity: usize,
    tables: Vec<Tensor<i32>>,
    queries: Arc<Tensor<f32>>,
    inputs: [Arc<Tensor<f32>>; 2],
    original: [Tensor<f8e4m3fn>; 2],
    candidate: [Tensor<f8e4m3fn>; 2],
}

impl Fixture {
    fn new(device: &CudaDevice, slots: usize) -> TestResult<Self> {
        Self::with_capacity(device, slots, CAPACITY)
    }
    fn with_capacity(device: &CudaDevice, slots: usize, capacity: usize) -> TestResult<Self> {
        let blocks = capacity / KV_BLOCK_TOKENS;
        let tables = (0..slots)
            .map(|slot| {
                upload(
                    device,
                    (0..blocks)
                        .map(|block| {
                            i32::try_from(((slot * blocks + block) * 3 + 7) % (slots * blocks))
                                .unwrap()
                        })
                        .collect(),
                    &[blocks],
                )
            })
            .collect::<TestResult<Vec<_>>>()?;
        let queries = device.upload(
            (0..slots * VERIFY_LANES * HEADS * DIM)
                .map(|i| (f32::from(u16::try_from(i % 131).unwrap()) - 65.0) / 37.0)
                .collect(),
            &[slots * VERIFY_LANES * HEADS * DIM],
        )?;
        let input = |seed: usize| {
            device.upload(
                (0..slots * VERIFY_LANES * KV * DIM)
                    .map(|i| {
                        if i % 97 == 0 {
                            if i % 2 == 0 { 1000.0 } else { -1000.0 }
                        } else {
                            (f32::from(u16::try_from((i + seed) % 137).unwrap()) - 68.0) / 31.0
                        }
                    })
                    .collect(),
                &[slots * VERIFY_LANES * KV * DIM],
            )
        };
        let cache = |seed: usize| {
            upload(
                device,
                (0..KV * slots * capacity * DIM)
                    .map(|i| {
                        f8e4m3fn(
                            32 + u8::try_from((i + seed) % 48).unwrap()
                                + if i % 3 == 0 { 128 } else { 0 },
                        )
                    })
                    .collect(),
                &[KV, slots * capacity, DIM],
            )
        };
        Ok(Self {
            slots,
            capacity,
            tables,
            queries,
            inputs: [input(0)?, input(13)?],
            original: [cache(0)?, cache(11)?],
            candidate: [cache(0)?, cache(11)?],
        })
    }

    fn check(&mut self, device: &CudaDevice, position: i32, window: i32) -> TestResult {
        let lanes = self.slots * VERIFY_LANES;
        // Mix active, fully inactive and partial slots; field zero deliberately differs from
        // the cache position, which is field two.
        let words: Vec<_> = (0..lanes)
            .flat_map(|lane| {
                let slot = lane / VERIFY_LANES;
                let candidate = lane % VERIFY_LANES;
                let active = slot % 5 != 1 && (!slot.is_multiple_of(3) || candidate == 0);
                let p = position + i32::try_from(candidate).unwrap();
                [p + 17, 0, if active { p } else { -1 }, 0]
            })
            .collect();
        let metadata = device.upload(words.clone(), &[words.len()])?;
        let lane_metadata = words
            .as_chunks::<METADATA_FIELDS>()
            .0
            .iter()
            .map(|row| device.upload(row.to_vec(), &[METADATA_FIELDS]))
            .collect::<Result<Vec<_>>>()?;
        let node = OpId::ONE;
        let geometry = Geometry {
            heads: HEADS,
            kv_heads: KV,
            dim: DIM,
        };
        let mut header = [0; CONTROL_TILE];
        header[0] = i32::from_ne_bytes(SCALES[0].to_ne_bytes());
        header[1] = i32::from_ne_bytes(SCALES[1].to_ne_bytes());
        header[2] = window;
        let buffer = Buffers::new(
            device,
            Blueprint {
                layout: Layout::new(geometry, self.slots, self.capacity, PARTS)?,
                parameters: BTreeMap::from([(node, header)]),
            },
        )?;
        let mut work = Workspace {
            buffers: BTreeMap::from([(geometry, buffer)]),
            nodes: BTreeMap::from([(node, geometry)]),
        };
        let mut actual = api::zeros::<f32>(&[lanes * HEADS, DIM]).sync_on(&device.stream)?;
        let mut expected = (0..lanes)
            .map(|_| api::zeros::<f32>(&[HEADS, DIM]).sync_on(&device.stream))
            .collect::<StdResult<Vec<_>, _>>()?;
        self.reference(device, &lane_metadata, window, &mut expected)?;
        let [keys, values] = &mut self.candidate;
        let candidate = CudaGraph::scope(&device.stream, |scope| {
            work.append(
                scope,
                Append {
                    node,
                    keys,
                    values,
                    inputs: [&self.inputs[0], &self.inputs[1]],
                    metadata: &metadata,
                    tables: &self.tables,
                },
            )?;
            work.attend(
                scope,
                Attend {
                    node,
                    output: &mut actual,
                    queries: &self.queries,
                    keys,
                    values,
                },
            )?;
            Ok(())
        })?;
        candidate.launch().sync_on(&device.stream)?;
        let actual = device.read_borrowed(&actual)?;
        for (lane, expected) in expected.iter().enumerate() {
            let expected = device.read_borrowed(expected)?;
            let start = lane * HEADS * DIM;
            for (column, (&a, &b)) in actual[start..start + HEADS * DIM]
                .iter()
                .zip(&expected)
                .enumerate()
            {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "slots={} p={position} window={window} lane={lane} column={column}: {a} != {b}",
                    self.slots
                );
            }
        }
        for (actual, expected) in self.candidate.iter().zip(&self.original) {
            let actual = device.read_borrowed(actual)?;
            let expected = device.read_borrowed(expected)?;
            for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.0, b.0,
                    "cache slots={} p={position} window={window} byte={index}",
                    self.slots
                );
            }
        }
        Ok(())
    }
    fn reference(
        &mut self,
        device: &CudaDevice,
        lane_metadata: &[Arc<Tensor<i32>>],
        window: i32,
        expected: &mut [Tensor<f32>],
    ) -> TestResult {
        let mut numerator =
            api::zeros::<f32>(&[KV * PARTS, STACKED_ROWS, DIM]).sync_on(&device.stream)?;
        let mut maxima = api::zeros::<f32>(&[KV * PARTS, STACKED_ROWS]).sync_on(&device.stream)?;
        let mut sums = api::zeros::<f32>(&[KV * PARTS, STACKED_ROWS]).sync_on(&device.stream)?;
        let [keys, values] = &mut self.original;
        let physical = self.slots * self.capacity;
        let reference = CudaGraph::scope(&device.stream, |scope| {
            for (lane, metadata) in lane_metadata.iter().enumerate() {
                let range = lane * KV * DIM..(lane + 1) * KV * DIM;
                scope.record(
                    attention::append(
                        (&mut *keys).partition([1, physical, DIM]),
                        (&mut *values).partition([1, physical, DIM]),
                        &self.inputs[0].slice(from_ref(&range))?.view(&[KV, DIM])?,
                        &self.inputs[1].slice(from_ref(&range))?.view(&[KV, DIM])?,
                        metadata.as_ref(),
                        &self.tables[lane / VERIFY_LANES],
                        SCALES[0],
                        SCALES[1],
                    )
                    .generics(vec![
                        f8e4m3fn::DTYPE.as_str().into(),
                        DIM.to_string(),
                        physical.to_string(),
                        "1".into(),
                    ]),
                )?;
            }
            for slot in 0..self.slots {
                let base = slot * VERIFY_LANES;
                let queries = (base..base + VERIFY_LANES)
                    .map(|lane| {
                        self.queries
                            .slice(from_ref(&(lane * HEADS * DIM..(lane + 1) * HEADS * DIM)))
                    })
                    .collect::<StdResult<Vec<_>, _>>()?;
                scope.record(
                    super::super::kernels::partial_gqa3(
                        (&mut numerator).partition([1, STACKED_ROWS, DIM]),
                        (&mut maxima).partition([1, STACKED_ROWS]),
                        (&mut sums).partition([1, STACKED_ROWS]),
                        &queries[0].view(&[KV, HEADS / KV, DIM])?,
                        &queries[1].view(&[KV, HEADS / KV, DIM])?,
                        &queries[2].view(&[KV, HEADS / KV, DIM])?,
                        &lane_metadata[base],
                        &lane_metadata[base + 1],
                        &lane_metadata[base + 2],
                        &*keys,
                        &*values,
                        &self.tables[slot],
                        window,
                        SCALES[0],
                        SCALES[1],
                    )
                    .generics(vec![
                        f8e4m3fn::DTYPE.as_str().into(),
                        DIM.to_string(),
                        PARTS.to_string(),
                        KV_BLOCK_TOKENS.to_string(),
                    ])
                    .compile_options(grouped_options()),
                )?;
                for (lane, output) in expected[base..base + VERIFY_LANES].iter_mut().enumerate() {
                    scope.record(
                        super::super::kernels::merge_gqa(
                            output.partition([1, DIM]),
                            &numerator.view(&[KV, PARTS, STACKED_ROWS, DIM])?,
                            &maxima.view(&[KV, PARTS, STACKED_ROWS])?,
                            &sums.view(&[KV, PARTS, STACKED_ROWS])?,
                            i32::try_from(lane * STACKED_LANE_ROWS).map_err(error)?,
                        )
                        .generics(
                            [DIM, PARTS, HEADS / KV, STACKED_ROWS]
                                .iter()
                                .map(ToString::to_string)
                                .collect(),
                        ),
                    )?;
                }
            }
            Ok(())
        })?;
        reference.launch().sync_on(&device.stream)?;
        Ok(())
    }
}

#[test]
#[ignore = "CUDA/safe-run; short bounded context, extended slot groups, full output/cache bytes"]
fn extended_concurrency_preserves_attention_outputs_and_pages() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for slots in [17, 32, 64] {
        let mut fixture = Fixture::with_capacity(&device, slots, 128)?;
        for (position, window) in [(0, 0), (30, 0), (125, 32), (125, 0)] {
            fixture.check(&device, position, window)?;
        }
    }
    Ok(())
}
