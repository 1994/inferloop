//! Compare production chunk scans with the original 32-column arithmetic and state.
use super::{SCAN_COLUMNS, Scratch, WIDE_SCAN_COLUMNS, WIDE_SCAN_VALUE_HEADS, kernels, record};
use crate::device::CudaDevice;
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_cuda_kernels::tile::runtime::Graph;
use std::error::Error as StdError;
use std::time::Duration;

type TestResult<T = ()> = Result<T, Box<dyn StdError>>;
const DIM: usize = 128;

struct Inputs {
    shape: [usize; 3],
    qkv: Arc<Tensor<f32>>,
    beta: Arc<Tensor<f32>>,
    decay: Arc<Tensor<f32>>,
    metadata: Arc<Tensor<i32>>,
    initial: Arc<Tensor<f32>>,
}

fn values(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (f32::from(u16::try_from((i * salt + i / 7) % 131).unwrap()) - 65.0) / 200.0)
        .collect()
}
fn capture(
    device: &CudaDevice,
    input: &Inputs,
    scratch: &mut Scratch,
    (state, output): (&mut Tensor<f32>, &mut Tensor<f32>),
    optimized: bool,
) -> TestResult<Graph> {
    let [kh, vh, lanes] = input.shape;
    let qkv = input.qkv.view(&[lanes, 2 * kh + vh, DIM])?;
    Ok(Graph::scope(&device.stream, |scope| {
        scope.record(baseline::reset(
            (&mut *state).partition([1, DIM, 32]),
            &input.initial,
        ))?;
        if optimized {
            return record(
                scope,
                scratch,
                &qkv,
                (&input.beta, &input.decay),
                [state, output],
                &input.metadata,
                (kh, vh, DIM, lanes),
            );
        }
        scope.record(
            kernels::chunk_normalize(
                (&mut scratch.queries).partition([1, 64, DIM]),
                (&mut scratch.keys).partition([1, 64, DIM]),
                &qkv,
                &input.metadata,
            )
            .generics(vec![kh.to_string(), DIM.to_string()]),
        )?;
        scope.record(
            kernels::chunk_prepare(
                (&mut scratch.u).partition([1, 64, DIM]),
                (&mut scratch.w).partition([1, 64, DIM]),
                (&mut scratch.p).partition([1, 64, 64]),
                (&mut scratch.cumulative).partition([1, 64]),
                &qkv,
                &scratch.queries,
                &scratch.keys,
                &input.beta,
                &input.decay,
                &input.metadata,
            )
            .generics(vec![kh.to_string(), vh.to_string(), DIM.to_string()]),
        )?;
        scope.record(
            baseline::legacy_scan32(
                state.partition([1, DIM, 32]),
                output.partition([1, lanes, 32]),
                &scratch.u,
                &scratch.w,
                &scratch.queries,
                &scratch.keys,
                &scratch.p,
                &scratch.cumulative,
                &input.metadata,
            )
            .generics([kh, vh, lanes].iter().map(ToString::to_string).collect()),
        )?;
        Ok(())
    })?)
}

fn equal(device: &CudaDevice, a: Tensor<f32>, b: Tensor<f32>) -> TestResult<[Arc<Tensor<f32>>; 2]> {
    let (a, b) = (Arc::new(a), Arc::new(b));
    let x = (&a).to_host_vec().sync_on(&device.stream)?;
    let y = (&b).to_host_vec().sync_on(&device.stream)?;
    assert_eq!(x.len(), y.len());
    for (i, (&x, &y)) in x.iter().zip(&y).enumerate() {
        assert!(x.is_finite() && y.is_finite(), "nonfinite at {i}");
        assert_eq!(x.to_bits(), y.to_bits(), "at {i}: {x} != {y}");
    }
    Ok([a, b])
}
fn compare_case(device: &CudaDevice, shape: [usize; 3]) -> TestResult {
    let [kh, vh, lanes] = shape;
    let columns = if vh == WIDE_SCAN_VALUE_HEADS {
        WIDE_SCAN_COLUMNS
    } else {
        SCAN_COLUMNS
    };
    let input = Inputs {
        shape,
        qkv: device.upload(
            values(lanes * (2 * kh + vh) * DIM, 17),
            &[lanes, 2 * kh + vh, DIM],
        )?,
        beta: device.upload(vec![0.2; lanes * vh], &[lanes, vh])?,
        decay: device.upload(vec![0.99; lanes * vh], &[lanes, vh])?,
        metadata: device.upload(vec![0, i32::try_from(lanes - 3)?, 0, 0], &[4])?,
        initial: device.upload(values(vh * DIM * DIM, 13), &[vh, DIM, DIM])?,
    };
    let mut scratch = Scratch::new(device, (kh, vh, DIM), lanes)?;
    let mut a = api::zeros::<f32>(&[vh, lanes, DIM]).sync_on(&device.stream)?;
    let mut b = api::zeros::<f32>(&[vh, lanes, DIM]).sync_on(&device.stream)?;
    let mut sa = api::zeros::<f32>(&[vh, DIM, DIM]).sync_on(&device.stream)?;
    let mut sb = api::zeros::<f32>(&[vh, DIM, DIM]).sync_on(&device.stream)?;
    let ga = capture(device, &input, &mut scratch, (&mut sa, &mut a), false)?;
    let gb = capture(device, &input, &mut scratch, (&mut sb, &mut b), true)?;
    ga.launch().sync_on(&device.stream)?;
    gb.launch().sync_on(&device.stream)?;
    let _outputs = equal(device, a, b)?;
    let _states = equal(device, sa, sb)?;
    println!(
        "{}",
        serde_json::json!({
            "kind":"gdn_scan_tiling_bits","lanes":lanes,"columns":columns,"key_heads":kh,"value_heads":vh,
            "output_elements":vh*lanes*DIM,"state_elements":vh*DIM*DIM,"all_finite_bitwise_equal":true
        })
    );
    for reverse in [false, true] {
        let (first, second) = if reverse { (&gb, &ga) } else { (&ga, &gb) };
        let options = BenchOptions {
            warmup: Duration::from_millis(30),
            rep: Duration::from_millis(100),
            min_reps: 20,
            max_reps: 100,
            clear_l2: true,
        };
        let (x, y) = do_bench_paired(
            &device.stream,
            &options,
            |_| {
                first
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|e| tensor_error(&e.to_string()))
            },
            |_| {
                second
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|e| tensor_error(&e.to_string()))
            },
        )?;
        let (base, candidate) = if reverse { (y, x) } else { (x, y) };
        println!(
            "{}",
            serde_json::json!({
                "kind":"gdn_scan_tiling_latency","lanes":lanes,"columns":columns,"key_heads":kh,"value_heads":vh,
                "reverse":reverse,"baseline_ms":base.median_ms(),"candidate_ms":candidate.median_ms(),
                "speedup":base.median_ms()/candidate.median_ms(),
                "baseline_samples_ms":base.times_ms(),"candidate_samples_ms":candidate.times_ms(),
                "scope":"original chunk normalization, preparation and scan plus identical state reset"
            })
        );
    }
    Ok(())
}
#[test]
#[ignore = "CUDA release, protected runner; GDN complete output/state and original-tile performance comparison"]
fn chunk_scan_tiling_screen() -> TestResult {
    if cfg!(debug_assertions) {
        return Err("release required".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for shape in [
        [2, 4, 64],
        [16, 16, 256],
        [16, 32, 1024],
        [16, 48, 64],
        [16, 48, 256],
        [16, 48, 1024],
    ] {
        compare_case(&device, shape)?;
    }
    Ok(())
}

#[cutile::module]
mod baseline {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_0, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tile_1, Tile_2, bf16, constant,
        convert_tile, exp, extract, get_tile_block_id, mmaf, tile_to_scalar,
    };
    #[cutile::entry()]
    pub fn reset(out: &mut Tensor<f32, { [1, 128, 32] }>, input: &Tensor<f32, { [-1, -1, -1] }>) {
        out.store(input.load_like(out));
    }
    /// `left * right + acc` from BF16 operands with F32 accumulation.
    fn bf16_product<const M: i32, const K: i32, const N: i32>(
        left: Tile<f32, { [M, K] }>,
        right: Tile<f32, { [K, N] }>,
        acc: Tile<f32, { [M, N] }>,
    ) -> Tile<f32, { [M, N] }> {
        let lh: Tile<bf16, { [M, K] }> = convert_tile(left);
        let rh: Tile<bf16, { [K, N] }> = convert_tile(right);
        mmaf(lh, rh, acc)
    }

    /// Carry one 32-column block of the state across the active chunks.
    #[expect(clippy::too_many_arguments, reason = "explicit CUDA tensor bindings")]
    #[cutile::entry()]
    pub fn legacy_scan32<const KH: i32, const VH: i32, const LANES: i32>(
        state: &mut Tensor<f32, { [1, 128, 32] }>,
        out: &mut Tensor<f32, { [1, LANES, 32] }>,
        u: &Tensor<bf16, { [-1, -1, 128] }>,
        w: &Tensor<bf16, { [-1, -1, 128] }>,
        queries: &Tensor<bf16, { [-1, -1, 128] }>,
        keys: &Tensor<bf16, { [-1, -1, 128] }>,
        p: &Tensor<bf16, { [-1, -1, 64] }>,
        cumulative: &Tensor<f32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let head = pid.0;
        let key = head / (VH / KH);
        let count: i32 =
            tile_to_scalar(metadata.partition(shape![1]).load([1i32]).reshape(shape![]));
        let mut old = state.load_like(state).reshape(shape![128, 32]);
        let zero_block: Tile<f32, { [64, 32] }> = constant(0.0f32, shape![64, 32]);
        let wp = w.partition(shape![1, 64, 128]);
        let qp = queries.partition(shape![1, 64, 128]);
        let kp = keys.partition(shape![1, 64, 128]);
        let pp = p.partition(shape![1, 64, 64]);
        let gp = cumulative.partition(shape![1, 64]);
        for chunk in 0i32..((count + 63i32) / 64i32) {
            let g: Tile<f32, { [64] }> = gp.load([head, chunk]).reshape(shape![64]);
            let g_last: Tile<f32, { [1] }> = extract(g, [63i32.broadcast(shape![])]);
            let wb: Tile<bf16, { [64, 128] }> =
                wp.load([head, chunk, 0i32]).reshape(shape![64, 128]);
            let wc: Tile<f32, { [64, 128] }> = convert_tile(wb);
            let predicted: Tile<f32, { [64, 32] }> = bf16_product(wc, old, zero_block);
            let target: Tile<bf16, { [64, 32] }> = u
                .partition(shape![1, 64, 32])
                .load([head, chunk, pid.2])
                .reshape(shape![64, 32]);
            let target: Tile<f32, { [64, 32] }> = convert_tile(target);
            let updates = target - predicted;
            let qb: Tile<bf16, { [64, 128] }> =
                qp.load([key, chunk, 0i32]).reshape(shape![64, 128]);
            let qf: Tile<f32, { [64, 128] }> = convert_tile(qb);
            let qc = qf * exp(g).reshape(shape![64, 1]).broadcast(shape![64, 128]);
            let pb: Tile<bf16, { [64, 64] }> = pp.load([head, chunk, 0i32]).reshape(shape![64, 64]);
            let pc: Tile<f32, { [64, 64] }> = convert_tile(pb);
            let carried: Tile<f32, { [64, 32] }> = bf16_product(qc, old, zero_block);
            let output: Tile<f32, { [64, 32] }> = bf16_product(pc, updates, carried);
            out.partition_mut(shape![1, 64, 32])
                .store(output.reshape(shape![1, 64, 32]), [0i32, chunk, 0i32]);
            let kb: Tile<bf16, { [64, 128] }> =
                kp.load([key, chunk, 0i32]).reshape(shape![64, 128]);
            let kf: Tile<f32, { [64, 128] }> = convert_tile(kb);
            let remaining = exp(g_last.broadcast(shape![64]) - g);
            let kc = kf * remaining.reshape(shape![64, 1]).broadcast(shape![64, 128]);
            let decayed: Tile<f32, { [128, 32] }> =
                old * exp(g_last).reshape(shape![1, 1]).broadcast(shape![128, 32]);
            let transposed: Tile<f32, { [128, 64] }> = kc.transpose();
            old = bf16_product(transposed, updates, decayed);
        }
        state.store(old.reshape(shape![1, 128, 32]));
    }
}
