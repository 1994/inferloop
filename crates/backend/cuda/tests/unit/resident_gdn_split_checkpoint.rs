//! Full independent numeric and lifecycle qualification of the shared split workspace.
use super::*;
use crate::constants::KV_BLOCK_TOKENS;
use crate::resident::small_gemm::kernels as pair_kernels;
use crate::resident::verify_projection_pair::{self, plan_for};
use serde_json::json;
fn reference(
    input: &[f32],
    wb: &[bf16],
    wa: &[bf16],
    (rows, columns, outputs): (usize, usize, usize),
) -> (Vec<f64>, Vec<f64>) {
    let mut rb = vec![0.0f64; rows * outputs];
    let mut ra = vec![0.0f64; rows * outputs];
    for row in 0..rows {
        for out in 0..outputs {
            let mut db = 0.0f64;
            let mut da = 0.0f64;
            for k in 0..columns {
                let x = f64::from(input[row * columns + k]);
                db = x.mul_add(f64::from(wb[out * columns + k]), db);
                da = x.mul_add(f64::from(wa[out * columns + k]), da);
            }
            rb[row * outputs + out] = db;
            ra[row * outputs + out] = da;
        }
    }
    (rb, ra)
}
fn gate_errors(actual: &[f32], expected: &[f64]) -> (usize, f64) {
    let mut failed = 0usize;
    let mut max_abs = 0.0f64;
    for (actual, expected) in actual.iter().zip(expected) {
        let error = (f64::from(*actual) - expected).abs();
        max_abs = max_abs.max(error);
        if !actual.is_finite() || error > expected.abs().mul_add(2.0e-5, 5.0e-4) {
            failed += 1;
        }
    }
    (failed, max_abs)
}
fn check_real_rows(loaded: &LoadedModel, rows: usize) -> Result<(), Box<dyn Error>> {
    let pairs = gdn_ba_pairs(loaded.graph(), loaded.weights());
    assert_eq!(pairs.len(), 48);
    let baseline = pair_work(loaded, &pairs, PairMode::Separate, rows)?;
    let before = verify_projection_pair::captures();
    let candidate = pair_work(loaded, &pairs, PairMode::Split, rows)?;
    assert_eq!(verify_projection_pair::captures() - before, 48);
    baseline.graph.launch().sync_on(&loaded.device().stream)?;
    candidate.graph.launch().sync_on(&loaded.device().stream)?;
    let input = loaded.device().read_borrowed(candidate.input.as_ref())?;
    let mut elements = 0;
    let mut baseline_max = 0.0f64;
    let mut split_max = 0.0f64;
    for (((&first, &second), original), split) in
        pairs.iter().zip(&baseline.outputs).zip(&candidate.outputs)
    {
        let (first, second) = (&loaded.graph().nodes[first], &loaded.graph().nodes[second]);
        let (Some(ProjectionWeight::Dense(wb)), Some(ProjectionWeight::Dense(wa))) = (
            loaded.weights().projections.get(&first.inputs[1]),
            loaded.weights().projections.get(&second.inputs[1]),
        ) else {
            return Err("GDN format changed".into());
        };
        let columns = usize::try_from(wb.shape()[1])?;
        let outputs = usize::try_from(wb.shape()[0])?;
        let wb = loaded.device().read_borrowed(wb)?;
        let wa = loaded.device().read_borrowed(wa)?;
        let (rb, ra) = reference(&input, &wb, &wa, (rows, columns, outputs));
        for (a, b, expected) in [(&original.0, &split.0, &rb), (&original.1, &split.1, &ra)] {
            let a = loaded.device().read_borrowed(a)?;
            let b = loaded.device().read_borrowed(b)?;
            assert_eq!(a.len(), rows * outputs);
            assert_eq!(a.len(), b.len());
            assert!(expected.iter().all(|v| v.is_finite()));
            let (af, ae) = gate_errors(&a, expected);
            let (bf, be) = gate_errors(&b, expected);
            assert_eq!(af, 0, "original two-term baseline F64 gate");
            assert_eq!(bf, 0, "shared split workspace F64 gate");
            baseline_max = baseline_max.max(ae);
            split_max = split_max.max(be);
            elements += b.len();
        }
    }
    eprintln!(
        "{}",
        json!({"kind":"shared_split_full_f64","rows":rows,"pairs":pairs.len(),
        "full_output_elements":elements,"baseline_max_abs":baseline_max,"split_max_abs":split_max,
        "gate":"abs(reference)*2e-5+5e-4","scratch_bytes":VerifyPairWorkspace::storage_bytes(
            loaded.graph(),loaded.weights(),rows,loaded.device().profile()?)?})
    );
    Ok(())
}

fn load_checkpoint() -> Result<LoadedModel, Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    Ok(LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL")?,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            verification_width: 3,
            autotune: false,
            mtp_depth: 2,
            ..LoadOptions::default()
        },
    )?)
}

#[test]
#[ignore = "CUDA checkpoint, independent F64 on every actual GDN output"]
fn split_workspace_matches_full_independent_reference() -> Result<(), Box<dyn Error>> {
    let loaded = load_checkpoint()?;
    for rows in [45, 48, 47] {
        check_real_rows(&loaded, rows)?;
    }
    Ok(())
}

#[test]
#[ignore = "CUDA, row/column/K tails and distinct nonempty split windows"]
fn split_kernel_covers_all_tail_windows() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (rows, outputs, columns) = (17usize, 33usize, 257usize);
    let plan = plan_for(
        rows,
        outputs,
        columns,
        usize::try_from(device.profile()?.multiprocessors)?,
    )
    .ok_or("no tail split")?;
    assert_eq!(plan.splits, 3);
    let input = device.upload(vec![1.0f32; rows * columns], &[rows, columns])?;
    let weight = device.upload(
        vec![bf16::from_f32(1.0); outputs * columns],
        &[outputs, columns],
    )?;
    let mut pb = api::zeros::<f32>(&plan.shape).sync_on(&device.stream)?;
    let mut pa = api::zeros::<f32>(&plan.shape).sync_on(&device.stream)?;
    let mut ob = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
    let mut oa = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            pair_kernels::dense_pair_split(
                (&mut pb).partition([16, 32]),
                (&mut pa).partition([16, 32]),
                input.as_ref(),
                weight.as_ref(),
                weight.as_ref(),
                i32::try_from(plan.k_tiles).map_err(error)?,
                i32::try_from(rows.div_ceil(16)).map_err(error)?,
            )
            .generics(vec![columns.to_string()]),
        )?;
        scope.record(pair_kernels::reduce_pair_split(
            (&mut ob).partition([16, 32]),
            (&mut oa).partition([16, 32]),
            &pb,
            &pa,
            i32::try_from(plan.splits).map_err(error)?,
            i32::try_from(rows.div_ceil(16)).map_err(error)?,
        ))?;
        Ok(())
    })?;
    graph.launch().sync_on(&device.stream)?;
    for tensor in [&ob, &oa] {
        assert!(
            device
                .read_borrowed(tensor)?
                .iter()
                .all(|v| v.to_bits() == 257.0f32.to_bits())
        );
    }
    for tensor in [&pb, &pa] {
        let values = device.read_borrowed(tensor)?;
        for (split, expected) in [128.0f32, 128.0, 1.0].into_iter().enumerate() {
            for row in 0..rows {
                for col in 0..outputs {
                    assert_eq!(values[(split * 32 + row) * plan.shape[1] + col], expected);
                }
            }
        }
        assert!(values.iter().all(|v| v.is_finite()));
    }
    Ok(())
}

#[test]
#[ignore = "CUDA real48pairs, production Workspace, fixed ABBA microbenchmark"]
fn actual_split_workspace_latency() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err("release required".into());
    }
    let compile = CompileOptions::default();
    assert!(!compile.device_debug && compile.opt_level.is_none_or(|level| level == 3));
    let loaded = load_checkpoint()?;
    let pairs = gdn_ba_pairs(loaded.graph(), loaded.weights());
    assert_eq!(pairs.len(), 48);
    for rows in [45, 48] {
        let before = verify_projection_pair::captures();
        let baseline = pair_work(&loaded, &pairs, PairMode::Paired, rows)?;
        let candidate = pair_work(&loaded, &pairs, PairMode::Split, rows)?;
        assert_eq!(verify_projection_pair::captures() - before, 48);
        for cold in [true, false] {
            let options = BenchOptions {
                warmup: Duration::from_millis(50),
                rep: Duration::from_millis(180),
                min_reps: 20,
                max_reps: 100,
                clear_l2: cold,
            };
            for reverse in [false, true] {
                let (a, b) = if reverse {
                    (&candidate, &baseline)
                } else {
                    (&baseline, &candidate)
                };
                let (a, b) = do_bench_paired(
                    &loaded.device().stream,
                    &options,
                    |_| {
                        a.graph
                            .launch()
                            .sync_on(&loaded.device().stream)
                            .map_err(|e| tensor_error(&e.to_string()))
                    },
                    |_| {
                        b.graph
                            .launch()
                            .sync_on(&loaded.device().stream)
                            .map_err(|e| tensor_error(&e.to_string()))
                    },
                )?;
                let (old, new) = if reverse { (b, a) } else { (a, b) };
                eprintln!(
                    "{}",
                    json!({"kind":"gdn_split_workspace_latency","rows":rows,
                    "pairs":pairs.len(),"cold":cold,"reverse":reverse,
                    "baseline_ms":old.median_ms(),"candidate_ms":new.median_ms(),
                    "speedup":old.median_ms()/new.median_ms(),
                    "baseline_raw_ms":old.times_ms(),"candidate_raw_ms":new.times_ms(),
                    "scratch_bytes":VerifyPairWorkspace::storage_bytes(loaded.graph(),
                        loaded.weights(),rows,loaded.device().profile()?)?,
                    "service_acceptance":false})
                );
            }
        }
    }
    Ok(())
}

fn committed_states(
    loaded: &LoadedModel,
    a: &SlotPool,
    b: &SlotPool,
    slots: usize,
) -> Result<usize, Box<dyn Error>> {
    assert_eq!(a.cursors(), b.cursors());
    let mut elements = 0;
    for slot in 0..slots {
        let (left, lf) = a.test_states(slot);
        let (right, rf) = b.test_states(slot);
        for (id, parts) in left {
            for (x, y) in parts.iter().zip(&right[id]) {
                let x = loaded.device().read_borrowed(x)?;
                let y = loaded.device().read_borrowed(y)?;
                bits_equal(&x, &y);
                elements += x.len();
            }
        }
        let ta = loaded.device().read_borrowed(a.test_table(slot))?;
        let tb = loaded.device().read_borrowed(b.test_table(slot))?;
        for (id, (ak, av)) in lf {
            let (bk, bv) = &rf[id];
            for (x, y) in [(ak, bk), (av, bv)] {
                let heads = usize::try_from(x.shape()[0])?;
                let physical = usize::try_from(x.shape()[1])?;
                let dim = usize::try_from(x.shape()[2])?;
                let x = loaded.device().read_borrowed(x)?;
                let y = loaded.device().read_borrowed(y)?;
                for h in 0..heads {
                    for pos in 0..a.cursors()[slot] {
                        let xa = (h * physical
                            + usize::try_from(ta[pos / KV_BLOCK_TOKENS])? * KV_BLOCK_TOKENS
                            + pos % KV_BLOCK_TOKENS)
                            * dim;
                        let ya = (h * physical
                            + usize::try_from(tb[pos / KV_BLOCK_TOKENS])? * KV_BLOCK_TOKENS
                            + pos % KV_BLOCK_TOKENS)
                            * dim;
                        assert!(
                            x[xa..xa + dim]
                                .iter()
                                .zip(&y[ya..ya + dim])
                                .all(|(a, b)| a.0 == b.0),
                            "committed FP8 prefix"
                        );
                        elements += dim;
                    }
                }
            }
        }
    }
    Ok(elements)
}

fn prefix_round(
    loaded: &LoadedModel,
    reference: &mut SlotPool,
    candidate: &mut SlotPool,
    active: &[usize],
    round: usize,
) -> Result<usize, Box<dyn Error>> {
    assert_eq!(reference.cursors(), candidate.cursors());
    let mut long = Vec::new();
    let mut short = Vec::new();
    for &slot in active {
        let position = reference.cursors()[slot];
        let accepted = (slot + round) % 3;
        for lane in 0..3 {
            let row = (
                slot,
                lane,
                u32::try_from(31 + slot * 7 + round * 3 + lane)?,
                position + lane,
                position + lane,
            );
            if lane <= accepted {
                short.push(row);
            }
            long.push(row);
        }
    }
    let expected = if short.is_empty() {
        Vec::new()
    } else {
        reference.run_verify(loaded.device(), &short)?
    };
    let actual = candidate.run_verify(loaded.device(), &long)?;
    // Readout retains fixed slot-major positions, including inactive empty rows.
    let total = reference.cursors().len() * 3;
    assert_eq!(expected.len(), total);
    assert_eq!(actual.len(), total);
    for &(slot, lane, ..) in &short {
        let row = slot * 3 + lane;
        bits_equal(&expected[row].0, &actual[row].0);
        bits_equal(&expected[row].1, &actual[row].1);
    }
    let mut elements = 0;
    for row in actual {
        assert!(row.0.iter().chain(&row.1).all(|v| v.is_finite()));
        elements += row.0.len() + row.1.len();
    }
    for &slot in active {
        let accepted = (slot + round) % 3;
        // accepted excludes the submitted token; both replays include it.
        reference.commit_verify(loaded.device(), slot, accepted)?;
        candidate.commit_verify(loaded.device(), slot, accepted)?;
    }
    Ok(elements)
}

#[test]
#[ignore = "CUDA MTP checkpoint; split default vs independent accepted-prefix replay"]
fn default_split_pool_preserves_commit_discard_and_reuse() -> Result<(), Box<dyn Error>> {
    let loaded = load_checkpoint()?;
    let seed = loaded.sequence_pooled(64)?;
    for (slots, paged) in [(15, false), (16, true)] {
        let before = verify_projection_pair::captures();
        let mut reference = loaded.slot_pool(slots, 64, paged.then_some(slots * 64))?;
        let mut candidate = loaded.slot_pool(slots, 64, paged.then_some(slots * 64))?;
        assert_eq!(verify_projection_pair::captures() - before, 96);
        for slot in 0..slots {
            assert_eq!(reference.take_free(), Some(slot));
            assert_eq!(candidate.take_free(), Some(slot));
            assert!(reference.assign(slot, 64, loaded.device())?);
            assert!(candidate.assign(slot, 64, loaded.device())?);
            reference.bind(slot, seed.states(), seed.fp8_states(), 0, loaded.device())?;
            candidate.bind(slot, seed.states(), seed.fp8_states(), 0, loaded.device())?;
        }
        let all: Vec<_> = (0..slots).collect();
        let mut elements = 0;
        let mut states = 0;
        for (round, active) in [all.clone(), vec![0, slots - 1], all].iter().enumerate() {
            elements += prefix_round(&loaded, &mut reference, &mut candidate, active, round)?;
            states += committed_states(&loaded, &reference, &candidate, slots)?;
        }
        let position = candidate.cursors()[0];
        let cancelled: Vec<_> = (0..3)
            .map(|lane| (0, lane, 77, position + lane, position + lane))
            .collect();
        assert_eq!(
            candidate.run_verify(loaded.device(), &cancelled)?.len(),
            slots * 3
        );
        candidate.discard(0);
        for slot in 0..slots {
            reference.release(slot);
            candidate.release(slot);
        }
        let slot = reference.take_free().ok_or("released slot not reusable")?;
        assert_eq!(candidate.take_free(), Some(slot));
        for pool in [&mut reference, &mut candidate] {
            assert!(pool.assign(slot, 64, loaded.device())?);
            pool.bind(slot, seed.states(), seed.fp8_states(), 0, loaded.device())?;
        }
        for round in [3, 4] {
            elements += prefix_round(&loaded, &mut reference, &mut candidate, &[slot], round)?;
            states += committed_states(&loaded, &reference, &candidate, slots)?;
        }
        eprintln!(
            "{}",
            json!({"default_split_prefix_replay":true,"slots":slots,"paged":paged,
            "full_finite_outputs":elements,"checked_state_and_committed_cache_elements":states,
            "sparse_active":true,"zero_partial_commit":true,"release_rebind_reuse":true})
        );
    }
    Ok(())
}
