//! Read-only memory accounting diagnosis with the production (non-test) CUDA library.
#[cfg(cuda_backend)]
use std::error::Error as StdError;
#[cfg(cuda_backend)]
mod profile {
    use cuda_core::Device;
    use infer_backend_cuda::{
        device::CudaDevice,
        loading::{LoadOptions, LoadedModel},
        resident::DeviceProgram,
    };
    use infer_core::{Error, ModelId};
    use infer_ir::OutputReadout;
    use serde::Serialize;
    use serde_json::json;
    use std::env::args as env_args;
    use std::fs::File;
    use std::io::Error as IoError;
    use std::{error::Error as StdError, sync::Arc, thread, time::Duration};

    type Failure = Box<dyn StdError>;
    const CAPACITY: usize = 32768;
    const GIB: u64 = 1 << 30;

    #[derive(Clone, Copy, Serialize)]
    struct Counters {
        driver_free: u64,
        total: u64,
        pool_reserved: u64,
        pool_used: u64,
        pool_reserved_high: u64,
        pool_used_high: u64,
        pool_reusable: u64,
        context_allocatable: u64,
    }

    fn emit(stage: &str, counters: Counters) {
        println!("{}", json!({"stage": stage, "memory": counters}));
    }

    fn sample(query: &CudaDevice, core: &Arc<Device>) -> Result<Counters, Failure> {
        let (driver_free, total) = query.memory_info()?;
        let stats = core.default_mem_pool()?.mem_stats()?;
        let pool_reusable = stats.reserved_current.saturating_sub(stats.used_current);
        Ok(Counters {
            driver_free,
            total,
            pool_reserved: stats.reserved_current,
            pool_used: stats.used_current,
            pool_reserved_high: stats.reserved_high,
            pool_used_high: stats.used_high,
            pool_reusable,
            context_allocatable: driver_free.saturating_add(pool_reusable).min(total),
        })
    }

    fn settled(query: &CudaDevice, core: &Arc<Device>, stage: &str) -> Result<Counters, Failure> {
        let mut previous = None;
        let mut stable = 0;
        for index in 0..20 {
            query.synchronize()?;
            let current = sample(query, core)?;
            emit(&format!("{stage}-{index}"), current);
            let identity = (current.pool_reserved, current.pool_used);
            if previous == Some(identity) {
                stable += 1;
            } else {
                previous = Some(identity);
                stable = 1;
            }
            if stable >= 3 {
                return Ok(current);
            }
            thread::sleep(Duration::from_millis(250));
        }
        Err(IoError::other("allocator counters did not settle in20samples").into())
    }

    fn allocation(
        loaded: &LoadedModel,
        query: &CudaDevice,
        core: &Arc<Device>,
    ) -> Result<(DeviceProgram, DeviceProgram), Failure> {
        let before = settled(query, core, "before-program-pair")?;
        let target = loaded.sequence(CAPACITY)?;
        let after_target = settled(query, core, "after-target-program")?;
        let draft = loaded
            .draft(CAPACITY)?
            .ok_or_else(|| Error::invalid("MTP2 draft missing"))?;
        let after_draft = settled(query, core, "after-draft-program")?;
        println!(
            "{}",
            json!({
                "stage": "program-components",
                "target_base_activation_bytes": target.activation_bytes,
                "draft_base_activation_bytes": draft.activation_bytes,
                "target_private_pool_bytes": after_target.pool_used.saturating_sub(before.pool_used),
                "draft_private_pool_bytes": after_draft.pool_used.saturating_sub(after_target.pool_used),
                "before": before,
                "after_target": after_target,
                "after_draft": after_draft,
                "model_generation_streams": 0
            })
        );
        Ok((target, draft))
    }

    fn allocation_fits(
        before: Counters,
        now: Counters,
        limit: u64,
        quote: u64,
        headroom: u64,
    ) -> bool {
        let active_engine = before
            .context_allocatable
            .saturating_sub(now.context_allocatable);
        active_engine
            .checked_add(quote)
            .is_some_and(|used| used <= limit)
            && quote <= now.context_allocatable.saturating_sub(headroom)
    }

    struct AllocationObservation {
        allocated_pairs: usize,
        allocation_error: Option<String>,
        stop_reason: Option<String>,
        after_pairs: Counters,
        after_drop_pairs: Counters,
    }

    fn observe_allocations(
        loaded: &LoadedModel,
        query: &CudaDevice,
        core: &Arc<Device>,
        before: Counters,
        after_load: Counters,
        limit: u64,
    ) -> Result<AllocationObservation, Failure> {
        let quote = loaded.sequence_budget(CAPACITY, OutputReadout::Logits, true)?;
        let headroom = loaded.profile().device_headroom_bytes();
        let mut pairs = Vec::with_capacity(2);
        let mut current = after_load;
        let mut allocation_error = None;
        let mut stop_reason = None;
        for index in 0..2 {
            if !allocation_fits(before, current, limit, quote, headroom) {
                stop_reason = Some(format!("preflight prevents allocation{}", index + 1));
                break;
            }
            match allocation(loaded, query, core) {
                Ok(pair) => {
                    pairs.push(pair);
                    current = settled(query, core, &format!("after-pair{}", index + 1))?;
                }
                Err(error) => {
                    allocation_error = Some(error.to_string());
                    break;
                }
            }
        }
        let allocated_pairs = pairs.len();
        let after_pairs = settled(query, core, "after-pairs-before-drop")?;
        drop(pairs);
        let after_drop_pairs = settled(query, core, "after-drop-pairs")?;
        Ok(AllocationObservation {
            allocated_pairs,
            allocation_error,
            stop_reason,
            after_pairs,
            after_drop_pairs,
        })
    }

    fn arguments() -> Result<(String, f64), Failure> {
        let mut args = env_args().skip(1);
        let model = args
            .next()
            .ok_or_else(|| IoError::other("model path required"))?;
        let utilization: f64 = args
            .next()
            .ok_or_else(|| IoError::other("frozen utilization fraction required"))?
            .parse()?;
        if utilization.to_bits() != 0.88_f64.to_bits() {
            return Err(IoError::other("diagnostic requires frozen0.88utilization").into());
        }
        if args.next().is_some() {
            return Err(IoError::other("unexpected extra argument").into());
        }
        Ok((model, utilization))
    }

    pub fn run() -> Result<(), Failure> {
        let (model, utilization) = arguments()?;
        CudaDevice::enable_kernel_cache()?;
        let query = CudaDevice::new(0)?;
        // cuda-core0.4.0 Device::new retains device0's primary context, as does CudaDevice.
        // Observing its default pool creates no second CUDA context or tensor allocation.
        let core = Device::new(0)?;
        let before = settled(&query, &core, "before-model")?;
        #[expect(
            clippy::cast_precision_loss,
            reason = "Same total-device-memory multiplication as the production CLI"
        )]
        let scaled = (before.total as f64) * utilization;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "Frozen0.88fraction is non-negative and bounded by actual device memory"
        )]
        let share = scaled as u64;
        let limit = share.min(before.driver_free.saturating_sub(GIB));
        let loaded = LoadedModel::open(
            query.clone(),
            model,
            ModelId::ONE,
            LoadOptions {
                prefill_width: 0,
                mtp_depth: 2,
                autotune: true,
                ..LoadOptions::default()
            },
        )?;
        let immediate = sample(&query, &core)?;
        emit("model-immediate-production-query", immediate);
        let original_model_charge = before.driver_free.saturating_sub(immediate.driver_free);
        let original_state_budget = limit.saturating_sub(original_model_charge);
        let after_load = settled(&query, &core, "model-settled")?;
        let active_model_estimate = before
            .context_allocatable
            .saturating_sub(after_load.context_allocatable);
        let pool_aware_budget_estimate = limit.saturating_sub(active_model_estimate);
        let execution = loaded.execution_profile();
        let mut graph = loaded.graph().clone();
        graph.plan_lifetimes()?;
        serde_json::to_writer(File::create_new("target-dataflow.json")?, &graph)?;
        if execution.prefill_width != 256
            || execution.batch_width != 3
            || execution.draft_depth != 2
        {
            return Err(IoError::other("load geometry differs from fixed serving profile").into());
        }
        let quote = loaded.sequence_budget(CAPACITY, OutputReadout::Logits, true)?;
        println!(
            "{}",
            json!({
                "stage": "accounting",
                "capacity": CAPACITY,
                "prefill_width": execution.prefill_width,
                "batch_width": execution.batch_width,
                "whole_engine_limit": limit,
                "original_model_charge": original_model_charge,
                "original_state_budget": original_state_budget,
                "active_model_estimate": active_model_estimate,
                "pool_aware_budget_estimate": pool_aware_budget_estimate,
                "private_target_and_draft_quote": quote,
                "production_default_admission_changed": false,
                "context_pool_formula_is_estimate_until_allocations_are_measured": true
            })
        );
        let observation = observe_allocations(&loaded, &query, &core, before, after_load, limit)?;
        drop(loaded);
        let after_drop_model = settled(&query, &core, "after-drop-model")?;
        println!(
            "{}",
            json!({
                "stage": "final",
                "completed": true,
                "allocated_target_and_draft_capacity_pairs": observation.allocated_pairs,
                "allocation_error": observation.allocation_error,
                "stop_reason": observation.stop_reason,
                "before": before,
                "after_load": after_load,
                "after_pairs": observation.after_pairs,
                "after_drop_pairs": observation.after_drop_pairs,
                "after_drop_model": after_drop_model,
                "original_state_budget": original_state_budget,
                "pool_aware_budget_estimate": pool_aware_budget_estimate,
                "private_target_and_draft_quote": quote,
                "direct_capture_only_not_default_backend_admission": true,
                "model_generation_streams": 0,
                "performance_accepted": false,
                "quality_accepted": false,
                "no_memory_pool_policy_or_context_limit_changed": true
            })
        );
        Ok(())
    }
}

#[cfg(cuda_backend)]
fn main() -> Result<(), Box<dyn StdError>> {
    profile::run()
}

#[cfg(not(cuda_backend))]
fn main() {
    eprintln!("This diagnostic requires Linux and the cuda feature.");
}
