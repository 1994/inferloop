use super::*;
use crate::{CalibratedCosts, FallbackCosts};
use infer_core::{ProgramId, StepId};
use infer_ir::{
    BackendKind, CostModelConfig, CostObservation, ExecutionTiming, PageGrowth, TimingSource,
};
use infer_spi::CostModelProvider;
#[test]
fn native_summaries_match_complete_queries_after_mixed_phase_growth_and_calibration() -> Result<()>
{
    let mut costs = CalibratedCosts::new("summary-parity".into(), CostModelConfig::default())?;
    for round in 0..64 {
        let mut summary = BatchCostSummary::default();
        let mut queries = Vec::new();
        for index in 0..8 {
            let mut query = CostQuery::from_unit(
                ProgramId::ONE,
                BackendKind::Metal,
                [
                    ExecutionRole::Prefill,
                    ExecutionRole::Decode,
                    ExecutionRole::Forward,
                ][index % 3],
                index + 1,
                32 + round,
                CostEstimate {
                    gpu_us: 2,
                    workspace_bytes: 16 + index as u64,
                    num_gpu_blocks: 1,
                    state_bytes: 32,
                    transfer_us: 3,
                    encoder_us: 4,
                    ..CostEstimate::default()
                },
            );
            query.page_growth = Some(PageGrowth {
                block_size: 16,
                allocated_pages: 1,
                bytes_per_page: 64,
                cow_tail: true,
            });
            query.logical_growth = Some(PageGrowth {
                block_size: 8,
                allocated_pages: 1,
                bytes_per_page: 0,
                cow_tail: false,
            });
            summary = update_summary(summary, None, query)?;
            queries.push(query);
            assert_eq!(
                FallbackCosts.estimate_summary(&summary)?,
                Some(aggregate(&queries)?)
            );
            assert_eq!(
                costs.estimate_summary(&summary)?,
                Some(costs.estimate(&queries)?)
            );
        }
        for index in 0..queries.len() {
            let previous = queries[index];
            queries[index].tokens += 1;
            queries[index].context_tokens += 1;
            summary = update_summary(summary, Some(previous), queries[index])?;
            assert_eq!(
                FallbackCosts.estimate_summary(&summary)?,
                Some(aggregate(&queries)?)
            );
            assert_eq!(
                costs.estimate_summary(&summary)?,
                Some(costs.estimate(&queries)?)
            );
        }
        costs.observe(&CostObservation {
            step: StepId::ONE,
            work: queries.into(),
            timing: ExecutionTiming {
                elapsed_us: 100 + round as u64,
                source: TimingSource::MetalGpu,
            },
        })?;
    }
    Ok(())
}
