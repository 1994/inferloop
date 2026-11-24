use super::*;
use infer_core::{RequestId, StateId};

fn plan() -> StepPlan {
    StepPlan {
        id: StepId::ONE,
        decision: DecisionId::ONE,
        program: ProgramId::ONE,
        role: ExecutionRole::Decode,
        work: vec![infer_ir::PlannedWork {
            request: RequestId::ONE,
            state: StateId::ONE,
            token_count: 1,
            role: ExecutionRole::Decode,
        }],
        cost: CostEstimate::default(),
        graph: None,
        quantum_overrun: false,
    }
}
#[test]
fn metadata_readers_hold_fixed_slots_without_overwrite_or_growth() -> Result<()> {
    let mut pool = StepPool::new(1, ProgramId::ONE)?;
    let mut held = Vec::new();
    let mut input = plan();
    for id in 1..=4 {
        input.id = StepId::new(id)?;
        input.work.push(infer_ir::PlannedWork {
            request: RequestId::ONE,
            state: StateId::ONE,
            token_count: 1,
            role: ExecutionRole::Decode,
        });
        if input.work.len() == 2 {
            input.work.pop();
        }
        held.push(pool.seal(&mut input)?);
    }
    assert_eq!(
        pool.seal(&mut input).err().map(|e| e.code),
        Some(ErrorCode::Capacity)
    );
    for (index, reader) in held.iter().enumerate() {
        assert_eq!(reader.id.get(), index as u64 + 1);
    }
    held.remove(0);
    input.id = StepId::new(5)?;
    input.work.push(infer_ir::PlannedWork {
        request: RequestId::ONE,
        state: StateId::ONE,
        token_count: 1,
        role: ExecutionRole::Decode,
    });
    let reused = pool.seal(&mut input)?;
    assert_eq!(reused.id.get(), 5);
    assert_eq!(held[0].id.get(), 2);
    assert_eq!(input.work, [] as [infer_ir::PlannedWork; 0]);
    Ok(())
}
#[test]
fn feedback_readers_share_records_and_exhaustion_returns_backpressure() -> Result<()> {
    let mut pool = QueryPool::new(1, 1)?;
    let query = infer_ir::CostQuery::from_unit(
        ProgramId::ONE,
        infer_ir::BackendKind::Metal,
        ExecutionRole::Decode,
        1,
        1024,
        CostEstimate::default(),
    );
    let mut held = Vec::new();
    for _ in 0..4 {
        held.push(pool.seal(&[query])?);
    }
    assert_eq!(
        pool.seal(&[query]).err().map(|e| e.code),
        Some(ErrorCode::Capacity)
    );
    let saved = held[0].clone();
    assert!(saved.shares_storage(&held[0]));
    held.remove(0);
    assert!(pool.seal(&[query]).is_err());
    drop(saved);
    let next = pool.seal(&[infer_ir::CostQuery {
        context_tokens: 2048,
        ..query
    }])?;
    assert_eq!(next[0].context_tokens, 2048);
    assert_eq!(held[0][0].context_tokens, 1024);
    Ok(())
}
