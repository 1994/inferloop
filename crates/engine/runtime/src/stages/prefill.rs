use crate::RequestRecord;
use infer_core::{Error, Result};
use infer_ir::{ExecutionInput, OutputReadout, Workload};

pub fn prepare(record: &RequestRecord, count: usize) -> Result<ExecutionInput> {
    let end = record
        .prefill_done
        .checked_add(count)
        .ok_or_else(|| Error::invariant("prefill cursor overflow"))?;
    let readout = readout(record, end);
    let prompt = record.context.prompt.len();
    let span = if end <= prompt {
        record.context.prompt.span(record.prefill_done..end)?
    } else {
        let start = record
            .prefill_done
            .checked_sub(prompt)
            .ok_or_else(|| Error::invariant("prefill crossed token storage boundary"))?;
        record.generated.span_at(start..end - prompt, prompt)?
    };
    Ok(ExecutionInput::Prefill { span, readout })
}
pub fn readout(record: &RequestRecord, end: usize) -> OutputReadout {
    if end < record.prefill_target {
        OutputReadout::None
    } else {
        record.request.workload.final_readout()
    }
}
pub fn commit(record: &mut RequestRecord, count: usize) -> Result<usize> {
    let end = record
        .prefill_done
        .checked_add(count)
        .ok_or_else(|| Error::invariant("prefill completion overflow"))?;
    if end > record.prefill_target {
        return Err(Error::invariant("prefill completion exceeds target"));
    }
    record.prefill_done = end;
    Ok(end)
}

pub const fn retained_readout(workload: &Workload) -> OutputReadout {
    workload.final_readout()
}
