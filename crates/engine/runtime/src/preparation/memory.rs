//! Conservative retained native request/projection bounds, independent of device storage.
use infer_ir::{CanonicalRequest, DecisionQuestion, RequestInput, Workload};

/// Fixed request bookkeeping retained beyond input tokens and workload projections.
const REQUEST_OVERHEAD_BYTES: usize = 256;

pub fn request_bytes(request: &CanonicalRequest) -> usize {
    let input = match &request.input {
        RequestInput::Sequence { tokens, media } => tokens
            .capacity()
            .saturating_mul(size_of::<u32>())
            .saturating_add(
                media
                    .capacity()
                    .saturating_mul(size_of::<infer_ir::MediaInput>()),
            )
            .saturating_add(media.iter().map(infer_ir::MediaInput::retained_bytes).sum()),
        RequestInput::Pairs { query, documents } => documents.iter().fold(
            query
                .capacity()
                .saturating_mul(size_of::<u32>())
                .saturating_add(
                    documents
                        .capacity()
                        .saturating_mul(size_of::<infer_ir::TokenBuffer>()),
                ),
            |bytes, document| {
                bytes.saturating_add(document.capacity().saturating_mul(size_of::<u32>()))
            },
        ),
    };
    let workload = match &request.workload {
        Workload::Decision(schema) => schema.questions.iter().fold(
            schema
                .questions
                .capacity()
                .saturating_mul(size_of::<DecisionQuestion>()),
            |bytes, question| {
                bytes.saturating_add(match question {
                    DecisionQuestion::Categorical { options } => {
                        options.capacity().saturating_mul(size_of::<u32>())
                    }
                    DecisionQuestion::Ordinal { options, values } => options
                        .capacity()
                        .saturating_add(values.capacity())
                        .saturating_mul(size_of::<u32>()),
                    DecisionQuestion::Binary { .. } | DecisionQuestion::Continuous { .. } => 0,
                })
            },
        ),
        Workload::Extension { provider, payload } => {
            provider.capacity().saturating_add(payload.capacity())
        }
        _ => 0,
    };
    input
        .saturating_add(workload)
        .saturating_add(request.qos.tenant.capacity())
        .saturating_add(size_of::<CanonicalRequest>() + REQUEST_OVERHEAD_BYTES)
}
pub fn projection_bytes(request: &CanonicalRequest) -> usize {
    match &request.workload {
        Workload::Decision(schema) => schema
            .questions
            .iter()
            .fold(0usize, |bytes, question| {
                let options = match question {
                    DecisionQuestion::Binary { .. } => 2,
                    DecisionQuestion::Continuous { .. } => 1,
                    DecisionQuestion::Categorical { options }
                    | DecisionQuestion::Ordinal { options, .. } => options.len(),
                };
                bytes.saturating_add(
                    options
                        .saturating_mul(size_of::<u32>())
                        .saturating_add(size_of::<infer_ir::DecisionAnswer>()),
                )
            })
            .saturating_mul(2),
        _ => 0,
    }
}
