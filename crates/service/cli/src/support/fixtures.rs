//! Fixtures CLI support.
use infer_core::{ModelId, RequestId, Result};
use infer_ir::{CanonicalRequest, Qos, RequestInput, Sampling, Workload};
use std::collections::BTreeMap;

/// Token identifiers shared by every example request's input sequence.
const EXAMPLE_TOKENS: [u32; 5] = [1, 2, 3, 4, 5];
pub fn example(id: u64, workload: Workload) -> Result<CanonicalRequest> {
    Ok(CanonicalRequest {
        id: RequestId::new(id)?,
        model: ModelId::new(1)?,
        session: None,
        input: RequestInput::Sequence {
            tokens: EXAMPLE_TOKENS.to_vec().into(),
            media: vec![],
        },
        workload,
        qos: Qos::default(),
        sampling: Sampling::default(),
        prefix_boundary: None,
        extensions: BTreeMap::new(),
    })
}
