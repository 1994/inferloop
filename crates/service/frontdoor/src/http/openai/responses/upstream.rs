//! The engine runtime as the agentic layer's [`Upstream`] for one Responses request.
use super::super::TextState;
use super::super::execution::Prepared as Generation;
use super::execute;
use super::output::ResponseBody;
use super::request::Spec;
use infer_agentic::Upstream;
use infer_core::{RequestId, Result};

/// Completes admitted generations into Responses objects.
pub(super) struct TextUpstream {
    pub(super) state: TextState,
}

/// One admitted generation and the request values its response object echoes.
pub(super) struct Admitted {
    pub(super) generation: Generation,
    pub(super) spec: Spec,
}

impl Upstream for TextUpstream {
    type Admitted = Admitted;
    type Document = ResponseBody;

    async fn complete(&self, admitted: Admitted) -> Result<ResponseBody> {
        execute::generate_prepared(&self.state, admitted.generation, &admitted.spec).await
    }

    async fn cancel(&self, request: RequestId) -> Result<()> {
        self.state.handle.cancel(request).await
    }
}
