use crate::{MediaInput, Workload};
use infer_core::{Error, ModelId, RequestId, Result, SessionId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Longest accepted `QoS` tenant identifier, in bytes.
const MAX_TENANT_ID_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RequestInput {
    Sequence {
        tokens: crate::TokenBuffer,
        #[serde(default)]
        media: Vec<MediaInput>,
    },
    Pairs {
        query: crate::TokenBuffer,
        documents: crate::SharedOutput<crate::TokenBuffer>,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Qos {
    pub tenant: String,
    pub weight: u32,
    /// Absolute deadline in the caller-provided monotonic microsecond clock.
    pub deadline_us: Option<u64>,
    pub ttft_slo_us: Option<u64>,
    pub tpot_slo_us: Option<u64>,
}
impl Default for Qos {
    fn default() -> Self {
        Self {
            tenant: "default".into(),
            weight: 1,
            deadline_us: None,
            ttft_slo_us: None,
            tpot_slo_us: None,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: f32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub eos_tokens: Vec<u32>,
    pub seed: u64,
    pub eos_token: Option<u32>,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: 1.0,
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            eos_tokens: vec![],
            seed: 0,
            eos_token: None,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalRequest {
    pub id: RequestId,
    pub model: ModelId,
    #[serde(default)]
    pub session: Option<SessionId>,
    pub input: RequestInput,
    pub workload: Workload,
    #[serde(default)]
    pub qos: Qos,
    #[serde(default)]
    pub sampling: Sampling,
    /// Prompt position whose prefix later requests are expected to repeat: the prompt is
    /// prefilled to exactly this position before continuing, and the backend may keep a
    /// reusable snapshot there (a cache breakpoint). A request carrying an evidence block that
    /// many follow-up questions share names the end of that block.
    #[serde(default)]
    pub prefix_boundary: Option<usize>,
    #[serde(default)]
    pub extensions: BTreeMap<String, Vec<u8>>,
}
impl CanonicalRequest {
    /// # Errors
    /// Returns an invalid-input or invariant error if dimensions, identities, ranges, or ownership are inconsistent.
    pub fn validate(&self) -> Result<()> {
        if self.qos.tenant.is_empty()
            || self.qos.tenant.len() > MAX_TENANT_ID_BYTES
            || self.qos.weight == 0
        {
            return Err(Error::invalid("tenant and positive weight required"));
        }
        if self.qos.ttft_slo_us == Some(0)
            || self.qos.tpot_slo_us == Some(0)
            || (!matches!(self.workload, Workload::Generate { .. })
                && (self.qos.ttft_slo_us.is_some() || self.qos.tpot_slo_us.is_some()))
        {
            return Err(Error::invalid(
                "positive TTFT/TPOT SLOs require a generate workload",
            ));
        }
        self.sampling.validate()?;
        if !self.extensions.is_empty() {
            return Err(Error::unsupported(
                "no feature provider registered for extension fields",
            ));
        }
        match &self.input {
            RequestInput::Sequence { tokens, media } => {
                if tokens.is_empty() && media.is_empty() {
                    return Err(Error::invalid("empty input"));
                }
                if self
                    .prefix_boundary
                    .is_some_and(|boundary| boundary == 0 || boundary >= tokens.len())
                {
                    return Err(Error::invalid(
                        "prefix boundary must fall strictly inside the prompt",
                    ));
                }
                if matches!(self.workload, Workload::Rerank { .. }) {
                    return Err(Error::invalid("rerank requires query/document pairs"));
                }
            }
            RequestInput::Pairs { query, documents } => {
                if self.prefix_boundary.is_some() {
                    return Err(Error::invalid("prefix boundaries apply to sequence input"));
                }
                if query.is_empty()
                    || documents.is_empty()
                    || documents.iter().any(|doc| doc.is_empty())
                {
                    return Err(Error::invalid("empty query/document"));
                }
                if !matches!(self.workload, Workload::Rerank { .. }) {
                    return Err(Error::invalid("pair input requires rerank workload"));
                }
            }
        }
        self.workload.validate()
    }
}

impl Sampling {
    /// Whether validated parameters select the highest F32 total-order logit directly.
    /// This does not validate the parameter ranges.
    #[must_use]
    pub fn uses_plain_greedy(&self) -> bool {
        self.temperature == 0.0
            && self.presence_penalty == 0.0
            && self.repetition_penalty.to_bits() == 1.0_f32.to_bits()
    }

    /// # Errors
    /// Rejects non-finite values and invalid probability/penalty ranges.
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite()
            || self.temperature < 0.0
            || self.top_k == Some(0)
            || !self.top_p.is_finite()
            || self.top_p <= 0.0
            || self.top_p > 1.0
            || !self.min_p.is_finite()
            || !(0.0..=1.0).contains(&self.min_p)
            || !self.presence_penalty.is_finite()
            || !(-2.0..=2.0).contains(&self.presence_penalty)
            || !self.repetition_penalty.is_finite()
            || self.repetition_penalty <= 0.0
        {
            return Err(Error::invalid("invalid sampling parameters"));
        }
        Ok(())
    }

    #[must_use]
    pub fn is_eos(&self, token: u32) -> bool {
        self.eos_token == Some(token) || self.eos_tokens.contains(&token)
    }
}
