use crate::{Head, OutputReadout};
use infer_core::credits::CreditLease;
use infer_core::{Error, ModelId, ProgramId, RequestId, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pooling {
    Mean,
    Last,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecisionQuestion {
    Binary {
        negative_token: u32,
        positive_token: u32,
    },
    Categorical {
        options: Vec<u32>,
    },
    Ordinal {
        options: Vec<u32>,
        values: Vec<f32>,
    },
    Continuous {
        token: u32,
        min: f32,
        max: f32,
    },
}
/// Which scores a decision reads its option tokens from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionReadout {
    /// An installed decision projection over the final hidden row; options index its channels.
    #[default]
    Projection,
    /// The language-model head at the last prompt position; options are vocabulary tokens.
    ///
    /// A generation model answers through the first token of its reply, so a deployment that
    /// generates can also decide without an extra head. The prompt must end where that reply
    /// begins; the option tokens are what the reply's first token would be for each label.
    LanguageModel,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionSchema {
    pub questions: Vec<DecisionQuestion>,
    pub calibration_temperature: f32,
    pub abstain_below: Option<f32>,
    /// Absent in older requests and snapshots, which always read an installed projection.
    #[serde(default)]
    pub readout: DecisionReadout,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Workload {
    Generate {
        max_new_tokens: usize,
    },
    Embed {
        pooling: Pooling,
        dimensions: Option<usize>,
        normalize: bool,
    },
    Rerank {
        top_k: usize,
    },
    Decision(DecisionSchema),
    Classify {
        classes: usize,
    },
    Reward,
    LateInteraction,
    Extension {
        provider: String,
        payload: Vec<u8>,
    },
}
impl Workload {
    #[must_use]
    pub const fn head(&self) -> Option<Head> {
        Some(match self {
            Self::Generate { .. }
            | Self::Decision(DecisionSchema {
                readout: DecisionReadout::LanguageModel,
                ..
            }) => Head::LanguageModel,
            Self::Embed { .. } => Head::Embedding,
            Self::Rerank { .. } => Head::Rank,
            Self::Decision(DecisionSchema {
                readout: DecisionReadout::Projection,
                ..
            }) => Head::Decision,
            Self::Classify { .. } => Head::Classification,
            Self::Reward => Head::Reward,
            Self::LateInteraction => Head::LateInteraction,
            Self::Extension { .. } => return None,
        })
    }
    /// The readout the last prompt position needs.
    ///
    /// Generation and language-model decisions consume only the final logits, so they keep the
    /// pooled logits-only state; every other workload projects hidden rows and needs them all.
    #[must_use]
    pub const fn final_readout(&self) -> OutputReadout {
        match self {
            Self::Generate { .. }
            | Self::Decision(DecisionSchema {
                readout: DecisionReadout::LanguageModel,
                ..
            }) => OutputReadout::Logits,
            _ => OutputReadout::Full,
        }
    }
    /// # Errors
    /// Returns an invalid-input or invariant error if dimensions, identities, ranges, or ownership are inconsistent.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Generate { max_new_tokens: 0 }
            | Self::Embed {
                dimensions: Some(0),
                ..
            }
            | Self::Rerank { top_k: 0 }
            | Self::Classify { classes: 0 } => {
                return Err(Error::invalid("workload size must be positive"));
            }
            Self::Decision(schema) => {
                if schema.questions.is_empty()
                    || !schema.calibration_temperature.is_finite()
                    || schema.calibration_temperature <= 0.0
                    || schema
                        .abstain_below
                        .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
                {
                    return Err(Error::invalid("invalid decision calibration/schema"));
                }
                for question in &schema.questions {
                    match question {
                        DecisionQuestion::Binary {
                            negative_token,
                            positive_token,
                        } if negative_token == positive_token => {
                            return Err(Error::invalid("binary answers must differ"));
                        }
                        DecisionQuestion::Categorical { options }
                        | DecisionQuestion::Ordinal { options, .. } => {
                            if options.len() < 2 {
                                return Err(Error::invalid(
                                    "at least two decision options required",
                                ));
                            }
                            let mut unique = options.clone();
                            unique.sort_unstable();
                            unique.dedup();
                            if unique.len() != options.len() {
                                return Err(Error::invalid("duplicate decision options"));
                            }
                        }
                        DecisionQuestion::Continuous { min, max, .. }
                            if !min.is_finite() || !max.is_finite() || min >= max =>
                        {
                            return Err(Error::invalid("invalid continuous decision bounds"));
                        }
                        _ => {}
                    }
                    if let DecisionQuestion::Ordinal { options, values } = question
                        && (options.len() != values.len()
                            || values.iter().any(|v| !v.is_finite())
                            || values.windows(2).any(|w| w[0] >= w[1]))
                    {
                        return Err(Error::invalid(
                            "ordinal values must match options and increase",
                        ));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadPlan {
    pub request: RequestId,
    pub model: ModelId,
    pub program: ProgramId,
    pub units: Vec<crate::TokenBuffer>,
    pub reserved_tokens: usize,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeaturePlan {
    pub speculation: Option<SpeculationPlan>,
    pub grammar: Option<String>,
    pub adapter: Option<String>,
    pub prefix_reuse: bool,
}
/// How a verification graph accepts the proposals a draft produced.
///
/// The rule is part of the shared contract rather than a string each side interprets: a provider
/// declares which sampling its verifier can accept, and the device implementation follows the
/// declaration. An untyped field accepted a typo as readily as a policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceptancePolicy {
    /// A proposal is accepted only where it matches the verifier's argmax.
    #[default]
    Greedy,
    /// A divergence is resampled from the residual, so acceptance needs the residual distribution.
    Residual,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculationPlan {
    pub proposal: ProgramId,
    pub verification: ProgramId,
    pub hidden_state_taps: Vec<usize>,
    pub max_candidates: usize,
    pub acceptance_policy: AcceptancePolicy,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionAnswer {
    pub probabilities: Vec<f32>,
    pub expected: Option<f32>,
    pub abstained: bool,
    /// The options' joint probability over every readout score, before renormalizing to them.
    ///
    /// Low mass means the model preferred something other than the offered labels, which the
    /// renormalized `probabilities` alone cannot show. `None` for a continuous answer.
    #[serde(default)]
    pub label_mass: Option<f32>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedDocument {
    pub index: usize,
    pub score: f32,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WorkloadOutput {
    Tokens(crate::TokenBuffer),
    Embedding(crate::SharedOutput<f32>),
    Ranking(crate::SharedOutput<RankedDocument>),
    Decisions(crate::SharedOutput<DecisionAnswer>),
}

impl WorkloadOutput {
    pub fn attach_credit(&mut self, credit: CreditLease) {
        match self {
            Self::Tokens(tokens) => tokens.attach_byte_credit(credit),
            Self::Embedding(values) => values.attach_credit(credit),
            Self::Ranking(values) => values.attach_credit(credit),
            Self::Decisions(values) => values.attach_credit(credit),
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/workload_speculation.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/unit/workload_decision.rs"]
mod decision_tests;
