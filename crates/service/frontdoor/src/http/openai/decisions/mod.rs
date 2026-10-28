//! `POST /v1/decisions`: typed answers read from a generation model's first reply token.
//!
//! Each question is one engine request: the shared evidence, then the question with lettered
//! options, ending exactly where the model's reply begins. The engine reads the language-model
//! logits at that position and renormalizes them over the option letters, so a deployment that
//! generates can decide with the same weights and no extra head. Thinking is disabled because
//! the reply's first token has to be the answer.
mod request;
#[cfg(test)]
#[path = "../../../../tests/unit/http_openai_decisions_request.rs"]
mod tests;

pub(super) use request::DecisionRequest;

use super::execution::finish_reason;
use super::{ApiResult, TextState};
use crate::constants::{TEXT_STAGING_EXPANSION, TEXT_STAGING_OVERHEAD_BYTES};
use axum::http::HeaderMap;
use futures_util::{StreamExt, TryStreamExt, stream};
use infer_core::{Error, ErrorCode, ModelId, RequestId};
use infer_ir::{
    CanonicalRequest, DecisionQuestion, DecisionReadout, DecisionSchema, RequestInput, Workload,
    WorkloadOutput,
};
use infer_models::{ChatMessage, ChatOptions, TextAssets};
use infer_observe::trace::TraceContext;
use infer_runtime::EngineOutput;
use request::{Answering, LABELS, ReadPlan};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Probe content for finding where a reply begins; the template's reply prompt follows it.
const PROBE_TEXT: &str = "x";
/// Option scores are read untempered: the answer is the model's own distribution.
const CALIBRATION_TEMPERATURE: f32 = 1.0;

/// The tokens a reply starts with for each label, checked against this package's template.
struct ReplyLabels {
    /// The first reply token of each label, in label order.
    tokens: Vec<u32>,
    /// The template's reply prompt; every rendered question must end with it.
    reply_prompt: Vec<u32>,
    /// Tokens every question's prompt starts with: the system turn and the shared evidence. The
    /// engine keeps a snapshot there so later questions prefill only their own text.
    shared_prefix: Option<usize>,
}

/// One answered question and the prompt it read.
struct Read {
    answer: Value,
    prompt_tokens: usize,
    /// Prompt tokens restored from the shared prefix instead of prefilled.
    cached_tokens: usize,
}

/// The template options under which a reply's first token is the answer.
const fn reading() -> ChatOptions {
    ChatOptions {
        add_generation_prompt: true,
        enable_thinking: false,
        tools: None,
    }
}

impl ReplyLabels {
    /// Find each label's first reply token for this package.
    ///
    /// # Errors
    /// Returns an unsupported error when the template's reply prompt does not end at a token
    /// boundary, or a label is not one distinct token at the start of the reply.
    fn discover(assets: &TextAssets, plan: &ReadPlan, count: usize) -> infer_core::Result<Self> {
        let probe = [ChatMessage::new("user", PROBE_TEXT)];
        let opening = assets.render_chat(
            &probe,
            &ChatOptions {
                add_generation_prompt: false,
                ..reading()
            },
        )?;
        let prompt = assets.render_chat(&probe, &reading())?;
        let reply = prompt.strip_prefix(&opening).ok_or_else(|| {
            Error::unsupported("the chat template's reply prompt does not extend the conversation")
        })?;
        let prompt_tokens = assets.encode(&prompt, false)?;
        let reply_prompt = assets.encode(reply, false)?;
        if !prompt_tokens.ends_with(&reply_prompt) {
            return Err(Error::unsupported(
                "the chat template's reply prompt does not start at a token boundary",
            ));
        }
        let mut tokens = Vec::with_capacity(count);
        for label in LABELS.iter().take(count) {
            let extended = assets.encode(&format!("{prompt}{label}"), false)?;
            match extended.split_last() {
                Some((token, head)) if head == prompt_tokens && !tokens.contains(token) => {
                    tokens.push(*token);
                }
                _ => {
                    return Err(Error::unsupported(format!(
                        "label {label:?} is not one distinct token at the start of this model's reply"
                    )));
                }
            }
        }
        Ok(Self {
            tokens,
            reply_prompt,
            shared_prefix: shared_prefix(assets, plan)?,
        })
    }

    /// Encode one question as the chat prompt that ends where its answer begins.
    ///
    /// # Errors
    /// Returns a template, tokenization or capacity error, or an unsupported error when the
    /// rendered question does not end with the probed reply prompt.
    fn encode(
        &self,
        assets: &TextAssets,
        plan: &ReadPlan,
        index: usize,
    ) -> infer_core::Result<Vec<u32>> {
        let question = plan
            .questions
            .get(index)
            .ok_or_else(|| Error::invariant("decision question index out of range"))?;
        let tokens = assets.encode_chat(&messages(plan, &question.text), &reading())?;
        if !tokens.ends_with(&self.reply_prompt) {
            return Err(Error::unsupported(
                "this conversation ends the chat prompt differently, so the label tokens are unknown",
            ));
        }
        Ok(tokens)
    }
}

/// The chat turns one question reads: the system turn, then the evidence and the question.
fn messages(plan: &ReadPlan, question: &str) -> Vec<ChatMessage> {
    let content = if plan.evidence.is_empty() {
        question.to_owned()
    } else {
        format!("{}\n\n{question}", plan.evidence)
    };
    let mut messages = Vec::with_capacity(2);
    if let Some(system) = &plan.system {
        messages.push(ChatMessage::new("system", system.as_str()));
    }
    messages.push(ChatMessage::new("user", content));
    messages
}

/// The token prefix every question shares, found from two prompts that differ only in their
/// question text. Token equality is checked again when the engine reuses it, so this only
/// decides where a snapshot is worth keeping.
fn shared_prefix(assets: &TextAssets, plan: &ReadPlan) -> infer_core::Result<Option<usize>> {
    if plan.questions.len() < 2 || (plan.evidence.is_empty() && plan.system.is_none()) {
        return Ok(None);
    }
    let first = assets.encode_chat(&messages(plan, LABELS[0]), &reading())?;
    let second = assets.encode_chat(&messages(plan, LABELS[1]), &reading())?;
    let common = first
        .iter()
        .zip(&second)
        .take_while(|(left, right)| left == right)
        .count();
    Ok((common > 0).then_some(common))
}

/// Answer every question of one request.
///
/// # Errors
/// Returns a parameter error before any engine work for an invalid request, an unsupported
/// error when the package cannot be read this way, and a capacity or backend error from a read.
pub(super) async fn decide(
    state: TextState,
    headers: HeaderMap,
    request: DecisionRequest,
) -> ApiResult<Value> {
    let plan = Arc::new(request.plan()?);
    tracing::debug!(
        target: "infer::openai",
        questions = plan.questions.len(),
        safety_identifier = ?request.safety_identifier,
        prompt_cache_key = ?request.prompt_cache_key,
        explicit_cache = request.explicit_cache(),
        "decision request"
    );
    let inspection = state.handle.inspect().await?;
    if request.model != inspection.model_name {
        return Err(Error::new(ErrorCode::NotFound, "model is not served; see /v1/models").into());
    }
    // Questions share the device as concurrent requests do; more in flight than the funded pool
    // would only queue, and would hold ingress credit other clients need.
    let window = inspection
        .execution_profile
        .as_ref()
        .and_then(|profile| profile.concurrency.as_ref())
        .map_or(1, |concurrency| concurrency.selected.max(1));
    let options = plan.questions.iter().map(|q| q.options).max().unwrap_or(0);
    let assets = state.assets.clone();
    let labels = Arc::new(
        state
            .handle
            .cpu_pool()
            .run(probe_bytes(&plan)?, {
                let plan = Arc::clone(&plan);
                move |context| {
                    context.check()?;
                    ReplyLabels::discover(&assets, &plan, options)
                }
            })
            .await?,
    );
    let trace = super::super::trace_parent(&headers);
    // The first question publishes the shared prefix; the rest start from it. Starting them
    // together would prefill the evidence once per question before any snapshot exists.
    let mut reads = Vec::with_capacity(plan.questions.len());
    let rest = if labels.shared_prefix.is_some() {
        reads.push(
            read(
                &state,
                inspection.model,
                Arc::clone(&plan),
                Arc::clone(&labels),
                0,
                trace.clone(),
            )
            .await?,
        );
        1
    } else {
        0
    };
    let later: Vec<Read> = stream::iter(rest..plan.questions.len())
        .map(|index| {
            read(
                &state,
                inspection.model,
                Arc::clone(&plan),
                Arc::clone(&labels),
                index,
                trace.clone(),
            )
        })
        .buffered(window)
        .try_collect()
        .await?;
    reads.extend(later);
    let input_tokens: usize = reads.iter().map(|read| read.prompt_tokens).sum();
    let cached_tokens: usize = reads.iter().map(|read| read.cached_tokens).sum();
    Ok(json!({
        "model": inspection.model_name,
        "answers": reads.into_iter().map(|read| read.answer).collect::<Vec<_>>(),
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens,
        },
    }))
}

/// CPU staging for probing labels and the shared prefix: two renders of the evidence.
fn probe_bytes(plan: &ReadPlan) -> infer_core::Result<usize> {
    plan.evidence
        .len()
        .checked_add(plan.system.as_ref().map_or(0, String::len))
        .and_then(|bytes| bytes.checked_mul(2 * TEXT_STAGING_EXPANSION))
        .and_then(|bytes| bytes.checked_add(TEXT_STAGING_OVERHEAD_BYTES))
        .ok_or_else(|| Error::invalid("decision preparation size overflow"))
}

/// Submit one question and wait for its answer.
async fn read(
    state: &TextState,
    model: ModelId,
    plan: Arc<ReadPlan>,
    labels: Arc<ReplyLabels>,
    index: usize,
    trace: Option<TraceContext>,
) -> ApiResult<Read> {
    let question = plan
        .questions
        .get(index)
        .ok_or_else(|| Error::invariant("decision question index out of range"))?;
    let bytes = plan
        .evidence
        .len()
        .checked_add(plan.system.as_ref().map_or(0, String::len))
        .and_then(|bytes| bytes.checked_add(question.text.len()))
        .and_then(|bytes| bytes.checked_mul(TEXT_STAGING_EXPANSION))
        .and_then(|bytes| bytes.checked_add(TEXT_STAGING_OVERHEAD_BYTES))
        .ok_or_else(|| Error::invalid("decision preparation size overflow"))?;
    let options = labels
        .tokens
        .get(..question.options)
        .ok_or_else(|| Error::invariant("decision labels were not discovered"))?
        .to_vec();
    let id = state.handle.allocate_request_id()?;
    let prompt_tokens = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&prompt_tokens);
    let assets = state.assets.clone();
    let shared = Arc::clone(&plan);
    let mut receiver = state
        .handle
        .submit_preparing(id, bytes, trace, move |context| {
            context.check()?;
            let tokens = labels.encode(&assets, &shared, index)?;
            count.store(tokens.len(), Ordering::Relaxed);
            context.check()?;
            let boundary = labels.shared_prefix.filter(|prefix| *prefix < tokens.len());
            Ok(decision_request(id, model, (tokens, boundary), options))
        })
        .await?;
    while let Some(event) = receiver.recv().await {
        if let EngineOutput::Finished(result) = event? {
            finish_reason(&result.reason)?;
            let Some(WorkloadOutput::Decisions(answers)) = result.output else {
                return Err(Error::invariant("decision completed without answers").into());
            };
            let read = answers
                .first()
                .ok_or_else(|| Error::invariant("decision completed without answers"))?;
            return Ok(Read {
                answer: answer(
                    question.name.as_deref(),
                    &question.answering,
                    &read.probabilities,
                    read.label_mass.unwrap_or(1.0),
                )?,
                prompt_tokens: prompt_tokens.load(Ordering::Relaxed),
                cached_tokens: result.measurement.cached_tokens,
            });
        }
    }
    Err(Error::new(
        ErrorCode::Backend,
        "decision stream closed before completion",
    )
    .into())
}

fn decision_request(
    id: RequestId,
    model: ModelId,
    (tokens, prefix_boundary): (Vec<u32>, Option<usize>),
    options: Vec<u32>,
) -> CanonicalRequest {
    CanonicalRequest {
        id,
        model,
        session: None,
        input: RequestInput::Sequence {
            tokens: tokens.into(),
            media: Vec::new(),
        },
        workload: Workload::Decision(DecisionSchema {
            questions: vec![DecisionQuestion::Categorical { options }],
            calibration_temperature: CALIBRATION_TEMPERATURE,
            abstain_below: None,
            readout: DecisionReadout::LanguageModel,
        }),
        qos: infer_ir::Qos::default(),
        sampling: infer_ir::Sampling::default(),
        prefix_boundary,
        extensions: BTreeMap::new(),
    }
}

/// Build one protocol answer from the option probabilities.
///
/// `confidence` is the winning option's probability over the whole vocabulary, so a reply that
/// mostly wanted some other token reports low confidence even when one option dominates the rest.
///
/// # Errors
/// Returns an invariant error when the engine answered a different number of options.
fn answer(
    name: Option<&str>,
    answering: &Answering,
    probabilities: &[f32],
    label_mass: f32,
) -> infer_core::Result<Value> {
    let expected = match answering {
        Answering::Predicate => 2,
        Answering::Choice(values) => values.len(),
        Answering::Score(labels) => labels.len(),
    };
    if probabilities.len() != expected {
        return Err(Error::invariant(
            "decision answered a different option count",
        ));
    }
    let top = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
        .map_or(0, |(index, _)| index);
    let confidence = probabilities[top] * label_mass;
    Ok(match answering {
        Answering::Predicate => json!({
            "type": "predicate", "name": name, "probability": probabilities[1],
        }),
        Answering::Choice(values) => json!({
            "type": "choice",
            "name": name,
            "choice": values[top],
            "probabilities": values
                .iter()
                .zip(probabilities)
                .map(|(value, probability)| json!({"value": value, "probability": probability}))
                .collect::<Vec<_>>(),
            "confidence": confidence,
        }),
        Answering::Score(labels) => {
            // Level indices are the score's units: the expected index under the distribution.
            let (score, _) =
                probabilities
                    .iter()
                    .fold((0.0f64, 0.0f64), |(score, level), probability| {
                        (level.mul_add(f64::from(*probability), score), level + 1.0)
                    });
            json!({
                "type": "score",
                "name": name,
                "score": score,
                "probabilities": labels
                    .iter()
                    .zip(probabilities)
                    .enumerate()
                    .map(|(level, (label, probability))| {
                        json!({"value": level, "label": label, "probability": probability})
                    })
                    .collect::<Vec<_>>(),
                "confidence": confidence,
            })
        }
    })
}
