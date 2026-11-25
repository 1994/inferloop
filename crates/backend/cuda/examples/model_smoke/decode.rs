use super::{Model, mtp::Mtp};
use infer_core::Result;
use infer_ir::Sampling;
use infer_workloads::{
    SamplingHistory, Verification, draw_distribution, probabilities, sampling_uniform, verify_draft,
};
use serde::Serialize;
use std::iter::once;

#[derive(Default, Serialize)]
pub struct DecodeResult {
    pub tokens: Vec<u32>,
    pub proposals: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub draft_state_restores: usize,
    pub target_steps: usize,
    pub target_batches: usize,
    pub draft_steps: usize,
}

struct Proposal {
    token: u32,
    probabilities: Vec<f64>,
}

pub fn decode(
    model: &mut Model,
    mut mtp: Option<&mut Mtp>,
    prompt: &[u32],
    mut logits: Vec<f32>,
    sampling: &Sampling,
    limit: usize,
    depth: usize,
) -> Result<DecodeResult> {
    let mut result = DecodeResult::default();
    let mut pending = None;
    while result.tokens.len() < limit {
        let first = if let Some(token) = pending.take() {
            token
        } else {
            let p = probabilities(
                &logits,
                sampling,
                SamplingHistory {
                    prompt,
                    generated: &result.tokens,
                    pending: &[],
                },
            )?;
            draw_distribution(&p, sampling_uniform(sampling.seed, 1, result.tokens.len()))?
        };
        result.tokens.push(first);
        if sampling.is_eos(first) || result.tokens.len() == limit {
            break;
        }
        let Some(draft) = mtp.as_deref_mut() else {
            logits = model.step(first, prompt.len() + result.tokens.len() - 1, true)?;
            result.target_steps += 1;
            continue;
        };
        let checkpoint = draft.checkpoint();
        let proposals = propose(
            draft,
            model,
            prompt,
            &result.tokens,
            sampling,
            depth.min(limit - result.tokens.len()),
        )?;
        result.proposals += proposals.len();
        result.draft_steps += proposals.len();
        let mut replay = vec![(first, model.hidden.clone())];
        let position = prompt.len() + result.tokens.len() - 1;
        let mut steps = super::verification::TargetSteps::new(
            model,
            once(first).chain(proposals.iter().map(|p| p.token)),
            position,
            &mut result,
        )?;
        logits = steps.next(model, first, position, &mut result)?;
        for proposal in proposals {
            let p = probabilities(
                &logits,
                sampling,
                SamplingHistory {
                    prompt,
                    generated: &result.tokens,
                    pending: &[],
                },
            )?;
            let position = result.tokens.len();
            match verify_draft(
                &p,
                &proposal.probabilities,
                proposal.token,
                sampling_uniform(sampling.seed, 3, position),
                sampling_uniform(sampling.seed, 4, position),
            )? {
                Verification::Accepted(token) => {
                    result.accepted += 1;
                    result.tokens.push(token);
                    if sampling.is_eos(token) || result.tokens.len() == limit {
                        break;
                    }
                    replay.push((token, model.hidden.clone()));
                    logits = steps.next(
                        model,
                        token,
                        prompt.len() + result.tokens.len() - 1,
                        &mut result,
                    )?;
                }
                Verification::Replaced(token) => {
                    result.rejected += 1;
                    pending = Some(token);
                    break;
                }
            }
        }
        steps.finish(model)?;
        draft.restore(checkpoint)?;
        result.draft_state_restores += 1;
        if result.tokens.last().is_some_and(|t| sampling.is_eos(*t)) || result.tokens.len() == limit
        {
            break;
        }
        let start = prompt.len() + result.tokens.len() - replay.len();
        for (offset, (token, hidden)) in replay.into_iter().enumerate() {
            draft.step(token, &hidden, start + offset, false)?;
            result.draft_steps += 1;
        }
    }
    Ok(result)
}

fn propose(
    draft: &mut Mtp,
    model: &Model,
    prompt: &[u32],
    generated: &[u32],
    sampling: &Sampling,
    depth: usize,
) -> Result<Vec<Proposal>> {
    let mut history = generated.to_vec();
    let mut hidden = model.hidden.clone();
    let mut proposals = Vec::new();
    for _ in 0..depth {
        let previous = history.last().copied().unwrap_or_default();
        let logits = draft.step(previous, &hidden, prompt.len() + history.len() - 1, true)?;
        let distribution = probabilities(
            &logits,
            sampling,
            SamplingHistory {
                prompt,
                generated: &history,
                pending: &[],
            },
        )?;
        let token = draw_distribution(
            &distribution,
            sampling_uniform(sampling.seed, 2, history.len()),
        )?;
        proposals.push(Proposal {
            token,
            probabilities: distribution,
        });
        hidden.clone_from(&draft.model.hidden);
        history.push(token);
        if sampling.is_eos(token) {
            break;
        }
    }
    Ok(proposals)
}
