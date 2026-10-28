//! Outputs workload responsibilities.
use super::math::log_normalizer;
use super::{pool, softmax};
use infer_core::{Error, Result};
use infer_ir::{
    DecisionAnswer, DecisionQuestion, ModelOutput, Pooling, RankedDocument, WorkloadOutput,
};

#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
pub fn embedding(
    first: &ModelOutput,
    output_count: usize,
    pooling: Pooling,
    dimensions: Option<usize>,
    normalize: bool,
) -> Result<WorkloadOutput> {
    if output_count != 1 {
        return Err(Error::invariant("embedding expects one output"));
    }
    let mut v = pool(&first.hidden, pooling)?;
    if let Some(d) = dimensions {
        if d == 0 || d > v.len() {
            return Err(Error::invalid("invalid embedding width"));
        }
        v.truncate(d);
    }
    if normalize {
        let norm = v.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x = (f64::from(*x) / norm) as f32;
            }
        }
    }
    Ok(WorkloadOutput::Embedding(v.into()))
}

pub fn ranking(outputs: &[ModelOutput], top_k: usize) -> Result<WorkloadOutput> {
    let mut ranking = outputs
        .iter()
        .enumerate()
        .map(|(index, o)| {
            Ok(RankedDocument {
                index,
                score: *o
                    .logits
                    .first()
                    .filter(|x| x.is_finite())
                    .ok_or_else(|| Error::invariant("invalid rank readout"))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ranking.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.index.cmp(&b.index)));
    ranking.truncate(top_k);
    Ok(WorkloadOutput::Ranking(ranking.into()))
}

pub fn decisions(
    first: &ModelOutput,
    output_count: usize,
    schema: &infer_ir::DecisionSchema,
) -> Result<WorkloadOutput> {
    if output_count != 1 {
        return Err(Error::invariant("decision expects one backbone output"));
    }
    // Every question reads the same scores, so their normalizer is computed once.
    let normalizer = log_normalizer(&first.logits, schema.calibration_temperature)?;
    let mut answers = Vec::with_capacity(schema.questions.len());
    for question in &schema.questions {
        answers.push(answer_question(
            &first.logits,
            normalizer,
            question,
            schema,
        )?);
    }
    Ok(WorkloadOutput::Decisions(answers.into()))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "Separate multiply/add rounding preserves the numerical contract of the independent Torch golden and the scalar reference"
)]
pub fn answer_question(
    logits: &[f32],
    normalizer: f64,
    question: &DecisionQuestion,
    schema: &infer_ir::DecisionSchema,
) -> Result<DecisionAnswer> {
    let read = |token: u32| {
        logits
            .get(token as usize)
            .copied()
            .filter(|x| x.is_finite())
            .ok_or_else(|| Error::invariant("invalid decision logit"))
    };
    let (probabilities, expected) = match question {
        DecisionQuestion::Binary {
            negative_token,
            positive_token,
        } => (
            softmax(
                &[read(*negative_token)?, read(*positive_token)?],
                schema.calibration_temperature,
            )?,
            None,
        ),
        DecisionQuestion::Categorical { options } => (
            softmax(
                &options
                    .iter()
                    .map(|t| read(*t))
                    .collect::<Result<Vec<_>>>()?,
                schema.calibration_temperature,
            )?,
            None,
        ),
        DecisionQuestion::Ordinal { options, values } => {
            let p = softmax(
                &options
                    .iter()
                    .map(|t| read(*t))
                    .collect::<Result<Vec<_>>>()?,
                schema.calibration_temperature,
            )?;
            let expected = (p
                .iter()
                .zip(values)
                .map(|(p, v)| f64::from(*p) * f64::from(*v))
                .sum::<f64>()
                / p.iter().map(|p| f64::from(*p)).sum::<f64>())
            .clamp(
                f64::from(values[0]),
                f64::from(
                    *values
                        .last()
                        .ok_or_else(|| Error::invariant("validated ordinal values"))?,
                ),
            ) as f32;
            (p, Some(expected))
        }
        DecisionQuestion::Continuous { token, min, max } => {
            let v = read(*token)? / schema.calibration_temperature;
            let s = if v >= 0.0 {
                1.0 / (1.0 + (-v).exp())
            } else {
                let e = v.exp();
                e / (1.0 + e)
            };
            (
                vec![],
                Some((f64::from(*min) + (f64::from(*max) - f64::from(*min)) * f64::from(s)) as f32),
            )
        }
    };
    let mass = |tokens: &[u32]| -> Result<Option<f32>> {
        let scale = f64::from(schema.calibration_temperature);
        tokens
            .iter()
            .try_fold(0.0f64, |sum, token| {
                Ok(sum + (f64::from(read(*token)?) / scale - normalizer).exp())
            })
            .map(|mass| Some(mass.min(1.0) as f32))
    };
    let label_mass = match question {
        DecisionQuestion::Binary {
            negative_token,
            positive_token,
        } => mass(&[*negative_token, *positive_token])?,
        DecisionQuestion::Categorical { options } | DecisionQuestion::Ordinal { options, .. } => {
            mass(options)?
        }
        DecisionQuestion::Continuous { .. } => None,
    };
    let abstained = schema.abstain_below.is_some_and(|t| {
        probabilities
            .iter()
            .copied()
            .reduce(f32::max)
            .is_some_and(|p| p < t)
    });
    Ok(DecisionAnswer {
        probabilities,
        expected,
        abstained,
        label_mass,
    })
}
