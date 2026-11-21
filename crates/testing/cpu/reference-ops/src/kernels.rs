use crate::moe::execute as moe_execute;
use infer_core::{Error, Result};
use infer_ir::TensorOp;
use infer_state::physical::PhysicalTensor;

/// Above this magnitude `softplus(x)` returns `x` directly to avoid `exp` overflow.
const SOFTPLUS_LINEAR_THRESHOLD: f32 = 20.0;
/// Floor added to the query/key sum of squares before the reciprocal square root.
const DELTA_NORM_EPSILON: f64 = 1e-6;

#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn softplus(x: f32) -> f32 {
    if x > SOFTPLUS_LINEAR_THRESHOLD {
        x
    } else {
        x.exp().ln_1p()
    }
}
#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
)]
fn norm(x: &[f32], weight: &[f32], dim: usize, epsilon: f32, offset: f32) -> Result<Vec<f32>> {
    if dim == 0 || !x.len().is_multiple_of(dim) || weight.len() != dim {
        return Err(Error::invalid("norm binding shape"));
    }
    let mut out = Vec::with_capacity(x.len());
    for head in x.chunks_exact(dim) {
        let scale = (head.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / dim as f64
            + f64::from(epsilon))
        .sqrt();
        out.extend(
            head.iter()
                .zip(weight)
                .map(|(x, w)| (f64::from(*x) / scale) as f32 * (w + offset)),
        );
    }
    Ok(out)
}
/// Execute an operation against validated test-backend bindings.
/// # Errors
/// Returns an error for inconsistent operation dimensions or physical state.
pub fn execute(
    op: &TensorOp,
    inputs: &[&[f32]],
    state: Option<&mut PhysicalTensor>,
    token: u32,
    position: usize,
) -> Result<Vec<Vec<f32>>> {
    let one = |x| Ok(vec![x]);
    match *op {
        TensorOp::Embedding => {
            let weight = inputs[0];
            // Shape/row width is resolved by the caller; input is the selected row.
            one(weight.to_vec())
        }
        TensorOp::MoeRoute { .. }
        | TensorOp::ExpertLinear { .. }
        | TensorOp::MoeCombine { .. }
        | TensorOp::Scale => moe_execute(op, inputs),
        TensorOp::Linear => linear(inputs),
        TensorOp::Norm {
            epsilon,
            offset,
            head_dim,
        } => one(norm(inputs[0], inputs[1], head_dim, epsilon, offset)?),
        TensorOp::Split { ref widths, heads } => split(inputs, widths, heads),
        TensorOp::Rope {
            heads,
            head_dim,
            rotary_dim,
            theta,
        } => rope(inputs, position, heads, head_dim, rotary_dim, theta),
        TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            window,
        } => attention(
            inputs,
            state.ok_or_else(|| Error::invalid("attention state binding"))?,
            position,
            query_heads,
            kv_heads,
            head_dim,
            window,
        ),
        TensorOp::Conv { channels, kernel } => conv(
            inputs,
            state.ok_or_else(|| Error::invalid("conv state binding"))?,
            channels,
            kernel,
        ),
        TensorOp::Delta {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            value_order,
        } => delta(
            inputs,
            state.ok_or_else(|| Error::invalid("delta state binding"))?,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            value_order.key_run(key_heads, value_heads),
        ),
        TensorOp::GatedNorm { head_dim, epsilon } => {
            let normalized = norm(inputs[0], inputs[2], head_dim, epsilon, 0.0)?;
            one(normalized
                .iter()
                .zip(inputs[1])
                .map(|(x, z)| x * silu(*z))
                .collect())
        }
        // The reference executes text tokens only, which no media feature reaches.
        TensorOp::MediaFeature { .. } => one(inputs[0].to_vec()),
        TensorOp::Silu => one(inputs[0].iter().map(|x| silu(*x)).collect()),
        TensorOp::Sigmoid => one(inputs[0].iter().map(|x| sigmoid(*x)).collect()),
        TensorOp::Multiply | TensorOp::Add => {
            if inputs[0].len() != inputs[1].len() {
                return Err(Error::invalid("elementwise dimensions"));
            }
            one(inputs[0]
                .iter()
                .zip(inputs[1])
                .map(|(a, b)| {
                    if matches!(op, TensorOp::Add) {
                        a + b
                    } else {
                        a * b
                    }
                })
                .collect())
        }
    }
    .and_then(|outputs| {
        if outputs.iter().flatten().any(|v| !v.is_finite()) {
            Err(Error::invalid(format!(
                "non-finite host op at token {token}, position {position}"
            )))
        } else {
            Ok(outputs)
        }
    })
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
fn linear(inputs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
    let one = |x| Ok(vec![x]);
    let x = inputs[0];
    if !inputs[1].len().is_multiple_of(x.len()) {
        return Err(Error::invalid("linear weight shape"));
    }
    one(inputs[1]
        .chunks_exact(x.len())
        .map(|row| {
            row.iter()
                .zip(x)
                .map(|(w, x)| f64::from(*w) * f64::from(*x))
                .sum::<f64>() as f32
        })
        .collect())
}

fn split(inputs: &[&[f32]], widths: &[usize], heads: usize) -> Result<Vec<Vec<f32>>> {
    let stride: usize = widths.iter().sum();
    if inputs[0].len() != stride * heads {
        return Err(Error::invalid("split shape"));
    }
    let mut offset = 0;
    let mut outputs = Vec::new();
    for width in widths {
        let mut out = Vec::with_capacity(width * heads);
        for head in 0..heads {
            out.extend_from_slice(
                &inputs[0][head * stride + offset..head * stride + offset + width],
            );
        }
        outputs.push(out);
        offset += width;
    }
    Ok(outputs)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "Separate multiply/add rounding preserves the numerical contract of the independent Torch golden and the scalar reference"
)]
fn rope(
    inputs: &[&[f32]],
    position: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f64,
) -> Result<Vec<Vec<f32>>> {
    let one = |x| Ok(vec![x]);
    let mut x = inputs[0].to_vec();
    if x.len() != heads * head_dim {
        return Err(Error::invalid("rope shape"));
    }
    for head in x.chunks_exact_mut(head_dim) {
        for i in 0..rotary_dim / 2 {
            let angle = position as f64 / theta.powf((2 * i) as f64 / rotary_dim as f64);
            let (sin, cos) = angle.sin_cos();
            let a = f64::from(head[i]);
            let b = f64::from(head[i + rotary_dim / 2]);
            head[i] = (a * cos - b * sin) as f32;
            head[i + rotary_dim / 2] = (a * sin + b * cos) as f32;
        }
    }
    one(x)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
)]
fn attention(
    inputs: &[&[f32]],
    state: &mut PhysicalTensor,
    position: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: Option<usize>,
) -> Result<Vec<Vec<f32>>> {
    let one = |x| Ok(vec![x]);
    let PhysicalTensor::Kv { keys, values } = state else {
        return Err(Error::invalid("attention state binding"));
    };
    if keys.rows != position || values.rows != position || inputs[0].len() != query_heads * head_dim
    {
        return Err(Error::invalid("attention position/shape"));
    }
    keys.push(inputs[1])?;
    values.push(inputs[2])?;
    let start = window.map_or(0, |w| (position + 1).saturating_sub(w));
    let mut out = vec![0.0; query_heads * head_dim];
    for head in 0..query_heads {
        let key_head = head / (query_heads / kv_heads);
        let q = &inputs[0][head * head_dim..(head + 1) * head_dim];
        let scores = (start..=position)
            .map(|p| {
                let k = keys.row(p)?;
                Ok(q.iter()
                    .zip(&k[key_head * head_dim..(key_head + 1) * head_dim])
                    .map(|(q, k)| f64::from(*q) * f64::from(*k))
                    .sum::<f64>()
                    / (head_dim as f64).sqrt())
            })
            .collect::<Result<Vec<_>>>()?;
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let probs: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f64 = probs.iter().sum();
        for (i, prob) in probs.iter().enumerate() {
            let v = values.row(start + i)?;
            for d in 0..head_dim {
                out[head * head_dim + d] +=
                    (prob / sum * f64::from(v[key_head * head_dim + d])) as f32;
            }
        }
    }
    one(out)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "Separate multiply/add rounding preserves the numerical contract of the independent Torch golden and the scalar reference"
)]
fn conv(
    inputs: &[&[f32]],
    state: &mut PhysicalTensor,
    channels: usize,
    kernel: usize,
) -> Result<Vec<Vec<f32>>> {
    let one = |x| Ok(vec![x]);
    let PhysicalTensor::Conv { history, .. } = state else {
        return Err(Error::invalid("conv state binding"));
    };
    if inputs[0].len() != channels
        || inputs[1].len() != channels * kernel
        || history.len() != channels * (kernel - 1)
    {
        return Err(Error::invalid("conv shape"));
    }
    let mut out = vec![0.0; channels];
    for channel in 0..channels {
        let row = &inputs[1][channel * kernel..(channel + 1) * kernel];
        let past = &mut history[channel * (kernel - 1)..(channel + 1) * (kernel - 1)];
        let sum = past
            .iter()
            .zip(row)
            .map(|(x, w)| f64::from(*x) * f64::from(*w))
            .sum::<f64>()
            + f64::from(inputs[0][channel]) * f64::from(row[kernel - 1]);
        out[channel] = silu(sum as f32);
        if !past.is_empty() {
            past.rotate_left(1);
            *past
                .last_mut()
                .ok_or_else(|| Error::invariant("nonempty"))? = inputs[0][channel];
        }
    }
    one(out)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
)]
fn delta(
    inputs: &[&[f32]],
    state: &mut PhysicalTensor,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    key_run: usize,
) -> Result<Vec<Vec<f32>>> {
    let one = |x| Ok(vec![x]);
    let PhysicalTensor::Delta { recurrent, .. } = state else {
        return Err(Error::invalid("delta state binding"));
    };
    let key_size = key_heads * key_dim;
    let value_size = value_heads * value_dim;
    if inputs[0].len() != 2 * key_size + value_size
        || recurrent.len() != value_heads * key_dim * value_dim
    {
        return Err(Error::invalid("delta dimensions"));
    }
    let mut out = vec![0.0; value_size];
    for head in 0..value_heads {
        let key_head = head / key_run % key_heads;
        let q = &inputs[0][key_head * key_dim..(key_head + 1) * key_dim];
        let k = &inputs[0][key_size + key_head * key_dim..key_size + (key_head + 1) * key_dim];
        let v = &inputs[0][2 * key_size + head * value_dim..2 * key_size + (head + 1) * value_dim];
        let qscale = (q.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() + DELTA_NORM_EPSILON)
            .sqrt()
            * (key_dim as f64).sqrt();
        let kscale =
            (k.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() + DELTA_NORM_EPSILON).sqrt();
        let q: Vec<_> = q.iter().map(|v| f64::from(*v) / qscale).collect();
        let k: Vec<_> = k.iter().map(|v| f64::from(*v) / kscale).collect();
        let beta = f64::from(sigmoid(inputs[1][head]));
        let log_decay = -inputs[3][head].exp() * softplus(inputs[2][head] + inputs[4][head]);
        let decay = f64::from(log_decay).exp();
        let matrix = &mut recurrent[head * key_dim * value_dim..(head + 1) * key_dim * value_dim];
        for s in matrix.iter_mut() {
            *s = (f64::from(*s) * decay) as f32;
        }
        for d in 0..value_dim {
            let predicted: f64 = (0..key_dim)
                .map(|i| f64::from(matrix[i * value_dim + d]) * k[i])
                .sum();
            let delta = (f64::from(v[d]) - predicted) * beta;
            for i in 0..key_dim {
                matrix[i * value_dim + d] += (k[i] * delta) as f32;
            }
            out[head * value_dim + d] = (0..key_dim)
                .map(|i| f64::from(matrix[i * value_dim + d]) * q[i])
                .sum::<f64>() as f32;
        }
    }
    one(out)
}
