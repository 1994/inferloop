//! Accepted-prefix reference for recurrent target state.
//!
//! R1 commits recurrent state by replaying the accepted prefix of a recorded window instead of
//! snapshotting every candidate. Two steps carry that: the gated-delta update and the causal
//! convolution window. Their numbers lived in the CPU test executor that E1 deletes, so they were
//! preserved as goldens first (`examples/recurrent-delta`, `examples/recurrent-conv`); this module
//! is the reference those goldens are compared against from inside the codebase, and the place a
//! device kernel can be compared against as well.
//!
//! The arithmetic keeps the reference's rounding points: the state is `f32`, the accumulation is
//! `f64`, and a step rounds exactly where the recorded implementation rounded. The `expect`
//! attributes below are the workspace's usual way of naming those deliberate casts.

use infer_core::{Error, Result};

/// The input rows one gated-delta step reads: `[q | k | v]` and the four per-value-head controls,
/// in the order the recorded reference indexes them.
const DELTA_ROWS: usize = 5;
const DELTA_GATE_ROW: usize = 1;
const DELTA_SOFTPLUS_ROW: usize = 2;
const DELTA_DECAY_ROW: usize = 3;
const DELTA_DECAY_BIAS_ROW: usize = 4;

/// Geometry of one gated-delta state step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeltaGeometry {
    /// Key heads; each is shared by `value_heads / key_heads` value heads.
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub norm_epsilon: f64,
}

impl DeltaGeometry {
    #[must_use]
    pub const fn state_len(&self) -> usize {
        self.value_heads * self.key_dim * self.value_dim
    }

    /// The number of input rows one step reads.
    #[must_use]
    pub const fn rows(&self) -> usize {
        DELTA_ROWS
    }

    /// One step: normalize q and k, decay the state, apply the rank-one update, read out.
    ///
    /// `inputs[0]` is `[q | k | v]`; rows 1 to 4 carry the per-value-head gate, the softplus input,
    /// and the log-decay inputs.
    ///
    /// # Errors
    /// Rejects a degenerate geometry, an input whose width does not match it, or a state of the
    /// wrong length.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The state is stored in F32 and the reference rounds back to it exactly here"
    )]
    #[expect(
        clippy::cast_precision_loss,
        reason = "Head dimensions are small counts; the reference scales by their square root"
    )]
    pub fn step(&self, state: &mut [f32], inputs: &[&[f32]]) -> Result<Vec<f32>> {
        let key_size = self.key_heads * self.key_dim;
        let value_size = self.value_heads * self.value_dim;
        if self.key_heads == 0
            || self.key_dim == 0
            || self.value_dim == 0
            || !self.value_heads.is_multiple_of(self.key_heads)
        {
            return Err(Error::invalid("delta geometry"));
        }
        if inputs.len() != self.rows()
            || inputs[0].len() != 2 * key_size + value_size
            || state.len() != self.state_len()
        {
            return Err(Error::invalid("delta input shape"));
        }
        let mut out = vec![0.0_f32; value_size];
        let head_ratio = self.value_heads / self.key_heads;
        for head in 0..self.value_heads {
            let key_head = head / head_ratio;
            let q = &inputs[0][key_head * self.key_dim..(key_head + 1) * self.key_dim];
            let k = &inputs[0]
                [key_size + key_head * self.key_dim..key_size + (key_head + 1) * self.key_dim];
            let v = &inputs[0]
                [2 * key_size + head * self.value_dim..2 * key_size + (head + 1) * self.value_dim];
            let q_scale =
                (sum_squares(q) + self.norm_epsilon).sqrt() * (self.key_dim as f64).sqrt();
            let k_scale = (sum_squares(k) + self.norm_epsilon).sqrt();
            let q: Vec<f64> = q.iter().map(|value| f64::from(*value) / q_scale).collect();
            let k: Vec<f64> = k.iter().map(|value| f64::from(*value) / k_scale).collect();
            let beta = sigmoid(f64::from(inputs[DELTA_GATE_ROW][head]));
            let log_decay = -f64::from(inputs[DELTA_DECAY_ROW][head]).exp()
                * softplus(
                    f64::from(inputs[DELTA_SOFTPLUS_ROW][head])
                        + f64::from(inputs[DELTA_DECAY_BIAS_ROW][head]),
                );
            let decay = log_decay.exp();
            let base = head * self.key_dim * self.value_dim;
            for value in &mut state[base..base + self.key_dim * self.value_dim] {
                *value = (f64::from(*value) * decay) as f32;
            }
            for d in 0..self.value_dim {
                let predicted: f64 = k
                    .iter()
                    .enumerate()
                    .map(|(i, key)| f64::from(state[base + i * self.value_dim + d]) * key)
                    .sum();
                let delta = (f64::from(v[d]) - predicted) * beta;
                for (i, key) in k.iter().enumerate() {
                    let index = base + i * self.value_dim + d;
                    state[index] += (key * delta) as f32;
                }
                out[head * self.value_dim + d] = q
                    .iter()
                    .enumerate()
                    .map(|(i, query)| f64::from(state[base + i * self.value_dim + d]) * query)
                    .sum::<f64>() as f32;
            }
        }
        Ok(out)
    }

    /// R1's commit: replay only the accepted steps of the recorded window from the base state.
    ///
    /// # Errors
    /// Rejects an accepted count beyond what was recorded, or a step that does not fit the geometry.
    pub fn fold(
        &self,
        base: &[f32],
        recorded: &[Vec<Vec<f32>>],
        accepted: usize,
    ) -> Result<Vec<f32>> {
        if accepted > recorded.len() {
            return Err(Error::invalid("accepted prefix beyond the recorded window"));
        }
        let mut state = base.to_vec();
        for rows in recorded.iter().take(accepted) {
            let borrowed: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
            self.step(&mut state, &borrowed)?;
        }
        Ok(state)
    }
}

/// Geometry of one causal convolution step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvGeometry {
    pub channels: usize,
    pub kernel: usize,
}

impl ConvGeometry {
    /// The trailing window one channel keeps, which is also the state length.
    #[must_use]
    pub const fn history_len(&self) -> usize {
        self.channels * (self.kernel - 1)
    }

    /// One step: weigh the window and the current input, round through a `SiLU`, rotate the
    /// window.
    ///
    /// `weights` holds one row of `kernel` weights per channel.
    ///
    /// # Errors
    /// Rejects a kernel smaller than two, a mismatched activation or weight row, or a state of the
    /// wrong length.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The reference rounds the accumulated f64 sum back to f32 through the SiLU"
    )]
    #[expect(
        clippy::suboptimal_flops,
        reason = "Separate multiply and add keep the rounding the recorded reference used"
    )]
    pub fn step(
        &self,
        history: &mut [f32],
        activation: &[f32],
        weights: &[f32],
    ) -> Result<Vec<f32>> {
        if self.kernel < 2
            || self.channels == 0
            || activation.len() != self.channels
            || weights.len() != self.channels * self.kernel
            || history.len() != self.history_len()
        {
            return Err(Error::invalid("convolution input shape"));
        }
        let window = self.kernel - 1;
        let mut out = vec![0.0_f32; self.channels];
        for channel in 0..self.channels {
            let row = &weights[channel * self.kernel..(channel + 1) * self.kernel];
            let past = &history[channel * window..(channel + 1) * window];
            let total = past
                .iter()
                .zip(row)
                .map(|(x, w)| f64::from(*x) * f64::from(*w))
                .sum::<f64>()
                + f64::from(activation[channel]) * f64::from(row[self.kernel - 1]);
            out[channel] = silu(total) as f32;
            let past = &mut history[channel * window..(channel + 1) * window];
            if !past.is_empty() {
                past.rotate_left(1);
                past[window - 1] = activation[channel];
            }
        }
        Ok(out)
    }

    /// R1's commit for the window: the trailing `kernel - 1` accepted inputs per channel.
    ///
    /// # Errors
    /// Rejects an accepted input whose width is not `channels`, or a base window of the wrong size.
    pub fn fold(&self, base: &[f32], accepted: &[Vec<f32>]) -> Result<Vec<f32>> {
        if base.len() != self.history_len() {
            return Err(Error::invalid("convolution base window"));
        }
        if self.channels > 0 && accepted.iter().any(|row| row.len() != self.channels) {
            return Err(Error::invalid("accepted activation width"));
        }
        let window = self.kernel - 1;
        let retained = window.saturating_sub(accepted.len());
        let suffix = &accepted[accepted.len().saturating_sub(window)..];
        let mut folded = vec![0.0_f32; self.history_len()];
        for channel in 0..self.channels {
            let start = channel * window;
            let row = &mut folded[start..start + window];
            row[..retained].copy_from_slice(&base[start + window - retained..start + window]);
            for (value, activation) in row[retained..].iter_mut().zip(suffix) {
                *value = activation[channel];
            }
        }
        Ok(folded)
    }
}

fn sum_squares(values: &[f32]) -> f64 {
    values.iter().map(|value| f64::from(*value).powi(2)).sum()
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn softplus(value: f64) -> f64 {
    value.exp().ln_1p()
}

fn silu(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}
