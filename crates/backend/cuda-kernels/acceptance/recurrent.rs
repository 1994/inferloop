//! Acceptance of the recurrent step cells, shared by every kernel source.
//!
//! A source records a case through the `Recurrent` contract and hands the outputs and the
//! advanced state to `check`. The references compute in F64 in index order; a source reduces in
//! its own order, so results compare within a tolerance and never bit for bit across sources.

/// Absolute tolerance on outputs and state, whose magnitudes stay below a few units here.
const ABSOLUTE_TOLERANCE: f64 = 2.0e-5;
/// Relative tolerance on larger values.
const RELATIVE_TOLERANCE: f64 = 1.0e-5;
/// Metadata fields of one lane, as the resident metadata lays them out.
pub const METADATA_FIELDS: usize = 4;
/// The field holding the lane's position; a negative position marks an idle lane.
const POSITION_FIELD: usize = 2;
/// The convolution's width: three history rows and the current token.
pub const CONV_WIDTH: usize = 4;
/// Inputs are sixteenths centred on zero, exact in F32.
const STEPS: f32 = 16.0;
/// Residues of the deterministic inputs.
const MODULUS: usize = 17;
/// The residue that maps to zero.
const CENTRE: f32 = 8.0;
/// Squared-norm floor of the query and key normalization.
const NORM_EPSILON: f64 = 1.0e-6;
/// Above this the softplus of the gate is the identity.
const SOFTPLUS_LINEAR: f64 = 20.0;

fn sixteenths(count: usize, stride: usize, shift: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let step = u16::try_from((index * stride + shift) % MODULUS).unwrap_or(0);
            (f32::from(step) - CENTRE) / STEPS
        })
        .collect()
}

fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().copied().map(f64::from).collect()
}

fn silu(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

/// One lane's metadata at `position`.
#[must_use]
pub fn metadata(position: i32) -> Vec<i32> {
    let mut metadata = vec![0; METADATA_FIELDS];
    metadata[POSITION_FIELD] = position;
    metadata
}

/// Compare `actual` with `expected`, naming the first difference.
/// # Errors
/// The first element that is not finite or lies outside the tolerance, or a wrong length.
pub fn compare(label: &str, actual: &[f32], expected: &[f64]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("{label}: {} values", actual.len()));
    }
    for (index, (value, reference)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(*value) - reference).abs();
        if !value.is_finite()
            || error
                > reference
                    .abs()
                    .mul_add(RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)
        {
            return Err(format!("{label} [{index}]: {value} vs {reference}"));
        }
    }
    Ok(())
}

/// One token of the causal depthwise convolution over `channels`.
#[derive(Clone, Copy, Debug)]
pub struct Conv {
    pub channels: usize,
    /// An idle lane writes zeros and leaves its history untouched.
    pub active: bool,
}

impl Conv {
    #[must_use]
    pub fn input(self) -> Vec<f32> {
        sixteenths(self.channels, 7, 1)
    }

    /// History rows, oldest first, each `[channels]`.
    #[must_use]
    pub fn history(self) -> [Vec<f32>; CONV_WIDTH - 1] {
        [3, 5, 9].map(|stride| sixteenths(self.channels, stride, stride))
    }

    /// `[channels, CONV_WIDTH]`, the oldest tap first.
    #[must_use]
    pub fn weight(self) -> Vec<f32> {
        sixteenths(self.channels * CONV_WIDTH, 11, 4)
    }

    #[must_use]
    pub fn metadata(self) -> Vec<i32> {
        metadata(if self.active { 0 } else { -1 })
    }

    /// The output and the history after the step.
    #[must_use]
    pub fn expected(self) -> (Vec<f64>, [Vec<f64>; CONV_WIDTH - 1]) {
        let (input, history, weight) = (self.input(), self.history(), self.weight());
        let history = history.map(|rows| widen(&rows));
        if !self.active {
            return (vec![0.0; self.channels], history);
        }
        let output = (0..self.channels)
            .map(|channel| {
                let taps = &weight[channel * CONV_WIDTH..(channel + 1) * CONV_WIDTH];
                let window = [
                    history[0][channel],
                    history[1][channel],
                    history[2][channel],
                    f64::from(input[channel]),
                ];
                silu(
                    window
                        .iter()
                        .zip(taps)
                        .map(|(value, tap)| value * f64::from(*tap))
                        .sum(),
                )
            })
            .collect();
        let [_, middle, newest] = history;
        (output, [middle, newest, widen(&input)])
    }

    /// # Errors
    /// Names the first output or history element outside the tolerance.
    pub fn check(self, output: &[f32], history: [&[f32]; CONV_WIDTH - 1]) -> Result<(), String> {
        let (expected, rows) = self.expected();
        compare(&format!("{self:?} output"), output, &expected)?;
        for (index, (actual, expected)) in history.iter().zip(&rows).enumerate() {
            compare(&format!("{self:?} history {index}"), actual, expected)?;
        }
        Ok(())
    }
}

/// One token of the gated delta rule over `value_heads` square `dim × dim` states.
#[derive(Clone, Copy, Debug)]
pub struct Delta {
    pub key_heads: usize,
    pub value_heads: usize,
    pub dim: usize,
    pub active: bool,
}

impl Delta {
    /// `[q (key_heads × dim), k (key_heads × dim), v (value_heads × dim)]`.
    #[must_use]
    pub fn qkv(self) -> Vec<f32> {
        sixteenths((2 * self.key_heads + self.value_heads) * self.dim, 7, 2)
    }

    /// `[value_heads, dim (key), dim (value)]`.
    #[must_use]
    pub fn state(self) -> Vec<f32> {
        sixteenths(self.value_heads * self.dim * self.dim, 5, 3)
            .into_iter()
            .map(|value| value / STEPS)
            .collect()
    }

    /// Per value head: the update gate's logit, the decay gate's logit, its log rate and bias.
    #[must_use]
    pub fn gates(self) -> [Vec<f32>; 4] {
        [(3, 0), (5, 2), (7, 9), (11, 4)]
            .map(|(stride, shift)| sixteenths(self.value_heads, stride, shift))
    }

    #[must_use]
    pub fn metadata(self) -> Vec<i32> {
        metadata(if self.active { 0 } else { -1 })
    }

    /// The output `[value_heads × dim]` and the state after the step.
    #[must_use]
    pub fn expected(self) -> (Vec<f64>, Vec<f64>) {
        let (qkv, state) = (widen(&self.qkv()), widen(&self.state()));
        let dim = self.dim;
        if !self.active {
            return (vec![0.0; self.value_heads * dim], state);
        }
        let [beta, alpha, rate, bias] = self.gates().map(|gate| widen(&gate));
        let width = f64::from(u32::try_from(dim).unwrap_or(u32::MAX));
        let unit = |values: &[f64], scale: f64| {
            let norm: f64 = values.iter().map(|x| x * x).sum::<f64>() + NORM_EPSILON;
            values
                .iter()
                .map(|x| x * scale / norm.sqrt())
                .collect::<Vec<_>>()
        };
        let mut output = Vec::with_capacity(self.value_heads * dim);
        let mut next = state.clone();
        let group = self.value_heads / self.key_heads;
        for head in 0..self.value_heads {
            let key = head / group;
            let q = unit(&qkv[key * dim..(key + 1) * dim], 1.0 / width.sqrt());
            let k = unit(
                &qkv[(self.key_heads + key) * dim..(self.key_heads + key + 1) * dim],
                1.0,
            );
            let v = &qkv[(2 * self.key_heads + head) * dim..(2 * self.key_heads + head + 1) * dim];
            let update = 1.0 / (1.0 + (-beta[head]).exp());
            let logit = alpha[head] + bias[head];
            let softplus = if logit > SOFTPLUS_LINEAR {
                logit
            } else {
                logit.exp().ln_1p()
            };
            let decay = (-rate[head].exp() * softplus).exp();
            let block = &mut next[head * dim * dim..(head + 1) * dim * dim];
            for value in block.iter_mut() {
                *value *= decay;
            }
            for column in 0..dim {
                let predicted: f64 = (0..dim).map(|row| block[row * dim + column] * k[row]).sum();
                let difference = (v[column] - predicted) * update;
                for row in 0..dim {
                    block[row * dim + column] += k[row] * difference;
                }
                output.push((0..dim).map(|row| block[row * dim + column] * q[row]).sum());
            }
        }
        (output, next)
    }

    /// # Errors
    /// Names the first output or state element outside the tolerance.
    pub fn check(self, output: &[f32], state: &[f32]) -> Result<(), String> {
        let (expected, next) = self.expected();
        compare(&format!("{self:?} output"), output, &expected)?;
        compare(&format!("{self:?} state"), state, &next)
    }
}
