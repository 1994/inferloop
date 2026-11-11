//! Acceptance of the rotary cell, shared by every kernel source.
//!
//! A source records [`Case`] through the `Rope` contract and hands the output to
//! [`Case::check`]. The angle is the F32 product of frequency and position, as the contract
//! defines it; the reference rotates by that angle in F64.

/// Absolute tolerance: rotated values are bounded by the input magnitude, below two here.
const ABSOLUTE_TOLERANCE: f64 = 2.0e-5;
/// Residues of the deterministic inputs.
const MODULUS: usize = 19;
/// Inputs are sixteenths centred on zero, exact in F32.
const STEPS: f32 = 16.0;
/// The residue that maps to zero.
const CENTRE: f32 = 9.0;
/// Ratio between consecutive frequencies, as a long-context base gives them.
const FREQUENCY_DECAY: f32 = 0.75;

/// Where each row's position comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Positions {
    /// One sequence at `metadata[0]`.
    Single,
    /// A prompt: lane `row / heads` sits at `metadata[0] + lane`.
    Prompt,
    /// Pooled lanes: lane `row / heads` sits at `metadata[lane * fields]`.
    Pooled { fields: usize },
}

/// One rotary launch over `rows` rows of `head_dim`, rotating the first `2 * half` values.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub positions: Positions,
    pub rows: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub half: usize,
    /// The first lane's position.
    pub base: i32,
}

impl Case {
    fn lanes(self) -> usize {
        self.rows.div_ceil(self.heads)
    }

    #[must_use]
    pub fn input(self) -> Vec<f32> {
        (0..self.rows * self.head_dim)
            .map(|index| {
                let step = u16::try_from((index * 7 + 2) % MODULUS).unwrap_or(0);
                (f32::from(step) - CENTRE) / STEPS
            })
            .collect()
    }

    #[must_use]
    pub fn frequencies(self) -> Vec<f32> {
        (0..self.half)
            .scan(1.0f32, |frequency, _| {
                let current = *frequency;
                *frequency *= FREQUENCY_DECAY;
                Some(current)
            })
            .collect()
    }

    /// Lane `lane`'s position; pooled lanes run backwards from the base so lanes differ.
    fn position(self, lane: usize) -> i32 {
        let lane = i32::try_from(lane).unwrap_or(i32::MAX);
        match self.positions {
            Positions::Single => self.base,
            Positions::Prompt => self.base + lane,
            Positions::Pooled { .. } => self.base - lane,
        }
    }

    /// The metadata the launch reads positions from.
    #[must_use]
    pub fn metadata(self) -> Vec<i32> {
        match self.positions {
            Positions::Single | Positions::Prompt => vec![self.base],
            Positions::Pooled { fields } => {
                let mut metadata = vec![0; self.lanes() * fields];
                for lane in 0..self.lanes() {
                    metadata[lane * fields] = self.position(lane);
                }
                metadata
            }
        }
    }

    /// The independent reference.
    #[must_use]
    pub fn expected(self) -> Vec<f64> {
        let (input, frequencies) = (self.input(), self.frequencies());
        let mut out: Vec<f64> = input.iter().copied().map(f64::from).collect();
        for row in 0..self.rows {
            let lane = match self.positions {
                Positions::Single => 0,
                Positions::Prompt | Positions::Pooled { .. } => row / self.heads,
            };
            // Cases keep positions within `i16`, which F32 holds exactly.
            let position = f32::from(i16::try_from(self.position(lane)).unwrap_or(i16::MAX));
            let start = row * self.head_dim;
            for (index, frequency) in frequencies.iter().enumerate() {
                let angle = f64::from(frequency * position);
                let (first, second) = (start + index, start + self.half + index);
                let (x, y) = (f64::from(input[first]), f64::from(input[second]));
                let (cos, sin) = (angle.cos(), angle.sin());
                out[first] = x.mul_add(cos, -y * sin);
                out[second] = y.mul_add(cos, x * sin);
            }
        }
        out
    }

    /// # Errors
    /// Names the first element that is not finite or lies outside the tolerance, or a wrong
    /// output length.
    pub fn check(self, actual: &[f32]) -> Result<(), String> {
        let expected = self.expected();
        if actual.len() != expected.len() {
            return Err(format!("{self:?}: {} outputs", actual.len()));
        }
        for (index, (value, reference)) in actual.iter().zip(&expected).enumerate() {
            if !value.is_finite() || (f64::from(*value) - reference).abs() > ABSOLUTE_TOLERANCE {
                return Err(format!("{self:?} [{index}]: {value} vs {reference}"));
            }
        }
        Ok(())
    }
}
