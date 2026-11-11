//! Acceptance of the RMS norm cell, shared by every kernel source.
//!
//! A source records [`Case`] through the `Norm` contract and hands the output to
//! [`Case::check`]. The reference reduces in F64 in index order; a source reduces in its own
//! order, so results compare within a tolerance and never bit for bit across sources.

/// Relative tolerance on each element: the F32 sum of squares and `rsqrt` round a few units.
const RELATIVE_TOLERANCE: f64 = 1.0e-5;
/// Absolute floor for elements near zero.
const ABSOLUTE_TOLERANCE: f64 = 1.0e-6;
/// Residues of the deterministic inputs.
const MODULUS: usize = 23;
/// Inputs are sixteenths centred on zero, exact in F32.
const STEPS: f32 = 16.0;
/// The residue that maps to zero.
const CENTRE: f32 = 11.0;

/// One normalization: `rows` rows of `dimension` values, optionally gated by `SiLU(gate)`.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub rows: usize,
    pub dimension: usize,
    pub gated: bool,
    pub epsilon: f32,
    pub offset: f32,
}

fn sixteenths(count: usize, stride: usize, shift: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let step = u16::try_from((index * stride + shift) % MODULUS).unwrap_or(0);
            (f32::from(step) - CENTRE) / STEPS
        })
        .collect()
}

impl Case {
    #[must_use]
    pub fn input(self) -> Vec<f32> {
        sixteenths(self.rows * self.dimension, 7, 3)
    }

    #[must_use]
    pub fn weight(self) -> Vec<f32> {
        sixteenths(self.dimension, 5, 1)
    }

    #[must_use]
    pub fn gate(self) -> Vec<f32> {
        sixteenths(self.rows * self.dimension, 11, 5)
    }

    /// The independent F64 reference.
    #[must_use]
    pub fn expected(self) -> Vec<f64> {
        let (input, weight, gate) = (self.input(), self.weight(), self.gate());
        let dimension = self.dimension;
        let width = f64::from(u32::try_from(dimension).unwrap_or(u32::MAX));
        let mut out = Vec::with_capacity(input.len());
        for (index, values) in input.chunks(dimension).enumerate() {
            let square: f64 = values.iter().map(|x| f64::from(*x).powi(2)).sum();
            let inverse = 1.0 / (square / width + f64::from(self.epsilon)).sqrt();
            for (column, x) in values.iter().enumerate() {
                let scale = f64::from(weight[column]) + f64::from(self.offset);
                let mut value = f64::from(*x) * inverse * scale;
                if self.gated {
                    let z = f64::from(gate[index * dimension + column]);
                    value *= z / (1.0 + (-z).exp());
                }
                out.push(value);
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
            let error = (f64::from(*value) - reference).abs();
            if !value.is_finite()
                || error
                    > reference
                        .abs()
                        .mul_add(RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)
            {
                return Err(format!("{self:?} [{index}]: {value} vs {reference}"));
            }
        }
        Ok(())
    }
}
