//! Acceptance of the `pool_gemv_bf16` cell, shared by every kernel source.
//!
//! A source launches [`Case`] however it records, then hands the output to [`Case::check`]. The
//! reference is an independent scalar sum; each source has its own reduction order, so results
//! compare against it within a tolerance and never bit for bit across sources.

/// Activation rows of every case.
pub const ROWS: usize = 4;
/// Output elements of every case.
pub const OUTPUTS: usize = 16;
/// Relative tolerance: a few F32 rounding units on the summed magnitude of the element.
const RELATIVE_TOLERANCE: f32 = 1.0e-5;
/// Bits a BF16 keeps from the top of an F32.
const BF16_SHIFT: u32 = 16;

/// One reduction length; inputs and weights are eighths, exact in both F32 and BF16.
#[derive(Clone, Copy)]
pub struct Case {
    pub columns: usize,
}

fn eighths(index: usize, stride: usize, modulus: usize, offset: f32) -> f32 {
    let step = u16::try_from((index * stride) % modulus).unwrap_or(0);
    f32::from(step) / 8.0 - offset
}

impl Case {
    #[must_use]
    pub fn input(self) -> Vec<f32> {
        (0..ROWS * self.columns)
            .map(|index| eighths(index, 7, 13, 0.5))
            .collect()
    }

    #[must_use]
    pub fn weight(self) -> Vec<f32> {
        (0..OUTPUTS * self.columns)
            .map(|index| eighths(index, 11, 17, 1.0))
            .collect()
    }

    /// The weight as BF16 bit patterns; truncating these F32 values loses nothing.
    #[must_use]
    pub fn weight_bits(self) -> Vec<u16> {
        self.weight()
            .iter()
            .map(|value| u16::try_from(value.to_bits() >> BF16_SHIFT).unwrap_or(0))
            .collect()
    }

    /// # Errors
    /// Names the first element outside the tolerance, or a wrong output length.
    pub fn check(self, actual: &[f32]) -> Result<(), String> {
        if actual.len() != ROWS * OUTPUTS {
            return Err(format!(
                "{} columns: {} outputs",
                self.columns,
                actual.len()
            ));
        }
        let (input, weight) = (self.input(), self.weight());
        for (index, value) in actual.iter().enumerate() {
            let (row, element) = (index / OUTPUTS, index % OUTPUTS);
            let terms = (0..self.columns)
                .map(|k| weight[element * self.columns + k] * input[row * self.columns + k]);
            let expected: f32 = terms.clone().sum();
            let magnitude: f32 = terms.map(f32::abs).sum();
            if (value - expected).abs() > magnitude * RELATIVE_TOLERANCE {
                return Err(format!(
                    "{} columns, [{row}, {element}]: {value} vs {expected}",
                    self.columns
                ));
            }
        }
        Ok(())
    }
}
