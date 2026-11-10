//! Acceptance of the element-wise state cells, shared by every kernel source.
//!
//! A source records [`Case`] through the `State` contract and hands the output to
//! [`Case::check`]. Add and multiply of these inputs are exact in F32, so they compare bit for
//! bit; the activations compare against F64 within a tolerance, since `exp` differs by source.

/// Absolute tolerance on activations, whose outputs stay below a few units here.
const ABSOLUTE_TOLERANCE: f64 = 2.0e-6;
/// Relative tolerance on activations.
const RELATIVE_TOLERANCE: f64 = 2.0e-6;
/// Inputs are eighths centred on zero, exact in F32 with exact sums and products.
const STEPS: f32 = 8.0;
/// Residues of the deterministic inputs.
const MODULUS: usize = 29;
/// The residue that maps to zero.
const CENTRE: f32 = 14.0;

/// The operation a case records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Silu,
    Sigmoid,
    Add,
    Multiply,
}

/// One element-wise launch over `elements` values, a partial tile included.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub op: Op,
    pub elements: usize,
}

fn eighths(count: usize, stride: usize, shift: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let step = u16::try_from((index * stride + shift) % MODULUS).unwrap_or(0);
            (f32::from(step) - CENTRE) / STEPS
        })
        .collect()
}

impl Case {
    #[must_use]
    pub fn first(self) -> Vec<f32> {
        eighths(self.elements, 7, 3)
    }

    /// The second operand of a binary case.
    #[must_use]
    pub fn second(self) -> Vec<f32> {
        eighths(self.elements, 11, 5)
    }

    #[must_use]
    pub fn expected(self) -> Vec<f64> {
        let (first, second) = (self.first(), self.second());
        first
            .iter()
            .zip(&second)
            .map(|(&x, &y)| {
                let (x, y) = (f64::from(x), f64::from(y));
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                match self.op {
                    Op::Silu => x * sigmoid,
                    Op::Sigmoid => sigmoid,
                    Op::Add => x + y,
                    Op::Multiply => x * y,
                }
            })
            .collect()
    }

    /// # Errors
    /// Names the first element that is not finite, differs from an exact result, or lies
    /// outside the activation tolerance, or a wrong output length.
    pub fn check(self, actual: &[f32]) -> Result<(), String> {
        let expected = self.expected();
        if actual.len() != expected.len() {
            return Err(format!("{self:?}: {} outputs", actual.len()));
        }
        let exact = matches!(self.op, Op::Add | Op::Multiply);
        for (index, (value, reference)) in actual.iter().zip(&expected).enumerate() {
            let error = (f64::from(*value) - reference).abs();
            let bound = if exact {
                0.0
            } else {
                reference
                    .abs()
                    .mul_add(RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)
            };
            if !value.is_finite() || error > bound {
                return Err(format!("{self:?} [{index}]: {value} vs {reference}"));
            }
        }
        Ok(())
    }
}
