//! `Q4_K` int8 two-term activation quantization: main int8 plus an int8 residual.
use infer_models::{GgufPackage, Q4KTiled};
use std::env::args;
use std::error::Error;

const ROWS: usize = 16;
const OUTPUTS: usize = 16;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = args().skip(1);
    let path = arguments
        .next()
        .ok_or("usage: cuda-q4k-two-term <gguf> <slot>")?;
    let slot_name = arguments
        .next()
        .ok_or("usage: cuda-q4k-two-term <gguf> <slot>")?;
    let package = GgufPackage::open(path)?;
    let slot = package
        .slots()
        .get(&slot_name)
        .ok_or("missing slot")?
        .clone();
    let columns = slot.shape[1];
    let layout = Q4KTiled::from_device(&package.q4k(&slot_name, u64::MAX)?)?;
    let values = package.dense(&slot_name, u64::MAX)?;
    let x: Vec<f32> = (0..ROWS * columns)
        .map(|index| {
            let main = f32::from(i8::try_from(index % 19).unwrap_or(0) - 9);
            let fine = f32::from(i8::try_from(index % 7).unwrap_or(0) - 3);
            fine.mul_add(0.001, main)
        })
        .collect();
    let groups = columns / 32;
    let mut one_term = vec![0.0f32; ROWS * OUTPUTS];
    let mut two_term = vec![0.0f32; ROWS * OUTPUTS];
    let mut reference = vec![0.0f32; ROWS * OUTPUTS];
    for row in 0..ROWS {
        for group in 0..groups {
            let start = row * columns + group * 32;
            let slice = &x[start..start + 32];
            let scale = 1.0f32;
            let scale2 = 0.001f32;
            let mut q1 = [0i8; 32];
            let mut q2 = [0i8; 32];
            for index in 0..32 {
                q1[index] = i8::try_from((start + index) % 19).unwrap_or(0) - 9;
                q2[index] = i8::try_from((start + index) % 7).unwrap_or(0) - 3;
            }
            let sum: f32 = slice.iter().sum();
            for output in 0..OUTPUTS {
                let factor = layout.factors()[group * OUTPUTS + output];
                let bias = layout.biases()[group * OUTPUTS + output];
                let mut main = 0i32;
                let mut fine = 0i32;
                for value in 0..32 {
                    let packed = layout.codes()[(group / 2) * 512 + (value % 32) * 16 + output];
                    let code = if group.is_multiple_of(2) {
                        packed & 0x0F
                    } else {
                        packed >> 4
                    };
                    let code = i32::from(code);
                    main += i32::from(q1[value]) * code;
                    fine += i32::from(q2[value]) * code;
                }
                let main = f32::from(i16::try_from(main).unwrap_or(0));
                let fine = f32::from(i16::try_from(fine).unwrap_or(0));
                let one = f32::mul_add(bias, -sum, factor * main * scale);
                let two = f32::mul_add(bias, -sum, factor * fine.mul_add(scale2, main * scale));
                one_term[row * OUTPUTS + output] += one;
                two_term[row * OUTPUTS + output] += two;
                reference[row * OUTPUTS + output] += values[output * columns + group * 32..][..32]
                    .iter()
                    .zip(slice)
                    .map(|(weight, activation)| weight * activation)
                    .sum::<f32>();
            }
        }
    }
    let mut one_error = 0.0f32;
    let mut two_error = 0.0f32;
    for index in 0..reference.len() {
        one_error = one_error.max((one_term[index] - reference[index]).abs());
        two_error = two_error.max((two_term[index] - reference[index]).abs());
    }
    println!("one-term max-abs={one_error:.3e} two-term max-abs={two_error:.3e}");
    Ok(())
}
