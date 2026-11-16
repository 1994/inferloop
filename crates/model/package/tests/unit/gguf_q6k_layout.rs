use super::Q6KTiled;
use infer_format::{GgufType, dequantize};

fn block() -> Vec<u8> {
    let mut block = vec![0u8; 210];
    // d = 1.0, all scales = 1, all codes = 0 (dequantizes to -32).
    block[208..210].copy_from_slice(&[0x00, 0x3c]);
    block[192..208].fill(1);
    block
}

#[test]
fn packs_codes_and_signed_factors() {
    let tiled = Q6KTiled::from_blocks(1, 256, &block()).unwrap();
    assert_eq!(tiled.rows(), 1);
    assert_eq!(tiled.columns(), 256);
    assert_eq!(tiled.codes().len(), 256 * 16);
    assert_eq!(tiled.factors().len(), 16 * 16);
    assert_eq!(tiled.codes()[0], 0);
    assert_eq!(tiled.factors()[0], 1.0);
}

#[test]
fn host_dot_matches_the_reference_decode() {
    let block = block();
    let tiled = Q6KTiled::from_blocks(1, 256, &block).unwrap();
    let x = vec![1.0f32; 256];
    let mut values = [0.0f32; 256];
    dequantize(GgufType::Q6K, &block, &mut values).unwrap();
    let expected: f32 = values.iter().sum();
    let mut actual = 0.0f32;
    for value in 0..256 {
        let code = f32::from(tiled.codes()[value * 16]);
        let factor = tiled.factors()[(value / 16) * 16];
        actual += factor * (code - 32.0) * x[value];
    }
    assert!((actual - expected).abs() < 1e-3);
}
