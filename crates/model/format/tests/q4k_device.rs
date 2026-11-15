use infer_core::ErrorCode;
use infer_format::{GgufType, Q4KDevice, dequantize};

const ONE: [u8; 2] = [0x00, 0x3c];
const HALF: [u8; 2] = [0x00, 0x38];
const QUARTER: [u8; 2] = [0x00, 0x34];

fn block(d: [u8; 2], dmin: [u8; 2], scales: [u8; 12], qs: u8) -> Vec<u8> {
    let mut block = vec![0u8; 144];
    block[..2].copy_from_slice(&d);
    block[2..4].copy_from_slice(&dmin);
    block[4..16].copy_from_slice(&scales);
    block[16..].fill(qs);
    block
}

fn unit_scales() -> [u8; 12] {
    let mut scales = [0u8; 12];
    scales[..4].fill(1);
    scales[8..].fill(1);
    scales
}

fn unit_scales_with_min() -> [u8; 12] {
    let mut scales = [0u8; 12];
    scales[..8].fill(1);
    scales[8..].fill(0x11);
    scales
}

#[test]
fn repacks_codes_scales_and_super_block_factors() {
    let payload = block(ONE, [0, 0], unit_scales(), 0x21);
    let device = Q4KDevice::from_blocks(1, 256, &payload).unwrap();
    assert_eq!(device.rows(), 1);
    assert_eq!(device.columns(), 256);
    assert_eq!(device.codes().len(), 128);
    assert_eq!(device.codes()[0], 0x21);
    assert_eq!(device.scales(), &[1u8; 8]);
    assert_eq!(device.mins(), &[0u8; 8]);
    assert_eq!(device.d(), &[1.0]);
    assert_eq!(device.dmin(), &[0.0]);
}

#[test]
fn host_gemv_matches_the_grouped_formula() {
    let payload = block(ONE, [0, 0], unit_scales(), 0x21);
    let device = Q4KDevice::from_blocks(1, 256, &payload).unwrap();
    let x = vec![1.0f32; 256];
    // Every 64-group contributes 32 * 1 + 32 * 2.
    assert_eq!(device.row_dot(0, &x).unwrap(), 384.0);

    let values = {
        let mut values = [0.0f32; 256];
        dequantize(GgufType::Q4K, &payload, &mut values).unwrap();
        values
    };
    let expected: f32 = values.iter().map(|value| value * 1.0).sum();
    assert!((device.row_dot(0, &x).unwrap() - expected).abs() < 1e-4);
}

#[test]
fn minimum_term_is_applied_per_group() {
    let payload = block(ONE, ONE, unit_scales_with_min(), 0x21);
    let device = Q4KDevice::from_blocks(1, 256, &payload).unwrap();
    assert_eq!(device.mins(), &[1u8; 8]);
    // codes 1 and 2 minus the unit minimum: 0 and 1 per half-group.
    assert_eq!(device.row_dot(0, &vec![1.0; 256]).unwrap(), 128.0);
}

#[test]
fn multiple_rows_and_blocks_keep_their_offsets() {
    let first = block(ONE, [0, 0], unit_scales(), 0x21);
    let second = block(HALF, QUARTER, unit_scales_with_min(), 0x43);
    let mut payload = first.clone();
    payload.extend_from_slice(&first);
    payload.extend_from_slice(&second);
    payload.extend_from_slice(&second);
    let device = Q4KDevice::from_blocks(2, 512, &payload).unwrap();
    assert_eq!(device.d(), &[1.0, 0.5, 1.0, 0.5]);
    assert_eq!(device.dmin(), &[0.0, 0.25, 0.0, 0.25]);
    let x = vec![1.0f32; 512];
    assert_eq!(device.row_dot(0, &x).unwrap(), 768.0);
    // Second row: d=0.5, dmin=0.25, scales 1, mins 1, codes 3 and 4.
    // Per 64-group: 32 * (1.5 - 0.25) + 32 * (2.0 - 0.25); one block is 384, two are 768.
    assert!((device.row_dot(1, &x).unwrap() - 768.0).abs() < 1e-3);
}

#[test]
fn rejects_malformed_shapes_and_payloads() {
    let payload = block(ONE, [0, 0], unit_scales(), 0x21);
    assert_eq!(
        Q4KDevice::from_blocks(1, 128, &payload).unwrap_err().code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        Q4KDevice::from_blocks(1, 256, &payload[..143])
            .unwrap_err()
            .code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        Q4KDevice::from_blocks(1, 256, &payload)
            .unwrap()
            .row_dot(1, &vec![0.0; 256])
            .unwrap_err()
            .code,
        ErrorCode::InvalidInput
    );
}
