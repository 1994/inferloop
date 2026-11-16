use infer_core::ErrorCode;
use infer_format::{GgufType, dequantize};

const ONE_HALF: [u8; 2] = [0x00, 0x3c];
fn q4_k_block() -> Vec<u8> {
    let mut block = vec![0u8; 144];
    block[..2].copy_from_slice(&ONE_HALF);
    block[4..8].fill(1);
    block[12..16].fill(1);
    block[16..].fill(0x21);
    block
}

fn q5_k_block() -> Vec<u8> {
    let mut block = vec![0u8; 176];
    block[..2].copy_from_slice(&ONE_HALF);
    block[4..8].fill(1);
    block[12..16].fill(1);
    block[16..48].fill(0xff);
    block
}

fn q6_k_block() -> Vec<u8> {
    let mut block = vec![0u8; 210];
    block[192..208].fill(1);
    block[208..].copy_from_slice(&ONE_HALF);
    block
}

#[test]
fn scalar_types_keep_their_bits() {
    let mut out = [0.0f32; 3];
    dequantize(
        GgufType::F32,
        &[
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x80, 0x3e,
        ],
        &mut out,
    )
    .unwrap();
    assert_eq!(out, [1.0, -2.0, 0.25]);

    dequantize(
        GgufType::F16,
        &[0x00, 0x3c, 0x00, 0xc0, 0x01, 0x00],
        &mut out,
    )
    .unwrap();
    assert_eq!(out, [1.0, -2.0, 2.0f32.powi(-24)]);

    dequantize(
        GgufType::Bf16,
        &[0x80, 0x3f, 0x00, 0xc0, 0x00, 0x00],
        &mut out,
    )
    .unwrap();
    assert_eq!(out, [1.0, -2.0, 0.0]);
}

#[test]
fn legacy_q4_0_puts_low_nibbles_first() {
    // GGML stores value j in the low nibble of byte j and value j + 16 in its high nibble.
    let mut out = [0.0f32; 32];
    let mut block = vec![0u8; 18];
    block[..2].copy_from_slice(&ONE_HALF);
    block[2] = 0x21;
    block[3] = 0xf0;
    dequantize(GgufType::Q4_0, &block, &mut out).unwrap();
    assert_eq!([out[0], out[16], out[1], out[17]], [-7.0, -6.0, -8.0, 7.0]);
    assert!(
        out.iter()
            .enumerate()
            .filter(|(index, _)| ![0, 1, 16, 17].contains(index))
            .all(|(_, value)| value.to_bits() == (-8.0f32).to_bits())
    );
}

#[test]
fn legacy_q4_1_adds_the_block_minimum() {
    let mut out = [0.0f32; 32];
    let mut block = vec![0u8; 20];
    block[..2].copy_from_slice(&ONE_HALF);
    // m = -2.0
    block[2..4].copy_from_slice(&[0x00, 0xc0]);
    block[4] = 0x31;
    dequantize(GgufType::Q4_1, &block, &mut out).unwrap();
    assert_eq!(out[0], -1.0);
    assert_eq!(out[16], 1.0);
    assert_eq!(out[1], -2.0);
}

#[test]
fn legacy_q5_0_takes_the_fifth_bit_from_the_high_word() {
    let mut out = [0.0f32; 32];
    let mut block = vec![0u8; 22];
    block[..2].copy_from_slice(&ONE_HALF);
    // Bit 0 sets value 0's fifth bit; bit 16 sets value 16's.
    block[2..6].copy_from_slice(&0x0001_0001u32.to_le_bytes());
    block[6] = 0x21;
    dequantize(GgufType::Q5_0, &block, &mut out).unwrap();
    assert_eq!(out[0], 1.0);
    assert_eq!(out[16], 2.0);
    assert_eq!(out[1], -16.0);
    assert_eq!(out[17], -16.0);
}

#[test]
fn legacy_q5_1_combines_fifth_bit_and_minimum() {
    let mut out = [0.0f32; 32];
    let mut block = vec![0u8; 24];
    block[..2].copy_from_slice(&ONE_HALF);
    block[2..4].copy_from_slice(&ONE_HALF);
    block[4..8].copy_from_slice(&0x8000_0000u32.to_le_bytes());
    block[8 + 15] = 0xf0;
    dequantize(GgufType::Q5_1, &block, &mut out).unwrap();
    assert_eq!(out[31], 32.0);
    assert_eq!(out[15], 1.0);
}

#[test]
fn legacy_q8_0_keeps_signed_codes() {
    let mut out = [0.0f32; 64];
    let mut blocks = vec![0u8; 68];
    blocks[..2].copy_from_slice(&ONE_HALF);
    blocks[2] = 0x80;
    blocks[3] = 0x7f;
    blocks[4] = 0xff;
    // Second block: d = 2.0.
    blocks[34..36].copy_from_slice(&[0x00, 0x40]);
    blocks[36] = 3;
    dequantize(GgufType::Q8_0, &blocks, &mut out).unwrap();
    assert_eq!(out[..4], [-128.0, 127.0, -1.0, 0.0]);
    assert_eq!(out[32], 6.0);
}

#[test]
fn q4_k_applies_group_scales_and_mins() {
    let mut out = [0.0f32; 256];
    dequantize(GgufType::Q4K, &q4_k_block(), &mut out).unwrap();
    // d=1, dmin=0, every 6-bit scale is 1; quants are low=1 and high=2.
    assert_eq!(out[0], 1.0);
    assert_eq!(out[31], 1.0);
    assert_eq!(out[32], 2.0);
    assert_eq!(out[63], 2.0);
    assert_eq!(out[64], 1.0);
    assert_eq!(out[255], 2.0);
}

#[test]
fn q5_k_adds_the_high_bit_plane() {
    let mut out = [0.0f32; 256];
    dequantize(GgufType::Q5K, &q5_k_block(), &mut out).unwrap();
    // d=1, dmin=0, scale=1, quants zero, every high bit set.
    assert_eq!(out[0], 16.0);
    assert_eq!(out[32], 16.0);
    assert_eq!(out[255], 16.0);
}

#[test]
fn q6_k_applies_signed_scales_and_zero_point() {
    let mut out = [0.0f32; 256];
    let mut block = q6_k_block();
    dequantize(GgufType::Q6K, &block, &mut out).unwrap();
    assert_eq!(out[0], -32.0);
    assert_eq!(out[255], -32.0);

    block[128] = 0x03;
    dequantize(GgufType::Q6K, &block, &mut out).unwrap();
    assert_eq!(out[0], 16.0);
    assert_eq!(out[1], -32.0);
    assert_eq!(out[32], -32.0);
    assert_eq!(out[64], -32.0);
    assert_eq!(out[96], -32.0);
}

#[test]
fn geometry_and_unsupported_types_are_explicit() {
    assert_eq!(GgufType::F32.byte_len(4).unwrap(), 16);
    assert_eq!(GgufType::Q4K.byte_len(256).unwrap(), 144);
    assert_eq!(
        GgufType::Q4K.byte_len(128).unwrap_err().code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        GgufType::Other(99).byte_len(256).unwrap_err().code,
        ErrorCode::Unsupported
    );
    let mut out = [0.0f32; 256];
    assert_eq!(
        dequantize(GgufType::Q2K, &[0u8; 84], &mut out)
            .unwrap_err()
            .code,
        ErrorCode::Unsupported
    );
}
