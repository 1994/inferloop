//! Regression test for the vision attention kernel.
//!
//! The kernel streams keys in 32-token blocks and keeps its softmax state across blocks. A
//! single-block sequence cannot fail there, so the fixtures below deliberately span two and four
//! blocks, one of them partially masked. The device result is compared against an independent
//! host implementation in F32. Both streaming state and operand precision are checked: the
//! multi-layer tower amplifies errors from simply rounding every MMA input to BF16.
use super::program::{AttentionMode, attention_kernel, attention_tiles, compact_attention};
use crate::device::CudaDevice;
use infer_core::{Error, Result};

/// Heads in the fixture: enough to prove the column-block arithmetic, small enough to stay fast.
const HEADS: usize = 2;
/// Padded half-head width; the real head is twice this.
const HALF: usize = 32;
/// Padded head width.
const WIDTH: usize = HALF * 2;
/// Relative tolerance between the device result and the host oracle.
const TOLERANCE: f32 = 1e-4;

/// Deterministic pseudo-random fixture values, so the test needs no assets.
fn pseudo_random(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = f32::from(u16::try_from(state >> 48).unwrap_or(0)) / 65_536.0;
            2.0f32.mul_add(unit, -1.0)
        })
        .collect()
}

/// `RoPE` tables in the layout the kernel reads: both halves carry the same cosine and sine.
fn rope_tables(tokens: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cosines = vec![1.0f32; tokens * WIDTH];
    let mut sines = vec![0.0f32; tokens * WIDTH];
    for token in 0..tokens {
        for lane in 0..HALF {
            #[expect(
                clippy::cast_precision_loss,
                reason = "Fixture indices are tiny exact integers"
            )]
            let angle = (token as f32 + 1.0) * 0.017 * (lane as f32 + 1.0);
            for half in 0..2 {
                cosines[token * WIDTH + half * HALF + lane] = angle.cos();
                sines[token * WIDTH + half * HALF + lane] = angle.sin();
            }
        }
    }
    (cosines, sines)
}

/// Index of one lane in the padded `[tokens, heads, 2, half]` layout.
fn slot(token: usize, head: usize, half: usize, lane: usize) -> usize {
    ((token * HEADS + head) * WIDTH) + half * HALF + lane
}

/// Host oracle: `RoPE` per half-head, full-head scores, non-causal softmax, weighted values.
///
/// No device tiling, operand decomposition or online rescaling is reproduced here.
fn oracle(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    cosines: &[f32],
    sines: &[f32],
    tokens: usize,
    frame_tokens: usize,
) -> Result<Vec<f32>> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "Fixture widths are tiny exact integers"
    )]
    let scale = 1.0 / (WIDTH as f32).sqrt();
    let mut output = vec![0.0f32; tokens * HEADS * WIDTH];
    for head in 0..HEADS {
        let rotated = |source: &[f32], token: usize, half: usize, lane: usize| -> (f32, f32) {
            let lo = source[slot(token, head, 0, lane)];
            let hi = source[slot(token, head, 1, lane)];
            let cosine = cosines[token * WIDTH + half * HALF + lane];
            let sine = sines[token * WIDTH + half * HALF + lane];
            (
                lo.mul_add(cosine, -(hi * sine)),
                hi.mul_add(cosine, lo * sine),
            )
        };
        for token in 0..tokens {
            let mut scores = vec![0.0f32; tokens];
            for (other, score) in scores.iter_mut().enumerate() {
                if token / frame_tokens != other / frame_tokens {
                    *score = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0f32;
                for lane in 0..HALF {
                    let (q_lo, q_hi) = rotated(query, token, 0, lane);
                    let (k_lo, k_hi) = rotated(key, other, 0, lane);
                    dot = q_lo.mul_add(k_lo, q_hi.mul_add(k_hi, dot));
                }
                *score = dot * scale;
            }
            let maximum = scores.iter().copied().fold(f32::MIN, f32::max);
            let probabilities: Vec<f32> =
                scores.iter().map(|score| (score - maximum).exp()).collect();
            let total: f32 = probabilities.iter().sum();
            if total <= 0.0 {
                return Err(Error::invariant("oracle softmax collapsed"));
            }
            for half in 0..2 {
                for lane in 0..HALF {
                    let mut numerator = 0.0f32;
                    for (other, weight) in probabilities.iter().enumerate() {
                        numerator = weight.mul_add(value[slot(other, head, half, lane)], numerator);
                    }
                    output[slot(token, head, half, lane)] = numerator / total;
                }
            }
        }
    }
    Ok(output)
}

/// Maximum difference relative to the oracle's own scale.
fn deviation(actual: &[f32], expected: &[f32]) -> Result<f32> {
    if actual.len() != expected.len() {
        return Err(Error::invalid("attention outputs differ in shape"));
    }
    let mut worst = 0f32;
    for (left, right) in actual.iter().zip(expected.iter()) {
        if !left.is_finite() || !right.is_finite() {
            return Err(Error::invariant("attention contains nonfinite values"));
        }
        worst = worst.max((left - right).abs());
    }
    let scale = expected
        .iter()
        .fold(0f32, |scale, value| scale.max(value.abs()));
    Ok(if scale > 0.0 { worst / scale } else { worst })
}

#[test]
fn attention_comparison_rejects_nonfinite_and_wrong_shapes() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(deviation(&[invalid], &[0.0]).is_err());
        assert!(deviation(&[0.0], &[invalid]).is_err());
    }
    assert!(deviation(&[0.0], &[]).is_err());
}

#[test]
fn attention_comparison_handles_zero_reference_without_division() -> Result<()> {
    assert_eq!(deviation(&[0.0, 0.0], &[0.0, 0.0])?, 0.0);
    assert_eq!(deviation(&[0.25, 0.0], &[0.0, 0.0])?, 0.25);
    Ok(())
}

/// Every streaming form must agree with the host oracle at one, two and four key blocks.
#[test]
#[ignore = "requires an NVIDIA GPU; run under tools/bench/safe-run.sh"]
fn attention_matches_the_host_oracle_across_key_blocks() -> Result<()> {
    let device = CudaDevice::new(0)?;
    // 100 tokens span four blocks, the last of them partially masked; 32 and 64 are exact.
    for tokens in [32usize, 33, 64, 65, 100] {
        let query = pseudo_random(tokens * HEADS * WIDTH, 0x51ed_2701 + tokens as u64);
        let key = pseudo_random(tokens * HEADS * WIDTH, 0x9e37_79b9 + tokens as u64);
        let value = pseudo_random(tokens * HEADS * WIDTH, 0x85eb_ca6b + tokens as u64);
        let (cosines, sines) = rope_tables(tokens);
        for frame_tokens in [tokens, 16] {
            let expected = oracle(&query, &key, &value, &cosines, &sines, tokens, frame_tokens)?;
            for tiles in [[32, 32, 0], [64, 64, 0]] {
                for mode in [AttentionMode::Online, AttentionMode::Exact] {
                    let actual = attention_kernel(
                        &device,
                        &query,
                        &key,
                        &value,
                        &cosines,
                        &sines,
                        tokens,
                        HEADS,
                        HALF,
                        mode,
                        frame_tokens,
                        WIDTH,
                        tiles,
                        false,
                    )?;
                    let error = deviation(&actual, &expected)?;
                    assert!(
                        error <= TOLERANCE,
                        "{mode:?} deviates by {error} at {tokens} tokens"
                    );
                }
            }
        }
    }
    Ok(())
}

/// Cancellation exposes input bits that would disappear in a single BF16 conversion.
#[test]
#[ignore = "requires an NVIDIA GPU; run under tools/bench/safe-run.sh"]
fn projection_preserves_sub_bf16_residuals() -> Result<()> {
    use cutile::half::bf16;
    let device = CudaDevice::new(0)?;
    let rows = 33;
    let weights: Vec<bf16> = (0..WIDTH * WIDTH)
        .map(|index| bf16::from_f32(if index % 2 == 0 { 1.0 } else { -1.0 }))
        .collect();
    let weight = device.upload(weights, &[WIDTH, WIDTH])?;
    let bias = device.upload(vec![bf16::ZERO; WIDTH], &[WIDTH])?;
    let input: Vec<f32> = (0..rows * WIDTH)
        .map(|index| {
            if index % 2 == 0 {
                1.0 + 1.0 / 1024.0
            } else {
                1.0
            }
        })
        .collect();
    let actual = super::program::project_with(&device, &weight, &bias, &input, rows, WIDTH)?;
    // 32 pairs, each contributing exactly 1/1024. Naive BF16 would return zero.
    assert_eq!(actual.len(), rows * WIDTH);
    for value in actual {
        assert!(
            (value - 0.03125).abs() < 1e-6,
            "projection lost residual: {value}"
        );
    }
    Ok(())
}

/// Legal power-of-two indexed `RoPE` over compact physical half48/head96, with old128 and 2B
/// controls. This is an F32 host/device comparison, not an attention-stage claim.
#[test]
#[ignore = "requires an NVIDIA GPU; run under tools/bench/safe-run.sh"]
#[expect(
    clippy::cast_precision_loss,
    reason = "Fixture indices are small exact integers"
)]
fn indexed_rope_compact_half48_matches_host_reference_on_sm120() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let heads = 4usize;
    let tokens = 37usize;
    for (half, real) in [(48usize, 36usize), (64, 36), (64, 64), (32, 32)] {
        let width = half * 2;
        let projections = heads * width;
        let mut query = vec![0.0f32; tokens * projections];
        for (index, value) in query.iter_mut().enumerate() {
            *value = ((index * 37 % 101) as f32 - 50.0) / 17.0;
        }
        let mut cos = vec![0.0f32; tokens * width];
        let mut sin = vec![0.0f32; tokens * width];
        for token in 0..tokens {
            for lane in 0..half {
                let angle = (token as f32 + 1.0) * 0.013 * (lane as f32 + 1.0);
                for h in 0..2 {
                    cos[token * width + h * half + lane] = angle.cos();
                    sin[token * width + h * half + lane] = angle.sin();
                }
            }
        }
        let actual =
            super::program::rope_compact_kernel(&device, &query, &cos, &sin, tokens, half, real)?;
        let mut expected = vec![0.0f32; query.len()];
        for token in 0..tokens {
            for head in 0..heads {
                for lane in 0..real {
                    let lo = query[(token * heads + head) * width + lane];
                    let hi = query[(token * heads + head) * width + half + lane];
                    let c = cos[token * width + lane];
                    let s = sin[token * width + lane];
                    expected[(token * heads + head) * width + lane] = lo.mul_add(c, -(hi * s));
                    expected[(token * heads + head) * width + half + lane] = hi.mul_add(c, lo * s);
                }
            }
        }
        let error = actual
            .iter()
            .zip(&expected)
            .fold(0.0f32, |error, (left, right)| {
                error.max((left - right).abs())
            });
        assert_eq!(actual.len(), expected.len());
        assert!(
            error <= 1e-5,
            "half={half} real={real} tokens={tokens}: max abs error {error}"
        );
    }
    Ok(())
}

/// Physical compact head layout helpers for the attention stage.
const COMPACT_WIDTH: usize = 96;
const WIDE_WIDTH: usize = 128;
const COMPACT_HALF: usize = 48;
const WIDE_HALF: usize = 64;
const REAL_HALF: usize = 36;
const REAL_HEAD: usize = 72;

fn compact_to_wide(values: &[f32], tokens: usize, heads: usize, per_head: bool) -> Vec<f32> {
    let width = if per_head {
        heads * WIDE_WIDTH
    } else {
        WIDE_WIDTH
    };
    let compact_head = if per_head {
        heads * COMPACT_WIDTH
    } else {
        COMPACT_WIDTH
    };
    let source_head = if per_head {
        heads * COMPACT_WIDTH
    } else {
        COMPACT_WIDTH
    };
    let _ = compact_head;
    let mut wide = vec![0.0f32; tokens * width];
    for token in 0..tokens {
        let source_base = token * source_head;
        let target_base = token * width;
        let heads_to_copy = if per_head { heads } else { 1 };
        for head in 0..heads_to_copy {
            for half in 0..2 {
                for lane in 0..REAL_HALF {
                    let src = source_base + head * COMPACT_WIDTH + half * COMPACT_HALF + lane;
                    let dst = target_base + head * WIDE_WIDTH + half * WIDE_HALF + lane;
                    wide[dst] = values[src];
                }
            }
        }
    }
    wide
}

fn wide_to_compact(values: &[f32], tokens: usize, heads: usize) -> Vec<f32> {
    let mut compact = vec![0.0f32; tokens * heads * COMPACT_WIDTH];
    for token in 0..tokens {
        for head in 0..heads {
            for half in 0..2 {
                for lane in 0..REAL_HALF {
                    let src = (token * heads + head) * WIDE_WIDTH + half * WIDE_HALF + lane;
                    let dst = (token * heads + head) * COMPACT_WIDTH + half * COMPACT_HALF + lane;
                    compact[dst] = values[src];
                }
            }
        }
    }
    compact
}

#[expect(
    clippy::too_many_arguments,
    reason = "The host oracle receives the full compact fixture explicitly"
)]
fn compact_oracle(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    cos: &[f32],
    sin: &[f32],
    tokens: usize,
    heads: usize,
    frame_tokens: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; tokens * heads * COMPACT_WIDTH];
    let scale = 1.0 / 72.0f32.sqrt();
    for head in 0..heads {
        let rotated = |source: &[f32], token: usize, half: usize, lane: usize| {
            let base = (token * heads + head) * COMPACT_WIDTH + half * COMPACT_HALF + lane;
            let lo = source[(token * heads + head) * COMPACT_WIDTH + lane];
            let hi = source[(token * heads + head) * COMPACT_WIDTH + COMPACT_HALF + lane];
            let c = cos[token * COMPACT_WIDTH + lane];
            let s = sin[token * COMPACT_WIDTH + lane];
            let _ = base;
            if half == 0 {
                lo.mul_add(c, -(hi * s))
            } else {
                hi.mul_add(c, lo * s)
            }
        };
        for token in 0..tokens {
            let mut scores = vec![f32::NEG_INFINITY; tokens];
            let mut maximum = f32::NEG_INFINITY;
            for (other, score) in scores.iter_mut().enumerate() {
                if token / frame_tokens != other / frame_tokens {
                    continue;
                }
                let mut dot = 0.0f32;
                for lane in 0..REAL_HALF {
                    dot = rotated(query, token, 0, lane).mul_add(rotated(key, other, 0, lane), dot);
                    dot = rotated(query, token, 1, lane).mul_add(rotated(key, other, 1, lane), dot);
                }
                *score = dot * scale;
                maximum = maximum.max(*score);
            }
            let mut denominator = 0.0f32;
            for score in &mut scores {
                if score.is_finite() {
                    *score = (*score - maximum).exp();
                    denominator += *score;
                } else {
                    *score = 0.0;
                }
            }
            for (other, score) in scores.iter().enumerate() {
                if *score == 0.0 {
                    continue;
                }
                let weight = *score / denominator;
                for half in 0..2 {
                    for lane in 0..REAL_HALF {
                        let src =
                            (other * heads + head) * COMPACT_WIDTH + half * COMPACT_HALF + lane;
                        let dst =
                            (token * heads + head) * COMPACT_WIDTH + half * COMPACT_HALF + lane;
                        output[dst] = value[src].mul_add(weight, output[dst]);
                    }
                }
            }
        }
    }
    output
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Fixture indices are small exact integers"
)]
fn compact_attention_case(device: &CudaDevice, tokens: usize, frame_tokens: usize) -> Result<()> {
    let heads = 4usize;
    let compact_len = tokens * heads * COMPACT_WIDTH;
    let mut q = vec![0.0f32; compact_len];
    let mut k = vec![0.0f32; compact_len];
    let mut v = vec![0.0f32; compact_len];
    for i in 0..compact_len {
        q[i] = ((i * 29 % 97) as f32 - 48.0) / 23.0;
        k[i] = ((i * 31 % 89) as f32 - 44.0) / 19.0;
        v[i] = ((i * 37 % 83) as f32 - 41.0) / 17.0;
    }
    let mut cos = vec![0.0f32; tokens * COMPACT_WIDTH];
    let mut sin = vec![0.0f32; tokens * COMPACT_WIDTH];
    for token in 0..tokens {
        for lane in 0..COMPACT_HALF {
            let angle = (token as f32 + 1.0) * 0.011 * (lane as f32 + 1.0);
            for half in 0..2 {
                cos[token * COMPACT_WIDTH + half * COMPACT_HALF + lane] = angle.cos();
                sin[token * COMPACT_WIDTH + half * COMPACT_HALF + lane] = angle.sin();
            }
        }
    }
    let tiles = attention_tiles(tokens);
    let compact = compact_attention(
        device,
        (&q, &k, &v),
        &cos,
        &sin,
        tokens,
        heads,
        frame_tokens,
        tiles,
    )?;
    let wide_q = compact_to_wide(&q, tokens, heads, true);
    let wide_k = compact_to_wide(&k, tokens, heads, true);
    let wide_v = compact_to_wide(&v, tokens, heads, true);
    let wide_cos = compact_to_wide(&cos, tokens, heads, false);
    let wide_sin = compact_to_wide(&sin, tokens, heads, false);
    let old = attention_kernel(
        device,
        &wide_q,
        &wide_k,
        &wide_v,
        &wide_cos,
        &wide_sin,
        tokens,
        heads,
        WIDE_HALF,
        AttentionMode::Online,
        frame_tokens,
        REAL_HEAD,
        tiles,
        false,
    )?;
    let old_compact = wide_to_compact(&old, tokens, heads);
    let expected = compact_oracle(&q, &k, &v, &cos, &sin, tokens, heads, frame_tokens);
    assert_eq!(compact.len(), expected.len());
    assert_eq!(old_compact.len(), expected.len());
    assert!(compact.iter().all(|value| value.is_finite()));
    assert!(old_compact.iter().all(|value| value.is_finite()));
    assert!(expected.iter().all(|value| value.is_finite()));
    let scale = expected
        .iter()
        .fold(0.0f32, |scale, value| scale.max(value.abs()))
        .max(1.0);
    let compact_error = compact
        .iter()
        .zip(&expected)
        .fold(0.0f32, |error, (left, right)| {
            error.max((left - right).abs())
        });
    let old_error = old_compact
        .iter()
        .zip(&expected)
        .fold(0.0f32, |error, (left, right)| {
            error.max((left - right).abs())
        });
    assert!(
        compact_error / scale <= TOLERANCE,
        "tokens={tokens} compact: relative error {}",
        compact_error / scale
    );
    assert!(
        old_error / scale <= TOLERANCE,
        "tokens={tokens} old128: relative error {}",
        old_error / scale
    );
    Ok(())
}

#[test]
#[ignore = "requires an NVIDIA GPU; run under tools/bench/safe-run.sh"]
fn compact_attention_head72_matches_host_and_old128_on_sm120() -> Result<()> {
    let device = CudaDevice::new(0)?;
    for (tokens, frame_tokens) in [(37usize, 8usize), (65, 16)] {
        compact_attention_case(&device, tokens, frame_tokens)?;
    }
    Ok(())
}
