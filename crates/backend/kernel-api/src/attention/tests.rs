use super::*;

fn descriptor() -> AttentionDescriptor {
    AttentionDescriptor {
        queries: 3,
        keys: 7,
        query_heads: 8,
        kv_heads: 2,
        qk_dim: 72,
        value_dim: 48,
        scale: 0.125,
        dtype: DType::F32,
        mask: AttentionMask::None,
    }
}

#[test]
fn arbitrary_dimensions_and_gqa_are_semantics_not_device_policy() -> Result<()> {
    descriptor().validate()?;
    let mut d = descriptor();
    d.kv_heads = 1;
    d.dtype = DType::Bf16;
    d.validate()?;
    d.kv_heads = 3;
    assert!(d.validate().is_err());
    Ok(())
}

#[test]
fn causal_alignment_and_fully_masked_rows_are_explicit() -> Result<()> {
    let mut d = descriptor();
    d.mask = AttentionMask::Causal { query_start: 4 };
    d.validate()?;
    assert!(d.visible(0, 4));
    assert!(!d.visible(0, 5));
    assert!(d.visible(2, 6));
    d.mask = AttentionMask::Causal { query_start: -2 };
    d.validate()?;
    assert!(!(0..d.keys).any(|k| d.visible(0, k)));
    assert!(d.visible(2, 0));
    Ok(())
}

#[test]
fn window_and_segments_do_not_cross_boundaries() -> Result<()> {
    let mut d = descriptor();
    d.mask = AttentionMask::Window {
        query_start: 4,
        left: 1,
        right: 0,
    };
    d.validate()?;
    assert!(!d.visible(0, 2));
    assert!(d.visible(0, 3));
    assert!(d.visible(0, 4));
    assert!(!d.visible(0, 5));
    d.queries = d.keys;
    d.mask = AttentionMask::Segments { tokens: 3 };
    d.validate()?;
    assert!(!d.visible(2, 3));
    assert!(d.visible(6, 6));
    assert!(!d.visible(6, 7));
    Ok(())
}

#[test]
fn malformed_descriptors_fail_before_device_submission() {
    let mut d = descriptor();
    d.scale = f32::NAN;
    assert!(d.validate().is_err());
    d = descriptor();
    d.queries = usize::MAX;
    assert!(d.validate().is_err());
    d = descriptor();
    d.mask = AttentionMask::Segments { tokens: 0 };
    assert!(d.validate().is_err());
    d = descriptor();
    d.mask = AttentionMask::Causal {
        query_start: i64::MAX,
    };
    assert!(d.validate().is_err());
    d = descriptor();
    d.kv_heads = 0;
    assert!(d.validate().is_err());
}

#[test]
fn window_checks_last_query_offset_before_submission() {
    let mut d = descriptor();
    d.mask = AttentionMask::Window {
        query_start: i64::MAX - 3,
        left: 0,
        right: 3,
    };
    assert!(d.validate().is_err());
}
