use super::projection_output;
use infer_core::Result;

#[test]
fn aligned_projection_preserves_the_readback_buffer() -> Result<()> {
    let values = vec![1.0, -2.0, 3.5, 4.0];
    let address = values.as_ptr();
    let result = projection_output(values, 2, 2)?;
    assert_eq!(result, [1.0, -2.0, 3.5, 4.0]);
    assert_eq!(result.as_ptr(), address);
    Ok(())
}

#[test]
fn padded_projection_discards_tail_without_retaining_it() -> Result<()> {
    let values = vec![1.0, -2.0, 3.5, 4.0, 7.0, 8.0];
    let address = values.as_ptr();
    let result = projection_output(values, 2, 2)?;
    assert_eq!(result, [1.0, -2.0, 3.5, 4.0]);
    assert_ne!(result.as_ptr(), address);
    Ok(())
}

#[test]
fn projection_output_rejects_short_or_overflowing_shapes() {
    assert!(projection_output(vec![1.0, 2.0, 3.0], 2, 2).is_err());
    assert!(projection_output(vec![1.0], usize::MAX, 2).is_err());
}
