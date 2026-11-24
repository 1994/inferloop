use super::*;
#[test]
fn large_masks_preserve_word_boundaries_and_reject_overflow() -> Result<()> {
    let mut mask = Mask::new(8193)?;
    for bit in [0, 63, 64, 1023, 1024, 8192] {
        mask.insert(bit)?;
        assert!(mask.contains(bit));
    }
    assert!(!mask.contains(8191));
    assert!(!mask.contains(mask.bits()));
    assert!(mask.insert(mask.bits()).is_err());
    assert_eq!(mask.bytes(), mask.bits() / 8);
    Ok(())
}
