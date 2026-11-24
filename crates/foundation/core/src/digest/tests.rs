use super::lower_hex;

#[test]
fn lower_hex_writes_two_lowercase_digits_per_byte_in_order() {
    assert_eq!(lower_hex(&[]), "");
    assert_eq!(lower_hex(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
}
