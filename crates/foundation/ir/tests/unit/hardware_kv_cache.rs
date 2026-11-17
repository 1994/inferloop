use super::KvCacheEncoding;

#[test]
fn every_encoding_parses_back_from_its_label() {
    for encoding in [
        KvCacheEncoding::F32,
        KvCacheEncoding::F16,
        KvCacheEncoding::Fp8E4m3,
    ] {
        assert_eq!(
            encoding.label().parse::<KvCacheEncoding>().ok(),
            Some(encoding)
        );
    }
}

#[test]
fn an_unknown_dtype_is_rejected() {
    assert!("q4_0".parse::<KvCacheEncoding>().is_err());
}

#[test]
fn element_bytes_follow_the_storage_width() {
    assert_eq!(KvCacheEncoding::F32.element_bytes(), 4);
    assert_eq!(KvCacheEncoding::F16.element_bytes(), 2);
    assert_eq!(KvCacheEncoding::Fp8E4m3.element_bytes(), 1);
}
