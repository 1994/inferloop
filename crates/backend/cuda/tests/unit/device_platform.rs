use super::parse_driver_build;

#[test]
fn driver_build_parser_preserves_linux_cache_identity() {
    assert_eq!(
        parse_driver_build("module info\nNVRM version: NVIDIA UNIX 570.12  \r\ncompiler info\n"),
        Some("NVRM version: NVIDIA UNIX 570.12")
    );
    assert_eq!(
        parse_driver_build("NVRM version:\nNVRM version: later"),
        Some("NVRM version:")
    );
    assert_eq!(parse_driver_build(" NVRM version: indented"), None);
    assert_eq!(parse_driver_build("compiler info"), None);
    assert_eq!(parse_driver_build(""), None);
}
