use super::*;

#[expect(
    unsafe_code,
    reason = "test versions are written to the query's output pointer"
)]
unsafe extern "C" fn nvjpeg_version(property: i32, value: *mut i32) -> i32 {
    let version = match property {
        MAJOR_VERSION => 13,
        MINOR_VERSION => 2,
        PATCH_LEVEL => 3,
        _ => return 1,
    };
    // SAFETY: the query contract provides a live, exclusive i32 pointer.
    unsafe { value.write(version) };
    NVJPEG_SUCCESS
}

#[expect(
    unsafe_code,
    reason = "test versions are written to the query's output pointer"
)]
unsafe extern "C" fn cudart_version(property: i32, value: *mut i32) -> i32 {
    let version = match property {
        MAJOR_VERSION => 13,
        MINOR_VERSION => 4,
        PATCH_LEVEL => 0,
        _ => return 1,
    };
    // SAFETY: the query contract provides a live, exclusive i32 pointer.
    unsafe { value.write(version) };
    NVJPEG_SUCCESS
}

#[expect(unsafe_code, reason = "the fake uses the nvJPEG C function ABI")]
unsafe extern "C" fn failing_query(_: i32, _: *mut i32) -> i32 {
    6
}

#[expect(
    unsafe_code,
    reason = "test versions are written to the query's output pointer"
)]
unsafe extern "C" fn old_version(property: i32, value: *mut i32) -> i32 {
    // SAFETY: the query contract provides a live, exclusive i32 pointer.
    unsafe { value.write(if property == MAJOR_VERSION { 12 } else { 0 }) };
    NVJPEG_SUCCESS
}

#[expect(
    unsafe_code,
    reason = "test versions are written to the query's output pointer"
)]
unsafe extern "C" fn invalid_version(_: i32, value: *mut i32) -> i32 {
    // SAFETY: the query contract provides a live, exclusive i32 pointer.
    unsafe { value.write(-1) };
    NVJPEG_SUCCESS
}

#[test]
fn version_queries_return_both_independent_versions() -> Result<()> {
    let info = version_info(nvjpeg_version, cudart_version)?;
    assert_eq!(
        info.nvjpeg,
        Version {
            major: 13,
            minor: 2,
            patch: 3
        }
    );
    assert_eq!(
        info.cudart,
        Version {
            major: 13,
            minor: 4,
            patch: 0
        }
    );
    Ok(())
}

#[test]
fn unavailable_abi_and_failed_or_invalid_queries_are_explicit() {
    assert_eq!(
        version_info(old_version, cudart_version).unwrap_err().code,
        ErrorCode::Unsupported
    );
    assert_eq!(
        version_info(failing_query, cudart_version)
            .unwrap_err()
            .code,
        ErrorCode::Backend
    );
    assert_eq!(
        version_info(nvjpeg_version, failing_query)
            .unwrap_err()
            .code,
        ErrorCode::Backend
    );
    assert_eq!(
        version_info(invalid_version, cudart_version)
            .unwrap_err()
            .code,
        ErrorCode::Backend
    );
}

#[test]
fn missing_library_returns_unsupported_with_loader_error() {
    match Api::load_names(&["libnvjpeg-infer-missing-test-library.so"]) {
        Ok(_) => panic!("nonexistent test library unexpectedly loaded"),
        Err(error) => {
            assert_eq!(error.code, ErrorCode::Unsupported);
            assert!(
                error
                    .message
                    .contains("libnvjpeg-infer-missing-test-library.so")
            );
        }
    }
}

#[test]
fn installed_library_version_is_queried_or_explicitly_unsupported() -> Result<()> {
    match library_info() {
        Ok(info) => {
            assert_eq!(info.nvjpeg.major, SUPPORTED_ABI_MAJOR);
            assert!(info.cudart.major > 0);
            Ok(())
        }
        Err(error)
            if error.code == ErrorCode::Unsupported
                && (error.message.starts_with("nvJPEG library is unavailable:")
                    || error.message.starts_with("nvJPEG ABI major ")) =>
        {
            Ok(())
        }
        Err(error) => Err(error),
    }
}
