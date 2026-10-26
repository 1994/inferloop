use crate::tuning::resident_gemm::{ResidentTuningSession, SearchFamily};

#[test]
fn completed_pool_search_does_not_complete_or_expire_swiglu_or_nvfp4() {
    let session = ResidentTuningSession::default();
    let shape = [16, 5120, 128];
    for family in [
        SearchFamily::PoolDense,
        SearchFamily::FusedSwiGLU,
        SearchFamily::Nvfp4,
    ] {
        session.plan_gemm_search(family, &[shape], 4).unwrap();
    }
    assert!(
        session
            .permits_gemm_search(SearchFamily::PoolDense, shape, 4)
            .unwrap()
    );
    session
        .complete_gemm_search(SearchFamily::PoolDense, shape)
        .unwrap();
    assert!(
        !session
            .permits_gemm_search(SearchFamily::PoolDense, shape, 4)
            .unwrap()
    );
    assert!(
        session
            .permits_gemm_search(SearchFamily::FusedSwiGLU, shape, 4)
            .unwrap()
    );
    assert!(
        session
            .permits_gemm_search(SearchFamily::Nvfp4, shape, 4)
            .unwrap()
    );
    assert!(
        !session
            .gemm_search_exhausted(SearchFamily::FusedSwiGLU, shape)
            .unwrap()
    );
    assert!(
        !session
            .gemm_search_exhausted(SearchFamily::Nvfp4, shape)
            .unwrap()
    );
}

#[test]
fn expired_search_is_uncovered_and_does_not_mark_complete() {
    let session = ResidentTuningSession::without_search_budget();
    let shape = [16, 5120, 128];
    assert!(
        !session
            .permits_gemm_search(SearchFamily::PoolDense, shape, 4)
            .unwrap()
    );
    assert!(
        session
            .gemm_search_exhausted(SearchFamily::PoolDense, shape)
            .unwrap()
    );
    // A completion for another family must not reset or complete this one.
    session
        .complete_gemm_search(SearchFamily::FusedSwiGLU, shape)
        .unwrap();
    assert!(
        !session
            .permits_gemm_search(SearchFamily::PoolDense, shape, 4)
            .unwrap()
    );
}
