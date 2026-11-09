use super::*;
#[test]
#[ignore = "requires CUDA/safe-run; actual automatic tuning, full proofs and persisted cache"]
fn exact_gdn_hardware_selection() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for width in [64, 256] {
        assert!(select(
            &device,
            [16, 48, 128, width],
            true,
            &ResidentTuningSession::default()
        ));
        let entry = memo()
            .lock()
            .map_err(|_| Error::invariant("exact GDN test memo"))?
            .values()
            .find(|entry| entry.key.shape == [16, 48, 128, width])
            .cloned()
            .ok_or_else(|| Error::invariant("exact GDN evidence missing"))?;
        assert!(valid(&entry));
        let mut missing = entry.clone();
        missing.reverse.pop();
        assert!(!valid(&missing));
        let mut stale = entry.clone();
        stale.key.kernel_fingerprint.clear();
        assert!(!valid(&stale));
        let mut unsafe_numeric = entry.clone();
        unsafe_numeric.max_f64_error = super::super::MAX_F64_ERROR;
        assert!(!valid(&unsafe_numeric));
        let mut slow = entry;
        slow.forward[0].warm.candidate_ms.fill(1e3);
        assert!(!valid(&slow));
    }
    Ok(())
}
