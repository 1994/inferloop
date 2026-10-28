//! Admission quotes against the actual tensors allocated for small draft geometries.
use super::*;

fn geometry() -> Geometry {
    Geometry {
        hidden: 512,
        intermediate: 1024,
        heads: 4,
        kv_heads: 2,
        head_dim: 128,
        vocabulary: 4096,
        block: 8,
        window: 100,
        rank: 16,
        top_k: 16,
        groups: 32,
        group_size: 16,
        mask_token: 0,
        epsilon: 1e-6,
    }
}

#[test]
fn draft_quotes_reject_overflow_before_allocation() {
    let g = geometry();
    assert!(budget::fixed_bytes(&g, usize::MAX, usize::MAX).is_err());
    assert!(budget::pooled_bytes(&g, usize::MAX, 128).is_err());
}

fn bytes<T: DType>(tensor: &Tensor<T>) -> u64 {
    (tensor.size() * size_of::<T>()) as u64
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn draft_buffer_quotes_cover_allocated_tensors_at_every_width() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let g = geometry();
    let multiprocessors =
        usize::try_from(device.profile()?.multiprocessors).map_err(device_error)?;
    for slots in [1, 2, 4, 8] {
        let rows = slots * g.block;
        let work = Ingest::new(&device, &g, rows)?;
        let ingestion = bytes(&work.meta)
            + bytes(&work.normed)
            + bytes(&work.k)
            + bytes(&work.v)
            + bytes(&work.k_normed)
            + bytes(&work.k_rotated);
        let w = ProposeBuffers::new(&device, &g, slots)?;
        let floats = [
            &w.stream[0],
            &w.stream[1],
            &w.normed,
            &w.coefficients,
            &w.conv,
            &w.q,
            &w.k,
            &w.v,
            &w.q_normed,
            &w.k_normed,
            &w.q_rotated,
            &w.k_rotated,
            &w.attention,
            &w.projected,
            &w.gate,
            &w.up,
            &w.activated,
            &w.logits,
            &w.partial_values,
            &w.unary,
            &w.selector,
        ]
        .into_iter()
        .map(bytes)
        .sum::<u64>();
        let ints = [&w.meta, &w.partial_ids, &w.candidates, &w.tokens]
            .into_iter()
            .map(bytes)
            .sum::<u64>();
        let partials = w
            .partials
            .values()
            .map(|(buffer, ..)| bytes(buffer))
            .sum::<u64>();
        assert_eq!(
            budget::pooled_bytes(&g, slots, multiprocessors)?,
            ingestion + floats + ints + partials,
            "{slots} slots"
        );
    }
    Ok(())
}
