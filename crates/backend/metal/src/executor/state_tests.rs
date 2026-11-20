use super::*;
use infer_ir::{OutputReadout, StateMemory};
use infer_spi::BackendProvider;
use std::path::Path;
#[test]
fn physical_recipe_matches_real_buffers_and_reset_keeps_allocations() -> Result<()> {
    let mut package = infer_models::ModelPackage::open(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny"),
        infer_core::ModelId::ONE,
    )?;
    let mut backend = MetalBackend::from_package(
        &mut package,
        MetalConfig {
            block_size: 2,
            prefix_cache_bytes: 0,
            probe_bytes: 1 << 20,
            ..MetalConfig::default()
        },
    )?;
    for (index, readout) in [
        OutputReadout::None,
        OutputReadout::Logits,
        OutputReadout::Full,
    ]
    .into_iter()
    .enumerate()
    {
        let id = StateId::new(index as u64 + 1)?;
        backend.reserve_state_for(id, 8, readout)?;
        let sequence = &backend.sequences[&id];
        let bytes: u64 = sequence
            .tensors
            .values()
            .map(|buffer| buffer.length())
            .chain([
                sequence.hidden.length(),
                sequence.logits.length(),
                sequence.page_table.length(),
                sequence.token_buffer.length(),
            ])
            .chain(sequence.probes.iter().map(|buffer| buffer.length()))
            .sum();
        let layout = backend.state_recipe.layout(8, readout)?;
        assert_eq!(layout.private_bytes, bytes);
        assert_eq!(layout.kv_block_bytes, backend.kv_block_bytes);
        let mut host = 0;
        backend.state_recipe.visit(8, readout, |region| {
            if region.region.memory == StateMemory::HostMirror {
                host += region.bytes;
            }
            Ok(())
        })?;
        assert_eq!(host, layout.host_bytes);
        let hidden = sequence.hidden.contents();
        let table = sequence.page_table.contents();
        let tokens = sequence.tokens.as_ptr();
        backend.reset_state(id)?;
        let sequence = &backend.sequences[&id];
        assert_eq!(sequence.hidden.contents(), hidden);
        assert_eq!(sequence.page_table.contents(), table);
        assert_eq!(sequence.tokens.as_ptr(), tokens);
        assert_eq!(sequence.blocks.len(), 0);
        assert!(sequence.table_mirror.iter().all(|index| *index == u32::MAX));
        backend.release_state(id)?;
    }
    Ok(())
}
