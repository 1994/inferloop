use super::*;
use infer_core::{Error, Result};
fn recipe() -> StateRecipe {
    StateRecipe {
        block_size: 16,
        max_tokens: 64,
        regions: vec![
            StateRegion {
                kind: StateRegionKind::Tensor {
                    tensor: TensorId::ONE,
                    kind: StateKind::AttentionKv,
                },
                memory: StateMemory::KvBlock,
                extent: StateExtent::Fixed(16 * 8 * 2),
                element_bytes: 2,
                reset: StateReset::ReleaseLease,
            },
            StateRegion {
                kind: StateRegionKind::Tensor {
                    tensor: TensorId::ONE,
                    kind: StateKind::LinearAttention,
                },
                memory: StateMemory::DevicePrivate,
                extent: StateExtent::Fixed(256),
                element_bytes: 4,
                reset: StateReset::Zero,
            },
            StateRegion {
                kind: StateRegionKind::Hidden,
                memory: StateMemory::DevicePrivate,
                extent: StateExtent::Hidden(32),
                element_bytes: 4,
                reset: StateReset::Zero,
            },
            StateRegion {
                kind: StateRegionKind::PageTable,
                memory: StateMemory::HostMirror,
                extent: StateExtent::Pages,
                element_bytes: 4,
                reset: StateReset::InvalidPage,
            },
        ],
    }
}
#[test]
fn kv_blocks_and_recurrent_state_are_distinct_from_readout_and_host_mirrors() -> Result<()> {
    let recipe = recipe();
    let full = recipe.layout(33, OutputReadout::Full)?;
    let logits = recipe.layout(33, OutputReadout::Logits)?;
    assert_eq!(
        (full.max_pages, full.kv_block_bytes, full.host_bytes),
        (3, 512, 12)
    );
    assert_eq!(full.private_bytes, 1024 + 33 * 32 * 4);
    assert_eq!(logits.private_bytes, 1024 + 32 * 4);
    let mut end = 0;
    recipe.visit(33, OutputReadout::Full, |region| {
        if region.region.memory == StateMemory::DevicePrivate {
            assert_eq!(region.offset, end);
            end += region.bytes;
        }
        Ok(())
    })?;
    assert_eq!(end, full.private_bytes);
    Ok(())
}
#[test]
fn recipe_rejects_invalid_capacity_and_arithmetic_overflow() {
    let mut recipe = recipe();
    assert!(recipe.layout(0, OutputReadout::Full).is_err());
    assert!(recipe.layout(65, OutputReadout::Full).is_err());
    recipe.regions[1].extent = StateExtent::Tokens(usize::MAX);
    assert!(recipe.layout(2, OutputReadout::Full).is_err());
    recipe.block_size = 0;
    assert!(recipe.layout(1, OutputReadout::Full).is_err());
}
#[test]
fn visitor_propagates_physical_allocation_failure() {
    assert!(
        recipe()
            .visit(1, OutputReadout::Full, |_| Err(Error::invalid(
                "allocation failure"
            )))
            .is_err()
    );
}
