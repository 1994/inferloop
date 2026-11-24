use infer_core::{ErrorCode, Result};
use std::iter::empty;
use std::iter::once;

use infer_state::{blocks::BlockLease, kv::*};
#[derive(Clone)]
struct Prefix {
    tokens: Vec<u32>,
    blocks: Vec<BlockLease>,
    bytes: u64,
}
impl KvPrefix for Prefix {
    fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    fn blocks(&self) -> &[BlockLease] {
        &self.blocks
    }
    fn bytes(&self) -> u64 {
        self.bytes
    }
}
fn manager(blocks: usize) -> Result<KvPages<Prefix>> {
    KvPages::new(KvCacheConfig {
        namespace: b"model-weights-layout-v1".to_vec(),
        block_size: 2,
        blocks,
        bytes_per_block: 32,
        prefix_bytes: 128,
        max_prefixes: 4,
    })
}
#[test]
fn prefix_ownership_attachment_growth_and_cow_use_one_ledger() -> Result<()> {
    let mut manager = manager(5)?;
    let mut table = vec![];
    assert!(manager.prepare_append(&mut table, 0, 3)?.is_none());
    assert!(manager.publish_prefix(Prefix {
        tokens: vec![1, 2],
        blocks: table[..1].to_vec(),
        bytes: 32
    })?);
    let attached = manager
        .attach_prefix(&[1, 2, 3], 2)?
        .ok_or_else(|| infer_core::Error::invariant("prefix expected"))?;
    let mut child = attached.blocks;
    manager.check_owners([table.as_slice(), child.as_slice()].into_iter())?;
    let view = manager.inspect([table.as_slice(), child.as_slice()].into_iter());
    assert_eq!(
        (view.active_blocks, view.cached_blocks, view.shared_blocks),
        (2, 1, 1)
    );
    assert!(manager.prepare_append(&mut child, 2, 3)?.is_none());
    let sibling = table.clone();
    manager.retain(&sibling)?;
    let growth = manager.page_growth(&table, 3)?;
    assert!(growth.cow_tail);
    assert_eq!(growth.required_pages(4), Some(1));
    let copy = manager
        .prepare_append(&mut table, 3, 4)?
        .ok_or_else(|| infer_core::Error::invariant("copy expected"))?;
    assert_eq!(copy.source, sibling[1]);
    assert_eq!(copy.destination, table[1]);
    assert_ne!(copy.source, copy.destination);
    manager.check_owners([table.as_slice(), child.as_slice(), sibling.as_slice()].into_iter())?;
    manager.release(&table)?;
    manager.release(&child)?;
    manager.release(&sibling)?;
    manager.check_owners(empty())?;
    assert_eq!(manager.inspect(empty()).cached_blocks, 1);
    manager.reclaim(5)?;
    assert_eq!(manager.free_blocks(), 5);
    assert_eq!(manager.prefix_count(), 0);
    Ok(())
}
#[test]
fn active_pages_cannot_be_evicted_and_recycled_generations_reject_stale_leases() -> Result<()> {
    let mut manager = manager(2)?;
    let active = manager.allocate(2)?;
    manager.publish_prefix(Prefix {
        tokens: vec![1, 2],
        blocks: active[..1].to_vec(),
        bytes: 32,
    })?;
    assert_eq!(
        manager.allocate(1).err().map(|e| e.code),
        Some(ErrorCode::Capacity)
    );
    assert_eq!(manager.free_blocks(), 0);
    manager.check_owners(once(active.as_slice()))?;
    manager.release(&active)?;
    let recycled = manager.allocate(2)?;
    assert!(recycled.iter().all(|l| l.generation > 1));
    assert!(manager.retain(&active).is_err());
    assert!(manager.release(&active).is_err());
    manager.check_owners(once(recycled.as_slice()))?;
    Ok(())
}
#[test]
fn invalid_prefix_and_budget_rejection_do_not_leak_cache_references() -> Result<()> {
    let mut manager = manager(2)?;
    let pages = manager.allocate(1)?;
    assert!(
        manager
            .publish_prefix(Prefix {
                tokens: vec![1],
                blocks: pages.clone(),
                bytes: 32
            })
            .is_err()
    );
    assert!(
        manager
            .publish_prefix(Prefix {
                tokens: vec![1, 2, 3, 4],
                blocks: pages.clone(),
                bytes: 32
            })
            .is_err()
    );
    assert!(!manager.publish_prefix(Prefix {
        tokens: vec![1, 2],
        blocks: pages.clone(),
        bytes: 129
    })?);
    assert_eq!(manager.references(pages[0])?, 1);
    assert_eq!(manager.prefix_count(), 0);
    assert!(manager.attach_prefix(&[1, 2], 2)?.is_none());
    assert!(manager.page_growth(&pages, 0).is_err());
    manager.check_owners(once(pages.as_slice()))?;
    Ok(())
}
