use super::*;
use crate::{blocks::BlockLease, kv::KvCacheConfig};
#[derive(Clone)]
struct PageSnapshot {
    tokens: Vec<u32>,
    blocks: Vec<BlockLease>,
}
impl KvPrefix for PageSnapshot {
    fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    fn blocks(&self) -> &[BlockLease] {
        &self.blocks
    }
    fn bytes(&self) -> u64 {
        4
    }
}
#[test]
fn stale_page_acquisition_does_not_commit_a_cache_hit() {
    let mut manager = KvCacheManager::new(KvCacheConfig {
        namespace: b"pages".to_vec(),
        block_size: 2,
        blocks: 2,
        bytes_per_block: 4,
        prefix_bytes: 8,
        max_prefixes: 2,
    })
    .unwrap();
    let pages = manager.allocate(1).unwrap();
    manager
        .publish_prefix(PageSnapshot {
            tokens: vec![1, 2],
            blocks: pages.clone(),
        })
        .unwrap();
    manager.release(&pages).unwrap();
    // Fault injection: invalidate the cache's lease after publication.
    manager.pool.release(&pages).unwrap();
    assert!(manager.attach_prefix(&[1, 2], 2).is_err());
    assert_eq!(
        (
            manager.prefixes.lookups,
            manager.prefixes.hits,
            manager.prefixes.reused_tokens
        ),
        (1, 0, 0)
    );
    assert_eq!((manager.prefix_count(), manager.prefix_bytes()), (1, 4));
}
