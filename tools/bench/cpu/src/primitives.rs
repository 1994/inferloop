use crate::measure::{Measurement, measure};
use infer_core::{
    Error, RequestHandle, Result, arena::Arena, credits::CreditPool, map::BoundedMap,
};
use infer_spi::{ResourcePool, ResourceReply};
use infer_state::blocks::BlockLease;
use infer_state::kv::KvCacheConfig;
use infer_state::kv::KvPages;
use infer_state::kv::KvPrefix;
#[derive(Clone)]
struct NoPrefix;
impl KvPrefix for NoPrefix {
    fn tokens(&self) -> &[u32] {
        &[]
    }
    fn blocks(&self) -> &[BlockLease] {
        &[]
    }
    fn bytes(&self) -> u64 {
        0
    }
}
pub fn run() -> Result<Measurement> {
    let mut arena = Arena::<RequestHandle, u64>::with_capacity(64)?;
    let mut map = BoundedMap::new(64)?;
    let credits = CreditPool::new(64, 1024)?;
    let replies = ResourcePool::new(64)?;
    let mut next = 0u64;
    let mut state = infer_state::SequenceStates::with_capacity(256, 16, 64)?;
    let first = state.reserve_incremental(
        infer_core::RequestId::ONE,
        infer_ir::StateKind::AttentionKv,
        1024,
    )?;
    let second = state.reserve_incremental(
        infer_core::RequestId::new(2)?,
        infer_ir::StateKind::AttentionKv,
        1024,
    )?;

    let mut physical = KvPages::<NoPrefix>::new(KvCacheConfig {
        namespace: vec![1],
        block_size: 16,
        blocks: 256,
        bytes_per_block: 64,
        prefix_bytes: 0,
        max_prefixes: 16,
    })?;
    let mut table = Vec::with_capacity(64);
    let mut fork = Vec::with_capacity(64);
    measure(10_000, || {
        next += 1;
        state.ensure_batch(&[(first, 33), (second, 65)])?;
        state.commit(first, 33)?;
        state.commit(second, 65)?;
        state.reset(first)?;
        state.reset(second)?;
        if state.free_pages() != 256 {
            return Err(Error::invariant("logical KV page leak"));
        }

        physical.prepare_append(&mut table, 0, 33)?;
        physical.retain(&table)?;
        fork.extend(table.iter().copied());
        let source = *table
            .last()
            .ok_or_else(|| Error::invariant("physical tail missing"))?;
        physical.retain(&[source])?;
        let copy = physical.prepare_append(&mut table, 33, 34)?;
        if copy.map(|copy| copy.source) != Some(source) {
            return Err(Error::invariant("physical COW plan missing"));
        }
        physical.release(&[source])?;
        physical.release(&table)?;
        physical.release(&fork)?;
        table.clear();
        fork.clear();
        if physical.free_blocks() != 256 {
            return Err(Error::invariant("physical KV page leak"));
        }
        let handle = arena.insert(next)?;
        map.insert(handle, next)?;
        let credit = credits.reserve(16)?;
        let reader = credit.clone();
        let (mut ticket, responder) = replies.channel()?;
        responder
            .send(Ok(ResourceReply::ReservationBytes(Some(next))))
            .map_err(|_| Error::invariant("reply receiver lost"))?;
        if !matches!(ticket.poll()?,Some(ResourceReply::ReservationBytes(value)) if value==Some(next))
            || map.remove(&handle) != Some(next)
            || arena.remove(handle) != Some(next)
        {
            return Err(Error::invariant("fixed CPU ownership mismatch"));
        }
        drop(credit);
        drop(reader);
        if credits.used() != 0 {
            return Err(Error::invariant("credit leak"));
        }
        Ok(())
    })
}
