//! Reservation responsibilities.
use super::{Page, PrefixKey, SequenceState, SequenceStates};
use infer_core::{
    Error, ErrorCode, IdAllocator, RequestId, Result, StateId, StatePageId, map::BoundedMap,
};
use infer_ir::StateKind;

/// Floor on the request capacity granted by `SequenceStates::new`, so small
/// page pools still provision enough sequence/owner ledger slots for the scheduler.
const MIN_REQUEST_CAPACITY: usize = 256;

impl SequenceStates {
    /// # Errors
    /// Returns an invalid-input error for a zero state capacity or page size.
    pub fn new(total_pages: usize, block_size: usize) -> Result<Self> {
        Self::with_capacity(
            total_pages,
            block_size,
            total_pages.max(MIN_REQUEST_CAPACITY),
        )
    }
    /// # Errors
    /// Rejects invalid dimensions or inability to allocate the fixed page and ownership ledgers.
    pub fn with_capacity(total_pages: usize, block_size: usize, requests: usize) -> Result<Self> {
        if total_pages == 0 || block_size == 0 {
            return Err(Error::invalid("state capacity/page size must be positive"));
        }
        Ok(Self {
            spare_tables: Vec::with_capacity(requests),
            total_pages,
            block_size,
            ids: IdAllocator::default(),
            pages: BoundedMap::new(total_pages)?,
            sequences: BoundedMap::new(requests)?,
            owners: BoundedMap::new(requests)?,
            prefixes: Vec::new(),
        })
    }
    /// Restore vector capacities on the cold checkpoint path, before dispatch resumes.
    /// # Errors
    /// Reports invalid ledgers or failure to reserve sequence page tables.
    pub fn prepare_storage(&mut self) -> Result<()> {
        self.check_invariants()?;
        self.spare_tables.reserve(
            self.sequences
                .capacity()
                .saturating_sub(self.spare_tables.len()),
        );
        for state in self.sequences.values_mut() {
            state
                .pages
                .try_reserve_exact(
                    state.capacity_tokens.div_ceil(self.block_size) - state.pages.len(),
                )
                .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        }
        Ok(())
    }
    /// Reserve identity/capacity only. Physical/logical pages grow at dispatch.
    ///
    /// # Errors
    /// Returns an invalid-input error for an empty capacity or duplicate owner, or an invariant error if identities are exhausted.
    pub fn reserve_incremental(
        &mut self,
        owner: RequestId,
        kind: StateKind,
        capacity: usize,
    ) -> Result<StateId> {
        if capacity == 0 || self.owners.contains_key(&owner) {
            return Err(Error::invalid("invalid incremental state owner/capacity"));
        }
        if self.sequences.len() == self.sequences.capacity() {
            return Err(Error::new(
                ErrorCode::Capacity,
                "state owner pool exhausted",
            ));
        }
        let required = capacity.div_ceil(self.block_size);
        let mut pages = self.spare_tables.pop().unwrap_or_default();
        pages
            .try_reserve_exact(required)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        let id = self.ids.allocate()?;
        self.sequences.insert(
            id,
            SequenceState {
                owner,
                kind,
                pages,
                capacity_tokens: capacity,
                committed_tokens: 0,
            },
        )?;
        self.owners.insert(owner, id)?;
        Ok(id)
    }
    /// # Errors
    /// Returns a not-found, invalid-input, or capacity error for unknown state, growth past its capacity, or insufficient logical pages.
    pub fn ensure_tokens(&mut self, id: StateId, tokens: usize) -> Result<()> {
        self.ensure_batch(&[(id, tokens)])
    }
    /// Validate and reserve every logical page in a dispatch before changing ownership.
    /// # Errors
    /// Returns not-found, invalid-input, capacity, or identity-exhaustion errors; no
    /// sequence, page table, or allocator cursor changes on failure.
    pub fn ensure_batch(&mut self, growth: &[(StateId, usize)]) -> Result<()> {
        let mut total = 0usize;
        for (index, &(id, tokens)) in growth.iter().enumerate() {
            if growth[..index].iter().any(|(prior, _)| *prior == id) {
                return Err(Error::invalid("dispatch contains duplicate sequence state"));
            }
            let state = self.get(id)?;
            if tokens > state.capacity_tokens
                || state.pages.capacity() < tokens.div_ceil(self.block_size)
            {
                return Err(Error::invalid("state growth exceeds prepared capacity"));
            }
            total = total
                .checked_add(
                    tokens
                        .div_ceil(self.block_size)
                        .saturating_sub(state.pages.len()),
                )
                .ok_or_else(|| Error::invalid("logical page growth overflow"))?;
        }
        if total > self.free_pages() {
            return Err(Error::new(
                ErrorCode::Capacity,
                "logical page pool exhausted",
            ));
        }
        // All fallible capacity/cursor checks precede publication. The range is checked once;
        // converting its nonzero members and inserting into the validated fixed maps cannot fail.
        let mut ids = self.ids.clone();
        let mut range = ids.reserve_range(total)?;
        for &(id, tokens) in growth {
            let state = self
                .sequences
                .get_mut(&id)
                .ok_or_else(|| Error::invariant("validated state missing"))?;
            let count = tokens
                .div_ceil(self.block_size)
                .saturating_sub(state.pages.len());
            for raw in range.by_ref().take(count) {
                let page = StatePageId::new(raw)?;
                self.pages.insert(
                    page,
                    Page {
                        references: 1,
                        sealed: false,
                    },
                )?;
                state.pages.push(page);
            }
        }
        self.ids = ids;
        Ok(())
    }
    /// # Errors
    /// Returns an invalid-input, conflict, or capacity error for an invalid owner, reservation, or exhausted storage.
    pub fn reserve(
        &mut self,
        owner: RequestId,
        kind: StateKind,
        tokens: usize,
        prefix: Option<&PrefixKey>,
    ) -> Result<StateId> {
        if tokens == 0 {
            return Err(Error::invalid("cannot reserve empty state"));
        }
        if self.owners.contains_key(&owner) {
            return Err(Error::new(
                ErrorCode::Conflict,
                "request already owns state",
            ));
        }
        if self.sequences.len() == self.sequences.capacity() {
            return Err(Error::new(
                ErrorCode::Capacity,
                "state owner pool exhausted",
            ));
        }
        let count = tokens.div_ceil(self.block_size);
        let shared = prefix
            .and_then(|key| {
                self.prefixes
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, p)| p.clone())
            })
            .unwrap_or_default();
        if shared.len() > count || prefix.is_some_and(|p| p.tokens.len() > tokens) {
            return Err(Error::invalid("prefix exceeds reservation"));
        }
        let needed = count - shared.len();
        if needed > self.total_pages - self.pages.len() {
            return Err(Error::new(
                ErrorCode::Capacity,
                format!(
                    "state pages required {needed}, available {}",
                    self.total_pages - self.pages.len()
                ),
            ));
        }
        // Stage IDs before changing ownership so ID exhaustion cannot leak pages.
        let mut ids = self.ids.clone();
        let state_id = ids.allocate()?;
        let fresh = (0..needed)
            .map(|_| ids.allocate::<StatePageId>())
            .collect::<Result<Vec<_>>>()?;
        for page in &shared {
            let p = self
                .pages
                .get(page)
                .ok_or_else(|| Error::invariant("prefix refers to missing page"))?;
            if !p.sealed || p.references == usize::MAX {
                return Err(Error::invariant("invalid shared page"));
            }
        }
        self.ids = ids;
        for page in &shared {
            self.pages
                .get_mut(page)
                .ok_or_else(|| Error::invariant("shared page disappeared"))?
                .references += 1;
        }
        for page in &fresh {
            self.pages.insert(
                *page,
                Page {
                    references: 1,
                    sealed: false,
                },
            )?;
        }
        let committed_tokens = shared.len() * self.block_size;
        let mut pages = self.spare_tables.pop().unwrap_or_default();
        pages.extend(shared);
        pages.extend(fresh);
        self.sequences.insert(
            state_id,
            SequenceState {
                owner,
                kind,
                pages,
                capacity_tokens: tokens,
                committed_tokens,
            },
        )?;
        self.owners.insert(owner, state_id)?;
        Ok(state_id)
    }
}
