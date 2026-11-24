//! Model- and device-independent scaled dot-product attention semantics.
//!
//! A descriptor describes one sequence. A batch/varlen caller supplies a descriptor per
//! sequence and retains ownership of its offsets and KV storage. `RoPE` is a separate operator.
use infer_core::{Error, Result};
use infer_ir::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMask {
    None,
    /// Key index must be at most `query_index + query_start`. Negative offsets allow
    /// bottom-right causal alignment when the query is longer than the key sequence.
    Causal {
        query_start: i64,
    },
    /// Inclusive key interval around `query_index + query_start`.
    Window {
        query_start: i64,
        left: usize,
        right: usize,
    },
    /// Queries and keys attend only within equal-sized independent segments.
    Segments {
        tokens: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttentionDescriptor {
    pub queries: usize,
    pub keys: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub qk_dim: usize,
    pub value_dim: usize,
    pub scale: f32,
    pub dtype: DType,
    pub mask: AttentionMask,
}

impl AttentionDescriptor {
    /// Validate SDPA/GQA semantics independently of any kernel's capability limits.
    /// # Errors
    /// Rejects empty/overflowing geometry, invalid head grouping or mask coordinates.
    pub fn validate(&self) -> Result<()> {
        if [
            self.queries,
            self.keys,
            self.query_heads,
            self.kv_heads,
            self.qk_dim,
            self.value_dim,
        ]
        .contains(&0)
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || !self.scale.is_finite()
            || !matches!(self.dtype, DType::F32 | DType::F16 | DType::Bf16)
        {
            return Err(Error::invalid("attention geometry, precision or scale"));
        }
        for dims in [
            [self.queries, self.query_heads, self.qk_dim],
            [self.keys, self.kv_heads, self.qk_dim],
            [self.keys, self.kv_heads, self.value_dim],
            [self.queries, self.query_heads, self.value_dim],
        ] {
            dims.into_iter()
                .try_fold(1usize, usize::checked_mul)
                .ok_or_else(|| Error::invalid("attention size overflow"))?;
        }
        let offset = match self.mask {
            AttentionMask::None => 0,
            AttentionMask::Segments { tokens } => {
                if tokens == 0 || self.queries != self.keys {
                    return Err(Error::invalid("attention segments"));
                }
                0
            }
            AttentionMask::Causal { query_start } => query_start,
            AttentionMask::Window {
                query_start,
                left,
                right,
            } => {
                let left =
                    i64::try_from(left).map_err(|_| Error::invalid("attention window overflow"))?;
                let right = i64::try_from(right)
                    .map_err(|_| Error::invalid("attention window overflow"))?;
                query_start
                    .checked_sub(left)
                    .and_then(|_| query_start.checked_add(right))
                    .and_then(|end| {
                        i64::try_from(self.queries - 1)
                            .ok()
                            .and_then(|q| end.checked_add(q))
                    })
                    .ok_or_else(|| Error::invalid("attention window offset overflow"))?;
                query_start
            }
        };
        let queries =
            i64::try_from(self.queries).map_err(|_| Error::invalid("attention query overflow"))?;
        offset
            .checked_add(queries)
            .ok_or_else(|| Error::invalid("attention offset overflow"))?;
        Ok(())
    }

    /// Host mask oracle; useful for independent backend contract tests.
    #[must_use]
    pub fn visible(&self, query: usize, key: usize) -> bool {
        if query >= self.queries || key >= self.keys {
            return false;
        }
        let (Ok(q), Ok(k)) = (i64::try_from(query), i64::try_from(key)) else {
            return false;
        };
        match self.mask {
            AttentionMask::None => true,
            AttentionMask::Segments { tokens } => tokens != 0 && query / tokens == key / tokens,
            AttentionMask::Causal { query_start } => q
                .checked_add(query_start)
                .is_some_and(|position| k <= position),
            AttentionMask::Window {
                query_start,
                left,
                right,
            } => {
                let (Ok(left), Ok(right)) = (i64::try_from(left), i64::try_from(right)) else {
                    return false;
                };
                q.checked_add(query_start).is_some_and(|position| {
                    position.checked_sub(left).is_some_and(|start| k >= start)
                        && position.checked_add(right).is_some_and(|end| k <= end)
                })
            }
        }
    }
}

#[cfg(test)]
mod tests;
