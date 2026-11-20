use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Bytes per stored `f32` state element, used by the allocation accounting.
const F32_BYTES: usize = size_of::<f32>();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagedRows {
    pub width: usize,
    pub page_rows: usize,
    pub capacity: usize,
    pub rows: usize,
    pages: Vec<Arc<Vec<f32>>>,
}
impl PagedRows {
    /// # Errors
    /// Returns an invalid-input error for zero or overflowing page dimensions.
    pub fn new(width: usize, page_rows: usize, capacity: usize) -> Result<Self> {
        if width == 0 || page_rows == 0 || capacity == 0 || width.checked_mul(page_rows).is_none() {
            return Err(Error::invalid("invalid physical page dimensions"));
        }
        Ok(Self {
            width,
            page_rows,
            capacity,
            rows: 0,
            pages: vec![],
        })
    }
    /// # Errors
    /// Returns an invalid-input error for incompatible or non-finite rows, or when all rows are occupied.
    pub fn push(&mut self, row: &[f32]) -> Result<()> {
        if row.len() != self.width
            || self.rows >= self.capacity
            || row.iter().any(|v| !v.is_finite())
        {
            return Err(Error::invalid("physical page row/capacity mismatch"));
        }
        let page = self.rows / self.page_rows;
        let offset = self.rows % self.page_rows * self.width;
        if page == self.pages.len() {
            self.pages
                .push(Arc::new(vec![0.0; self.page_rows * self.width]));
        }
        Arc::make_mut(&mut self.pages[page])[offset..offset + self.width].copy_from_slice(row);
        self.rows += 1;
        Ok(())
    }
    /// # Errors
    /// Returns an invalid-input error for an uncommitted row.
    pub fn row(&self, row: usize) -> Result<&[f32]> {
        if row >= self.rows {
            return Err(Error::invalid("physical row out of bounds"));
        }
        let offset = row % self.page_rows * self.width;
        Ok(&self.pages[row / self.page_rows][offset..offset + self.width])
    }
    #[must_use]
    pub const fn allocated_bytes(&self) -> usize {
        self.pages.len() * self.page_rows * self.width * F32_BYTES
    }
    #[must_use]
    pub fn shared_pages(&self) -> usize {
        self.pages
            .iter()
            .filter(|p| Arc::strong_count(p) > 1)
            .count()
    }
    /// # Errors
    /// Returns an invalid-input or invariant error if dimensions, identities, ranges, or ownership are inconsistent.
    pub fn validate(&self) -> Result<()> {
        if self.width == 0
            || self.page_rows == 0
            || self.capacity == 0
            || self.rows > self.capacity
            || self.pages.len() != self.rows.div_ceil(self.page_rows)
            || self.pages.iter().any(|p| {
                p.len() != self.width.saturating_mul(self.page_rows)
                    || p.iter().any(|v| !v.is_finite())
            })
        {
            return Err(Error::invalid("corrupt physical paged rows"));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhysicalTensor {
    Kv {
        keys: PagedRows,
        values: PagedRows,
    },
    Conv {
        channels: usize,
        kernel: usize,
        history: Vec<f32>,
    },
    Delta {
        heads: usize,
        key_dim: usize,
        value_dim: usize,
        recurrent: Vec<f32>,
    },
}
impl PhysicalTensor {
    #[must_use]
    pub const fn allocated_bytes(&self) -> usize {
        match self {
            Self::Kv { keys, values } => keys.allocated_bytes() + values.allocated_bytes(),
            Self::Conv { history, .. } => history.len() * F32_BYTES,
            Self::Delta { recurrent, .. } => recurrent.len() * F32_BYTES,
        }
    }
}
#[cfg(test)]
mod recurrent_reference;
#[cfg(test)]
mod tests;
