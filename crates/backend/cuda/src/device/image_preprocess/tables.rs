//! Build both fixed-point filter axes into one upload.
use super::{axis::AxisTable, geometry::Geometry};
use infer_core::{Error, ErrorCode, Result};
use std::sync::Arc;
/// Horizontal and vertical axes each carry index and coefficient arrays.
const TABLE_PARTS: usize = 4;

pub(super) struct Tables {
    pub values: Arc<Vec<i32>>,
    /// Byte offsets of horizontal indices, horizontal weights, vertical indices and weights.
    pub offsets: [u64; TABLE_PARTS],
    pub horizontal_taps: i32,
    pub horizontal_denominator: i32,
    pub vertical_taps: i32,
    pub vertical_denominator: i32,
}

impl Tables {
    pub fn new(geometry: &Geometry) -> Result<Self> {
        let horizontal = AxisTable::new(
            to_usize(geometry.source_width)?,
            to_usize(geometry.target_width)?,
        )?;
        let vertical = AxisTable::new(
            to_usize(geometry.source_height)?,
            to_usize(geometry.target_height)?,
        )?;
        let parts = [
            horizontal.indices.as_slice(),
            horizontal.weights.as_slice(),
            vertical.indices.as_slice(),
            vertical.weights.as_slice(),
        ];
        let mut offsets = [0_u64; TABLE_PARTS];
        let mut total = 0_usize;
        for (offset, values) in offsets.iter_mut().zip(parts) {
            *offset = u64::try_from(
                total
                    .checked_mul(size_of::<i32>())
                    .ok_or_else(|| Error::invalid("image filter offset overflow"))?,
            )
            .map_err(|_| Error::invalid("image filter offset overflow"))?;
            total = total
                .checked_add(values.len())
                .ok_or_else(|| Error::invalid("image filter length overflow"))?;
        }
        i32::try_from(total)
            .map_err(|_| Error::invalid("image filter exceeds CUDA integer ABI"))?;
        let mut values = Vec::new();
        values.try_reserve_exact(total).map_err(|error| {
            Error::new(
                ErrorCode::Capacity,
                format!("image filter allocation failed: {error}"),
            )
        })?;
        for part in parts {
            values.extend_from_slice(part);
        }
        Ok(Self {
            values: Arc::new(values),
            offsets,
            horizontal_taps: i32::try_from(horizontal.taps)
                .map_err(|_| Error::invalid("horizontal filter taps exceed CUDA integer ABI"))?,
            horizontal_denominator: horizontal.denominator,
            vertical_taps: i32::try_from(vertical.taps)
                .map_err(|_| Error::invalid("vertical filter taps exceed CUDA integer ABI"))?,
            vertical_denominator: vertical.denominator,
        })
    }
}

fn to_usize(value: i32) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::invalid("image dimension out of range"))
}
