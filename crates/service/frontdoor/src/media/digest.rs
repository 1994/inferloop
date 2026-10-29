//! Stable media digest, with independent images processed together.
use infer_core::{Error, Result};
use infer_models::PromptImage;
use std::array::from_fn;

#[cfg(test)]
#[path = "../../tests/unit/media_digest.rs"]
mod tests;

/// Number of independent hash chains processed at once.
pub(super) const BATCH_IMAGES: usize = 4;
/// Number of axes in a media grid.
const GRID_AXES: usize = 3;
/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[inline]
fn hash_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
}

fn finish(mut hash: u64, pixels: &[f32], grid: (usize, usize, usize)) -> u64 {
    for value in pixels {
        for byte in value.to_bits().to_le_bytes() {
            hash = hash_byte(hash, byte);
        }
    }
    for value in <[usize; GRID_AXES]>::from(grid) {
        for byte in u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes() {
            hash = hash_byte(hash, byte);
        }
    }
    hash
}

fn group_common<const N: usize>(images: [&PromptImage; N]) -> [u64; N] {
    let common = images
        .iter()
        .map(|image| image.pixels.len())
        .min()
        .unwrap_or(0);
    let mut hashes = [FNV_OFFSET_BASIS; N];
    for index in 0..common {
        let words = images.map(|image| image.pixels[index].to_bits().to_le_bytes());
        for byte in 0..size_of::<f32>() {
            for (hash, word) in hashes.iter_mut().zip(&words) {
                *hash = hash_byte(*hash, word[byte]);
            }
        }
    }
    from_fn(|lane| {
        finish(
            hashes[lane],
            &images[lane].pixels[common..],
            images[lane].grid,
        )
    })
}

struct Lane<'a> {
    image: &'a PromptImage,
    hash: u64,
    original: usize,
}

fn advance<const N: usize>(mut lanes: [&mut Lane<'_>; N], start: usize, end: usize) {
    for index in start..end {
        let words: [[u8; size_of::<f32>()]; N] =
            from_fn(|lane| lanes[lane].image.pixels[index].to_bits().to_le_bytes());
        for byte in 0..size_of::<f32>() {
            for (lane, word) in lanes.iter_mut().zip(&words) {
                lane.hash = hash_byte(lane.hash, word[byte]);
            }
        }
    }
}

fn group_uneven<const N: usize>(images: [&PromptImage; N]) -> Result<[u64; N]> {
    let mut lanes: [Lane<'_>; N] = from_fn(|index| Lane {
        image: images[index],
        hash: FNV_OFFSET_BASIS,
        original: index,
    });
    lanes.sort_unstable_by_key(|lane| lane.image.pixels.len());
    let mut start = 0;
    for index in 0..N {
        let end = lanes[index].image.pixels.len();
        if end > start {
            match &mut lanes[index..] {
                [a] => advance([a], start, end),
                [a, b] => advance([a, b], start, end),
                [a, b, c] => advance([a, b, c], start, end),
                [a, b, c, d] => advance([a, b, c, d], start, end),
                _ => return Err(Error::invariant("invalid media digest lane count")),
            }
        }
        start = end;
        lanes[index].hash = finish(lanes[index].hash, &[], lanes[index].image.grid);
    }
    let mut hashes = [0; N];
    for lane in lanes {
        hashes[lane.original] = lane.hash;
    }
    Ok(hashes)
}

fn group<const N: usize>(images: [&PromptImage; N]) -> Result<[u64; N]> {
    // Two lanes cannot overlap a tail with another tail. Preserve the established fast path
    // for that case and for equal sizes, where every lane has the same end point.
    if N <= 2
        || images
            .iter()
            .all(|image| image.pixels.len() == images[0].pixels.len())
    {
        Ok(group_common(images))
    } else {
        group_uneven(images)
    }
}

/// Digest at most four images, preserving their individual byte order and input order.
///
/// Each float is hashed by its exact little-endian bits, followed by each grid axis as little-
/// endian u64. This retains the original `fnv1a64` contract for every image independently.
///
/// # Errors
/// Rejects batches larger than the fixed number of independent chains.
pub(super) fn hashes(images: &[PromptImage]) -> Result<[u64; BATCH_IMAGES]> {
    Ok(match images {
        [] => [0; BATCH_IMAGES],
        [a] => [finish(FNV_OFFSET_BASIS, &a.pixels, a.grid), 0, 0, 0],
        [a, b] => {
            let [a, b] = group([a, b])?;
            [a, b, 0, 0]
        }
        [a, b, c] => {
            let [a, b, c] = group([a, b, c])?;
            [a, b, c, 0]
        }
        [a, b, c, d] => group([a, b, c, d])?,
        _ => return Err(Error::invalid("media digest batch too large")),
    })
}
