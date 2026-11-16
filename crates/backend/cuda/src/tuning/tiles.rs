//! Tile baselines and search spaces owned by the load-time projection tuners.
//!
//! Load-time measurement picks the tile for each `(device, driver, kernel, M/N/K)`. These
//! values do only two things: they are the configuration kept when no measurement applies
//! (tuning disabled, budget exhausted, or a candidate failed verification), and they list
//! the tile variants the native kernels compile. Do not select a tile from a GPU model.

/// Output tile of the activation-quantized GEMMs on decode-width graphs; the first row and
/// column variant of every resident search.
pub const QUANT_GEMM_TILE: [usize; 2] = [16, 64];
/// Baseline row tile for graphs wider than one decode tile.
///
/// A 16-row tile splits a prompt graph into row blocks that each reread the same weight tile.
/// A 64-row CTA on a one-row decode wastes most of its tensor-core work, so decode keeps
/// [`QUANT_GEMM_TILE`].
pub const PROMPT_GEMM_TILE_ROWS: usize = 64;
/// Baseline row tile once a graph has at least this many rows: fewer CTAs reread each
/// weight tile. Shorter prompt tails keep [`PROMPT_GEMM_TILE_ROWS`].
pub const WIDE_PROMPT_GEMM_TILE_ROWS: usize = 128;
/// MMA output tile of the dense and scaled pooled verification projections.
pub const SMALL_GEMM_TILE: [usize; 2] = [16, 32];
