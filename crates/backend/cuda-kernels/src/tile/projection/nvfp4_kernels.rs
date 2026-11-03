//! Pack unchanged positive E4M3 bits into the SDK scale layout.
use crate::contract::plan::Fp4ScalePlan;
use infer_core::{Error, Result};

/// Elements one FP4 block scale covers.
const GROUP: usize = 16;
/// Rows within one SDK scale block.
const BLOCK_ROWS: usize = 128;
/// K groups within one SDK scale block.
const BLOCK_GROUPS: usize = 4;
/// Bytes one swizzle tile writes, one SDK scale block.
pub(in crate::tile) const BLOCK_BYTES: usize = BLOCK_ROWS * BLOCK_GROUPS;

/// How a scale swizzle launches: its generics and grid.
pub(in crate::tile) struct Swizzle {
    pub generics: Vec<String>,
    pub grid: (u32, u32, u32),
    /// Groups of one scale row.
    pub groups: usize,
    /// Bytes of one destination row.
    pub row: usize,
}

/// The launch of `plan`, when it tiles into whole SDK blocks.
///
/// # Errors
/// The plan leaves a partial block, or its grid exceeds the launch limits.
pub(in crate::tile) fn swizzle(plan: &Fp4ScalePlan) -> Result<Swizzle> {
    let Fp4ScalePlan {
        rows,
        outputs,
        columns,
    } = *plan;
    let groups = columns / GROUP;
    if rows == 0
        || outputs == 0
        || !rows.is_multiple_of(BLOCK_ROWS)
        || !outputs.is_multiple_of(BLOCK_ROWS)
        || !columns.is_multiple_of(GROUP * BLOCK_GROUPS)
    {
        return Err(Error::invalid("FP4 scale swizzle: partial SDK block"));
    }
    let blocks = |value: usize| {
        u32::try_from(value).map_err(|_| Error::invalid("FP4 scale swizzle: grid exceeds u32"))
    };
    Ok(Swizzle {
        generics: vec![outputs.to_string()],
        grid: (
            blocks((outputs + rows) / BLOCK_ROWS)?,
            blocks(groups / BLOCK_GROUPS)?,
            1,
        ),
        groups,
        row: groups * BLOCK_ROWS,
    })
}

#[cutile::module]
pub mod entries {
    use cutile::core::{
        Array_3, BroadcastScalar, Reshape, Shape_1, Shape_2, Shape_3, StoreTileAtCurrentBlock,
        Tensor_2, Tile_1, Tile_2, Tile_3, convert_tile, f8e4m3fn, get_tile_block_id, iota, permute,
    };
    #[cutile::entry()]
    pub fn pack_pair<const N: i32>(
        out: &mut Tensor<f8e4m3fn, { [1, 512] }>,
        weights: &Tensor<f8e4m3fn, { [-1, -1] }>,
        input: &Tensor<f8e4m3fn, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let raw: Tile<f8e4m3fn, { [128, 4] }> = if pid.0 < N / 128i32 {
            // Modulo preserves this branch's index and exposes its bound to launch validation.
            let values: Tile<f8e4m3fn, { [128, 4] }> = weights
                .partition(shape![128, 4])
                .load([pid.0 % (N / 128i32), pid.1]);
            values
        } else {
            let values: Tile<f8e4m3fn, { [128, 4] }> = input
                .partition(shape![128, 4])
                .load([pid.0 - N / 128i32, pid.1]);
            values
        };
        let tile: Tile<f8e4m3fn, { [4, 32, 4] }> = raw.reshape(shape![4, 32, 4]);
        let packed: Tile<f8e4m3fn, { [32, 4, 4] }> = permute(tile, const_array![1, 0, 2]);
        out.store(packed.reshape(shape![1, 512]));
    }
    /// Deterministic signed probe; the product slot is unused after its codes are prepared.
    #[cutile::entry()]
    pub fn probe_input<const K: i32>(out: &mut Tensor<f32, { [16, 512] }>) {
        let pid = get_tile_block_id();
        let column: Tile<i32, { [512] }> = iota(shape![512]);
        let row: Tile<i32, { [16] }> = iota(shape![16]);
        let r = (row + (pid.0 * 16i32).broadcast(shape![16]))
            .reshape(shape![16, 1])
            .broadcast(shape![16, 512]);
        let c = (column + (pid.1 * 512i32).broadcast(shape![512]))
            .reshape(shape![1, 512])
            .broadcast(shape![16, 512]);
        let index = (r * K.broadcast(shape![16, 512]) + c) * 17i32.broadcast(shape![16, 512])
            % 131i32.broadcast(shape![16, 512]);
        let values: Tile<f32, { [16, 512] }> = convert_tile(index);
        out.store(
            (values - 65.0f32.broadcast(shape![16, 512])) / 37.0f32.broadcast(shape![16, 512]),
        );
    }
}
