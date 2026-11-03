//! Scalar launch arguments update resident metadata without temporary device allocations.
use crate::tile::backend_error;
use crate::tile::runtime::{Graph, GraphUpdate};
use cuda_core::Stream;
use cutile::prelude::*;
use infer_core::Result;
use std::sync::Arc;

/// Fields of one program's control metadata.
const METADATA_FIELDS: usize = 4;
/// Axes of an independent rotary coordinate.
const ROTARY_AXES: usize = 3;

/// Opaque bit pattern through a float argument: cuTile 0.4 specializes integer scalars by
/// `DivHint`, causing runtime JIT as tokens change. No float arithmetic touches these arguments.
const fn bits(value: i32) -> f32 {
    f32::from_ne_bytes(value.to_ne_bytes())
}

/// Write control metadata before the graph's next replay, on the graph's stream.
/// # Errors
/// Reports a rejected launch.
pub fn update(
    graph: &Graph,
    output: &mut Tensor<i32>,
    values: [i32; METADATA_FIELDS],
) -> Result<()> {
    // Graph::update retains the output lease until replay completes; same-stream ordering replaces
    // an upload sync.
    graph
        .update(GraphUpdate(kernels::write(
            output.partition([METADATA_FIELDS]),
            bits(values[0]),
            bits(values[1]),
            bits(values[2]),
            bits(values[3]),
        )))
        .map_err(backend_error)
}

/// Write control metadata and independent rotary coordinates on the graph's stream.
/// # Errors
/// Reports a rejected launch or a coordinate beyond the kernel's range.
pub fn update_rotary(
    graph: &Graph,
    output: &mut Tensor<i32>,
    values: [i32; METADATA_FIELDS],
    position: [usize; ROTARY_AXES],
) -> Result<()> {
    let height = i32::try_from(position[1]).map_err(backend_error)?;
    let width = i32::try_from(position[2]).map_err(backend_error)?;
    graph
        .update(GraphUpdate(kernels::write_rotary(
            output.partition([METADATA_FIELDS * 2]),
            bits(values[0]),
            bits(values[1]),
            bits(values[2]),
            bits(values[3]),
            bits(height),
            bits(width),
        )))
        .map_err(backend_error)
}

#[cutile::module]
pub mod kernels {
    use cutile::core::{
        BroadcastScalar, Shape_1, StoreTileAtCurrentBlock, Tensor_1, Tile_1, bitcast, eq_tile,
        iota, select,
    };

    #[cutile::entry()]
    fn write(out: &mut Tensor<i32, { [4] }>, position: f32, token: f32, kv: f32, external: f32) {
        let offsets: Tile<i32, { [4] }> = iota(shape![4]);
        let position: Tile<i32, { [4] }> = bitcast(position.broadcast(shape![4]));
        let token: Tile<i32, { [4] }> = bitcast(token.broadcast(shape![4]));
        let kv: Tile<i32, { [4] }> = bitcast(kv.broadcast(shape![4]));
        let values: Tile<i32, { [4] }> = bitcast(external.broadcast(shape![4]));
        let values = select(eq_tile(offsets, 2i32.broadcast(shape![4])), kv, values);
        let values = select(eq_tile(offsets, 1i32.broadcast(shape![4])), token, values);
        let values = select(
            eq_tile(offsets, 0i32.broadcast(shape![4])),
            position,
            values,
        );
        out.store(values);
    }
    #[cutile::entry()]
    fn write_rotary(
        out: &mut Tensor<i32, { [8] }>,
        position: f32,
        token: f32,
        kv: f32,
        external: f32,
        height: f32,
        width: f32,
    ) {
        let offsets: Tile<i32, { [8] }> = iota(shape![8]);
        let position_values: Tile<i32, { [8] }> = bitcast(position.broadcast(shape![8]));
        let token_values: Tile<i32, { [8] }> = bitcast(token.broadcast(shape![8]));
        let kv_values: Tile<i32, { [8] }> = bitcast(kv.broadcast(shape![8]));
        let external_values: Tile<i32, { [8] }> = bitcast(external.broadcast(shape![8]));
        let height_values: Tile<i32, { [8] }> = bitcast(height.broadcast(shape![8]));
        let width_values: Tile<i32, { [8] }> = bitcast(width.broadcast(shape![8]));
        let result = select(
            eq_tile(offsets, 1i32.broadcast(shape![8])),
            token_values,
            position_values,
        );
        let result = select(
            eq_tile(offsets, 2i32.broadcast(shape![8])),
            kv_values,
            result,
        );
        let result = select(
            eq_tile(offsets, 3i32.broadcast(shape![8])),
            external_values,
            result,
        );
        let result = select(
            eq_tile(offsets, 5i32.broadcast(shape![8])),
            height_values,
            result,
        );
        let result = select(
            eq_tile(offsets, 6i32.broadcast(shape![8])),
            width_values,
            result,
        );
        out.store(result);
    }
}

/// Run the rotary metadata write once so its module is loaded before any submission: a module
/// load waits for the GPU, which would otherwise stall the first decode behind running work.
///
/// # Errors
/// Reports the driver's launch failure.
pub fn load_rotary(stream: &Arc<Stream>, output: &mut Tensor<i32>) -> Result<()> {
    kernels::write_rotary(
        output.partition([output.size()]),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    .sync_on(stream)
    .map(|_| ())
    .map_err(backend_error)
}
