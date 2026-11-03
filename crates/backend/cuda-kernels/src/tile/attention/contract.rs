//! cuTile implementation of the attention family.
use super::{attention, pooled_device, resident_attention_decode, resident_attention_prefill};
use crate::contract::attention::{
    AppendBuffers, Attention, DecodeBuffers, DecodePlan, GROUPED_ROWS, KvBuffers, KvCache,
    LanesAppendPlan, LanesDecodePlan, Partials, PooledAppendBuffers, PooledDecodeBuffers,
    PooledPlan, PrefillAppendPlan, PrefillDecodePlan, STACKED_LANE_ROWS, STACKED_ROWS,
    SplitDecodePlan, VERIFY_LANES, VerifyDecodeBuffers,
};
use crate::contract::buffer::Buffer;
use crate::contract::recording::Recording;
use crate::tile::projection::retyped_fp8;
use crate::tile::{backend_error, borrowed, capture_scope};
use cuda_core::DType;
use cutile::prelude::*;
use cutile::tile_kernel::CompileOptions;
use infer_core::Result;
use std::slice::from_ref;

/// Compiler occupancy hint of the grouped decode kernels. Without it the compiler spends the whole
/// register file on one CTA per SM, so the 1.5-wave split grid runs in two waves.
pub const GROUPED_OCCUPANCY: i32 = 2;

/// Two CTAs per SM: the measured optimum of both grouped kernels on RTX 5090 at 32K tokens
/// (the stacked three-lane kernel 150 → 59 µs, the one-lane kernel 53 → 46 µs per layer).
#[must_use]
pub fn grouped_options() -> CompileOptions {
    CompileOptions::default().occupancy(GROUPED_OCCUPANCY)
}

/// Attention kernels compiled with cuTile.
pub struct TileAttention {
    ordinal: usize,
}

impl TileAttention {
    /// Build the implementation for one device ordinal.
    #[must_use]
    pub const fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }

    fn tensor<T: DType>(
        &self,
        recording: &Recording<'_>,
        buffer: Buffer<'_, T>,
        name: &str,
        shape: &[usize],
    ) -> Result<Tensor<T>> {
        recording.retain(buffer);
        borrowed(
            buffer.require(name, shape.iter().product())?,
            self.ordinal,
            shape,
        )
    }

    fn flat<T: DType>(
        &self,
        recording: &Recording<'_>,
        buffer: Buffer<'_, T>,
    ) -> Result<Tensor<T>> {
        recording.retain(buffer);
        borrowed(buffer, self.ordinal, &[buffer.len()])
    }

    fn pair<T: DType>(
        &self,
        recording: &Recording<'_>,
        (keys, values): (Buffer<'_, T>, Buffer<'_, T>),
        cache: KvCache<'_>,
    ) -> Result<Kv<T>> {
        let rows = keys.len() / (cache.groups * cache.dim);
        let shape = [cache.groups, rows, cache.dim];
        Ok(Kv {
            keys: self.tensor(recording, keys, "attention keys", &shape)?,
            values: self.tensor(recording, values, "attention values", &shape)?,
            groups: cache.groups,
            rows,
            dim: cache.dim,
            scales: cache.scales,
        })
    }

    fn partials(
        &self,
        recording: &Recording<'_>,
        partials: Partials<'_>,
        shape: (&[usize], &[usize]),
    ) -> Result<PartialTensors> {
        Ok(PartialTensors {
            numerator: self.tensor(recording, partials.numerator, "numerator", shape.0)?,
            maxima: self.tensor(recording, partials.maxima, "maxima", shape.1)?,
            sums: self.tensor(recording, partials.sums, "sums", shape.1)?,
        })
    }
}

/// A cache as one launch binds it.
struct Kv<E: DType> {
    keys: Tensor<E>,
    values: Tensor<E>,
    groups: usize,
    rows: usize,
    dim: usize,
    scales: [f32; 2],
}

struct PartialTensors {
    numerator: Tensor<f32>,
    maxima: Tensor<f32>,
    sums: Tensor<f32>,
}

/// Call a generic launch with the cache typed for cuTile.
macro_rules! typed {
    ($self:ident, $recording:ident, $cache:expr, $launch:ident($($argument:expr),*)) => {
        typed!(@kv $self, $recording, $cache, $launch, kv, kv, $($argument),*)
    };
    (read $self:ident, $recording:ident, $cache:expr, $launch:ident($($argument:expr),*)) => {
        typed!(@kv $self, $recording, $cache, $launch, kv, &kv, $($argument),*)
    };
    (@kv $self:ident, $recording:ident, $cache:expr, $launch:ident, $kv:ident, $bound:expr,
     $($argument:expr),*) => {
        match $cache.buffers {
            KvBuffers::F32 { keys, values } => {
                let $kv = $self.pair($recording, (keys, values), $cache)?;
                $self.$launch($recording, $bound, $($argument),*)
            }
            KvBuffers::Fp8 { keys, values } => {
                let $kv = $self.pair($recording, (retyped_fp8(keys), retyped_fp8(values)), $cache)?;
                $self.$launch($recording, $bound, $($argument),*)
            }
        }
    };
}

fn quantized<E: DType>() -> i32 {
    i32::from(E::DTYPE != f32::DTYPE)
}

const fn query_group(plan: DecodePlan) -> usize {
    plan.heads / plan.kv_heads
}

impl TileAttention {
    fn append_one<E: DType>(
        &self,
        recording: &Recording<'_>,
        mut kv: Kv<E>,
        buffers: AppendBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let input = [kv.groups, kv.dim];
        let keys = self.tensor(recording, buffers.keys, "append keys", &input)?;
        let values = self.tensor(recording, buffers.values, "append values", &input)?;
        scope
            .record(
                attention::append(
                    (&mut kv.keys).partition([1, kv.rows, kv.dim]),
                    (&mut kv.values).partition([1, kv.rows, kv.dim]),
                    &keys,
                    &values,
                    &self.flat(recording, buffers.metadata)?,
                    &self.flat(recording, buffers.table)?,
                    kv.scales[0],
                    kv.scales[1],
                )
                .generics(vec![
                    E::DTYPE.as_str().into(),
                    kv.dim.to_string(),
                    kv.rows.to_string(),
                    quantized::<E>().to_string(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn append_chunk<E: DType>(
        &self,
        recording: &Recording<'_>,
        mut kv: Kv<E>,
        plan: PrefillAppendPlan,
        buffers: AppendBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let input = [plan.lanes, kv.groups, kv.dim];
        let keys = self.tensor(recording, buffers.keys, "append keys", &input)?;
        let values = self.tensor(recording, buffers.values, "append values", &input)?;
        let lookup = plan.lanes.div_ceil(plan.block).next_power_of_two();
        scope
            .record(
                resident_attention_prefill::append(
                    (&mut kv.keys).partition([1, plan.rows, kv.dim]),
                    (&mut kv.values).partition([1, plan.rows, kv.dim]),
                    &keys,
                    &values,
                    &self.flat(recording, buffers.metadata)?,
                    &self.flat(recording, buffers.table)?,
                    kv.scales[0],
                    kv.scales[1],
                )
                .generics(vec![
                    E::DTYPE.as_str().into(),
                    kv.dim.to_string(),
                    plan.rows.to_string(),
                    quantized::<E>().to_string(),
                    plan.block.to_string(),
                    lookup.to_string(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn append_pool<E: DType>(
        &self,
        recording: &Recording<'_>,
        mut kv: Kv<E>,
        plan: LanesAppendPlan,
        buffers: AppendBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let input = [plan.lanes * plan.kv_heads, kv.dim];
        let keys = self.tensor(recording, buffers.keys, "append keys", &input)?;
        let values = self.tensor(recording, buffers.values, "append values", &input)?;
        let table = self.tensor(
            recording,
            buffers.table,
            "page tables",
            &[plan.table_rows, buffers.table.len() / plan.table_rows],
        )?;
        scope
            .record(
                attention::append_lanes(
                    (&mut kv.keys).partition([1, kv.rows, kv.dim]),
                    (&mut kv.values).partition([1, kv.rows, kv.dim]),
                    &keys,
                    &values,
                    &self.flat(recording, buffers.metadata)?,
                    &table,
                    kv.scales[0],
                    kv.scales[1],
                )
                .generics(vec![
                    E::DTYPE.as_str().into(),
                    kv.dim.to_string(),
                    kv.rows.to_string(),
                    quantized::<E>().to_string(),
                    plan.kv_heads.to_string(),
                    plan.lanes.to_string(),
                    i32::from(plan.per_lane).to_string(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn decode_one<E: DType>(
        &self,
        recording: &Recording<'_>,
        kv: &Kv<E>,
        plan: DecodePlan,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let shape = [plan.heads, plan.dim];
        let mut output = self.tensor(recording, buffers.output, "attention output", &shape)?;
        let query = self.tensor(recording, buffers.query, "attention query", &shape)?;
        scope
            .record(
                attention::decode(
                    (&mut output).partition([1, plan.dim]),
                    &query,
                    &kv.keys,
                    &kv.values,
                    &self.flat(recording, buffers.metadata)?,
                    &self.flat(recording, buffers.table)?,
                    plan.window,
                    kv.scales[0],
                    kv.scales[1],
                )
                .generics(vec![
                    E::DTYPE.as_str().into(),
                    plan.dim.to_string(),
                    query_group(plan).to_string(),
                    plan.block.to_string(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn decode_chunk<E: DType>(
        &self,
        recording: &Recording<'_>,
        kv: &Kv<E>,
        plan: PrefillDecodePlan,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let PrefillDecodePlan { decode, lanes } = plan;
        let shape = [lanes * decode.heads, decode.dim];
        let mut output = self.tensor(recording, buffers.output, "attention output", &shape)?;
        let query = self.tensor(recording, buffers.query, "attention query", &shape)?;
        scope
            .record(
                resident_attention_prefill::decode(
                    (&mut output).partition([1, decode.dim]),
                    &query,
                    &kv.keys,
                    &kv.values,
                    &self.flat(recording, buffers.metadata)?,
                    &self.flat(recording, buffers.table)?,
                    decode.window,
                    kv.scales[0],
                    kv.scales[1],
                )
                .generics(vec![
                    E::DTYPE.as_str().into(),
                    decode.dim.to_string(),
                    query_group(decode).to_string(),
                    decode.heads.to_string(),
                    decode.block.to_string(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn decode_chunk_tiled<E: DType>(
        &self,
        recording: &Recording<'_>,
        kv: &Kv<E>,
        plan: PrefillDecodePlan,
        query_tile: usize,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let PrefillDecodePlan { decode, lanes } = plan;
        let shape = [lanes, decode.heads * decode.dim];
        let mut output = self.tensor(recording, buffers.output, "attention output", &shape)?;
        let query = self.tensor(recording, buffers.query, "attention query", &shape)?;
        let scale = 1.0f32 / f32::from(u16::try_from(decode.dim).map_err(backend_error)?).sqrt();
        scope
            .record(
                resident_attention_prefill::decode_tiled(
                    (&mut output).partition([query_tile, decode.dim]),
                    &query,
                    &kv.keys,
                    &kv.values,
                    &self.flat(recording, buffers.metadata)?,
                    &self.flat(recording, buffers.table)?,
                    decode.window,
                    kv.scales[0],
                    kv.scales[1],
                    scale,
                )
                .generics(vec![
                    E::DTYPE.as_str().into(),
                    decode.dim.to_string(),
                    query_group(decode).to_string(),
                    query_tile.to_string(),
                    decode.block.to_string(),
                    decode.block.to_string(),
                    "1".into(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn split_grouped<E: DType>(
        &self,
        recording: &Recording<'_>,
        kv: &Kv<E>,
        plan: SplitDecodePlan,
        groups: usize,
        partials: Partials<'_>,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let SplitDecodePlan { decode, parts, .. } = plan;
        let (dim, group) = (decode.dim, query_group(decode));
        let blocks = groups * parts;
        let mut partial = self.partials(
            recording,
            partials,
            (&[blocks, GROUPED_ROWS, dim], &[blocks, GROUPED_ROWS]),
        )?;
        let query = self.tensor(recording, buffers.query, "query", &[groups, group, dim])?;
        scope
            .record(
                resident_attention_decode::partial_gqa(
                    (&mut partial.numerator).partition([1, GROUPED_ROWS, dim]),
                    (&mut partial.maxima).partition([1, GROUPED_ROWS]),
                    (&mut partial.sums).partition([1, GROUPED_ROWS]),
                    &query,
                    &self.flat(recording, buffers.metadata)?,
                    &kv.keys,
                    &kv.values,
                    &self.flat(recording, buffers.table)?,
                    decode.window,
                    kv.scales[0],
                    kv.scales[1],
                )
                .generics(vec![
                    E::DTYPE.as_str().into(),
                    dim.to_string(),
                    parts.to_string(),
                    decode.block.to_string(),
                ])
                .compile_options(grouped_options()),
            )
            .map_err(backend_error)?;
        self.merge_rows(
            recording,
            buffers.output,
            (decode, parts, groups, GROUPED_ROWS, 0),
            &partial,
        )
    }

    /// Merge one lane's rows of grouped partials.
    fn merge_rows(
        &self,
        recording: &Recording<'_>,
        output: Buffer<'_, f32>,
        (decode, parts, groups, rows, offset): (DecodePlan, usize, usize, usize, i32),
        partial: &PartialTensors,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let dim = decode.dim;
        let mut output = self.tensor(
            recording,
            output,
            "attention output",
            &[output.len() / dim, dim],
        )?;
        let numerator = partial
            .numerator
            .view(&[groups, parts, rows, dim])
            .map_err(backend_error)?;
        let maxima = partial
            .maxima
            .view(&[groups, parts, rows])
            .map_err(backend_error)?;
        let sums = partial
            .sums
            .view(&[groups, parts, rows])
            .map_err(backend_error)?;
        scope
            .record(
                resident_attention_decode::merge_gqa(
                    (&mut output).partition([1, dim]),
                    &numerator,
                    &maxima,
                    &sums,
                    offset,
                )
                .generics(vec![
                    dim.to_string(),
                    parts.to_string(),
                    query_group(decode).to_string(),
                    rows.to_string(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn split_per_head<E: DType>(
        &self,
        recording: &Recording<'_>,
        kv: &Kv<E>,
        plan: SplitDecodePlan,
        partials: Partials<'_>,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let SplitDecodePlan { decode, parts, .. } = plan;
        let (heads, dim) = (decode.heads, decode.dim);
        let mut partial = self.partials(
            recording,
            partials,
            (&[heads * parts, dim], &[heads * parts, 1]),
        )?;
        let query = self.tensor(recording, buffers.query, "query", &[heads, dim])?;
        scope
            .record(
                resident_attention_decode::partial(
                    (&mut partial.numerator).partition([1, dim]),
                    (&mut partial.maxima).partition([1, 1]),
                    (&mut partial.sums).partition([1, 1]),
                    &query,
                    &kv.keys,
                    &kv.values,
                    &self.flat(recording, buffers.metadata)?,
                    &self.flat(recording, buffers.table)?,
                    decode.window,
                    kv.scales[0],
                    kv.scales[1],
                )
                .generics(vec![
                    E::DTYPE.as_str().into(),
                    dim.to_string(),
                    query_group(decode).to_string(),
                    parts.to_string(),
                    decode.block.to_string(),
                ]),
            )
            .map_err(backend_error)?;
        let mut output = self.tensor(recording, buffers.output, "output", &[heads, dim])?;
        let numerator = partial
            .numerator
            .view(&[heads, parts, dim])
            .map_err(backend_error)?;
        let maxima = partial
            .maxima
            .view(&[heads, parts])
            .map_err(backend_error)?;
        let sums = partial.sums.view(&[heads, parts]).map_err(backend_error)?;
        scope
            .record(
                resident_attention_decode::merge(
                    (&mut output).partition([1, dim]),
                    &numerator,
                    &maxima,
                    &sums,
                )
                .generics(vec![dim.to_string(), parts.to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn split<E: DType>(
        &self,
        recording: &Recording<'_>,
        kv: &Kv<E>,
        plan: SplitDecodePlan,
        partials: Partials<'_>,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()> {
        plan.groups.map_or_else(
            || self.split_per_head(recording, kv, plan, partials, buffers),
            |groups| self.split_grouped(recording, kv, plan, groups, partials, buffers),
        )
    }

    fn verify<E: DType>(
        &self,
        recording: &Recording<'_>,
        kv: &Kv<E>,
        plan: SplitDecodePlan,
        partials: Partials<'_>,
        buffers: &VerifyDecodeBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let SplitDecodePlan {
            decode,
            parts,
            groups,
        } = plan;
        let groups = groups.ok_or_else(|| backend_error("grouped verification workspace"))?;
        let (dim, group) = (decode.dim, query_group(decode));
        let blocks = groups * parts;
        let mut partial = self.partials(
            recording,
            partials,
            (&[blocks, STACKED_ROWS, dim], &[blocks, STACKED_ROWS]),
        )?;
        let query = |lane: usize| {
            self.tensor(
                recording,
                buffers.queries[lane],
                "query",
                &[groups, group, dim],
            )
        };
        let metadata = |lane: usize| self.flat(recording, buffers.metadata[lane]);
        scope
            .record(
                resident_attention_decode::partial_gqa3(
                    (&mut partial.numerator).partition([1, STACKED_ROWS, dim]),
                    (&mut partial.maxima).partition([1, STACKED_ROWS]),
                    (&mut partial.sums).partition([1, STACKED_ROWS]),
                    &query(0)?,
                    &query(1)?,
                    &query(2)?,
                    &metadata(0)?,
                    &metadata(1)?,
                    &metadata(2)?,
                    &kv.keys,
                    &kv.values,
                    &self.flat(recording, buffers.table)?,
                    decode.window,
                    kv.scales[0],
                    kv.scales[1],
                )
                .generics(vec![
                    E::DTYPE.as_str().into(),
                    dim.to_string(),
                    parts.to_string(),
                    decode.block.to_string(),
                ])
                .compile_options(grouped_options()),
            )
            .map_err(backend_error)?;
        for lane in 0..VERIFY_LANES {
            let offset = i32::try_from(lane * STACKED_LANE_ROWS).map_err(backend_error)?;
            self.merge_rows(
                recording,
                buffers.outputs[lane],
                (decode, parts, groups, STACKED_ROWS, offset),
                &partial,
            )?;
        }
        Ok(())
    }

    fn decode_pool<E: DType>(
        &self,
        recording: &Recording<'_>,
        kv: &Kv<E>,
        plan: LanesDecodePlan,
        partials: Partials<'_>,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let decode = plan.decode;
        let (dim, group) = (decode.dim, query_group(decode));
        let groups = plan.lanes * decode.kv_heads;
        let blocks = groups * plan.parts;
        let mut partial = self.partials(
            recording,
            partials,
            (&[blocks, GROUPED_ROWS, dim], &[blocks, GROUPED_ROWS]),
        )?;
        let query = self.tensor(recording, buffers.query, "query", &[groups, group, dim])?;
        let table = self.tensor(
            recording,
            buffers.table,
            "page tables",
            &[plan.table_rows, buffers.table.len() / plan.table_rows],
        )?;
        scope
            .record(
                resident_attention_decode::partial_gqa_lanes(
                    (&mut partial.numerator).partition([1, GROUPED_ROWS, dim]),
                    (&mut partial.maxima).partition([1, GROUPED_ROWS]),
                    (&mut partial.sums).partition([1, GROUPED_ROWS]),
                    &query,
                    &self.flat(recording, buffers.metadata)?,
                    &kv.keys,
                    &kv.values,
                    &table,
                    decode.window,
                    kv.scales[0],
                    kv.scales[1],
                )
                .generics(vec![
                    E::DTYPE.as_str().into(),
                    dim.to_string(),
                    plan.parts.to_string(),
                    decode.kv_heads.to_string(),
                    i32::from(plan.per_lane).to_string(),
                ])
                .compile_options(grouped_options()),
            )
            .map_err(backend_error)?;
        self.merge_rows(
            recording,
            buffers.output,
            (decode, plan.parts, groups, GROUPED_ROWS, 0),
            &partial,
        )
    }
}

impl Attention for TileAttention {
    fn append(&self, recording: &Recording<'_>, buffers: AppendBuffers<'_>) -> Result<()> {
        typed!(self, recording, buffers.cache, append_one(buffers))
    }

    fn append_prefill(
        &self,
        recording: &Recording<'_>,
        plan: PrefillAppendPlan,
        buffers: AppendBuffers<'_>,
    ) -> Result<()> {
        typed!(self, recording, buffers.cache, append_chunk(plan, buffers))
    }

    fn append_lanes(
        &self,
        recording: &Recording<'_>,
        plan: LanesAppendPlan,
        buffers: AppendBuffers<'_>,
    ) -> Result<()> {
        typed!(self, recording, buffers.cache, append_pool(plan, buffers))
    }

    fn decode(
        &self,
        recording: &Recording<'_>,
        plan: DecodePlan,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()> {
        typed!(read self, recording, buffers.cache, decode_one(plan, buffers))
    }

    fn decode_prefill(
        &self,
        recording: &Recording<'_>,
        plan: PrefillDecodePlan,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()> {
        typed!(read self, recording, buffers.cache, decode_chunk(plan, buffers))
    }

    fn decode_prefill_tiled(
        &self,
        recording: &Recording<'_>,
        plan: PrefillDecodePlan,
        query_tile: usize,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()> {
        typed!(
            read self,
            recording,
            buffers.cache,
            decode_chunk_tiled(plan, query_tile, buffers)
        )
    }

    fn decode_split(
        &self,
        recording: &Recording<'_>,
        plan: SplitDecodePlan,
        partials: Partials<'_>,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()> {
        typed!(read self, recording, buffers.cache, split(plan, partials, buffers))
    }

    fn decode_verify(
        &self,
        recording: &Recording<'_>,
        plan: SplitDecodePlan,
        partials: Partials<'_>,
        buffers: VerifyDecodeBuffers<'_>,
    ) -> Result<()> {
        typed!(read self, recording, buffers.cache, verify(plan, partials, &buffers))
    }

    fn decode_lanes(
        &self,
        recording: &Recording<'_>,
        plan: LanesDecodePlan,
        partials: Partials<'_>,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()> {
        typed!(read self, recording, buffers.cache, decode_pool(plan, partials, buffers))
    }

    fn append_pooled(
        &self,
        recording: &Recording<'_>,
        plan: PooledPlan,
        buffers: PooledAppendBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let (dim, kv_heads) = (plan.dim, plan.kv_heads);
        let lanes = plan.batch * VERIFY_LANES;
        let first_lane = plan.first * VERIFY_LANES;
        let control_rows = buffers.control.len() / plan.columns;
        let mut control = self.tensor(
            recording,
            buffers.control,
            "control",
            &[control_rows, plan.columns],
        )?;
        let metadata = self.flat(recording, buffers.metadata)?;
        let words = first_lane * plan.fields..(first_lane + lanes) * plan.fields;
        let fragment = metadata.slice(from_ref(&words)).map_err(backend_error)?;
        let [first, second, third, fourth] =
            buffers.tables.map(|table| self.flat(recording, table));
        scope
            .record(
                pooled_device::pack_control(
                    (&mut control).partition([1, plan.columns]),
                    &first?,
                    &second?,
                    &third?,
                    &fourth?,
                    &self.flat(recording, buffers.header)?,
                    &fragment,
                )
                .generics(vec![plan.columns.to_string()]),
            )
            .map_err(backend_error)?;
        let physical = buffers.keys.len() / (kv_heads * dim);
        let cache = [kv_heads, physical, dim];
        let mut keys = self.tensor(recording, retyped_fp8(buffers.keys), "keys", &cache)?;
        let mut values = self.tensor(recording, retyped_fp8(buffers.values), "values", &cache)?;
        let range = first_lane * kv_heads * dim..(first_lane + lanes) * kv_heads * dim;
        let rows = [lanes, kv_heads, dim];
        let key_rows = self.flat(recording, buffers.key_rows)?;
        let value_rows = self.flat(recording, buffers.value_rows)?;
        let key_rows = key_rows.slice(from_ref(&range)).map_err(backend_error)?;
        let value_rows = value_rows.slice(from_ref(&range)).map_err(backend_error)?;
        scope
            .record(
                pooled_device::append_pages(
                    (&mut keys).partition([1, physical, dim]),
                    (&mut values).partition([1, physical, dim]),
                    &key_rows.view(&rows).map_err(backend_error)?,
                    &value_rows.view(&rows).map_err(backend_error)?,
                    &control,
                )
                .generics(
                    [dim, physical, kv_heads, lanes, plan.fields]
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                ),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn decode_pooled(
        &self,
        recording: &Recording<'_>,
        plan: PooledPlan,
        buffers: PooledDecodeBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let (dim, kv_heads, parts) = (plan.dim, plan.kv_heads, plan.parts);
        let group = plan.heads / kv_heads;
        let groups = plan.full_batch * kv_heads;
        let blocks = groups * parts;
        let mut partial = self.partials(
            recording,
            buffers.partials,
            (&[blocks, STACKED_ROWS, dim], &[blocks, STACKED_ROWS]),
        )?;
        let lanes = plan.batch * VERIFY_LANES;
        let first_row = plan.first * VERIFY_LANES * plan.heads * dim;
        let range = first_row..first_row + lanes * plan.heads * dim;
        let queries = self.flat(recording, buffers.queries)?;
        let queries = queries.slice(from_ref(&range)).map_err(backend_error)?;
        let physical = buffers.keys.len() / (kv_heads * dim);
        let cache = [kv_heads, physical, dim];
        let keys = self.tensor(recording, retyped_fp8(buffers.keys), "keys", &cache)?;
        let values = self.tensor(recording, retyped_fp8(buffers.values), "values", &cache)?;
        let control_rows = buffers.control.len() / plan.columns;
        let control = self.tensor(
            recording,
            buffers.control,
            "control",
            &[control_rows, plan.columns],
        )?;
        scope
            .record(
                pooled_device::partial_slots(
                    (&mut partial.numerator).partition_prefix([1, STACKED_ROWS, dim]),
                    (&mut partial.maxima).partition_prefix([1, STACKED_ROWS]),
                    (&mut partial.sums).partition_prefix([1, STACKED_ROWS]),
                    &queries
                        .view(&[lanes * kv_heads, group, dim])
                        .map_err(backend_error)?,
                    &keys,
                    &values,
                    &control,
                )
                .generics(
                    [dim, kv_heads, parts, plan.fields]
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                )
                .compile_options(grouped_options())
                .grid((
                    u32::try_from(plan.batch * kv_heads * parts).map_err(backend_error)?,
                    1,
                    1,
                )),
            )
            .map_err(backend_error)?;
        let mut output = self.tensor(
            recording,
            buffers.output,
            "output",
            &[buffers.output.len() / dim, dim],
        )?;
        let numerator = partial
            .numerator
            .view(&[groups, parts, STACKED_ROWS, dim])
            .map_err(backend_error)?;
        let maxima = partial
            .maxima
            .view(&[groups, parts, STACKED_ROWS])
            .map_err(backend_error)?;
        let sums = partial
            .sums
            .view(&[groups, parts, STACKED_ROWS])
            .map_err(backend_error)?;
        scope
            .record(
                pooled_device::merge_slots(
                    (&mut output).partition([1, dim]),
                    &numerator,
                    &maxima,
                    &sums,
                    i32::try_from(plan.first).map_err(backend_error)?,
                    i32::try_from(plan.batch).map_err(backend_error)?,
                )
                .generics(
                    [dim, parts, group, plan.heads]
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                ),
            )
            .map_err(backend_error)?;
        Ok(())
    }
}
