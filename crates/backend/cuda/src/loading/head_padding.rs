//! Attention heads wider than any power of two the kernels tile, stored padded.
//!
//! cuTile tiles are powers of two, so a head width such as 96 has no attention, rotary or
//! cache kernel. Instead of a second set of kernels, the load stores each such head in the
//! next power of two: projection rows and columns for the padding lanes are zero, so the
//! padded lanes stay zero through rotary, attention and the output projection, and the real
//! lanes compute exactly what the checkpoint describes.
//!
//! Two quantities depend on the width rather than on the values, and are corrected here:
//! - A per-head `RMSNorm` averages over the padded width. With its epsilon scaled by `D / P`
//!   it returns the real norm times `sqrt(P / D)`.
//! - Attention scales scores by `1 / sqrt(P)`. Queries carry `sqrt(P / D)`, from the query
//!   norm or from scaled query rows, and a key norm's gain is scaled by `sqrt(D / P)`, so
//!   scores keep `1 / sqrt(D)`.
//!
//! Full-width rotary pads each half on its own, keeping the rotated pairs together, and its
//! frequency table carries the real frequencies. Partial rotary pads only the tail.
use infer_core::{Error, OpId, Result, TensorId};
use infer_ir::{DataflowGraph, TensorOp, TensorStorage};
use infer_models::DeviceWeight;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Where each padded lane of one head reads its real lane, `None` for a zero lane.
type Layout = Arc<[Option<usize>]>;

/// How one weight's checkpoint values map into its padded storage.
#[derive(Clone, Debug, PartialEq)]
pub enum Pad {
    /// Output rows, `parts` stacked head-wide blocks per head (the query, then its gate).
    Rows {
        heads: usize,
        parts: usize,
        real: usize,
        layout: Layout,
        scale: f32,
    },
    /// Input columns, one head-wide block per head.
    Columns {
        heads: usize,
        real: usize,
        layout: Layout,
    },
    /// A per-head norm gain `[real]`, applied as `gain + offset`.
    Gain {
        layout: Layout,
        scale: f32,
        offset: f32,
    },
}

/// Weights to pad and rotary tables to fill, for a graph whose heads were widened.
#[derive(Default, Debug)]
pub struct HeadPadding {
    pub weights: BTreeMap<TensorId, Pad>,
    /// Real rotary width of each widened rotary node, whose frequencies its table carries.
    pub rotary: BTreeMap<OpId, usize>,
}

impl Pad {
    /// Rows the padded matrix has for `rows` checkpoint rows, or columns likewise.
    fn widen(&self, count: usize) -> Result<usize> {
        let (heads, parts, real, stored) = match self {
            Self::Rows {
                heads,
                parts,
                real,
                layout,
                ..
            } => (*heads, *parts, *real, layout.len()),
            Self::Columns {
                heads,
                real,
                layout,
            } => (*heads, 1, *real, layout.len()),
            Self::Gain { layout, .. } => (1, 1, count, layout.len()),
        };
        if count != heads * parts * real {
            return Err(Error::invalid("padded weight does not match its heads"));
        }
        Ok(heads * parts * stored)
    }

    /// Pad a `[rows, columns]` matrix; `zero` fills the padding, and `scale` applies the
    /// query-row correction to real rows.
    pub fn apply<T: Copy>(
        &self,
        values: &[T],
        (rows, columns): (usize, usize),
        zero: T,
        scale: impl Fn(T, f32) -> T,
    ) -> Result<(Vec<T>, [usize; 2])> {
        if values.len() != rows * columns {
            return Err(Error::invalid("padded weight size"));
        }
        match self {
            Self::Rows {
                parts,
                real,
                layout,
                scale: factor,
                ..
            } => {
                let stored = self.widen(rows)?;
                let mut out = vec![zero; stored * columns];
                for (block, chunk) in out.chunks_mut(layout.len() * columns).enumerate() {
                    let source = block * real;
                    // Only a query block is scaled; its gate block keeps its values.
                    let factor = if block % parts == 0 { *factor } else { 1.0 };
                    for (lane, row) in chunk.chunks_mut(columns).enumerate() {
                        if let Some(real_lane) = layout[lane] {
                            let from = &values[(source + real_lane) * columns..][..columns];
                            for (to, &value) in row.iter_mut().zip(from) {
                                *to = scale(value, factor);
                            }
                        }
                    }
                }
                Ok((out, [stored, columns]))
            }
            Self::Columns { real, layout, .. } => {
                let stored = self.widen(columns)?;
                let mut out = vec![zero; rows * stored];
                for (row, to) in out.chunks_mut(stored).enumerate() {
                    let from = &values[row * columns..][..columns];
                    for (head, block) in to.chunks_mut(layout.len()).enumerate() {
                        for (lane, value) in block.iter_mut().enumerate() {
                            if let Some(real_lane) = layout[lane] {
                                *value = from[head * real + real_lane];
                            }
                        }
                    }
                }
                Ok((out, [rows, stored]))
            }
            Self::Gain { .. } => Err(Error::invariant("matrix padding for a gain")),
        }
    }

    /// A norm gain's padded values, scaled so `gain + offset` carries the correction.
    pub fn gain(&self, values: &[f32]) -> Result<Vec<f32>> {
        let Self::Gain {
            layout,
            scale,
            offset,
        } = self
        else {
            return Err(Error::invariant("gain padding for a matrix"));
        };
        if values.len() != layout.iter().flatten().count() {
            return Err(Error::invalid("padded gain size"));
        }
        Ok(layout
            .iter()
            .map(|lane| lane.map_or(0.0, |lane| (values[lane] + offset).mul_add(*scale, -offset)))
            .collect())
    }
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "Bounded rotary dimensions are intentionally rounded to the F32 device frequency contract"
)]
/// Rotary frequencies for a `half`-lane table: the first `real / 2` lanes rotate as a
/// `real`-wide rotary does, padding lanes do not rotate.
pub fn rotary_frequencies(half: usize, real: usize, theta: f64) -> Vec<f32> {
    (0..half)
        .map(|lane| {
            if lane < real / 2 {
                theta.powf(-((2 * lane) as f64) / real as f64) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// `D / P` for a head of `real` lanes stored in `stored`.
#[expect(
    clippy::cast_precision_loss,
    reason = "head widths are far below the f32 mantissa"
)]
fn width_ratio(real: usize, stored: usize) -> f32 {
    real as f32 / stored as f32
}

/// Lanes of one head: real lanes in order, each half followed by its padding when `halves`.
fn layout(real: usize, stored: usize, halves: bool) -> Layout {
    if halves {
        let (real, stored) = (real / 2, stored / 2);
        (0..2)
            .flat_map(|half| {
                (0..stored).map(move |lane| (lane < real).then_some(half * real + lane))
            })
            .collect()
    } else {
        (0..stored)
            .map(|lane| (lane < real).then_some(lane))
            .collect()
    }
}

/// Which attention input a chain feeds.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Query,
    Key,
    Value,
}

/// One attention node's head chain, found from the attention node outwards.
struct Chain {
    /// Nodes between each attention input and its projection, rotary and norm included.
    nodes: Vec<usize>,
    /// Activation tensors whose last dimension holds the heads.
    tensors: BTreeSet<TensorId>,
    rotary: Option<(usize, usize)>,
}

/// Widen every attention whose head width no kernel tiles, in place, and say how to pad the
/// weights it reads. Refuses chains the padding cannot preserve: speculative drafts,
/// multimodal rotary and weight encodings without an exact padded form.
/// # Errors
/// Returns `Unsupported` for a geometry this load cannot pad, `Invalid` for a malformed chain.
pub fn pad_heads(
    graph: &mut DataflowGraph,
    weights: &mut BTreeMap<String, DeviceWeight>,
    drafted: bool,
    multimodal_rotary: bool,
) -> Result<HeadPadding> {
    let mut padding = HeadPadding::default();
    let narrow: Vec<usize> = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            matches!(node.op, TensorOp::Attention { head_dim, .. } if !head_dim.is_power_of_two())
        })
        .map(|(index, _)| index)
        .collect();
    if narrow.is_empty() {
        return Ok(padding);
    }
    if drafted || multimodal_rotary {
        return Err(Error::unsupported(
            "padded CUDA attention heads support neither speculative drafts nor multimodal rotary",
        ));
    }
    let producer: BTreeMap<TensorId, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(index, node)| node.outputs.iter().map(move |&id| (id, index)))
        .collect();
    for attention in narrow {
        widen(graph, weights, &producer, attention, &mut padding)?;
    }
    graph.plan_lifetimes()?;
    Ok(padding)
}

#[expect(
    clippy::too_many_lines,
    reason = "one pass rewrites a head chain's nodes, tensors and weights together"
)]
fn widen(
    graph: &mut DataflowGraph,
    weights: &mut BTreeMap<String, DeviceWeight>,
    producer: &BTreeMap<TensorId, usize>,
    attention: usize,
    padding: &mut HeadPadding,
) -> Result<()> {
    let node = &graph.nodes[attention];
    let TensorOp::Attention {
        query_heads,
        kv_heads,
        head_dim: real,
        window,
    } = node.op
    else {
        return Err(Error::invariant("attention node"));
    };
    let (query, key, value) = match node.inputs.as_slice() {
        &[query, key, value] => (query, key, value),
        _ => return Err(Error::invalid("attention binds query, key and value")),
    };
    let output = node.outputs[0];
    let state = node.states[0];
    let queries = chain(graph, producer, query, real)?;
    let keys = chain(graph, producer, key, real)?;
    let values = chain(graph, producer, value, real)?;
    let width = |chain: &Chain| chain.rotary.map(|(_, width)| width);
    if values.rotary.is_some() || width(&queries) != width(&keys) {
        return Err(Error::unsupported(
            "padded heads need matching query and key rotary",
        ));
    }
    let (stored, halves) = match queries.rotary {
        Some((_, rotary)) if rotary == real => {
            if !real.is_multiple_of(2) {
                return Err(Error::unsupported("padded rotary heads must be even"));
            }
            let half = (real / 2)
                .checked_next_power_of_two()
                .ok_or_else(|| Error::invalid("padded head overflow"))?;
            (half * 2, true)
        }
        Some((_, rotary)) if (rotary / 2).is_power_of_two() => (
            real.checked_next_power_of_two()
                .ok_or_else(|| Error::invalid("padded head overflow"))?,
            false,
        ),
        Some(_) => {
            return Err(Error::unsupported(
                "padded heads need full rotary or a power-of-two rotary half",
            ));
        }
        None => (
            real.checked_next_power_of_two()
                .ok_or_else(|| Error::invalid("padded head overflow"))?,
            false,
        ),
    };
    let lanes = layout(real, stored, halves);
    let ratio = width_ratio(real, stored);
    // The attention output, an output gate, and the projection they reach.
    let consumers = |tensor: TensorId| -> Vec<usize> {
        graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.inputs.contains(&tensor))
            .map(|(index, _)| index)
            .collect()
    };
    let mut tensors: BTreeSet<TensorId> = [output].into();
    let mut nodes: Vec<usize> = Vec::new();
    let mut frontier = vec![output];
    let mut outputs = Vec::new();
    while let Some(tensor) = frontier.pop() {
        for index in consumers(tensor) {
            let node = &graph.nodes[index];
            match node.op {
                TensorOp::Linear if node.inputs[0] == tensor => outputs.push(index),
                TensorOp::Multiply | TensorOp::Sigmoid => {
                    nodes.push(index);
                    for &next in &node.outputs {
                        if tensors.insert(next) {
                            frontier.push(next);
                        }
                    }
                }
                _ => {
                    return Err(Error::unsupported(
                        "padded heads reach an unsupported consumer",
                    ));
                }
            }
        }
    }
    // The multiply's other operand is the gate after its sigmoid.
    for &index in &nodes {
        tensors.extend(graph.nodes[index].inputs.iter().copied());
    }
    let normed = |chain: &Chain| {
        chain
            .nodes
            .iter()
            .any(|&index| matches!(graph.nodes[index].op, TensorOp::Norm { .. }))
    };
    let query_norm = normed(&queries);
    if query_norm != normed(&keys) {
        return Err(Error::unsupported(
            "padded heads need norms on both queries and keys or on neither",
        ));
    }
    let gated = queries
        .nodes
        .iter()
        .any(|&index| matches!(graph.nodes[index].op, TensorOp::Split { .. }));
    let mut pads: Vec<(TensorId, Pad)> = Vec::new();
    for (chain, heads, role) in [
        (&queries, query_heads, Role::Query),
        (&keys, kv_heads, Role::Key),
        (&values, kv_heads, Role::Value),
    ] {
        for &index in &chain.nodes {
            let node = &mut graph.nodes[index];
            let weight = node.inputs.get(1).copied();
            match &mut node.op {
                TensorOp::Linear => {
                    let query = role == Role::Query;
                    // Without a query norm the query rows carry the score correction.
                    let scale = if query && !query_norm {
                        ratio.recip().sqrt()
                    } else {
                        1.0
                    };
                    pads.push((
                        weight.ok_or_else(|| Error::invalid("projection weight"))?,
                        Pad::Rows {
                            heads,
                            parts: if query && gated { 2 } else { 1 },
                            real,
                            layout: lanes.clone(),
                            scale,
                        },
                    ));
                }
                TensorOp::Norm {
                    epsilon,
                    offset,
                    head_dim,
                } => {
                    if *head_dim != real {
                        return Err(Error::unsupported("padded heads reach a non-head norm"));
                    }
                    *head_dim = stored;
                    *epsilon *= ratio;
                    // Key norms cancel the query's extra `sqrt(P / D)` in every score.
                    let scale = if role == Role::Key { ratio.sqrt() } else { 1.0 };
                    pads.push((
                        weight.ok_or_else(|| Error::invalid("norm weight"))?,
                        Pad::Gain {
                            layout: lanes.clone(),
                            scale,
                            offset: *offset,
                        },
                    ));
                }
                TensorOp::Split { widths, .. } => {
                    if widths.iter().any(|&width| width != real) {
                        return Err(Error::unsupported("padded heads reach an uneven split"));
                    }
                    widths.fill(stored);
                    // The gate output widens with the query.
                    tensors.extend(node.outputs.iter().copied());
                }
                TensorOp::Rope {
                    head_dim,
                    rotary_dim,
                    ..
                } => {
                    if halves {
                        padding.rotary.insert(node.id, *rotary_dim);
                        *rotary_dim = stored;
                    }
                    *head_dim = stored;
                }
                _ => return Err(Error::invariant("padded head chain node")),
            }
        }
        tensors.extend(chain.tensors.iter().copied());
    }
    for index in outputs {
        let weight = graph.nodes[index].inputs[1];
        pads.push((
            weight,
            Pad::Columns {
                heads: query_heads,
                real,
                layout: lanes.clone(),
            },
        ));
    }
    graph.nodes[attention].op = TensorOp::Attention {
        query_heads,
        kv_heads,
        head_dim: stored,
        window,
    };
    for spec in &mut graph.tensors {
        if tensors.contains(&spec.id) || spec.id == state {
            let last = spec
                .shape
                .last_mut()
                .ok_or_else(|| Error::invalid("padded head tensor rank"))?;
            if !last.is_multiple_of(real) {
                return Err(Error::invalid("padded head tensor width"));
            }
            *last = *last / real * stored;
        }
    }
    for (weight, pad) in pads {
        let spec = graph
            .tensors
            .iter_mut()
            .find(|spec| spec.id == weight)
            .ok_or_else(|| Error::invalid("padded weight tensor"))?;
        let TensorStorage::Weight { slot } = &spec.storage else {
            return Err(Error::invalid("padded weight is not a weight"));
        };
        let source = weights
            .get_mut(slot)
            .ok_or_else(|| Error::invalid("padded weight source"))?;
        let shape = match (&pad, spec.shape.as_slice()) {
            (Pad::Rows { .. }, &[rows, columns]) => vec![pad.widen(rows)?, columns],
            (Pad::Columns { .. }, &[rows, columns]) => vec![rows, pad.widen(columns)?],
            (Pad::Gain { .. }, &[width]) => vec![pad.widen(width)?],
            _ => return Err(Error::invalid("padded weight shape")),
        };
        spec.shape.clone_from(&shape);
        source.shape = shape;
        if padding.weights.insert(weight, pad).is_some() {
            return Err(Error::unsupported("a weight shared between padded heads"));
        }
    }
    Ok(())
}

/// Walk from one attention input back to its projection through rotary, norm and split.
fn chain(
    graph: &DataflowGraph,
    producer: &BTreeMap<TensorId, usize>,
    input: TensorId,
    real: usize,
) -> Result<Chain> {
    let mut chain = Chain {
        nodes: Vec::new(),
        tensors: BTreeSet::new(),
        rotary: None,
    };
    let mut tensor = input;
    loop {
        chain.tensors.insert(tensor);
        let index = *producer
            .get(&tensor)
            .ok_or_else(|| Error::unsupported("padded heads need a projected input"))?;
        let node = &graph.nodes[index];
        chain.nodes.push(index);
        match node.op {
            TensorOp::Linear => return Ok(chain),
            TensorOp::Rope {
                head_dim,
                rotary_dim,
                ..
            } if head_dim == real => {
                chain.rotary = Some((index, rotary_dim));
                tensor = node.inputs[0];
            }
            TensorOp::Norm { .. } | TensorOp::Split { .. } => tensor = node.inputs[0],
            _ => {
                return Err(Error::unsupported(
                    "padded heads reach an unsupported producer",
                ));
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/head_padding.rs"]
mod tests;
