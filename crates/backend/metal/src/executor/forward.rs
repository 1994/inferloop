use super::{MetalBackend, Sequence, u32_size};
use crate::constants::INPUT_BINDING_COUNT;
use crate::{device::Bindings, device::MetalDevice, device::Params, registry::shader};
use infer_core::{Error, Result};
use infer_ir::{ExecutionProgram, ExecutionTask, OpTrace, StepPlan, TensorNode, TensorOp};
use infer_models::TensorDtype;
use metal::Buffer;
use std::array::from_fn;
use std::{ops::Range, time::Instant};

pub(super) struct ForwardNode {
    inputs: [Buffer; INPUT_BINDING_COUNT],
    outputs: Vec<OutputDispatch>,
    state: Option<infer_core::TensorId>,
    params: Params,
    threads: usize,
    shader: &'static str,
    sequential: bool,
    head: bool,
}
struct OutputDispatch {
    buffer: Buffer,
    n: u32,
    split_offset: u32,
    split_width: u32,
}

impl MetalBackend {
    pub(super) fn encode_chunk(
        &mut self,
        command: &metal::CommandBufferRef,
        program: &ExecutionProgram,
        step: &StepPlan,
        task: &ExecutionTask,
        chunk: Range<usize>,
    ) -> Result<usize> {
        let logits = chunk.end == task.tokens.len()
            && task.tokens.readout() != infer_ir::OutputReadout::None
            || self.config.prefix_cache_bytes > 0
                && chunk.end.is_multiple_of(self.config.block_size);
        let mut dispatches = 0;
        for ((node, prepared), compiled) in self
            .graph
            .nodes
            .iter()
            .zip(&self.forward_nodes)
            .zip(&program.operations)
        {
            let s = &self.sequences[&task.state];
            let timestamp_ns = u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let start = Instant::now();
            let encoded = self.encode_node(command, prepared, s, chunk.clone(), logits)?;
            if encoded == 0 {
                continue;
            }
            dispatches += encoded;
            if let Some(probes) = &s.probes
                && let Some(layer) = self.layer_outputs.get(&node.id)
            {
                for (row, position) in chunk.clone().enumerate() {
                    MetalDevice::copy_range(
                        command,
                        &self.scratch[self.slots[&node.outputs[0]]],
                        row * self.model.hidden_size,
                        probes,
                        (position * self.model.mixers.len() + layer) * self.model.hidden_size,
                        self.model.hidden_size,
                    )?;
                }
            }
            if self.traces.len() == self.config.trace_capacity {
                self.traces.pop_front();
                self.trace_dropped += 1;
            }
            self.traces.push_back(OpTrace {
                request: task.request,
                step: step.id,
                state: task.state,
                op: node.id,
                kernel: compiled.kernel,
                position: if prepared.head {
                    chunk.end - 1
                } else {
                    chunk.start
                },
                token_count: if prepared.head { 1 } else { chunk.len() },
                elapsed_ns: u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
                timestamp_ns,
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
                states: node.states.clone(),
            });
        }
        self.encode_readout(command, &self.sequences[&task.state], chunk, logits)?;
        Ok(dispatches)
    }

    pub(super) fn prepare_forward(&mut self) -> Result<()> {
        self.forward_nodes = self
            .graph
            .nodes
            .iter()
            .map(|node| {
                if node.inputs.len() > INPUT_BINDING_COUNT || node.states.len() > 1 {
                    return Err(Error::unsupported("Metal forward binding capacity"));
                }
                let head = self
                    .graph
                    .logits
                    .is_some_and(|id| node.outputs.contains(&id));
                let inputs = from_fn(|i| {
                    node.inputs.get(i).map_or_else(
                        || self.dummy.clone(),
                        |id| {
                            self.weights
                                .get(id)
                                .map_or_else(|| self.scratch[self.slots[id]].clone(), Clone::clone)
                        },
                    )
                });
                let mut weight_formats = 0;
                for (i, id) in node.inputs.iter().enumerate() {
                    let format = match self.weight_formats.get(id) {
                        None | Some(TensorDtype::F32) => 0,
                        Some(TensorDtype::BF16) => 1,
                        Some(TensorDtype::F16) => 2,
                        _ => return Err(Error::unsupported("Metal weight storage format")),
                    };
                    weight_formats |= format << (i * 2);
                }
                let base = Params {
                    n: u32_size(self.specs[&node.outputs[0]].elements()?)?,
                    rows: 1,
                    weight_formats,
                    ..Default::default()
                };
                let (mut params, threads) = self.node_params(node, base)?;
                let mut offset = 0;
                let outputs = node
                    .outputs
                    .iter()
                    .enumerate()
                    .map(|(i, id)| {
                        let width = if let TensorOp::Split { widths, .. } = &node.op {
                            widths[i]
                        } else {
                            0
                        };
                        let split_offset = u32_size(offset)?;
                        offset += width;
                        Ok(OutputDispatch {
                            buffer: self.scratch[self.slots[id]].clone(),
                            n: u32_size(self.specs[id].elements()?)?,
                            split_offset,
                            split_width: u32_size(width)?,
                        })
                    })
                    .collect::<Result<_>>()?;
                if matches!(node.op, TensorOp::Split { .. }) {
                    params.a = u32_size(offset)?;
                }
                Ok(ForwardNode {
                    inputs,
                    outputs,
                    state: node.states.first().copied(),
                    params,
                    threads,
                    shader: shader(node.op.operation(head)),
                    sequential: matches!(node.op, TensorOp::Conv { .. } | TensorOp::Delta { .. }),
                    head,
                })
            })
            .collect::<Result<_>>()?;
        Ok(())
    }

    pub(super) fn encode_node(
        &self,
        command: &metal::CommandBufferRef,
        node: &ForwardNode,
        s: &Sequence,
        chunk: Range<usize>,
        compute_logits: bool,
    ) -> Result<usize> {
        if node.head && !compute_logits {
            return Ok(0);
        }
        let inputs = node.inputs.each_ref().map(|b| &**b);
        let state = node.state.map_or(&*self.dummy, |id| {
            self.kv_buffers
                .get(&id)
                .map_or_else(|| &*s.tensors[&id], |b| &**b)
        });
        let mut p = Params {
            position: u32_size(chunk.start)?,
            capacity: u32_size(s.capacity)?,
            rows: u32_size(chunk.len())?,
            ..node.params
        };
        if node.head {
            p.start_row = p.rows - 1;
            p.rows = 1;
        }
        let mut dispatches = 0;
        if node.shader == "attention" {
            let append = Params {
                a: p.b
                    .checked_mul(p.c)
                    .ok_or_else(|| Error::invalid("KV width overflow"))?,
                ..p
            };
            self.gpu.encode(
                command,
                "kv_append",
                &Bindings {
                    inputs: &inputs,
                    state,
                    output: &self.dummy,
                    dummy: &self.dummy,
                    page_table: &s.page_table,
                    tokens: &s.token_buffer,
                },
                append,
                append.a as usize * append.rows as usize,
            )?;
            dispatches += 1;
        }
        for output in &node.outputs {
            p.n = output.n;
            if node.shader == "split" {
                p.b = output.split_offset;
                p.c = output.split_width;
            }
            let rows = if node.sequential { 1 } else { p.rows as usize };
            let threads = if node.shader == "split" {
                p.n as usize * rows
            } else {
                node.threads * rows
            };
            self.gpu.encode(
                command,
                node.shader,
                &Bindings {
                    inputs: &inputs,
                    state,
                    output: &output.buffer,
                    dummy: &self.dummy,
                    page_table: &s.page_table,
                    tokens: &s.token_buffer,
                },
                p,
                threads,
            )?;
            dispatches += 1;
        }
        Ok(dispatches)
    }

    pub(super) fn encode_readout(
        &self,
        command: &metal::CommandBufferRef,
        s: &Sequence,
        chunk: Range<usize>,
        logits: bool,
    ) -> Result<()> {
        let hidden = self
            .graph
            .hidden
            .ok_or_else(|| Error::invariant("compiled hidden"))?;
        let (source_offset, destination_offset, rows) =
            if s.readout == infer_ir::OutputReadout::Full {
                (0, chunk.start * self.model.hidden_size, chunk.len())
            } else {
                ((chunk.len() - 1) * self.model.hidden_size, 0, 1)
            };
        MetalDevice::copy_range(
            command,
            &self.scratch[self.slots[&hidden]],
            source_offset,
            &s.hidden,
            destination_offset,
            rows * self.model.hidden_size,
        )?;
        if logits {
            let id = self
                .graph
                .logits
                .ok_or_else(|| Error::invariant("compiled logits"))?;
            MetalDevice::copy(
                command,
                &self.scratch[self.slots[&id]],
                &s.logits,
                0,
                self.model.vocab_size,
            )?;
        }
        Ok(())
    }

    pub(super) fn next_chunk(&self, start: usize, end: usize) -> Range<usize> {
        let maximum = end.min(start.saturating_add(self.config.prefill_chunk_tokens));
        let next = if self.config.prefix_cache_bytes == 0 {
            maximum
        } else {
            maximum
                .min(start.saturating_add(self.config.block_size - start % self.config.block_size))
        };
        start..next
    }
}

impl MetalBackend {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
    )]
    fn node_params(&self, node: &TensorNode, mut p: Params) -> Result<(Params, usize)> {
        let mut threads = p.n as usize;
        match &node.op {
            TensorOp::MoeRoute { .. }
            | TensorOp::ExpertLinear { .. }
            | TensorOp::MoeCombine { .. }
            | TensorOp::Scale => {
                return Err(Error::unsupported(
                    "MoE currently requires the CUDA backend",
                ));
            }
            TensorOp::Linear => p.a = u32_size(self.specs[&node.inputs[0]].elements()?)?,
            TensorOp::Norm {
                epsilon,
                offset,
                head_dim,
            } => {
                p.a = u32_size(*head_dim)?;
                p.epsilon = *epsilon;
                p.offset = *offset;
                threads /= head_dim;
            }
            TensorOp::Rope {
                head_dim,
                rotary_dim,
                theta,
                ..
            } => {
                p.a = u32_size(*head_dim)?;
                p.b = u32_size(*rotary_dim)?;
                p.theta = *theta as f32;
            }
            TensorOp::Attention {
                query_heads,
                kv_heads,
                head_dim,
                window,
            } => {
                p.e = u32_size(self.config.block_size)?;
                p.f = u32_size(self.kv.capacity())?;
                p.a = u32_size(*query_heads)?;
                p.b = u32_size(*kv_heads)?;
                p.c = u32_size(*head_dim)?;
                p.d = u32_size(window.unwrap_or(0))?;
                threads = *query_heads;
            }
            TensorOp::Conv { channels, kernel } => {
                p.a = u32_size(*channels)?;
                p.b = u32_size(*kernel)?;
                threads = *channels;
            }
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
                value_order,
            } => {
                p.a = u32_size(*key_heads)?;
                p.b = u32_size(*value_heads)?;
                p.c = u32_size(*key_dim)?;
                p.d = u32_size(*value_dim)?;
                p.e = u32_size(value_order.key_run(*key_heads, *value_heads))?;
                threads = *value_heads;
            }
            TensorOp::GatedNorm { head_dim, epsilon } => {
                p.a = u32_size(*head_dim)?;
                p.epsilon = *epsilon;
                threads /= head_dim;
            }
            TensorOp::Sigmoid | TensorOp::Multiply => p.a = 1,
            TensorOp::MediaFeature { .. } => {
                return Err(Error::unsupported(
                    "media feature injection requires a backend with a vision encoder",
                ));
            }
            _ => {}
        }
        Ok((p, threads))
    }
}
