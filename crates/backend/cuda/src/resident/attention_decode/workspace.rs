//! Capture-time split-KV planning and reusable scratch, independent of model identity.

use crate::constants::KV_BLOCK_TOKENS;
use crate::device::CudaDevice;
use crate::resident::buffers::buffer;
use crate::resident::capture::error;
use crate::tuning::attention::Choice;
use crate::tuning::attention::Shape;
use crate::tuning::attention::decode::{GROUPED_ROWS as GQA_ROWS, grouped_parts, partitions};
use crate::tuning::attention::enabled as attention_enabled;
use crate::tuning::attention::exact::select;
use crate::tuning::attention::mma::select as mma_select;
use crate::tuning::attention::mma_f32::select as mma_f32_select;
use crate::tuning::attention::select as attention_select;
use crate::tuning::attention::shared::select as shared_select;
use crate::tuning::attention::vector::select as vector_select;
use crate::tuning::attention::vector_combined::select as vector_combined_select;
use infer_core::{Result, TensorId};
use infer_cuda_kernels::contract::{
    Attention, DecodeBuffers, DecodePlan, Partials as ContractPartials, Recording, SplitDecodePlan,
    VerifyDecodeBuffers,
};
use infer_cuda_kernels::runtime::Allocation;
use infer_ir::{DataflowGraph, TensorOp};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::mem::swap;

#[cfg(test)]
#[path = "../../../tests/unit/resident_attention_prefill_taps.rs"]
pub mod taps;

pub use infer_cuda_kernels::contract::VERIFY_LANES;
pub(super) use infer_cuda_kernels::contract::{STACKED_LANE_ROWS, STACKED_ROWS};

struct Buffers {
    /// KV heads of the grouped kernel's partial layout, or `None` for one partial per query head.
    grouped: Option<usize>,
    parts: usize,
    numerator: Allocation<f32>,
    maxima: Allocation<f32>,
    sums: Allocation<f32>,
    /// Stacked partials of the three-lane verification kernel, when a KV group fits one lane's
    /// rows of the stacked query tile.
    verify: Option<Partials>,
}

/// One lane's grouped split partials.
struct Partials {
    numerator: Allocation<f32>,
    maxima: Allocation<f32>,
    sums: Allocation<f32>,
}

pub struct Workspace {
    oxide_available: bool,
    mma: Option<super::super::attention_mma::Workspace>,
    mma_f16: Option<super::super::attention_mma::Workspace>,
    pub(in crate::resident) oxide: Option<super::super::attention_oxide::Support>,
    pub(in crate::resident) shared: Option<super::super::attention_oxide::Support>,
    vector: Option<super::super::attention_oxide::Support>,
    combined: Option<super::super::attention_oxide::Support>,
    #[cfg(test)]
    pub taps: taps::Taps,
    buffers: BTreeMap<(usize, usize), Buffers>,
    pub(super) pooled: Option<super::pooled::Workspace>,
    prefill: BTreeMap<(TensorId, usize), Choice>,
    precise_prefill: BTreeMap<(TensorId, usize), Choice>,
    precise_active: bool,
}

impl Workspace {
    pub(crate) fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        _capacity: usize,
    ) -> Result<Self> {
        let mut buffers = BTreeMap::new();
        let sms = usize::try_from(device.profile()?.multiprocessors)
            .map_err(|_| infer_core::Error::invalid("CUDA multiprocessor count"))?;
        for node in &graph.nodes {
            let TensorOp::Attention {
                query_heads,
                kv_heads,
                head_dim,
                window,
            } = node.op
            else {
                continue;
            };
            if query_heads == 0 || head_dim == 0 {
                return Err(infer_core::Error::invalid(
                    "attention dimensions must be positive",
                ));
            }
            let parts = partitions(query_heads, window, sms);
            if parts == 1 || buffers.contains_key(&(query_heads, head_dim)) {
                continue;
            }
            let runtime = device.runtime();
            let zeros = |shape: &[usize]| runtime.zeros::<f32>(shape.iter().product());
            let entry = if let Some(grouped) = grouped_parts(query_heads, kv_heads, sms) {
                let blocks = kv_heads * grouped;
                let verify = if query_heads / kv_heads <= STACKED_LANE_ROWS {
                    Some(Partials {
                        numerator: zeros(&[blocks, STACKED_ROWS, head_dim])?,
                        maxima: zeros(&[blocks, STACKED_ROWS])?,
                        sums: zeros(&[blocks, STACKED_ROWS])?,
                    })
                } else {
                    None
                };
                Buffers {
                    grouped: Some(kv_heads),
                    parts: grouped,
                    numerator: zeros(&[blocks, GQA_ROWS, head_dim])?,
                    maxima: zeros(&[blocks, GQA_ROWS])?,
                    sums: zeros(&[blocks, GQA_ROWS])?,
                    verify,
                }
            } else {
                Buffers {
                    grouped: None,
                    verify: None,
                    parts,
                    numerator: zeros(&[query_heads * parts, head_dim])?,
                    maxima: zeros(&[query_heads * parts, 1])?,
                    sums: zeros(&[query_heads * parts, 1])?,
                }
            };
            buffers.insert((query_heads, head_dim), entry);
        }
        Ok(Self {
            #[cfg(test)]
            taps: taps::Taps::default(),
            oxide_available: super::super::attention_oxide::Support::eligible(device, graph),
            mma: None,
            mma_f16: None,
            oxide: None,
            shared: None,
            vector: None,
            combined: None,
            buffers,
            pooled: None,
            prefill: BTreeMap::new(),
            precise_prefill: BTreeMap::new(),
            precise_active: false,
        })
    }

    pub(crate) fn prepare_mma(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &super::super::ProgramWeights,
        capacity: usize,
        lanes: usize,
    ) -> Result<()> {
        #[cfg(test)]
        if super::super::attention_oxide::disabled() {
            return Ok(());
        }
        self.mma = None;
        self.mma_f16 = None;
        // FA4 gathers whole KV pages, but a private program's arena holds exactly the request's
        // capacity. An unaligned one (a short prompt before the pool exists) keeps the exact
        // attention paths, which bound reads by the arena instead of failing the request.
        if !capacity.is_multiple_of(KV_BLOCK_TOKENS) {
            return Ok(());
        }
        let architecture = device.profile()?.architecture.clone();
        let target = weights.fusion.is_none();
        let any_f16 = target
            && weights.attention_profile.is_fp16()
            && graph.nodes.iter().any(|node| {
                let TensorOp::Attention {
                    query_heads,
                    kv_heads,
                    head_dim,
                    window,
                } = node.op
                else {
                    return false;
                };
                node.states
                    .first()
                    .is_some_and(|id| weights.kv_scales.contains_key(id))
                    && super::super::attention_mma::f16::supported(
                        &architecture,
                        query_heads,
                        kv_heads,
                        head_dim,
                        lanes,
                        window.unwrap_or(0),
                    )
            });
        if any_f16 {
            match super::super::attention_mma::Workspace::new(
                device,
                graph,
                weights,
                capacity,
                lanes,
                super::super::attention_mma::Arithmetic::F16,
            )
            .and_then(|workspace| {
                workspace
                    .map(|mut workspace| {
                        workspace.support.prepare_variant(
                            device,
                            super::super::attention_mma::catalog::Variant::Sm120,
                        )?;
                        Ok(workspace)
                    })
                    .transpose()
            }) {
                Ok(Some(workspace)) => self.mma_f16 = Some(workspace),
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(target:"infer::tuning",%error,"FP16 target attention unavailable: canonical fallback");
                }
            }
        }
        if !any_f16 {
            self.mma = super::super::attention_mma::Workspace::new(
                device,
                graph,
                weights,
                capacity,
                lanes,
                super::super::attention_mma::Arithmetic::Bf16,
            )?;
        }
        Ok(())
    }

    pub(crate) fn prepare_prefill(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &super::super::ProgramWeights,
        lanes: usize,
    ) {
        let enabled = attention_enabled();
        #[cfg(test)]
        let enabled = enabled || taps::forced_choice().is_some();
        if !enabled && !self.oxide_available && self.mma.is_none() && self.mma_f16.is_none() {
            self.prefill.retain(|(_, width), _| *width != lanes);
            self.precise_prefill.retain(|(_, width), _| *width != lanes);
            return;
        }
        let architecture = device
            .profile()
            .map(|p| p.architecture.clone())
            .unwrap_or_default();
        let mut choices = BTreeMap::new();
        for node in &graph.nodes {
            let TensorOp::Attention {
                query_heads,
                kv_heads,
                head_dim,
                window,
            } = node.op
            else {
                continue;
            };
            let Some(state) = node.states.first() else {
                continue;
            };
            let fp8 = weights.kv_scales.contains_key(state);
            let f16_route = weights.fusion.is_none()
                && weights.attention_profile.is_fp16()
                && fp8
                && self.mma_f16.is_some()
                && super::super::attention_mma::f16::supported(
                    &architecture,
                    query_heads,
                    kv_heads,
                    head_dim,
                    lanes,
                    window.unwrap_or(0),
                );
            if f16_route {
                self.prefill
                    .insert((node.outputs[0], lanes), Choice::MmaF16);
                continue;
            }
            // The target experimental profiles are limited to FP8-packed KV. Only the draft
            // proposal program keeps its existing F32-KV staging path, so a target graph that
            // mixes FP8 and F32 attention cannot leak a profile into the F32 nodes.
            if !fp8 && (self.mma.is_none() || weights.fusion.is_none()) {
                continue;
            }
            let shape = Shape {
                heads: query_heads,
                kv_heads,
                dim: head_dim,
                lanes,
                window: window.unwrap_or(0),
            };
            let (choice, precise) = *choices.entry((shape, fp8)).or_insert_with(|| {
                if fp8 {
                    self.fp8_prefill_choices(device, shape, weights)
                } else {
                    let choice = mma_f32_select(device, shape, false, &weights.resident_tuning);
                    (choice, choice)
                }
            });
            #[cfg(test)]
            let (choice, precise) = (
                taps::forced_choice().unwrap_or(choice),
                taps::forced_choice().unwrap_or(precise),
            );
            self.prefill.insert((node.outputs[0], lanes), choice);
            self.precise_prefill
                .insert((node.outputs[0], lanes), precise);
        }
        self.prepare_mma_modules(device);
        self.prepare_modules(device);
        swap(&mut self.prefill, &mut self.precise_prefill);
        self.prepare_modules(device);
        swap(&mut self.prefill, &mut self.precise_prefill);
    }

    fn fp8_prefill_choices(
        &self,
        device: &CudaDevice,
        shape: Shape,
        weights: &super::super::ProgramWeights,
    ) -> (Choice, Choice) {
        let baseline = if self.oxide_available {
            let baseline = select(device, shape, weights.autotune, &weights.resident_tuning);
            shared_select(
                device,
                shape,
                weights.autotune,
                &weights.resident_tuning,
                baseline,
            )
        } else if attention_enabled() {
            attention_select(device, shape, weights.autotune, &weights.resident_tuning)
        } else {
            Choice::Simt
        };
        let baseline = vector_select(device, shape, false, &weights.resident_tuning, baseline);
        let baseline = vector_combined_select(device, shape, baseline);
        #[cfg(test)]
        if taps::forced_choice().is_some() {
            return (baseline, baseline);
        }
        let selected = if self.mma.is_some() {
            mma_select(device, shape, false, &weights.resident_tuning, baseline)
        } else {
            baseline
        };
        (selected, baseline)
    }

    pub(crate) const fn precise_prefill(&mut self, enabled: bool) {
        self.precise_active = enabled;
    }

    fn prefill_choice(&self, output: TensorId, lanes: usize) -> Option<&Choice> {
        if self.precise_active {
            self.precise_prefill.get(&(output, lanes))
        } else {
            self.prefill.get(&(output, lanes))
        }
    }

    fn prepare_mma_modules(&mut self, device: &CudaDevice) {
        if let Some(mma) = self.mma.as_mut() {
            let variants = self
                .prefill
                .values()
                .filter_map(|choice| {
                    if let Choice::Mma { variant } = choice {
                        Some(*variant)
                    } else {
                        None
                    }
                })
                .collect::<BTreeSet<_>>();
            for variant in variants {
                if let Err(error) = mma.support.prepare_variant(device, variant) {
                    tracing::warn!(target:"infer::tuning",%error,?variant,"MMA launch module unavailable: canonical fallback");
                    for choice in self.prefill.values_mut() {
                        if *choice == (Choice::Mma { variant }) {
                            *choice = if self.oxide_available {
                                Choice::RustSimt
                            } else {
                                Choice::Simt
                            };
                        }
                    }
                }
            }
        }
    }

    fn prepare_modules(&mut self, device: &CudaDevice) {
        self.prepare_oxide_module(device);
        self.prepare_shared_module(device);
        self.prepare_vector_module(device);
        self.prepare_combined_module(device);
    }

    fn prepare_oxide_module(&mut self, device: &CudaDevice) {
        if self.oxide.is_none() && self.prefill.values().any(|choice| choice.uses_oxide()) {
            match super::super::attention_oxide::Support::load(device) {
                Ok(Some(module)) => self.oxide = Some(module),
                result => {
                    tracing::warn!(target:"infer::tuning",error=?result.err(),"Rust attention module unavailable: cuTile fallback");
                    self.oxide_available = false;
                    for choice in self.prefill.values_mut() {
                        if choice.uses_oxide() {
                            *choice = Choice::Simt;
                        }
                    }
                }
            }
        }
    }

    fn prepare_shared_module(&mut self, device: &CudaDevice) {
        if self.shared.is_none()
            && self.prefill.values().any(|choice| {
                matches!(
                    choice,
                    Choice::SharedKv | Choice::VectorSharedKv | Choice::VectorSharedKvCombined
                )
            })
        {
            match super::super::attention_oxide::Support::load_shared(device) {
                Ok(Some(module)) => self.shared = Some(module),
                result => {
                    tracing::warn!(target:"infer::tuning",error=?result.err(),"shared KV module unavailable: exact fallback");
                    for choice in self.prefill.values_mut() {
                        if *choice == Choice::SharedKv {
                            *choice = Choice::RustSimt;
                        }
                    }
                }
            }
        }
    }

    fn prepare_vector_module(&mut self, device: &CudaDevice) {
        if self.vector.is_none()
            && self.prefill.values().any(|choice| {
                matches!(
                    choice,
                    Choice::VectorSharedKv | Choice::VectorSharedKvCombined
                )
            })
        {
            match super::super::attention_oxide::Support::load_vector(device) {
                Ok(Some(module)) => self.vector = Some(module),
                result => {
                    tracing::warn!(target:"infer::tuning",error=?result.err(),"vector shared module unavailable: qualified shared fallback");
                    let fallback = if self.shared.is_some() {
                        Choice::SharedKv
                    } else {
                        Choice::RustSimt
                    };
                    for choice in self.prefill.values_mut() {
                        if *choice == Choice::VectorSharedKv {
                            *choice = fallback;
                        }
                    }
                }
            }
        }
    }

    fn prepare_combined_module(&mut self, device: &CudaDevice) {
        if self.combined.is_none()
            && self
                .prefill
                .values()
                .any(|choice| *choice == Choice::VectorSharedKvCombined)
        {
            match super::super::attention_oxide::Support::load_combined(device) {
                Ok(Some(module)) => self.combined = Some(module),
                result => {
                    tracing::warn!(target:"infer::tuning",error=?result.err(),"vector combined module unavailable: qualified vector fallback");
                    let fallback = if self.vector.is_some() {
                        Choice::VectorSharedKv
                    } else if self.shared.is_some() {
                        Choice::SharedKv
                    } else {
                        Choice::RustSimt
                    };
                    for choice in self.prefill.values_mut() {
                        if *choice == Choice::VectorSharedKvCombined {
                            *choice = fallback;
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn prefill_mma(
        &mut self,
        output: TensorId,
        lanes: usize,
    ) -> Option<(
        &mut super::super::attention_mma::Support,
        super::super::attention_mma::catalog::Variant,
    )> {
        match *self.prefill_choice(output, lanes)? {
            Choice::Mma { variant } => Some((&mut self.mma.as_mut()?.support, variant)),
            Choice::MmaF16 => Some((
                &mut self.mma_f16.as_mut()?.support,
                super::super::attention_mma::catalog::Variant::Sm120,
            )),
            _ => None,
        }
    }

    pub(crate) fn prefill_oxide(
        &self,
        output: TensorId,
        lanes: usize,
    ) -> Option<&super::super::attention_oxide::Support> {
        match self.prefill_choice(output, lanes) {
            Some(Choice::RustSimt) => self.oxide.as_ref(),
            Some(Choice::SharedKv) => self.shared.as_ref(),
            Some(Choice::VectorSharedKv) => self.vector.as_ref(),
            Some(Choice::VectorSharedKvCombined) => self.combined.as_ref(),
            _ => None,
        }
    }

    pub(crate) fn prefill_query_tile(
        &self,
        output: TensorId,
        lanes: usize,
        fp8: bool,
    ) -> Option<usize> {
        if !fp8 {
            return None;
        }
        self.prefill_choice(output, lanes)
            .and_then(|choice| choice.query_tile())
    }

    pub(crate) fn supports(&self, heads: usize, dim: usize) -> bool {
        self.buffers.contains_key(&(heads, dim))
    }

    /// Whether three verification lanes of this geometry share one grouped launch.
    pub(crate) fn supports_verify(&self, heads: usize, dim: usize) -> bool {
        self.buffers
            .get(&(heads, dim))
            .is_some_and(|work| work.grouped.is_some() && work.verify.is_some())
    }

    /// Decode attention of the three lanes of one verification step: one grouped launch reads
    /// each KV block once, then each lane merges its own splits.
    pub(crate) fn record_verify(
        &self,
        attention: &dyn Attention,
        recording: &Recording<'_>,
        decode: DecodePlan,
        buffers: &VerifyDecodeBuffers<'_>,
    ) -> Result<()> {
        let work = self
            .buffers
            .get(&(decode.heads, decode.dim))
            .ok_or_else(|| error("split KV workspace"))?;
        let (Some(groups), Some(stacked)) = (work.grouped, work.verify.as_ref()) else {
            return Err(error("grouped verification workspace"));
        };
        attention
            .decode_verify(
                recording,
                SplitDecodePlan {
                    decode,
                    parts: work.parts,
                    groups: Some(groups),
                },
                partials(stacked)?,
                *buffers,
            )
            .map_err(error)
    }

    /// Decode one lane through this geometry's split partials.
    pub(crate) fn record(
        &self,
        attention: &dyn Attention,
        recording: &Recording<'_>,
        decode: DecodePlan,
        buffers: DecodeBuffers<'_>,
    ) -> Result<()> {
        let work = self
            .buffers
            .get(&(decode.heads, decode.dim))
            .ok_or_else(|| error("split KV workspace"))?;
        attention
            .decode_split(
                recording,
                SplitDecodePlan {
                    decode,
                    parts: work.parts,
                    groups: work.grouped,
                },
                ContractPartials {
                    numerator: buffer(&work.numerator)?,
                    maxima: buffer(&work.maxima)?,
                    sums: buffer(&work.sums)?,
                },
                buffers,
            )
            .map_err(error)
    }
}

fn partials(work: &Partials) -> Result<ContractPartials<'_>> {
    Ok(ContractPartials {
        numerator: buffer(&work.numerator)?,
        maxima: buffer(&work.maxima)?,
        sums: buffer(&work.sums)?,
    })
}
