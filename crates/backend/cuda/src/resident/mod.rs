//! Fixed device activations and captured, model-independent tensor operations.
pub(crate) mod arena;
mod batch;
pub use batch::BatchOutput;
mod attention;
mod attention_decode;
pub(crate) mod attention_mma;
pub(crate) mod attention_oxide;
pub(crate) mod attention_prefill;
mod batch_projection;
pub(crate) mod capacity;
mod capture;
mod capture_attention;
mod capture_checkpoint;
mod capture_rope;
mod capture_state;
pub(crate) mod fp8_cache;
pub(crate) mod fp8_gemm;
pub(crate) mod fusion_batch;
pub(crate) mod kernels;
mod linear_batch;
mod metadata;
pub(crate) mod nvfp4_gemm;
#[cfg(test)]
pub(crate) mod prefill_demand;
mod prefill_gemm;
mod prefill_projection;
mod profile;
mod program;
pub(crate) mod prompt_logits;
pub(crate) mod prompt_scratch;
mod recurrent;
mod recurrent_checkpoint;
pub(crate) mod recurrent_prefill;
pub(crate) mod slot_batch;
mod slot_commit;
mod slot_convolution;
mod slot_recurrent;
mod slot_rope;
mod slot_states;
mod slot_verify;
mod slot_verify_attention;
mod small_gemm;
mod validation;
pub use arena::ActivationArena;
pub(crate) use program::ProgramGeometry;
pub use program::{DeviceProgram, FusionWeights, ProgramWeights};
pub use prompt_scratch::Registry as PromptScratchRegistry;

mod conv_prefill;

pub mod moe;

mod reset;

pub(crate) mod greedy;

mod token_readout;

#[cfg(test)]
pub(crate) use attention_decode::taps as attention_prefill_taps;

#[cfg(test)]
pub(crate) use small_gemm::{pool_f32_capture_records, with_pool_f32, with_precise_dense};

pub(crate) mod fp8_quant_reduction;

pub(crate) mod pool_gemv;

#[cfg(test)]
pub(crate) use small_gemm::kernels as small_gemm_kernels;

#[cfg(test)]
#[path = "../../tests/unit/resident_gated_norm_discovery.rs"]
pub(crate) mod gated_norm_discovery;

#[cfg(test)]
pub(crate) use prefill_projection::{narrow_dense_captures, with_dense_reference};

#[cfg(test)]
#[path = "../../tests/unit/resident_pooled_verify_model.rs"]
mod pooled_verify_model_tests;

mod norm_maximum;
mod norm_maximum_kernel;
#[cfg(test)]
pub(crate) use norm_maximum::{
    captures as norm_maximum_captures, consumptions as norm_maximum_consumptions,
    with_separate as with_separate_norm_maximum,
};

mod canonical_recurrent;
