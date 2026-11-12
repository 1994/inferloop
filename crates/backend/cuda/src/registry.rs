//! Compiler registrations for the resident graph dispatcher; costs are bootstrap placeholders.
//!
//! Each registration names the lowering that decides what records for its operation, so a
//! registration points at the code that runs. Shapes no cell records are refused when the
//! model loads (`resident::require_kernel_geometry`), after narrow heads are padded.
use infer_core::KernelId;
use infer_ir::{BackendKind, CapabilityRequirements, DType, Operation, PrecisionPlan};
use infer_spi::{KernelProvider, KernelRegistration, SourceLocation};
use std::num::NonZeroU64;

/// First kernel id assigned to this provider's operation registrations.
const KERNEL_ID_BASE: u64 = 3000;

pub struct CudaKernels;
impl KernelProvider for CudaKernels {
    fn kernels(&self) -> Vec<KernelRegistration> {
        [
            Operation::TokenEmbedding,
            Operation::MatMul,
            Operation::RmsNorm,
            Operation::Split,
            Operation::Rope,
            Operation::Attention,
            Operation::Convolution,
            Operation::LinearAttention,
            Operation::GatedNorm,
            Operation::MoeRoute,
            Operation::ExpertMatMul,
            Operation::MoeCombine,
            Operation::Silu,
            Operation::Sigmoid,
            Operation::Multiply,
            Operation::Residual,
            Operation::LmHead,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, operation)| KernelRegistration {
            backend: BackendKind::Cuda,
            id: KernelId::from_nonzero(
                NonZeroU64::MIN.saturating_add((KERNEL_ID_BASE + i as u64) - 1),
            ),
            operation,
            precision: PrecisionPlan::f32(),
            requirements: CapabilityRequirements {
                compute_dtypes: vec![DType::F32],
                ..Default::default()
            },
            max_shape_elements: usize::MAX,
            workspace_bytes: 0,
            priority: 0,
            estimated_ns: 1,
            source: {
                let (file, function) = lowering(operation);
                SourceLocation {
                    crate_name: "infer-backend-cuda".into(),
                    file: format!("crates/backend/cuda/src/resident/{file}"),
                    function: function.into(),
                }
            },
        })
        .collect()
    }
}

/// The module and function that lower `operation` into recorded kernels.
const fn lowering(operation: Operation) -> (&'static str, &'static str) {
    match operation {
        Operation::MatMul | Operation::LmHead => {
            ("projection_lowering.rs", "projection_lowering::lower")
        }
        Operation::RmsNorm | Operation::GatedNorm => ("norm_lowering.rs", "norm_lowering::layout"),
        Operation::Attention => ("attention_lowering.rs", "attention_lowering::lower"),
        Operation::Convolution | Operation::LinearAttention => {
            ("recurrent_lowering.rs", "recurrent_lowering::lower")
        }
        Operation::Rope => ("capture_rope.rs", "Capture::record_rope"),
        Operation::MoeRoute | Operation::ExpertMatMul | Operation::MoeCombine => {
            ("moe.rs", "Capture::record_moe")
        }
        Operation::TokenEmbedding => ("capture.rs", "Capture::embedding"),
        Operation::Split => ("capture.rs", "Capture::record_split"),
        Operation::Silu
        | Operation::Sigmoid
        | Operation::Multiply
        | Operation::Residual
        | Operation::Pool
        | Operation::Rank
        | Operation::Decision => ("capture.rs", "Capture::operation"),
    }
}
