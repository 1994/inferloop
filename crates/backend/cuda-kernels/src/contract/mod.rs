//! Device, geometry and kernel contracts shared by the kernel sources.
//!
//! The host speaks only these types; `tile` and `aot` implement the traits behind them.
pub mod buffer;
pub mod kernels;
pub mod plan;
pub mod recording;
pub mod target;

pub use buffer::{Bf16, Buffer, Fp4Pair, Fp8, Owner};
pub use kernels::{
    CONV_HISTORY, ChunkBuffers, ConvBuffers, DeltaBuffers, ExpertWeight, FusionNormBuffers,
    GateBuffers, GreedyPartials, Kernels, Mlp, Moe, Norm, NormBuffers, NormFp4Buffers,
    NormMaximumBuffers, PreparedDeltaBuffers, ProductBuffers, Projection, Recurrent, Rope,
    RopeBuffers, Sampling, State, WeightRef,
};
pub use plan::{
    BinaryOp, ConvKernel, DeltaKernel, DeltaStepPlan, DenseKernel, DenseRowsPlan, DownPlan,
    EmbeddingPlan, ExpertPlan, Fp4QuantKind, Fp4QuantPlan, Fp8QuantKind, Fp8QuantPlan,
    FusionNormPlan, FusionShape, GemmLoop, GemmPlan, GreedyPlan, LastRowsKind, LastRowsPlan,
    NormLayout, NormPlan, PoolGemvPlan, PromptPlan, QuantNormPlan, RecurrentShape, RopeMode,
    RopePlan, RoutePlan, RowNormPlan, RowsKernel, RowsPlan, SplitGemmPlan, SplitPlan, THREE_ROWS,
    UnaryOp,
};
pub use recording::Recording;
pub use target::Arch;
