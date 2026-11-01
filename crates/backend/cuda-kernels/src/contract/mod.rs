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
    ExpertWeight, FusionNormBuffers, GreedyPartials, Kernels, Mlp, Moe, Norm, NormBuffers,
    NormFp4Buffers, NormMaximumBuffers, ProductBuffers, Projection, Rope, RopeBuffers, Sampling,
    State, WeightRef,
};
pub use plan::{
    BinaryOp, DownPlan, ExpertPlan, FusionNormPlan, FusionShape, GreedyPlan, LastRowsKind,
    LastRowsPlan, NormLayout, NormPlan, PoolGemvPlan, QuantNormPlan, RopeMode, RopePlan, RoutePlan,
    RowNormPlan, RowsKernel, RowsPlan, SplitPlan, THREE_ROWS, UnaryOp,
};
pub use recording::Recording;
pub use target::Arch;
