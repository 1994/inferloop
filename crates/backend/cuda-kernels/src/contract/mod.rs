//! Device, geometry and kernel contracts shared by the kernel sources.
//!
//! The host speaks only these types; `tile` and `aot` implement the traits behind them.
pub mod attention;
pub mod buffer;
pub mod draft;
pub mod kernels;
pub mod plan;
pub mod recording;
pub mod target;
pub mod vision;

pub use attention::{
    AppendBuffers, Attention, DecodeBuffers, DecodePlan, DenseBuffers, DensePlan, GROUPED_ROWS,
    KvBuffers, KvCache, KvElement, LanesAppendPlan, LanesDecodePlan, POOLED_TABLES, Partials,
    PooledAppendBuffers, PooledDecodeBuffers, PooledPlan, PrefillAppendPlan, PrefillDecodePlan,
    STACKED_LANE_ROWS, STACKED_ROWS, SplitDecodePlan, VERIFY_LANES, VerifyDecodeBuffers,
};
pub use buffer::{Bf16, Buffer, Fp4Pair, Fp8, Owner, Width};
pub use draft::{
    DRAFT_KEY_TILE, DRAFT_ROW_TILE, Draft, DraftAddNorm, DraftAttention, DraftConv, DraftNorm,
    DraftProjection, DraftRingStore, DraftRope, DraftSplit, DraftTop, DraftWalk, DraftWeight,
};
pub use kernels::{
    CONV_HISTORY, ChunkBuffers, ConvBuffers, DeltaBuffers, ExpertWeight, FUSED_LANES,
    FusedConvBuffers, FusedDeltaBuffers, FusionNormBuffers, GateBuffers, GreedyPartials, Kernels,
    LaneBuffers, Mlp, Moe, Norm, NormBuffers, NormFp4Buffers, NormMaximumBuffers, PoolDeltaBuffers,
    PreparedDeltaBuffers, ProductBuffers, Projection, Recurrent, RecurrentPool, Rope, RopeBuffers,
    Sampling, State, WeightRef,
};
pub use plan::{
    BinaryOp, ConvKernel, DeltaKernel, DeltaStepPlan, DenseKernel, DenseRowsPlan, DownPlan,
    EmbeddingBatchPlan, EmbeddingPlan, ExpertPlan, Fp4QuantKind, Fp4QuantPlan, Fp4ScalePlan,
    Fp8QuantKind, Fp8QuantPlan, FusionNormPlan, FusionShape, GemmLoop, GemmPlan, GreedyPlan,
    LastRowsKind, LastRowsPlan, LinearRowsPlan, NARROW_GRANULE, NormLayout, NormPlan, PoolConvPlan,
    PoolDeltaPlan, PoolGemvPlan, PromptPlan, QuantNormPlan, RecurrentShape, RopeMode, RopePlan,
    RoutePlan, RowNormPlan, RowsKernel, RowsPlan, SplitGemmPlan, SplitPlan, THREE_ROWS, UnaryOp,
};
pub use recording::{Foreign, Recording};
pub use target::Arch;
pub use vision::{Gelu, Pack, PositionTaps, Vision, VisionDense, VisionNorm, VisionRope};
