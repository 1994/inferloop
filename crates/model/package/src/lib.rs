//! HF packages, metadata preflight, bounded weight loading, and text assets.
//! Device allocation and execution remain owned by independent backends.
mod constants;
mod input;
mod providers;
mod storage;
pub use input::{image, mrope, prompt, video, vision};
mod workload;
/// Canonical provider import result, re-exported for package consumers.
pub use infer_spi::ImportedModel;
pub use input::constraint::{CONSTRAINT_REQUEST_BYTES, ToolConstraints};
pub use input::image::{ImageProcessor, PreprocessedImage, RgbImage};
pub use input::multimodal::{MediaLimits, MultimodalAssets};
pub use input::prompt::{
    ExpandedPrompt, MediaLayout, MediaSpan, PreparedPrompt, PromptImage, RawImage, VisualPlacement,
    expand_placeholders, image_token_counts, media_layout, place_visual_embeddings, placeholders,
    prepare, prepare_video, require_encoder, visual_tokens,
};
pub use input::text::{ChatMessage, ChatOptions, TextAssets, TextStreamDecoder};
pub use input::tool::{
    OutputStreamParser, ParsedOutput, ParsedToolCall, StreamEvent, ToolCall, ToolDialect, assemble,
    parse as parse_model_output, parse_with as parse_model_output_with,
};
pub use input::video::{PreprocessedVideo, RawVideo, VideoProcessor};
pub use providers::dflash2::{
    ARCHITECTURE as DFLASH2_ARCHITECTURE, DFlash2Config, DFlash2DraftConfig, DraftGeometry,
    DraftQuote, check_declared_architecture,
};
pub use providers::qwen::{QwenProvider, VisionConfig};
pub use providers::registry::{ModelRegistry, default_registry};
pub use storage::index::SafetensorsIndex;
pub use storage::loader::{
    LoadOptions, LoadedWeights, TensorLoadPlan, WeightLoadPlan, WeightStorage, WeightTarget,
    load_weights,
};
pub use storage::memory::{MemoryEstimate, memory_estimate};
pub use storage::package::{
    ModelPackage, PackageManifest, QwenPackage, WeightBinding, package_path,
};
pub use storage::quantized::{
    DeviceWeight, QuantizedPackage, WeightEncoding, WeightSource, WeightView,
};
pub use storage::safetensors::{
    HostTensor, SafetensorsFile, TensorDtype, TensorHeader, convert_float_bytes, write_safetensors,
};
pub use workload::{ProjectionCatalog, ProjectionWorkload};

mod generation;
pub use generation::{GenerationDefaults, ResolvedGeneration, SamplingOverrides};
