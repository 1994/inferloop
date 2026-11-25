//! Bounded engine service with asynchronous CPU workers and HTTP adapters.
mod actor;
mod constants;
pub mod cpu;
mod http;
mod ingress;
mod media;
pub use actor::RuntimeHandle;
pub use http::{
    HttpError, NativeJsonAdapter, NativeTextRequest, router, router_with_text,
    router_with_text_and_media, router_with_text_media_and_responses, serve, serve_with_text,
    serve_with_text_and_media, serve_with_text_media_and_responses,
};
pub use media::{ImageDecoder, ImageProcessingOptions, MediaIngest, MediaRequest};
