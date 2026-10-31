//! Pooled verification twins of the established per-lane kernels.
//! Keep their arithmetic separate and unchanged: extra stores can change compiler lowering,
//! so exact output, state and rollback tests compare against the original kernels.

pub use infer_cuda_kernels::tile::recurrent::recurrent_checkpoint as checkpoint;

#[cfg(test)]
#[path = "../../tests/unit/resident_recurrent_checkpoint.rs"]
mod tests;
