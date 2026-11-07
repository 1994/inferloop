#[cfg(test)]
#[cfg(feature = "cuda")]
pub use infer_cuda_kernels::tile::attention::attention;

#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "../../tests/unit/resident_embedding_rows.rs"]
mod embedding_rows_tests;
