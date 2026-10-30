pub use infer_cuda_kernels::tile::attention;

#[cfg(test)]
#[path = "../../tests/unit/resident_embedding_rows.rs"]
mod embedding_rows_tests;
