//! Model-to-performance adapter; contains no device or kernel decisions.
use crate::{QuantizedPackage, WeightEncoding};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ProjectionWorkload {
    pub name: String,
    pub rows: usize,
    pub columns: usize,
    pub encoding: WeightEncoding,
    pub storage_dtype: crate::TensorDtype,
    pub mtp: bool,
}

/// Adapter boundary for model-specific storage and architecture metadata.
pub trait ProjectionCatalog {
    fn projections(&self) -> Vec<ProjectionWorkload>;
}

impl ProjectionCatalog for QuantizedPackage {
    fn projections(&self) -> Vec<ProjectionWorkload> {
        self.weights
            .iter()
            .map(|(name, weight)| (name, weight, false))
            .chain(self.mtp.iter().map(|(name, weight)| (name, weight, true)))
            .filter_map(|(name, weight, mtp)| {
                let shape = weight.shape();
                let [rows, columns] = shape.as_slice() else {
                    return None;
                };
                // Embeddings are indexed reads, not linear projections.
                if name.contains("embed_tokens") {
                    return None;
                }
                Some(ProjectionWorkload {
                    name: name.clone(),
                    rows: *rows,
                    columns: *columns,
                    encoding: weight.encoding(),
                    storage_dtype: weight.data().dtype,
                    mtp,
                })
            })
            .collect()
    }
}
