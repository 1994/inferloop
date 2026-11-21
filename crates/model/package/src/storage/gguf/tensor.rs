//! Stored GGUF tensors and where an adapter maps them.
use infer_core::{Error, Result};
use infer_format::GgufType;
use std::collections::BTreeMap;

/// One stored tensor, as adapters see it before any data is read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Stored GGUF name.
    pub name: String,
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// Storage type.
    pub ty: GgufType,
}

/// The inverse of a value change a converter applied, undone when a float tensor is decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueTransform {
    /// Stored as is.
    Identity,
    /// Stored as `1 + w`, as llama.cpp stores the norms of families that scale by `1 + w`.
    PlusOne,
    /// Stored as `-exp(x)`, as llama.cpp stores gated-delta `A_log`.
    NegatedExp,
}

impl ValueTransform {
    /// Recover the original values in place.
    pub fn undo(self, values: &mut [f32]) {
        match self {
            Self::Identity => {}
            Self::PlusOne => values.iter_mut().for_each(|value| *value -= 1.0),
            Self::NegatedExp => values.iter_mut().for_each(|value| *value = (-*value).ln()),
        }
    }
}

/// Where one stored text tensor goes and how to read it back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappedTensor {
    /// Canonical HF name.
    pub slot: String,
    pub transform: ValueTransform,
    /// The canonical shape when the converter reshaped the tensor; `None` keeps the stored one.
    pub shape: Option<Vec<usize>>,
}

impl MappedTensor {
    /// A tensor stored under its canonical meaning.
    #[must_use]
    pub fn named(slot: impl Into<String>) -> Self {
        Self {
            slot: slot.into(),
            transform: ValueTransform::Identity,
            shape: None,
        }
    }

    /// The same mapping with a value transform to undo.
    #[must_use]
    pub const fn transformed(mut self, transform: ValueTransform) -> Self {
        self.transform = transform;
        self
    }

    /// The same mapping with a canonical shape.
    #[must_use]
    pub fn reshaped(mut self, shape: Vec<usize>) -> Self {
        self.shape = Some(shape);
        self
    }
}

/// Where one stored mmproj tensor goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisionTensor {
    /// A tensor name in the HF vision inventory.
    Slot(String),
    /// One temporal slice of the `Conv3D` patch embedding, which llama.cpp stores as two
    /// `Conv2D` weights; the package fuses them.
    PatchSlice(usize),
}

/// The tensors an architecture expects, by stored name and shape.
#[derive(Debug, Default)]
pub struct ExpectedTensors(BTreeMap<String, Vec<usize>>);

impl ExpectedTensors {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Expect `name` with `shape`.
    /// # Errors
    /// Rejects a name expected twice.
    pub fn insert(&mut self, name: impl Into<String>, shape: Vec<usize>) -> Result<()> {
        let name = name.into();
        if self.0.contains_key(&name) {
            return Err(Error::invalid(format!("duplicate expected tensor {name}")));
        }
        self.0.insert(name, shape);
        Ok(())
    }

    /// Check that `tensors` is exactly the expected set, each in a type `supported` accepts.
    /// # Errors
    /// Rejects a missing or extra tensor, a shape mismatch, or an unsupported type.
    pub fn check(
        mut self,
        tensors: &[TensorSpec],
        supported: impl Fn(GgufType) -> bool,
    ) -> Result<()> {
        for tensor in tensors {
            let shape = self
                .0
                .remove(&tensor.name)
                .ok_or_else(|| Error::invalid(format!("unexpected tensor {}", tensor.name)))?;
            if tensor.shape != shape {
                return Err(Error::invalid(format!(
                    "tensor {} shape {:?}, expected {shape:?}",
                    tensor.name, tensor.shape
                )));
            }
            if !supported(tensor.ty) {
                return Err(Error::invalid(format!(
                    "tensor {} has unsupported type {}",
                    tensor.name,
                    tensor.ty.name()
                )));
            }
        }
        if let Some(name) = self.0.keys().next() {
            return Err(Error::invalid(format!("missing tensor {name}")));
        }
        Ok(())
    }
}

/// Whether `ty` is a float storage type.
#[must_use]
pub const fn is_float(ty: GgufType) -> bool {
    matches!(ty, GgufType::F32 | GgufType::F16 | GgufType::Bf16)
}
