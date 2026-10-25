use crate::constants::F32_BYTES_U64;
use crate::native::decision_tokens_fit;
use infer_core::{Error, ProgramId, Result};
use infer_ir::{CanonicalRequest, ModelIr, ModelOutput, Workload, WorkloadOutput, WorkloadPlan};
use infer_models::{HostTensor, SafetensorsFile, package_path};
use infer_spi::WorkloadProvider;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};

/// Maximum bytes for one readout tensor read and for the accumulated readout package.
const READOUT_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// Explicit installed readouts. No fabricated rank/decision score from an unrelated LM channel.
#[derive(Clone)]
pub struct ProjectionWorkloads {
    weights: BTreeMap<String, HostTensor>,
    hidden: usize,
    vocab: usize,
    identity: String,
}
impl ProjectionWorkloads {
    /// # Errors
    /// Returns an I/O or invalid-input error for missing, malformed, unsupported, or oversized assets.
    pub fn open(root: impl AsRef<Path>, hidden: usize, vocab: usize) -> Result<Self> {
        let mut file = SafetensorsFile::open(package_path(root.as_ref(), "readouts.safetensors")?)?;
        let names: Vec<_> = file.tensors.keys().cloned().collect();
        let mut weights = BTreeMap::new();
        let mut bytes = 0u64;
        for name in names {
            if ![
                "embedding.weight",
                "rank.weight",
                "rank.bias",
                "decision.weight",
                "decision.bias",
            ]
            .contains(&name.as_str())
            {
                return Err(Error::invalid("unknown readout tensor"));
            }
            let tensor = file.read_f32(&name, READOUT_MAX_BYTES)?;
            let tensor_bytes = (tensor.data.len() as u64)
                .checked_mul(F32_BYTES_U64)
                .ok_or_else(|| Error::invalid("readout size overflow"))?;
            bytes = bytes
                .checked_add(tensor_bytes)
                .ok_or_else(|| Error::invalid("readout size overflow"))?;
            if bytes > READOUT_MAX_BYTES {
                return Err(Error::invalid("readout package exceeds 64 MiB"));
            }
            weights.insert(name, tensor);
        }
        for (name, rows) in [
            ("embedding.weight", None),
            ("rank.weight", Some(1)),
            ("decision.weight", None),
        ] {
            if let Some(t) = weights.get(name)
                && (t.shape.len() != 2
                    || t.shape[1] != hidden
                    || t.shape[0] == 0
                    || rows.is_some_and(|n| t.shape[0] != n)
                    || name == "decision.weight" && t.shape[0] > vocab)
            {
                return Err(Error::invalid(format!("{name}: readout shape mismatch")));
            }
        }
        for kind in ["rank", "decision"] {
            if let Some(bias) = weights.get(&format!("{kind}.bias")) {
                let weight = weights
                    .get(&format!("{kind}.weight"))
                    .ok_or_else(|| Error::invalid("readout bias without projection"))?;
                if bias.shape != vec![weight.shape[0]] {
                    return Err(Error::invalid("readout bias shape mismatch"));
                }
            }
        }
        let mut hash = Sha256::new();
        hash.update((hidden as u64).to_le_bytes());
        hash.update((vocab as u64).to_le_bytes());
        for (name, tensor) in &weights {
            hash.update(name.as_bytes());
            for v in &tensor.data {
                hash.update(v.to_le_bytes());
            }
        }
        Ok(Self {
            weights,
            hidden,
            vocab,
            identity: format!("projection-workloads-f32-v1:{:x}", hash.finalize()),
        })
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
    )]
    fn project(&self, kind: &str, x: &[f32]) -> Result<Vec<f32>> {
        let weight = self
            .weights
            .get(&format!("{kind}.weight"))
            .ok_or_else(|| Error::unsupported("readout projection not installed"))?;
        if x.len() != self.hidden {
            return Err(Error::invalid("readout hidden width"));
        }
        let bias = self.weights.get(&format!("{kind}.bias"));
        let output: Vec<_> = weight
            .data
            .chunks_exact(self.hidden)
            .enumerate()
            .map(|(row, w)| {
                w.iter()
                    .zip(x)
                    .map(|(w, x)| f64::from(*w) * f64::from(*x))
                    .sum::<f64>() as f32
                    + bias.map_or(0.0, |b| b.data[row])
            })
            .collect();
        if output.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid("readout produced non-finite values"));
        }
        Ok(output)
    }
}
impl WorkloadProvider for ProjectionWorkloads {
    fn fork(&self) -> Option<Box<dyn WorkloadProvider + Send + Sync>> {
        Some(Box::new(self.clone()))
    }
    fn identity(&self) -> &str {
        &self.identity
    }
    fn supports(&self, w: &Workload) -> bool {
        match w {
            Workload::Generate { .. } => true,
            Workload::Embed { .. } => self.weights.contains_key("embedding.weight"),
            Workload::Rerank { .. } => self.weights.contains_key("rank.weight"),
            Workload::Decision(_) => self.weights.contains_key("decision.weight"),
            _ => false,
        }
    }
    fn plan(&self, r: &CanonicalRequest, m: &ModelIr, p: ProgramId) -> Result<WorkloadPlan> {
        if m.hidden_size != self.hidden || m.vocab_size != self.vocab || !self.supports(&r.workload)
        {
            return Err(Error::unsupported("readout/model/workload mismatch"));
        }
        let mut declared = m.clone();
        if let Some(head) = r.workload.head()
            && !declared.heads.contains(&head)
        {
            declared.heads.push(head);
        }
        let plan = crate::NativeWorkloads.plan(r, &declared, p)?;
        if let Workload::Embed {
            dimensions: Some(d),
            ..
        } = r.workload
            && d > self.weights["embedding.weight"].shape[0]
        {
            return Err(Error::invalid(
                "embedding request exceeds installed projection width",
            ));
        }
        if let Workload::Decision(schema) = &r.workload {
            let limit = self.weights["decision.weight"].shape[0];
            for q in &schema.questions {
                if !decision_tokens_fit(q, limit) {
                    return Err(Error::invalid(
                        "decision option exceeds installed readout channels",
                    ));
                }
            }
        }
        Ok(plan)
    }
    fn postprocess(&self, r: &CanonicalRequest, outputs: &[ModelOutput]) -> Result<WorkloadOutput> {
        let transformed = outputs
            .iter()
            .map(|o| {
                // Native postprocessing consumes only the selected readout. Keep the borrowed
                // backbone buffers intact without copying fields the selected head never reads.
                let mut projected = ModelOutput {
                    next_token: o.next_token,
                    logits: Vec::new(),
                    hidden: Vec::new(),
                    tokens: Vec::new(),
                };
                match r.workload {
                    Workload::Embed { .. } => {
                        projected.hidden = o
                            .hidden
                            .iter()
                            .map(|x| self.project("embedding", x))
                            .collect::<Result<_>>()?;
                    }
                    Workload::Rerank { .. } => {
                        projected.logits = self.project(
                            "rank",
                            o.hidden
                                .last()
                                .ok_or_else(|| Error::invalid("readout has no hidden row"))?,
                        )?;
                    }
                    Workload::Decision(_) => {
                        projected.logits = self.project(
                            "decision",
                            o.hidden
                                .last()
                                .ok_or_else(|| Error::invalid("readout has no hidden row"))?,
                        )?;
                    }
                    _ => return Ok(o.clone()),
                }
                Ok(projected)
            })
            .collect::<Result<Vec<_>>>()?;
        crate::NativeWorkloads.postprocess(r, &transformed)
    }
}

#[cfg(test)]
#[path = "../tests/unit/readouts.rs"]
mod tests;
