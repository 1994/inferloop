//! Model-lifetime qualifications; no data is persisted or shared with another model session.
use super::super::ALGO_WORDS;
use super::{Context, Operands, Prepared, Shape, WeightOwners, error};
use crate::mlp::ProjectionWeight;
use crate::tuning::resident_gemm::GemmConfig;
use cutile::prelude::DeviceError;
use std::{
    collections::BTreeMap,
    sync::{Mutex, MutexGuard},
};
// Opaque bytes identify only this process's returned algorithm. Every handle still queries
// its own algorithm; no stored blob is restored, serialized, or submitted to another handle.
pub(super) type DecisionKey = (usize, usize, Shape, GemmConfig, [u64; ALGO_WORDS]);
type NumericalKey = (DecisionKey, u64, u64, u32, u32);
#[derive(Default)]
struct Entries {
    timing: BTreeMap<DecisionKey, bool>,
    numerical: BTreeMap<NumericalKey, bool>,
    owners: BTreeMap<(u64, u64), WeightOwners>,
}
#[derive(Default)]
pub struct ModelCache {
    entries: Mutex<Entries>,
}
pub(super) const fn key(
    context: &Context,
    prepared: &Prepared,
    shape: Shape,
    native: GemmConfig,
) -> DecisionKey {
    (
        context.ordinal,
        context.primary,
        shape,
        native,
        prepared.algo.data,
    )
}
fn numerical_key(key: DecisionKey, op: &Operands<'_>) -> NumericalKey {
    let proof = op.proof(key.2);
    (key, proof.1, proof.2, proof.3, proof.4)
}
impl ModelCache {
    fn lock(&self) -> Result<MutexGuard<'_, Entries>, DeviceError> {
        self.entries
            .lock()
            .map_err(|_| error("vendor FP4 model qualification mutex"))
    }
    pub(super) fn qualification(
        &self,
        key: DecisionKey,
        op: &Operands<'_>,
    ) -> Result<(Option<bool>, Option<bool>), DeviceError> {
        let entries = self.lock()?;
        let result = (
            entries.timing.get(&key).copied(),
            entries.numerical.get(&numerical_key(key, op)).copied(),
        );
        drop(entries);
        Ok(result)
    }
    pub(super) fn save_timing(
        &self,
        key: DecisionKey,
        accepted: bool,
    ) -> Result<bool, DeviceError> {
        Ok(*self.lock()?.timing.entry(key).or_insert(accepted))
    }
    pub(super) fn save_numerical(
        &self,
        key: DecisionKey,
        op: &Operands<'_>,
        accepted: bool,
    ) -> Result<(), DeviceError> {
        let proof = numerical_key(key, op);
        let mut entries = self.lock()?;
        if let ProjectionWeight::Fp4(w, s, _) = op.weight {
            // Retain immutable allocations so pointer identities cannot be recycled beneath a proof.
            entries
                .owners
                .entry((proof.1, proof.2))
                .or_insert_with(|| (w.clone(), s.clone()));
        }
        entries.numerical.entry(proof).or_insert(accepted);
        drop(entries);
        Ok(())
    }
}
