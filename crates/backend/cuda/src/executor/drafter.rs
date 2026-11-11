//! The draft beside the slot pool. A pooled speculative step asks it for candidate chains,
//! verifies them on the target, then lets it settle what the target accepted; the step itself
//! does not know which draft is loaded. A new draft is one more variant here.
use super::Sequence;
use crate::{
    device::CudaDevice,
    resident::{
        dflash2::{Engine, VerifiedRows},
        slot_batch::SlotPool,
    },
};
use infer_core::{Error, Result, StateId};
use infer_ir::{ExecutionTask, TaskOutput};
use infer_workloads::DraftPooling;
use std::collections::BTreeMap;

/// The loaded draft, exclusive at load: no draft, a head over the target's own layers, or a
/// separate draft model.
pub(super) enum Drafter {
    None,
    /// Multi-token-prediction layers of the target. Pooled steps batch them in their own slot
    /// pool, captured with the target pool when the head's graph allows it.
    Head(Option<SlotPool>),
    /// A separate draft model that owns per-sequence rings and proposes from target features.
    Model(Engine),
}

/// Candidate chains for one pooled verify step: each starts with the submitted token.
pub(super) struct Proposal {
    pub(super) chains: Vec<Vec<u32>>,
    /// The draft rebuilds its timelines once after verification instead of per sequence.
    pub(super) settles_after: bool,
}

impl Drafter {
    pub(super) fn new(head_depth: usize, model: Option<Engine>) -> Self {
        match model {
            Some(engine) => Self::Model(engine),
            None if head_depth > 0 => Self::Head(None),
            None => Self::None,
        }
    }

    /// How this draft runs beside the slot pool; the pool policy that differs by draft
    /// follows from it.
    pub(super) const fn pooling(&self) -> DraftPooling {
        match self {
            Self::Model(_) => DraftPooling::Pooled,
            Self::None | Self::Head(_) => DraftPooling::Separate,
        }
    }

    /// Whether this draft takes a slot pool of its own beside the target's.
    pub(super) const fn wants_pool(&self) -> bool {
        matches!(self, Self::Head(_))
    }

    /// The draft's own slot pool, which homes and prefix snapshots bind beside the target's.
    pub(super) const fn pool(&self) -> Option<&SlotPool> {
        match self {
            Self::Head(pool) => pool.as_ref(),
            Self::None | Self::Model(_) => None,
        }
    }

    pub(super) const fn pool_mut(&mut self) -> Option<&mut SlotPool> {
        match self {
            Self::Head(pool) => pool.as_mut(),
            Self::None | Self::Model(_) => None,
        }
    }

    /// The draft model, which the serial path primes and which leases sequence rings.
    pub(super) const fn model(&mut self) -> Option<&mut Engine> {
        match self {
            Self::Model(engine) => Some(engine),
            Self::None | Self::Head(_) => None,
        }
    }

    /// No sequence holds any of the draft's pooled storage.
    pub(super) fn idle(&self) -> bool {
        self.pool()
            .is_none_or(|pool| pool.free_slots() == pool.width())
    }

    /// Bind the draft to a newly captured target pool: keep the head's pool, or capture the
    /// model's pooled proposal over the target's verify features.
    pub(super) fn attach(
        &mut self,
        device: &CudaDevice,
        target: &SlotPool,
        head: Option<SlotPool>,
    ) -> Result<()> {
        match self {
            Self::None => Ok(()),
            Self::Head(pool) => {
                *pool = head;
                Ok(())
            }
            Self::Model(engine) => {
                target
                    .verify_features()
                    .ok_or_else(|| {
                        Error::invariant("pooled verification carries no draft features")
                    })
                    .and_then(|graphs| engine.attach_pool(device, &graphs))?;
                tracing::info!(target: "infer::executor", width = target.width(),
                    "pooled draft proposal captured");
                Ok(())
            }
        }
    }

    /// Whether the pooled capture holds everything this draft needs.
    pub(super) const fn attached(&self) -> bool {
        !matches!(self, Self::Head(None))
    }

    /// Drop the draft's pooled storage before the target pool goes.
    pub(super) fn detach(&mut self) {
        match self {
            Self::None => {}
            Self::Head(pool) => drop(pool.take()),
            Self::Model(engine) => engine.detach_pool(),
        }
    }

    /// Propose one chain per participant, or `None` when this step drafts per sequence.
    pub(super) async fn propose(
        &mut self,
        states: &mut BTreeMap<StateId, Sequence>,
        device: &CudaDevice,
        tasks: &[ExecutionTask],
        participants: &[(usize, usize)],
        width: usize,
    ) -> Result<Option<Proposal>> {
        let chains = match self {
            Self::Model(engine) if engine.attached() => Some(
                super::execution::model_chains(engine, states, device, tasks, participants, width)
                    .await?,
            ),
            Self::Head(Some(pool)) => {
                super::drafting::propose(pool, states, device, tasks, participants, width).await?
            }
            Self::None | Self::Head(None) | Self::Model(_) => None,
        };
        Ok(chains.map(|chains| Proposal {
            chains,
            settles_after: matches!(self, Self::Head(_)),
        }))
    }

    /// After the target committed a verify step: the model ingests the verified feature rows,
    /// the head replays accepted tokens when it proposed for the whole step.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the pooled execution transaction"
    )]
    pub(super) fn settle(
        &mut self,
        target: &SlotPool,
        states: &mut BTreeMap<StateId, Sequence>,
        device: &CudaDevice,
        tasks: &[ExecutionTask],
        participants: &[(usize, usize)],
        outputs: &[Option<TaskOutput>],
        (rows, width): (&[(Vec<f32>, Vec<f32>)], usize),
        (verified, settles_after): (&[VerifiedRows], bool),
    ) -> Result<()> {
        match self {
            Self::Model(engine) if engine.attached() && !verified.is_empty() => {
                engine.ingest_verified(device, verified, target.verified_slots())
            }
            Self::Head(Some(pool)) if settles_after => super::drafting::catch_up(
                pool,
                states,
                device,
                tasks,
                participants,
                outputs,
                rows,
                width,
            ),
            Self::None | Self::Head(_) | Self::Model(_) => Ok(()),
        }
    }
}
