//! Bounded aggregation of backend device timing samples.
use crate::constants::NANOS_PER_MICROSECOND;
use crate::metrics::LatencyHistogram;
use infer_ir::{DeviceTimingSample, OpClass, TimedPhase};
use serde::{Deserialize, Serialize};

/// Phases of [`TimedPhase::ALL`].
pub const TIMED_PHASE_COUNT: usize = TimedPhase::ALL.len();
/// Operator classes of [`OpClass::ALL`].
pub const OP_CLASS_COUNT: usize = OpClass::ALL.len();
/// Layers tracked per phase; deeper layers fold into the last slot rather than grow the table.
pub const MAX_TIMED_LAYERS: usize = 256;
/// Implementation variants tracked; a backend declaring more folds the rest into the last slot.
pub const MAX_TIMED_VARIANTS: usize = 256;

/// Accumulated device time and the number of segments it came from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceTotal {
    pub device_ns: u64,
    pub segments: u64,
}

impl DeviceTotal {
    const fn add(&mut self, device_ns: u64) {
        self.device_ns = self.device_ns.saturating_add(device_ns);
        self.segments = self.segments.saturating_add(1);
    }
}

/// Device time of one phase: whole graphs, and where the node-level samples spent it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseTiming {
    /// Device time of each sampled graph, in microseconds.
    pub graphs: LatencyHistogram,
    /// Rows summed over the sampled graphs, to normalise time per sequence.
    pub rows: u64,
    /// By [`OpClass::ALL`] position.
    pub ops: [DeviceTotal; OP_CLASS_COUNT],
    /// By layer; entries past the model's depth stay zero.
    pub layers: Vec<DeviceTotal>,
}

impl Default for PhaseTiming {
    fn default() -> Self {
        Self {
            graphs: LatencyHistogram::default(),
            rows: 0,
            ops: [DeviceTotal::default(); OP_CLASS_COUNT],
            layers: vec![DeviceTotal::default(); MAX_TIMED_LAYERS],
        }
    }
}

/// Every device timing sample a deployment read back, in fixed storage: recording a sample never
/// allocates, so the aggregate can stay on in production.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeviceTimingMetrics {
    /// Samples recorded, whether or not they carried node segments.
    pub samples: u64,
    /// By [`TimedPhase::ALL`] position.
    pub phases: Vec<PhaseTiming>,
    /// By the backend's variant index.
    pub variants: Vec<DeviceTotal>,
    /// The backend's implementation names, which [`Self::variants`] indexes.
    pub variant_names: Vec<String>,
}

impl Default for DeviceTimingMetrics {
    fn default() -> Self {
        Self {
            samples: 0,
            phases: vec![PhaseTiming::default(); TIMED_PHASE_COUNT],
            variants: vec![DeviceTotal::default(); MAX_TIMED_VARIANTS],
            variant_names: Vec::new(),
        }
    }
}

impl DeviceTimingMetrics {
    /// An aggregate whose variant indices name `variants`, the backend's declared list.
    #[must_use]
    pub fn with_variants(variants: Vec<String>) -> Self {
        Self {
            variant_names: variants,
            ..Self::default()
        }
    }

    /// Fold one sample into the aggregate.
    pub fn record(&mut self, sample: &DeviceTimingSample) {
        self.samples = self.samples.saturating_add(1);
        let Some(phase) = self.phases.get_mut(sample.phase.index()) else {
            return;
        };
        phase.graphs.record(sample.total_ns / NANOS_PER_MICROSECOND);
        phase.rows = phase.rows.saturating_add(u64::from(sample.rows));
        for segment in &sample.segments {
            if let Some(total) = phase.ops.get_mut(segment.op.index()) {
                total.add(segment.device_ns);
            }
            if let Some(layer) = segment.layer
                && let Some(total) = phase
                    .layers
                    .get_mut(usize::from(layer).min(MAX_TIMED_LAYERS - 1))
            {
                total.add(segment.device_ns);
            }
            if let Some(variant) = segment.variant
                && let Some(total) = self
                    .variants
                    .get_mut(usize::from(variant).min(MAX_TIMED_VARIANTS - 1))
            {
                total.add(segment.device_ns);
            }
        }
    }

    /// The aggregate of one phase.
    #[must_use]
    pub fn phase(&self, phase: TimedPhase) -> Option<&PhaseTiming> {
        self.phases.get(phase.index())
    }
}

#[cfg(test)]
#[path = "../tests/unit/device.rs"]
mod tests;
