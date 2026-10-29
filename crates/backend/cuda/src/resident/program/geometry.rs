//! One description of the graph rows retained by capture, workspace preparation and admission.
use super::ProgramWeights;
use crate::constants::PREFILL_LANES;
use std::iter::once;

/// Prompt graphs between the narrow and the wide graph, each half the next wider one.
const PROMPT_LADDER_RUNGS: usize = 2;

/// Each prompt width owns an activation arena; projection buffers share distinct widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramGeometry {
    pub verification: usize,
    /// Wide and narrow prompt graphs, respectively; zero means absent.
    pub prompts: [usize; 2],
}
impl ProgramGeometry {
    pub(crate) const fn new(weights: &ProgramWeights, requested: bool) -> Self {
        let verification = requested || weights.prefill_width < PREFILL_LANES;
        let wide_prompt = verification || weights.narrow_prefill_width < PREFILL_LANES;
        Self::from_flags(
            weights.batch_width,
            weights.prefill_width,
            weights.narrow_prefill_width,
            verification,
            wide_prompt,
        )
    }
    pub(crate) const fn from_flags(
        batch: usize,
        wide: usize,
        narrow: usize,
        verification: bool,
        wide_prompt: bool,
    ) -> Self {
        Self {
            verification: if verification && batch > 1 { batch } else { 0 },
            prompts: [
                if wide_prompt && wide >= PREFILL_LANES {
                    wide
                } else {
                    0
                },
                if narrow >= PREFILL_LANES && (narrow < wide || !wide_prompt) {
                    narrow
                } else {
                    0
                },
            ],
        }
    }
    /// Shared workers can trade prompt scratch for more resident decode slots.
    pub(crate) const fn worker(weights: &ProgramWeights, width: usize) -> Self {
        let wide = if width < weights.prefill_width {
            width
        } else {
            weights.prefill_width
        };
        let narrow = if weights.narrow_prefill_width < wide {
            weights.narrow_prefill_width
        } else {
            wide
        };
        Self::from_flags(weights.batch_width, wide, narrow, false, true)
    }
    pub(crate) const fn prompt_lanes(self) -> usize {
        self.prompts[0] + self.prompts[1]
    }
    pub(crate) const fn arena_lanes(self) -> usize {
        1 + self.verification + self.prompt_lanes()
    }
    pub(crate) fn widths(self) -> Vec<usize> {
        let mut widths = vec![1];
        let prompts = self.prompt_widths();
        for width in once(self.verification).chain(prompts.into_iter().rev()) {
            if width > 1 && !widths.contains(&width) {
                widths.push(width);
            }
        }
        widths
    }
    /// Every prompt graph width, widest first: the wide graph, the rungs that halve down from
    /// it while they stay above the narrow graph, then the narrow graph. Without a narrow graph
    /// there is no gap to fill and no rung.
    ///
    /// A prompt chunk costs one fixed-width replay whatever its token count, so a chunk
    /// replays on the narrowest graph that holds it. On a 27B hybrid model a 65-token chunk
    /// took 99 ms on the 1024-row graph and 37 ms on a 256-row one, while a 64-token chunk took
    /// 25 ms on the 64-row graph. Narrower graphs capture against the widest graph's
    /// activation scratch, so a rung costs graphs, not another arena.
    pub(crate) fn prompt_widths(self) -> Vec<usize> {
        let [wide, narrow] = self.prompts;
        let mut widths = Vec::with_capacity(PROMPT_LADDER_RUNGS + 2);
        if wide > 0 {
            widths.push(wide);
        }
        // Rungs only fill the gap between two captured graphs; a lone graph keeps its width.
        if wide > 0 && narrow > 0 {
            let mut rung = wide;
            for _ in 0..PROMPT_LADDER_RUNGS {
                rung /= 2;
                if rung <= narrow || rung < PREFILL_LANES || !rung.is_multiple_of(PREFILL_LANES) {
                    break;
                }
                widths.push(rung);
            }
        }
        if narrow > 0 && !widths.contains(&narrow) {
            widths.push(narrow);
        }
        widths
    }
}
