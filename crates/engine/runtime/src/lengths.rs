//! Resolved single-sequence length limits shared by every request entry point.
//!
//! `--max-model-len` is the total context of one sequence, prompt and output together. It is
//! resolved once, after model metadata is known and before any backend resource is planned, so
//! the HTTP adapters, the workload plan, admission and capacity planning read the same numbers.
//! The input cap and the total context stay separate: an input cap is not a `--max-model-len`.
use crate::RuntimeConfig;
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};

/// Where an effective length limit came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitSource {
    /// The limit follows the model's supported context ceiling.
    Model,
    /// The limit follows an explicit service setting.
    Service,
    /// The limit follows a built-in bounded default.
    Default,
}

/// Effective source of each limit in [`ResolvedLengthLimits`], for configuration readback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LengthLimitSources {
    pub total: LimitSource,
    pub input_cap: LimitSource,
    pub output_cap: LimitSource,
}

/// Single-sequence length limits after the model ceiling, service overrides and defaults combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedLengthLimits {
    /// Model-supported context ceiling (`M`).
    pub model_limit: usize,
    /// Effective service total context, prompt and output together (`L`).
    pub total: usize,
    /// Effective accepted prompt-token cap.
    pub input_cap: usize,
    /// Effective generated-token cap.
    pub output_cap: usize,
    pub sources: LengthLimitSources,
}

impl ResolvedLengthLimits {
    /// Combine the model ceiling with the service settings exactly once.
    ///
    /// # Errors
    /// Rejects a zero or unattainable explicit limit instead of silently shrinking it.
    pub fn resolve(model_limit: usize, config: &RuntimeConfig) -> Result<Self> {
        if model_limit == 0 {
            return Err(Error::invalid("model context limit must be positive"));
        }
        let (total, total_source) = match config.max_model_len {
            None => (model_limit, LimitSource::Model),
            Some(0) => return Err(Error::invalid("max-model-len must be positive")),
            Some(requested) if requested > model_limit => {
                return Err(Error::invalid(format!(
                    "max-model-len {requested} exceeds the model context limit {model_limit}"
                )));
            }
            Some(requested) => (requested, LimitSource::Service),
        };
        // The input cap never exceeds the total context, so a prompt can always leave room for
        // at least one output token when the total is positive.
        let (input_cap, input_source) = if config.max_input_tokens <= total {
            (config.max_input_tokens, LimitSource::Service)
        } else {
            (total, LimitSource::Model)
        };
        if input_cap == 0 {
            return Err(Error::invalid("input token cap must be positive"));
        }
        let (output_cap, output_source) = match config.max_output_tokens {
            None => (total, LimitSource::Default),
            Some(0) => return Err(Error::invalid("max-output-tokens must be positive")),
            Some(requested) if requested > total => {
                return Err(Error::invalid(format!(
                    "max-output-tokens {requested} exceeds the total context {total}"
                )));
            }
            Some(requested) => (requested, LimitSource::Service),
        };
        Ok(Self {
            model_limit,
            total,
            input_cap,
            output_cap,
            sources: LengthLimitSources {
                total: total_source,
                input_cap: input_source,
                output_cap: output_source,
            },
        })
    }
    /// Check one request's encoded prompt against its output budget.
    ///
    /// # Errors
    /// Returns an invalid-input error when the request cannot fit, without truncating it.
    pub fn check(&self, prompt_tokens: usize, output_tokens: usize) -> Result<()> {
        if prompt_tokens == 0 {
            return Err(Error::invalid("encoded prompt must not be empty"));
        }
        if prompt_tokens > self.input_cap {
            return Err(Error::invalid(format!(
                "prompt of {prompt_tokens} tokens exceeds the input cap {}",
                self.input_cap
            )));
        }
        if output_tokens > self.output_cap {
            return Err(Error::invalid(format!(
                "requested output of {output_tokens} tokens exceeds the output cap {}",
                self.output_cap
            )));
        }
        let requested = prompt_tokens
            .checked_add(output_tokens)
            .ok_or_else(|| Error::invalid("prompt + output token count overflow"))?;
        if requested > self.total {
            return Err(Error::invalid(format!(
                "prompt {prompt_tokens} + output {output_tokens} exceeds the total context {}",
                self.total
            )));
        }
        Ok(())
    }
    /// Generated-token budget left for a prompt, after the output cap and the total context.
    ///
    /// # Errors
    /// Returns an invalid-input error when the prompt leaves no room for output.
    pub fn remaining_output(&self, prompt_tokens: usize) -> Result<usize> {
        let remaining = self.total.saturating_sub(prompt_tokens);
        if remaining == 0 {
            return Err(Error::invalid(format!(
                "prompt of {prompt_tokens} tokens leaves no room in the {} token total context",
                self.total
            )));
        }
        Ok(remaining.min(self.output_cap))
    }
}

#[cfg(test)]
mod tests;
