//! Bounded grammar compilation and per-request matching for tool output dialects.
mod grammar;
mod schema;
use super::text::TextAssets;
use infer_core::{Error, ErrorCode, Result};
use infer_spi::{ConstraintHandle, GenerationConstraint};
use llguidance::toktrie::{ApproximateTokEnv, SimpleVob, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory, api::TopLevelGrammar};
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::Arc;

/// Retained reservation includes bounded lexer/matcher state and output-prefix history.
pub const CONSTRAINT_REQUEST_BYTES: usize = 32 * 1024 * 1024;
const GRAMMAR_CACHE_ENTRIES: usize = 4;
const MAX_LEXER_STATES: usize = 2048;
const MAX_GRAMMAR_SIZE: usize = 8192;
const MAX_ITEMS_PER_ROW: usize = 256;
const MAX_STEP_ITEMS: usize = 8192;
const MAX_OUTPUT_BYTES: usize = 4096;

#[derive(Debug, Clone)]
pub struct ToolConstraints {
    pub tools: Value,
    pub required: bool,
    pub named: Option<String>,
    pub parallel: bool,
}
#[derive(Default)]
pub(super) struct GrammarCache {
    entries: VecDeque<(String, Matcher)>,
    environment: Option<(String, Arc<ParserFactory>)>,
}
struct ToolMatcher {
    matcher: Matcher,
    environment: TokEnv,
    mask: SimpleVob,
    eos: u32,
    bytes: usize,
    words: usize,
}
impl GenerationConstraint for ToolMatcher {
    fn mask(&mut self) -> Result<&[u32]> {
        self.mask = self
            .matcher
            .compute_mask_or_eos()
            .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))?;
        if !self.matcher.stop_reason().is_ok() {
            return Err(Error::new(
                ErrorCode::Backend,
                format!("grammar stopped: {}", self.matcher.stop_reason()),
            ));
        }
        self.mask
            .as_slice()
            .get(..self.words)
            .ok_or_else(|| Error::invariant("grammar mask shorter than vocabulary"))
    }
    fn commit(&mut self, token: u32) -> Result<()> {
        if token == self.eos && self.matcher.is_stopped() {
            return Ok(());
        }
        self.bytes = self
            .bytes
            .saturating_add(self.environment.tok_trie().token(token).len());
        if self.bytes > MAX_OUTPUT_BYTES {
            return Err(Error::new(
                ErrorCode::Backend,
                "constrained output exceeds 4096-byte grammar budget",
            ));
        }
        self.matcher
            .consume_token(token)
            .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))
    }
}
impl TextAssets {
    /// Compile or reuse a grammar, with independent mutable lexer state for every request.
    /// # Errors
    /// Rejects unsupported schemas/tokenizers, missing EOS, and bounded compilation failures.
    pub fn tool_constraint(
        &self,
        spec: &ToolConstraints,
        sampling: &infer_ir::Sampling,
        thinking: bool,
        open: bool,
        vocabulary: usize,
    ) -> Result<ConstraintHandle> {
        if !self.supports_streaming_decode() {
            return Err(Error::unsupported(
                "tool constraints require a byte-level decoder",
            ));
        }
        let eos = sampling
            .eos_token
            .or_else(|| sampling.eos_tokens.first().copied())
            .ok_or_else(|| Error::unsupported("tool constraints require a configured EOS token"))?;
        let grammar = grammar::build(spec, self.tool_dialect(), thinking, open)?;
        let key = format!(
            "{vocabulary}:{:?}:{:?}:{grammar}",
            sampling.eos_token, sampling.eos_tokens
        );
        let mut cache = self
            .constraint_cache
            .lock()
            .map_err(|_| Error::invariant("grammar cache poisoned"))?;
        let matcher =
            if let Some(index) = cache.entries.iter().position(|(stored, _)| stored == &key) {
                let entry = cache
                    .entries
                    .remove(index)
                    .ok_or_else(|| Error::invariant("grammar cache entry lost"))?;
                let matcher = entry.1.deep_clone();
                cache.entries.push_back(entry);
                matcher
            } else {
                let factory = self.constraint_factory(sampling, vocabulary, eos, &mut cache)?;
                drop(cache);
                let parser = factory
                    .create_parser(TopLevelGrammar::from_lark(grammar))
                    .map_err(|e| Error::invalid(format!("tool grammar: {e}")))?;
                let mut parser = parser;
                if !parser.grammar_warnings().is_empty() {
                    return Err(Error::unsupported(
                        "tool grammar emitted unsupported-schema warnings",
                    ));
                }
                let matcher = Matcher::new(Ok(parser));
                if let Some(error) = matcher.get_error() {
                    return Err(Error::invalid(error));
                }
                let request = matcher.deep_clone();
                let mut cache = self
                    .constraint_cache
                    .lock()
                    .map_err(|_| Error::invariant("grammar cache poisoned"))?;
                if cache.entries.len() == GRAMMAR_CACHE_ENTRIES {
                    cache.entries.pop_front();
                }
                cache.entries.push_back((key, matcher));
                request
            };
        let environment = matcher
            .tok_env()
            .map_err(|e| Error::invalid(e.to_string()))?;
        Ok(ConstraintHandle::new(
            Box::new(ToolMatcher {
                matcher,
                environment,
                mask: SimpleVob::new(),
                eos,
                bytes: 0,
                words: vocabulary.div_ceil(u32::BITS as usize),
            }),
            vocabulary,
            CONSTRAINT_REQUEST_BYTES,
        ))
    }
    fn constraint_factory(
        &self,
        sampling: &infer_ir::Sampling,
        vocabulary: usize,
        eos: u32,
        cache: &mut GrammarCache,
    ) -> Result<Arc<ParserFactory>> {
        let environment_key = format!(
            "{vocabulary}:{:?}:{:?}",
            sampling.eos_token, sampling.eos_tokens
        );
        let factory = if let Some((cached_eos, factory)) = &cache.environment
            && *cached_eos == environment_key
        {
            factory.clone()
        } else {
            let encoded = self
                .tokenizer
                .to_string(false)
                .map_err(|e| Error::invalid(e.to_string()))?;
            let value: Value =
                serde_json::from_str(&encoded).map_err(|e| Error::invalid(e.to_string()))?;
            let special_eos = value["added_tokens"].as_array().is_some_and(|tokens| {
                tokens.iter().any(|token| {
                    token["id"].as_u64() == Some(u64::from(eos)) && token["special"] == true
                })
            });
            if !special_eos {
                return Err(Error::unsupported(
                    "constrained EOS must be a tokenizer special token",
                ));
            }
            let mut tokens = llguidance::token_bytes_from_tokenizer_json(&value)
                .map_err(|e| Error::unsupported(e.to_string()))?;
            if eos as usize >= vocabulary || tokens.len() > vocabulary {
                return Err(Error::invalid("grammar tokenizer vocabulary mismatch"));
            }
            tokens.resize(vocabulary, Vec::new());
            // Non-selected EOS IDs must never stop the engine inside a tool argument.
            for token in sampling
                .eos_tokens
                .iter()
                .copied()
                .chain(sampling.eos_token)
            {
                if token != eos && (token as usize) < tokens.len() {
                    tokens[token as usize] = vec![];
                }
            }
            let info = TokRxInfo::new(
                u32::try_from(vocabulary).map_err(|_| Error::invalid("vocabulary overflow"))?,
                eos,
            );
            let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &tokens)));
            let mut factory =
                ParserFactory::new_simple(&env).map_err(|e| Error::invalid(e.to_string()))?;
            factory.quiet();
            let limits = factory.limits_mut();
            limits.max_lexer_states = MAX_LEXER_STATES;
            limits.max_grammar_size = MAX_GRAMMAR_SIZE;
            limits.max_items_in_row = MAX_ITEMS_PER_ROW;
            limits.step_max_items = MAX_STEP_ITEMS;
            limits.verbose_errors = false;
            let factory = Arc::new(factory);
            cache.environment = Some((environment_key, factory.clone()));
            factory
        };

        Ok(factory)
    }
}

#[cfg(test)]
mod tests;
