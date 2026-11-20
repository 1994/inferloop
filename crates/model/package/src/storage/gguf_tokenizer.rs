//! Build the `tokenizer.json` and `tokenizer_config.json` assets from GGUF metadata.
//!
//! The byte-level BPE layout is shared by every GPT-2-style GGUF tokenizer; what varies is the
//! pre-tokenization split, which llama.cpp names in `tokenizer.ggml.pre`.
use infer_core::{Error, Result};
use infer_format::{Metadata, MetadataValue};
use serde_json::{Map, Value, json};

/// Qwen2's pre-tokenizer split pattern, as stored by the HF tokenizer.
const QWEN2_SPLIT: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
/// Qwen3.5's split: letters keep their combining marks.
const QWEN35_SPLIT: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// The split pattern `tokenizer.ggml.pre` names. An unknown name is an error, not Qwen2: a wrong
/// split tokenizes text differently without failing.
fn split_pattern(metadata: &Metadata) -> Result<&'static str> {
    match metadata.require_str("tokenizer.ggml.pre")? {
        "qwen2" => Ok(QWEN2_SPLIT),
        "qwen35" => Ok(QWEN35_SPLIT),
        other => Err(Error::unsupported(format!(
            "GGUF pre-tokenizer {other} has no split pattern"
        ))),
    }
}
/// GGUF token types that are normal vocabulary entries rather than added tokens.
const TOKEN_TYPE_NORMAL: u64 = 1;
/// GGUF token type for raw byte fallbacks; they stay in the vocab, not the added-token list.
const TOKEN_TYPE_BYTE: u64 = 6;

/// Build a `tokenizer.json` with the same components as the HF Qwen2 tokenizer.
/// # Errors
/// Rejects missing token/merge arrays, a malformed merge, or a token id that does not fit.
pub fn tokenizer_json(metadata: &Metadata) -> Result<Vec<u8>> {
    let tokens = metadata.require_strings("tokenizer.ggml.tokens")?;
    let merges = metadata.require_strings("tokenizer.ggml.merges")?;
    let split = split_pattern(metadata)?;
    let token_types = u64_array(metadata, "tokenizer.ggml.token_type")?;
    if tokens.len() != token_types.len() {
        return Err(Error::invalid("GGUF tokenizer token/type length mismatch"));
    }
    let mut vocab = Map::new();
    for (id, token) in tokens.iter().enumerate() {
        vocab.insert((*token).to_owned(), Value::from(id as u64));
    }
    let mut split_merges = Vec::with_capacity(merges.len());
    for merge in &merges {
        let (left, right) = merge
            .split_once(' ')
            .ok_or_else(|| Error::invalid("GGUF merge is not a pair"))?;
        split_merges.push(Value::String(format!("{left} {right}")));
    }
    let added: Vec<Value> = tokens
        .iter()
        .zip(&token_types)
        .enumerate()
        .filter(|(_, (_, ty))| **ty != TOKEN_TYPE_NORMAL && **ty != TOKEN_TYPE_BYTE)
        .map(|(id, (token, _))| {
            json!({
                "id": id as u64,
                "content": token,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect();
    let value = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": split}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
            ]
        },
        "post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": vocab,
            "merges": split_merges,
        }
    });
    serde_json::to_vec(&value).map_err(|error| Error::invalid(error.to_string()))
}

/// Build `tokenizer_config.json` with the special tokens and chat template.
/// # Errors
/// Rejects a missing token id or chat template.
pub fn tokenizer_config(metadata: &Metadata) -> Result<Vec<u8>> {
    let tokens = metadata.require_strings("tokenizer.ggml.tokens")?;
    // Families such as Qwen3.5 prepend no BOS and their files store none.
    let bos = metadata
        .get("tokenizer.ggml.bos_token_id")
        .is_some()
        .then(|| token(metadata, "tokenizer.ggml.bos_token_id", &tokens))
        .transpose()?;
    let eos = token(metadata, "tokenizer.ggml.eos_token_id", &tokens)?;
    let pad = token(metadata, "tokenizer.ggml.padding_token_id", &tokens)?;
    let template = metadata.require_str("tokenizer.chat_template")?;
    let add_bos = metadata
        .get("tokenizer.ggml.add_bos_token")
        .and_then(MetadataValue::as_bool)
        .unwrap_or(false);
    let value = json!({
        "add_bos_token": add_bos,
        "add_eos_token": false,
        "bos_token": bos,
        "eos_token": eos,
        "pad_token": pad,
        "chat_template": template,
        "clean_up_tokenization_spaces": false,
        "model_max_length": metadata.require_u64(&format!(
            "{}.context_length",
            metadata.require_str("general.architecture")?
        ))?,
    });
    serde_json::to_vec(&value).map_err(|error| Error::invalid(error.to_string()))
}

fn token(metadata: &Metadata, key: &str, tokens: &[&str]) -> Result<String> {
    let id = usize::try_from(metadata.require_u64(key)?)
        .map_err(|_| Error::invalid(format!("{key} exceeds usize")))?;
    tokens
        .get(id)
        .map(|token| (*token).to_owned())
        .ok_or_else(|| Error::invalid(format!("{key} is outside the vocabulary")))
}

fn u64_array(metadata: &Metadata, key: &str) -> Result<Vec<u64>> {
    metadata
        .get(key)
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| Error::invalid(format!("GGUF metadata {key} is not an array")))?
        .iter()
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                Error::invalid(format!("GGUF metadata {key} is not an integer array"))
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "../../tests/unit/providers_gguf_tokenizer.rs"]
mod tests;
