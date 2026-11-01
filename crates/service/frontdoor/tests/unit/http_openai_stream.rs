use super::*;
use crate::cpu::CpuConfig;
use crate::cpu::CpuPool;
use infer_models::{ParsedToolCall, TextAssets};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::env::temp_dir;
use std::fs::create_dir_all;
use std::fs::write;
use std::process::id as process_id;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;

/// The GPT-2 byte alphabet: printable bytes keep their character, the rest are shifted past
/// the printable range so every byte has a distinct character.
fn byte_alphabet() -> Vec<char> {
    let mut printable: Vec<u8> = (b'!'..=b'~').collect();
    printable.extend(b'\xa1'..=b'\xac');
    printable.extend(b'\xae'..=b'\xff');
    let mut alphabet: Vec<char> = printable.iter().map(|byte| char::from(*byte)).collect();
    let shifted: Vec<u8> = (0u8..=255)
        .filter(|value| !printable.contains(value))
        .collect();
    alphabet.extend(shifted.iter().enumerate().map(|(index, _)| {
        char::from_u32(256 + u32::try_from(index).expect("alphabet index fits u32"))
            .expect("alphabet code is a scalar value")
    }));
    alphabet
}

/// A byte-level tokenizer over the whole byte alphabet, so any text round-trips one token per
/// byte. The served packages use `ByteLevel` too; this one is small enough to build in a
/// test.
fn byte_assets() -> Arc<TextAssets> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let alphabet = byte_alphabet();
    let vocab: serde_json::Map<String, Value> = (0u16..=255)
        .map(|byte| (alphabet[byte as usize].to_string(), json!(u64::from(byte))))
        .collect();
    let tokenizer = json!({
        "version": "1.0",
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false},
        "post_processor": null,
        "decoder": {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false},
        "model": {"type":"BPE","dropout":null,"unk_token":null,
            "continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,
            "vocab":vocab,"merges":[]}
    });
    let root = temp_dir().join(format!(
        "infer-byte-tokenizer-{}-{}",
        process_id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    create_dir_all(&root).expect("temporary package");
    write(
        root.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).expect("tokenizer serializes"),
    )
    .expect("tokenizer written");
    Arc::new(TextAssets::open(&root, 4096).expect("byte tokenizer opens"))
}

/// A stream wired to a real decoder and parser, as the service runs it.
fn parsing_stream(assets: &TextAssets, declared: &[&str]) -> Stream {
    let mut stream = stream();
    stream.decoder = Some(assets.stream_decoder());
    stream.parser = Some(
        OutputStreamParser::new(
            assets.tool_dialect(),
            Some(
                declared
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect::<BTreeSet<_>>(),
            ),
        )
        .with_argument_deltas(),
    );
    stream.parse = true;
    stream
}

fn stream() -> Stream {
    let (sender, receiver) = mpsc::channel(1);
    drop(sender);
    Stream {
        request: RequestId::new(7).unwrap(),
        receiver,
        delivery: CpuPool::new(CpuConfig::default()).unwrap(),
        decoder: None,
        parser: None,
        prompt_reasoning: None,
        queue: VecDeque::new(),
        id: "chatcmpl-7".into(),
        object: "chat.completion.chunk",
        model: "served".into(),
        created: 1,
        chat: true,
        include_usage: false,
        prompt_tokens: Arc::new(AtomicUsize::new(3)),
        completion_tokens: 0,
        call_index: 0,
        parse: true,
        phase: Phase::Running,
    }
}

fn chunks(stream: &Stream) -> Vec<Value> {
    stream
        .queue
        .iter()
        .filter(|data| data.as_str() != DONE)
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

#[test]
fn a_tool_call_delta_carries_the_complete_reply_identifier() {
    let mut stream = stream();
    stream.publish(StreamEvent::Call(ParsedToolCall {
        name: "get_weather".into(),
        arguments: "{\"city\":\"Paris\"}".into(),
    }));
    stream.publish(StreamEvent::Call(ParsedToolCall {
        name: "ping".into(),
        arguments: "{}".into(),
    }));
    stream.finish("tool_calls");
    let chunks = chunks(&stream);
    let first = &chunks[0]["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(first["index"], 0);
    assert_eq!(first["id"], "call_7_0");
    assert_eq!(first["type"], "function");
    assert_eq!(first["function"]["name"], "get_weather");
    assert_eq!(first["function"]["arguments"], "{\"city\":\"Paris\"}");
    // A second call is appendable and keeps its own index and identifier.
    assert_eq!(
        chunks[1]["choices"][0]["delta"]["tool_calls"][0]["id"],
        "call_7_1"
    );
    assert_eq!(chunks[2]["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(stream.queue.back().unwrap(), DONE);
    assert_eq!(stream.phase, Phase::Done);
}

#[test]
fn usage_is_published_once_when_requested() {
    let mut stream = stream();
    stream.include_usage = true;
    stream.completion_tokens = 4;
    stream.finish("stop");
    let chunks = chunks(&stream);
    assert!(chunks[0].get("usage").unwrap().is_null());
    assert_eq!(chunks[1]["choices"].as_array().unwrap().len(), 0);
    assert_eq!(chunks[1]["usage"]["prompt_tokens"], 3);
    assert_eq!(chunks[1]["usage"]["completion_tokens"], 4);
    assert_eq!(chunks[1]["usage"]["total_tokens"], 7);
}

#[test]
fn a_failure_ends_the_stream_with_one_error_event() {
    let mut stream = stream();
    stream.fail(&Error::new(ErrorCode::Backend, "device fault"));
    let chunks = chunks(&stream);
    assert_eq!(chunks[0]["error"]["message"], "device fault");
    assert_eq!(chunks[0]["error"]["code"], "backend_error");
    assert_eq!(chunks[0]["error"]["type"], "server_error");
    assert!(chunks[0]["error"].get("param").unwrap().is_null());
    assert_eq!(stream.queue.back().unwrap(), DONE);
    assert_eq!(stream.phase, Phase::Done);
}

#[test]
fn a_reasoning_delta_is_published_separately_from_content() {
    let mut stream = stream();
    stream.publish(StreamEvent::Reasoning("why".into()));
    stream.publish(StreamEvent::Text("answer".into()));
    let chunks = chunks(&stream);
    assert_eq!(chunks[0]["choices"][0]["delta"]["reasoning_content"], "why");
    assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "answer");
}

#[tokio::test]
async fn streamed_tool_calls_are_decoded_parsed_and_published() {
    let assets = byte_assets();
    let text = concat!(
        "Checking.\n",
        "<tool_call>\n",
        "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n",
        "</tool_call>"
    );
    let tokens = assets.encode(text, false).expect("byte tokenizer encodes");
    assert!(tokens.len() > text.len() / 2, "one token per byte");
    let mut stream = parsing_stream(&assets, &["get_weather"]);
    for (index, token) in tokens.iter().enumerate() {
        stream
            .advance(EngineOutput::Token {
                request: stream.request,
                token: *token,
                index,
            })
            .await
            .expect("token advances");
    }
    stream.finish("tool_calls");
    let chunks = chunks(&stream);
    let content: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, "Checking.");
    let call = chunks
        .iter()
        .find_map(|chunk| chunk["choices"][0]["delta"]["tool_calls"].as_array())
        .expect("one tool call delta")
        .first()
        .cloned()
        .expect("one call");
    let call = &call;
    assert_eq!(call["id"], "call_7_0");
    assert_eq!(call["function"]["name"], "get_weather");
    // The arguments are the model's own bytes, not a re-serialized object.
    let arguments: String = chunks
        .iter()
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str()
        })
        .collect();
    assert_eq!(arguments, "{\"city\":\"Paris\"}");
    assert!(
        chunks
            .iter()
            .filter(|chunk| chunk["choices"][0]["delta"]["tool_calls"].is_array())
            .count()
            > 1
    );
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "tool_calls"
    );
    assert_eq!(stream.queue.back().unwrap(), DONE);
}

#[tokio::test]
async fn a_streamed_call_that_is_not_declared_stays_in_the_content() {
    let assets = byte_assets();
    let text = "<tool_call>{\"name\":\"invented\",\"arguments\":{}}</tool_call>";
    let tokens = assets.encode(text, false).expect("byte tokenizer encodes");
    let mut stream = parsing_stream(&assets, &["declared"]);
    for (index, token) in tokens.iter().enumerate() {
        stream
            .advance(EngineOutput::Token {
                request: stream.request,
                token: *token,
                index,
            })
            .await
            .expect("token advances");
    }
    stream.finish("stop");
    let chunks = chunks(&stream);
    assert!(
        chunks
            .iter()
            .all(|chunk| chunk["choices"][0]["delta"]["tool_calls"].is_null())
    );
    let content: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert!(content.contains("invented"), "{content:?}");
}

#[tokio::test]
async fn utf8_and_nonpublishing_tokens_still_count_as_completion_tokens() {
    let assets = byte_assets();
    let tokens = assets.encode("北京", false).unwrap();
    let mut stream = parsing_stream(&assets, &[]);
    for (index, token) in tokens.iter().enumerate() {
        stream
            .advance(EngineOutput::Token {
                request: stream.request,
                token: *token,
                index,
            })
            .await
            .unwrap();
    }
    assert_eq!(stream.completion_tokens, tokens.len());
}

#[test]
fn incomplete_calls_cannot_end_with_a_successful_tool_finish() {
    assert!(completion_reason("stop", true, true).is_err());
    assert_eq!(completion_reason("length", true, true).unwrap(), "length");
    assert_eq!(
        completion_reason("stop", false, true).unwrap(),
        "tool_calls"
    );
}

#[tokio::test]
async fn prompt_opened_reasoning_is_initialized_once_before_streaming_tokens() {
    let assets = byte_assets();
    let tokens = assets.encode("why</think>answer", false).unwrap();
    let mut stream = parsing_stream(&assets, &[]);
    stream.prompt_reasoning = Some(Arc::new(AtomicBool::new(true)));
    for (index, token) in tokens.iter().enumerate() {
        stream
            .advance(EngineOutput::Token {
                request: stream.request,
                token: *token,
                index,
            })
            .await
            .unwrap();
    }
    let events = stream.parser.as_mut().unwrap().finish().unwrap();
    for event in events {
        stream.publish(event);
    }
    let chunks = chunks(&stream);
    let reasoning: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["reasoning_content"].as_str())
        .collect();
    let content: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(reasoning, "why");
    assert_eq!(content, "answer");
    assert!(stream.prompt_reasoning.is_none());
    assert_eq!(stream.completion_tokens, tokens.len());
}
