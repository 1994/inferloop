use super::*;
use std::iter::once;

#[test]
fn the_reasoning_markers_match_the_served_package_tokens() {
    // Tokens 151667 and 151668 of the served packages spell these in plain ASCII.
    assert_eq!(REASONING_OPEN, "\u{3c}think\u{3e}");
    assert_eq!(REASONING_CLOSE, "\u{3c}/think\u{3e}");
    assert_eq!(TOOL_CALL_OPEN, "\u{3c}tool_call\u{3e}");
    assert_eq!(TOOL_CALL_CLOSE, "\u{3c}/tool_call\u{3e}");
}

#[test]
fn a_json_block_becomes_one_call_with_verbatim_arguments() {
    let parsed = parse(
            "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"units\": \"c\"}}\n</tool_call>\n",
        )
        .unwrap();
    assert_eq!(parsed.content, "Let me check.");
    assert_eq!(parsed.calls.len(), 1);
    assert_eq!(parsed.calls[0].name, "get_weather");
    assert_eq!(
        parsed.calls[0].arguments,
        r#"{"city": "Paris", "units": "c"}"#
    );
    assert!(!parsed.truncated && parsed.reasoning.is_none());
}

#[test]
fn a_string_argument_payload_must_be_a_json_object() {
    let parsed =
        parse(r#"<tool_call>{"name":"f","arguments":"{ not json }"}</tool_call>"#).unwrap();
    assert_eq!(parsed.calls.len(), 0);
    let parsed =
        parse(r#"<tool_call>{"name":"f","arguments":"{ \"x\": 1 }"}</tool_call>"#).unwrap();
    assert_eq!(parsed.calls[0].arguments, r#"{ "x": 1 }"#);
}

#[test]
fn multiple_calls_keep_their_order_and_leave_no_marker_text() {
    let parsed = parse(concat!(
        "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>",
        "between",
        "<tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>"
    ))
    .unwrap();
    assert_eq!(
        parsed
            .calls
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(parsed.content, "between");
}

#[test]
fn an_unterminated_block_is_withheld_instead_of_repaired() {
    let parsed = parse("answer<tool_call>{\"name\":\"a\",\"arguments\":{}").unwrap();
    assert_eq!(parsed.content, "answer");
    assert_eq!(parsed.calls.len(), 0);
    assert!(parsed.truncated);
}

#[test]
fn a_malformed_payload_stays_text() {
    let parsed = parse("<tool_call>{not json}</tool_call>").unwrap();
    assert_eq!(parsed.calls.len(), 0);
    assert!(parsed.content.contains(TOOL_CALL_OPEN));
    assert!(!parsed.truncated);
}

#[test]
fn a_payload_without_a_declared_name_is_not_a_call() {
    for body in [
        r#"{"arguments":{"a":1}}"#,
        r#"{"name":"","arguments":{}}"#,
        r#"{"name":"f","arguments":12}"#,
        r#"[{"name":"f"}]"#,
    ] {
        let text = format!("{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}");
        assert_eq!(parse(&text).unwrap().calls.len(), 0, "{body}");
    }
}

#[test]
fn a_leading_reasoning_block_is_separated_from_the_answer() {
    let text = format!("{REASONING_OPEN}weigh the options{REASONING_CLOSE}\n\nFinal answer.");
    let parsed = parse(&text).unwrap();
    assert_eq!(parsed.reasoning.as_deref(), Some("weigh the options"));
    assert_eq!(parsed.content, "Final answer.");
}

#[test]
fn an_unterminated_reasoning_block_publishes_nothing() {
    let text = format!("{REASONING_OPEN}still going");
    let parsed = parse(&text).unwrap();
    assert_eq!(parsed.reasoning.as_deref(), Some("still going"));
    assert_eq!(parsed.content, "");
    // The model ended its turn while thinking; that is an answer, not a malformed output.
    assert!(!parsed.truncated);
}

#[test]
fn ordinary_text_is_returned_unchanged() {
    let parsed = parse("plain answer").unwrap();
    assert_eq!(parsed.content, "plain answer");
    assert!(parsed.reasoning.is_none());
    assert_eq!(parsed.calls.len(), 0);
}

#[test]
fn oversized_arguments_are_rejected_rather_than_truncated() {
    let huge = "x".repeat(MAX_TOOL_ARGUMENT_BYTES + 1);
    let text =
        format!("{TOOL_CALL_OPEN}{{\"name\":\"f\",\"arguments\":\"{huge}\"}}{TOOL_CALL_CLOSE}");
    assert!(parse(&text).is_err());
}

#[test]
fn an_undeclared_name_is_left_as_text_instead_of_published() {
    let text = concat!(
        "<tool_call>{\"name\":\"declared\",\"arguments\":{}}</tool_call>",
        "<tool_call>{\"name\":\"invented\",\"arguments\":{}}</tool_call>"
    );
    let declared = once("declared".to_string()).collect();
    let parsed = parse_with(text, ToolDialect::JsonBlock, Some(&declared)).unwrap();
    assert_eq!(
        parsed
            .calls
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["declared"]
    );
    assert!(parsed.content.contains("invented"));
    // Without a declaration list every structurally valid call is published.
    assert_eq!(parse(text).unwrap().calls.len(), 2);
}

#[test]
fn the_dialect_is_detected_from_the_template_that_asks_for_it() {
    assert_eq!(
        ToolDialect::detect("<tool_call>\n<function=NAME>\n<parameter=K>"),
        ToolDialect::FunctionParameters
    );
    assert_eq!(
        ToolDialect::detect("{\"name\": <function-name>, \"arguments\": ...}"),
        ToolDialect::JsonBlock
    );
}

#[test]
fn a_function_block_becomes_one_call_with_typed_parameters() {
    let text = concat!(
        "Checking.\n<tool_call>\n<function=get_weather>\n",
        "<parameter=city>\nParis\n</parameter>\n",
        "<parameter=days>\n3\n</parameter>\n",
        "<parameter=notes>\nline one\nline two\n</parameter>\n",
        "</function>\n</tool_call>"
    );
    let parsed = parse_with(text, ToolDialect::FunctionParameters, None).unwrap();
    assert_eq!(parsed.content, "Checking.");
    assert_eq!(parsed.calls.len(), 1);
    assert_eq!(parsed.calls[0].name, "get_weather");
    let arguments: serde_json::Value = serde_json::from_str(&parsed.calls[0].arguments).unwrap();
    assert_eq!(arguments["city"], "Paris");
    // A bare number is valid JSON, so it stays a number; free text stays a string.
    assert_eq!(arguments["days"], 3);
    assert_eq!(arguments["notes"], "line one\nline two");
}

#[test]
fn a_function_block_without_parameters_is_an_empty_object() {
    let text = "<tool_call><function=ping>\n</function></tool_call>";
    let parsed = parse_with(text, ToolDialect::FunctionParameters, None).unwrap();
    assert_eq!(parsed.calls[0].arguments, "{}");
}

#[test]
fn malformed_function_blocks_are_left_as_text() {
    for body in [
        // No function header at all.
        "<parameter=city>\nParis\n</parameter>",
        // An empty function name.
        "<function=>\n<parameter=city>\nParis\n</parameter>",
        // A parameter that never closes.
        "<function=f>\n<parameter=city>\nParis",
        // A duplicate parameter would drop one value.
        "<function=f>\n<parameter=k>\n1\n</parameter>\n<parameter=k>\n2\n</parameter>",
        // A parameter without a name.
        "<function=f>\n<parameter=>\n1\n</parameter>",
    ] {
        let text = format!("{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}");
        let parsed = parse_with(&text, ToolDialect::FunctionParameters, None).unwrap();
        assert_eq!(parsed.calls.len(), 0, "{body}");
        assert!(
            parsed.content.contains(FUNCTION_MARKER) || parsed.content.contains(PARAMETER_MARKER)
        );
    }
}

#[test]
fn a_json_block_is_not_a_function_block_and_the_reverse() {
    let json = "<tool_call>{\"name\":\"f\",\"arguments\":{\"k\":1}}</tool_call>";
    let parsed = parse_with(json, ToolDialect::FunctionParameters, None).unwrap();
    assert_eq!(parsed.calls.len(), 0);
    let function =
        "<tool_call><function=f>\n<parameter=k>\n1\n</parameter>\n</function></tool_call>";
    let parsed = parse_with(function, ToolDialect::JsonBlock, None).unwrap();
    assert_eq!(parsed.calls.len(), 0);
}

#[test]
fn a_tool_call_argument_object_round_trips_for_the_template() {
    let call = ToolCall {
        id: "call_1".into(),
        name: "get_weather".into(),
        arguments: "{\"city\":\"Paris\"}".into(),
    };
    assert_eq!(call.argument_object().unwrap()["city"], "Paris");
    let malformed = ToolCall {
        arguments: "{not json}".into(),
        ..call
    };
    assert!(malformed.argument_object().is_err());
}

/// Feed `text` in fixed character groups and assemble the result.
fn feed_all(text: &str, dialect: ToolDialect, chunk: usize) -> ParsedOutput {
    let chars: Vec<char> = text.chars().collect();
    let mut parser = OutputStreamParser::new(dialect, None);
    let mut events = Vec::new();
    for piece in chars.chunks(chunk.max(1)) {
        let piece: String = piece.iter().collect();
        events.extend(parser.feed(&piece).unwrap());
    }
    events.extend(parser.finish().unwrap());
    assemble(events, parser.truncated())
}

fn samples() -> Vec<(ToolDialect, String)> {
    let json_call =
        format!("{TOOL_CALL_OPEN}{{\"name\":\"a\",\"arguments\":{{\"x\":1}}}}{TOOL_CALL_CLOSE}");
    let function_call = format!(
        "{TOOL_CALL_OPEN}\n<function=b>\n<parameter=k>\nv\n</parameter>\n</function>\n{TOOL_CALL_CLOSE}"
    );
    vec![
        (ToolDialect::JsonBlock, "plain answer".to_string()),
        (ToolDialect::JsonBlock, "  padded \n\n".to_string()),
        (ToolDialect::JsonBlock, format!("Sure.\n{json_call}\n")),
        (ToolDialect::JsonBlock, format!("{json_call}and{json_call}")),
        (
            ToolDialect::JsonBlock,
            format!("{TOOL_CALL_OPEN}{{not json}}{TOOL_CALL_CLOSE}"),
        ),
        (
            ToolDialect::JsonBlock,
            format!("{TOOL_CALL_OPEN}{{\"name\":\"a\",\"arguments\":{{}}"),
        ),
        (
            ToolDialect::FunctionParameters,
            format!("Checking.\n{function_call}\nDone."),
        ),
        (
            ToolDialect::FunctionParameters,
            format!("{TOOL_CALL_OPEN}<function=ping>\n</function>{TOOL_CALL_CLOSE}"),
        ),
        (
            ToolDialect::FunctionParameters,
            format!("{REASONING_OPEN}weigh it{REASONING_CLOSE}\n\nAnswer."),
        ),
        (
            ToolDialect::FunctionParameters,
            format!("{REASONING_OPEN}never closed"),
        ),
        (ToolDialect::FunctionParameters, String::new()),
        // Ordinary text that resembles a marker must stay text.
        (
            ToolDialect::JsonBlock,
            "use <tool_call> in prose, not </tool_call>".to_string(),
        ),
        (
            ToolDialect::JsonBlock,
            "a <tool_cal> typo and a trailing <tool_call".to_string(),
        ),
        (
            ToolDialect::FunctionParameters,
            format!("```json\n{json_call}\n```"),
        ),
        // Escapes and nesting inside the argument payload.
        (
            ToolDialect::JsonBlock,
            format!(
                "{TOOL_CALL_OPEN}{{\"name\":\"f\",\"arguments\":{{\"q\":\"a\\\"b\",\"n\":[1,{{\"k\":\"v\"}}]}}}}{TOOL_CALL_CLOSE}"
            ),
        ),
        // A truncated payload must not be repaired into a call.
        (
            ToolDialect::JsonBlock,
            format!("{TOOL_CALL_OPEN}{{\"name\":\"f\",\"arguments\":{{\"a\":1}}"),
        ),
    ]
}

#[test]
fn streaming_and_complete_parsing_agree_at_every_chunking() {
    for (dialect, text) in samples() {
        let complete = parse_with(&text, dialect, None).unwrap();
        for chunk in [1usize, 2, 3, 5, 17, 4096] {
            assert_eq!(
                feed_all(&text, dialect, chunk),
                complete,
                "chunk {chunk} disagreed on {text:?}"
            );
        }
    }
}

#[test]
fn a_marker_split_across_chunks_is_still_one_call() {
    let text = format!("hi{TOOL_CALL_OPEN}{{\"name\":\"a\",\"arguments\":{{}}}}{TOOL_CALL_CLOSE}");
    // Feed up to the middle of the opening marker, then the rest.
    let split = "hi<too".len();
    let mut parser = OutputStreamParser::new(ToolDialect::JsonBlock, None);
    let mut events = parser.feed(&text[..split]).unwrap();
    // The partial marker must be held, not published as text.
    assert_eq!(events, vec![StreamEvent::Text("hi".into())]);
    events.extend(parser.feed(&text[split..]).unwrap());
    events.extend(parser.finish().unwrap());
    assert_eq!(assemble(events, parser.truncated()).calls.len(), 1);
}

#[test]
fn a_call_is_published_as_soon_as_its_block_closes() {
    let mut parser = OutputStreamParser::new(ToolDialect::JsonBlock, None);
    let head = format!("{TOOL_CALL_OPEN}{{\"name\":\"a\",\"arguments\":{{}}}}");
    assert_eq!(parser.feed(&head).unwrap().len(), 0);
    let events = parser.feed(TOOL_CALL_CLOSE).unwrap();
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], StreamEvent::Call(_)));
}

#[test]
fn whitespace_around_a_call_is_never_published() {
    let mut parser = OutputStreamParser::new(ToolDialect::JsonBlock, None);
    let mut events = parser.feed("  ").unwrap();
    events.extend(parser.feed("\n").unwrap());
    assert_eq!(events.len(), 0);
    events.extend(parser.feed("answer").unwrap());
    assert_eq!(events, vec![StreamEvent::Text("answer".into())]);
    events.extend(parser.finish().unwrap());
    assert_eq!(assemble(events, parser.truncated()).content, "answer");
}

#[test]
fn a_reasoning_block_streams_before_the_answer() {
    let mut parser = OutputStreamParser::new(ToolDialect::FunctionParameters, None);
    let mut events = parser.feed(REASONING_OPEN).unwrap();
    assert!(events.is_empty(), "the marker alone is not reasoning yet");
    events.extend(parser.feed("why").unwrap());
    assert_eq!(events, vec![StreamEvent::Reasoning("why".into())]);
    events.extend(parser.feed(REASONING_CLOSE).unwrap());
    events.extend(parser.feed("Answer").unwrap());
    events.extend(parser.finish().unwrap());
    let parsed = assemble(events, parser.truncated());
    assert_eq!(parsed.reasoning.as_deref(), Some("why"));
    assert_eq!(parsed.content, "Answer");
}

#[test]
fn an_unterminated_call_is_withheld_without_publishing_partial_arguments() {
    let mut parser = OutputStreamParser::new(ToolDialect::JsonBlock, None);
    let mut events = parser.feed("text").unwrap();
    events.extend(parser.feed(TOOL_CALL_OPEN).unwrap());
    events.extend(parser.feed("{\"name\":\"a\",").unwrap());
    events.extend(parser.finish().unwrap());
    let parsed = assemble(events, parser.truncated());
    assert_eq!(parsed.content, "text");
    assert_eq!(parsed.calls.len(), 0);
    assert!(parsed.truncated);
}

#[test]
fn ordinary_text_that_resembles_a_marker_is_not_a_call() {
    for text in [
        "a </tool_call> without an opener",
        "a <tool_cal> typo",
        "a <tool_call without a closing bracket",
        "plain < and > characters",
        "use <tool_call> in prose, closed by </tool_call>",
    ] {
        let parsed = parse_with(text, ToolDialect::JsonBlock, None).unwrap();
        assert_eq!(parsed.calls.len(), 0, "{text}");
        assert!(!parsed.truncated, "{text}");
        // Whatever is not a call stays available as text.
        assert!(!parsed.content.is_empty(), "{text}");
    }
}

#[test]
fn a_bare_opener_withholds_the_tail_instead_of_publishing_half_a_call() {
    // An opener with no closer is indistinguishable from a truncated call, so the tail is
    // withheld and reported rather than published as text that might be half a call.
    let parsed = parse_with(
        "the <tool_call> tag appears in documentation",
        ToolDialect::JsonBlock,
        None,
    )
    .unwrap();
    assert_eq!(parsed.calls.len(), 0);
    assert!(parsed.truncated);
    assert_eq!(parsed.content, "the");
}

#[test]
fn a_marker_inside_a_code_fence_is_still_a_call() {
    // The dialect is not markdown-aware: a well-formed block is a call wherever it appears.
    let text = format!(
        "```json\n{TOOL_CALL_OPEN}{{\"name\":\"f\",\"arguments\":{{}}}}{TOOL_CALL_CLOSE}\n```"
    );
    let parsed = parse_with(&text, ToolDialect::JsonBlock, None).unwrap();
    assert_eq!(parsed.calls.len(), 1);
}

#[test]
fn escaped_and_nested_arguments_survive_verbatim() {
    let body = r#"{"name":"f","arguments":{"q":"a\"b","path":"C:\\tmp","n":[1,{"k":"v"}]}}"#;
    let text = format!("{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}");
    let parsed = parse_with(&text, ToolDialect::JsonBlock, None).unwrap();
    assert_eq!(parsed.calls.len(), 1);
    let arguments: serde_json::Value = serde_json::from_str(&parsed.calls[0].arguments).unwrap();
    assert_eq!(arguments["q"], "a\"b");
    assert_eq!(arguments["path"], "C:\\tmp");
    assert_eq!(arguments["n"][1]["k"], "v");
}

#[test]
fn an_unbalanced_payload_is_never_repaired() {
    for body in [
        r#"{"name":"f","arguments":{"a":1"#,
        r#"{"name":"f","arguments":{"a":}}}"#,
        r#"{"name":"f","arguments":}}"#,
    ] {
        let text = format!("{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}");
        let parsed = parse_with(&text, ToolDialect::JsonBlock, None).unwrap();
        assert_eq!(parsed.calls.len(), 0, "{body}");
        assert!(parsed.content.contains(TOOL_CALL_OPEN), "{body}");
    }
}

#[test]
fn a_utf8_character_split_across_chunks_is_reassembled_in_the_call() {
    let text = format!(
        "{TOOL_CALL_OPEN}{{\"name\":\"f\",\"arguments\":{{\"city\":\"北京\"}}}}{TOOL_CALL_CLOSE}"
    );
    let complete = parse_with(&text, ToolDialect::JsonBlock, None).unwrap();
    assert_eq!(feed_all(&text, ToolDialect::JsonBlock, 1), complete);
    let arguments: serde_json::Value = serde_json::from_str(&complete.calls[0].arguments).unwrap();
    assert_eq!(arguments["city"], "北京");
}

#[test]
fn a_function_requires_its_closer_and_only_its_declared_parameters() {
    for body in [
        "<function=f><parameter=x>1</parameter>",
        "noise<function=f></function>",
        "<function=f></function>noise",
        "<function=f></function><function=g></function>",
        "<function=f>garbage</function>",
    ] {
        let text = format!("{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}");
        for chunk in [1, 3, 17, 4096] {
            let parsed = feed_all(&text, ToolDialect::FunctionParameters, chunk);
            assert!(parsed.calls.is_empty(), "{text}");
            assert!(parsed.content.contains(body));
        }
    }
}

#[test]
fn delimiters_inside_json_strings_are_argument_data() {
    let text = r#"<tool_call>{"name":"f","arguments":{"text":"literal </tool_call> here","quote":"\\\""}}</tool_call>"#;
    for chunk in [1, 2, 7, 4096] {
        let parsed = feed_all(text, ToolDialect::JsonBlock, chunk);
        assert_eq!(parsed.calls.len(), 1, "{parsed:?}");
        assert!(
            parsed.calls[0]
                .arguments
                .contains("literal </tool_call> here")
        );
    }
}

fn streaming_events(text: &str, dialect: ToolDialect, chunk: usize) -> (Vec<StreamEvent>, bool) {
    let mut parser = OutputStreamParser::new(dialect, None).with_argument_deltas();
    let mut events = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    for chars in chars.chunks(chunk) {
        events.extend(parser.feed(&chars.iter().collect::<String>()).unwrap());
    }
    events.extend(parser.finish().unwrap());
    (events, parser.truncated())
}

#[test]
fn argument_deltas_assemble_to_the_same_validated_completion() {
    for (dialect, text) in samples() {
        // After publication, invalid generated bytes are terminal rather than retractable text.
        if !parse_with(&text, dialect, None).unwrap().calls.is_empty() {
            for chunk in [1, 3, 17, 4096] {
                let (events, truncated) = streaming_events(&text, dialect, chunk);
                assert_eq!(
                    assemble(events, truncated),
                    parse_with(&text, dialect, None).unwrap()
                );
            }
        }
    }
}

#[test]
fn json_arguments_publish_before_the_tool_block_closes() {
    let mut parser = OutputStreamParser::new(ToolDialect::JsonBlock, None).with_argument_deltas();
    let head = r#"<tool_call>{"name":"f","arguments":{"city":"北"#;
    let events = parser.feed(head).unwrap();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::CallStart { name, .. } if name == "f"))
    );
    assert!(events.iter().any(
        |e| matches!(e, StreamEvent::CallArguments { arguments, .. } if arguments.contains("北"))
    ));
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, StreamEvent::CallEnd { .. }))
    );
    let tail = "京\"}}</tool_call>";
    let events = parser.feed(tail).unwrap();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::CallEnd { index: 0 }))
    );
}

#[test]
fn function_parameters_publish_at_parameter_boundaries() {
    let mut parser =
        OutputStreamParser::new(ToolDialect::FunctionParameters, None).with_argument_deltas();
    let events = parser
        .feed("<tool_call><function=f><parameter=x>21</parameter>")
        .unwrap();
    let arguments: String = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::CallArguments { arguments, .. } => Some(arguments.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(arguments, "{\"x\":21");
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, StreamEvent::CallEnd { .. }))
    );
    assert!(
        parser
            .feed("</function></tool_call>")
            .unwrap()
            .iter()
            .any(|e| matches!(e, StreamEvent::CallEnd { index: 0 }))
    );
}

#[test]
fn invalid_generated_suffix_after_publication_cannot_become_plain_text() {
    for text in [
        r#"<tool_call>{"name":"f","arguments":{"x":1},"name":"g"}</tool_call>"#,
        r#"<tool_call>{"name":"f","arguments":{"x":1} garbage}</tool_call>"#,
    ] {
        let mut parser =
            OutputStreamParser::new(ToolDialect::JsonBlock, None).with_argument_deltas();
        let mut rejected = false;
        for ch in text.chars() {
            if parser.feed(&ch.to_string()).is_err() {
                rejected = true;
                break;
            }
        }
        assert!(rejected, "{text}");
    }
}

#[test]
fn tool_recognition_can_be_disabled_without_disabling_reasoning() {
    let mut parser = OutputStreamParser::new(ToolDialect::JsonBlock, None).with_tools(false);
    let mut events = parser
        .feed("<think>why</think><tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>")
        .unwrap();
    events.extend(parser.finish().unwrap());
    let parsed = assemble(events, parser.truncated());
    assert_eq!(parsed.calls.len(), 0);
    assert_eq!(parsed.reasoning.as_deref(), Some("why"));
    assert!(parsed.content.contains("<tool_call>"));
}

#[test]
fn unsupported_templates_have_no_tool_parser_capability() {
    assert!(ToolDialect::from_template("{{ messages }}").is_none());
    assert_eq!(
        ToolDialect::from_template("<tool_call>{\"name\":F,\"arguments\":A}</tool_call>"),
        Some(ToolDialect::JsonBlock)
    );
}

#[test]
fn literal_tool_closers_inside_function_values_are_data() {
    let text = "<tool_call><function=f><parameter=x>literal </tool_call> text</parameter></function></tool_call>";
    for chunk in [1, 3, 17, 4096] {
        let parsed = feed_all(text, ToolDialect::FunctionParameters, chunk);
        assert_eq!(parsed.calls.len(), 1);
        assert!(
            parsed.calls[0]
                .arguments
                .contains("literal </tool_call> text")
        );
    }
}

#[test]
fn json_escape_parity_survives_chunk_boundaries_and_literal_markers() {
    for slashes in 0..16 {
        let value = format!(
            "{}\"北</tool_call>{}",
            "\\".repeat(slashes),
            "\\".repeat(slashes)
        );
        let arguments = serde_json::json!({"text": value}).to_string();
        let text = format!("<tool_call>{{\"name\":\"f\",\"arguments\":{arguments}}}</tool_call>");
        let complete = parse(&text).unwrap();
        assert_eq!(complete.calls[0].arguments, arguments);
        for chunk in [1, 2, 3, 7, 1024] {
            let (events, truncated) = streaming_events(&text, ToolDialect::JsonBlock, chunk);
            assert_eq!(assemble(events, truncated), complete);
        }
    }
}

#[test]
fn reasoning_can_start_in_the_rendered_generation_prefix() {
    assert!(OutputStreamParser::prompt_opens_reasoning(
        "assistant\n<think>\n"
    ));
    assert!(!OutputStreamParser::prompt_opens_reasoning(
        "assistant\n<think>\n</think>\n"
    ));
    let text = "why</think><tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>";
    let mut parser = OutputStreamParser::new(ToolDialect::JsonBlock, None)
        .with_open_reasoning(true)
        .with_argument_deltas();
    let mut events = Vec::new();
    for ch in text.chars() {
        events.extend(parser.feed(&ch.to_string()).unwrap());
    }
    events.extend(parser.finish().unwrap());
    let parsed = assemble(events, parser.truncated());
    assert_eq!(parsed.reasoning.as_deref(), Some("why"));
    assert_eq!(parsed.content, "");
    assert_eq!(parsed.calls.len(), 1);
}
