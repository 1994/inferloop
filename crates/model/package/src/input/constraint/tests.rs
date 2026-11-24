use super::*;
use crate::ToolDialect;
use serde_json::json;

fn spec(required: bool, parallel: bool) -> ToolConstraints {
    ToolConstraints {
        required,
        parallel,
        named: None,
        tools: json!([
            {"type":"function","function":{"name":"weather","strict":true,"parameters":{
                "type":"object","properties":{"city":{"type":"string","enum":["Paris","北京"]}},
                "required":["city"],"additionalProperties":false}}},
            {"type":"function","function":{"name":"ping","strict":true,"parameters":{
                "type":"object","properties":{},"required":[],"additionalProperties":false}}}
        ]),
    }
}
fn matcher(spec: &ToolConstraints, dialect: ToolDialect, thinking: bool, open: bool) -> Matcher {
    let grammar = grammar::build(spec, dialect, thinking, open).unwrap();
    let factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env()).unwrap();
    let parser = factory
        .create_parser(TopLevelGrammar::from_lark(grammar))
        .unwrap();
    let matcher = Matcher::new(Ok(parser));
    assert!(matcher.get_error().is_none(), "{:?}", matcher.get_error());
    matcher
}
fn accepts(mut matcher: Matcher, text: &str) -> bool {
    for byte in text.bytes() {
        let Ok(mask) = matcher.compute_mask_or_eos() else {
            return false;
        };
        if mask.as_slice()[usize::from(byte) / 32] & (1 << (byte % 32)) == 0 {
            return false;
        }
        if matcher.consume_token(u32::from(byte)).is_err() {
            return false;
        }
    }
    matcher.is_accepting().unwrap_or(false) || (matcher.is_stopped() && !matcher.is_error())
}

#[test]
fn strict_json_blocks_enforce_schema_names_and_call_obligations() {
    let spec = spec(true, false);
    let parser = matcher(&spec, ToolDialect::JsonBlock, false, false);
    for text in [
        r#"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#,
        r#"Checking.<tool_call>{"name":"weather","arguments":{"city":"北京"}}</tool_call>Done."#,
        r#"<tool_call>{"name":"ping","arguments":{}}</tool_call>"#,
    ] {
        assert!(accepts(parser.deep_clone(), text), "{text}");
    }
    for text in [
        "Just text.",
        r#"<tool_call>{"name":"invented","arguments":{}}</tool_call>"#,
        r#"<tool_call>{"name":"weather","arguments":{"city":3}}</tool_call>"#,
        r#"<tool_call>{"name":"weather","arguments":{}}</tool_call>"#,
        r#"<tool_call>{"name":"weather","arguments":{"city":"London"}}</tool_call>"#,
        r#"<tool_call>{"name":"weather","arguments":{"city":"Paris","extra":true}}</tool_call>"#,
    ] {
        assert!(!accepts(parser.deep_clone(), text), "{text}");
    }
}
#[test]
fn auto_named_and_parallel_modes_have_distinct_languages() {
    let one = r#"<tool_call>{"name":"ping","arguments":{}}</tool_call>"#;
    let two = format!("{one}{one}");
    let parser = matcher(&spec(false, false), ToolDialect::JsonBlock, false, false);
    assert!(accepts(parser.deep_clone(), "ordinary answer < 3"));
    assert!(accepts(parser.deep_clone(), one));
    assert!(!accepts(parser, &two));
    let parser = matcher(&spec(true, true), ToolDialect::JsonBlock, false, false);
    assert!(accepts(parser, &two));
    let mut named = spec(false, true);
    named.named = Some("weather".into());
    let parser = matcher(&named, ToolDialect::JsonBlock, false, false);
    assert!(!accepts(parser.deep_clone(), "ordinary answer"));
    assert!(!accepts(parser.deep_clone(), one));
    let weather = r#"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#;
    assert!(accepts(parser.deep_clone(), weather));
    assert!(!accepts(parser, &format!("{weather}{weather}")));
}
#[test]
fn function_parameters_use_json_values_and_preserve_schema() {
    let parser = matcher(
        &spec(true, false),
        ToolDialect::FunctionParameters,
        false,
        false,
    );
    let valid = r#"<tool_call><function=weather><parameter=city>"Paris"</parameter></function></tool_call>"#;
    assert!(accepts(parser.deep_clone(), valid));
    assert!(!accepts(
        parser.deep_clone(),
        &valid.replace("\"Paris\"", "true")
    ));
    assert!(!accepts(
        parser,
        &valid.replace("<parameter=city>\"Paris\"</parameter>", "")
    ));
    let parsed =
        crate::parse_model_output_with(valid, ToolDialect::FunctionParameters, None).unwrap();
    assert_eq!(parsed.calls[0].arguments, r#"{"city":"Paris"}"#);
}
#[test]
fn reasoning_and_prompt_open_reasoning_do_not_satisfy_required_calls() {
    let call = r#"<tool_call>{"name":"ping","arguments":{}}</tool_call>"#;
    for open in [false, true] {
        let thought = if open {
            "why</think>"
        } else {
            "<think>why</think>"
        };
        let parser = matcher(&spec(true, false), ToolDialect::JsonBlock, true, open);
        assert!(accepts(parser.deep_clone(), &format!("{thought}{call}")));
        assert!(!accepts(parser, thought));
    }
}
#[test]
fn unknown_schema_keywords_and_invalid_strict_objects_are_rejected() {
    let mut value = spec(true, false);
    value.tools[0]["function"]["parameters"]["patternProperties"] = json!({});
    assert!(grammar::build(&value, ToolDialect::JsonBlock, false, false).is_err());
    value = spec(true, false);
    value.tools[0]["function"]["parameters"]["required"] = json!([]);
    assert!(grammar::build(&value, ToolDialect::JsonBlock, false, false).is_err());
}

#[test]
fn nested_nullable_arguments_and_marker_strings_preserve_their_values() {
    let mut spec = spec(true, false);
    spec.tools = json!([{"type":"function","function":{"name":"weather","strict":true,"parameters":{
        "type":"object","properties":{"data":{"type":"array","items":{"anyOf":[{"type":"string"},{"type":"null"}]}}},
        "required":["data"],"additionalProperties":false
    }}}]);
    let value = json!(["</parameter></tool_call>\"\\", null]);
    for dialect in [ToolDialect::JsonBlock, ToolDialect::FunctionParameters] {
        let text = match dialect {
            ToolDialect::JsonBlock => format!(
                "<tool_call>{}</tool_call>",
                json!({"name":"weather","arguments":{"data":value}})
            ),
            ToolDialect::FunctionParameters => format!(
                "<tool_call><function=weather><parameter=data>{value}</parameter></function></tool_call>"
            ),
        };
        assert!(accepts(matcher(&spec, dialect, false, false), &text));
        let parsed = crate::parse_model_output_with(&text, dialect, None).unwrap();
        assert_eq!(parsed.calls.len(), 1);
        let arguments: Value = serde_json::from_str(&parsed.calls[0].arguments).unwrap();
        assert_eq!(arguments["data"], value);
    }
}
#[test]
fn required_call_masks_eos_until_complete_and_cache_clones_are_independent() {
    let mut first = matcher(&spec(true, false), ToolDialect::JsonBlock, false, false);
    let mut second = first.deep_clone();
    assert!(!first.is_accepting().unwrap());
    for byte in br#"<tool_call>{"name":"ping","arguments":{}}</tool_call>"# {
        first.compute_mask_or_eos().unwrap();
        first.consume_token(u32::from(*byte)).unwrap();
    }
    assert!(first.is_accepting().unwrap());
    assert!(!second.is_accepting().unwrap());
}
