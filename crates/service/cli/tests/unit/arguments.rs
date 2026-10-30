use super::*;
use std::path::Path;

#[test]
fn model_path_serves_with_automatic_defaults() {
    let cli = Cli::try_parse_from(["infer", "/models/example"]).unwrap();
    assert_eq!(cli.backend, backend::BackendChoice::Auto);
    assert!(!cli.no_autotune);
    assert!(cli.num_speculative_tokens.is_none());
    assert!(Cli::try_parse_from(["infer", "/models/example", "--chunked-recurrent"]).is_err());
    assert!(cli.block_size.is_none());
    assert!(cli.max_num_batched_tokens.is_none());
    let Command::Serve(options) = cli.into_command().unwrap() else {
        panic!("expected server")
    };
    assert_eq!(
        options.package.as_deref(),
        Some(Path::new("/models/example"))
    );
    assert!(options.config.is_none());
    assert_eq!(options.listen.to_string(), "127.0.0.1:8080");
}

#[test]
fn speculation_distinguishes_automatic_selection_from_explicit_zero() {
    for args in [
        vec![
            "infer",
            "/models/example",
            "--num-speculative-tokens",
            "auto",
        ],
        vec![
            "infer",
            "serve",
            "--package",
            "/models/example",
            "--num-speculative-tokens",
            "auto",
        ],
    ] {
        let cli = Cli::try_parse_from(args).unwrap();
        assert_eq!(
            cli.num_speculative_tokens,
            Some(backend::SpeculativeTokens::Auto)
        );
    }
    let legacy = Cli::try_parse_from(["infer", "serve", "--package", "/models/example"]).unwrap();
    assert!(legacy.num_speculative_tokens.is_none());
    for invalid in ["automatic", "AUTO", "-1", "1.5", "18446744073709551616"] {
        assert!(
            Cli::try_parse_from([
                "infer",
                "/models/example",
                "--num-speculative-tokens",
                invalid
            ])
            .is_err()
        );
    }
    for depth in [0, 1, 2, 8] {
        let value = depth.to_string();
        let cli = Cli::try_parse_from([
            "infer",
            "/models/example",
            "--num-speculative-tokens",
            &value,
        ])
        .unwrap();
        assert_eq!(
            cli.num_speculative_tokens,
            Some(backend::SpeculativeTokens::Depth(depth))
        );
    }
}

#[test]
fn attention_profile_distinguishes_omission_and_rejects_unknown_values() {
    let default = Cli::try_parse_from(["infer", "/models/example"]).unwrap();
    assert!(default.cuda_attention_profile.is_none());
    for (name, expected) in [
        ("canonical", backend::CudaAttentionProfile::Canonical),
        ("bf16", backend::CudaAttentionProfile::Bf16),
        ("fp16", backend::CudaAttentionProfile::Fp16),
    ] {
        let cli =
            Cli::try_parse_from(["infer", "/models/example", "--cuda-attention-profile", name])
                .unwrap();
        assert_eq!(cli.cuda_attention_profile, Some(expected));
    }
    // The profile is a closed enum; a near-miss precision name is not a numerical option.
    assert!(
        Cli::try_parse_from([
            "infer",
            "/models/example",
            "--cuda-attention-profile",
            "f16",
        ])
        .is_err()
    );
}

#[test]
fn shorthand_preserves_overrides_and_legacy_commands() {
    for args in [
        vec![
            "infer",
            "./model",
            "--listen",
            "127.0.0.1:9000",
            "--config",
            "custom.json",
        ],
        vec![
            "infer",
            "serve",
            "--package",
            "./model",
            "--listen",
            "127.0.0.1:9000",
            "--config",
            "custom.json",
        ],
    ] {
        let Command::Serve(options) = Cli::try_parse_from(args).unwrap().into_command().unwrap()
        else {
            panic!("expected server")
        };
        assert_eq!(options.package.as_deref(), Some(Path::new("./model")));
        assert_eq!(options.config.as_deref(), Some(Path::new("custom.json")));
        assert_eq!(options.listen.port(), 9000);
    }
    assert!(matches!(
        Cli::try_parse_from(["infer", "--backend", "cuda", "doctor"])
            .unwrap()
            .into_command()
            .unwrap(),
        Command::Doctor
    ));
}

#[test]
fn ambiguous_or_missing_launch_input_is_rejected() {
    for args in [
        vec!["infer"],
        vec!["infer", "--listen", "127.0.0.1:9000"],
        vec!["infer", "./model", "--package", "./other"],
    ] {
        assert!(Cli::try_parse_from(args).is_err());
    }
    assert!(
        Cli::try_parse_from(["infer", "./model", "doctor"])
            .unwrap()
            .into_command()
            .is_err()
    );
    assert!(Cli::try_parse_from(["infer", "./model", "--backend", "cuda"]).is_ok());
    assert!(Cli::try_parse_from(["infer", "--backend", "cuda", "./model"]).is_ok());
    assert!(Cli::try_parse_from(["infer", "--", "./serve"]).is_ok());
}

#[test]
fn deployment_length_limits_are_parsed_as_separate_budgets() {
    let cli = Cli::try_parse_from([
        "infer",
        "./model",
        "--max-model-len",
        "8192",
        "--max-output-tokens",
        "1024",
    ])
    .unwrap();
    assert_eq!(cli.max_model_len, Some(8192));
    assert_eq!(cli.max_output_tokens, Some(1024));
    // Zero is a configuration error the length resolver reports, not a clap special value.
    let zero = Cli::try_parse_from(["infer", "./model", "--max-model-len", "0"]).unwrap();
    assert_eq!(zero.max_model_len, Some(0));
    // The total context and output cap stay distinct from the per-step batching budget.
    let batching =
        Cli::try_parse_from(["infer", "./model", "--max-num-batched-tokens", "2048"]).unwrap();
    assert_eq!(batching.max_num_batched_tokens, Some(2048));
    assert!(batching.max_model_len.is_none());
    assert!(batching.max_output_tokens.is_none());
}

#[test]
fn capacity_and_identity_overrides_are_parsed() {
    let cli = Cli::try_parse_from([
        "infer",
        "./model",
        "--max-num-seqs",
        "8",
        "--served-model-name",
        "qwen3-8-27b",
    ])
    .unwrap();
    assert_eq!(cli.max_num_seqs, Some(8));
    assert_eq!(cli.served_model_name.as_deref(), Some("qwen3-8-27b"));
    let default = Cli::try_parse_from(["infer", "./model"]).unwrap();
    assert!(default.max_num_seqs.is_none());
    assert!(default.served_model_name.is_none());
}

#[test]
fn multimodal_limits_are_parsed_and_all_zero_means_text_only() {
    let cli = Cli::try_parse_from([
        "infer",
        "/models/example",
        "--limit-mm-per-prompt",
        r#"{"image":2,"video":1}"#,
    ])
    .unwrap();
    assert_eq!(
        cli.limit_mm_per_prompt,
        Some(infer_models::MediaLimits {
            images: 2,
            videos: 1
        })
    );
    let text_only = Cli::try_parse_from([
        "infer",
        "/models/example",
        "--limit-mm-per-prompt",
        r#"{"image":0,"video":0}"#,
    ])
    .unwrap();
    assert!(text_only.limit_mm_per_prompt.unwrap().text_only());
    for invalid in [
        r#"{"audio":1}"#,
        r#"{"image":1,"video":1,"audio":0}"#,
        r#"{"image":-1}"#,
        r#"{"image":"one"}"#,
        "{}",
        "not-json",
    ] {
        assert!(
            Cli::try_parse_from(["infer", "/models/example", "--limit-mm-per-prompt", invalid])
                .is_err(),
            "{invalid}"
        );
    }
}

#[test]
fn image_processor_options_are_accepted_by_serve() {
    let options = r#"{"min_pixels":1024,"max_pixels":65536}"#;
    let cli = Cli::try_parse_from(["infer", "/models/example", "--mm-processor-kwargs", options])
        .unwrap();
    assert_eq!(cli.serve.mm_processor_kwargs.as_deref(), Some(options));
}

#[test]
fn image_decoder_override_accepts_cpu_and_rejects_unknown_modes() {
    let default = Cli::try_parse_from(["infer", "/models/example"]).unwrap();
    assert_eq!(default.serve.image_decoder, commands::ImageDecoder::Auto);
    for args in [
        vec!["infer", "/models/example", "--image-decoder", "cpu"],
        vec![
            "infer",
            "serve",
            "--package",
            "/models/example",
            "--image-decoder",
            "cpu",
        ],
    ] {
        let Command::Serve(options) = Cli::try_parse_from(args).unwrap().into_command().unwrap()
        else {
            panic!("expected server")
        };
        assert_eq!(options.image_decoder, commands::ImageDecoder::Cpu);
    }
    assert!(
        Cli::try_parse_from(["infer", "/models/example", "--image-decoder", "invalid"]).is_err()
    );
}
