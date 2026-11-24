use super::*;
use crate::arguments::Cli;
use clap::Parser;

#[test]
fn omitted_cli_options_require_the_complete_fast_capability_gate() {
    let cli = Cli::try_parse_from(["infer", "/models/example"]).unwrap();
    assert_eq!(
        resolve(cli.num_speculative_tokens, cli.cuda_attention_profile, true),
        (SpeculativeTokens::Auto, CudaAttentionProfile::Bf16),
    );
    // A missing head, unsupported geometry/device or non-FP8 KV cannot change prior loadability.
    assert_eq!(
        resolve(
            cli.num_speculative_tokens,
            cli.cuda_attention_profile,
            false
        ),
        (SpeculativeTokens::Depth(0), CudaAttentionProfile::Canonical),
    );
}

#[test]
fn explicit_cli_execution_overrides_preserve_the_other_conservative_option() {
    for (args, expected) in [
        (
            vec!["--num-speculative-tokens", "0"],
            (SpeculativeTokens::Depth(0), CudaAttentionProfile::Canonical),
        ),
        (
            vec!["--num-speculative-tokens", "2"],
            (SpeculativeTokens::Depth(2), CudaAttentionProfile::Canonical),
        ),
        (
            vec!["--num-speculative-tokens", "auto"],
            (SpeculativeTokens::Auto, CudaAttentionProfile::Canonical),
        ),
        (
            vec!["--cuda-attention-profile", "canonical"],
            (SpeculativeTokens::Depth(0), CudaAttentionProfile::Canonical),
        ),
        (
            vec!["--cuda-attention-profile", "bf16"],
            (SpeculativeTokens::Depth(0), CudaAttentionProfile::Bf16),
        ),
        (
            vec!["--cuda-attention-profile", "fp16"],
            (SpeculativeTokens::Depth(0), CudaAttentionProfile::Fp16),
        ),
        (
            vec![
                "--num-speculative-tokens",
                "2",
                "--cuda-attention-profile",
                "bf16",
            ],
            (SpeculativeTokens::Depth(2), CudaAttentionProfile::Bf16),
        ),
    ] {
        let cli =
            Cli::try_parse_from(["infer", "/models/example"].into_iter().chain(args)).unwrap();
        for eligible in [false, true] {
            assert_eq!(
                resolve(
                    cli.num_speculative_tokens,
                    cli.cuda_attention_profile,
                    eligible
                ),
                expected
            );
        }
    }
}
