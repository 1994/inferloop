use super::*;
use infer_ir::Mixer;
use std::fs::read;
use std::path::Path;

fn geometry(taps: Vec<usize>) -> DraftGeometry {
    DraftGeometry {
        target_taps: taps,
        ..DraftGeometry::official()
    }
}

fn decoder(hidden_size: usize, layers: usize) -> ModelIr {
    ModelIr {
        id: infer_core::ModelId::ONE,
        backbone: BackboneKind::Decoder,
        vocab_size: 32,
        hidden_size,
        max_sequence: 128,
        mixers: vec![
            Mixer::Attention {
                query_heads: 1,
                kv_heads: 1,
                head_dim: hidden_size,
                sliding_window: None,
                output_gate: false,
                qk_norm: false,
            };
            layers
        ],
        feed_forward: infer_ir::FeedForward::Dense { intermediate: 16 },
        position: infer_ir::PositionSpec {
            rope_theta: 10_000.0,
            rotary_fraction: 1.0,
            multimodal_sections: vec![],
            interleaved: false,
        },
        norm_epsilon: 1e-5,
        norm_weight_offset: 0.0,
        heads: vec![infer_ir::Head::LanguageModel],
        modalities: vec![infer_ir::Modality::Text],
        state: vec![],
        tied_embeddings: false,
    }
}

#[test]
fn the_official_geometry_validates() {
    DraftGeometry::official().validate().unwrap();
}

#[test]
fn key_value_heads_must_divide_the_attention_heads() {
    let broken = DraftGeometry {
        attention_heads: 32,
        key_value_heads: 5,
        ..DraftGeometry::official()
    };
    assert!(broken.validate().is_err());
}

#[test]
fn taps_must_be_a_non_empty_ascending_list() {
    assert!(geometry(vec![]).validate().is_err());
    assert!(geometry(vec![5, 5]).validate().is_err());
    assert!(geometry(vec![19, 5]).validate().is_err());
}

#[test]
fn the_selector_may_not_take_more_candidates_than_it_scores() {
    let broken = DraftGeometry {
        selector_top_k: 0,
        ..DraftGeometry::official()
    };
    assert!(broken.validate().is_err());
    let broken = DraftGeometry {
        selector_top_k: 512,
        ..DraftGeometry::official()
    };
    assert!(broken.validate().is_err());
}

#[test]
fn a_block_wider_than_the_window_is_rejected() {
    let broken = DraftGeometry {
        block_size: 4096,
        ..DraftGeometry::official()
    };
    assert!(broken.validate().is_err());
}

#[test]
fn the_target_hidden_size_must_match_the_draft() {
    let draft = DraftGeometry::official();
    draft.check_target(&decoder(5120, 64)).unwrap();
    let error = draft.check_target(&decoder(4096, 64)).unwrap_err();
    assert!(error.to_string().contains("hidden"), "{error}");
}

#[test]
fn the_target_layer_count_must_match_the_draft() {
    let draft = DraftGeometry::official();
    // The largest official tap is 61, but the layer count is checked first: a 61-layer target is
    // not the 64-layer target the draft was trained against.
    let error = draft.check_target(&decoder(5120, 61)).unwrap_err();
    assert!(error.to_string().contains("64 target layers"), "{error}");
}

#[test]
fn a_tap_reads_the_residual_stream_of_any_mixer() {
    let draft = DraftGeometry::official();
    let mut target = decoder(5120, 64);
    // Qwen3.8 taps recurrent layers: the residual stream after a layer does not depend on its mixer.
    target.backbone = BackboneKind::Hybrid;
    target.mixers[5] = Mixer::Recurrent { state_width: 128 };
    draft.check_target(&target).unwrap();
}

#[test]
fn a_tap_past_the_last_target_layer_is_rejected() {
    let error = geometry(vec![5, 70])
        .check_target(&decoder(5120, 64))
        .unwrap_err();
    assert!(error.to_string().contains("tap 70"), "{error}");
}

#[test]
fn a_non_decoder_target_is_rejected() {
    let draft = DraftGeometry::official();
    let mut target = decoder(5120, 64);
    target.backbone = BackboneKind::Encoder;
    assert!(draft.check_target(&target).is_err());
}

#[test]
fn an_unsupported_operation_is_named() {
    let draft = DraftGeometry::official();
    let mut missing: BTreeSet<Operation> = REQUIRED_OPERATIONS.into_iter().collect();
    draft.check_operations(&missing).unwrap();
    missing.remove(&Operation::Convolution);
    let error = draft.check_operations(&missing).unwrap_err();
    assert!(error.to_string().contains("Convolution"), "{error}");
}

#[test]
fn the_quote_scales_with_geometry_and_precision() {
    let config = fixture();
    let quote = config.quote(2);
    // Five layers, eight KV heads of 128 values, K and V, two bytes each.
    assert_eq!(quote.kv_bytes_per_token, 5 * 8 * 128 * 2 * 2);
    assert_eq!(quote.window_kv_bytes, quote.kv_bytes_per_token * 2048);
    // Five taps of hidden 5120 at two bytes.
    assert_eq!(quote.feature_bytes_per_token, 5 * 5120 * 2);
    let wide = config.quote(4);
    assert_eq!(wide.kv_bytes_per_token, quote.kv_bytes_per_token * 2);
    assert_eq!(
        wide.feature_bytes_per_token,
        quote.feature_bytes_per_token * 2
    );
}

#[test]
fn the_quote_covers_the_state_a_draft_sequence_carries() {
    let config = fixture();
    let quote = config.quote(2);
    // Weights are the same number the inventory sums to, so the quote and the checks agree.
    assert_eq!(quote.weights_bytes, config.weights_bytes(2));
    assert_eq!(quote.weights_bytes, 3_848_808_960);
    // Two dynamic convolutions per layer, each keeping `kernel - 1` slots over the hidden channels.
    assert_eq!(quote.conv_history_bytes_per_sequence, 5 * 2 * 5120 * 2);
    // A round that accepts only its first candidate discards the rest of the block.
    assert_eq!(
        quote.rollback_bytes_per_round,
        quote.kv_bytes_per_token * (8 - 1)
    );
    // A sequence's resident state is its window plus the convolution histories; the features a
    // sequence retains grow with the tokens it has seen, which is what the per-token figure is for.
    let resident_without_features = quote.window_kv_bytes + quote.conv_history_bytes_per_sequence;
    // K and V for the 2048-token window, plus both convolution histories.
    assert_eq!(resident_without_features, 41_943_040 + 102_400);
    assert!(quote.feature_bytes_per_token > quote.kv_bytes_per_token / 2);
}

#[test]
fn another_architecture_does_not_answer_an_explicit_dflash2_request() {
    check_declared_architecture(&[ARCHITECTURE.to_owned()]).unwrap();
    let error = check_declared_architecture(&["Qwen3ForCausalLM".to_owned()]).unwrap_err();
    assert!(error.to_string().contains(ARCHITECTURE), "{error}");
}

fn fixture() -> DFlash2Config {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/qwen3.8-27b-dflash2/config.json");
    DFlash2Config::parse(&read(path).unwrap()).unwrap()
}

#[test]
fn the_registered_fixture_parses_to_the_official_geometry() {
    let config = fixture();
    assert_eq!(config.geometry(), DraftGeometry::official());
    config.geometry().validate().unwrap();
    config.check_block_semantics().unwrap();
}

#[test]
fn the_inventory_covers_the_released_draft() {
    let config = fixture();
    let expected = config.expected_weight_shapes();
    // The released checkpoint has 81 tensors: six shared and fifteen per draft layer.
    assert_eq!(expected.len(), 6 + 15 * config.num_hidden_layers);
    assert_eq!(expected["fc.weight"], vec![5120, 25600]);
    assert_eq!(
        expected["layers.0.self_attn.q_proj.weight"],
        vec![4096, 5120]
    );
    assert_eq!(
        expected["layers.0.self_attn.k_proj.weight"],
        vec![1024, 5120]
    );
    assert_eq!(expected["layers.4.mlp.down_proj.weight"], vec![5120, 17408]);
    // The convolution shapes come from the reference implementation's layout.
    assert_eq!(
        expected["layers.0.attention_conv.base_kernel"],
        vec![2, 2, 5120]
    );
    assert_eq!(
        expected["layers.0.mlp_conv.kernel_projection.weight"],
        vec![1280, 5120]
    );
}

#[test]
fn a_complete_inventory_is_accepted_and_each_shortfall_is_named() {
    let config = fixture();
    let mut actual: BTreeMap<String, Vec<usize>> =
        config.expected_weight_shapes().into_iter().collect();
    config.check_weight_inventory(&actual).unwrap();

    let complete = actual.clone();
    let missing = actual.keys().next().unwrap().clone();
    actual.remove(&missing);
    let error = config.check_weight_inventory(&actual).unwrap_err();
    assert!(error.to_string().contains(&missing), "{error}");

    actual = complete;
    actual.insert("layers.0.mystery.weight".to_owned(), vec![1]);
    let error = config.check_weight_inventory(&actual).unwrap_err();
    assert!(error.to_string().contains("unknown tensor"), "{error}");

    actual.remove("layers.0.mystery.weight");
    actual.insert(
        "layers.0.self_attn.q_proj.weight".to_owned(),
        vec![4096, 4096],
    );
    let error = config.check_weight_inventory(&actual).unwrap_err();
    assert!(error.to_string().contains("q_proj"), "{error}");
}

#[test]
fn a_causal_or_tied_draft_configuration_is_rejected() {
    let mut causal = fixture();
    causal.is_causal = true;
    assert!(causal.check_block_semantics().is_err());
    let mut tied = fixture();
    tied.tie_word_embeddings = true;
    assert!(tied.check_block_semantics().is_err());
}

#[test]
fn a_configuration_for_another_architecture_does_not_parse() {
    let bytes = read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen3.8-27b/config.json"),
    )
    .unwrap();
    assert!(DFlash2Config::parse(&bytes).is_err());
}

fn golden() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/qwen3.8-27b-dflash2/fusion-golden.json");
    serde_json::from_slice(&read(path).unwrap()).unwrap()
}

#[test]
fn the_fusion_golden_matches_the_registered_contract() {
    let config = fixture();
    let geometry = config.geometry();
    let golden = golden();

    // The golden is only usable if it came from the same configuration the fixture registers.
    let digest = {
        use sha2::{Digest, Sha256};
        let bytes = read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../examples/qwen3.8-27b-dflash2/config.json"),
        )
        .unwrap();
        format!("{:x}", Sha256::digest(&bytes))
    };
    assert_eq!(golden["model"]["config_sha256"], digest);
    let taps: Vec<usize> = golden["target_layer_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| usize::try_from(value.as_u64().unwrap()).unwrap())
        .collect();
    assert_eq!(taps, geometry.target_taps);
    assert_eq!(
        golden["fused"].as_array().unwrap().len(),
        geometry.hidden_size
    );
    let expected = config.expected_weight_shapes();
    let shape: Vec<usize> = golden["model"]["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| usize::try_from(value.as_u64().unwrap()).unwrap())
        .collect();
    assert_eq!(shape, expected["fc.weight"]);
}

#[test]
fn the_fusion_projection_is_recorded_in_float32() {
    let golden = golden();
    assert_eq!(golden["model"]["tensor"], "fc.weight");
    assert_eq!(golden["model"]["dtype"], "BF16");
    assert!(
        golden["formula"]["projection"]
            .as_str()
            .unwrap()
            .contains("no bias"),
        "the golden records the projection's semantics"
    );
    assert!(golden["fused"].as_array().unwrap().iter().all(|value| {
        let value = value.as_f64().unwrap();
        value.is_finite()
    }));
}

fn selector_golden() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/qwen3.8-27b-dflash2/selector-golden.json");
    serde_json::from_slice(&read(path).unwrap()).unwrap()
}

#[test]
fn the_selector_golden_matches_the_registered_contract() {
    let config = fixture();
    let geometry = config.geometry();
    let golden = selector_golden();
    let expected = config.expected_weight_shapes();

    assert_eq!(
        golden["rank"].as_u64().unwrap(),
        geometry.selector_rank as u64
    );
    assert_eq!(
        golden["top_k"].as_u64().unwrap(),
        geometry.selector_top_k as u64
    );
    for tensor in [
        "candidate_selector.hidden_projection.weight",
        "candidate_selector.predecessor_codebook",
        "candidate_selector.successor_codebook",
    ] {
        let shape: Vec<usize> = golden["model"]["tensors"][tensor]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| usize::try_from(value.as_u64().unwrap()).unwrap())
            .collect();
        assert_eq!(shape, expected[tensor], "{tensor}");
    }
    let scores = golden["scores"].as_array().unwrap();
    assert_eq!(scores.len(), geometry.selector_top_k);
    for row in scores {
        assert_eq!(row.as_array().unwrap().len(), geometry.selector_top_k);
    }
    let logits = golden["inputs"]["unary_logits"].as_array().unwrap();
    assert_eq!(logits[0], logits[1], "the recorded tie stays visible");
    assert_eq!(
        golden["inputs"]["candidate_ids"].as_array().unwrap().len(),
        geometry.selector_top_k
    );
    assert!(golden["scores"].as_array().unwrap().iter().all(|row| {
        row.as_array()
            .unwrap()
            .iter()
            .all(|value| value.as_f64().unwrap().is_finite())
    }));
}

#[test]
fn the_conv_golden_matches_the_registered_contract() {
    let config = fixture();
    let geometry = config.geometry();
    let draft = &config.dflash_config;
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/qwen3.8-27b-dflash2/conv-golden.json");
    let golden: serde_json::Value = serde_json::from_slice(&read(path).unwrap()).unwrap();

    // The convolution's own parameters come from the configuration, not from the exporter.
    assert_eq!(
        golden["block_size"].as_u64().unwrap(),
        geometry.block_size as u64
    );
    assert_eq!(
        golden["group_size"].as_u64().unwrap(),
        draft.conv_group_size as u64
    );
    assert_eq!(
        golden["taps"].as_u64().unwrap(),
        draft.conv_kernel_size as u64
    );
    assert_eq!(golden["rows"].as_u64().unwrap(), geometry.block_size as u64);

    // Both tensors the convolution reads are names the inventory already requires.
    let expected = config.expected_weight_shapes();
    let tensors = golden["model"]["tensors"].as_object().unwrap();
    for name in tensors.keys() {
        assert!(expected.contains_key(name), "{name} is not a draft tensor");
    }
    let projection: Vec<usize> = tensors
        .iter()
        .find(|(name, _)| name.ends_with("kernel_projection.weight"))
        .map(|(_, shape)| {
            shape
                .as_array()
                .unwrap()
                .iter()
                .map(|value| usize::try_from(value.as_u64().unwrap()).unwrap())
                .collect()
        })
        .unwrap();
    // `[2 * taps * groups, hidden]`, which is where the released 1280 comes from.
    let groups = geometry.hidden_size / draft.conv_group_size;
    assert_eq!(
        projection,
        vec![2 * draft.conv_kernel_size * groups, geometry.hidden_size]
    );

    let output = golden["output"].as_array().unwrap();
    assert_eq!(output.len(), geometry.block_size);
    for row in output {
        assert_eq!(row.as_array().unwrap().len(), geometry.hidden_size);
        assert!(
            row.as_array()
                .unwrap()
                .iter()
                .all(|value| value.as_f64().unwrap().is_finite())
        );
    }
}

#[test]
fn the_weight_quote_reproduces_the_released_checkpoint_exactly() {
    let config = fixture();
    // The released `model.safetensors` holds 81 bfloat16 tensors totalling 3,848,808,960 bytes.
    // Every shape in the inventory is derived, so this one number checks all of them at once: a
    // wrong group count, tap count or head geometry would miss it.
    assert_eq!(config.weights_bytes(2), 3_848_808_960);
    // The quote has to follow the dtype it is asked for.
    assert_eq!(config.weights_bytes(4), 2 * config.weights_bytes(2));
    // The draft's own footprint, for the report: 1.92 B parameters.
    assert_eq!(config.weights_bytes(2) / 2, 1_924_404_480);
}

#[test]
fn a_query_sees_its_own_block_in_both_directions() {
    let draft = DraftGeometry::official();
    // Block 1 spans positions 8..16: its tokens see each other forwards and backwards.
    assert!(draft.attends(8, 9));
    assert!(draft.attends(15, 8));
    assert_eq!(draft.block_of(8), 1);
    assert_eq!(draft.block_of(15), 1);
}

#[test]
fn a_query_does_not_see_later_blocks() {
    let draft = DraftGeometry::official();
    assert!(!draft.attends(7, 8));
    assert!(!draft.attends(15, 16));
    // ... but it does see the whole of an earlier block, up to the window.
    assert!(draft.attends(16, 15));
    assert!(draft.attends(16, 8));
}

#[test]
fn the_sliding_window_bounds_how_far_back_a_query_reaches() {
    let draft = DraftGeometry::official();
    let window = DraftGeometry::official().sliding_window;
    assert!(draft.attends(window, 1));
    assert!(!draft.attends(window, 0));
    assert!(!draft.attends(window + 1, 1));
    // The furthest visible key is exactly window - 1 positions back.
    assert!(draft.attends(window + 1, 2));
}

#[test]
fn a_draft_block_never_exceeds_its_window() {
    // `validate` keeps the block inside the window, so within-block visibility never has to consult
    // the window at all.
    let draft = DraftGeometry::official();
    assert!(draft.block_size <= draft.sliding_window);
}

fn target_config() -> serde_json::Value {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen3.8-27b/config.json");
    serde_json::from_slice(&read(path).unwrap()).unwrap()
}

#[test]
fn the_registered_fixture_declares_its_special_tokens() {
    let config = fixture();
    config.check_special_tokens().unwrap();
    assert_eq!(config.dflash_config.mask_token_id, Some(248_070));
    assert_eq!(config.eos_token_id, Some(248_044));
    assert_eq!(config.pad_token_id, Some(248_044));
    assert!(config.dflash_config.mask_token_id.unwrap() < config.vocab_size);
}

#[test]
fn a_draft_without_a_mask_token_or_outside_its_vocabulary_is_rejected() {
    let mut absent = fixture();
    absent.dflash_config.mask_token_id = None;
    assert!(absent.check_special_tokens().is_err());

    let mut outside = fixture();
    outside.dflash_config.mask_token_id = Some(outside.vocab_size);
    let error = outside.check_special_tokens().unwrap_err();
    assert!(
        error.to_string().contains("outside its vocabulary"),
        "{error}"
    );

    let mut bad_eos = fixture();
    bad_eos.eos_token_id = Some(bad_eos.vocab_size + 1);
    assert!(bad_eos.check_special_tokens().is_err());
}

#[test]
fn the_draft_shares_its_targets_token_boundaries() {
    let config = fixture();
    let target = target_config();
    let text = &target["text_config"];
    let eos = usize::try_from(text["eos_token_id"].as_u64().unwrap()).unwrap();
    // The target declares no padding token, so only the end-of-sequence boundary is compared.
    let pad = text["pad_token_id"]
        .as_u64()
        .map(|value| usize::try_from(value).unwrap());
    config.check_target_tokens(eos, pad).unwrap();
    assert_eq!(eos, 248_044);
    // The vocabularies have to agree as well, since the draft's head scores the target's tokens.
    assert_eq!(
        text["vocab_size"].as_u64().unwrap(),
        config.vocab_size as u64
    );

    let error = config.check_target_tokens(eos + 1, pad).unwrap_err();
    assert!(
        error.to_string().contains("does not match the target"),
        "{error}"
    );
}

#[test]
fn an_admission_says_what_it_could_not_price() {
    let quote = fixture().quote(2);
    let admission = quote.admit(quote.required_bytes(1, 2048), 1, 2048);
    // The plan's resource row also asks for the arena and the captured graph; both need device
    // facts, so the result names them instead of implying the total is complete.
    assert_eq!(
        DraftQuote::deferred(),
        vec![DraftCost::Arena, DraftCost::Graph]
    );
    assert_eq!(admission.deferred, DraftQuote::deferred());
}

#[test]
fn the_required_bytes_are_the_components_of_the_quote() {
    let quote = fixture().quote(2);
    assert_eq!(
        quote.required_bytes(3, 512),
        quote.weights_bytes
            + 3 * (quote.window_kv_bytes + quote.conv_history_bytes_per_sequence)
            + 512 * (quote.kv_bytes_per_token + quote.feature_bytes_per_token)
            + quote.rollback_bytes_per_round
    );
}

#[test]
fn a_budget_that_covers_the_request_admits_it() {
    let quote = fixture().quote(2);
    let required = quote.required_bytes(1, 2048);
    for budget in [required, required + 1, required * 2] {
        let admission = quote.admit(budget, 1, 2048);
        assert!(admission.admitted(), "{admission:?}");
        assert_eq!(admission.shortfall, Vec::<DraftCost>::new());
        assert_eq!(admission.required_bytes, required);
        assert_eq!(admission.budget_bytes, budget);
    }
}

#[test]
fn a_shortfall_names_what_the_budget_does_not_cover() {
    let quote = fixture().quote(2);
    // Components are charged weights, window, cache, rollback; each of these budgets reaches one
    // step further than the last, so the uncovered list is exactly the remainder.
    let weights = quote.weights_bytes;
    let window = quote.window_kv_bytes + quote.conv_history_bytes_per_sequence;
    let cache = 2048 * (quote.kv_bytes_per_token + quote.feature_bytes_per_token);
    assert_eq!(
        quote.admit(weights - 1, 1, 2048).shortfall,
        vec![
            DraftCost::Weights,
            DraftCost::Window,
            DraftCost::Cache,
            DraftCost::Rollback
        ]
    );
    assert_eq!(
        quote.admit(weights, 1, 2048).shortfall,
        vec![DraftCost::Window, DraftCost::Cache, DraftCost::Rollback]
    );
    assert_eq!(
        quote.admit(weights + window, 1, 2048).shortfall,
        vec![DraftCost::Cache, DraftCost::Rollback]
    );
    assert_eq!(
        quote.admit(weights + window + cache, 1, 2048).shortfall,
        vec![DraftCost::Rollback]
    );
    let covered = weights + window + cache + quote.rollback_bytes_per_round;
    assert!(quote.admit(covered, 1, 2048).admitted());
    assert!(!quote.admit(covered - 1, 1, 2048).admitted());
}

#[test]
fn the_released_draft_is_admitted_or_rejected_with_a_reason() {
    let quote = fixture().quote(2);
    // The BF16 draft needs a little over 3.8 GiB of weights, so a 24 GiB device holds it and a
    // 4 GiB device still does - the window, cache and rollback buffer fit beside the weights.
    let admitted = quote.admit(24 * 1024 * 1024 * 1024, 1, 2048);
    assert!(admitted.admitted(), "{admitted:?}");
    let tight = quote.admit(4 * 1024 * 1024 * 1024, 1, 2048);
    assert!(tight.admitted(), "{tight:?}");
    assert!(tight.required_bytes <= tight.budget_bytes);
    // One byte more than the weights is enough to load them and nothing else, and the result says
    // which parts are missing rather than only that it failed.
    let weights_only = quote.admit(quote.weights_bytes + 1, 1, 2048);
    assert!(!weights_only.admitted());
    assert_eq!(
        weights_only.shortfall,
        vec![DraftCost::Window, DraftCost::Cache, DraftCost::Rollback]
    );
    // One byte less than the weights names the weights first.
    let too_small = quote.admit(quote.weights_bytes - 1, 1, 2048);
    assert_eq!(too_small.shortfall.first(), Some(&DraftCost::Weights));
    assert!(too_small.required_bytes > too_small.budget_bytes);
}
