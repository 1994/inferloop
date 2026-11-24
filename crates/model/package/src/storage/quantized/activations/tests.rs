use super::*;
#[test]
fn activation_rules_resolve_exact_regex_and_weight_only_modules() -> Result<()> {
    let quant = serde_json::json!({"config_groups":{
        "quantized":{"weights":{"num_bits":8},"targets":["re:.*attention\\.q_proj$", "lm_head"],
            "input_activations":{"num_bits":8,"type":"float","strategy":"token","dynamic":true,"symmetric":true}},
        "weight_only":{"weights":{"num_bits":8},"targets":["re:.*attention\\.v_proj$"],"input_activations":null}
    }});
    let rules = Rule::parse(&quant)?;
    assert!(Rule::for_module(&rules, "model.layers.3.attention.q_proj")?);
    assert!(Rule::for_module(&rules, "lm_head")?);
    assert!(!Rule::for_module(
        &rules,
        "model.layers.3.attention.v_proj"
    )?);
    assert!(!Rule::for_module(
        &rules,
        "model.layers.3.attention.k_proj"
    )?);
    let conflict = serde_json::json!({"config_groups":{
            "a":{"weights":{"num_bits":8},"targets":["Linear"],"input_activations":null},
            "b":quant["config_groups"]["quantized"]}});
    assert!(Rule::for_module(&Rule::parse(&conflict)?, "lm_head").is_err());
    Ok(())
}
