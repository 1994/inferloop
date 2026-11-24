//! Resolve activation policy from checkpoint module targets, independently of the architecture.
use super::FP8_WEIGHT_BITS;
use infer_core::{Error, Result};

pub(super) struct Rule {
    targets: Vec<Target>,
    fp8_token: bool,
    /// Weight block shape when the group declares block scales instead of per-channel ones.
    weight_block: Option<[usize; 2]>,
}
enum Target {
    AllLinear,
    Exact(String),
    Regex(regex::Regex),
}
impl Rule {
    pub(super) fn parse(quant: &serde_json::Value) -> Result<Vec<Self>> {
        let Some(groups) = quant["config_groups"].as_object() else {
            return Ok(Vec::new());
        };
        let mut rules = Vec::new();
        for group in groups.values() {
            if group["weights"]["num_bits"] != FP8_WEIGHT_BITS {
                continue;
            }
            let input = &group["input_activations"];
            let fp8_token = !input.is_null();
            // "block" is the same dynamic per-token scheme with one scale per input block
            // instead of one per row; the weight group's block size carries that distinction.
            if fp8_token
                && !(input["num_bits"] == FP8_WEIGHT_BITS
                    && input["type"] == "float"
                    && matches!(input["strategy"].as_str(), Some("token" | "block"))
                    && input["dynamic"] == true
                    && input["symmetric"] == true)
            {
                return Err(Error::unsupported(
                    "unsupported FP8 activation quantization",
                ));
            }
            let weight_block = group["weights"]["block_size"].as_array().and_then(|sizes| {
                let values = sizes
                    .iter()
                    .map(|size| size.as_u64().and_then(|n| usize::try_from(n).ok()))
                    .collect::<Option<Vec<_>>>()?;
                let [rows, columns] = values.as_slice() else {
                    return None;
                };
                Some([*rows, *columns])
            });
            let targets = group["targets"]
                .as_array()
                .ok_or_else(|| Error::invalid("quantization module targets"))?
                .iter()
                .map(|target| {
                    let name = target
                        .as_str()
                        .ok_or_else(|| Error::invalid("quantization target must be a string"))?;
                    if name == "Linear" {
                        Ok(Target::AllLinear)
                    } else if let Some(pattern) = name.strip_prefix("re:") {
                        regex::Regex::new(pattern)
                            .map(Target::Regex)
                            .map_err(|e| Error::invalid(e.to_string()))
                    } else {
                        Ok(Target::Exact(name.to_owned()))
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            rules.push(Self {
                targets,
                fp8_token,
                weight_block,
            });
        }
        Ok(rules)
    }
    /// Weight block shape declared for `module`, when its group uses block scales.
    pub(super) fn weight_block_for_module(
        rules: &[Self],
        module: &str,
    ) -> Result<Option<[usize; 2]>> {
        let mut declared = None;
        for rule in rules {
            if !rule.targets.iter().any(|target| match target {
                Target::AllLinear => true,
                Target::Exact(name) => name == module,
                Target::Regex(pattern) => pattern.is_match(module),
            }) {
                continue;
            }
            if declared.is_some_and(|value| value != rule.weight_block) {
                return Err(Error::invalid("conflicting weight block sizes for module"));
            }
            declared = Some(rule.weight_block);
        }
        Ok(declared.flatten())
    }

    pub(super) fn for_module(rules: &[Self], module: &str) -> Result<bool> {
        let mut declared = None;
        for rule in rules {
            if rule.targets.iter().any(|target| match target {
                Target::AllLinear => true,
                Target::Exact(name) => name == module,
                Target::Regex(pattern) => pattern.is_match(module),
            }) {
                if declared.is_some_and(|value| value != rule.fp8_token) {
                    return Err(Error::invalid("conflicting activation policies for module"));
                }
                declared = Some(rule.fp8_token);
            }
        }
        Ok(declared.unwrap_or(false))
    }
}

#[cfg(test)]
mod tests;
