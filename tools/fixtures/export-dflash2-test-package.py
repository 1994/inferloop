"""Create an untrained, deterministic FP8 draft for small-model integration tests.

The target's real weights and tokenizer remain unchanged. This fixture exercises
draft loading, rejection, graph replay and admission; it cannot measure acceptance
rate or trained-draft quality. No third-party Python packages are required.
"""

import argparse
import json
import math
import struct
from pathlib import Path


def export(target, output):
    target_config = json.loads((target / "config.json").read_text())
    hidden = target_config["hidden_size"]
    vocabulary = target_config["vocab_size"]
    if hidden > 2048 or hidden % 512:
        raise ValueError("test fixture requires a small target with 512-aligned hidden size")
    config = {
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": hidden,
        "intermediate_size": hidden,
        "vocab_size": vocabulary,
        "num_hidden_layers": 1,
        "num_target_layers": target_config["num_hidden_layers"],
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "sliding_window": 100,
        "rms_norm_eps": 1e-6,
        "is_causal": False,
        "tie_word_embeddings": False,
        "eos_token_id": target_config["eos_token_id"],
        "pad_token_id": target_config["eos_token_id"],
        "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "selector_rank": 16,
            "selector_top_k": 16,
            "mask_token_id": target_config["eos_token_id"],
            "target_layer_ids": [target_config["num_hidden_layers"] - 1],
        },
        "quantization_config": {"quant_method": "compressed-tensors"},
    }
    tensors = {}

    def add(name, shape, dtype="BF16", one=False):
        code = b"\0" if dtype == "F8_E4M3" else b"\x80\x3f" if one else b"\0\0"
        tensors[name] = (shape, dtype, code * math.prod(shape))

    for name, shape in [
        ("fc.weight", [hidden, hidden]),
        ("hidden_norm.weight", [hidden]),
        ("norm.weight", [hidden]),
        ("candidate_selector.hidden_projection.weight", [16, hidden]),
        ("candidate_selector.predecessor_codebook", [vocabulary, 16]),
        ("candidate_selector.successor_codebook", [vocabulary, 16]),
    ]:
        add(name, shape, one="norm" in name)
    for name in ["input_layernorm", "post_attention_layernorm"]:
        add(f"layers.0.{name}.weight", [hidden], one=True)
    for name in ["q_norm", "k_norm"]:
        add(f"layers.0.self_attn.{name}.weight", [128], one=True)
    for name, shape in [
        ("self_attn.q_proj", [512, hidden]),
        ("self_attn.k_proj", [256, hidden]),
        ("self_attn.v_proj", [256, hidden]),
        ("self_attn.o_proj", [hidden, 512]),
        ("mlp.gate_proj", [hidden, hidden]),
        ("mlp.up_proj", [hidden, hidden]),
        ("mlp.down_proj", [hidden, hidden]),
    ]:
        add(f"layers.0.{name}.weight", shape, "F8_E4M3")
        add(f"layers.0.{name}.weight_scale", [shape[0], 1], one=True)
    for name in ["attention_conv", "mlp_conv"]:
        add(f"layers.0.{name}.base_kernel", [2, 2, hidden])
        add(f"layers.0.{name}.kernel_projection.weight", [4 * hidden // 16, hidden])
    header = {}
    offset = 0
    for name, (shape, dtype, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (output / "model.safetensors").open("wb") as file:
        file.write(struct.pack("<Q", len(raw)))
        file.write(raw)
        for _shape, _dtype, data in tensors.values():
            file.write(data)
    print(f"untrained DFlash2 test fixture: {offset} weight bytes at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.target, args.output)
