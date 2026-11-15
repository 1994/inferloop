#!/usr/bin/env python3
"""Convert a Qwen3-VL GGUF (and optional mmproj) into an HF-style package.

This is the fast functional path used before the native GGUF package source lands.
It dequantizes to F16/F32, maps names, and reuses the existing Safetensors loader.

Requires `numpy`, `safetensors`, and llama.cpp's `gguf-py`:
  PYTHONPATH=/path/to/llama.cpp/gguf-py python3 tools/gguf/convert.py ...
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors.numpy import save_file

TEXT_FIELDS = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
VISION_FIELDS = {
    "attn_qkv.weight": "attn.qkv.weight",
    "attn_qkv.bias": "attn.qkv.bias",
    "attn_out.weight": "attn.proj.weight",
    "attn_out.bias": "attn.proj.bias",
    "ffn_up.weight": "mlp.linear_fc1.weight",
    "ffn_up.bias": "mlp.linear_fc1.bias",
    "ffn_down.weight": "mlp.linear_fc2.weight",
    "ffn_down.bias": "mlp.linear_fc2.bias",
    "ln1.weight": "norm1.weight",
    "ln1.bias": "norm1.bias",
    "ln2.weight": "norm2.weight",
    "ln2.bias": "norm2.bias",
}
ASSETS = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def text_name(name: str) -> str | None:
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    if name == "output.weight":
        return "lm_head.weight"
    if not name.startswith("blk."):
        return None
    layer, field = name[4:].split(".", 1)
    target = TEXT_FIELDS.get(field)
    return None if target is None else f"model.layers.{layer}.{target}"


def vision_name(name: str, deepstack: list[int]) -> str | None:
    if name == "v.patch_embd.weight" or name == "v.patch_embd.weight.1":
        return "patch"
    if name == "v.patch_embd.bias":
        return "model.visual.patch_embed.proj.bias"
    if name == "v.position_embd.weight":
        return "model.visual.pos_embed.weight"
    if name.startswith("v.post_ln."):
        return "model.visual.merger." + name[len("v.post_ln.") :]
    if name.startswith("mm."):
        _, index, field = name.split(".", 2)
        target = {"0": "merger.linear_fc1", "2": "merger.linear_fc2"}.get(index)
        return None if target is None else f"model.visual.{target}.{field}"
    if name.startswith("v.deepstack."):
        _, _, layer, field = name.split(".", 3)
        index = deepstack.index(int(layer))
        return f"model.visual.deepstack_merger_list.{index}.{field}"
    if name.startswith("v.blk."):
        _, _, layer, field = name.split(".", 3)
        target = VISION_FIELDS.get(field)
        return None if target is None else f"model.visual.blocks.{layer}.{target}"
    return None


def load(path: Path) -> list[tuple[str, np.ndarray]]:
    reader = GGUFReader(str(path))
    out: list[tuple[str, np.ndarray]] = []
    for tensor in reader.tensors:
        values = dequantize(tensor.data, tensor.tensor_type)
        out.append((tensor.name, np.ascontiguousarray(values)))
    return out


def convert(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assets = Path(args.assets)
    for name in ASSETS:
        source = assets / name
        if source.exists():
            shutil.copy2(source, out / name)
    config = json.loads((assets / "config.json").read_text())
    if args.text_only:
        for key in ("vision_config", "vision_start_token_id", "vision_end_token_id"):
            config.pop(key, None)
        config["language_model_only"] = True
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    tensors: dict[str, np.ndarray] = {}
    for name, values in load(Path(args.gguf)):
        target = text_name(name)
        if target is None:
            raise SystemExit(f"unmapped text tensor {name}")
        tensors[target] = values.astype(np.float16 if values.ndim > 1 else np.float32)
    if args.mmproj and not args.text_only:
        deepstack = list(config.get("vision_config", {}).get("deepstack_visual_indexes", []))
        patch: list[np.ndarray] = []
        for name, values in load(Path(args.mmproj)):
            target = vision_name(name, deepstack)
            if target == "patch":
                patch.append(values)
                continue
            if target is None:
                raise SystemExit(f"unmapped vision tensor {name}")
            tensors[target] = values.astype(np.float16 if values.ndim > 1 else np.float32)
        if len(patch) != 2:
            raise SystemExit(f"expected two patch slices, got {len(patch)}")
        tensors["model.visual.patch_embed.proj.weight"] = np.stack(patch, axis=2)
    save_file(tensors, out / "model.safetensors")
    print(f"wrote {len(tensors)} tensors to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--assets", required=True, help="HF directory with config/tokenizer")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mmproj")
    parser.add_argument("--text-only", action="store_true")
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
