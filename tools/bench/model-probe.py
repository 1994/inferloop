#!/usr/bin/env python3
"""Serve tiny random checkpoints of varied geometry through the real path.

This answers whether a new model only needs its provider: each geometry must load and generate,
alone and concurrently, or be refused at load. Random weights prove the path runs and is batch
invariant. They say nothing about numerical correctness, which belongs to the kernel acceptance.
"""

import argparse
import json
import os
import random
import shutil
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

DTYPES = {"BF16": ("bfloat16", 2), "F16": ("float16", 2), "F32": ("float32", 4)}
BLOCK_VALUES = 16384
INIT_STD = 0.02
LOAD_LIMIT_S = 1500
REQUEST_LIMIT_S = 60
CONCURRENCY = 4
MAX_TOKENS = 16
MAX_POSITIONS = 4096
HEADER_ALIGN = 8
# The load-time refusal of a geometry without kernels, not any unsupported error.
REFUSAL = "need a power-of-two"


class Geometry(NamedTuple):
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    layers: int
    dtype: str = "BF16"
    refused: bool = False


CASES = {
    "control-128": Geometry(512, 4, 2, 128, 1536, 2),
    "pow2-64": Geometry(256, 4, 2, 64, 768, 2),
    "no-gqa": Geometry(512, 4, 4, 128, 1536, 2),
    "one-layer": Geometry(512, 4, 2, 128, 1536, 1),
    "wide-mlp": Geometry(512, 4, 2, 128, 2048, 2),
    "head-256": Geometry(1024, 4, 2, 256, 3072, 2),
    "hidden-ne-heads": Geometry(640, 4, 2, 128, 1920, 2),
    "f16": Geometry(512, 4, 2, 128, 1536, 2, "F16"),
    "f32": Geometry(512, 4, 2, 128, 1536, 2, "F32"),
    "head-48": Geometry(192, 4, 2, 48, 576, 2, refused=True),
    "head-96": Geometry(384, 4, 2, 96, 1152, 2, refused=True),
    "head-100": Geometry(400, 4, 2, 100, 1200, 2, refused=True),
}


def tensor_shapes(geometry, vocab):
    """Qwen3 weight names and shapes for one geometry, with tied embeddings."""
    g = geometry
    q, kv = g.heads * g.head_dim, g.kv_heads * g.head_dim
    shapes = {"model.embed_tokens.weight": [vocab, g.hidden], "model.norm.weight": [g.hidden]}
    for layer in range(g.layers):
        prefix = f"model.layers.{layer}."
        shapes |= {
            f"{prefix}input_layernorm.weight": [g.hidden],
            f"{prefix}post_attention_layernorm.weight": [g.hidden],
            f"{prefix}mlp.down_proj.weight": [g.hidden, g.intermediate],
            f"{prefix}mlp.gate_proj.weight": [g.intermediate, g.hidden],
            f"{prefix}mlp.up_proj.weight": [g.intermediate, g.hidden],
            f"{prefix}self_attn.k_norm.weight": [g.head_dim],
            f"{prefix}self_attn.q_norm.weight": [g.head_dim],
            f"{prefix}self_attn.k_proj.weight": [kv, g.hidden],
            f"{prefix}self_attn.v_proj.weight": [kv, g.hidden],
            f"{prefix}self_attn.o_proj.weight": [g.hidden, q],
            f"{prefix}self_attn.q_proj.weight": [q, g.hidden],
        }
    return shapes


def safetensors_header(shapes, dtype):
    width = DTYPES[dtype][1]
    header, offset = {}, 0
    for name, shape in shapes.items():
        size = width
        for dimension in shape:
            size *= dimension
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode()
    return header, encoded + b" " * (-len(encoded) % HEADER_ALIGN)


def value_block(dtype):
    """A repeated small normal block: finite, cheap and deterministic."""
    rng = random.Random(7)
    values = [rng.gauss(0, INIT_STD) for _ in range(BLOCK_VALUES)]
    if dtype == "F32":
        return struct.pack(f"<{len(values)}f", *values)
    if dtype == "F16":
        return struct.pack(f"<{len(values)}e", *values)
    return b"".join(struct.pack("<f", value)[2:] for value in values)


def write_checkpoint(path, source, geometry):
    path.mkdir(parents=True)
    for item in source.iterdir():
        if item.suffix in {".json", ".txt"} and item.name != "config.json":
            shutil.copy(item, path / item.name)
    g = geometry
    config = json.loads((source / "config.json").read_text())
    config |= {
        "hidden_size": g.hidden,
        "num_attention_heads": g.heads,
        "num_key_value_heads": g.kv_heads,
        "head_dim": g.head_dim,
        "intermediate_size": g.intermediate,
        "num_hidden_layers": g.layers,
        "max_window_layers": g.layers,
        "max_position_embeddings": MAX_POSITIONS,
        "torch_dtype": DTYPES[g.dtype][0],
    }
    config.pop("layer_types", None)
    (path / "config.json").write_text(json.dumps(config))
    header, encoded = safetensors_header(tensor_shapes(g, config["vocab_size"]), g.dtype)
    block = value_block(g.dtype)
    with (path / "model.safetensors").open("wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for entry in header.values():
            left = entry["data_offsets"][1] - entry["data_offsets"][0]
            while left > 0:
                chunk = block[:left]
                output.write(chunk)
                left -= len(chunk)


def generate(port, prompt):
    body = {
        "model": "1",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(),
        {"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_LIMIT_S) as response:
            return json.load(response)["choices"][0]["message"]["content"], None
    except (urllib.error.URLError, TimeoutError, KeyError) as error:
        return None, str(error)


def wait_ready(server, port):
    for _ in range(LOAD_LIMIT_S):
        if server.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1).close()
            return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    return False


def probe(args, name, geometry, port):
    """One case's result. `passed` means it served correctly or was refused as declared."""
    with tempfile.TemporaryDirectory(dir=args.work) as directory:
        checkpoint = Path(directory) / name
        write_checkpoint(checkpoint, args.source, geometry)
        env = dict(os.environ, XDG_CACHE_HOME=str(args.work / "cache"))
        log = args.work / f"{name}.log"
        with log.open("w") as sink:
            server = subprocess.Popen(
                [args.binary, "serve", "--package", checkpoint, "--listen", f"127.0.0.1:{port}"],
                env=env,
                stdout=sink,
                stderr=subprocess.STDOUT,
            )
            try:
                if not wait_ready(server, port):
                    refused = server.poll() not in (None, 0) and REFUSAL in log.read_text()
                    return {
                        "served": False,
                        "refused": refused,
                        "passed": geometry.refused and refused,
                    }
                prompts = [f"q{index}" for index in range(CONCURRENCY)]
                alone = [generate(port, prompt) for prompt in prompts]
                with ThreadPoolExecutor(CONCURRENCY) as pool:
                    together = list(pool.map(lambda prompt: generate(port, prompt), prompts))
            finally:
                server.terminate()
                server.wait(30)
    errors = [error for _, error in alone + together if error]
    invariant = [text for text, _ in alone] == [text for text, _ in together]
    passed = not geometry.refused and not errors and invariant
    return {"served": True, "errors": errors[:2], "batch_invariant": invariant, "passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument(
        "--source", required=True, type=Path, help="Qwen3 package for its tokenizer"
    )
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES))
    parser.add_argument("--port", type=int, default=18100)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    results = {}
    for offset, name in enumerate(args.cases):
        results[name] = probe(args, name, CASES[name], args.port + offset)
        print(name, json.dumps(results[name]), flush=True)
    (args.work / "result.json").write_text(json.dumps(results, indent=1))
    failed = [name for name, result in results.items() if not result["passed"]]
    if failed:
        raise SystemExit(f"failed geometries: {failed}")


if __name__ == "__main__":
    main()
