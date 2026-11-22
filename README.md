# inferloop

[![Quality](https://github.com/1994/inferloop/actions/workflows/ci.yml/badge.svg)](https://github.com/1994/inferloop/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**A GPU inference engine that serves Safetensors and GGUF models from one native binary, behind an OpenAI-compatible API.**

Point `infer` at a Hugging Face model folder or a `.gguf` file and it starts an HTTP server. There is no Python in the serving path, no conversion step, and no config file. The engine sizes its memory budget, batch width and KV cache from the device and the model.

> **Status:** active development. CUDA (Linux) and Metal (macOS) are the execution backends; there is no CPU backend. Model coverage is the Qwen 3.x family plus dense Llama/Mistral. See [limits](#limits).

## Quick start

```sh
git clone https://github.com/1994/inferloop.git
cd inferloop
make local-build                               # release CLI for this host's GPU backend

target/release/infer /models/Qwen3-8B          # a Safetensors folder
target/release/infer /models/qwen3-vl-2b.Q4_K_M.gguf   # or a GGUF file (CUDA)
```

The server listens on `http://127.0.0.1:8080`:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"model","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

The `model` field must match the name from `GET /v1/models`. Set it with `--served-model-name`.

## Models and formats

| Architecture | Safetensors | GGUF |
|---|---|---|
| Qwen3, Qwen3-MoE | ✓ | — |
| Qwen3-Next, Qwen3.5 / Qwen3.5-MoE (hybrid attention, incl. vision) | ✓ | — |
| Qwen3-VL | ✓ | ✓ (CUDA), with a sibling `mmproj-*.gguf` for vision |
| Llama, Mistral (dense) | ✓ | — |

| Weights | CUDA | Metal |
|---|---|---|
| Safetensors F32 / BF16 / F16 | ✓ | ✓ |
| Safetensors FP8 (per-channel, block) and NVFP4 | ✓ | — |
| GGUF F32 / F16 / BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K | ✓ | — |

- **Safetensors:** weights are memory-mapped and read in place. The tokenizer and chat template come from the same folder.
- **GGUF:** quantized weights stay quantized in device memory and run on the engine's own grouped-integer kernels; nothing is dequantized to BF16 at load. The tokenizer and chat template are rebuilt from the GGUF metadata.
- **KV cache:** stored as `f32`, `f16` or `fp8`; select with `--kv-cache-dtype`. GGUF models default to `f16`, everything else follows the checkpoint.

An architecture that is not listed is rejected at load. A listed one is not guaranteed to work for every package variant, so validate it with `infer inspect-package --package <path>` first.

## API

| Route | |
|---|---|
| `POST /v1/chat/completions` | Chat with SSE streaming, tool calls, strict JSON schemas, images and video (CUDA) |
| `POST /v1/completions` | Text completion |
| `POST /v1/responses` | Responses text and function-tool subset; optional in-process store with `--enable-response-store` |
| `POST /v1/decisions` | Predicate / choice / score decisions answered by the served model |
| `GET /v1/models` | The served model |
| `/native/v1/*` | Generate, embed, rerank and decision on one engine, plus runtime state and device timelines |

Unsupported request fields are rejected instead of silently ignored. Field-level support is in the [OpenAI API guide](docs/guides/openai-api.md); media handling is in the [vision guide](docs/guides/vision.md).

## Serving engine

- **Continuous batching.** Requests share a slot pool with paged KV. A slot addresses the whole model context; memory is spent on pages actually used.
- **Prefix reuse.** Committed full KV blocks are shared between requests with the same prefix.
- **Speculative decoding.** Checkpoints with an MTP head select depth 2 automatically on CUDA. `--draft-model` uses a separate DFlash2 draft instead.
- **Load-time tuning.** Kernel choices are measured on the device at load and cached per hardware, driver and model. Nothing is tuned through environment variables.
- **Bounded and observable.** Admission, queues and host memory are bounded, with backpressure and cancellation. Health, events, Chrome Trace / Perfetto, OTLP and per-step device timing are built in.

## Common options

| Option | Effect |
|---|---|
| `--served-model-name NAME` | Model ID clients send |
| `--max-model-len TOKENS` | Context per sequence (prompt + output); defaults to the model limit |
| `--max-num-seqs N` | Cap on concurrent sequences |
| `--gpu-memory-utilization F` | Fraction of device memory the engine may use |
| `--kv-cache-dtype f32\|f16\|fp8` | KV storage |
| `--num-speculative-tokens N\|auto` | MTP draft depth; `0` disables it |
| `--draft-model PATH` | Separate DFlash2 draft (CUDA) |
| `--limit-mm-per-prompt JSON` | Media items per prompt, e.g. `{"image":1}`; all zero skips loading the vision tower |
| `--backend auto\|cuda\|metal` | Backend; `auto` fails if no supported GPU is present |

`infer --help` lists everything. `infer doctor` reports the backends and host environment it detects.

## Requirements

| | |
|---|---|
| Linux / CUDA | NVIDIA GPU and driver, and the cuTile JIT toolchain (`tileiras`) to run; CUDA Toolkit headers, a C compiler and libclang to build |
| macOS / Metal | A Metal-capable Apple GPU |
| Rust | `1.99.1`, pinned by `rust-toolchain.toml` (MSRV `1.90`) |
| Build helper | Python 3.12+ for the `make` entry points; a plain `cargo build` does not need it |
| Video input | `ffmpeg` and `ffprobe` on the host |

Hosts without a supported GPU can build, lint and run the reference tests, but cannot execute a model.

## Release packages

`make build-linux-cuda` and `make build-macos-metal` produce self-contained archives for Linux CUDA (x86_64, AArch64; glibc baselines 2.17–2.41) and macOS Metal (Intel, Apple Silicon). Each holds `bin/infer`, run instructions, `SHA256SUMS` and a manifest of target, features and verification scope:

```sh
./bin/infer --backend cuda serve --package /path/to/model
```

See the [packaging guide](docs/guides/packaging.md).

## Architecture

```text
Service   CLI, native HTTP/SSE, OpenAI-compatible routes, diagnostic agent
Runtime   admission, scheduling, state/KV lifecycle, completion, observability
Model     Safetensors and GGUF packages, provider registry, execution graph, compiler
Backend   CUDA (cuTile / CUDA C++) and Metal (MSL) kernels
```

Model providers own topology and weight semantics. The compiler lowers their graph through a shared IR and kernel API, so model names stay out of the scheduler and the device code. A GGUF file is presented to the providers as an equivalent Hugging Face package, so the rest of the engine does not know about GGUF. Details: [code layout](docs/architecture/layout.md), [backend contracts](docs/architecture/backends.md), [runtime](docs/architecture/runtime.md).

## Performance

Speed claims in this repository come from paired release runs on the same hardware, configuration and workload, with raw samples kept. This README does not quote a headline number. Current measurements and open gaps are in the [performance baseline](docs/plans/performance/baseline.md), the [GGUF performance plan](docs/plans/performance/gguf-grouped-int.md) and the [roadmap](docs/plans/README.md).

## Limits

- Model coverage is the architectures listed above.
- GGUF runs on CUDA only and covers the Qwen3-VL architecture. Q2_K, Q3_K, IQ-family types and quantized (Q8_0/Q4_0) KV are not supported.
- Quantized GGUF prompt processing and decode are still slower than llama.cpp on the same file.
- Media requests do not use prefix reuse or speculative decoding, and one prompt carries one media modality.
- CUDA kernels depend on the cuTile JIT toolchain; ahead-of-time coverage is in progress.
- No CPU execution backend.

## Documentation

Start at the [documentation index](docs/README.md). Most in-depth guides are in Chinese.

- [Development](docs/guides/development.md) · [Quality gates](docs/guides/quality-gates.md)
- [Model execution](docs/guides/model-execution.md) · [Adding a model](docs/guides/adding-a-model.md)
- [OpenAI-compatible API](docs/guides/openai-api.md) · [Vision](docs/guides/vision.md)
- [CUDA performance](docs/guides/cuda-performance.md) · [Tuning for new hardware and models](docs/guides/tuning-new-hardware-and-models.md)
- [Roadmap](docs/plans/README.md)
- [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

## License

[Apache License 2.0](LICENSE).
