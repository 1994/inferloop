# inferloop

[![Quality](https://github.com/1994/inferloop/actions/workflows/ci.yml/badge.svg)](https://github.com/1994/inferloop/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.90%2B-orange?logo=rust)](rust-toolchain.toml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

A native Rust execution engine for NVIDIA CUDA and Apple Metal. It loads Safetensors model packages, compiles them into device-resident graphs, and serves native and OpenAI-compatible HTTP/SSE APIs from one `infer` binary.

inferloop is built around a general request loop rather than a single model family or workload. **Generate, Embed, Rerank, and Decision** requests share scheduling, admission, state/KV lifecycle, and observability; model providers and device kernels are extensions, not forks of the runtime.

> **Status: active development.** Built-in model coverage is limited and production performance is not yet certified. There is no CPU execution backend.

## Highlights

- **Workload-agnostic runtime** for generation, embedding, ranking, and decision requests.
- **Pluggable model providers** so new model families do not require runtime branches.
- **CUDA and Metal backends** with device-resident execution graphs.
- **Bounded request batching**, admission control, backpressure, and cancellation.
- **Serving** through native token APIs and an OpenAI-compatible subset with SSE.
- **Operational tooling** for health, events, traces, checkpoints, and diagnostics.

## Architecture

```text
Service   CLI, native HTTP/SSE, OpenAI subset, diagnostic agent
Runtime   admission, scheduling, state/KV lifecycle, completion, observability
Model     package loading, provider registry, execution graph, compiler
Backend   CUDA (cuTile / SIMT) and Metal (MSL) kernels
```

Model providers own model semantics. Shared IR and kernel APIs keep model names out of the scheduler and device code, so adding a provider or workload does not require a separate engine.

See [code layout](docs/architecture/layout.md), [backend capabilities](docs/architecture/backends.md), and the [runtime design](docs/architecture/runtime.md).

## Workloads

| Workload | Purpose |
|---|---|
| **Generate** | Token generation with batching, optional speculative decoding, and streaming |
| **Embed** | Embedding extraction from a compatible model head |
| **Rerank** | Candidate ranking from a compatible model head |
| **Decision** | Structured decisions from a declared schema |

Workload support depends on the model provider, head, and backend. The built-in provider set is intentionally small; an arbitrary Hugging Face package is not guaranteed to load or run. See [model execution](docs/guides/model-execution.md) and [adding a model](docs/guides/adding-a-model.md).

## Requirements

| Component | Requirement |
|---|---|
| Rust | `1.99.0` pinned by `rust-toolchain.toml`; MSRV `1.90` |
| Linux / CUDA | NVIDIA GPU, CUDA Toolkit, `tileiras`, C compiler, and libclang |
| macOS / Metal | A Metal-capable Apple GPU and Xcode command-line tools |
| Model | A supported local Safetensors package; tokenizer assets are required for text and OpenAI routes |

`auto` selects CUDA or Metal and fails if no supported GPU is available. CPU-only hosts can run tests and lint checks, but cannot execute a model.

## Quick start

```sh
git clone https://github.com/1994/inferloop.git
cd inferloop
make local-build
target/release/infer /path/to/model
```

`make local-build` builds the native release CLI with the platform backend: CUDA on Linux, Metal on macOS. For explicit Cargo builds:

```sh
cargo build --locked --release -p infer-cli --features cuda  # Linux CUDA
cargo build --locked --release -p infer-cli                 # macOS Metal
```

CUDA builds need the environment in [crates/backend/cuda/README.md](crates/backend/cuda/README.md).

The server listens on `http://127.0.0.1:8080`. On supported CUDA checkpoints, the default profile may enable BF16 prefill attention and multi-token prediction (MTP); `--num-speculative-tokens 0` keeps the canonical path.

```sh
target/release/infer /path/to/model --served-model-name model
```

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"model","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

OpenAI routes require a package tokenizer; chat completions also require a chat template. See the [OpenAI API guide](docs/guides/openai-api.md) for streaming, tools, and parameter support.

## Backends

| Backend | Scope |
|---|---|
| **CUDA on Linux** | Native Rust/cuTile and SIMT kernels, device graphs, serving, bounded batching, optional MTP |
| **Metal on macOS** | Prefill/decode, KV and prefix caching, service validation with repository fixtures |
| **Host (no GPU)** | Tests, lint, and scalar reference operators; no model execution |

## Testing and packaging

```sh
make test              # Host tests; no GPU required
make check-rust        # Lint, tests, docs, CPU allocation, golden checks
make test-cuda-hosted  # CUDA compile/lint/tests; Toolkit only
make test-cuda         # CUDA device checks
make test-metal        # macOS Metal checks
```

`make check` is the full host gate. Device gates are separate because they need a specific GPU and driver.

```sh
make setup-build
make package TARGET=x86_64-unknown-linux-gnu.2.28
```

Packaging supports Linux CUDA and macOS Metal on x86_64 and AArch64. See the [packaging guide](docs/guides/packaging.md).

## Limitations

- Built-in model coverage is narrow; validate each package before relying on it.
- No CPU execution backend.
- CUDA uses a synchronous provider and runtime-compiled kernels.
- Multimodal execution is limited to examples, not the serving scheduler.
- Performance and production SLO validation are incomplete.

## Documentation

Start with the [documentation index](docs/README.md). In-depth guides are currently in Chinese.

- [Development](docs/guides/development.md)
- [Adding a model](docs/guides/adding-a-model.md)
- [Model execution](docs/guides/model-execution.md)
- [OpenAI-compatible API](docs/guides/openai-api.md)
- [CUDA performance](docs/guides/cuda-performance.md)
- [Roadmap](docs/plans/README.md)
- [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

## License

[Apache License 2.0](LICENSE).
