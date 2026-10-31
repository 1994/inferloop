# inferloop

[![Quality](https://github.com/1994/inferloop/actions/workflows/ci.yml/badge.svg)](https://github.com/1994/inferloop/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**Serve local language models from a single binary.**

inferloop turns a Safetensors model folder into an OpenAI-compatible endpoint on GPUs you already own. One native executable, no Python, no conversion step, no serving framework to operate. Built for teams that want the model behind a familiar API without adopting a Python stack to get there.

> **Status.** Active development: built-in providers cover the Qwen 3.x family, there is no CPU backend, and performance certification is in progress. See [current boundaries](#current-boundaries).

## Why inferloop

- **Deploy a file, not a stack.** The runtime artifact is `bin/infer`: no interpreter, virtualenv, conda, PyTorch, or Python serving framework in the serving path. Build and QA tooling uses Python; the thing you run does not.
- **Your client already speaks it.** `/v1/chat/completions` and `/v1/completions` with SSE streaming, tool calls, and strict JSON schemas. Point the base URL at the server and keep your client code.
- **The model folder is the artifact.** Safetensors weights are read in place through memory maps, with the tokenizer and chat template taken from the same directory. No export, re-shard, or quantization script between you and a running endpoint.
- **One engine, four workloads.** Generate, embed, rerank, and structured decision requests share admission, scheduling, state/KV lifecycle, and observability instead of growing a separate service each.
- **Fast paths with receipts.** CUDA compute uses non-blocking submit/poll by default; supported checkpoints auto-select MTP depth 2 and BF16 target attention, with a logged fallback to the canonical path. A separate DFlash2 draft is available via `--draft-model`. Defaults pass numerical and quality gates before shipping, and speed claims wait for paired, signed baselines rather than single runs.
- **Operable by default.** Health, events, Chrome Trace, OTLP, checkpoints, bounded queues, backpressure, and cancellation are part of the engine, not a bolt-on dashboard.
- **One CLI across CUDA and Metal.** The same commands and HTTP surface on Linux CUDA hosts and Apple Metal machines.

## Start serving

```sh
git clone https://github.com/1994/inferloop.git
cd inferloop
make local-build
target/release/infer /path/to/model
```

That builds the release CLI with the platform backend and starts an HTTP server on `http://127.0.0.1:8080`, with memory budget and KV sizing derived from the device. No config file, no `serve` subcommand, no flags required.

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"model","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Use `--served-model-name model` to control the model ID your clients send. OpenAI routes require a tokenizer in the package; chat completions also require a chat template. See the [OpenAI API guide](docs/guides/openai-api.md) for streaming, tools, and parameter support.

## What you can serve

| Workload | What it gives you |
|---|---|
| **Generate** | Token generation with batching, optional speculative decoding, and streaming |
| **Embed** | Embedding extraction from a compatible model head |
| **Rerank** | Candidate ranking from a compatible model head |
| **Decision** | Structured decisions from a declared schema |

The native API covers all four. The OpenAI-compatible routes cover chat and text completions. Built-in providers currently handle the Qwen 3.x family, including hybrid and MoE variants; an arbitrary Hugging Face package is not guaranteed to load or run. See [model execution](docs/guides/model-execution.md) and [adding a model](docs/guides/adding-a-model.md).

## Requirements

| Component | Requirement |
|---|---|
| Model | A supported local Safetensors package; tokenizer assets for text and OpenAI routes |
| Linux / CUDA run | NVIDIA GPU and driver, plus the cuTile JIT toolchain (`tileiras`) |
| macOS / Metal run | A Metal-capable Apple GPU |
| Linux / CUDA build | CUDA Toolkit headers, C compiler, and libclang |
| Build toolchain | `1.99.0` pinned by `rust-toolchain.toml`; MSRV `1.90` |
| Build helper | `make` entry points use Python 3.12+; a direct `cargo build` does not |

`auto` selects CUDA or Metal and fails if no supported GPU is available. CPU-only hosts can run tests and lint checks, but cannot execute a model.

## Deployment

- **Packages per target.** Linux CUDA (x86_64, AArch64; selectable glibc baselines from 2.17 to 2.41) and macOS Metal (Intel, Apple Silicon) archives come from one cross-compilation pipeline. Each ships `bin/infer`, standalone run instructions, `SHA256SUMS`, and a manifest recording target, features, and verification scope.
- **Packaged binary, same start.** `./bin/infer --backend cuda serve --package /path/to/model` serves exactly like the native build.
- **Portable by design.** Weights are read in place rather than rewritten per host, and an archive runs on any host at or above its selected glibc baseline. Metal needs only the OS.
- **No CPU execution backend.** A supported GPU is required to run a model; hosts without one can still build, lint, and run the reference tests.

## Under the hood

```text
Service   CLI, native HTTP/SSE, OpenAI subset, diagnostic agent
Runtime   admission, scheduling, state/KV lifecycle, completion, observability
Model     package loading, provider registry, execution graph, compiler
Backend   CUDA (cuTile / SIMT) and Metal (MSL) kernels
```

Model providers own model semantics. A shared IR and kernel API keep model names out of the scheduler and device code, so a new provider or workload does not require a second engine. Read the [code layout](docs/architecture/layout.md), [backend capabilities](docs/architecture/backends.md), and [runtime design](docs/architecture/runtime.md).

## Performance, measured

Performance is a requirement with evidence attached, not a tagline. A candidate is accepted on paired runs against a signed baseline profile: same release, same effective configuration read back from the runtime, raw samples retained and recomputable. A single run, a different configuration, or another engine's setup is not republished here as proof of speed.

That is why this README has no "Nx faster" headline. Speed claims are kept where they can be audited, in the [performance baseline plan](docs/plans/performance/baseline.md) and the [roadmap](docs/plans/README.md).

## Current boundaries

- Built-in model coverage is narrow: the Qwen 3.x family today. Validate a package before relying on it.
- No CPU execution backend.
- CUDA still depends on the cuTile JIT toolchain; ahead-of-time kernel coverage is in progress.
- Execution overlap is currently per submit/poll, not a depth-2 batch queue.
- Multimodal execution exists in examples, not in the serving scheduler.
- Performance and production SLO validation are incomplete; no cross-engine parity is claimed.

## Documentation

Start with the [documentation index](docs/README.md). In-depth guides are currently in Chinese.

- [Development](docs/guides/development.md)
- [Adding a model](docs/guides/adding-a-model.md)
- [Model execution](docs/guides/model-execution.md)
- [OpenAI-compatible API](docs/guides/openai-api.md)
- [CUDA performance](docs/guides/cuda-performance.md)
- [Tuning for new hardware and models](docs/guides/tuning-new-hardware-and-models.md)
- [Roadmap](docs/plans/README.md)
- [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

## Implementation notes

The engine is written in Rust, with CUDA paths in cuTile/CUDA C++ and Metal paths in Metal Shading Language. These are implementation choices made for memory safety without a garbage collector, predictable latency, and direct access to the two target GPU APIs.

They are not a goal in themselves and do not shape the project's contracts. The CLI, the HTTP/OpenAI surface, and the Safetensors package format are language-agnostic; changing a toolchain is an engineering decision that has to be justified by correctness and measured performance, not a change in what the project is.

## License

[Apache License 2.0](LICENSE).
