# inferloop

[![Quality](https://github.com/1994/inferloop/actions/workflows/ci.yml/badge.svg)](https://github.com/1994/inferloop/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

inferloop serves local language models on NVIDIA CUDA and Apple Metal GPUs. It loads a Safetensors model package, compiles it into a device-resident execution graph, and exposes the model through native and OpenAI-compatible HTTP/SSE APIs from a single `infer` binary.

The engine is organized around a request loop, not around one model family or one workload. Generate, Embed, Rerank, and Decision requests share the same admission, scheduling, state/KV lifecycle, and observability; model providers and device kernels extend that loop instead of forking it.

> **Status: active development.** Built-in model coverage is narrow and production performance is not certified. There is no CPU execution backend.

## Highlights

- **Workload-agnostic runtime** for generation, embedding, ranking, and decision requests.
- **Pluggable model providers**, so a new model family does not require runtime branches.
- **CUDA and Metal backends** with device-resident execution graphs; CUDA compute uses non-blocking submit/poll by default.
- **Speculative decoding** with the built-in MTP head or an external DFlash2 draft package.
- **Bounded request batching**, admission control, backpressure, and cancellation.
- **Serving** through native token APIs and an OpenAI-compatible subset with SSE.
- **One binary, one command, no Python at runtime**: `infer /path/to/model` serves with device-derived defaults.
- **Operational tooling** for health, events, traces, checkpoints, and diagnostics.

## Architecture

```text
Service   CLI, native HTTP/SSE, OpenAI subset, diagnostic agent
Runtime   admission, scheduling, state/KV lifecycle, completion, observability
Model     package loading, provider registry, execution graph, compiler
Backend   CUDA (cuTile / SIMT) and Metal (MSL) kernels
```

Model providers own model semantics. A shared IR and kernel API keep model names out of the scheduler and device code, so adding a provider or a workload does not require a second engine.

See [code layout](docs/architecture/layout.md), [backend capabilities](docs/architecture/backends.md), and the [runtime design](docs/architecture/runtime.md).

## Workloads

| Workload | Purpose |
|---|---|
| **Generate** | Token generation with batching, optional speculative decoding, and streaming |
| **Embed** | Embedding extraction from a compatible model head |
| **Rerank** | Candidate ranking from a compatible model head |
| **Decision** | Structured decisions from a declared schema |

Workload support depends on the model provider, head, and backend. Built-in providers currently cover the Qwen 3.x family, including hybrid and MoE variants; an arbitrary Hugging Face package is not guaranteed to load or run. See [model execution](docs/guides/model-execution.md) and [adding a model](docs/guides/adding-a-model.md).

## Requirements

| Component | Requirement |
|---|---|
| Build toolchain | `1.99.0` pinned by `rust-toolchain.toml`; MSRV `1.90` |
| Build helper | `make` entry points use Python 3.12+; a direct `cargo build` does not |
| Linux / CUDA build | CUDA Toolkit headers, C compiler, and libclang |
| Linux / CUDA run | NVIDIA GPU and driver, plus the cuTile JIT toolchain (`tileiras`) |
| macOS / Metal | A Metal-capable Apple GPU; Xcode command-line tools to build |
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

The server listens on `http://127.0.0.1:8080`. With no execution flags, a supported CUDA package automatically selects MTP depth 2 and BF16 target attention, and falls back to canonical attention without MTP (logging the reason) when capacity is short. `--num-speculative-tokens 0` disables speculation; `--draft-model PATH` uses a separate DFlash2 draft instead. The [CUDA performance guide](docs/guides/cuda-performance.md) records what is enabled by default and how each profile is accepted.

```sh
target/release/infer /path/to/model --served-model-name model
```

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"model","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

OpenAI routes require a package tokenizer; chat completions also require a chat template. See the [OpenAI API guide](docs/guides/openai-api.md) for streaming, tools, and parameter support.

## Deployment

The runtime artifact is a single native executable (`bin/infer`). Python appears in this repository's build, packaging, and check tooling, but nothing in the released package needs an interpreter, virtualenv, conda, PyTorch, or a Python serving framework.

- **Packages per target.** Linux CUDA (x86_64, AArch64; selectable glibc baselines from 2.17 to 2.41) and macOS Metal (Intel, Apple Silicon) archives come from one cross-compilation pipeline. Each archive ships `bin/infer`, standalone run instructions, `SHA256SUMS`, and a manifest recording target, features, and verification scope.
- **No conversion step.** A Hugging Face Safetensors package is opened in place through read-only memory maps; tokenizer and chat template come from the same directory.
- **No config file to start.** `infer /path/to/model` serves on `127.0.0.1:8080`; memory budget and KV sizing are derived from the device, and flags exist when you need to override them. A packaged binary starts the same way: `./bin/infer --backend cuda serve --package /path/to/model`.
- **Existing clients.** The OpenAI-compatible subset works with clients that only need a new base URL.

What it still needs: on CUDA hosts, an NVIDIA driver and the cuTile JIT toolchain (`tileiras`); Metal hosts need only the OS. Work to remove that toolchain dependency is in progress. There is no CPU backend, and only supported packages run.

## Backends

| Backend | Scope |
|---|---|
| **CUDA on Linux** | cuTile and SIMT kernels, device graphs, non-blocking submit/poll, bounded batching, optional MTP or DFlash2 draft |
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

## Performance

Performance is a requirement to be measured, not a label. A change is accepted on paired runs against a signed baseline profile: same release, same configuration read back from the runtime, raw samples retained and recomputable. A single run, a different configuration, or another engine's setup is not republished here as evidence of speed.

There is therefore no headline "Nx faster" claim in this README. What exists is a measurement contract and three signed baselines whose full profile acceptance is still pending; current status and evidence rules are maintained in the [performance baseline plan](docs/plans/performance/baseline.md) and the [roadmap](docs/plans/README.md).

## Limitations

- Built-in model coverage is narrow; validate each package before relying on it.
- No CPU execution backend, so a GPU is required to run a model.
- A single compute flight: execution overlap is per submit/poll, not a depth-2 batch queue yet.
- CUDA still depends on the cuTile JIT toolchain; AOT coverage is incomplete.
- Multimodal execution is limited to examples, not the serving scheduler.
- Performance and production SLO validation are incomplete; no cross-engine parity is claimed.

## Implementation notes

The engine is written in Rust, with CUDA paths in cuTile/CUDA C++ and Metal paths in Metal Shading Language. These are implementation choices made for memory safety without a garbage collector, predictable latency, and direct access to the two target GPU APIs.

They are not a goal in themselves and do not shape the project's contracts. The CLI, the HTTP/OpenAI surface, and the Safetensors package format are language-agnostic; changing a toolchain is an engineering decision that has to be justified by correctness and measured performance, not a change in what the project is.

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

## License

[Apache License 2.0](LICENSE).
