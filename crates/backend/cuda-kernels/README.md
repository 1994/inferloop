# CUDA kernel implementations

Kernel bodies for `infer-backend-cuda`. The host owns device allocation, CUDA graphs, KV, slot
batching and tuning; this crate owns launches and device artifacts.

## Kernel sources

`tile` and `aot` can be built together. A device without cuTile builds with `aot` alone.

| Feature | Build needs | Runtime needs |
|---|---|---|
| `tile` | CUDA 13.2+ headers | `tileiras` JIT on a 13.x driver |
| `aot` | nothing from a toolkit | a cubin built for the device, loaded through the driver |

## Layout

```text
src/contract/                 what the host may see: arch, plans, recording, one trait per family
src/tile/<family>/            cuTile kernels; only this source may name `cutile`
src/aot/<family>.rs           driver launcher
src/aot/<family>/*.cu         its kernels
tools/cuda/kernels.json       every device artifact; `sidecar` lists AOT entries and toolchain pins
```

To bring a family to a device without cuTile, read `src/tile/<family>/`, write the `.cu`, register it
in the `sidecar` section of `tools/cuda/kernels.json`, and compare against an independent CPU reference on the device.

## Launch contract

A family method records a launch and never waits, because the resident graph captures once and
replays. `Recording` says where it goes: a cuTile scope, or a driver stream that is capturing.
`pool_gemv_bf16` is the worked example: the host records it through the contract on the inference
path, and `examples/aot-pool-gemv.rs` captures and replays the AOT version on a device.

See `docs/plans/engineering/cuda-kernel-provider.md` for the target dimensions and extension rules.
