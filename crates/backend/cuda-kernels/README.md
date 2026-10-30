# CUDA kernel implementations

Kernel bodies for `infer-backend-cuda`. The host crate owns device allocation, streams, CUDA
graphs, KV, slot batching and tuning; this crate owns launches and device artifacts.

## Profiles

Exactly one profile is compiled per build; enabling both is a compile error.

| Feature | Toolchain | Runtime |
|---|---|---|
| `tile` | CUDA 13.x with `tileiras` | cuTile JIT or per-arch cubins |
| `aot` | none at build time | precompiled SASS through the driver |

## Layout

```text
src/tile/<family>_kernels.rs   cuTile modules; re-exported by `tile/<family>`
src/aot/<family>.rs            driver launcher + artifacts (next)
```

The device/stream/buffer contract moves here next; until then the host keeps its own allocation
and passes tensors to the exported launch functions. See
`docs/plans/engineering/cuda-kernel-provider.md` for the target dimensions and extension rules.
