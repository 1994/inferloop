# CUDA kernel implementations

Kernel bodies for `infer-backend-cuda`. The host owns device allocation, CUDA graphs, KV, slot
batching and tuning; this crate owns launches and device artifacts.

## Profiles

Exactly one profile is compiled per build; enabling both is a compile error.

| Feature | Build needs | Runtime needs |
|---|---|---|
| `tile` | CUDA 13.2+ headers | `tileiras` JIT on a 13.x driver |
| `aot` | nothing from a toolkit | a cubin built for the device, loaded through the driver |

## Layout

```text
src/contract/            what the host may see: target, plans, one trait per operator family
src/tile/<family>/       cuTile kernels; only this profile may name `cutile`
src/aot/<family>         driver launcher and its `.cu` source, side by side
tools/cuda/aot-kernels.json   AOT entries and the toolchain pinned for each architecture
```

Both profiles use the same family names. To bring a family to a device that has no cuTile, read
`src/tile/<family>/`, write the `.cu` next to `src/aot/<family>`, register it in
`aot-kernels.json`, and compare against an independent CPU reference on the device.

## Launch contract

`contract::kernels` states it once: a launch is ordered on the stream and completes by the next
synchronization. An implementation may wait earlier, but no caller can rely on that.

See `docs/plans/engineering/cuda-kernel-provider.md` for the target dimensions and extension rules.
