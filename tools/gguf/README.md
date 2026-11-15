# GGUF functional conversion

`convert.py` is the fast functional path until the native GGUF package source lands. It reads a
Qwen3-VL text GGUF and an optional mmproj, dequantizes to F16/F32, maps names, and writes an
HF-style Safetensors package that the existing loader already accepts.

It is explicit about being temporary: weights are materialized on disk, so it does not preserve
GGUF's memory or bandwidth advantage. Native Q4_K loading and the virtual package source replace
it later.

## Dependencies

- Python 3.12+
- `numpy`, `safetensors`
- llama.cpp `gguf-py` on `PYTHONPATH`

```sh
PYTHONPATH=/path/to/llama.cpp/gguf-py python3 tools/gguf/convert.py \
  --gguf Qwen3VL-2B-Instruct-Q4_K_M.gguf \
  --assets /path/to/Qwen3-VL-2B-Instruct \
  --out /path/to/converted \
  --text-only
```

`--assets` supplies `config.json` and tokenizer files; `--mmproj` adds the vision tower (including
the two temporal patch slices and deepstack mergers). Drop `--text-only` for the VL package once
deepstack is wired into the engine.

## Verified

Qwen3-VL-2B-Instruct Q4_K_M: 310 text tensors convert, `infer-cli inspect-package` accepts the
package, and `infer-cli run --backend cuda` generates 8 tokens (TTFT ~1.9 s, TPOT ~3.5 ms on the
RTX 5090). The model output is a functional smoke test, not a parity result.
