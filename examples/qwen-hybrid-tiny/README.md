# qwen-hybrid-tiny：数值对照样例包

这是一个未训练的小型测试包，由官方 Transformers `Qwen3_5ForCausalLM`（Transformers 5.18.0、Torch 2.14.1、seed 20261004、F32 CPU eager attention）导出；`golden.json` 由该 CPU 参考实现生成，Rust 执行路径在 CUDA/Metal 上对照这些 golden。

模型包含一个 linear-attention 层和一个带 GQA、QK normalization、partial RoPE 与 output gating 的 full-attention 层；norm、decay 与 gate 参数被刻意设为非平凡值。

## 内容

| 文件 | 说明 |
|---|---|
| `config.json` | 模型配置 |
| `model.safetensors` | 实际生成的权重 |
| `golden.json` | 独立参考：整段 layer 输出、prefix logits/hidden 与 cached greedy decode token |
| `readouts.safetensors` | embedding / rank / decision projection 参考 |
| `tokenizer.json`、`tokenizer_config.json` | 服务夹具用的微型 WordLevel tokenizer |

这些是特定架构与配置下的数值参考，不能代表 Qwen3.8-27B 的质量或性能。真实 Qwen3.8 tokenizer 另用固定版本的官方资产单独测试。

## 使用

```sh
# 数值验证（CUDA 或 Metal）
target/release/infer verify --package examples/qwen-hybrid-tiny \
  --golden examples/qwen-hybrid-tiny/golden.json --atol 0.000002 --rtol 0.00002

# 执行样例请求
target/release/infer run --package examples/qwen-hybrid-tiny \
  --requests examples/requests.json --config examples/runtime.json
```

模型执行需要 CUDA 或 Metal；无 GPU 的 `make test` 覆盖标量参考算子、协议与不依赖设备的测试。构建见[开发指南](../../docs/guides/development.md)。

## 重新导出

```sh
uv venv artifacts/golden-env --python 3.12
uv pip install --python artifacts/golden-env/bin/python -r tools/fixtures/requirements.txt
artifacts/golden-env/bin/python tools/fixtures/export-qwen-golden.py --output examples/qwen-hybrid-tiny
```

Python 依赖只用于导出 golden；Rust 可执行文件直接读取已保存的包，不需要 Python 或 Torch。另一组分组比例的样例见 [qwen-hybrid-grouped](../qwen-hybrid-grouped/README.md)。
