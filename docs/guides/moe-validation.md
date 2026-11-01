# 单张 5090 上的 MoE 验证

完整权重超过显存时，先保留模型计算结构，缩小层数、隐藏维度、词表和专家数，与独立的官方实现对照；另外保留真实注意力头尺寸，覆盖 CUDA 的形状分支。缩小模型通过不代表大模型的容量、吞吐和长上下文已经通过。

## 当前实现

执行图包含路由、选中专家投影和加权合并。路由留在 GPU，只计算 top-k 专家；所有专家权重仍需常驻显存。支持 top-k 归一化开关、共享专家与 sigmoid 门控、部分层使用普通 MLP。

CUDA 支持 BF16 专家、按通道 FP8 和 128 列分块 FP8，遵守 checkpoint 的输入量化策略。支持逐专家矩阵与合并专家 bank。Next 的浮点融合 Q/K/V/Z 和 B/A 投影通过受边界检查的行视图转换到执行图布局。

导入支持 Qwen3 MoE、Qwen3-Next、Qwen3.5 MoE。独立完整对照目前覆盖前两者；Qwen3.5 MoE 的导入支持不能替代其完整模型验收。

当前限制：

- NVFP4 专家、MoE MTP、量化的 Next 融合线性注意力投影尚未实现。
- 普通 `ModelPackage` 主机加载入口尚不支持合并专家与融合行视图；CUDA 与本项验证使用 `QuantizedPackage`。
- 没有专家 CPU 卸载或多 GPU 专家并行。top-k 减少计算量，不会按同样比例减少常驻权重。
- MoE 的 prompt Delta 使用逐行路径。128 维头尺寸的内存检查发现 chunk Delta 生成内核存在非法 scratch atomic，因此目前关闭该优化；卷积和普通投影仍可批量执行。
- [Qwen3.8-Flash-Next 官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) 声明独立的 `qwen4_exp` 架构，包含 HyperConnection、PLE 和稀疏注意力。它尚未实现，不能当成 Qwen3-Next 注册后宣称支持。

## 独立参考

| 目录 | 验证目的 |
|---|---|
| `examples/qwen-moe-tiny` | 普通 Qwen3 MoE，归一化 top-k |
| `examples/qwen-moe-packed-tiny` | 合并专家 bank，不归一化 top-k |
| `examples/qwen-next-moe-tiny` | Gated DeltaNet、完整注意力、共享专家、融合投影 |
| `examples/qwen-next-moe-head128` | 保留 128 维两类注意力头，隐藏维度 256 |

导出器先保存精确 BF16 权重，再让 Transformers 在 CPU 使用这些权重做 F32 eager 前向。每个 token 前缀独立计算，无缓存。生成器、种子和依赖版本记录在 golden，与 Rust/CUDA 实现分开。

已验证环境：torch `2.8.0+cu128`、Transformers `4.57.0`、safetensors `0.6.2`、tokenizers `0.22.1`。参考导出无需 GPU；独立清单在 `tools/fixtures/requirements-moe.txt`，不改变其他 fixture 的版本。

```sh
python3 tools/fixtures/export-moe-golden.py --output examples/qwen-moe-tiny
python3 tools/fixtures/export-moe-golden.py --output examples/qwen-moe-packed-tiny --packed --no-normalize
python3 tools/fixtures/export-moe-golden.py --output examples/qwen-next-moe-tiny --family next
python3 tools/fixtures/export-moe-golden.py --output examples/qwen-next-moe-head128 --family next --production-heads
python3 tools/fixtures/export-moe-operators.py --output examples/moe-operators/golden.json
```

FP8 算子 golden 使用独立 PyTorch 转换、输入量化、缩放和矩阵乘法，覆盖 channel/block scale、输入量化开关与已路由输入开关，共八组。

## 运行验证

在 CUDA Linux 机器运行；本次开发路径是远端 `/home/r/code/inference.rs`。GPU 验证使用保护入口，限制主机内存并串行占用 GPU。

```sh
export CUDA_TOOLKIT_PATH=/opt/cuda
export BINDGEN_EXTRA_CLANG_ARGS="-isystem $(gcc -print-file-name=include)"
cargo build --locked -p infer-backend-cuda --features cuda --example cuda-moe-check

bash tools/bench/safe-run.sh --memory-gib 8 env CUDA_TOOLKIT_PATH=/opt/cuda \
    target/debug/examples/cuda-moe-check examples/qwen-next-moe-head128
```

decode attention 按[数值合同](../plans/performance/baseline.md)使用 BF16 乘积加 F32 累加，所以对照的口径是整行最大误差不超过该行最大幅值的 2%，且最大元素的位置一致，不是 F32 舍入级别。

默认检查每个前缀的 decode hidden/logits、32 行 prompt 图最后有效行、三 token verification、重复 reset 和 loader 释放后的权重生命周期。容差是 `5e-5 + 5e-4 * abs(expected)`。非有限结果、形状不符或超出容差都会失败，末行 JSON 记录实际路径与最大绝对误差。

`--decode-only` 只检查逐 token、重置和生命周期；`--verify-only` 另外检查三 token verification，不会报告未执行的 prompt 检查。

```sh
cargo test --locked -p infer-backend-cuda --features cuda --lib --no-run
# 使用上条命令实际打印的测试二进制路径：
bash tools/bench/safe-run.sh --memory-gib 8 env CUDA_TOOLKIT_PATH=/opt/cuda \
    target/debug/deps/infer_backend_cuda-<hash> resident::moe::tests --ignored --test-threads=1
bash tools/bench/safe-run.sh --memory-gib 8 env CUDA_TOOLKIT_PATH=/opt/cuda \
    target/debug/deps/infer_backend_cuda-<hash> resident::reset::tests --ignored --test-threads=1
```

路由测试覆盖平分、大幅度分数、非二次幂专家数、`k=1`/`k=experts`、归一化开关、小于 `-1e30` 的有限分数，以及 512 专家选 10 的实际路由尺寸。重置测试覆盖部分 tile、三维循环状态与 KV 形状，要求每个元素位模式为零。

```sh
bash tools/bench/safe-run.sh --memory-gib 8 env CUDA_TOOLKIT_PATH=/opt/cuda \
    /opt/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 77 \
    target/debug/examples/cuda-moe-check examples/qwen-next-moe-head128
```

## 大模型仍需验收的范围

只获取真实配置与 safetensors 索引/文件头，可以检查架构、权重名、维度、dtype、scale 布局和容量估算，无需下载全部权重。这是导入与容量检查，不能证明推理正确。

再在单卡上加载一个真实层或少数专家，保留真实隐藏维度、中间维度与量化格式，对照算子和层输出。最后在足够内存的机器验收完整 checkpoint、长上下文、并发与性能。当前缩小模型提供可重复的正确性回归，不提供大模型性能结论。
