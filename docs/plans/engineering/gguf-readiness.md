# GGUF 接入前置改造

状态：**G1–G3 已实施并合入 #260；GGUF 本体进行中**。方案源码基点为 `b0abe338`（2026-11-15），实施记录见下节。

## 实施状态（2026-11-15）

G1–G3 已实施并合入 #260（merge `4544a479`），方案文档由 #254 合入。

- G1 已落地 `PackageSource` / `TensorFile` / `PackageProbe` / `TensorSet`。可选资产按首次使用读取，避免只加载权重时读取 tokenizer；因此 tokenizer 以字节交给 `TextAssets` 构造，而不是由 source 直接交出构造好的对象。
- G2 已把 `DeviceWeight` 改为 `Float` / `Fp8Channel` / `Fp8Block` / `Nvfp4`；`Fp8Block` 额外保留 `token_input`，因为现有 checkpoint 允许 block scale 与 token 量化分别声明。
- G3 已把投影编码行为收进 `ProjectionWeight` 方法；穷举 match 只剩 `projection_lowering.rs` 与编码专属模块。新增编码仍需要改 `loading/weights.rs`（上传）和 `loading/experts.rs`（expert bank），`bindings.rs`/`draft.rs` 的少量特例也可能需要处理，因此“只改四处”的目标并未字面达成。
- 验收：CI #260 全绿；45 上 CUDA hosted/device 门禁通过；27B NVFP4 的 1536 个 hidden 分量与 64 步贪心 token 与基线逐位一致；release 服务配对在 short/long/batch4 上无回退。`hot_long` 因原生 `/native/v1/stream` 固定 `prefix_boundary: None` 无法产生前缀复用而未纳入配对，完整 `check-release` 仍需单独窗口。
- GGUF 本体已启动：容器解析与 Q4_K/Q5_K/Q6_K 参考解码进入 `infer-format`。参考解码只用于验证自研 kernel，生产路径不会在加载时反量化。Q4_K GEMV 已归入 `projection::grouped_int` 精度分支，当前 845.2 GB/s（4096×5120，roofline 47.2%），仍未过性能门槛；Qwen3-VL 2B 制品已下载并盘点，接入方案见 [Qwen3-VL GGUF 接入](gguf-qwen3vl.md)。

本文只做 GGUF 接入前的三项改造。它们不改变现有行为，即使最后不接 GGUF 也值得做。GGUF 本身的接入（容器解析、k-quant kernel、性能目标）另行立项，这里只记录结论，作为改造方向的依据。

## 背景

### 为什么考虑 GGUF

- 社区量化模型大量只以 GGUF 发布（Unsloth、bartowski 等）。目前项目只能加载 Hugging Face Safetensors 包，见[代码布局](../../architecture/layout.md)。
- vLLM 的 GGUF 支持在官方文档里写着 "highly experimental and under-optimized"，并且已经从主仓库移到 out-of-tree 插件。SGLang 的 GGUF 路径没有优化（Q4_K_M 启动时会弹警告），只支持纯文本，没有 mmproj。"服务级引擎原生、优化过的 GGUF"目前是一个空位。
- 只看格式数量，我们不占优势：safetensors 量化目前只支持 FP8（channel/block）和 NVFP4。差异化的前提是 GGUF 的服务性能明显好于 vLLM/SGLang，并接近 llama.cpp。

### 已确定的方向

- **原生实现 k-quant**，不走"加载时反量化成 BF16"的捷径。后者会丢掉 GGUF 的主要价值：省显存、省带宽。
- **GGUF 作为独立模块**：新建 `crates/model/format`（`infer-format`），只依赖 `infer-core` 和 `infer-ir`。cargo feature 只控制入口，也就是 CLI 和 package 是否依赖 `infer-format`、是否识别 `.gguf` 路径，不要用 `cfg` 一路包到 IR 和 kernel。原因见下文"不做的事"。feature 是实验期的成熟度闸门：设备验收通过后改成默认开启或直接删除，不保留成长期的用户开关，与"性能收益成为自动默认"的惯例一致。
- **GGUF 适配层对外表现为一个虚拟 HF 包**：合成 HF 形态的配置，重建 tokenizer，把 tensor 名映射到 HF 名，并在主机端反转转换脚本对权重做的变换。这样 provider、compiler、runtime 都不需要知道 GGUF。
- **权重编码按量化语义命名**，比如 `Q4K`、`Q6K`、`Q8_0`，不带 gguf 字样。以后 compressed-tensors 的 int4（llm-compressor 产出的 AWQ/GPTQ）可以复用同一套按组整数编码和大部分 decode kernel。旧版 AutoAWQ 格式不在支持范围内。

### 性能可行性（结论，待原型验证）

参考数据：RTX 5090 上 Qwen3.6-27B Q4_K_M，llama.cpp 社区测得 pp512 约 3826 tok/s、tg128 约 77 tok/s。llama.cpp 在这里只是格式参考和性能参照，不是运行时依赖；执行 kernel 由本项目实现。

- **decode**：折合带宽利用率约 72%，有机会追平。
- **prefill**：有效算力约 207 TFLOPS，接近 BF16 dense 峰值，说明 llama.cpp 用的是 int8 MMA（MMQ）。"按 tile 反量化到 BF16 再 MMA"从原理上追不上，必须做 int8 路径。cuTile 0.4 提供了 `mmai` 和 `shri`/`andi`/`bitcast`，具备实现条件。
- llama.cpp 已经有原生 MTP（PR #22673，2026-05）和融合的 GDN kernel，这两项不构成我们的优势。

GGUF 立项后的第一步是 1–2 周的原型：在本项目内实现 Q4_K GEMV（M=1/4）和 int8 MMQ tile，直接读取 GGUF block，不依赖 llama.cpp 运行时。数值上用 `infer-format` 的参考解码校验；性能上以 5090 的带宽/算力 roofline 和自研 BF16/FP8 路径为门槛，`llama.cpp test-backend-ops` 的 Q4_K MUL_MAT 数据只作为可选的外部参照。若外部参照不可用（模型架构或网络限制），不阻塞原型。本文的改造不依赖这个原型的结论。

## GGUF kernel 原型（自研，不依赖 llama.cpp）

llama.cpp 只用于理解 GGUF block 布局和外部性能参照；生产 kernel、调度和内存布局都由本项目实现。`infer-format` 只负责容器、metadata、张量描述和参考解码。

### 数据布局

- 设备上保留 4-bit code，不在加载时反量化成 BF16。GGUF 的原始 super-block 是 Q4_K 每 256 个值 144 字节（`d`、`dmin`、12 字节 scale/min、128 字节 nibble），Q5_K 176 字节，Q6_K 210 字节；这是加载格式，不强制成为设备布局。
- 设备布局由本项目定义：加载时可以把 6-bit scale/min 解包成 kernel 友好的 u8/f16 数组，或按 tile 重排 code；仍然只保留量化数据，不产生 BF16 权重。kernel 侧不依赖 llama.cpp 的结构体。
- host 侧按张量做 shape/name 映射和视图切分；kernel 只接受 `[rows, columns]` 的 row-major block 矩阵，`columns` 必须是 block 元素数的整数倍。
- `GroupedInt` 编码携带块大小、scale/min 布局和 super-block 几何；`ProjectionWeight` 增加对应分支，`projection_lowering.rs` 仍是唯一的 kernel 选择点。

### Q4_K GEMV（decode，M=1/4）

- 一个 CTA 负责一个或多个输出行；每个 K 维 block 由 32 个 nibble 字节覆盖 64 个值，先解出 6-bit scale/min，再按 `d*scale*code - dmin*min` 累加。
- 目标是把 `d`/`dmin`/scales 的读取和 nibble 解包藏在带宽等待里；M=1 完全受权重带宽限制，M=4 复用同一 block。
- 先做 F32 累加的简单版本，验证逐位/容差；再考虑把 nibble 解包成 int8 并用 `dp4a`/int8 MMA 降低 ALU 压力。

### int8 MMQ（prefill）

- 按 K 维 tile 把 Q4_K block 反量化到 shared memory 的 int8 tile，scale/min 留在寄存器或 shared；用 int8 MMA 累加 int32，再在每个 block 边界乘 `d*scale`、减去 `dmin*min` 的贡献。
- 不做“整张权重反量化成 BF16”的中间态；逐 tile 解包，避免带宽和显存峰值回退。
- 需要覆盖 K 尾块、N 尾块、M<MMA tile 的掩码。

- **当前进展（2026-11-15）**：cuTile `mmai(signed, signed)` 已在 5090 上通过 16×64×16 的逐元素整数参考（exact match）。Q4_K 的 host 量化 + int8 MMA 原型能运行，但与参考解码仍有约 6% 偏差；因子/偏置映射或 32-value 分组修正尚未定位，未合入生产路径。单 group 隔离结果：code、low/high factor、bias、x_scale 都逐项对上（相对误差 <1e-6）；把完整 kernel 限制到 1/2/4/8 个 32-value group 时最大绝对误差分别为 1.8e-3/3.4e-3/6.0e-3/1.2e-2，完整 64 group 时升到 9.5e-2。剩余偏差在多 group 累加/激活量化上，尚未定位。

### 验证与门槛

- 正确性：用 `infer-format` 的参考解码对同一批真实 Q4_K/Q5_K/Q6_K 张量逐位比较；kernel 先在 F32 路径上做到逐位一致，再对 int8 路径定义明确的累加/舍入契约。
- 性能：decode 与 5090 带宽 roofline 比较，prefill 与 int8 MMA 峰值比较；同时跑自研 BF16/FP8 路径作为同机基线。`llama.cpp test-backend-ops` 的 Q4_K MUL_MAT 只作可选外部参照，环境不支持时不阻塞。
- **M0 基线（2026-11-15，45 上的 RTX 5090，`~/work/llama.cpp-patched-build/bin/llama-bench`，CUDA，`ngl=99`）**：

  | 测试 | llama.cpp |
  |---|---:|
  | pp16 | 2402.03 tok/s |
  | pp128 | 11432.47 tok/s |
  | tg16 | 554.37 ± 25.84 tok/s |

  同一制品（Qwen3-VL-2B Q4_K_M）在本项目 native 路径上 16 token 解码 `max_tpot=5.573 ms`，约 **179 tok/s**，decode 差距约 **3.1×**。prefill 目前是逐行 grouped-int GEMV，不能与 llama.cpp 的 int8 MMQ 直接比较；Q4_K 单张量实测 608 GB/s、Q6_K 679 GB/s（kernel 微基准 835/741 GB/s）。
- M0 由此完成；模型级 pp/tg 的下一步是把 prompt 的逐行 GEMV 换成 int8 MMQ/GEMM。

## 为什么先改造

如果直接接 GGUF，会碰到三处和格式、编码强耦合的地方：

1. **package 层写死了 safetensors 目录结构。** 分片类型、元数据文件名、模型识别规则都是按 HF 目录写的。GGUF 要伪装成虚拟 HF 包，就得在多个读取点各自插分支，文件元数据还会被重复读取。
2. **权重编码的参数是平铺的可选字段。** 哪个字段对哪种编码有效只靠约定，新增一种带两级 scale/min 的编码时，类型系统帮不上忙。
3. **CUDA 投影按编码分支散在各处。** 新增一种编码要沿二十多个文件逐个补分支，漏掉一处就可能在 budget、并发准入、MoE、融合或 draft 路径上出现运行时 unsupported，或者算错。

先把这三处收成单一接缝，GGUF 和以后的 compressed-tensors int4 都只需要在接缝上加实现。

## 现状（`b0abe338`）

### package 层

- `ModelPackage::shards`（`crates/model/package/src/storage/package.rs`）和 `QuantizedPackage::shards`（`crates/model/package/src/storage/quantized.rs`）的类型都是 `BTreeMap<String, SafetensorsFile>`。
- `config.json` 在多处按根目录路径各自读取：`storage/package.rs`、`storage/quantized.rs`、`storage/quantized/draft.rs`、`input/multimodal.rs`（另外还读 `preprocessor_config.json`）、`generation/read.rs`（另外还读 `generation_config.json`）。
- tokenizer 和 chat template 由 `input/text.rs` 按文件名探测并读取，包括 `tokenizer.json`、`tokenizer_config.json` 里的 `chat_template`、`chat_template.jinja`。
- CLI 的模型识别靠文件是否存在：`crates/service/cli/src/commands/server.rs` 检查 `tokenizer.json`，`crates/service/cli/src/commands/models.rs` 读 `config.json`。
- **越层**：`crates/engine/workloads/src/readouts.rs` 直接 `use infer_models::SafetensorsFile` 去打开 `readouts.safetensors`，engine 层知道了文件格式。

### 编码数据

`DeviceWeight`（`storage/quantized.rs`）由 `encoding: WeightEncoding` 加上 `scale`、`global_scale`、`input_global_scale`、`weight_block`、`fp8_token_input` 这些平铺的可选字段组成。`WeightEncoding` 有 `Float`、`Fp8Channel`、`Fp8Block`、`Nvfp4` 四个分支。`DeviceWeight` 派生了 `Serialize`/`Deserialize`，但用 grep 没有找到它被写进文件或缓存的调用点。实施前要再确认一次，决定是否需要兼容迁移。

### CUDA 投影

- `ProjectionWeight`（`crates/backend/cuda/src/mlp/projection.rs`）已经是唯一的投影权重类型，有 `Dense`、`Fp8`、`Fp8Block`、`Fp4` 四个分支，并已有 `bytes()` 这类方法。
- `resident/projection_lowering.rs` 已经统一了各种捕获模式下的投影选择。
- kernel 的公开面已经收进 `crates/backend/cuda-kernels` 的 `contract`，每个算子族一个 trait，比如 `gemm_fp4`、`gemm_fp8`、`dense`。
- 但在编码专属模块之外，仍有约 15 个文件对 `ProjectionWeight` 做穷举 match，或者写了只认 `Dense` 的分支：
  - `loading/`：`bindings`、`concurrency`、`experts`、`budget`
  - `resident/`：`batch_projection`、`capture`、`dflash2/mod`、`fusion_batch`、`moe`、`pool_gemv`、`prefill_projection`、`program`、`batch`
  - 另外还有 `tuning.rs` 和 `mlp/mod.rs`
- `nvfp4_gemm/*`、`device/vendor_fp8/*`、`tuning/fp8_gemm.rs` 属于编码专属模块，只认自己的编码是合理的，不在改造范围内。

## 技术方案

### G1：package source 与 `TensorFile`

目标是把"包从哪里来"收成一个接缝，后面的代码只消费字节和已构造好的对象。

- 新增具体的 enum `TensorFile { Safetensors(SafetensorsFile) }`，替换两个 package 里的 `SafetensorsFile` 字段。GGUF 以后加一个分支。按 AGENTS.md 的要求，用具体 enum，不引入 trait。
- 新增 package source，打开包时一次性读完并做边界校验，然后提供：
  - config
  - generation config
  - preprocessor config
  - tokenizer（已构造好的 `tokenizers::Tokenizer`）
  - chat template
  - tensor 文件表

  上面列出的读取点都改成从 source 取，不再各自拼路径。读取上限继续沿用 `read_bounded` 和现有常量。
- `input/text.rs` 改为接收已构造好的 tokenizer 和模板来源。现有"哪个模板优先"的规则和错误信息保持不变。
- CLI 的模型识别改成调用 package 层的探测函数，不再检查文件是否存在。`models` 和 `server` 两条入口共用这个函数。
- `readouts.rs` 改为通过 `infer-models` 暴露的读取接口拿 `readouts.safetensors`，workloads 不再引用 `SafetensorsFile`。

验收：

- 全部在 Mac 上可以运行的测试通过：现有 loader、package 测试，`verify --golden`，CLI 识别测试。
- 新增用例：缺文件、超限、模板优先级、识别失败时的错误码和信息与改造前一致。
- 输出与改造前逐位一致。

### G2：编码分支携带参数

- 把 `DeviceWeight` 改成"分支携带自己的数据"：
  - `Float { data }`
  - `Fp8Channel { data, scale, token_input }`
  - `Fp8Block { data, scale, block }`
  - `Nvfp4 { data, scale, global, input_global }`
- 删掉平铺的可选字段。原本靠运行时检查"字段不能为空"的地方，改为由类型保证。
- 这一步不新增编码。GGUF 立项时再加 `GroupedInt { bits, group, scale, min_or_zero, super_block }` 这一类分支，同时覆盖 k-quant 和以后的 compressed-tensors int4。
- 消费方的改动集中在 `storage/quantized.rs`（13 处 `WeightEncoding::` 引用）和 CUDA 的 `loading/{weights,experts,bindings,head_padding}.rs`、`resident/dflash2/weights.rs`。

验收：

- 和 G1 一样做 package 层回归。
- 每种编码各补一个反例：缺 scale、block 不整除等，确认在构造阶段被拒绝。
- CUDA 部分在 Linux 上通过编译和 lint。

### G3：投影编码接缝

- 在 `ProjectionWeight` 上补齐按编码区分的方法，用来替换上面约 15 个文件里的穷举 match。方法大致有：
  - 形状校验（已有 `validate`）
  - 设备字节数（已有 `bytes`）
  - tuning key 和 dtype 名，取代 `tuning.rs` 里的 match
  - 并发准入时的激活和 workspace 开销，取代 `loading/concurrency.rs`
  - MoE 和 draft 权重的转换，取代 `resident/moe.rs`、`resident/dflash2/mod.rs`
  - 融合与 prompt 路径能否走某条 kernel 的判断
- 调用方改成调方法。只认 `Dense` 的 `let ... else` 分支改成显式的能力查询，返回现有的 unsupported 错误。
- 新编码的 kernel 族以后放进 `cuda-kernels` 的 `contract`（新增一个 trait 方法）和 `tile` 实现。host 侧放在独立的 `resident/<编码>/` 目录。`projection_lowering.rs` 是唯一的选择点。
- 目标是：新增一种编码时，只需要改 `ProjectionWeight` 所在模块、`projection_lowering.rs`、`cuda-kernels` 的对应族，以及新目录，不再碰上面那约 15 个文件。

验收：

- 纯重构，CUDA 输出必须和改造前**逐位一致**。
- 在 45 上按[质量门禁](../../guides/quality-gates.md)跑 CUDA 门禁，设备测试一进程一个用例。
- 按[基线契约](../performance/baseline.md)做一次 release 服务配对，确认没有性能回退。只报告"无回退"，不声称有收益。

## 顺序与 PR 拆分

1. **PR-A：G1。** 只改 Mac 上能运行的部分，风险最低，可以先合。
2. **PR-B：G2 + G3。** 两者都改 CUDA 的消费方，合在一起只需要一次设备验收窗口。如果 diff 太大，G2 可以先单独提一个 PR，但 CUDA 部分要等 G3 一起做设备验收后再合。
3. 合入后，GGUF 正式立项：先做 M0 基线和原型（1–2 周）。M0 以自研 kernel 的带宽/算力 roofline 与自研 BF16/FP8 路径为门槛，llama.cpp 仅作可选参照；通过门槛后再扩展 `infer-format`。

G1 和 G2/G3 之间没有代码依赖，可以并行开发。

## 不做的事

- **不用 `cfg(feature = "gguf")` 包裹 IR、`WeightEncoding` 或 CUDA kernel。** 原因有三：穷举 match 会到处跟着加 `cfg`；序列化形态会随 feature 变化；门禁矩阵会翻倍（`gate.sh` 目前只有 `cuda` 一个平台 feature）。
- **不引入 trait 抽象 package 格式。** 一个具体 enum 已经够用。
- **不顺带做平台层抽取**（KV 改用 `infer-state`、投机接受逻辑上移、budget 迁到 compiler）。这些另行评估，不作为 GGUF 的前提。
- **不新增编码、kernel 或依赖。**

## 待确认

- `DeviceWeight` 的 serde 派生是否被持久化。G2 实施前要确认。
- 模型识别改为探测之后，`models` 命令的输出格式是否需要保持完全一致。目前的规则是保留公开行为和输出格式。
- G3 的方法清单，以实施时逐个 match 点核对的结果为准。本文的文件列表只是起点。

## 参考

- [vLLM GGUF](https://docs.vllm.ai/en/latest/features/quantization/gguf/)
- [vllm-gguf-plugin](https://pypi.org/project/vllm-gguf-plugin/)
- [SGLang Model Loading](https://docs.sglang.io/docs/advanced_features/model_loading)
- [RTX 5090 llama-bench 数据集](https://huggingface.co/datasets/omegaprime669/rtx-5090-benchmarks)
- [llama.cpp MTP（PR #22673）](https://github.com/ggml-org/llama.cpp/pull/22673)
- [vLLM AutoAWQ 弃用说明](https://docs.vllm.ai/en/latest/features/quantization/auto_awq/)
