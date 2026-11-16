# GGUF KV cache design

状态：**设计评审稿**。本文件先定义 GGUF 模型的 KV cache 策略；实现分阶段落地，避免把 KV 编码塞进 GGUF 权重编码或 CUDA 专属逻辑。

## 既有设计基线

这份设计不另起一套 KV 架构，直接沿用现有约定：

- [KV Manager](../../architecture/kv-manager.md)：块池、引用计数、COW、前缀匹配和统计归平台层；CUDA/Metal 只提供各自的设备快照/页适配；
- [KV cache 分页方案](../../plans/performance/kv-cache-paged-attention.md)：SPI 已经定义 `kv_cache()`、`state_page_growth()`、`reusable_prefix*()` 和 `KvCacheInspection`；当前缺口是 CUDA 仍走 SPI 默认值、没有实现这套分页契约；
- [Backend 架构](../../architecture/backends.md)：`KernelRegistration`/`ExecutionProgram` 绑定 backend，公共 trace/probe/stats 留在 IR；GPU profile 是 command buffer 时间，OpTrace 是 CPU encoding，二者都不能当作逐 kernel GPU counter；
- [GGUF readiness](gguf-readiness.md)：GGUF 是虚拟 HF 包，KV cache 是运行时策略，不是文件里的权重编码。

因此 GGUF KV 的工作必须：

1. 先复用/补齐平台的 KV 分页契约，不能在 CUDA 里再建一套 `SlotPool` 策略；
2. 只把 GGUF 的几何和默认 KV 策略接进来；
3. 新增编码时扩展已有 `KvCacheInspection`/能力声明，而不是增加平行 inspection。

## 先澄清：GGUF 文件里没有 KV cache

GGUF 是权重容器。它包含：

- 文本/视觉权重；
- context length、head count、head dim、rope 等几何 metadata；
- tokenizer 和 chat template。

它**不包含** KV cache 的存储类型。llama.cpp 的 `--cache-type-k/v` 是运行时选项，不是 GGUF 文件字段。因此这里的目标不是“解析 GGUF KV tensor”，而是：

1. 从 GGUF 合成正确的 KV 几何；
2. 给 GGUF 模型选择 KV 存储编码；
3. 让 CUDA 和 Metal 共用同一套策略/能力/观测，而不是各自实现一份 GGUF 特判。

## 当前状态

- `GgufPackage` / `QuantizedPackage::open_gguf` 从合成文本 config 得到层数、KV heads、head dim、context；
- `kv_cache_dtype` 对 GGUF 被设为 `None`；
- CUDA `KvElement` 目前只实现 `f32` 和 `Fp8`，没有 F16/Q8_0；
- CUDA 也尚未实现 [分页方案](../../plans/performance/kv-cache-paged-attention.md) 中的 `kv_cache()`/`state_page_growth()` 契约；
- CLI 只有 `backend` 的 KV 状态展示，没有面向 GGUF 的 KV 编码选项；
- `inspect-package` 不报告 KV 几何、默认编码或每 block 字节数；
- 因此 GGUF 默认落到 F32 KV，显存占用远高于 llama.cpp 的 F16/Q8_0。

这是实际缺口，不是 GGUF 解析缺口。

## 目标

- 不新增平行的 KV 管理层；分页、prefix、预算继续由平台契约拥有；

- GGUF 默认 KV 使用 F16（与 llama.cpp 默认对齐），而不是 F32；
- 支持运行时选择 F16 / F32 / FP8 / Q8_0 / Q4_0（按 backend 能力）；
- K 和 V 可分别选择；
- page/block 字节数、prefix cache 身份、inspection 都包含 KV 编码；
- 平台侧定义策略和报告，backend 只实现存储/kernel；
- 不把 KV 编码写进 `WeightEncoding` 或 GGUF 文件元数据。

## 平台边界

| 层 | 内容 |
|---|---|
| `infer-ir` | `KvCacheEncoding`、`KvCachePolicy`、inspection 字段 |
| `infer-spi` | `BackendProvider` 的 KV 能力声明 |
| model/package | GGUF 合成文本几何、请求/CLI 策略、包默认值 |
| scheduler/state | 按编码计算 page bytes、prefix 兼容性 |
| CUDA/Metal | KV 存储、量化/反量化、attention、prefix copy |

GGUF provider 只贡献几何和默认值；KV 编码不进入 `GgufPackage` 的权重 slot。

## 类型

```rust
enum KvCacheEncoding {
    F16,
    F32,
    Fp8E4m3,
    Q8_0,
    Q4_0,
}

struct KvCachePolicy {
    key: KvCacheEncoding,
    value: KvCacheEncoding,
}

struct KvCacheFormats {
    encodings: &'static [KvCacheEncoding],
}
```

`KvCacheFormats` 和 `BackendFormats` 分开：权重能执行 Q4_K，不代表 KV 能存 Q8_0。

选择优先级：

```text
CLI / 请求显式指定
> 包 config 显式声明
> GGUF 默认（F16）
> backend 默认
```

backend 不支持显式编码时，在加载前失败，不静默回退。

## 存储

Q8_0（对齐 GGML）：

```text
每 32 个值：32 × i8 codes + 1 × f16 scale = 34 B
```

Q4_0：

```text
每 32 个值：16 B packed nibbles + 1 × f16 scale = 18 B
```

`KvElement` 不应继续用“元素类型”表达 block 编码。建议改成：

```rust
enum KvStorage<'a> {
    F32 { keys, values },
    F16 { keys, values },
    Fp8 { keys, values },
    Blocked { codes, scales, block },
}
```

attention kernel 按 tile 解码；append kernel 写入时量化；prefix copy 同时复制 codes 和 scales。

## 生命周期

- page bytes 用 `KvCachePolicy` 计算；
- prefix cache 的 key 必须包含 KV 编码，不能把 F16 prefix 复用到 Q8_0；
- KV inspection 增加 `key_encoding`、`value_encoding`、`bytes_per_block`；
- 层数/heads/head_dim/context 仍来自合成文本 config；
- 视觉 token 不改变 KV 编码。

## CUDA 落地顺序

1. F16 KV 存储 + attention/prefix copy；
2. `--kv-cache-dtype` 的加载前能力校验和 inspection；
3. Q8_0 append + attention + prefix copy；
4. Q4_0；
5. 与 F16/F32 的数值和显存对比；
6. Metal 只实现平台契约。

## 验收

- GGUF Qwen3-VL 默认 KV 从 F32 降到 F16，显存/容量可测；
- Q8_0 KV 的 attention 输出相对 F16 有明确误差门槛；
- long-context 和 prefix reuse 在 F16/Q8_0 下都正确；
- `inspect-package` 报告 KV 几何、默认编码、bytes/block；
- backend 不支持时在加载前失败；
- 平台侧没有 CUDA/Metal 专属字段。

## 开放问题

1. K/V 是否必须同编码？本设计允许不同，但 page 计算和 prefix key 要同时覆盖两者。
2. scale 用 f16 还是 f32？先按 GGML f16，后续按精度/性能决定。
3. GGUF 默认是否直接 F16，还是保留 F32 由 CLI 选择？建议默认 F16，与 llama.cpp 对齐。
4. Q8_0 KV 是否作为第一阶段唯一 quantized 编码，Q4_0 后置。
