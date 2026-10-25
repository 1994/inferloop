# DFlash2 模型接入方案

状态：设计方案。基于 2026 年 10 月 9 日的 `7c79d99`，目标为 `z-lab/Qwen3.8-27B-DFlash2`。

整体排期与任务状态见 [路线图](../README.md) 的 D1、D2。本文维护本项设计与验收。

DFlash2 作为独立草稿 provider 接入，共用 [推测解码 SPI 与状态协议](spi.md)。模型图由模型层定义，算法 provider 的规划由公共 workloads 层实现，CUDA 是优先落地的设备执行；其他 backend 按同一契约独立实现并报告能力。现有 MTP 的融合、位置错位和追赶方式不能直接套用到该模型。先完成兼容性检查和独立数值参考，再实现模型图、候选选择与目标验证；资源准入同时执行。

## 模型需求与接入边界

官方 [config.json](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/blob/main/config.json) 声明 `DFlash2DraftModel`、5 个草稿层、hidden size 5120、32 个 attention heads、8 个 KV heads、head dim 128。原生 block size 为 8，块内非因果；target taps 为 `[5, 19, 33, 47, 61]`，草稿 attention 使用长度 2048 的滑动窗口，selector rank 为 256、top-k 为 16。

权重结构与单卡约束的现有记录见 [性能实验记录](../../archive/research/cuda-performance-experiments.md)。接入时将其中的双抽头动态卷积、特征融合和 candidate selector 分别落实为模型配方与算子能力，不把它们解释为通用 Qwen decoder 的默认行为。

第一版只开放 greedy 和线性候选链。完整随机推测采样依赖实际 proposal 分布的推导与导出，不能把 selector 分数直接当成归一化的 `q`。草稿内部的候选 lattice 与 target 的树形验证是两个独立能力。

## 独立 package 与兼容性检查

草稿由独立 `DraftModelBinding` 绑定，模型 registry 按 `DFlash2DraftModel` 解析 provider。package 层负责配置、张量命名和拓扑；CUDA loader 负责设备存储、绑定与图录制。

| 检查项 | 接受条件 |
|---|---|
| target 身份 | target 架构、层数、hidden 几何及版本与草稿要求匹配 |
| target taps | layer ID 有效；明确抽取层输出的位置、归一化与位置映射 |
| tokenizer | token ID 对应关系兼容；不能只比较 vocab size |
| embedding 与 head | 声明可共享，且权重身份、形状、dtype 与设备所有权匹配 |
| 特殊 token | mask、EOS、padding 与请求采样规则一致 |
| 算子 | backend 支持动态卷积、块内非因果 attention、feature fusion 与 selector |
| 资源 | 权重、target/draft 状态、feature 历史、arena、graph 和回滚开销全部纳入报价 |

兼容性失败在加载或规划时报告具体原因。`auto` 可以依据明确策略回退 MTP 或普通 decode；显式要求 DFlash2 时，不能静默改用其他算法。草稿量化是单独的精度方案，需要记录格式和接受长度变化。

## 已实施：D1 第一步的配置与兼容性契约

`crates/model/package` 新增 `providers/dflash2.rs`（从 crate 根导出），只做与设备无关的部分：

| 项 | 实现 | 验收对应 |
|---|---|---|
| 几何 | `DraftGeometry`，`official()` 记录方案里的官方值（5 层、hidden 5120、32 heads、8 KV heads、head dim 128、block 8、taps `[5,19,33,47,61]`、窗口 2048、selector rank 256 / top-k 16） | 配置检查 |
| 内部一致性 | `validate()`：空维度、KV heads 必须整除 attention heads、taps 必须非空且严格递增、top-k 不得为 0 或超过 rank、block 不得宽于窗口 | 配置检查 |
| 架构声明 | `check_declared_architecture()`：显式 DFlash2 请求不被别的算法静默接管 | "显式要求 DFlash2 时不能静默改用其他算法" |
| target 兼容 | `check_target()`：target 必须是因果 decoder、hidden 必须与草稿一致、每个 tap 必须落在存在的 attention 层上，错误里带上 tap 编号 | target 身份 / target taps |
| 算子能力 | `check_operations()`：草稿声明需要的算子集合必须是 backend 上报集合的子集，报出第一个缺失项 | 算子 |
| 资源（几何部分） | `quote(dtype_bytes)`：每 token KV、窗口 KV、每 tap 每 token 的特征字节 | 资源（部分） |

**配置映射与权重清单（第二步，已完成）**

模型本机已有下载（`/home/r/models/Qwen3.8-27B-DFlash2`，apache-2.0），因此键名与张量名取自真实
制品而不是猜测：

- 官方 `config.json` 原样登记为 fixture：`examples/qwen3.8-27b-dflash2/config.json`，配套
  `source.json` 记录 repository/URL，并把**文件 sha256** 当作身份（本地下载未保留上游 revision，
  如实标注，不编造）。`DFlash2Config` 按真实键名反序列化（顶层 + 嵌套 `dflash_config`），
  `geometry()` 映射到 `DraftGeometry`，并有测试断言它与 `official()` 完全一致。
- 配置实测值补充进几何：`intermediate_size 17408`、`vocab_size 248320`、`num_target_layers 64`
  （taps 来自 `dflash_config.target_layer_ids`）。`check_target()` 因此新增**目标层数必须等于
  草稿训练时的层数**这一条（64 层，与 `examples/qwen3.8-27b` 的目标 fixture 一致）。
- `check_block_semantics()`：`is_causal` 必须为 false（块内双向注意力），且不得与目标共享词嵌入。
- `expected_weight_shapes()` / `check_weight_inventory()`：按真实 checkpoint 的结构生成 81 个张量
  名单（6 个共享 + 每层 15 个），其中几何能决定的形状逐项核对（`fc.weight [5120, 25600]`、
  `q_proj [4096, 5120]`、`k/v_proj [1024, 5120]`、`o_proj [5120, 4096]`、`gate/up [17408, 5120]`、
  `down [5120, 17408]`、`q/k_norm [128]`、selector 与 codebook）；两个双抽头卷积张量的形状由
  loader 依分组方式确认，清单里只查名字。缺少、多余、形状不符都会点名报出。

**特征融合参考（第三步，已完成）**

本机 vLLM 环境里有官方模型的实现（`artifacts/vllm-compare/.../vllm/model_executor/models/qwen3_dflash.py`），
因此融合语义是读出来的、不是猜的：`combine_hidden_states()` 对五个抽头 hidden 的**拼接**做一次
`fc` 投影到草稿 hidden，`fc` **无 bias**，投影内**没有归一化**（`hidden_norm` 用在别处）。这也解释了
实测的 `fc.weight [5120, 25600] = hidden × (taps × hidden)`。

据此产出可对照的参考制品：

- 导出器 `tools/fixtures/export-dflash2-fusion-golden.py`（冷路径，用 torch+safetensors，与既有
  golden 导出同一约定），从真实权重里只读 `fc.weight`，对确定性公式构造的输入做 fp32 投影，写出
  `examples/qwen3.8-27b-dflash2/fusion-golden.json`（5120 个值）。
- golden 记录了它来自哪个模型：config 的 sha256、张量名与形状、dtype（BF16）、算术语义、torch
  版本、容差与理由；重复运行结果完全一致（已验证）。
- Rust 侧两个测试把 golden 与契约绑在一起：`config_sha256` 必须等于登记的 fixture、抽头必须等于
  `geometry.target_taps`、`model.shape` 必须等于 `expected_weight_shapes()["fc.weight"]`、输出长度
  必须等于 hidden——契约或 fixture 漂移时测试会失败。

**selector 参考（第四步，已完成）**

`_score_edges()` 的语义同样从官方实现读出：`hidden_projection`（无 bias，5120→256）投影草稿 hidden，
每个候选带上自己的 unary logit，低秩项是**前驱 codebook 行（按投影 hidden 缩放）与后继 codebook 行
的双线性型**：

```
score[p, c] = unary_logits[c] + Σ_r predecessor_codebook[pred[p], r] * hidden[r] * successor_codebook[c, r]
pred[0] = anchor, pred[p] = candidate[p-1]
```

对应产物 `examples/qwen3.8-27b-dflash2/selector-golden.json`（导出器
`tools/fixtures/export-dflash2-selector-golden.py`，只按 id 行读 codebook）：记录 16×16 的分数矩阵、
rank/top-k、三个张量的形状与 dtype、输入公式、**故意相等的两个 unary logit**（让 tie-break 成为可查的
记录而不是脚注）、容差与理由、torch 版本；重复运行结果一致。Rust 侧一个测试把 rank/top-k、三个张量
形状（对照 `expected_weight_shapes()`）与分数矩阵维度绑到契约上。

**双抽头分组卷积参考（第五步，已完成）**

上一轮留白的问题在同一个文件里就有答案：vLLM 除 triton kernel 之外还有**非融合参考路径**
（`_grouped_conv` + `DFlashGroupedConv`），它把形状关系写全了，无需猜测：

- `base_kernel` 是 `[side, tap, channel]`（side 0 用于 `prepare`、side 1 用于 `finish`）；
- `kernel_projection` 输出 `[row, side, tap, group]`，`2 × taps × groups = 2×2×320 = 1280` ✓
  正是实测的 1280，`groups = hidden / conv_group_size = 5120/16 = 320`；
- 系数 = `base_kernel[side]`（按 channel）+ `coefficients[:, side]`（按 group）；输出 = 系数[0]×block，
  其后每个 tap 加 `系数[tap] × block[row - tap] × (row % block_size >= tap)`，末尾把 group 维摊回 channel。

于是用真实权重落了 `examples/qwen3.8-27b-dflash2/conv-golden.json`（导出器
`tools/fixtures/export-dflash2-conv-golden.py`）：**8 行（一个完整 block）× 5120**，side 1（finish），
确定性输入，值按 6 位有效数字记录（远小于 2e-2 容差），并记录 config 摘要、两个张量形状、block/group/
taps、公式、舍入与 torch 版本；重复运行一致。Rust 测试把它绑到契约：block/group/taps 必须等于配置、
两个张量名必须已在 weight inventory 中、`kernel_projection` 形状必须等于 `[2 × taps × groups, hidden]`、
输出维度必须等于 block × hidden。

**权重报价（第六步，已完成）**

卷积张量的形状确定之后，清单里不再有任何"只查名字"的项，于是权重报价可以完整给出并**被真实制品验证**：

- `expected_weight_shapes()` 现在是 `BTreeMap<String, Vec<usize>>`（去掉 Option）：`base_kernel` 为
  `[side, tap, channel]`、`kernel_projection` 为 `[2 × taps × groups, hidden]`，`groups` 取自
  `conv_group_size`；顺带把两个都叫 "taps" 的计数分开命名（`feature_taps` 是抽头数、`conv_taps` 是
  卷积核 tap 数）——它们在方案行文里同名，但在实现里毫无关系。
- 新增 `DFlash2Config::weights_bytes(dtype_bytes)`。
- 验收：`weights_bytes(2)` == **3,848,808,960** 字节，即 `model.safetensors` 里 81 个 BF16 张量的实际
  总和（同时得出 1,924,404,480 个参数，与研究记录里的 "1.924B" 吻合）。这一个数字同时校验了全部
  形状公式——`groups`、`conv_taps`、head 几何或 vocab 任一处错了都会对不上。
- 我另外用一份**独立的 Python 推导**（不共享 Rust 实现）逐张量对比了真实文件的 header：**81 个名字与
  形状全部精确一致**，无缺失、无多余、无形状差异。

**block mask 与滑动窗口（第七步，已完成）**

参考实现把 mask 交给 attention 后端，自己只解析每层的 `(sliding_window, causal)`：本 checkpoint 的
`layer_types` 全是 `sliding_attention`、`is_causal=false`，窗口取配置的 2048。方案对草稿 attention 的
要求更具体——**块内双向可见、跨块只看历史、且不越过滑动窗口**。这两条合起来就是本项目的规则，已在
`DraftGeometry` 上落地为 `attends(query, key)` 与 `block_of(position)`：

```
attends(q, k) = (q / block_size == k / block_size) || (k <= q && q - k < sliding_window)
```

这在文档里被标注为"本项目的规则陈述"而不是对参考实现的转写（参考只给了 `causal=false` 与窗口）。
`validate()` 已经保证 `block_size <= sliding_window`，因此块内可见性永远不需要再查窗口。四个测试覆盖：
块内双向、不看不见后续块、窗口边界（`window` 位置可见 `1` 而不可见 `0`；`window+1` 可见 `2`）、以及
块宽不超过窗口这一前提。

**特殊 token 与目标边界（第八步，已完成）**

配置里有两处容易放错位置的字段，实测确认了它们的位置：**mask token 在 `dflash_config` 内**
（`mask_token_id = 248070`），而 eos/pad 在顶层（都是 248044）。据此新增：

- `check_special_tokens()`：mask token 必须存在且落在自己的词表内（草稿用它填充"待提出"的槽位），
  eos/pad 也必须落在词表内，且不得与目标共享词嵌入。
- `check_target_tokens(eos, pad: Option<usize>)`：草稿的 eos/pad 必须与目标一致。目标 fixture
  （`examples/qwen3.8-27b`）的 `text_config.pad_token_id` 是 **null**，所以这一侧的比较是可选的、
  由目标决定——这本身也是配对信息的一部分，不是我们放宽了检查。
- 测试同时读两个登记 fixture：eos 一致（248044）、词表一致（248320），并验证 eos 不匹配时报错。

**selector 的 top-k 与路径选择（已确认语义，实现归属别的层）**

追到 `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` 后语义完整了，记在这里以免以后重新追：

1. **候选来自草稿自己的 LM head**：`compute_candidates()` 对草稿 hidden 取 top-k（k = `selector_top_k` = 16）
   得到 `candidate_ids` 与 `unary_logits`——**不是 selector 产生的**，selector 只负责给这些候选打分。
2. **路径依赖**：选择是一个自回归的 lattice walk——`previous` 初始为 0（anchor），每一步用
   `scores[step, previous, :]` 这一行，选中后 `previous = index` 交给下一步；被真正用到的行会写回
   `realized_scores`，于是"实际用了哪一行"是可复核的。
3. **排序与 tie break**：每步的抽取是 `gumbel_noised_argmax(scores_row, candidates, seed, position-1,
   temperature)`——噪声由 (seed, 位置) 决定，所以**同 seed 同位置可复现**；平局由该噪声决出，而不是
   取最小下标。`temperature = 0`（非概率模式）退化为贪心 argmax。
4. 这套 walk 是**提议策略**，按 I1 的分层属于 `crates/engine/workloads` 的 provider，不属于模型
   package，因此本轮只把语义确认并记录，没有把采样器塞进 `infer-models`。

**资源报价补齐到"草稿自身"能解释的部分（第九步）**

方案要求报价包含"权重、target/draft 状态、feature 历史、arena、graph 和回滚开销"，且**完整报价可解释**。
此前 `quote()` 只有每 token KV、窗口 KV 与每 tap 特征 ✗，权重与回滚都不在里面 ✗。现在报价移到
`DFlash2Config`（几何与张量形状都在那里），字段与口径如下（BF16，`dtype_bytes = 2`）：

| 字段 | 含义 | 值 |
|---|---|---|
| `weights_bytes` | 全部权重，与 weight inventory 同源 | 3,848,808,960 |
| `kv_bytes_per_token` | 5 层 × 8 KV head × 128 维 × (K+V) × 2B | 20,480 |
| `window_kv_bytes` | 每序列 2048 token 的窗口 | 41,943,040 |
| `conv_history_bytes_per_sequence` | 每层两个动态卷积各 `kernel−1` 槽 × 5120 通道 × 2B | 102,400 |
| `feature_bytes_per_token` | 每 tap 一条 hidden 特征（5 × 5120 × 2B） | 51,200 |
| `rollback_bytes_per_round` | 只接受第一个候选时该轮丢弃的草稿 KV（`block−1`） | 143,360 |

于是"一个序列的常驻草稿状态"= 窗口 + 卷积历史 = 42,045,440 字节（≈40 MB），而 feature 历史随已见 token
线性增长——这正是它按 token 记账的原因。**仍未计入的是 arena 与捕获图**，它们需要设备事实；这一条继续
写在方案里，不用估算糊过去。测试把每个字段与上述数字逐一钉住，含"权重报价与 inventory 同源"。

**明确还没有做的**：arena 与捕获图的报价（需要设备事实）；上述 walk 的实现（归 I1 的 provider 层，
等 SPI 契约落地）；tokenizer 文件的逐 token 对照（需要 tokenizer 制品，目前 fixture 只登记了配置）。

单测 19 个（`tests/unit/providers_dflash2.rs`），覆盖每个接受条件、对应反例与 golden 交叉核对，其中
7 个直接读登记的 fixture 与 golden。

### 已实施：fixture 的来源**可离线核验**（并如实登记两处缺口）

D1 要求参考制品能重新生成 ✓，而 fixture 目录里的 `source.json` 记录来源 ✓。此前只有**文字** ✓，
`config_sha256` 也**没人核对** ✗ —— 把 `examples/qwen3.8-27b/source.json` 拿来一看，它连**输入哈希都没有** ✗。

现在 `tools/check/golden-provenance.py` 也扫描 fixture 来源 ✓：必须有 `repository` 与 `source` ✓，必须有
**输入身份**（`config_sha256` **或** `revision` ✓），且当哈希存在时**必须与目录里 `config.json` 的实际
哈希一致** ✓（对不上会失败 ✓ —— 那份"说明"就不再指向 fixture 实际持有的文件 ✓）。实测补齐了
`examples/qwen3.8-27b/source.json` 的 `config_sha256` ✓（`191e0af2…` ✓）。

**两处缺口如实打印而不是编造** ✓：`qwen3.8-27b/source.json` **没有 license** ✓（其 HuggingFace 仓库离线
不可达 ✓ —— 编一个比说"没有"更糟 ✓，所以它只是提示 ✓ 不是硬性要求 ✓）；`qwen3.8-27b-dflash2/source.json`
**没有上游 revision** ✓（它是从本地下载复制而来 ✓，其身份由**已验证的 config 哈希**承担 ✓）。测试覆盖哈希不
匹配 ✓、缺输入身份 ✓、仅凭 revision 即可 ✓、缺来源 ✓、以及"缺 license 只是提示" ✓。

### 已实施：报价是"草稿的边际成本"，且**明确声明**了它报不出的部分

D1 的资源验收要求报价覆盖"**权重、target/draft 状态、feature 历史、arena、graph 和回滚开销**"✓。
`DraftQuote` 报得出四项 ✓（权重 ✓、attention 窗口即 draft 状态 ✓、KV+feature 历史 ✓、回滚缓冲 ✓），
而 **arena 与 graph 需要设备事实** ✗ —— 这一点此前只写在 `weights_bytes` 的文档注释里 ✗，读报价的人
**看不出报价是不完整的** ✗。

现在它由 API 说出来 ✓：`DraftCost` 增列 `Arena`/`Graph` ✓，`DraftQuote::deferred()` 返回
`[Arena, Graph]` ✓，`DraftAdmission` 带上 `deferred` 字段 ✓ —— 于是"准入结果"本身**不会**被误读成完整
答案 ✓。另外文档写明：**target 自身的状态与缓存不计入** ✓，因为无论有没有草稿，target 都要跑 ✓（所以这是
草稿的**边际**成本 ✓，不是整个 feature 的成本 ✓）。arena 与 graph 的正式报价仍**等设备窗口** ✓，
在此之前它们是被声明的缺口 ✓，而不是被忘记的字段 ✓。

## Target 特征捕获

模型 provider 声明抽头语义，backend 在 target 图中捕获指定位置。prefill 和 verification 必须使用同一特征定义。首先通过逐层独立参考确认 layer 编号、norm 前后位置、token 位置以及 fc 融合结果，再录制常驻图。

特征留在设备端，以有界 feature store 管理历史。保存范围由参考执行语义确定；draft 的 sliding window 不自动等于 raw target hidden 的保留长度。如果 fc 投影能够在特征捕获后等价计算，可存投影后的表示以减少长期缓存，但需先通过数值对照。

verification 产生的拒绝尾部特征只在事务内暂存，提交后立即失效。下一轮使用的特征必须覆盖 target 已物化前缀；校正或 bonus token 的特征要等该 token 被 target 消费后才能取得。

prefill chunk、prefix 命中和 provider 切换都需要检查特征覆盖范围。缓存缺失时选择可验证的重建路径，并将其成本计入 TTFT 或切换成本。

## 草稿模型图与 CUDA 算子

| 模块 | 拟议实现与验证重点 |
|---|---|
| 特征融合 | 绑定多层 target hidden 的拼接与 fc 投影；确认拼接顺序和归一化 |
| 双抽头动态卷积 | 分别实现 attention/MLP 的 kernel projection、分组与两抽头边界；对照独立公式 |
| 草稿 attention | 历史 prefix 与当前 block 使用正确 mask；允许块内双向可见，保持滑动窗口边界 |
| 草稿 KV | 与 target KV 分离；定义初始化、窗口淘汰、接受后的保留与拒绝后的修复 |
| candidate selector | 实现低秩打分、top-k 和路径选择；确认排序、tie break 与路径依赖 |
| target verification | 使用目标模型自身的因果 mask；草稿的非因果语义不传播到 target |

先实现可检查的算子组合，再由 backend 进行等价融合。selector 采用分块打分与 top-k，避免为方便集成而将完整词表矩阵传回 CPU。每个融合方案都与未融合参考对照，不能用最终接受率代替算子正确性。

[Infernix 调研](../../research/infernix-performance.md) 提供了两项资源候选：对草稿的大投影按角色量化，以及另存带 token-ID 映射的低精度 shortlist head。先建立完整 head 和 BF16 草稿参考，再分别评估量化、shortlist 及二者组合。target head 保留完整词表；DFlash2 候选 ID 映射回公共词表后进入 selector，随机模式的条件 `q` 从实际候选与分数计算。codec/kernel、额外 head、映射和 workspace 全部计入准入。

## Block 执行与状态结算

原生 block size 8 对应一个 seed 与最多 7 个候选，因此 target verification width 为 8。对外参数分别表达候选数量、draft block size 和 target verify width，避免与 MTP depth 混用。

执行顺序为准备已提交前缀的 target features、生成 block、运行 selector 得到一条候选链、target 因果验证、提交接受前缀、修复 draft KV 和特征历史。具体 KV 修复遵从 DFlash2 参考实现，由 provider 声明需求，不复用 MTP 的固定追赶代码。

显存受限时，可保留原生 8-token 草稿生成，只验证候选链的较短前缀。这会浪费部分草稿计算，但可以降低 target verification 的状态成本。直接缩短 draft block 会改变输入和 mask，必须独立验证模型语义与接受长度，不能仅修改 config 常量。

Infernix 的已实现路径接受 `K=1..15`，物理 block 为 `K+1`，说明该权重下长度可作为执行参数研究，而非固定张量维度。但块内非因果 attention 会使不同长度的候选同时改变，短块不保证等于长块的前缀。本项目先在原生 8-token block 建立参考，再做可变草稿宽度的独立对照；它与只验证短前缀是两个实验。

batching 按 target program、候选布局、验证宽度、状态 recipe 和 graph signature 分组。短链可以使用有明确有效长度的固定宽度图；padding lane 不得推进真实 KV、recurrent state 或 feature cursor。

## 实施依赖与验收

| 阶段 | 交付内容 | 进入下一阶段的条件 |
|---|---|---|
| 1 | 独立 package 解析、兼容校验和完整显存报价 | 不执行模型也能给出可解释的准入或拒绝结果 |
| 2 | 特征抽头、融合、卷积、attention 与 selector 的独立参考 | 各模块通过数值、mask、窗口与边界检查 |
| 3 | 草稿图与单请求线性链 | 与参考实现的中间结果、候选路径及状态对齐 |
| 4 | 共同 verifier 和事务结算 | 拒绝后下一步与普通 target decode 的状态、输出一致 |
| 5 | 池化、常驻图与设备端读写 | 多请求、prefix、取消、OOM 回退与混合负载通过验收 |
| 6 | 草稿量化和自适应验证长度 | 在完整成本和接受长度变化下取得服务收益 |

资源方案见 [CUDA 性能优化方案](../performance/cuda.md)。较大显存设备可先完成 BF16 正确性验证；目标小显存设备需要同时解决回滚快照和草稿权重开销。降低回滚成本并不等于草稿一定能装下。

package 校验、独立参考、draft 图/selector 模块与容量分析属于 D1，其中配置检查、参考制品和独立模块可以先准备；共同 verifier 与服务接入属于 D2，等待 I1 及对应资源能力。I1 契约稳定后，draft 模型模块的接线可与 R1 的 target Record/Fold 分别推进，最终在 D2 合流。若目标容量装不下 BF16 草稿，在 BF16/完整 head 参考可用后提前评估草稿量化或 shortlist，不必等到表中池化阶段失败后再处理；量化与 R1 各自验收，GPU 运行按资源窗口排队。

正确性矩阵覆盖首 token 拒绝、中间拒绝、全接受、EOS、输出上限、prefill chunk 尾部、2048 窗口边界、prefix 恢复与 provider 切换。greedy 的最终输出对照同一 target 权重和精度下的普通 decode；中间状态须在拒绝后继续运行多步验证。

性能比较执行 [性能基线方案](../performance/baseline.md) 的构建、配置、负载与统计要求，至少包括普通 decode、当前最佳 MTP 配置和 DFlash2。引用已验收的普通 decode/MTP baseline ID，另行登记 DFlash2 候选 profile；首次接入不预设 DFlash2 已有有效基线。报告真实启用的算法、图宽度、池化状态、显存与接受长度，不能把因容量失败而退回 serial 的结果标为 pooled DFlash2。
