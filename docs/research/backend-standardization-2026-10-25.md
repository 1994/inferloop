# Backend 标准化与接入 SOP 研究

日期：2026-10-25（Asia/Shanghai）。源码基线：`2bec30a`（`main`；代码与 `0babb94` 相同，差异仅 README 文档提交）。
范围：主机侧（host-side）接入成本、重复工作与可共享契约，不评估新设备的 kernel 性能。
本文只做静态盘点与方案比较，不修改代码、不改变排期，也未运行 GPU 验收。

## 结论

- **CUDA 的 45k 行不是新 backend 的接入基线。** 其中约 70% 是常驻图、slot、pooled、MTP 与自动调优，
  属于性能层；设备与主机层合计约 18%。Metal 用 3.3k 行源码就跑通了仓库内的 hybrid 夹具，
  说明 Tier-0 功能后端的规模可以低一个数量级。
- **真正的重复不在 kernel，而在主机侧：** kernel 能力清单、权重装载、状态布局与报价、prefix/checkpoint、
  CLI 枚举转发、测试场景。这六项每接一个 backend 就要重写一遍，且没有一致性检查。
- **共享原材料大多已经在仓库里，只是没有被强制使用：** `StateRecipe`、`load_weights` / `WeightTarget`、
  `KvCacheManager`、`reference-ops` 都是现成的，但 CUDA 只用了其中一部分，Metal 用了另一部分。
- **建议的 SOP 顺序：先“数据契约 + 一致性套件”，后“主机骨架”，最后才考虑通用执行器。**
  直接抽通用 kernel ABI 或通用执行框架，成本和风险都高于收益。
- **目标：** 新 Tier-0 backend 主机代码 ≤ 3,000 行（不含设备 kernel），2–4 周做到功能可用，
  `engine` / `runtime` / `compiler` 零改动；模型接入在只使用现有算子时不需要 backend 改动。

## 1. 规模对照：CUDA 是性能基线，Metal 是接入基线

| crate | `src` 行数 | `tests` 行数 | 设备源码 |
|---|---:|---:|---|
| `infer-backend-cuda` | 45,330 | 23,605 | 95 个 kernel 文件，约 4.4 MiB |
| `infer-backend-metal` | 3,304 | 1,660 | 1 个 MSL 文件，150 行、13 个 kernel |
| `infer-kernel-api` | 227 | 187 | — |

CUDA 源码按目录拆分：

| 模块 | 行数 | 占 `src` | 性质 |
|---|---:|---:|---|
| `resident/` | 23,165 | 51.1% | 性能层：常驻图、slot、pooled、MTP、recurrent |
| `tuning/` | 8,530 | 18.8% | 性能层：GEMM/attention 自动调优 |
| `executor/` | 3,509 | 7.7% | 主机层：`BackendProvider`、ticket、状态、checkpoint |
| `loading/` | 2,726 | 6.0% | 主机层：权重绑定、预算与报价（CUDA 自写） |
| `device/` | 2,086 | 4.6% | 设备层：上下文、能力、分配、FFI |
| `vision/` | 1,876 | 4.1% | 模型/特性：视觉塔 |
| `attention/` | 553 | 1.2% | 算子：dense SDPA provider |
| `mlp/` | 499 | 1.1% | 算子：融合 MLP |
| 顶层文件 | 2,386 | 5.3% | 注册表、常量、量化、查询等 |

由此得到三个判断：

1. **性能层约 70%。** 常驻图与调优是 CUDA 的竞争力，不是接入必需项。新 backend 首版不应复制。
2. **设备与主机层约 18%。** 这是任何 backend 都绕不开的部分，其中又有一半是可以共享的数据与生命周期。
3. **模型/算子特性约 7%。** 视觉、融合 MLP、dense attention 属于可选能力，可按 Tier 推进。

Metal 的 3.3k 行里，`executor/` 占 2.7k，`device.rs` 只有 438 行；它没有常驻图、没有调优、
没有视觉与 MTP，却已经通过 golden、forward、paging、retention、runtime 五组测试。
**Metal 是“功能完整、性能普通”的 Tier-0 参照实现；CUDA 是 Tier-1 性能参照实现。**
当前文档（[Backend 能力与扩展](../architecture/backends.md)、[Backend 分组](../../crates/backend/README.md)）
只描述了这两条路径的差异，没有把“新 backend 应该以哪条为参照”写成规则。

## 2. 新 backend 今天要写什么

| 接入面 | 当前成本 | 说明 |
|---|---|---|
| IR 封闭枚举 | 7 个文件出现 `DeviceBackend`，19 个文件出现 `BackendKind` | 编译期穷尽匹配，新增 variant 必须逐处处理；这是有意的，不是运行时插件 |
| CLI 分发 | [`selected.rs`](../../crates/service/cli/src/backend/selected.rs) 437 行、15 处 `match self` | `BackendProvider` 47 个方法逐个转发；新增 backend 要加 47 个分支 |
| CLI adapter | CUDA 431 行，Metal 51 行 | 默认值、利用率、覆盖检查、profile 选择 |
| capability 查询 | 设备层约 300–500 行 | `DeviceCapabilities` 与厂商要求结构 |
| 权重装载 | Metal 378 行（用共享 loader），CUDA 2,726 行（自写） | 新 backend 应走前者 |
| 状态布局与报价 | Metal 用 `StateRecipe`，CUDA 自写 `loading/budget.rs` 820 行 | 新 backend 应走前者 |
| 图执行 | Metal `forward.rs` 390 行解释 dense 算子（MoE 未注册）；CUDA 常驻图 | Tier-0 可照抄 Metal 的解释器结构 |
| 设备 kernel | 13–17 个；CUDA 为性能展开到 95 个文件 | 这是唯一无法标准化的部分，但可以先写朴素版本 |
| 测试 | Metal 1.4k 行，CUDA 23.6k 行 | 五类场景高度同构，适合共享 |
| 门禁与文档 | `tools/check/layout.py` 的 `BACKENDS` 集合、README、backends.md | 纯清单工作，可模板化 |

`BackendProvider` 共 47 个方法，只有 6 个没有默认实现：`identity`、`capabilities`、
`validate_program`、`execution_graph`、`submit`、`poll`。这意味着**编译一个空壳 backend 很容易，
但默认值会让 prefix 复用、容量报价、checkpoint 静默退化**；当前没有测试能区分
“显式不支持”与“忘了实现”。这是第 4 节一致性套件要解决的第一类问题。

## 3. 已有但未被强制使用的共享资产

| 资产 | 位置 | 现状 | 新 backend 能否直接用 |
|---|---|---|---|
| 图语义与 lowering | [`TensorOp`](../../crates/foundation/ir/src/dataflow.rs)、[`decoder::lower`](../../crates/model/recipes/src/decoder.rs) | 两条路径最终都来自同一 lowering；Metal 在 backend 内调用，CUDA 使用 package 已生成的图 | 能。模型 provider 产出的图与设备无关 |
| 权重装载流水线 | [`load_weights` / `WeightTarget`](../../crates/model/package/src/storage/loader.rs) | 只有 Metal 使用；CUDA 自写 2.7k 行 | 能。backend 只需实现 allocate/upload |
| 状态布局与报价 | [`StateRecipe`](../../crates/foundation/ir/src/state_recipe/mod.rs) | 只有 Metal 使用；CUDA 自写预算 | 能。backend 按 region 分配，报价由共享代码给出 |
| KV 页与 prefix | [`KvCacheManager`](../../crates/engine/state/src/kv.rs) | Metal 用共享 lease；CUDA 用私有 snapshot（`executor/prefix.rs`） | 能。Tier-0 应默认用共享 manager；snapshot 属性能选择 |
| 每算子数值参考 | [`reference-ops`](../../crates/testing/cpu/reference-ops/src/kernels.rs) | 覆盖全部 `TensorOp`，但只被 CUDA 使用 | 能。是跨 backend 一致性套件的现成 oracle |
| kernel 注册 | [`KernelRegistry`](../../crates/backend/kernel-api/src/registry.rs) | 两个 backend 都用，但注册项全是 `f32 / usize::MAX / 0 / 1`，选择退化为白名单 | 能注册，但无法表达真实适用域 |
| 协议级假 backend | [`runtime/tests/support`](../../crates/engine/runtime/tests/support/mod.rs) | 432 行，只模拟提交/轮询协议 | 能复用协议场景，但没有数值执行 |

结论：**标准化不是从零抽象，而是把 Metal 已经验证过的路径固化为“默认路径”，
把 CUDA 的偏离标注为“性能选择”。** 共享代码的存在不是问题；问题是新 backend 没有理由
必须使用它们，也没有测试在偏离时报警。

## 4. 重复工作分类

| 类别 | 内容 | 处理方式 |
|---|---|---|
| 设备必需 | 上下文、能力查询、分配/上传、提交/fence、ticket、错误映射 | 保留在 backend；提供模板与 conformance 契约 |
| 主机可共享 | 权重装载、状态布局与报价、prefix、checkpoint 编解码、CLI 枚举转发、kernel 清单 | 抽成数据契约或共享骨架，新 backend 只填设备操作 |
| 性能可选 | 常驻图、融合、自动调优、paged/量化 KV、MTP、视觉、BF16/FP16 profile | 不进入 Tier-0；通过 capability 显式声明 |
| 明确不做 | 通用 kernel ABI / 参数绑定、动态加载、CPU 执行 backend | 前者会把设备类型漏进共享层；后者已在 E1 删除，[不再恢复](../plans/engineering/remove-test-backends.md) |

需要强调：**“新模型不需要改 backend”只在模型只使用现有 `TensorOp` 与 Tier-0 形状时成立。**
新增算子、新的 KV 布局或新精度路径仍然是一次契约变更，必须同时有参考实现与一致性用例；
这不是 SOP 失败，而是防止“隐式 fallback”的必要成本。

## 5. 建议的 SOP

### 5.1 分层定义

| 层级 | 要求 | 参照实现 | 允许缺失 |
|---|---|---|---|
| Tier-0 功能后端 | F32 dense 图、dense 算子子集、paged KV、prefix、prefill/decode、checkpoint、CLI 可选 | Metal | MoE、融合、MTP、视觉、量化、常驻图、调优 |
| Tier-1 性能后端 | 在 Tier-0 之上增加融合、常驻图、调优、量化 KV/权重、BF16/FP16 profile | CUDA | 仍须通过 Tier-0 一致性套件 |
| Tier-2 模型特性 | MoE、视觉塔、MTP/draft、模型专属融合 | CUDA | 通过 capability 显式拒绝，不能静默降级 |

Tier-0 的“允许缺失”必须在 `DeviceCapabilities` 与命令入口显式报 `Unsupported`，
并通过一致性套件断言错误路径，而不是返回默认值。

### 5.2 五件标准件

**S1. Kernel 能力清单（声明式）**

把每个 backend 的 `KernelProvider` 从“算子列表 + 常量”改成带适用域的清单：
compute/storage dtype、head_dim 上下界、mask 种类、KV 布局（dense/paged、F32/FP16/BF16/FP8）、
对齐要求、元素数上限。backend 只填一张表，选择逻辑留在 `kernel-api`。
新 backend 的清单约 60–100 行；`compile` 成为真正的准入点，而不是到加载期才失败。

**S2. 状态与装载数据契约**

- Tier-0 backend 必须用 `StateRecipe::compile/layout` 计算容量与报价，只实现
  “按 region class 分配/释放”；不再自写字节公式。
- 必须用 `load_weights` + `WeightTarget` 装载权重，只实现 `allocate` / `upload`；
  量化、专家、视觉作为可选扩展，不进入首版。
- 这两项能把 CUDA 式的 3.5k 行主机代码从新 backend 中移除，且把“报价与真实分配一致”
  变成共享测试。

**S3. 一致性套件**

新增 `conformance` 模块（建议先放在 [`reference-ops`](../../crates/testing/cpu/reference-ops/src/lib.rs) 内，
避免新增 crate 与修改布局规则；规模变大后再拆 `crates/testing/backend-conformance`）。
套件只依赖 `BackendProvider`、`StateRecipe` 与 reference oracle，场景由清单驱动：

1. 加载夹具 → 编译 → 预约 → prefill/decode → 输出与参考一致；
2. 容量报价 ≥ 实际分配，释放后回到基线；
3. KV 页增长、prefix 命中/失效、COW 隔离；
4. 取消/排空后无残留 ticket 与状态；
5. checkpoint 捕获/恢复后输出不变；
6. 每个已注册 kernel id 可解析，且清单声明的能力与真实接受/拒绝一致；
7. 未实现能力必须返回 `Unsupported`，不能静默返回默认值；
8. 多请求批处理与单请求结果一致；
9. 重复提交/并发 owner 的错误路径；
10. 可选的控制面分配预算（只对声明支持的 backend）。

场景分 `required` 与 `optional(capability)` 两级；设备侧用 backend 自己提供的 factory 启动，
host-only 部分用协议 mock。**不恢复 CPU 执行 backend**：mock 只验证契约，不冒充设备。

**S4. CLI 枚举转发宏**

把 `SelectedBackend` / `SelectedTicket` 与 47 个转发方法收进一个 `backend_enum!` 声明，
新 backend 只加一个 variant 和一个 ≤100 行的 adapter。目标是把
[`selected.rs`](../../crates/service/cli/src/backend/selected.rs) 从 437 行降到 ~80 行，
且新增 backend 不再需要逐方法补 `match`。

**S5. SOP 文档、模板与能力矩阵**

- 新增 `docs/guides/adding-a-backend.md`：Tier 定义、checklist、命令、DoD、常见错误；
  替换 [backends.md](../architecture/backends.md) 里四步式的简述。
- `tools/check/layout.py` 的 `BACKENDS` 集合、README 支持表、backends.md 表格改为同一份
  能力矩阵生成或至少由同一清单驱动，避免“代码支持但文档没写”。
- 提供最小 backend 模板（目录、manifest、CLI adapter、conformance 入口），
  新 backend 从模板开始而不是从 CUDA 复制。

### 5.3 主机骨架（可选，最后做）

若完成 S1–S4 后，新 backend 的主机代码仍超过 1.5k 行，再考虑 `infer-backend-kit`：
把 Metal 的 `ForwardNode` 准备、KV append、readout、scratch lifetime、checkpoint 编解码
抽成 host-only 骨架，泛型参数为设备 buffer 句柄。决策门槛：

- Metal 迁移后行为与性能不回退（五组现有测试通过）；
- 热路径不增加动态分派或每步分配；
- 新 backend 的主机代码再降 30% 以上。

达不到任一条就停在 S1–S4，不做通用执行器。**不建议**把 CUDA 常驻图纳入该骨架。

## 6. 分阶段计划与验收

| 阶段 | 交付 | 验收 | 主要风险 |
|---|---|---|---|
| P0（本 PR） | 本文：盘点、分层、SOP 与待决问题 | 文档链接与布局门禁通过；维护者确认 Tier 划分 | 结论被误读为“立即重构 CUDA” |
| P1 | S3 一致性套件骨架 + S5 SOP 文档；不改变生产行为 | mock 层在 CI 运行；Metal 在设备窗口通过 required 场景；CUDA 通过 host 可检查组，缺口逐条登记 | 设备依赖导致 CI 无法覆盖；套件变成第二套测试 |
| P2 | S1 kernel 清单 + S2 `StateRecipe`/`load_weights` 强制；S4 CLI 宏 | 新 backend 只填表即可注册；报价与分配一致性由共享测试覆盖；CUDA 行为与报价不变 | StateRecipe 的内存分类不够用；CUDA 报价无法等价 |
| P3 | 用目标设备做一次真实接入试点 | Tier-0 主机代码 ≤3,000 行、2–4 周、`engine`/`runtime`/`compiler` 零改动；模型夹具通过 | 目标设备工具链不成熟；kernel 工作量被低估 |
| P4（可选） | S5 的主机骨架与 CUDA 收敛 | Metal 迁移无回退；CUDA 只收敛清单/报价/套件三处 | 抽象泄漏、热路径回退 |

P2 不要求 CUDA 改用 `StateRecipe`：常驻图与 pooled 程序有独立的报价模型，
强行统一可能改变准入行为。要求是“**新 backend 默认使用，CUDA 记录偏离并由一致性测试证明差异**”。

## 7. 量化目标（假设，供 P3 验证）

- 新 Tier-0 backend：主机代码 ≤3,000 行（不含设备 kernel），其中 backend 自有逻辑 ≤1,500 行；
- 设备 kernel：13–17 个朴素实现起步，性能优化不进入首版验收；
- 时间：有可用 kernel 工具链时 2–4 周；若厂商库已覆盖 GEMM/attention，可压缩到 5–10 天；
- 改动面：`engine`/`runtime`/`compiler` 0 处；CLI 1 个 variant + ≤100 行 adapter；
  IR 枚举按 checklist 补齐；文档与 `layout.py` 清单各 1 处；
- 测试：五类共享场景不重复实现；backend 只保留 kernel 数值、设备怪异行为与性能测试；
- 模型接入：只使用 Tier-0 算子与形状的新模型，不需要改 backend。

这些是**验收假设**，不是当前实测；P3 需要用真实试点回填。

## 8. 风险与非目标

- **抽象泄漏。** 设备 buffer 生命周期、统一内存、异步队列差异很大；主机骨架必须保持 host-only，
  不把 fence/stream 语义抽象成最小公分母。达不到门槛就不做。
- **`StateRecipe` 的内存分类只有 DevicePrivate / HostMirror / KvBlock。** 若新设备有 scratchpad、
  pinned 或分层显存，需要新增 region class；这是加法，但会触及共享布局数学与所有 backend 的分配器。
- **一致性套件的容差与 flaky。** 不同设备/精度的数值容差必须按 dtype 声明，
  skip 必须显式标记为 `optional`，不能把“没跑”算作通过。
- **Metal 重构风险。** 它是唯一 macOS 路径；任何骨架迁移都要保留现有五组测试作为行为基线。
- **CUDA 报价不可强行统一。** 常驻图、pooled slot、private verification 的成本模型与 `StateRecipe`
  不同；统一的前提是逐项证明等价，否则保留差异。
- **非目标：** 通用 kernel ABI、动态插件加载、CPU 执行 backend、跨 backend checkpoint、
  用一致性套件替代性能门禁。

## 9. 待决问题

1. 目标 backend 的设备与内存模型是什么（独显 / 统一内存 / 多级显存）？这决定 `StateRecipe`
   是否需要扩展 region class，以及主机骨架的 buffer 抽象。
2. 首版是否要求 checkpoint、prefix、连续批处理？还是允许显式 `Unsupported` 并进入 Tier-1？
3. 是否接受一致性套件先放进 `reference-ops`（避免新 crate 与布局规则变更）？
4. 是否接受新增 `infer-backend-kit` 作为 P4 选项（而不是现在就建）？
5. CUDA 是否接受“只收敛清单、报价对照与套件”三处，常驻图路径保持不动？
6. 新 backend 的首个目标模型族是什么？这决定 13–17 个 kernel 中哪些必须先做。

## 附 A：Tier-0 接入 checklist（草案，P1 移入 guide）

```text
[ ] 能力与枚举：BackendKind / DeviceBackend / DeviceCapabilities / BackendRequirements
[ ] tools/check/layout.py 的 BACKENDS、README 支持表、backends.md
[ ] KernelProvider 清单：目标模型用到的 TensorOp 子集（dtype、head_dim、KV 布局、mask）
[ ] 权重：ModelPackage → load_weights → WeightTarget（allocate/upload）
[ ] 状态：StateRecipe::compile/layout → 按 region class 分配；预约/释放/重置
[ ] BackendProvider：identity、capabilities、validate_program、execution_graph、submit、poll
[ ] 图执行：目标算子子集的 encode；KV append/readout；prefill/decode 分块
[ ] prefix：KvCacheManager 命中/复用；无共享 manager 时显式 Unsupported
[ ] checkpoint：capture/restore/validate（或显式 Unsupported 并登记）
[ ] CLI adapter + SelectedBackend variant
[ ] conformance required 场景通过；optional 场景逐条标记
[ ] 至少一个仓库夹具的 golden 输出通过；无 CPU fallback
[ ] 文档：支持矩阵、已知限制、性能未验收声明
```

建议的本地命令（P1 落地后补全实际 target）：

```sh
cargo fmt --all --check
make check-rust
python3 tools/check/layout.py
# 设备套件按 backend 提供 make target，例如 make test-<backend>
```

## 附 B：盘点方法

- 行数：`find crates/backend/<name>/src -name '*.rs' | xargs cat | wc -l`；测试同路径换 `tests`。
- 共享资产使用情况：`grep -rn "StateRecipe\|load_weights\|KvCacheManager\|reference_operation" crates`。
- 接入面：`grep -rn "BackendKind::\|DeviceBackend::" crates`、`wc -l crates/service/cli/src/backend/selected.rs`。
- 本文所有数字基于 `2bec30a` 的静态源码；未执行设备测试，未把编译通过当作功能验收。
