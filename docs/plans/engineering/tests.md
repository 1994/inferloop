# 测试目录与执行入口统一方案

状态：**已交付**。测试已完成按 crate 迁移，`src/` 内测试体为 0（见"验收对照"），实现基于 `7c79d99` 起步。

**记录只描述已提交的语料（实测教训）**：`test-inventory.py --record` 曾经**从磁盘读**所有测试文件 ✓，于是在
一个共享检出里、当另一个会话的测试改动尚未提交时 ✗，记录会把这些工作树中的用例算进去 ✗ —— 实测出现过
"记录 475 条、实际提交 467 条" ✗。现在 `--record` 与判定都会**排除工作树与索引不一致的文件** ✓，并明确
打印被排除的文件清单 ✓（"stage or commit them before judging the corpus" ✓）；判定路径也不再把它报成
"语料减少" ✗，因为减少的是**这棵树**而不是这条分支 ✓。CI 在干净检出上运行 ✓，因此不受影响 ✓；本地脏树下，
该棘轮的自检会**显式跳过** ✓ 而不是给出假失败 ✗。新增用例覆盖 porcelain 解析（含改名 ✓）、干净树无排除 ✓、
以及被排除文件确实不进计数 ✓。

整体排期与任务状态见 [路线图](../README.md) 的 E1。本文维护本项设计与验收。

迁移前的用例身份、门控与消费者清单已随 E1 结束清理；当前用例语料棘轮由 `tools/check/test-inventory.py` 与 `tools/check/test-inventory.json` 执行。

测试正文统一放在所属 crate 的 `tests/` 中，按功能域组织文件；`src/` 保留产品实现和必要的测试模块挂载声明。保留 Rust 单元测试与集成测试的编译边界，减少重复链接、单用例专属目录和不明确的硬件依赖。

完整 CPU 测试推理执行器按 [删除方案](remove-test-backends.md) 迁移；构建配置、CI 和命令见 [构建与依赖方案](build-and-dependencies.md)。

## 当前分布

按 workspace member 的 Rust 文件中 `#[test]`、`#[tokio::test]` 声明粗统计，119 个文件包含 361 个入口。其中 80 个在普通 `src` 文件内，136 个在 `src` 内的专用 test/check/bench 文件，143 个在 crate 的 `tests/`，2 个在 examples。45 个文件只有一个声明。统计未按 cfg、ignore、平台或宏展开区分实际运行数量。

例如 [waiters 测试](../../../crates/engine/runtime/tests/unit/resource_waiters.rs)、[shards 测试](../../../crates/model/package/tests/unit/storage_shards_tests.rs)、[GPU 投影对照](../../../crates/backend/cuda/tests/unit/resident_prefill_gemm_bench_check.rs) 和 [checkpoint owner 测试](../../../crates/engine/runtime/tests/checkpoint_owner.rs) 分别使用不同路径和执行方式。问题是同类测试没有共同规则，不能仅按文件里的函数数量判断测试价值。

## 统一目录

以下为每个 crate 的拟议布局，仅创建实际需要的文件和目录：

```text
crate/
  Cargo.toml
  src/                         产品实现与测试模块挂载声明
  tests/
    integration.rs             公开接口测试的聚合入口
    gpu.rs                     真实设备公开场景的聚合入口 按需设置
    unit/
      sampling.rs              私有逻辑 按功能域命名
      state.rs
    integration/
      lifecycle.rs             公开接口场景
      http.rs
    device/
      attention.rs             设备场景
      recurrent.rs
    support/
      fixtures.rs              被复用的输入和 helper
  benches/                     独立性能 harness 按需设置
```

不为每个测试函数创建一个目录。没有必要细分的单用例并入所属功能域；不同资源要求、隔离需求或数值职责的测试可以独立成文件。空目录、只有转发意义的 `mod.rs` 层级和重复 helper 在迁移后删除。

## 保留私有性与 Cargo 编译边界

Rust 的集成测试是独立 crate，只能访问公开 API；被测模块的子单元测试可访问其私有实现。目录统一不能靠公开私有字段或增加产品 `test-support` feature 实现。[Cargo 测试目标](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#tests)、[Rust 私有性](https://doc.rust-lang.org/reference/visibility-and-privacy.html)。

私有单元测试由原所属模块用 `#[cfg(test)]` 与 `#[path]` 挂载 `tests/unit/` 文件。源码中只保留声明，用例正文及专用 helper 移出。挂载后仍编译在原模块的测试子模块中，不改变 visibility，不通过 `include!` 大量复制测试，也不增加公开测试 API。

公开行为使用 `tests/integration.rs` 聚合功能模块。显式配置 `[[test]]`，按迁移情况设置 `autotests = false`，避免 unit/support 文件误成为 Cargo target。每个 crate 原则上一个 host integration target；GPU 场景或确需进程隔离的场景另设具名 target。

多文件单独成为 integration target 会分别编译、链接和执行。聚合后可以用模块过滤定位用例，是否改善编译时间必须实测。[Cargo 集成测试组织](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#integration-tests)。

私有 GPU 单元测试仍挂载到被测模块，编译在 backend 的 lib test target。公开 GPU 场景进入具名 device target；runner 分别运行两个编译边界，不为了同一个入口而开放 kernel 内部 API。

## 按资源要求执行

正式 host/device Rust 测试、模型验收与性能场景执行 [共同 release 构建要求](../performance/baseline.md#release-与当前配置是必检条件)，Rust 测试使用 `--locked --release` 并保存构建身份。Python 测试不使用 Cargo profile，其调用的 Rust/GPU 程序遵守同一要求。性能采样、配置对齐和基线证据统一由 [性能基线方案](../performance/baseline.md) 维护。

| 类别 | 内容 | 执行与结果要求 |
|---|---|---|
| Host | 配置、图、采样、纯算法、状态机、网络协议 | 无 GPU、Toolkit 或外部模型即可运行 |
| Device | CUDA/Metal kernel、state、graph capture、小模型 | 显式选择真实平台，缺失硬件或必需 fixture 为失败 |
| Model acceptance | 正式 CLI/服务与实际 checkpoint | 接受明确模型、golden 和设备参数，报告覆盖配置 |
| Performance | 计时、分配、吞吐、尾延迟与比较门禁 | 独立 release harness，记录硬件和基线，避免普通测试并行干扰 |

GPU 私有测试按明确的 suite 清单或过滤规则选择；检查期望用例数量，不能把零用例或全 ignored 当作设备验收成功。`cuda` 是生产设备能力 feature，不新增 `test-cuda`、`test-backends` 等产品 feature。

模型路径、golden 和显存需求由验收命令声明。仅因发现环境变量才偶然运行的模型测试，迁移为显式 acceptance 场景。普通 host 测试可以不选设备 suite；已经选择该 suite 后，不允许缺少设备而 `return Ok(())`。

涉及同一 GPU、全局环境变量或进程级状态的测试使用确定的串行规则或进程隔离。网络测试采用随机可用端口、有界等待和可靠清理；不依赖另一测试的执行顺序。

## Examples 与性能 harness

`examples/` 只保留演示公开 API 的程序。`*-check`、`*-probe` 中仅用于断言正确性的内容迁移为 device/model 场景；用于计时的内容进入 `benches/` 或已有独立性能工具。examples 不携带只有特定 gate 才会触发的测试正文。

可以给 GPU backend 保留一个按场景选择的验收 driver，统一参数与报告，但它只调度正式模型/设备执行。它不能加载新的 CPU 推理实现，也不能取代具有明确断言和非零失败退出的 tests。

benchmark、可执行 fixture 生成器和验证脚本都进入统一命令清单，避免同一个场景既在 `cargo test` 中计时，又在 example 和脚本中重复维护。

## Python 工具测试

Python 测试放在所属工具的 `tests/` 下，例如 `tools/package/tests/`、`tools/check/tests/`、`tools/bench/tests/`。保留标准 `unittest` 入口，将工具内部混杂的 `test_*.py` 正文移出；测试按照 package、布局规则、报告比较等功能域组织。

收集入口覆盖全部已登记模块，不继续在 `gate.sh` 中逐个写死文件名模式。目前 `tools/bench/test_serve_compare.py` 已存在，而 rust gate 只显式收集 `test_compare_results.py`；迁移后应通过 discovery 和收集数量检查避免这类遗漏。

Python helper 的 import 使用明确工具包或统一测试 bootstrap，不靠每个测试修改不同的 `sys.path`。生成器依赖按所属工具环境安装，普通测试不自动安装 PyTorch、Candle 或下载模型。

## Helper 复用与迁移门禁

优先将 helper 放在所属 crate 的 `tests/support/`。多个 crate 确实复用的少量 fixture/脚本/断言可共享测试源文件，并只从测试 target 或开发工具引用；不为少量 helper 新增 Cargo crate，不构建完整模型执行器。

新增规则检查测试正文是否位于 `tests/`，挂载路径是否唯一且存在，Cargo target 是否登记，support 是否被错误收集，设备/性能场景是否具有明确 runner。规则适用于生产、target-specific 和 build-script 测试；build contract 的用例移到其工具 tests 中，保留必要挂载。

迁移先记录现有用例身份、断言目标、平台、ignore 原因、fixture 和资源需求。按 crate 搬移后比较实际收集清单，将重命名、合并和转为设备验收的用例显式对应；文件少了不能成为删掉断言的理由。

先迁移 foundation、model、scheduler 与 workloads，再迁移 runtime/frontdoor/agent，最后迁移 GPU 私有测试、examples 与性能场景。每批只调整目录和收集，不同时改数值公式或产品行为。

完成条件是：源码不再包含用例正文；host 测试可一次收集运行；GPU/模型/性能测试具有明确入口；私有性与现有断言保持；收集变化有去向；完整 CPU 测试推理 backend 已按删除方案移除。
