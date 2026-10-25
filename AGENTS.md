# Agent 工作指南

本文件是 inferloop 的统一 agent 入口，适用于整个仓库。修改或审查 Rust、Cargo、测试与性能代码前先读本文件；更具体的目录指引若存在，也要一并阅读。无需安装个人插件。

Rust 实践提炼自 [BeautifulRust](https://github.com/morganlinton/BeautifulRust/tree/4f1c0d1602bf75c9718ccea42879defb8bfe89f4)，按本项目的推理引擎约束做了取舍。**项目的架构契约、Cargo 配置和门禁优先于上游通用建议**。这里只维护 agent 工作规则，使用与架构细节仍查现有文档，不复制另一套。

## 开始工作

- 先检查工作区改动与目标模块，保留其他任务尚未提交的内容；只修改当前任务需要的文件。
- 确认代码属于 foundation、backend、model、engine、diagnostics、service 或 testing 哪一层，再决定落点。边界见[代码布局](docs/architecture/layout.md)。
- 新功能先明确输入、输出、错误、资源所有权与验收方式；重构保留公开行为、错误语义和输出格式。发现可疑旧行为先核实，不夹带无关修正。
- 优先复用已有类型、错误、工具和测试入口；不为未来可能出现的需求增加 trait、builder、crate 或依赖。

## Rust 编写与审查

### 所有权与资源

- 读取参数优先用 `&str`、`&[T]`、`&T`，修改用 `&mut T`，只有需要保存或消费时才接收所有权。避免 `&String`、`&Vec<T>`、`&Box<T>`。
- 遇到借用冲突，先缩短借用、拆表达式、拆字段或使用 `mem::take` / `Option::take`；不要只为通过编译添加深拷贝。确有快照或共享所有权需求时，明确复制语义和成本。
- 优先单一所有者；`Arc`、锁和内部可变性必须对应真实的共享需求。设备 buffer、lease、fence、KV 页的释放时点要跟随实际使用完成，不能只跟随主机函数返回。
- 通过 RAII 保证提前返回与取消时清理资源；guard 绑定到具名变量，避免 `let _ = acquire_guard()` 立即释放返回值。可失败的关闭操作提供显式返回错误的接口，`Drop` 不 panic。

### 类型与 API

- 用 newtype 区分 ID、单位和语义，用 enum 表示模式与状态，用可失败构造函数维护不变量。模式选择参数避免含义不清的 `bool`；不要为风格统一破坏现有 API。
- 保持最窄可见性；公开字段意味着调用方可以绕过构造约束。不要为测试开放产品内部 API。
- 优先具体类型；需要替换实现或泛型算法时才引入 trait。泛型与 `dyn Trait` 按分发、异构集合、编译成本和现有 SPI 选择，不机械改写。
- 适用时派生 `Debug`、`Default`、比较和转换 trait；秘密或大块数据的调试输出要控制。手写排序时保证与相等关系一致，实现 `Hash` 时也须满足相等值同哈希。
- 命名遵循标准库的 `as_` / `to_` / `into_` 语义；显式导出 API，不增加 glob re-export 或无必要的 prelude。兼容性变化说明修改前后行为和迁移影响。

### 错误与日常写法

- 正常缺失用 `Option`，需要解释失败用 `Result`；复用项目已有错误类型与错误码，不因上游偏好引入 `anyhow` / `eyre` 或重做公共错误协议。
- 生产代码禁止以 `unwrap`、`expect`、`panic`、`todo`、`unimplemented`、`dbg` 或忽略 `Result` 处理问题；`main` 也不例外。不变量失败走现有错误/故障处理路径。已有经门禁认可的局部不变量例外，先审查其安全理由与回收语义，不做机械替换。
- 锁中毒不可用 `expect` 绕过，也不可未经证明通过 `into_inner` 继续使用可能损坏的状态；按所属组件的错误、隔离和退出协议处理。
- 短且清楚的转换适合迭代器与 combinator；多步骤、提前退出或涉及状态的逻辑保留清晰循环，不为“函数式”制造复杂链。
- 项路径用 `use` 引入；三段及以上的项路径不直接写在表达式或签名中，遵循 `clippy::absolute_paths`。导出按项列出。
- 数值常量遵循项目门禁：生产代码中的非豁免数字提取为带 `///` 说明的具名常量。不要靠调高长度、复杂度阈值或放宽 lint 过关。
- 运行日志使用现有结构化 `tracing`；CLI 的正式 stdout 输出按协议保留。避免在内层循环格式化字符串或逐 token 打日志。

### 并发、异步与 unsafe

- 保持现有 actor / 单所有者边界；跨线程类型与 future 按实际契约满足 `Send` / `Sync`，不能靠 `unsafe impl` 消除编译错误。
- 锁作用域保持短小，不跨 `.await`，不持锁调用外部回调；通道和队列必须考虑容量、背压、取消与关闭。
- 长 CPU 工作使用现有工作线程或有界阻塞执行机制，不能阻塞异步服务线程；不机械插入固定频率的 yield，也不任意新建 runtime 或无界任务。
- `unsafe_code` 默认禁止；沿用已隔离的 OS/设备 FFI 边界。新增例外必须有必要性、安全不变量和最小作用域，不能只凭“可能更快”开放。
- 每个 unsafe block / impl 写明 `SAFETY:`，unsafe API 写明调用方义务；核实指针范围、对齐、初始化、别名、生命周期、线程/stream 所属与异步完成时点。
- FFI 明确 ABI、分配/释放所属方与错误转换；不能让 unwind 穿过不允许它的边界。Miri 适用于可运行的主机安全抽象，不能代替设备侧验证。

## 项目适配：这些约束不能套用通用建议

| 主题 | 本项目规则 |
|---|---|
| 目录 | 保留 `crates/<职责>/<crate>` 分组；`lib.rs` 是小型 facade，实现放职责模块。不改为上游建议的扁平目录。 |
| 架构 | runtime 通过合约驱动 backend，不依赖具体 CUDA/Metal 实现；backend 不依赖 service。模型拓扑归 provider/recipes，compiler 校验与降低图，scheduler 管策略，state 管逻辑状态所有权。 |
| 工具链与依赖 | 按 `rust-toolchain.toml`、workspace 的 edition/MSRV、依赖版本与 lint 配置执行。保留有意的精确版本和 `Cargo.lock`；不因通用指南推荐版本范围而改动。 |
| Feature 与构建 | 使用项目门禁的 feature/target 矩阵；不要直接照抄上游 `--all-features`。CUDA 及其独立设备构建 workspace 有工具链与依赖隔离要求。 |
| Lint 例外 | 优先修正代码；确有局部例外时写明理由并满足门禁已有规则，不能新增大范围 `allow` / `expect` 或调整阈值逃避检查。 |
| 测试布局 | 正文放所属 crate 的 `tests/`，私有单测沿用 `#[cfg(test)]` / `#[path]` 挂载；不向 `src/` 增加测试正文，不增加产品测试 feature。 |
| CPU 与 GPU | CPU 参考实现只作独立数值对照；CPU 通过或 CUDA 无 feature 编译通过不代表设备支持。选定设备验收后，缺硬件、fixture、零用例或全 ignored 不能冒充成功。 |

## 测试、文档与性能

- Bug 修复先用失败用例复现；行为变化覆盖边界输入、错误、取消和资源回收。测可观察行为、性质与状态转换，不写复述常量或复制实现分支的测试。
- 测试放到既有 suite 并确认被收集、执行；不把“编译成功”当成“测试通过”。涉及时间、网络和随机性的用例要有确定输入、有界等待和可靠清理。
- 公共 API 文档说明签名不能表达的约束、所有权、失败与副作用；按需提供 `# Errors`、`# Safety` 和可测试的 `?` 示例。注释解释原因，不记会话过程。
- 推理热路径先测量再优化；复用 buffer、预分配、批处理，关注每 tick/token 的分配、锁、拷贝和同步成本。不机械调用 `shrink_to_fit`、更换 allocator/hasher 或增加缓存。
- 性能结论必须来自相同条件下的 release 测量，记录源码/构建身份、模型、输入、设备、精度、warmup、样本与统计；按[基线契约](docs/plans/performance/baseline.md)验收，不把单算子收益直接等同服务收益。
- 数值优化同时检查独立参考、容差、有限值、完整输出和资源生命周期；不能放宽精度来掩盖回归，也不能把单机结果外推到所有 GPU。

## 交付前

- Rust 改动至少执行 `cargo fmt --all --check` 与 `make check-rust`；按改动追加工具、安全、MSRV、Linux、Metal 或 CUDA 门禁。具体入口以 [Makefile](Makefile)、[统一门禁](tools/check/gate.sh)与[质量门禁说明](docs/guides/quality-gates.md)为准。
- 仅修改本文或文档时检查本地链接、`git diff --check`，并运行 `python3 tools/check/layout.py`；不为纯文档改动启动整套硬件验收。
- 无法运行的检查明确写出原因；分别报告已运行、仅编译、未运行和设备验收范围，不声明超出证据的支持或性能收益。
- 核对 diff 的任务范围、兼容性与依赖变化；保留 lockfile、微型 fixture 和可复算的正式证据，不提交模型、凭证或临时运行产物。

## 来源与维护

提炼基点：BeautifulRust commit [`4f1c0d1602bf75c9718ccea42879defb8bfe89f4`](https://github.com/morganlinton/BeautifulRust/commit/4f1c0d1602bf75c9718ccea42879defb8bfe89f4)，阅读日期 2026-10-24（America/Los_Angeles）。依据其 `rust-best-practices` skill 与五份摘要整理，未导入插件、eval 或自动安装脚本。

需要深入判断时查原始来源：[Google Rust Style Guide](https://google.github.io/styleguide/rustguide.html)、[Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)、[Effective Rust](https://www.lurklurk.org/effective-rust/)、[Rust Design Patterns](https://rust-unofficial.github.io/patterns/)、[Microsoft Pragmatic Rust Guidelines](https://microsoft.github.io/rust-guidelines/)。这些来源各自的许可见原站；BeautifulRust 的摘要与 skill 使用 MIT 许可。

后续更新只维护本文件：核对上游指定 commit 的差异，重新审查项目适配，再更新基点；不自动同步上游规则，也不把本指南拆散进现有项目文档。以下保留上游许可声明，仅适用于来自 BeautifulRust 的内容；项目代码仍遵循仓库原有许可。

```text
MIT License

Copyright (c) 2026 Morgan Linton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
