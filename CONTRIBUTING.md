# 贡献指南

感谢参与 inferloop。开始之前，请先阅读 [README](README.md)、[代码布局](docs/architecture/layout.md) 与 [能力边界](docs/architecture/backends.md)，了解模块边界和当前能力范围。

## 环境准备

1. 安装 `rust-toolchain.toml` 指定的 Rust 1.99.0（最低支持 1.90）与 C 编译器。
2. 克隆仓库。无 GPU 环境运行 `make test` 做协议与单元检查；需要本机模型执行时使用 `make local-build`（Linux 启用 CUDA，macOS 使用 Metal）。

3. 完整门禁还需要 Python 3.12+、`uv`、Go、`cargo-deny`、`cargo-audit` 等工具，版本与安装命令见[质量门禁](docs/guides/quality-gates.md)。

## 开发流程

1. 从最新主干创建分支，一个分支只解决一个明确问题。
2. 在负责该功能的 crate 内修改代码，并为行为变化补充回归测试。
3. 更新受影响的使用文档、示例或[能力边界](docs/architecture/backends.md)，遵循[文档维护约定](docs/README.md#维护约定)，不新增会话日志或进度文档。
4. 本地通过相关门禁后再提交。

```sh
cargo fmt --all --check
make check-rust
```

`make check-rust` 是提交前的最低要求；改动涉及工具、依赖、Linux 或 Metal 时，请一并运行对应的 `check-*` 门禁。

## 代码与提交约定

- 遵循现有 crate 分组与依赖方向（见[代码布局](docs/architecture/layout.md)）：实现层不依赖服务入口，引擎只依赖 backend 合约而非具体执行器。
- 不通过放宽 lint、`allow`/`expect` 或提高阈值绕过问题；确需局部例外时写明原因。
- 项路径统一用 `use` 引入，不在代码中直接书写三段及以上的全限定路径；`check-rust` 通过 `clippy::absolute_paths` 强制这一约定。
- 一个 PR 说明修改前后的行为、验证命令及其结果；未运行的硬件或工具检查要注明原因。
- 不把“编译通过”描述为生产支持。CPU 测试不能替代 Metal / CUDA 实机验收。
- 保留 `Cargo.lock` 与仓库内微型测试资产；不提交完整模型、凭证、`target/` 或 `artifacts/`。
- 新增依赖说明用途；涉及设备、状态所有权或公共协议的变更要说明兼容性影响。

## 报告问题

Bug 报告请包含：

- 操作系统、Rust 版本、后端与硬件；
- 完整执行命令；
- 预期行为与实际结果；
- 最小复现步骤或样例包。

模型相关问题请说明模型配置、精度与输入规模。日志请先移除凭证和私有内容。

功能需求请说明使用场景、需要补齐的行为以及如何验证。大型架构变更建议先通过 Issue 讨论边界与验收条件。

## 安全问题

漏洞请不要公开提交 Issue，按[安全策略](SECURITY.md)私下报告。

## 许可证

提交贡献即表示同意以 [Apache-2.0](LICENSE) 许可发布。
