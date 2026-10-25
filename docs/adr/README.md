# 架构决策记录

本目录保存已采纳的架构决策（ADR）。每条记录包含状态、日期、背景、决策与后果；修改已采纳决策时新增一条记录，而不是改写历史。

| 编号 | 决策 | 状态 |
|---|---|---|
| [ADR-0001](0001-runtime-boundaries.md) | 核心与执行边界 | 采用 |
| [ADR-0002](0002-cpu-runtime.md) | CPU 与 GPU 提交流水线 | 采用 |
| [ADR-0003](0003-model-providers.md) | 模型接入按 provider 注册，而不是代码分支 | 采用 |
| [ADR-0004](0004-performance-first.md) | 性能优先：先做执行重叠与 AOT 覆盖 | 采用 |

当前代码组织见[代码布局](../architecture/layout.md)，设计契约见[代码布局](../architecture/layout.md)。
