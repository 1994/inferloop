# 测试布局：改为 tokio 式

状态：**已决定，未实施**（2026-11-21）。取代 [测试目录方案](tests.md) 中"私有单元测试用 `#[path]` 挂载
`tests/unit/`"的部分；该方案的资源分类、执行入口与用例棘轮继续有效。

## 背景

现在每个私有单元测试的正文放在 `crates/<group>/<crate>/tests/unit/`，再由被测模块用
`#[cfg(test)] #[path = "../../tests/unit/<file>.rs"] mod tests;` 挂回去。问题：

- 前缀深度跟着模块深度变（`../` 到 `../../../../`），一个 `src/loading/mod.rs` 曾有 21 个挂载块；
- 正文看似在 `tests/`，实际是 `src` 模块的私有子模块，"隔离"只是表象，rustfmt / rust-analyzer 也要绕这层间接；
- 公开 API 测试与私有单元测试混在同一个 `tests/unit/` 里，看不出哪些用例只依赖公开接口。

规模（2026-11-21，main `ab98a677`，`tools/check/test-inventory.json` 的 `path_mounts`）：289 个挂载文件、约
60,460 行；按 `super::` 粗查，256 个引用被测模块的私有项。

| crate | 挂载文件 | 引用 `super::` | 行数 |
|---|---|---|---|
| `crates/backend/cuda` | 171 | 153 | 37330 |
| `crates/service/frontdoor` | 24 | 19 | 8916 |
| `crates/engine/runtime` | 16 | 13 | 2563 |
| `crates/model/package` | 16 | 14 | 4456 |
| `crates/foundation/core` | 10 | 10 | 437 |
| `crates/service/cli` | 10 | 10 | 860 |
| `crates/backend/cuda-kernels` | 5 | 3 | 524 |
| `crates/engine/scheduler` | 5 | 5 | 584 |
| `crates/engine/state` | 5 | 4 | 412 |
| `crates/engine/workloads` | 5 | 5 | 1188 |
| `crates/foundation/ir` | 5 | 5 | 268 |
| `crates/backend/metal` | 3 | 3 | 272 |
| `crates/backend/api` | 2 | 2 | 178 |
| `crates/backend/kernel-api` | 2 | 2 | 187 |
| `crates/diagnostics/observe` | 2 | 2 | 414 |
| `crates/foundation/spi` | 2 | 2 | 264 |
| `crates/service/agent` | 2 | 0 | 1069 |
| `crates/diagnostics/quality` | 1 | 1 | 45 |
| `crates/model/compiler` | 1 | 1 | 226 |
| `crates/model/recipes` | 1 | 1 | 179 |
| `crates/service/media` | 1 | 1 | 88 |

关闭的 #290 曾把正文直接移进 `src/<module>/tests.rs`，当时以"`src` 内放测试正文影响可读性"为由不合入；本方案
保留它的思路，但把正文集中到 `tests/` 子目录，并把只用公开接口的测试真正移出 crate。

## Rust 的约束

只有两种办法让测试访问私有项：测试在 crate 的模块树里（`src/` 下），或用 `#[path]` 挂进去。`tests/`
下的集成测试是独立 crate，只能用公开 API；AGENTS.md 禁止只为测试开放 API。所以"全部在 `tests/`、没有
`../`"只对公开接口测试成立。tokio 也是这样：私有单元测试在 `tokio/src/runtime/tests/`、
`tokio/src/sync/tests/`，公开行为在 `tokio/tests/`，跨 crate 场景在 `tests-integration` 等独立 crate。

## 目标布局

```text
crates/<group>/<crate>/
  src/
    storage/gguf/
      arch.rs                 末尾一行 #[cfg(test)] mod tests;
      tests/                  私有单元测试（模块名不变，过滤串不变）
        mod.rs
        arch.rs
  tests/
    <crate>/main.rs           唯一的 host 集成 target，只用公开 API
    <crate>/<domain>.rs
    device/main.rs            需要时：公开接口的设备场景，具名 target
crates/testing/
  cuda-acceptance/            跨 crate 共享的 CUDA 用例（host 测试与 AOT 示例共用）
  support/                    runtime / frontdoor / agent 的协议替身
```

规则：

1. 不再有测试用的 `#[path]`，也就没有 `../`；生产文件只保留一行 `#[cfg(test)] mod tests;`（或具名
   `mod <name>_tests;`，沿用现有模块名，使 `cargo test` 过滤串、设备测试清单与 CI 命令不变）。
2. 每个挂载文件按内容分类：
   - 只用公开 API → 移到该 crate 的 `tests/<crate>/`（参照已完成的 `crates/model/package/tests/gguf/`）；
   - 引用私有项 → 移到被测模块旁的 `tests/` 子目录；
   - 多个 crate 共用的用例或替身 → `crates/testing/` 下的 dev-only crate，用 dev-dependency 引入，不再
     `#[path]` 跨 crate 挂载。
3. 不为测试提升可见性；确需公开的接口要有产品理由。只镜像实现分支、没有行为价值的用例在迁移时删除并在 PR 里列出。
4. 每个 crate 原则上一个 host 集成 target；GPU 或需要进程隔离的场景另设具名 target。

## 工具与门禁

- `tools/check/test-inventory.py`：现在的棘轮是"`src` 内测试条目只减不增"且"每个 `#[path]` 挂载必须解析"。
  改为：测试用 `#[path]` 挂载数只减不增（迁移完成后为 0 并禁止新增）；`src` 内允许的位置限定为
  `src/**/tests/` 与 `mod tests` 声明；总用例数不得下降（删除需 `--record` 并在 PR 说明）。
- `tools/check/layout.py`：校验 `tests/` 下没有被 `src` 挂载的文件，`crates/testing/*` 只作 dev-dependency。
- 更新 `docs/architecture/cuda-layers.md` 中"`tests/` 按 `#[path]` 挂载"的描述与 [测试目录方案](tests.md)。

## 迁移顺序

每个 PR 一组 crate，迁移后 `test-inventory.py` 总数不变、`make check-rust` 通过：

1. 小 crate：core、ir、spi、api、kernel-api、observe、quality、compiler、recipes、media、scheduler、state、
   workloads、metal（约 50 个文件），顺带修工具规则；
2. package、cli、agent、runtime、frontdoor（约 70 个文件），抽出 `crates/testing/support`；
3. cuda-kernels 与 cuda（176 个文件），抽出 `crates/testing/cuda-acceptance`；设备测试按
   [质量门禁](../../guides/quality-gates.md) 在 CUDA 主机上逐进程跑一遍受影响清单。

#290 分支（`refactor/unit-tests-next-to-modules`）已做过机械移动和两个共享 crate，可复用其脚本与
`crates/testing` 部分。

## 性能影响

只移动测试文件与 dev-only crate，不改变生产代码、构建配置或执行路径；验证以编译、用例数与过滤串不变为准。
