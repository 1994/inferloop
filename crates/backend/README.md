# Backend 分组

受支持的设备架构只有 NVIDIA CUDA 与 Metal。CPU 对照位于 `testing`，默认发布版不构建也不注册。

```text
crates/backend/
  api/          # 设备描述与 CPU/GPU 交接协议
  kernel-api/   # kernel 注册与选择
  cuda/         # NVIDIA 生产目标；cuTile Rust 算子与量化参考
  metal/        # macOS GPU 执行器、MSL kernel 与受限 FFI
crates/testing/cpu/
  reference-ops/  # 标量参考算子：设备 kernel 的独立数值对照
crates/foundation/ir/src/hardware/
  mod.rs        # 后端判别、共有能力与兼容性接口
  cuda.rs       # NVIDIA architecture / capabilities / requirements
  metal.rs      # Metal capabilities / requirements
crates/service/cli/src/backend/
  mod.rs        # 选择、provider 转发与 ticket 匹配
  cuda.rs
  metal.rs
  selected.rs   # 编译期闭环的 provider 分发
```

crate 名称（`infer-backend-cuda`、`infer-backend-metal`）不因目录组织而改变。

## 执行合约

执行器实现 `BackendProvider` 的状态生命周期、submit/poll、身份、成本计时与 checkpoint，模型和调度只通过统一合约调用。`KernelProvider` 只注册本后端可执行的 kernel。设备 buffer、ticket 与 unsafe FFI 留在本组。

共享 `PrefixCache<V, M>` 统一管理 prefix publication、匹配、预算、LRU 与复用统计；
`PrefixPolicy` 可选择块指纹或快照扫描，`PrefixReuse<V>` 接入后端资源操作。
`KvCacheManager` 在此之上管理页 ledger、COW/append/rollback 与 copy pin；Metal 获取共享 lease，
CUDA 使用同一缓存核心恢复其独占设备快照，实际设备复制与 fence 保留在 backend。

## 新增 backend

1. 在本目录新增独立 crate，在 `hardware` 子模块定义该后端专属的能力与要求。
2. 注册 `BackendKind` / `DeviceBackend` / `BackendRequirements`，并实现 CLI 选择 adapter。
3. 公共结构不添加厂商 `Option` 字段；当前是编译期封闭枚举，新增 variant 必须通过穷尽匹配检查，不能悄悄退回 CPU。

动态加载 ABI 尚未实现。CUDA 接入边界见 [CUDA 说明](cuda/README.md)，KV 合约见 [KV Cache Manager](../../docs/architecture/kv-manager.md)，公共加载器与 Metal forward 见[模型执行](../../docs/guides/model-execution.md)。

## 依赖隔离

生产依赖树只包含 CUDA 与 Metal 执行器：CPU 测试执行器（`infer-backend-host` / `infer-backend-reference`）、`test-backends` feature 与 `test-cpu` CLI 选项已全部删除。`crates/testing/cpu/reference-ops` 只提供标量参考算子，是 `infer-backend-cuda` 的 dev-dependency，不注册 backend、不进入生产依赖树。supported / auto_priority 始终为 CUDA/Metal；`auto` 无可用 GPU 时返回错误。
