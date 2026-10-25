# 前缀指纹优化：进程内索引不需要 SHA-256

本评估是[容器评估](container-performance-2026-10-24.md)的第二阶段。
基线为提交 `78e5d9b`：摘要索引已改用 AHashMap，但每个 token block 仍重新执行 SHA-256。
候选同时简化指纹计算与键宽度，不将收益全部归因于某一个步骤。

## 为什么可以移除 SHA-256

`engine/state::PrefixCache` 的指纹只用于当前实例的内存索引，既不序列化、不跨进程交换，
也不作为制品身份、签名或权限判断。namespace 与 block size 初始化实例的哈希状态。
每个候选命中都先比较**完整 tokens**，然后检查 value 兼容性。因此：

- 指纹相同、tokens 不同：查询拒绝命中，不能恢复其他序列的 KV/私有状态。
- 插入遇到同一个指纹：替换并返回原 value，原来的 LRU 项及 bytes 正确移除。
  碰撞可能损失缓存命中率，但不会绕过 token 校验或泄漏页引用。
- 成功命中的 LRU 顺序仍由独立递增 clock 定义；指纹的随机种子不决定淘汰顺序。

这不要求密码学抗碰撞；两个使用不同种子的 64-bit aHash 状态组成 128-bit 内存指纹，
避免用单一短指纹过度增加偶然冲突。128-bit 是存储宽度，**不是密码学安全强度声明**。
[aHash 官方说明](https://github.com/tkaitchuck/aHash#goals-and-non-goals)明确将其限定为内存哈希表用途，
输出不保证跨版本/平台稳定，不应用于持久化、网络协议或 MAC。当前私有指纹符合该边界。

若未来要持久化或跨 worker 共享缓存身份，应单独定义稳定的内容身份协议；
不能直接导出这些内存指纹，也不能删除完整 tokens 的最终确认。

## 实现

构造时生成两个不同种子的 AHasher，并写入 namespace、block size 与域标记。
发布和查询共用 `keys()` 迭代器：复制初始状态，按相同 block 边界更新 token slice，
在需要查询的边界取得指纹。状态持续向前，不再每块新建 SHA hasher、写入父摘要、逐个转换 token 字节。
`matched_prefix` 先裁剪到 maximum 内的完整块，再一次遍历；部分块不参与匹配。
根状态和指纹都是固定大小值，没有新增热路径分配或 unsafe。infer-state 移除 sha2 依赖，
其他模块的制品/权重 SHA-256 保持原用途。

radix tree 可以进一步复用 token 前缀比较，但需要节点生命周期、内存上限和淘汰索引设计，
不是这次局部优化的必要复杂度。现有完整 token 比较仍可能成为大量连续命中块时的后续成本。

## 45 实测

i9-13900KF，Rust 1.99.0，release / thin LTO / codegen-units=1，CPU 8，
powersave governor、睿频与桌面任务不变。5 组串行交替二进制配对，每组 7 轮、
每轮 100,000 次调用；构建/测试在计时外。两侧使用同一基准源码，剩余依赖版本相同。
下表是各进程中位数的中位数，单位 ns；改善包含指纹从 32 字节缩为 16 字节对 map/LRU 的影响。

| 条目 | 2 tokens lookup+LRU 前→后 | 加速 | 128 tokens / block16 lookup+LRU 前→后 | 加速 |
|---:|---:|---:|---:|---:|
| 16 | 79.6 → 33.2 | 2.40× | 684.0 → 99.3 | 6.89× |
| 64 | 78.8 → 36.7 | 2.15× | 696.0 → 107.3 | 6.48× |
| 1,024 | 101.4 → 60.5 | 1.68× | 721.8 → 133.0 | 5.43× |
| 8,192 | 123.6 → 77.7 | 1.59× | 812.3 → 168.1 | 4.83× |

128-token contains 在 1,024 条目下 612.7 → 59.6 ns（10.28×），8,192 条目下 674.1 → 78.8 ns（8.56×）。
全部 5 组配对都改善，但桌面环境与频率变化仍有噪声，未用局部均值推算服务吞吐。
完整轮次、checksum、分配计数及源码/二进制身份见[证据](artifacts/prefix-fingerprint-2026-10-24.json)。
复现入口见[基准说明](../../tools/bench/containers/README.md)，应分别构建基线与候选版本。

范围：此结果只对应共享的 engine/state PrefixCache。CUDA 当前的 executor prefix cache
另用 VecDeque + tokens 精确前缀匹配，没有调用此处 SHA 路径；没有测量 CUDA 服务或 Metal 实机吞吐。

## 回归

保留 namespace、最长完整前缀、maximum 截断、byte budget、LRU、时钟回绕与页所有权回归。
新增两项直接验证：人为把其他 tokens 的 entry 放到查询指纹下，确认 contains、matched_tokens、
lookup/兼容性回调均拒绝，替换/排空正确释放 value 和 bytes；以及跨 block size 的增量边界、
部分块、长前缀、兼容性过滤回退和不同后缀的正确性。验证结果以最终 PR 的检查记录为准。
