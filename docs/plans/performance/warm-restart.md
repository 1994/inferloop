# 热重启：启动阶段测量与缓存

状态：第一批已实施（issue #250）。测量基于 main `589a1c2a`，日期 2026-11-19。设备为 RTX 5090（sm_120），release 构建。每次启动都用独立的 `XDG_CACHE_HOME`，计时从 exec 到 `listening for requests`。原始日志和脚本在 GPU 主机的 `~/benchmarks/inference.rs/warm-restart-2026-11-19/`。

## 测量

| 场景 | 2B（Qwen3.5-2B） | 27B（Qwen3.8-27B-NVFP4，MTP 2） |
|---|---|---|
| 冷启动：缓存全空，含 cuTile JIT 和调优 | 68 s | 422 s |
| 调优表已有、准备戳失效：重跑准备子进程（改动后还要首次写入身份缓存） | 20.2 s → 16.7 s | 改动前未测 → 90 s |
| 热启动：缓存完整，二进制和参数不变 | **10.1 s → 7.5 s** | **64.3 s → 22.1 s** |

热启动各重复两次以上，同一配置的波动不超过 0.1 s。改动前，27B 热启动的 64 s 拆开如下：

- 调优键要算 kernel 身份，每个候选 kernel 都会跑一遍 cuTile 前端（Rust→Tile IR→bytecode，取 L2 key）。这部分不计入 `CUTILE_JIT_TIMING`，在日志里表现为各个 "cached timings"、"choice"、"readout captured" 之前约 0.1–2.5 s 的空档，合计约 35 s。
- MTP 草稿头 BF16→FP8 的主机端量化花了 6.6 s，其中 88% 是逐元素在码表上二分查找。
- 真正的 kernel 编译只做前端：260 个 kernel 共 9.6 s，tileiras 全部命中磁盘缓存（0.09 s）。
- 权重读取和上传 3.0 s，属于 IO。

另外，准备子进程的完成戳以完整命令行为键，换一个 `--listen` 端口就会让准备整个重跑一遍，2B 上多 10 s。

## 已实施

1. **kernel 身份缓存**（`infer-cuda-kernels` 的 `tile::identities`）：从 L1 key 映射到 L2 key，追加写到 `$XDG_CACHE_HOME/infer-cuda/kernel-identities/<hash>.identities`。L1 key 覆盖 module、function、generics、布局、编译选项、设备和编译器身份，但不覆盖 kernel 链接进来的 helper 模块，所以只对同一份构建成立。因此 hash 包含可执行文件的路径、大小、mtime，以及 backend 传入的设备与驱动身份；重新构建会换一个新文件，目录里最多保留 16 个。损坏的行会被忽略，写入失败只会让下一个进程重新计算一次。
2. **E4M3 量化改为位运算**：按 round-to-nearest-even 舍入、饱和到 ±448，NaN 也饱和，语义与原来逐值一致。测试把旧的码表搜索作为独立参考，覆盖所有码值、中点及其相邻 ulp、特殊值，以及按 4099 步长扫过全部 f32。27B 草稿头 6.6 s → 1.4 s。
3. **准备键剔除只影响服务的参数**：`--listen`、`--served-model-name`、`--enable-response-store`、`--storage` 不再进入准备戳的键，其余参数（包括没登记的）照旧参与。

热启动时的调优选择与改动前一致：27B 两次都是 3251/3252 次 table-cache 命中，且没有新的搜索或回退。27B 贪心请求输出正确（17×23=391，MTP 2）。

## 后续

- **kernel 编译前端**（27B 9.6 s，2B 2.8 s）：cuTile 的磁盘缓存只覆盖 bytecode→cubin。跳过前端需要同时持久化 launch 校验用的 `Validator`，这要上游 cuTile 提供 L1 级的磁盘缓存，或者走 AOT kernel（C2）。
- **CUDA graph 捕获与 choice 校验**：27B 池捕获剩下约 6 s，每个缓存的 GEMM 选择仍要在真实权重上做一次逐位校验（约 0.1 s × 数十个）。后者的结果只取决于"权重 + 选择 + 二进制"，可以按权重内容摘要缓存，但先要证明摘要的成本低于校验本身。
- **权重读取**：3.0 s / 19 GB，接近 page cache 拷贝的带宽，暂不处理。
