# KV cache 分页重构方案（PagedAttention 的分层与落地）

入口修正（2026-10-13）：生产 CLI 的 `--chunked-recurrent` 及环境变量
`INFER_CUDA_CHUNKED_RECURRENT` 已撤下；chunk 候选仅通过独立诊断 loader 配置。
Tiled attention 也只保留 CUDA 内部测试入口。下文旧开关和命令属于历史实验协议，
复核时使用对应冻结制品，不再作为当前用户配置建议。

状态：**P0a / P1 / P2 已落地并实测（2026-10-11，`220699c` + `87ca6d0`）**；P3 的 `CB_DECODE_SLOTS=16`
验收在物理上不可达（见 §五 P3 行）；P4–P6 未做。第①步（块表间接寻址）早已落地。
源码基点为 `d1f313e`（2026-10-11）；更早的基线是 `d5afa9b`（2026-10-10）。

2026-10-25 复核：本项未进入当前 C1 主线（当前聚焦 attention 成本与真实显存画像），P4–P6 保留为容量与前缀复用储备；下文"下一步"是当时的交接记录，不再构成当前排期。

本文是阶段性交接文档：目标、分层判断、实测差距、已落地地基、硬边界账目、
分阶段计划、复现命令、代码地图、已否证路线与未决问题都在这里；
逐条实验数据与推理过程见[性能实验记录](../../archive/research/cuda-performance-experiments.md) §13–§15。

排期归属见[路线图](../README.md)；CUDA 侧其它优化见 [CUDA 方案](cuda.md)。

## 〇、任务目标（原文摘要）

彻底重构 CUDA 侧 KV cache 管理，使长上下文 / agentic 负载（长共享前缀、高并发、多轮）不再受
"每序列按最大长度预留 + 前缀整状态快照拷贝"拖累。目标形态：**PagedAttention 式块池 + 每序列块表 +
块级前缀共享（内容哈希、引用计数、COW）**；48 层 gated-delta/conv 的递推状态保持每序列，
并在与共享块相同的边界上做检查点。

约束：默认路径不变、新路径挂开关；每步官方 matrix 实测并与改动前对比、数据写进上述实验记录并提交；
**不得让工作区处于编译不过的状态**；不触碰并行进行中的 `crates/model/package/**`、`Makefile`、
`tools/package/**`、`docs/plans/speculation/**`。

## 一、结论先说：PagedAttention 应分三层放，现在只对了最下面一层

| 部分 | 应住哪 | 现在住哪 | 判定 |
|---|---|---|---|
| **块表间接寻址**（`table[逻辑块] → 物理块`，设备侧一次查表） | **CUDA 后端**（设备） | `resident/capture.rs` 的 `Capture.table`、`resident/slot_batch.rs` 的 `SlotPool.tables`、`resident/metadata.rs::identity_table`，被 5 处核使用 | ✅ 正确，已落地验证（§13.16–13.17） |
| **块池 / 按需增长 / 前缀共享 / 引用计数 / 按块计价** | **engine + SPI 契约** | SPI 已定义：`foundation/spi/src/backend.rs:96` 的 `kv_cache()`、`:102` 的 `state_page_growth()`、`reusable_prefix*()`；`foundation/ir/src/diagnostics.rs:65` 的 `KvCacheInspection`（`block_size`/`total_blocks`/`free_blocks`/`active_blocks`/`cached_blocks`/**`shared_blocks`**/`bytes_per_block`） | ✅ 契约已在，**CUDA 未实现** |
| 同一策略的第二份实现 | —— | CUDA 自建：`SlotPool` 的 per-slot arena + `free` + `bind`/`SlotCopy` | ❌ 重复，engine 看不见 |

**engine 已经在按"页"调度**：`pipeline/scheduling/ready.rs:50,118` 与 `overlap.rs:95` 每个调度步都调用
`backend.state_page_growth(state)?` 写入 `cost_query.page_growth`；`admission.rs:51`、`resources.rs:11,23`
用 `kv_cache()` 做按块核算。**实现者只有 Metal**（`backend/metal/src/executor/provider.rs:67,70`）与 runtime
测试替身；**CUDA 全部走 SPI 默认值**（`kv_cache() → None`、`state_page_growth() → Ok(None)`）。

**这一条有冻结证据**：交付基线的 `config_readback.readback.kv_cache` 对 native 记录的就是 **`null`**。
后果（均有实测，§13.18/13.25/13.29/13.31/13.33）：按块准入失效（退回配置 `num_gpu_blocks`）；
**按需增长不存在** ⇒ 只能按 `capacity` 一次性预留（即"每序列按最大长度预留"）；计价只能按
**字节 × 满容量**，而池在 backend 内部另算一套 ⇒ 两套词汇互不可见、池与请求抢同一份预算；
前缀共享只能停在"整状态快照拷贝"。

⇒ **因此本方案的核心动作**：不要继续把策略往 CUDA 里塞（例如"共享 arena + 偏移表"），
而是让 CUDA **实现 engine 已经依赖的那份分页契约**，把块池/增长/共享的词汇交回 engine 可见的层。

## 二、实测的 vLLM 差距（冻结基线 `27b-mtp2-serving-v1`）

来源：`benchmarks/baselines/serving/27b-mtp2-serving-v1/` 的 native 与 vllm 报告，2026-10-09，
源码 rev `2bda883`，RTX 5090，两组独立重复 g1/g2，配置 `--num-speculative-tokens 2
--max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.88`（**未开** `--chunked-recurrent`）。
下表由**逐请求原始记录**重算（每个 case 对 3 次重复 × 并发数的 `ttft_seconds`/`tpot_seconds` 取均值）。

| case | 指标 | native g1/g2 | vLLM g1/g2 | native/vLLM |
|---|---|---|---|---|
| short | TTFT | 54.5 / 58.0 ms | 36.4 / 33.6 ms | **1.50× / 1.73× 慢** |
| short | TPOT | 11.8 / 12.3 ms | 8.6 / 8.6 ms | 1.37× / 1.44× 慢 |
| long | TTFT | 319.8 / 309.3 ms | 71.8 / 72.2 ms | **4.45× / 4.29× 慢** |
| long | TPOT | 12.2 / 11.5 ms | 8.7 / 8.0 ms | 1.40× / 1.43× 慢 |
| batch4 | TTFT | 704.7 / 715.9 ms | 82.4 / 94.4 ms | **8.55× / 7.58× 慢** |
| batch4 | TPOT | 16.3 / 16.3 ms | 9.0 / 8.5 ms | 1.81× / 1.91× 慢 |
| **hot_long** | **TTFT** | **85.0 / 84.6 ms** | 216.9 / 207.1 ms | **0.39× / 0.41×（我们快 2.4–2.6×）** ✅ |
| hot_long | TPOT | 11.4 / 11.5 ms | 7.9 / 9.1 ms | 1.46× / 1.26× 慢 |

原始样本（batch4 repeat0 slot0，52→64 token）：native TTFT **146.3 ms** / TPOT 18.4 ms；
vLLM **81.5 ms** / 9.5 ms。`hot_long`（3647-token prompt）：native TTFT **87.5 ms 且复用 3584 token**；
vLLM **215.7 ms，复用 1600 token** ⇒ 我们唯一赢的那格，赢在**前缀复用更多**。

**读法**：

1. 总体落后 **1.3–8.6×**，最差的是 **TTFT（prefill）**，TPOT 只差 1.25–1.9×（decode 带宽量级）；
2. 差距集中在 **长 prompt TTFT（4.4×）** 与 **并发 TTFT（7.6–8.6×）**；
3. batch4 那一格里含**并发上限**：本机池宽 ≤ `CB_DECODE_SLOTS=4`，实测 8 并发会**分两批**
   （TTFT max ≈ 2× median），而 vLLM 是连续批处理；
4. **KV 分页重构的收益不在这四格比值上**，而在**容量/规模**（长上下文/多序列装得下、块级共享），
   即 agentic 场景的可持续性；前缀复用我们目前靠"整状态快照"已经赢，升级为按块共享是进一步收益。

**两个前提必须随数字一起引用**：

- 该基线**未开** `--chunked-recurrent`；该开关自身 A/B 实测 TTFT **−15.7%/−20.1%/−14.6%/−16.0%**
  （short/long/batch4/hot_long，§12.20）。未在同窗口对 vLLM 重测，故**不能直接相乘**。
- `statistical_basis.paired_units = 2`，而基线方案要求 **≥5** ⇒ 上表**指示性**，不是签发结论。
  要变成可签发的差距，需按 §六 的命令做**同窗口 ≥5 单元成对测量**。

### 2.1 顺手发现的 artifacts 缺陷（**已修，2026-10-11，`220699c`**）

冻结 manifest 的 `reports[].role` 写 `baseline=native / candidate=vllm`，但
`gate.performance.metrics[].baseline` 里放的却是 **vLLM 的数**、`.candidate` 是 **native 的数**
（batch4 wall：`baseline 0.679` = vLLM、`candidate 2.376` = native；已对原始文件核两遍）⇒
**gate 的 `passed` 标记是反的**：它恰好"通过"了唯一那格我们更快的指标
（`hot_long` TTFT `ratio 0.390, passed true`）。这会让读者把差距方向读反。

**修法与验证（已落地）**：根因在 `tools/bench/freeze-baseline.py`——`freeze()` 原按命令行顺序写
role，现改为用 `gate.orientation(baseline, candidate)` 判定真实方向后按 [baseline, candidate]
角色序写 `reports[]`/`engines`，`source`/`hardware` 跟随被测侧，gate 块新增 `ratio_orientation`
字段。三个冻结 manifest（`2b-mtp0`/`27b-mtp0`/`27b-mtp2-serving-v1`）的 role 已交换为
`baseline=vllm / candidate=native` 并与 gate 数值对齐；`verify-baseline.py` 对三个基线全部
重算通过（`compare()` 内部按 engine identity 重定向，与文件顺序无关），
`test_freeze_baseline`/`test_verify_baseline`/`test_compare_results` 79 个测试全绿。

## 三、已落地的地基（不要重做）

| 项 | 状态 | 实测 | 提交 |
|---|---|---|---|
| ① 块表间接寻址：预填充写 + 预填充读（SIMT/张量核）+ decode 读（SIMT/split-KV）五处核 + bind + 预算 | ✅ 默认路径 | 恒等映射下官方 matrix **12/12 trial token 逐位相同**、越界 0 | `953c0ea`、`9fb692e` |
| `CB_SLOT_TOKENS` 2048 → 8192 | ✅ 默认 | 4×2560 并发 wall 10.59 → **9.21 s（−13%）**、TTFT 中位 −22%、`gpu_budget` 递延 −32% | `921abe3` |
| 池内计价（占槽请求不再二次预留） | ✅ 默认（对交付 profile 尚不生效） | matrix 16/16、27 单测、4 GPU 测试 | `9421a8c` 等 |
| **P1+P2：CUDA 实现 `kv_cache()` 与 `state_page_growth()`**（本方案的 SPI 闭环） | ✅ 默认路径（池存在时上报；池缺席回退 None，行为与之前一致） | 官方 matrix 16/16、**28/28 请求流 token 逐位一致**（`artifacts/kv-paging-p1p2-20261011/`，run-id `kv-before`/`kv-after`）；runtime readback `kv_cache` 由 `null` → 真实块报（32 token/块 × 1024 块 × 1 MiB）；clippy/单测/GPU 测试全绿（含 release `attention_decode` 2/2）。明细见实验记录 §15 | `87ca6d0`（2026-10-11） |
| Chunk recurrent 内部候选（诊断 loader） | 生产 CLI 无算法开关 | 历史局部 TTFT −15.7%/−20.1%/−14.6%/−16.0%；未获默认完整验收 | §12.20 |
| CUDA 内部 tiled attention 候选（仅测试程序） | 内部比较 SIMT/QT16/QT32；生产不开放算法开关 | 六项补偿通过真实 F64；256 题原任务质量失败，生产保持 SIMT | [2026-10-13 记录](../../archive/research/cuda-attention-query-sharing-2026-10-13.md) |

## 四、硬边界账目（决定了并发上限）

| 量 | 值（capacity 8192） |
|---|---|
| 每槽 fp8 KV | **262 MiB**（16 个全注意力层 × 8192 × 4 头 × 256 × 2 × 1 B，无浪费） |
| 每槽 f32 递推状态（48 层 delta/conv） | **149 MiB** |
| 每槽合计 / 引擎 `state_budget` | **405 MiB** / **4.8 GiB** ⇒ **并发 ≈12 槽** |

⇒ **槽容量与槽数互斥**：容量大→长会话能进池但 KV 线性增长；槽数多→并发高但每槽 ×405 MiB。
只有"序列只持有实际用到的块"能消除这个互斥——这正是本方案要交付的东西（§二 的 batch4 那格也受它制约）。

## 五、分阶段计划

每阶段硬性要求：默认路径不变、新路径挂开关、单测与 GPU 测试全绿、官方 matrix 实测并与改动前对比、
数据写进实验记录并提交，**不得让工作区编译不过**。

| 阶段 | 内容 | 启动条件 | 完成标准（验收） |
|---|---|---|---|
| **P0** | **同窗口 ≥5 单元成对基线**：当前默认 + `--chunked-recurrent` 对 vLLM，按 §六 命令跑 | 无（可与任何线并行） | 产出可签发差距（替换 §二 的指示性表），并修 §2.1 的 manifest 方向缺陷。**进展（2026-10-11）**：§2.1 已修并验证（见该节）；≥5 单元成对测量**阻塞**——本机无 vLLM 环境 |
| **P1** ✅ | CUDA 实现 `kv_cache()`：把已有块池与表如实上报（`block_size`/`total`/`free`/`active`/`cached`/`shared`/`pool_bytes`/`bytes_per_block`） | 无 | **已验收（2026-10-11）**：`/native/v1/runtime` 的 `config_readback.kv_cache` 由 `null` 变为真实块报（`block_size=32, total_blocks=1024, bytes_per_block=1 MiB, pool_bytes=1 GiB`）；观测端口出现 `available_kv_blocks`/`active_kv_blocks`/`cached_kv_blocks`/`kv_pool_bytes`；engine 逻辑页池与后端调和（`serving.rs` 把 `block_size` 同调成 32，`num_gpu_blocks=total_blocks=1024`，逻辑容量 4096×16 → 1024×32 = 32768 token，恰好等于物理池与 `max_model_len`）；官方 matrix **16/16 trial、28/28 请求流 token 逐位一致**（run-id `kv-before`/`kv-after`，`artifacts/kv-paging-p1p2-20261011/`），两侧 checklist 均 compliant。实测数据见实验记录 §15 |
| **P2** ✅ | CUDA 实现 `state_page_growth()`：按当前 token 数报"下一步用/增长多少块" | P1 | **已落地（2026-10-11）**，验收口径修正见下：在租序列报 `ceil(history/32)`、私有序列报满容量预付（`required_pages` 恒 0，物理诚实）；`cost_query.page_growth` 不再恒 `None`。但**"deferred_reasons 显著下降"在计划点名的两组负载（4×2560 / 2×8000）上不成立——实测前后逐位相同**（4×2560：TokenBudget +24/+23、GpuBudget +72/+72、Preparation +8/+8；2×8000：各 +4 Preparation；`state_capacity` 两侧恒 0）：那两组负载的递延来自 token 预算与执行预算，KV/状态计价从来不是它们的瓶颈。P2 的价值是把真实增长数交给调度器（可证、可查），而不是改变这两组负载的递延结构 |
| **P3** | 计价改为块数（原④）：`state_reservation_bytes*` 与准入按块；池已覆盖的部分从请求报价扣除 | P1、P2 | `CB_SLOT_TOKENS` 不再是墙：`CB_DECODE_SLOTS=16` prewarm 成功、matrix 16/16；长上下文请求不再因"池占内存"被拒。**注意 §13.30**：报价大头是固定激活 arena 与块两件事，减免必须针对正确的项。**边界（2026-10-11 复核）**：`CB_DECODE_SLOTS=16` 的 prewarm 验收在本机物理不可达——16 槽 × 405 MiB/槽（262 KV + 149 递推）≈ 6.3 GiB > 4.8 GiB 状态预算，只有 P5 的共享 arena 才可能解开；本阶段剩余价值是"池覆盖部分从报价扣除"，其中池内计价部分已随 `9421a8c` 落地 |
| **P4** | 块级前缀共享 + COW（原③）：`reusable_prefix*()` 按页暴露 + 内容哈希 + 引用计数 | P3 | `hot_long` 类场景 TTFT 与块复用计数同时改善；重复前缀不再产生整状态拷贝。**未做（2026-10-11）**；§八 已否证"共享 arena + 偏移表"路线，参照实现应看 `infer_state::kv::KvCacheManager`（`crates/engine/state/src/kv.rs`，Metal 侧完整 BlockPool/前缀/COW） |
| **P5** | 共享 arena / 按需分配（原②，**可选实现细节**） | 仅当 P1–P4 后 per-slot arena 仍是浪费 | 若做：先按 §13.23 办法在**回放时**打印 `(lane, position, block, 表项)`，并核对 draft 私有 program 的 `identity_table` 与它 arena 的归属。**注意：这条路上 `attention.rs:113` 越界不在关键路径**。**未做（2026-10-11）**：§13.21–13.26 三轮实测失败已否证，数周级高风险 |
| **P6** | token-major 块内布局 + 张量核 SDPA（原⑤） | P3 后按块寻址稳定 | ≥3072 走张量核；与 SIMT 路径在 1e-4 相对误差内一致（沿用已验证的 6.2×）。**未做（2026-10-11）** |

### 5.1 P1 参考实现要点（最划算的第一步，改动小）

- 位置：`crates/backend/cuda/src/executor/provider.rs`（`impl BackendProvider for CudaBackend` 在同文件，可直接
  读 `self.slots: Option<SlotPool>`）。
- 数据已经在手上：`SlotPool` 的 `capacity`（`resident/slot_batch.rs:278`）、`slots`、`free`、
  `tables`；块粒度是 `constants::KV_BLOCK_TOKENS = 32`。
- 参照实现：`crates/backend/metal/src/executor/provider.rs:67`（Metal 的 `kv_cache()` 与
  `state_page_growth()` 是唯一现成范本）。
- 结构定义：`crates/foundation/ir/src/diagnostics.rs:65`（`KvCacheInspection`）、`PageGrowth`。

## 六、复现命令（新会话直接抄）

环境：CUDA 机器；构建需 `BINDGEN_EXTRA_CLANG_ARGS="-isystem /usr/lib/gcc/x86_64-pc-linux-gnu/16/include"`。

```bash
# 构建
BINDGEN_EXTRA_CLANG_ARGS="-isystem /usr/lib/gcc/x86_64-pc-linux-gnu/16/include" make local-build
# 门禁与测试
cargo clippy --locked -p infer-backend-cuda --features cuda -- -D warnings
cargo check --locked -p infer-backend-cuda --features cuda --all-targets
cargo test  --locked -p infer-backend-cuda --features cuda
cargo test  --locked -p infer-backend-cuda --features cuda -- --ignored resident::slot_batch resident::slot_verify
timeout 900 cargo test --locked --release -p infer-backend-cuda --features cuda --lib -- --ignored attention_decode --test-threads 1
timeout 300 ./tools/check/gate.sh tools

# 官方 matrix（内存守卫：safe-run 需要 --memory-gib，本机 12 够用）
bash tools/bench/safe-run.sh --memory-gib 12 python3 tools/bench/serve-compare.py \
  --engine native --model /home/r/models/Qwen3.8-27B-NVFP4 --executable target/release/infer \
  --inputs artifacts/workloads/Qwen3.8-27B-NVFP4-inputs.json --output-dir artifacts/stage-b-20261010 \
  --mtp 2 --tokens 64 --gpu-memory-utilization 0.88 --max-model-len 32768 --max-num-seqs 16 \
  --run-id <ID> --profile-id 27b-mtp2 --checklist benchmarks/profiles/27b-mtp2.json \
  --extra --chunked-recurrent
```

**手起服务（注意：默认子命令就是服务，不要加 `serve`，也不要把 `--model` 与子命令混用）：**

```bash
env INFER_CUDA_CHUNKED_RECURRENT=1 ./target/release/infer /home/r/models/Qwen3.8-27B-NVFP4 \
  --listen 127.0.0.1:38111 --max-model-len 32768 --max-num-seqs 16 \
  --gpu-memory-utilization 0.88 --num-speculative-tokens 2      # 末尾两者按需
# 观测：curl -s localhost:38111/native/v1/observability  → deferred_reasons / kv_cache 字段
```

**并发对照脚本（写入 `/tmp/conc.py`；判据：`ttft max ≈ 2× median` 即"分两批"）：**

```python
#!/usr/bin/env python3
import json, sys, threading, time, urllib.request
BASE, N, TOK = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
prompt = [(i * 7 + 11) % 240000 for i in range(TOK)]; out = {}
def one(slot):
    payload = {"id": 700000 + slot, "model": 1, "input": {"Sequence": {"tokens": prompt}},
               "workload": {"Generate": {"max_new_tokens": 64}},
               "sampling": {"temperature": 0.0, "eos_tokens": []}}
    req = urllib.request.Request(BASE + "/native/v1/stream", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter(); ttft = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            if line.startswith(b"data:") and b"Token" in line:
                ttft = ttft or (time.perf_counter() - start)
    out[slot] = ttft or 0.0
t0 = time.perf_counter()
ts = [threading.Thread(target=one, args=(i,)) for i in range(N)]
[t.start() for t in ts]; [t.join() for t in ts]
v = sorted(x * 1000 for x in out.values())
print(f"N={N} tok={TOK} wall={time.perf_counter()-t0:.3f}s ttft min/med/max={v[0]:.0f}/{v[len(v)//2]:.0f}/{v[-1]:.0f} ms")
```

**重算 §二 差距（对冻结基线逐请求原始记录，避免误读 manifest 的方向）：**

```python
import json, statistics as st
B = 'benchmarks/baselines/serving/27b-mtp2-serving-v1/'
def series(eng, group, case, field):
    d = json.load(open(f"{B}Qwen3.8-27B-NVFP4-mtp2-{eng}-27b-mtp2-{eng}-{group}.json"))
    x = [float(r[field]) for t in d['trials'] if t['case'] == case and not t.get('warmup')
         for r in t['results']]
    return st.mean(x)
for case in ['short', 'long', 'batch4', 'hot_long']:
    for f in ['ttft_seconds', 'tpot_seconds']:
        n = series('native', 'g1', case, f); v = series('vllm', 'g1', case, f)
        print(f"{case:9s} {f:14s} native={n*1000:7.1f}ms vllm={v*1000:7.1f}ms native/vllm={n/v:.2f}x")
```

**测量纪律**：跨小时绝对值不可比（≈10% 漂移）⇒ 必须**同窗口成对**，且**重复**（单次"图总量"曾经骗过人，§12.22）。
**实验改动先存补丁**（`git diff > /tmp/x.patch`）：本会话因未存补丁而把同一改动重写了三遍。

## 七、代码地图

| 关注点 | 位置 |
|---|---|
| 块表字段与来源 | `resident/capture.rs`（`Capture.table`/`capacity`）、`resident/metadata.rs::identity_table`、`resident/slot_batch.rs`（`SlotPool.tables`、`slot_tables`） |
| 五处 KV 访问点 | `resident/capture_attention.rs`（预填充写 `attention::append`/`kernels::append`；读 `decode`/`decode_tiled`）、`resident/attention.rs`（decode SIMT）、`resident/attention_decode.rs` + `attention_decode/workspace.rs`（split-KV） |
| 每槽状态与 arena | `resident/program.rs`（`allocate_states`）、`resident/fp8_cache.rs`（`allocate`）、`resident/slot_batch.rs::state_bytes` |
| 池的构建与降宽 | `executor/execution.rs:150-320`（`ensure_slot_pool`/`create_slot_pool`/`prepare_draft_pool`/`reclaim_cached_for_pool`） |
| 计价与准入 | `executor/provider.rs`（`state_reservation_bytes_for`）、`loading/budget.rs::sequence_budget`、`engine/runtime/src/pipeline/scheduling/admission.rs`、`resources.rs` |
| 页契约（要实现） | `foundation/spi/src/backend.rs:96,102`、`foundation/ir/src/diagnostics.rs:65`、范本 `backend/metal/src/executor/provider.rs:67,70` |
| 请求规划 | `engine/workloads/src/native.rs:111`（`reserved_tokens = max(unit.len()) + max_new_tokens`）、`engine/runtime/src/pipeline/admission.rs:137`（state capacity 取值） |
| 程序工作区上限 | `model/compiler/src/compilation.rs:109`、`device.rs:72`（`arena_budget_bytes`） |

## 八、不要再走的路（已实测否证）

| 路线 | 否证 |
|---|---|
| 加大 `CB_SLOT_TOKENS` 突破容量上限 | 16384 在 MTP 下准备挂起 30 s（§13.30/13.31）；不解决并发 |
| 加大 `CB_DECODE_SLOTS` 提并发 | 16 槽 prewarm 失败：12 槽是状态账目的硬上限（§13.33） |
| 池内减免 KV 计价 | 减错项，报价大头是激活 arena（§13.30） |
| prefill rung 预算 | 扣掉池 arena 后仍选最宽档（256），未解决 prewarm（§13.32） |
| "表/arena 错配"族假设 | 表的形状（§13.27）与取值（§13.28）都与 arena 自洽 |
| 共享 arena + 偏移表（`KvSpan` 双侧 / `arena_capacity` 拆分） | 编译与 clippy 全绿，但 draft 路径 `attention.rs:113` 设备越界，未定位到最后一处（§13.21–13.26） |
| cuBLASLt 委派 / GEMM tile 扫描 / SDPA `load_pipelined::<4>` / delta 代数重写 / value-dim 拆分 / outlier 归因 | §12.22 等 |

## 九、未决问题

1. **MTP 池几何挂起**：`(CB_SLOT_TOKENS=2048, 槽 16)` 与 `(16384, 4)` 在准备阶段挂起 30 s、0 token
   （`reason=Failed("resource preparation acknowledgement timed out")`），而 `(8192, 4)` 正常。
   怀疑与池化推测图捕获在特定 (capacity, width, verify) 组合下的行为有关；与 P1–P4 独立，可单独立项。
   **2026-10-11 补充（P1/P2 前后 A/B 实测）**：同一失败签名在**默认几何**上也能用负载打出——
   5×8000 并发（越 4 槽池）时，两个构建（有/无 `kv_cache()` 上报）都表现为 4 成功 + 第 5 条
   准备 30 s 超时失败，deferred 增量两侧同为 +9 Preparation。即这不是池几何特例，而是
   **"同时准入、准备期才发现物理装不下"的准入竞态**：5 条请求在彼此成为 active 之前全部通过
   准入，第 5 条的私有预留撞上已租满的池。P1 的真实块上报没有改变它（准入时刻余量仍然够），
   要修需要准入计入"准备中"的在途预留。
2. **状态侧并发上限**：149 MiB/槽的 f32 递推状态若改 bf16 可近乎翻倍并发，有数值风险；
   官方 matrix 的 token 对比可判定，需单独立项。
3. **跨层契约的测试面**：P1–P3 的验收涉及 engine 调度/准入分支，需要设备窗口与观测口径确认
   （观测入口：`/native/v1/observability` 的 `deferred_reasons` 与 `kv_cache` 字段）。

## 十、当时的下一步（2026-10-11 记录，不构成当前排期）

P1/P2 已落地并验收（§五 表），§2.1 已修。剩余工作按序：

1. **P0 收口（阻塞中）**：≥5 单元成对基线需要 vLLM 环境；本机 `import vllm` 不可用，
   先恢复 vLLM 环境（或换机）再按 §六 跑。§2.1 的 manifest 方向缺陷已修，不阻塞签发。
2. **P3 剩余部分**（启动条件已满足）：计价按块化——把"池已覆盖的部分从请求报价扣除"做完整
   （池内计价已随 `9421a8c` 落地，剩私有路径报价与准入的块化）。**不要**把
   `CB_DECODE_SLOTS=16` prewarm 当验收：物理账目（§四）决定了它在 4.8 GiB 状态预算下不可达。
3. **P4**（块级前缀共享 + COW）：参照 `crates/engine/state/src/kv.rs` 的
   `KvCacheManager`（Metal 侧完整范本）；§八 的"共享 arena + 偏移表"路线已否证，不走。
4. P5/P6 按 §五 的启动条件排队；§九.1 的准入竞态（5×8000 实测复现）可单独立项。

若用户更看重"比分"而非"容量/规模"，§二 读法第 3、4 条指出应优先 batch4/long 的 TTFT：
即池宽（受 §四 账目限制）与 prefill 计算（`--chunked-recurrent` 可直接开、张量核 attention 在 ≥3072 有 6.2×），
这与本方案的 P3–P4 是**两条不同但相容的线**。
