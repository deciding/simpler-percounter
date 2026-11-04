# 同一个算子，prefill 和 decode 编出来不是同一张图

`decode_attn_swa` 和 `prefill_attn_swa` 是 DeepSeek-V4.1-Flash 里同一个 kernel
家族的两半。在 a5 host_build_graph 上，percounter 对 legacy 的结果是
**decode +36.7%、prefill −17.4%**，同一个调度器，相差五十多个点。

本页讲这两张图在形状上到底差在哪，以及每个数字是怎么来的。**所有任务名和块数都
是从生成的编排代码里直读的**（`build_output/.../next_levels/<rank>/orchestration/
<rank>.cpp`，每个任务前面有 `// Spmd <名字>` 注释，后面跟 `set_core_num(N)`），
不是从 Python 推的。

缺口本身的成因在[percounter 为什么在 prefill attention 上慢于
legacy](prefill-attention-deficit.md) 和 [swa：为什么连排序都救不了
它](swa-why-no-order-helps.md)。本页只负责图的形状。

## 先排掉一个混淆：三个互不相干的 64

读这些代码最容易踩的坑是有三个 64，数值撞在一起但毫无关系：

| 哪个 64 | 来自 | 是什么 |
| ------- | ---- | ------ |
| `LOCAL_H = 64` | `H = 64`，`TP = 1` | **头数**，是工作单元的一个维度 |
| `_PREFILL_WORKERS = WORKER_TILE = 64` | 模型库里手写的常量 | **prefill 的任务宽度上限** |
| AIV lane = 64 | 硬件（32 cluster × 2） | 核数 |

本页下文凡是写"宽度 64"的，指第二个。

## 两个概念：工作单元和任务宽度

共用的算子工厂里只有两行是关键的（`attention_ops.py`，`make_rope` 一族）：

```python
blocks  = num_tokens * heads            # 工作单元总数
workers = pl.min(blocks, worker_limit)  # 任务宽度 = 用几个 block 去铺这些工作单元

with pl.spmd(workers) as tid:                       # 外层：一个任务，workers 个 block
    worker = pl.tile.get_block_idx()
    for block in pl.range(worker, blocks, workers): # 内层：block 内串行的跨步循环
        rotate_block(x, cos, sin, output, block)
```

`worker_limit` 的来源就是区别所在：

```python
worker_limit = 2**31 - 1 if max_workers is None else max_workers
```

**decode 不传 `max_workers`**，于是 `workers == blocks`，一个 block 一个工作单元，
内层只跑一轮。**prefill 处处传 `max_workers=_PREFILL_WORKERS = 64`**，于是宽度封在
64，余量进内层循环。

一个工作单元是什么，`_make_rope_block` 写得很死：

```python
t = block // heads      # 哪个 token
h = block % heads       # 哪个头
base = h * head_dim     # 只旋转这一行里这一个头的那段
```

**一个工作单元 = 一个 (token, 头) 对。**

## prefill 的循环层次

生成的编排代码的骨架（`num_tokens = 257`）：

```cpp
// 第 0 层：主机侧，query 分块
for (start = 0; start < num_tokens; start += 128) {       // 3 圈
    active = min(128, num_tokens - start);                 // 128, 128, 1

    // 第 1 层：主机侧，矩阵乘的行块（只有部分算子有）
    for (mt = 0; mt < (active + 31) / 32; mt++) { ... }

    // 第 2 层：设备侧，一个任务 = workers 个 block
    // 第 3 层：block 内的跨步循环（见上一节）
}
```

`QUERY_TILE = 128` 是**一次处理多少行 query**。第 0 层之所以存在，是因为按块开的
scratch 张量放不下整条序列：

```python
selected = pl.create_tensor([QUERY_TILE, 128, HEAD_DIM], dtype=pl.BF16)  # ≈16 MB
```

换成完整序列长度（`PREFILL_MAX_TOKENS = 4096`）就是四百多 MB。**分块是被内存逼的。**

而且这一层是**一条链**，不是可并行的循环：

```python
with pl.spmd(WORKER_TILE, name_hint="prefill_swa_stage", deps=[chunk_done]):
    ...
with pl.spmd(WORKER_TILE, name_hint="prefill_swa_collect") as collect_tid:
    ...
chunk_done = collect_tid        # 下一圈的 stage 等这一圈的 collect
```

scratch 跨圈复用，所以第 N+1 圈的第一个任务必须等第 N 圈的最后一个任务。
swa 的最长链占到任务数的 0.75，这是来源之一。

### 代入下标：每条 lane 实际在干什么

拿 `prefill_attention_q_rope` 这一个任务，`active = 128`、`heads = 64`：

```text
blocks  = 128 * 64 = 8192
workers = min(8192, 64) = 64
内层轮数 = 8192 / 64 = 128

第 k 轮：block = worker + k*64
  h = (worker + k*64) % 64 = worker     ← 和 k 无关，恒定
  t = (worker + k*64) // 64 = k         ← k = 0..127
```

**lane w 负责第 w 个头，把 128 行 query 从头走到尾。** 64 条 lane 各管一个头。

同一个任务在 decode 里（`num_tokens = 32`、不封顶）：

```text
blocks  = 32 * 64 = 2048
workers = min(2048, 2^31-1) = 2048
内层轮数 = 1       ← 每个 block 只做一个 (token, 头)
```

2048 个 block 只有 64 条 AIV lane 可跑，所以**调度器**分 32 轮发下去。

## 两张任务表

### decode（`num_tokens = 32`，`LOCAL_H = 64`，`M_TILE = 16`）

| 任务名 | 工作量公式 | 上限 | 宽度 |
| ------ | ---------- | ---- | ---: |
| attention_q_a | 常量 | — | 10 |
| attention_q_norm | ⌈32/8⌉ | 无 | 4 |
| attention_q_b | 常量 | — | 256 |
| attention_q_rope | 32 token × 64 头 | 无 | **2048** |
| attention_kv | 常量 | — | 4 |
| attention_kv_norm | ⌈32/8⌉ | 无 | 4 |
| attention_kv_rope | 32 token × 1 头 | 无 | 32 |
| swa_cache_publish | `num_tokens` | — | 32 |
| swa_cache_gather | 32 × 8 窗口组 | — | 256 |
| swa_online_attention | 32 × (64 头 ÷ 16) | — | 128 |
| attention_o_rope | 32 token × 64 头 | 无 | **2048** |
| attention_grouped_o_a | — | — | 128 |
| attention_o_b | 常量 | — | 40 |

这 13 个宽度把实测直方图里的 `4×3 / 10 / 32×2 / 40 / 128×2 / 256×2 / 2048×2`
全部对上了，所以 `num_tokens = 32` 是确定的。

**那两个 2048 是 q_rope 和 o_rope**，`token × 头`，不是窗口 gather。

### prefill（`num_tokens = 257`，每 chunk `active = 128`）

| 任务名 | 工作量公式 | 上限 | 宽度 |
| ------ | ---------- | ---- | ---: |
| prefill_swa_begin | `WORKER_TILE` | — | **64** |
| attention_kv | 常量，主机循环 ⌈257/32⌉ 次 | — | 4 |
| prefill_attention_kv_norm | ⌈257/8⌉ = 33 | min(·,64) | 33 |
| prefill_attention_kv_rope | 257 × 1 头 | min(·,64) | **64** |
| prefill_swa_cache_publish | `WORKER_TILE` | — | **64** |
| prefill_swa_stage | `WORKER_TILE` | — | **64** |
| attention_q_a | 常量 | — | 10 |
| prefill_attention_q_norm | ⌈128/8⌉ | 无 | 16 |
| attention_q_b | 常量 | — | 256 |
| prefill_attention_q_rope | 128 行 × 64 头 = 8192 | min(·,64) | **64** |
| prefill_swa_cache_gather | `WORKER_TILE` | — | **64** |
| prefill_swa_online_attention | 常量 | — | 32 |
| prefill_attention_o_rope | 128 × 64 头 = 8192 | min(·,64) | **64** |
| prefill_grouped_o_a_cube / cast | ⌈128/16⌉ × 64 | — | 512 |
| attention_o_b | 常量 | — | 40 |
| prefill_swa_collect | `WORKER_TILE` | — | **64** |
| prefill_tp_publish / reduce | `WORKER_TILE` | — | **64** |

`prefill_attention_kv_norm = 33` 这个怪数正好是 `⌈257/8⌉`，反过来坐实了
`num_tokens = 257`。

### 并排

同一个 `make_rope` 工厂，同一个 `blocks = num_tokens × heads`：

| 配置 | 工作量 blocks | 任务宽度 workers | 内层深度 |
| ---- | ------------: | ---------------: | -------: |
| decode `attention_q_rope` | 32 × 64 = **2048** | **2048** | **1** |
| prefill `attention_q_rope` | 128 × 64 = **8192** | **64** | **128** |

**prefill 这个任务的工作量是 decode 的 4 倍，却被压成 64 宽、内层跑 128 轮。**

这就是"prefill 任务更多、decode 任务更宽"的全部内容。同一份活：

| 配置 | 调度器看到的 | 每项多大 | 谁来摊 |
| ---- | -----------: | -------- | ------ |
| decode q_rope | **2048 个 block** | 1 个工作单元 | **调度器**：哪条 lane 空了就拿下一个 |
| prefill q_rope | **64 个 block** | 128 个工作单元 | **编译期写死**：拿到的 lane 得跑完 128 轮 |

## 这些 block 是不是同时发射的：不是

`pl.spmd` 的 `sync_start` 默认 **False**（文档原话：If True, all blocks start
execution simultaneously）。实测确认：`prefill_attn_swa` 的 **95 个任务全部
`sync=0`**。整个 DeepSeek-V4.1-Flash 模型里只有一处用了它
（`prefill_attn_c2a_full.py` 的 `prefill_compressor_cube`）。

所以这些 64 块的任务是**零散发的**，不是同时起跑。两个调度器在这一点上一样；
差别在**一个核空出来的时候它能拿什么**：

| 调度器 | 空核能拿什么 |
| ------ | ------------ |
| legacy | AICPU 手里有全局就绪集合，空核拿任意就绪任务的下一块 → **work-conserving** |
| percounter | 一条 lane 只能拿**自己顺序表当前位置**上那一块，抽到就在那儿自旋等就绪，不能跳过也不能退回 → **不是 work-conserving** |

percounter 其实实现了 `sync_start`：这种任务只占**一个**顺序表位置，走 cohort
队列，经过的 lane 各用一次 `fetch_add` 领一块，领走最后一块的那条宣布释放，然后
所有持有者才开跑。也就是说它的语义字面上就是"等齐 `block_num` 条 lane 再一起跑"。

## 为什么 prefill 这么写

**没有找到说明。** 两个常量旁边没有注释，`git log -S` 只找到一次纯搬运的重构提交。
下面把能查证的和推断的分开。

**能查证的一：没有硬限制逼它这么做。** simpler 对每任务块数的上限是
`logical_block_num ≤ 65535 / subtasks`（`host/runtime_maker.cpp`）。prefill 已经按
`QUERY_TILE = 128` 分块，所以 q_rope 不封顶也只有 8192 块，合法。**64 是选的，不是
被逼的。**

**能查证的二：64 对 legacy 恰好是最优。** 把所有 prefill 侧的 worker 上限改掉，
同一台机器、同一次会话、干净二进制实测 `prefill_attn_swa`：

| 上限 | legacy | percounter | 缺口 |
| ---: | -----: | ---------: | ---: |
| 16 | 17.976 ms | 23.241 ms | −22.65% |
| **64**（原样） | **16.313 ms** | 19.322 ms | −15.57% |
| 128 | 16.694 ms | 19.211 ms | −13.10% |
| 256 | 17.512 ms | **19.130 ms** | −8.46% |

legacy 在 64 上取到最小值：窄了浪费 lane，宽了多付 AICPU 派发，**正好 64 时一块一个
AIV lane、一次填满、派发次数最少**。这个常量看起来是照着 AICPU 派发器调出来的。

**推断（只有结构上的合理性，没有动机证据）：** prefill 的行数大且可变
（`PREFILL_MAX_TOKENS = 4096`），而且因为 scratch 的尺寸本来就必须按行分块流式处理；
既然已经有了一层分块，再把 worker 数封顶、余量塞进 block 内循环是顺手的，而且能让
任务数和每任务的簿记量不随 `num_tokens` 增长。decode 的行数又小又固定
（`DECODE_MAX_TOKENS` 是两位到三位数），没有东西可流，一个工作单元一个 block 是最
自然的写法。

**要坐实动机只有一条路**：问模型库的作者，或翻引入这两个常量的原始 PR。

## 复现

任务名和块数，从生成的编排代码读：

```bash
# 跑一次模块，日志里会打出 work_dir
CPP="$WORK_DIR/next_levels/$RANK/orchestration/$RANK.cpp"
awk '
  /\/\/ (Spmd|Group) / { n=$0; sub(/.*\/\/ (Spmd|Group) /,"",n); sub(/:.*/,"",n); pending=n }
  /set_core_num\(/ { if (pending!="") { c=$0; sub(/.*set_core_num\(/,"",c); sub(/\).*/,"",c);
                     printf "%-36s %s\n", pending, c; pending="" } }
' "$CPP"
# 表达式名（workers_inlineNN / blocks_inlineNN）的定义：
grep -nE 'int64_t (workers|blocks)_inline[0-9]+ =' "$CPP"
# 主机侧循环：
grep -nE '^ *for \(int64_t' "$CPP"
```

宽度直方图和 `sync=` 需要带 `EXPG` 的插桩构建；注意**插桩会让 legacy 慢约 15%、
percounter 几乎不动**，所以时间不能和干净二进制的数混着读
（见 [swa 那一页](swa-why-no-order-helps.md)的说明）。

宽度扫描：把模型库副本里的 `_PREFILL_WORKERS`（两处）和 `WORKER_TILE`（一处）改成
目标值，每个宽度跑 legacy 和 percounter 两个配置，同一次会话同一块卡。
