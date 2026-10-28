# percounter 调度器代码走读

本文面向第一次读 percounter 代码的人：先给文件结构和整体流程，再按文件、按功能给出伪代码和对应代码块。

所有代码链接都固定在提交
[`9769943a5`](https://github.com/deciding/simpler-percounter/commit/9769943a58f4d7580dd23e6f6eff93067e7e9bbc)
（`perf/aic-gap` 分支），链接文字就是 `文件路径#L起-L止`，可以直接在本地按路径和行号翻到同一位置。之后代码再改，链接仍指向这个版本。

## 1. 一句话概述

percounter 是 a5 `host_build_graph`（HBG）运行时的第三种调度器，与 legacy（AICPU 派发）、resident（每个 cluster 一个调度核）并列。它的核心思想：

- **Host 在 bind 时把整张图编译成表**：每个任务一个 64 B 计数器、每种核类型一个按 b-level 排好序的"顺序列表"、每个任务的生产者地址表。
- **AICore 上每条 lane 自己调度自己**：从票号（ticket）领一段列表位置 → 轮询生产者计数器直到"已发布" → 跑 kernel → 对自己的计数器做一次原子加。没有调度核、没有就绪队列、没有 AICPU 派发。
- **AICPU 几乎不参与调度**：只做启动握手、发布拓扑、看进度计数器（看门狗），以及 Graph 任务体的按需展开。

三层分工：

```text
Host (bind)                          AICPU (运行时)                       AICore lanes (运行时)
-----------------------------------  -----------------------------------  ------------------------------------
选择是否走 percounter                握手, 发布 cluster 拓扑               报到 -> 等 READY -> 等寄存器放行
规划 GM 区域布局                     给每个已报到的核发 READY              lane loop: 领号 -> 等依赖 -> 跑 -> 发布
填表: 计数器/顺序列表/fanin/控制块    非监督线程: 展开 Graph body            MIX 会合 / sync-start cohort
上传, 发布 bootstrap(mode=5)         监督线程: 看 resolved_task_count       退出: 等 EXIT, 应答 COND nonce
```

## 2. 文件结构

percounter 自有代码全部在 `simpler::hbg::percounter` 命名空间下、各层的 `percounter/` 子目录中，对上游文件只有少量钩子。

```text
src/a5/runtime/host_build_graph/
├── runtime/scheduler/percounter/
│   └── percounter_types.h          (691 行) 三层共用的 GM 线结构体与常量（Host/AICPU/AICore 都编译它）
├── host/percounter/                 Host 侧（只在 host 编译，可用 STL）
│   ├── percounter_selection.h      ( 70 行) 解析环境变量 SIMPLER_HBG_SCHEDULER_TYPE
│   ├── percounter_state.h          (302 行) bind 时的总入口：判定是否接管、收集输入、规划、填表、发布
│   ├── percounter_plan.h           (686 行) GM 区域布局规划 plan_layout + 填表 build_tables
│   └── percounter_graph.h          (230 行) Graph Definition -> 模板 (HostGraphTemplate)
├── aicpu/percounter/                AICPU 侧
│   ├── percounter_aicpu.h          (194 行) 拓扑发布、READY、监督线程、辅助线程入口
│   └── percounter_graph_prepare.h  (278 行) AICPU 线程按优先级展开 Graph body 并发布
└── aicore/percounter/
    └── percounter_executor.h      (1426 行) AICore lane：领号、等依赖、跑 kernel、MIX、cohort、退出

被改动的上游文件（只放钩子，见第 3 节）:
src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp
src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp
src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.h
src/a5/runtime/host_build_graph/host/runtime_maker.cpp
src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_layout.h
src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_topology.h
src/common/host_build_graph/runtime.h
docs/dfx/profiling-config-naming.md

测试:
tests/ut/cpp/a5/runtime/host_build_graph/
├── test_hbg_percounter_selection.cpp      环境变量解析
├── test_hbg_percounter_tables.cpp         领号/分组/布局/填表/排序
├── test_hbg_percounter_graph.cpp          Graph 模板与 Graph 填表
└── test_hbg_percounter_graph_prepare.cpp  AICPU 展开 Graph body
tests/st/a5/host_build_graph/
├── spmd_blocks/                 单槽 SPMD（多 block）与 N 路 join
├── mix_rendezvous/              MIX 会合（含多 block MIX）
├── sync_cohort/                 sync-start cohort（kernel 内真实屏障，缺 lane 即挂）
└── sync_cohort_interleaved/     两种 cohort 同时在途的回归测试
```

所有文件都是 header-only，原因是上游的单元测试按文件名编译 `aicore_executor.cpp` / `aicpu_executor.cpp` / `runtime_maker.cpp`，新增 `.cpp` 就得去改每个 UT 目标。

## 3. 对上游文件的改动（钩子）

| 上游文件 | 改动 | 位置 |
| -------- | ---- | ---- |
| `scheduler_layout.h` | 新增运行模式常量 `SCHEDULER_RUNTIME_MODE_PERCOUNTER = 5` | [src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_layout.h#L23-L25](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_layout.h#L23-L25) |
| `runtime.h` | `worker_context_base` 的注释：percounter 下它指向 `PercounterControl` | [src/common/host_build_graph/runtime.h#L196-L212](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/common/host_build_graph/runtime.h#L196-L212) |
| `scheduler_topology.h` | `scheduler_cluster_coordinate_from_worker` 加 `__aicore__`，让 lane 能算出自己所在的 cluster | [src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_topology.h#L26](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_topology.h#L26) |
| `runtime_maker.cpp` | bind 开头校验环境变量 | [src/a5/runtime/host_build_graph/host/runtime_maker.cpp#L1137](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/runtime_maker.cpp#L1137) |
| `runtime_maker.cpp` | 请求 percounter 时，不因 GRAPH 任务提前退回 legacy | [src/a5/runtime/host_build_graph/host/runtime_maker.cpp#L1144-L1152](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/runtime_maker.cpp#L1144-L1152) |
| `runtime_maker.cpp` | 默认选择的逐任务遍历跳过 GRAPH 任务并计数 | [src/a5/runtime/host_build_graph/host/runtime_maker.cpp#L1173-L1177](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/runtime_maker.cpp#L1173-L1177) |
| `runtime_maker.cpp` | 遍历结束后调用 `select_percounter_state`，接管则直接返回 | [src/a5/runtime/host_build_graph/host/runtime_maker.cpp#L1301-L1315](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/runtime_maker.cpp#L1301-L1315) |
| `aicpu_executor.cpp` | `init`：握手后的初始化换成 `percounter_post_handshake_init` | [src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L247-L293](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L247-L293) |
| `aicpu_executor.cpp` | `init`：发 READY 换成 `percounter_publish_partition`，并跳过 resident 的 bootstrap 等待 | [src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L308-L325](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L308-L325) |
| `aicpu_executor.cpp` | `run`：最后一个线程当监督线程 `percounter_supervise` | [src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L410-L419](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L410-L419) |
| `aicpu_executor.cpp` | `run`：其余线程做 `percounter_assist`（展开 Graph body）后等待关闭 | [src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L534-L540](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L534-L540) |
| `aicpu_executor.cpp` | `aicpu_execute`：percounter 模式不走 legacy 分支 | [src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L720](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L720) |
| `aicore_lifecycle.h` | 新增三个只读访问器：`core_count` / `core_type` / `core_reported` | [src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.h#L62-L68](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.h#L62-L68) |
| `aicore_executor.cpp` | `aicore_execute`：mode 为 5 时转入 `percounter_aicore_execute` | [src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp#L724-L727](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp#L724-L727) |
| `profiling-config-naming.md` | 登记环境变量 `SIMPLER_HBG_SCHEDULER_TYPE` | [docs/dfx/profiling-config-naming.md#L134](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/docs/dfx/profiling-config-naming.md#L134) |

环境变量未设置时，上面所有钩子都不生效：Host 不会选 mode 5，AICPU/AICore 只在 mode 5 下进入 percounter 分支。

## 4. 端到端流程

```text
# ---------- Host: bind (runtime_maker.cpp::create_scheduler_state) ----------
check_scheduler_request()                     # 环境变量拼错 -> bind 失败
默认选择的逐任务遍历 (上游代码)                # 分类任务形状、校验 kernel id、找出 inline 完成的任务
                                              # 请求 percounter 时跳过 GRAPH 任务
select_percounter_state(...)                  # percounter_state.h
    未请求 percounter             -> DEFAULT  (继续上游的 resident/legacy 选择)
    有 percounter 不支持的任务形状 -> DEFAULT  (打日志退回)
    gather_graph_inputs           # 每个 Definition 建一个模板, 每个 GRAPH 任务一个 invocation
    逐任务统计各列表条目数、边数
    plan_layout                   # 算 GM 区域每段的偏移
    acquire_scheduler_state_storage
    build_tables                  # 填控制块、计数器、fanin、顺序列表、Graph 模板和 invocation
    初始化每个 lane 的 DispatchPayload
    publish_scheduler_bootstrap(mode=5, control 地址)
    add_pending_metadata(上传前 upload_size 字节)

# ---------- AICPU: init ----------
握手 (上游)
leader: percounter_post_handshake_init      # 数 AIC/AIV, 写 cluster_count, aiv_per_cluster, 超时
每个线程: percounter_publish_partition       # 给自己那一片已报到的核写 aicpu_ready = 5
(跳过 resident 的 bootstrap 等待)
释放 DATA_MAIN_BASE 门 (上游)

# ---------- AICPU: run ----------
最后一个线程: percounter_supervise           # 轮询 resolved_task_count 直到 == executable_task_count
其余线程:     percounter_assist              # 抢着展开 Graph body, 然后等 shutdown_ready
发 EXIT, 收 COND nonce (上游)

# ---------- AICore: 每个核 ----------
percounter_aicore_execute
    报到 -> 等 aicpu_ready==5 -> 等 DATA_MAIN_BASE==IDLE
    run_percounter_lane -> percounter_lane_loop
    wait_for_aicpu_exit_signal; 应答 COND nonce
```

## 5. 共用线结构 `percounter_types.h`

这个头文件被三层都编译，只放 POD 结构体和 `__aicore__` 内联函数。GM 区域内所有引用都是"相对区域基址的偏移"，所以整块区域一次 memcpy 就能上 device。

### 5.1 GM 区域布局

由 `plan_layout` 按下面顺序保留（见 6.3），从前往后：

```text
PercounterControl            控制块 (384 B), 所有段的偏移都在这里
PercounterTicket[2][16]      每种核类型 16 张票 (按 8 条 lane 一组)
uint64 order[AIC]            AIC 顺序列表
uint64 order[AIV]            AIV 顺序列表
[有 MIX 时] MIX 列表, MIX 两张票(draw/done), 每 cluster 一个 ready 字
[有 sync-start 时] PercounterSyncQueue, 描述符环 + 每 cohort 的 claim 计数
[有 GRAPH 时] PercounterGraphInvocation[], PercounterGraphPrepare, 模板 blob
PercounterFanin[table_task_count]   每个任务的 (生产者个数, 起始下标)
uint64 fanin_addr[edge_count]       每条边: 生产者计数器的 device 地址
DispatchPayload[worker 容量]         每个 lane 一个 payload
PercounterCounter[task_count]        每个任务一个计数器 (最后一段; Graph body 的计数器不上传)
```

### 5.2 计数器协议

- 结构体 `PercounterCounter`（每个 64 B 独占一个 cache line）：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L139-L165](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L139-L165)
- 发布判定与预置值：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L167-L187](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L167-L187)

```text
preset(total_subtasks) = 1 - total_subtasks       # host 写
每个子任务跑完:  fetch_add(counter, 1)              # lane 写; MIX 的最低参与者一次加 popcount(mask)
published(v)   = (int64)v > 0                      # 消费者读; 恰好走到 1
inline 完成的任务: 直接预置为 1
```

用"有符号 > 0"而不是"== 1"：如果某个子任务被错误地跑了两次，表现为结果错，而不是消费者永远等不到。

计数器的后 56 字节只给 Graph body 任务用（`storage_address`、`body_index`、`flags`、`shell_task_id`、`fanin_count`、`fanin_offset`），由 AICPU 在发布条目前写好，lane 只需读这一个 line 就能跑这个任务。

### 5.3 顺序列表条目

- 条目编码、解码：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L439-L482](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L439-L482)
- cohort 的"武装条目"标记 `PERCOUNTER_ORDER_BLOCK_IDX_ARM`：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L484-L497](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L484-L497)

```text
entry (64 bit) = task_id[0..31] | block_idx[32..47] | block_num[48..63]
block_num == N 的 SPMD 任务占 N 个连续条目, block_idx = 0..N-1
sync-start 任务只占 1 个条目, block_idx = 0xffff (武装条目)
block_num == 0 的条目 = 尚未发布 (Graph body 的区间在 AICPU 展开前全为 0)
```

一个 64 位读就拿到 lane 需要的全部信息，避免在就绪扫描时多一次 device 读。

### 5.4 控制块 `PercounterControl`

[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L499-L600](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L499-L600)

要点：

- 前两个 128 B line 是只读的描述（各段偏移和长度、外部地址、拓扑、超时），lane 启动时读一遍。
- `cluster_count` / `aiv_per_cluster` 由 AICPU 握手后写，host 在 bind 时不知道 cluster 数。
- `lane_error`：第一个错误胜出（CAS 0 → 错误码），监督线程看到非 0 就停。
- `resolved_task_count` **独占最后一个 line**：它是运行期间唯一被 lane 原子写的字段。旁边放一个每任务都读的字段，曾让 paged_attention Case1 每任务多花 919 cycle。

### 5.5 领号（ticket）与分组

- 票的容量和分组大小：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L121-L137](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L121-L137)
- 每次领号的批量 `PERCOUNTER_CLAIM_BATCH = 4`、长 kernel 阈值、短列表阈值：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L602-L634](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L602-L634)
- 组内下标 → 列表位置、组的份额大小：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L636-L657](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L636-L657)
- 一次领号变成区间，以及无死锁论证：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L659-L689](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L659-L689)

```text
G = clamp(该类型 lane 数 / 8, 1, 16)              # 组数
lane 的组 g = cluster_index % G
列表按 4 个位置切块, 第 c 块属于组 c % G
组 g 的第 i 个下标 -> 列表位置 ((i/4)*G + g)*4 + i%4

领号:  base = fetch_add(tickets[g].next, batch)
       base >= group_size(g) -> 该组已领完, 换下一组 (按顺序, 不回头)
       否则得到区间 [base, min(base+batch, group_size))
```

例：AIC 有 32 条 lane → G = 4。列表位置 0-3 归组 0，4-7 归组 1，8-11 归组 2，12-15 归组 3，16-19 又归组 0……一组 8 条 lane 争一张票，而不是 32 条争一张。

### 5.6 MIX 会合信号（SSBUF）

- `PercounterMixSignal`，每个 cluster 一个 line，放在 SSBUF 里：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L38-L94](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L38-L94)
- 清空与"是否已武装"：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L96-L119](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L96-L119)
- MIX 的两张票（draw / done）：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L189-L205](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L189-L205)

SSBUF 没有原子操作，所以每个字段只有一个写者：`task_id` / `armed_generation` / `active_mask` / `block_*` / `cohort_block` 只由本 cluster 的 AIV0（claimer）写；`arrived[i]` / `finished[i]` 只由 cluster 内第 i 条 lane 写。`armed_generation` 单调递增，奇数表示已武装。

### 5.7 sync-start cohort 队列

- `PercounterSyncQueue`（5 个独立 line）：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L207-L264](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L207-L264)
- 描述符环与每 cohort 独立的 claim 计数：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L266-L315](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L266-L315)

两条规则：

1. **同一时刻只有"open"的那一个 cohort 可被领 block**。这样不会出现两个 cohort 各占一部分 lane、谁都凑不齐的死锁。
2. **cohort 一被领满就让出 open 槽**，不等它执行完。下一个 cohort 可以立刻开始收集，持有 block 的 lane 通过 `released > my_seq` 得知自己的 cohort 已凑齐。

### 5.8 Graph 相关结构

- body 任务、标志位（ROOT / SINK / PREDICATE）：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L317-L338](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L317-L338)
- `PercounterGraphTemplate`（每个 Definition 一份）：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L340-L375](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L340-L375)
- `PercounterGraphInvocation`（每个 GRAPH 任务一份）：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L377-L397](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L377-L397)
- `PercounterGraphPrepare`（AICPU 线程间共享的展开进度）与切片大小：[src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L399-L429](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/runtime/scheduler/percounter/percounter_types.h#L399-L429)

编号规则：任务表里的任务占 percounter id `0..table_task_count-1`；Graph body 任务接在后面，invocation k 的 body 任务 i 的 id 是 `body_base[k] + i`。GRAPH 任务本身（shell）不跑 kernel，它的计数器按"body 中所有 sink 子任务数"预置，由 sink 任务发布。

## 6. Host 侧

### 6.1 环境变量 `percounter_selection.h`

- 变量名：[src/a5/runtime/host_build_graph/host/percounter/percounter_selection.h#L29-L35](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_selection.h#L29-L35)
- 解析与读取：[src/a5/runtime/host_build_graph/host/percounter/percounter_selection.h#L37-L68](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_selection.h#L37-L68)

```text
未设置 或 ""     -> DEFAULT
"percounter"    -> PERCOUNTER
其他任何值       -> 打错误日志, bind 失败   (拼错不能悄悄退回默认调度器)
```

### 6.2 bind 总入口 `percounter_state.h`

| 功能 | 位置 |
| ---- | ---- |
| `percounter_requested` / `log_fallback` / `check_scheduler_request` / `percounter_takes_graph_tasks` | [src/a5/runtime/host_build_graph/host/percounter/percounter_state.h#L53-L77](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_state.h#L53-L77) |
| `GraphInputs` + `gather_graph_inputs`：每个 Definition 建模板，每个 GRAPH 任务建 invocation | [src/a5/runtime/host_build_graph/host/percounter/percounter_state.h#L79-L137](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_state.h#L79-L137) |
| `StateSelection`（DEFAULT / PERCOUNTER / FAILED） | [src/a5/runtime/host_build_graph/host/percounter/percounter_state.h#L139-L143](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_state.h#L139-L143) |
| `select_percounter_state` 全函数 | [src/a5/runtime/host_build_graph/host/percounter/percounter_state.h#L145-L300](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_state.h#L145-L300) |

`select_percounter_state` 伪代码：

```text
if !requested: return DEFAULT
标出 inline 完成的任务
gather_graph_inputs                                  # L167-L182
    MALFORMED   -> FAILED  (Definition 自相矛盾)
    UNSUPPORTED -> DEFAULT (body 里有 percounter 不跑的形状)
for 每个任务 t:                                       # L191-L231
    if GRAPH 任务: 累加其模板各列表长度、cohort 数; block_total += 1
    elif 非 inline 完成:
        placement = task_placement_from_active_mask(active_mask)
        UNSUPPORTED (AIV1 单独 / 无子任务) -> DEFAULT
        条目数 = sync_start ? 1 : block_num
        MIX -> mix 列表; 否则 -> AIC/AIV 列表
    记录 fanin 边
plan_layout                                          # L235-L242
acquire_scheduler_state_storage                      # L243-L253
build_tables                                         # L255-L270
初始化每个 lane 的 DispatchPayload: args 指向自己的 local/global context   # L272-L285
publish_scheduler_bootstrap(mode=5, control 的 device 地址)                # L287
add_pending_metadata(只上传前 upload_size 字节)                            # L288-L290
return PERCOUNTER
```

### 6.3 布局与填表 `percounter_plan.h`

| 功能 | 位置 |
| ---- | ---- |
| `PercounterLayout`：区域内每段的偏移 | [src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h#L33-L77](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h#L33-L77) |
| Graph 相关的 host 输入结构 | [src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h#L79-L137](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h#L79-L137) |
| `plan_layout` | [src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h#L139-L229](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h#L139-L229) |
| `task_placement_from_active_mask`：按 active mask 决定进哪个列表 | [src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h#L231-L268](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h#L231-L268) |
| `TaskInput`：填表器看到的每个任务 | [src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h#L270-L302](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h#L270-L302) |
| `BuildStatus` / `BuildResult` | [src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h#L304-L340](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h#L304-L340) |
| `build_tables` 全函数 | [src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h#L347-L684](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h#L347-L684) |

`plan_layout` 伪代码：

```text
task_count = table_task_count + graph.body_task_count
校验: 各列表条目数之和 <= block_total, ...
依次 reserve (对齐 + 溢出检查), 顺序见 5.1                   # L195-L224
upload_size = counters_offset + table_task_count * 64       # L225, body 计数器不上传
```

`task_placement_from_active_mask`：mask 1 → AIC；mask 2 → AIV；两到三个位 → MIX；0 或只有 AIV1（mask 4）→ 不支持。只有 AIV1 被拒是因为 AIV 列表被每个 cluster 的两条 AIV lane 共同领，任务落在哪条 lane 不确定，`sub_block_id` 会和 kernel 预期不符。

`build_tables` 伪代码：

```text
memset(region[0:upload_size], 0)
为每个 invocation 分配 body_base (接在任务表 id 之后)                       # L408-L421
for id in 0..table_task_count:                                           # L431-L507
    写 fanin[id] = (count, begin); 每条边写 fanin_addr = 生产者计数器 device 地址
        校验: 生产者 id < 消费者 id (hbg 中 id 升序就是拓扑序, 这里逐边检查)
    inline 完成  -> counter = 1, 不进列表
    GRAPH 任务   -> counter = preset(模板 sink 子任务总数); 在模板用到的每个列表里占一个位置
    off-lane    -> counter = preset, 不进列表
    普通任务     -> counter = preset(block_num * subtask_count)
                  条目数 = sync_start ? 1 : block_num, 进 MIX / AIC / AIV 列表
计算 b-level (到 sink 的最长路径), id 倒序一遍即可                          # L513-L528
    GRAPH 任务额外加 (body 深度 - 1)
每个列表: 按 b-level 降序稳定排序 (同级按 id), 再展开成条目                  # L530-L579
    GRAPH 任务: 只记录 order_base, 区间留 0 (等 AICPU 发布)
    sync-start: 一个武装条目 (block_idx = 0xffff)
    其他:       block_num 个条目
拷贝模板到区域                                                           # L582-L620
写 invocation 记录, 按 shell 的 b-level 排序 (AICPU 先展开最急的)            # L621-L641
填控制块各字段; sync 队列 open_task_id = -1; 超时用编译期默认值              # L643-L683
```

**为什么按 b-level 而不是按深度排序**：b-level 是"后面还挂着多少工作"。按深度排序时，paged_attention Case1 会把 16384 个 QK 全排在第一个 PV 前面，AIV 池空转。按 b-level 降序，每一层的工作同步交给两个池。

**为什么无死锁**：生产者的 b-level 严格大于消费者，所以在全局键（-b-level, id）下生产者一定排在消费者前面。取未完成任务中键最大的 T：它的生产者和它列表中前面的位置都已完成；如果 T 已被某条 lane 领走，那条 lane 没有其他需要等待的；如果还没被领，它所在组的票已经走到它，下一个从该组领号的 lane 就会拿到 T。完整论证见 `percounter_claim_range` 的注释（5.5 节链接）。

### 6.4 Graph 模板 `percounter_graph.h`

| 功能 | 位置 |
| ---- | ---- |
| `BodyTaskShape`、`TemplateStatus`、`TemplateResult` | [src/a5/runtime/host_build_graph/host/percounter/percounter_graph.h#L37-L57](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_graph.h#L37-L57) |
| `build_graph_template`：从 body 的形状和 fanin 建模板 | [src/a5/runtime/host_build_graph/host/percounter/percounter_graph.h#L59-L184](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_graph.h#L59-L184) |
| `read_graph_template`：用上游的解码器读 Definition | [src/a5/runtime/host_build_graph/host/percounter/percounter_graph.h#L186-L228](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/host/percounter/percounter_graph.h#L186-L228) |

```text
read_graph_template(staging, definition_offset):
    graph_definition_decode_framed                 # 上游解码器, 设备会拒绝的 Definition 这里也拒绝
    逐个读 SubTaskDefinition -> BodyTaskShape (active_mask, sync_start, predicate, block_num, 子任务数)
    读 fanin_offsets / fanin_indices
    build_graph_template(...)

build_graph_template:
    校验 fanin 行, 每条边必须指向更小的下标                        # L78-L101
    for 每个 body 任务 i:                                         # L103-L154
        active_mask == 0 (只表达依赖) -> 放进 AIV 列表, 当作 1 个子任务
        AIV1 单独 -> UNSUPPORTED;  predicate 只允许单 block 单子任务
        subtask_total = block_num * popcount(mask)
        无 fanin -> ROOT;  无消费者 -> SINK (累加 sink_subtask_total)
        MIX -> MIX 列表;  否则 -> AIC/AIV 列表
    body 内的 level (到 body sink 的最长链), depth = max(level)+1  # L156-L165
    每个列表按 level 降序稳定排序, 再展开成条目 (id 用 body 下标)   # L167-L181
```

模板只保存调度需要的东西；kernel id、参数、predicate 的实际物化仍由上游在 AICPU 上按 invocation 做。

## 7. AICPU 侧

### 7.1 `percounter_aicpu.h`

| 功能 | 位置 |
| ---- | ---- |
| `percounter_mode_enabled` / `percounter_control` | [src/a5/runtime/host_build_graph/aicpu/percounter/percounter_aicpu.h#L48-L57](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/percounter/percounter_aicpu.h#L48-L57) |
| `percounter_post_handshake_init`：发布拓扑（leader） | [src/a5/runtime/host_build_graph/aicpu/percounter/percounter_aicpu.h#L59-L101](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/percounter/percounter_aicpu.h#L59-L101) |
| `percounter_publish_partition`：给自己那片核发 READY | [src/a5/runtime/host_build_graph/aicpu/percounter/percounter_aicpu.h#L103-L125](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/percounter/percounter_aicpu.h#L103-L125) |
| `percounter_assist`：非监督线程展开 Graph body | [src/a5/runtime/host_build_graph/aicpu/percounter/percounter_aicpu.h#L127-L128](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/percounter/percounter_aicpu.h#L127-L128) |
| `percounter_supervise`：监督线程 | [src/a5/runtime/host_build_graph/aicpu/percounter/percounter_aicpu.h#L130-L192](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/percounter/percounter_aicpu.h#L130-L192) |

```text
percounter_post_handshake_init:
    数已报到的 AIC / AIV; 要求 AIV == AIC * 2, 否则失败 (拓扑不完整)
    control.cluster_count = AIC 数; control.aiv_per_cluster = 2; control.timeout = 覆盖值
    flush; percounter_graph_prepare_init (清掉 AICPU 缓存里上一轮的旧行)

percounter_publish_partition(thread_idx):
    for 本线程负责的 [lo, hi) 中已报到的核: handshake.aicpu_ready = 5
    flush; wmb

percounter_supervise:
    只有一个 AICPU 线程时, 自己先展开 Graph body
    expected = executable_task_count
    loop:
        resolved = control.resolved_task_count (invalidate 后读)
        有变化 -> 更新最后进展时间
        resolved >= expected -> 成功退出
        每 64 次: 查 lane_error、查 Graph 展开错误
        超过 timeout 没有进展 -> TIMEOUT
    打印 Graph 展开时间; 有错则 latch 到 sched_error_code
```

AICPU 不调度任何任务。监督的目标是 `executable_task_count`（lane 实际跑的 block 数）：inline 完成的任务不在任何列表里，没有 lane 会计它。

### 7.2 Graph body 展开 `percounter_graph_prepare.h`

| 功能 | 位置 |
| ---- | ---- |
| `percounter_graph_prepare_init`：清 AICPU 缓存中会被部分重写的行 | [src/a5/runtime/host_build_graph/aicpu/percounter/percounter_graph_prepare.h#L53-L105](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/percounter/percounter_graph_prepare.h#L53-L105) |
| `percounter_graph_prepare_fail`、`percounter_flush_body_task` | [src/a5/runtime/host_build_graph/aicpu/percounter/percounter_graph_prepare.h#L107-L126](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/percounter/percounter_graph_prepare.h#L107-L126) |
| `percounter_prepare_invocation`：展开一个 invocation | [src/a5/runtime/host_build_graph/aicpu/percounter/percounter_graph_prepare.h#L128-L222](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/percounter/percounter_graph_prepare.h#L128-L222) |
| `percounter_prepare_graphs`：多个 AICPU 线程抢 invocation | [src/a5/runtime/host_build_graph/aicpu/percounter/percounter_graph_prepare.h#L224-L244](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/percounter/percounter_graph_prepare.h#L224-L244) |
| 错误查询与一行耗时报告 | [src/a5/runtime/host_build_graph/aicpu/percounter/percounter_graph_prepare.h#L246-L276](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicpu/percounter/percounter_graph_prepare.h#L246-L276) |

```text
percounter_prepare_graphs:                      # 任何 AICPU 线程都可调用
    while 没有错误:
        rank = fetch_add(prepare.next, 1)       # invocation 记录已按优先级排好
        rank >= 总数 -> return
        percounter_prepare_invocation(rank)
        prepared += 1

percounter_prepare_invocation(rank):
    execution = graph_execution_localize(shell 的 slot, view)        # 上游: 在 shell 的堆里定位 body 存储
    invocation.storage_address = execution.task_storage; flush
    loop:
        graph_execution_materialize_slice(..., 16 个任务)             # 上游: 物化一片
        for 本片新物化的 body 任务 i:                                  # L170-L182
            flush 任务存储与参数
            counters[body_base+i] = preset(subtask_total), 并写 storage/body_index/flags/shell/fanin; flush
        for 每个列表 l:                                               # L184-L205
            把模板条目里 body 下标 < 已物化数 的最长前缀,
            加上 body_base 写进 invocation 在该列表的区间; flush
        全部物化 -> break
    校验每个列表都已全部发布
```

先写计数器、后写条目：lane 拿到一个条目就会读自己和生产者的计数器行，而生产者的 body 下标更小，一定在本片或更早的片里。

## 8. AICore 侧 `percounter_executor.h`

### 8.1 入口与启动

| 功能 | 位置 |
| ---- | ---- |
| `percounter_aicore_execute`：报到 → 等 READY → 等放行 → lane → 退出 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L1347-L1424](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L1347-L1424) |
| `run_percounter_lane`：从控制块读出 lane 的配置 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L1266-L1345](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L1266-L1345) |
| `LaneConfig`：lane 运行期间不变的配置 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L92-L139](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L92-L139) |
| lane 的可变状态：`PercounterSyncState` / `PercounterMixClaim` / `PercounterLaneState` | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L141-L187](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L141-L187) |
| 调优常量：完成批量 8、检查间隔 64、退避 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L58-L90](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L58-L90) |

```text
percounter_aicore_execute(block_idx, core_type):
    写 handshake 报到 (物理核号、核类型), dcci + dsb
    读 control.timeout (host 预填, 所以在 AICPU 写任何东西前就能限定等待)
    等 handshake.aicpu_ready == 5          (超时 -> lane_error = TIMEOUT)
    等 DATA_MAIN_BASE == AICPU_IDLE_TASK_ID (超时 -> lane_error = TIMEOUT)
    write COND = IDLE
    run_percounter_lane
    wait_for_aicpu_exit_signal; 应答 COND nonce   (与 resident 相同)

run_percounter_lane:
    用 cluster_count / aiv_per_cluster 算出 (cluster_index, cluster_lane)   # lane 0 = AIC, 1/2 = AIV0/AIV1
    所有字段都用 device load 读 (区域是 DMA 写的, 普通读可能拿到旧缓存)
    first_batch  = 列表长度 <= 64 * lane 数 ? 1 : 4                        # L1312-L1313
    组数 G       = clamp(lane 数 / 8, 1, 16); ticket_group = cluster_index % G   # L1314-L1320
    有 sync-start: 读队列地址;  有 MIX: 读 MIX 列表/票/ready 字/SSBUF, 并清 SSBUF 行
    percounter_lane_loop
```

### 8.2 lane 主循环

[src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L1108-L1264](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L1108-L1264)

```text
batch = first_batch; range_group = 自己的组; spent_groups = 0; pending = 0
loop:
    EXIT 或 lane_error -> break
    checkpoint_serve: 先服务已放行的 cohort block, 再看本 cluster 的 MIX 行     # L1169
    sync_take: 若有 open cohort 且本 lane 能服务, 领一个 block                  # L1171
    mix_tick: (仅 claimer) 释放已结束的会合, 领/武装下一个 MIX 任务             # L1173

    if pending == 0:                                                         # L1175-L1201
        从自己的组开始, 依次尝试每组的票:
            base = fetch_add(tickets[g].next, batch)
            落在该组份额内 -> 得到区间, break;  否则该组已空, 试下一组
        所有组都空 -> flush_done; mix_drain; break
        pending = 区间内每个位置一个 bit

    # 在区间内找第一个"已就绪"的位置, 而不是死等第一个                      # L1203-L1228
    for pending 中的每个位置 (从低到高):
        entry = order[group_position(...)]
        entry 未发布 (block_num == 0) -> 跳过
        resolve(task_id); 同一 SPMD 任务已证明就绪 或 fanin 全部已发布 -> 选中
    没找到 -> 取最低的 pending 位置, 等条目发布, 再等它的 fanin                 # L1229-L1239

    武装条目 -> sync_enqueue(把 cohort 放进队列), continue                     # L1244-L1247
    run_task(block_idx, block_num)                                           # L1249-L1255
    kernel 耗时 >= 3us ? batch = 1 : batch = 4                               # L1256-L1258
    publish_completion: counters[id] += 1 (sink 还要给 shell += 1)
    local_done 满 8 -> fetch_add(resolved_task_count, local_done)
```

区间内"先跑已就绪的"避免了区间被自己的第一个任务队头阻塞。阻塞时取最低位置是无死锁论证要求的：更高的位置可能依赖本区间中更低的位置，而那只有本 lane 能跑。

### 8.3 任务解析与运行

| 功能 | 位置 |
| ---- | ---- |
| `percounter_observe`：用 `ld_dev` 读跨核字 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L44-L56](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L44-L56) |
| `percounter_fanin_view` / `percounter_fail` | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L195-L215](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L195-L215) |
| `PercounterTask` + `percounter_resolve`：任务表任务 / Graph body 任务统一成一个视图 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L217-L286](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L217-L286) |
| `percounter_producer` / `percounter_fanin_ready`：非阻塞的一遍依赖检查 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L288-L307](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L288-L307) |
| `percounter_launch_kernel`：predicate、查 callable、物化 payload、跑 kernel | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L316-L375](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L316-L375) |
| `percounter_run_task` / `percounter_run_subtask` | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L377-L404](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L377-L404) |
| `percounter_publish_completion` / `percounter_should_stop` | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L406-L421](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L406-L421) |

```text
percounter_resolve(id):
    id < table_task_count:   graph = 任务表; fanin = fanin[id] 的生产者计数器地址
    否则 (Graph body 任务):  invalidate 并读 counters[id] 这一行
                            graph = invocation 的存储, local = body_index
                            ROOT -> 等 shell 的 fanin;  否则 -> fanin 是 body 下标 (加 body_base 得到计数器)
                            sink = flags & SINK

percounter_run_task:
    invalidate 整个 dcache (kernel 用普通读访问 GM)
    找到唯一非负的 kernel 槽 -> launch_kernel;  没有 kernel (只表达依赖) -> 直接算完成

percounter_launch_kernel:
    有 predicate 且为假 -> 不跑 kernel, 但仍算完成 (消费者照常放行)
    查 callable 地址; 用上游函数物化 payload 到本 lane 自己的 DispatchPayload
    写 block_idx / block_num / sub_block_id; writeback payload
    kernel(args); OUT_OF_ORDER_STORE_BARRIER   # 发布屏障: 输出先落地, 再原子加计数器
```

### 8.4 等待（阻塞路径）

| 功能 | 位置 |
| ---- | ---- |
| `percounter_wait_for_fanin`：等一个任务的所有生产者 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L1000-L1058](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L1000-L1058) |
| `percounter_wait_for_entry`：等 AICPU 发布 Graph body 的条目 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L1060-L1106](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L1060-L1106) |

```text
wait_for_fanin(task):
    for 每个生产者:
        while 未发布:
            checkpoint_serve (cohort block, 然后 MIX 行)  -- 每轮都做
            每 8 轮: sync_take                           -- 本 lane 可能正是 cohort 缺的那个
            服务过 -> 重置看门狗计时, continue
            连续 miss 16 次以上 -> 读 cycle 计数器空转 256 cycle (退避, 不是 sleep)
            每 64 轮: 查 EXIT / 错误, mix_tick, 看门狗
```

等待中仍然服务 MIX 和 cohort 是协议必需的：一个卡在生产者上的 lane，恰恰可能是某个会合或 cohort 在等的那条 lane。去掉这个检查点，`wide_240` 十次里死锁三次。

### 8.5 MIX 会合

| 功能 | 位置 |
| ---- | ---- |
| 概述与 claimer 角色（AIV0） | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L423-L432](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L423-L432) |
| `percounter_clear_mix_signal`：启动时清 SSBUF 行 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L434-L466](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L434-L466) |
| `percounter_mix_active_mask` / `percounter_mix_release` | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L468-L499](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L468-L499) |
| `percounter_mix_arm`：claimer 领 MIX 位置，就绪后武装本 cluster | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L706-L770](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L706-L770) |
| `percounter_mix_tick`：claimer 在每个检查点的职责 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L772-L791](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L772-L791) |
| `percounter_mix_wait`：等参与者的 token | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L815-L848](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L815-L848) |
| `percounter_serve_mix`：参与一次会合 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L850-L903](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L850-L903) |
| `percounter_mix_discover`：发现本 cluster 行上指名自己的任务 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L905-L935](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L905-L935) |

```text
# claimer (AIV0) 在每个检查点: mix_tick
mix_release: 行已武装 且 所有参与者 finished == generation -> armed_generation += 1 (变偶数, 空闲)
mix_arm (行空闲 且 本 lane 没持有 cohort block 时):
    没有持有的位置 -> draw = fetch_add(mix_tickets[DRAW], 1); 超出列表 -> list_spent
    读条目; 未发布 -> 下次再看
    resolve; fanin 未就绪 -> 保持持有, 下次再看     # 只宣布可立即运行的任务
    武装条目 -> sync_enqueue, 不武装本 cluster
    写 task_id / active_mask / block_* / cohort_block=0, 然后 release 写 armed_generation += 1 (奇数)

# 每条 lane 在每个检查点: mix_discover
gen = armed_generation (acquire); 未武装 或 已看过 -> return
读 payload; 再读 gen 校验 (seqlock)
不含本 lane -> 记为已看过, return
serve_mix:
    arrived[lane] = gen          # 不可撤销
    等所有参与者 arrived == gen
    跑本 lane 的子任务
    finished[lane] = gen
    若本 lane 是 mask 中最低位:
        等所有参与者 finished == gen
        counters[id] += popcount(mask)
        cohort block ? sync_queue.completed += 1 : mix_tickets[DONE] += 1
```

用 fetch_add 领 MIX 位置而不是 CAS 抢列表头：后者在硅上丢掉 83–95% 的就绪探测。

### 8.6 sync-start cohort

| 功能 | 位置 |
| ---- | ---- |
| 概述：一个 lane 先服务已放行的 block 再做别的 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L501-L517](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L501-L517) |
| `percounter_sync_can_serve`：哪类 lane 服务哪类 cohort | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L519-L528](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L519-L528) |
| `percounter_sync_promote`：把下一个排队的 cohort 提升到 open 槽 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L530-L557](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L530-L557) |
| `percounter_sync_enqueue`：武装条目把 cohort 入队 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L559-L569](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L559-L569) |
| `percounter_sync_take_block`：一次 fetch_add 领一个 block | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L571-L584](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L571-L584) |
| `percounter_sync_serve_block`：跑 block，或 MIX cohort 时武装本 cluster | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L586-L623](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L586-L623) |
| `percounter_sync_serve_released`：cohort 已放行则服务 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L625-L651](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L625-L651) |
| `percounter_sync_take`：决定是否领 open cohort 的 block | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L653-L704](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L653-L704) |

```text
sync_enqueue(task, block_num):         # lane 跑到武装条目、且 fanin 就绪时
    seq = fetch_add(queue.tail, 1); sync_pending[seq] = desc(task, block_num, mask)
    sync_promote

sync_promote:
    gen = queue.open_generation; 奇数 (有 open) -> return
    seq = gen / 2; seq >= tail 或 描述符还没写 -> return
    CAS(open_generation, gen, gen+1) 失败 -> return
    写 open_block_num / open_active_mask / open_seq, 最后写 open_task_id (读者以它为门)

sync_take (lane 没持有 block 时):
    gen = open_generation; 无 open 或 已对该 gen 做过决定 -> return
    读 open_task_id (<0 表示 payload 还没写完 -> 下次再看), open_seq, 形状; 再读 gen 校验 (seqlock)
    记下已决定; 本 lane 不在该 cohort 的服务池 -> return
    block = fetch_add(claims[seq], 1); 超出 block_num -> 丢弃
    持有 block, 记 taken 时间
    若是最后一个 block:
        open_task_id = -1; released = seq + 1; open_generation = gen + 1   # 先放行, 再让槽
        sync_promote; 立刻服务

sync_serve_released (每个检查点最先做):
    持有 block 且 released > my_seq -> serve_block
        单槽 cohort: 跑 block, counter += 1, completed += 1
        MIX cohort:  本 cluster 行空闲时武装 (cohort_block = 1), 由会合负责计数
    否则超时 -> TIMEOUT
```

claim 计数按 cohort 各自独立（`2 * cohort_count` 的环的后半段）：共享计数在提升时重置的话，迟到的 add 会吃掉下一个 cohort 的 block，导致它永远凑不满。

### 8.7 收尾与退出

| 功能 | 位置 |
| ---- | ---- |
| `percounter_lane_may_exit`：自己的列表领完后何时能走 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L793-L813](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L793-L813) |
| `percounter_checkpoint_serve` / `percounter_flush_done` | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L937-L954](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L937-L954) |
| `percounter_mix_drain`：列表领完后继续可达，直到所有 MIX / cohort 结束 | [src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L956-L998](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h#L956-L998) |

```text
lane_may_exit:
    持有 cohort block -> 不能走
    有 MIX 且 mix_tickets[DONE] < mix_rendezvous_count -> 不能走   # 数"已完成的会合", 不是"已领的位置"
    有 cohort 且 queue.completed < sync_block_total -> 不能走
    否则可以走

mix_drain:   # 图里没有 MIX 和 cohort 时直接返回
    loop:
        checkpoint_serve; mix_tick              # SSBUF 行是本 cluster 独有的, 每轮读
        每 64 轮: flush_done, 查 EXIT, sync_take, lane_may_exit, 看门狗(衡量空闲时长)
```

lane 退出循环后，回到 `percounter_aicore_execute` 等 AICPU 的 EXIT 信号，然后应答 COND nonce。

## 9. 关键不变量速查

| 不变量 | 由谁保证 | 参见 |
| ------ | -------- | ---- |
| 生产者 id < 消费者 id | `build_tables` 逐边检查 | 6.3 |
| 列表按 (-b-level, id) 全局有序，生产者严格在前 | `build_tables` 的稳定排序 | 6.3 |
| 领号区间不重不漏；只有本组领完才去别组 | `percounter_claim_range` + lane loop 的组顺序 | 5.5, 8.2 |
| 阻塞时只等区间内最低的未完成位置 | lane loop 的 fallback | 8.2 |
| 计数器从 `1 - total` 走到恰好 1，消费者看 `> 0` | `percounter_counter_preset` / `percounter_counter_published` | 5.2 |
| kernel 输出先于计数器原子加可见 | `OUT_OF_ORDER_STORE_BARRIER` | 8.3 |
| 跨核读用 `ld_dev`（`scheduler_gm_query`），不用 volatile 读 | `percounter_observe` | 8.3 |
| 区域是 DMA 写的，lane 启动时所有字段都用 device load | `run_percounter_lane` | 8.1 |
| SSBUF 每个字段只有一个写者 | MIX 信号设计 | 5.6 |
| 同一时刻只有一个 cohort 可领 block | cohort 队列 | 5.7, 8.6 |
| Graph body：先写计数器再写条目；条目为 0 表示未发布 | AICPU 展开 | 7.2 |
| 写热点各自独占 cache line（计数器、票、`resolved_task_count`） | `static_assert` | 5.2, 5.4 |

## 10. 测试

单元测试（`tests/ut/cpp/a5/runtime/host_build_graph/`，在 `CMakeLists.txt` 第 20–30 行注册）：

| 文件 | 覆盖内容 |
| ---- | -------- |
| [tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_percounter_selection.cpp](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_percounter_selection.cpp) | 环境变量：未设置/空串走默认，只接受 `percounter`，其他值报错 |
| [tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_percounter_tables.cpp](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_percounter_tables.cpp) | 领号区间升序不相交且全覆盖、分组份额按序划分列表（第 100 行）、mask 分类、布局对齐和 cache line 隔离、按 b-level 排序且生产者在前、MIX/SPMD/cohort 的条目展开、inline 完成的预置值、fanin 地址 |
| [tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_percounter_graph.cpp](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_percounter_graph.cpp) | 模板：ROOT/SINK 标记、body 内按 level 排序、MIX/cohort/只表达依赖的任务、拒绝不支持和自相矛盾的 Definition；Graph 填表：body 编号、区间留空、shell 预置、按 body 深度加权 |
| [tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_percounter_graph_prepare.cpp](https://github.com/deciding/simpler-percounter/blob/9769943a58f4d7580dd23e6f6eff93067e7e9bbc/tests/ut/cpp/a5/runtime/host_build_graph/test_hbg_percounter_graph_prepare.cpp) | AICPU 展开：发布存储地址、计数器、条目；shell 计数器留给 sink；localize 失败时什么都不发布 |

场景测试（`tests/st/a5/host_build_graph/`）：

| 目录 | 测什么 | 失败的表现 |
| ---- | ------ | ---------- |
| `spmd_blocks/` | 单槽多 block；`join_24` 是 N 路 join 唯一可证伪的测试（消费者必须等生产者所有 block） | 结果不对 |
| `mix_rendezvous/` | MIX 与单核任务跨依赖交错，`_bN` 为多 block MIX | 结果不对 |
| `sync_cohort/` | kernel 内真实跨 lane 屏障；`_nosync` 用同一张图但不设属性 | **挂住**（45 s op 超时），这正是测试本身 |
| `sync_cohort_interleaved/` | 单核 cohort 和 MIX cohort 同时在途 | 挂住：持有 block 的 lane 先去参加别的会合 |

## 11. 建议的走读顺序

1. 先读本文第 1、4 节，建立"host 编表、lane 自调度"的整体印象。
2. `percounter_types.h`：计数器（5.2）、条目（5.3）、控制块（5.4）、领号（5.5）。这四样看懂了，主循环就能读。
3. `percounter_executor.h` 的 lane loop（8.2）和任务运行（8.3），先忽略 MIX、cohort 和 Graph 分支。
4. Host 侧 `build_tables`（6.3），重点看 b-level 排序和无死锁论证。
5. 第二轮再读 MIX（8.5）、cohort（8.6）和收尾（8.7）。
6. 最后读 Graph：模板（6.4）→ AICPU 展开（7.2）→ lane 中的 `percounter_resolve` Graph 分支和 `wait_for_entry`。
