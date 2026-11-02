# percounter 性能对比的复现脚本

复现 [`docs/zh-cn/percounter/perf-results.md`](../../docs/zh-cn/percounter/perf-results.md)
里的三组实验：15 个合成 scene test 的三方对比、qwen3-14B decode 这个整图例子的对比，
以及 pypto-lib DeepSeek-V4.1-Flash 拆出来的细粒度 kernel。

## 为什么需要补丁

仓库里的 `SIMPLER_HBG_SCHEDULER_TYPE` 只认 `percounter` 和不设，没法指定 legacy
或 resident；而且 percounter 遇到自己跑不了的图会悄悄落回默认选择。两点都会让
"某调度器的数字"掺进别的调度器。

`patches/` 下的补丁解决这个问题，它们**只作用于一份实验副本，不改源码树**：

| 补丁 | 作用 |
| ---- | ---- |
| `counter_patch.py` | lane 计数器 |
| `window_patch.py` | AICore 窗口分解，kernel span 那一列读的就是它 |
| `force_sched_patch.py` | `SIMPLER_EXP_FORCE_SCHED`：指定谁就跑谁，跑不了就让 bind 失败 |
| `poison_patch.py` | `SIMPLER_EXP_POISON_OUT`：运行前把 `OUT` 张量灌成 NaN |
| `launchspec_core_num_patch.py` | 让 `LaunchSpec` 同时认 `core_num`；少了它所有 pypto 生成的 a5 HBG 程序都编不过 orchestration |

`spin_backoff_patch.py` 不在上面这组里，它不参与 `setup`：它改的是
`PERCOUNTER_SPIN_BACKOFF_CYCLES`，一个 AICore 侧的常量，用来量"lane 自旋的占空比值不值"。
这个值在 AICore 代码里，读不到环境变量，所以改完必须重建 AICore 目标并同步到**三处**
加载位置（`build/lib`、实验副本的 venv、pypto venv 的 `_assets`）——`pip install .` 在
产物已存在时会跳过运行时构建，只重建 wheel，于是改动悄悄不生效。

**换二进制一定要等波次跑完。** 在波次中途换，之后启动的运行会用新对象，数据被污染
且事后只能靠时间戳区分。

`tasktime_patch.py` 同样不参与 `setup`，它打在 `window_patch.py` 之上，给**两种调度器**
都加按任务 id 的设备时间戳（块数、最早 start、最晚 end、总忙时），用来把一条串行链的
时间分到每一级。`window_patch.py` 只按核类型聚合，一级吃掉大半时间也看不出来——
`lm_head` 的采样级就是这样被平摊掉的。两侧挂的位置不同：percounter 在 AICore 的 lane 上，
legacy 不用改 AICore，AICPU 本来就同时拿着 slot、派发时刻和完成时刻，只是被一个这些运行
不会设的 chip-swimlane 级别挡着。它的读出必须放在 `LegacyAicpuExecutor::deinit`——legacy
走 `legacy_aicpu_execute` 这个独立入口，永远到不了 percounter 读出所在的 `AicpuExecutor`。
**打了它的构建不能用来报时延**：AICPU 的打点被从级别门后放了出来，legacy 因此多约 5% 开销。

`cases/test_pa_exp.py` 提供 `pa_b16` / `pa_b64` 两个大 shape，仓库里没有对应的 case。

## DeepSeek 组需要的两样东西

这组跑的是 pypto-lib 的模块，不是本仓库的 scene test，所以 `run` 要多给两个路径：
`--pypto-lib` 指向 pypto-lib 检出，`--pypto-venv` 指向一个装了 pypto **和**打过补丁的
simpler 的 venv。pypto 的编译核心是 Python 3.10 的 `.so`，实验副本自己的 `.venv` 通常
不是这个版本，所以这是两个解释器，不能合成一个。

`helpers/pypto_module.py` 负责把模块切到 HBG：pypto 的运行时只由环境里的 `PassContext`
决定（`jit/decorator.py::_resolve_runtime()`，默认 TRB），所以在 `validate()` 外面建一个
`PassContext([], runtime=HOST_BUILD_GRAPH)` 就是全部的开关，pypto-lib 一个文件都不用改。

## 用法

```bash
# 1. 建实验副本（rsync + 打补丁 + venv + editable 安装）
python tools/percounter-perf/percounter_perf.py setup --name g --source /path/to/clean/checkout

# 2. 跑。整个 plan 在一次设备独占锁里跑完
python tools/percounter-perf/percounter_perf.py run --name g --out ~/percounter_runs/w1

# 3. 出表
python tools/percounter-perf/percounter_perf.py report --out ~/percounter_runs/w1
```

## 开关

`--cases` 选跑哪组：

| 值 | 内容 |
| -- | ---- |
| `synthetic` | 15 个 scene test，三方（legacy / resident / percounter） |
| `graph` | qwen3-14B decode，**整图例子**；resident 在 bind 阶段被拒，时延波次不为它花运行 |
| `deepseek` | pypto-lib DeepSeek-V4.1-Flash 的细粒度 kernel，legacy 与 percounter 两方 |
| `all` | 三组都跑（默认） |

`deepseek` 只跑两方：pypto 给 a5 发的全是 SPMD 形状，resident 在 bind 阶段逐个拒绝，
为它花运行只会把同一句拒绝记 34 遍。

`--deepseek` 选这组跑哪些模块：

| 值 | 内容 |
| -- | ---- |
| `runnable` | 已知能跑完并自带 golden 的那些（默认），也就是 perf-results.md 报的那几个 |
| `all` | 连记录为跑不了的也试一遍，用来复核那些拒绝还在不在 |

`--comm-pad64` 把 pypto 发出的 comm buffer 向上取整到 simpler 要的 64 字节再分配。
simpler 的 comm domain 要求每个具名 buffer 和窗口本身都是 64 的倍数，好让具名 buffer
不共享一个缓存维护块；pypto 取整到 32（它 `codegen/distributed/comm_layout.h` 里的
`kCommBufferAlignmentBytes`，这个数来自 A2/A3 的 FP16 peer MTE 粒度），于是 4 字节的
signal buffer 发出来是 32 字节宽，被 `allocate_domain` 直接拒掉。这个开关是**量具不是
修复**——真正的修法是 pypto 那个常量改成 64。

`--golden` 选正确性：

| 值 | 内容 | 代价 |
| -- | ---- | ---- |
| `skip` | 只测时延 | — |
| `compare` | 另跑一个正确性波次（默认） | 合成用例每个 case 每调度器一次；整图例子 5 次运行 |

**灌毒在 `compare` 下是无条件开的。** `ChildMemoryTaskArgs` 对 `OUT` 方向的张量
只分配不上传，设备上留的是那块内存上一次的残留字节；不灌毒的话，一次没写满自己
输出的运行会拿残留数据去比对并可能通过。整图例子的正确性波次还带一个负对照
（读回前故意把 `out` 写成 NaN），比对必须报错——一个负对照没报错的波次不说明任何事。

时延波次**从不开 golden**：参考值的计算会落进被测区间里。

## 波次怎么编排

三种调度器交错执行，且每遍轮换顺序（`lg→res→pc`、`res→pc→lg`、`pc→lg→res`），
使三者落在同一时段，不被机器负载的漂移拉开，也不让任何一个总是跟在同一个前驱后面。
每次运行丢掉第一轮（含预热）取中位数，再取各遍的中位数。

## 前提

- a5 硬件，`task-submit` 在 `PATH` 上（脚本自己取设备独占锁，并先跑 `onboard-arch-precheck`）
- 每次运行的 `ASCEND_PROCESS_LOG_PATH` 指到它自己的目录，不落进共享的 `~/ascend/log/debug/`

## 诊断补丁

`setup` 不打这一组。它们各自回答一个具体问题，打法相同——还原实验副本的源码、
打上补丁、重建相应目标、同步到三处加载位置——读出都走设备日志里的一行前缀。
每个补丁的 docstring 写了它测什么、为什么这样测、以及它自己的失真。

| 补丁 | 日志前缀 | 回答的问题 |
| ---- | -------- | ---------- |
| `tasktime_patch.py` | `EXPT` | 一条串行链的时间分到每一级，两种调度器都有 |
| `tasktime3_patch.py` | `EXPS` | 同上，但容量 64 个任务且不折叠：percounter 写每个任务自己的计数器行 |
| `blocktrace_patch.py` | `EXPB` | 一个任务**逐块**的领取时刻、执行时刻、执行它的 lane，以及该 lane 在这两刻的累计 kernel 数与忙时。分辨"领得早跑得晚"（lane 的问题）与"领得晚立刻跑"（列表的问题） |
| `drawahead_patch.py` | `EXPD` | 一次领取拿几个位置，按"任务块数是否超过本核类型 lane 数"分桶 |
| `mixgap_patch.py` | `EXPM` | 一个 cluster 两次认领 MIX 工作之间隔多久 |
| `graphdump_patch.py` | `EXPG` | 主机侧：每个任务的块数、MIX/sync、核类型、生产者。不上板 |
| `orderdump_patch.py` | `EXPO` | 主机侧：每条顺序表的实际排列，带 b-level 与 t-level。不上板 |

还有一组是**改行为**的，用来给某个假设定价，不是候选修法：

| 补丁 | 改什么 |
| ---- | ------ |
| `spin_backoff_patch.py` | `PERCOUNTER_SPIN_BACKOFF_CYCLES` |
| `drawwidth_patch.py` | 按任务宽度限制一次领取，`guard` / `noguard` 两种形态 |
| `obscache_patch.py` | 每块的整缓存失效改成只失效 payload 自己的行，或整个去掉 |
| `mixhold_patch.py` | 持有位置的 lane 不认领 MIX |
| `mixdraw_patch.py` | 有未结清 MIX 认领时不抽新位置 |

**改行为的补丁要连 `passed=` 一起读。** `mixdraw` 那次看着快 13~20%，实际是 golden
不匹配——挂住的 `claim.position` 让那条 lane 再不抽位置，它那份工作无人认领。
只看时间会把"漏跑了活"读成"变快了"。

这些实验各自的结论记在
[`docs/zh-cn/percounter/perf-results.md`](../../docs/zh-cn/percounter/perf-results.md)
的"排除掉的候选"一节。

## 顺序表的扫描

`helpers/order_misfit.py` 读一个波次的运行目录，找出**同一 b-level 内次序与就绪
先后相反**的任务对，并给每一对打分。它需要的三样数据来自 `graphdump_patch.py`、
`orderdump_patch.py` 和 `tasktime3_patch.py`，一次构建全打上即可。

```bash
python3 tools/percounter-perf/helpers/order_misfit.py ~/runs/scan1
```

只看同层是有理由的：跨层的"未就绪任务排在前面"是 b-level 的判断，而它是对的——
实测把 b-level 换成就绪序会亏 7.10%，因为一条停在"后面吊着大量工作"的任务上的
lane 是一个**预置的扳机**，生产者一完成它立刻就发。同层任务之间没有边，b-level
对它们没有意见，现在由 task id 决定，那才是可以免费改的地方。

打分是四项的乘积，每一项排除一种"救不回来"的情形：

```text
lanes_held = min(靠前者块数, lane 池, 靠后者块数, 紧邻前驱的块数)
score      = lanes_held × (靠前者就绪时刻 − 靠后者就绪时刻)
```

后两项都是被反例教出来的。**靠后者块数**：192 块的任务挡在 **3 块**的任务前面，
最多救三条 lane，两处这样的候选实测只值 0.17% 和 0.40%。**紧邻前驱的块数**：lane
只有在前一段工作把它们**同时**释放时才会一拥而上，值 10.3% 的那一对前面是个 256
块的任务，而 `prefill_attn_swa` 的两对前面只有 16 块——实测它们的靠后任务是在自己
的就绪时刻开始的，**从未被挡住**，交换反而亏 0.5%。

**高分是候选，不是结论。** 这个模型对 swa 仍然高估一个数量级以上。确认一个候选要
两步：

1. `blocktrace_patch.py` 盯靠前的那个任务。真的被挡住，会看到很大的"领取到执行"
   间隔、而同期 kernel 忙时很小。
2. `orderafter_patch.py` 配 `SIMPLER_EXP_ORDER_AFTER=靠前:靠后` 做同时段 A/B，
   **先读 `rc=` 和 `passed=`，再读时间**。

## 按 lane 的时间预算

其余探针都按任务或按块。两者都答不了"一个任务的块各自领到手就跑得很快，整体却慢
十倍"——块是陆续被领走的，所以问题在池，而池是别的探针不量的维度。

`patches/lanebudget_patch.py` 把每条 lane 的生命分成四份，按核类型汇总打 `EXPL`：

| 项 | 含义 |
| -- | ---- |
| `mix` | 在 MIX rendezvous 里执行（`percounter_run_subtask`） |
| `list` | 执行普通列表工作（`percounter_run_task`） |
| `wait` | 阻塞在生产者计数器的自旋上，手里握着一个未就绪的位置 |
| `life` | lane 生命周期，用来把上面三项读成比例 |

`wait` 直接取仓库自带的 `cfg.kx[2]`——那个 fanin 等待循环的出口本来就在累加它。
**四项之和会超过 `life`**：等待循环里会调 `percounter_checkpoint_serve`，lane 可能
在等待期间服务一个 MIX 块，那段时间被两边各算一次。所以 `wait` 是上界。

`patches/straggler_patch.py` 是 `blocktrace_patch.py` 的补充：后者每块四个字，只装得
下八块，而一个 64 块任务的头八块往往都正常。它只留一条记录——"领取到执行"间隔最大
的那一块——并把 `hold << 32 | block << 24 | kernels << 16 | lane` 打包进一个字，这样
max-CAS 搬运的是一条一致的记录，不会出现 hold 来自这一块、lane 来自另一块。
`kernels` 是判据：1 表示 lane 一直空坐（纯依赖等待），2 表示它先跑了另一个 kernel
才回来，远大于 2 才可能是连续进出多个 rendezvous。

## 在写实现之前先否掉它

`patches/scanready_patch.py` 是一个**只测量、不改行为**的探针，用来回答"只认领已就绪
的位置"这个设计值不值得写：抽取点向前探一个窗口，记录第一个就绪任务的偏移，什么都
不认领。一次运行就定论，不用先写几百行。

窗口按**任务游程**数，不是按位置——这一点第一版搞错了，而它是个容易重犯的错，因为
"窗口"这个词天然让人想到位置。一个任务的块在列表里连续，所以一个 256 块的任务自己
就填满了任何 32 位置的窗口，那样量到的是"抽取点那个任务有多宽"。按游程跳是 O(1)：
条目同时带 `block_idx` 和 `block_num`，下一个任务起点是 `p + block_num − block_idx`
（sync-start 任务只占一个 arming 条目，要单独处理）。

结论是否定的——往前跳 32 个任务只买到 2.2% 和 0.0%，因为**在拓扑序里向前扫是朝依赖
更深、更不可能就绪的方向走**。细节见
[percounter 为什么在 prefill attention 上慢于 legacy](../../docs/zh-cn/percounter/prefill-attention-deficit.md)。

这个探针的模式值得复用：**行为改动先用只读探针定上界**。这一天里调参、改 MIX 认领
时机、换排序一共试了十来个方案，大部分是写完才发现无效；这个是唯一在动手前就被数据
否掉的，成本是一次运行。

## 一个不可用的模拟器，以及它为什么留在这里

`helpers/order_sim.py` 用一个 lane 循环的模型回放一份顺序表，本意是把"比较两个
顺序"的成本从上板一次约十分钟降到毫秒，好支撑搜索。**它没有通过校准，不能用来排序
顺序。** 留在仓库里是因为下一个想做同样事情的人会重写一遍，而重写的代价远高于读完
这一节。

校准靶子是 `prefill_attn_c2a_reuse` 的三个**已知答案**的顺序：

| 顺序 | 第一版模型 | 第二版模型 | 实测 |
| ---- | ---------: | ---------: | ---: |
| pristine | 2,546,012 | 4,637,459 | 真机 makespan 2,360,113 |
| 一对交换 | 好 13.0% | 好 2.7% | **好 10.30%** |
| legacy 全序 | **好 4.6%** | **好 3.0%** | **差 7.07%** |

第一版单票据、一次抽一个位置，把一次实测 7.07% 的退化排成了 4.6% 的改善——因为那样
建模的 lane 一空出来就能够到任何位置，于是"把任务挪后"永远没有代价，而真机上
"4 条 lane 从一开始就握着那个位置、生产者一完成立刻发"这件事它看不见。

第二版补上了八个票据组、`CLAIM_BATCH=4` 和"持有一批、挑就绪的跑"。**仍然排反**，
而且绝对值从偏小变成偏大近一倍。再往下调就是在拟合这三个点而不是建模。

能从中学到的是：**让一个位置靠后变贵的，正是分组和批量本身**，不是可有可无的细节；
而把它们建对又很难，因为它们同时决定了 lane 的可达性和铺开方式。想重做的人至少要
先过这三点校准，尤其是 legacy 全序那一点——它是唯一一个方向相反的样本。
