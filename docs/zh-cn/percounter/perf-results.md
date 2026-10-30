# percounter 调度器性能对比

a5 `host_build_graph` 三种 AICore 调度器在同一台机器、同一时段内交错测得的延迟对比：

- **legacy** —— AICPU 逐个派发任务。
- **resident** —— 每个 cluster 留一个核做调度。
- **percounter** —— 每条 lane 自己调度自己，见[代码走读](code-walkthrough.md)。

被测代码固定在 `perf/aic-gap` 分支的
[`d0d7485`](https://github.com/deciding/simpler-percounter/commit/d0d748581e8d8e05726e83e4b8113aac2949172f)。

## 结论

15 个 case 中，percounter 快于 legacy 的有 15 个，快于 resident 的有 14 个
（第 15 个 case resident 跑不了）。相对 legacy 的降幅是 14%~94%，相对 resident 是 5%~70%。

收益极不均匀，而且**只有在任务数足够大时才显著**。2048 个 kernel 以上的 5 个 case，
相对 resident 降 16%~70% 且随规模单调上升；其余 10 个 case 降幅都在 5%~20%，
与规模无关。把这 15 个数平均成一个"提升 X%"会同时高估小 case 和低估大 case。

## 测试方法

每个 case 在三种调度器下各跑 3 遍，三种调度器**交错**执行，且每遍轮换顺序
（`legacy→resident→percounter`、`resident→percounter→legacy`、
`percounter→legacy→resident`），使三者落在同一时段，不被机器负载的漂移拉开。

每次运行跑 4 轮，丢掉第一轮（含预热），取后 3 轮的中位数；再取 3 遍的中位数作为该格的值。
全部数值单位为微秒。

三个指标：

| 指标 | 来源 | 含义 |
| ---- | ---- | ---- |
| sys wall | `chip.run.runner_run` | host 端每轮从发起到返回 |
| device wall | `chip.run.runner_run.device_wall` | 设备端一轮的耗时 |
| kernel span | lane 计数器 | 最后一个 kernel 结束减去第一个 kernel 开始 |

**调度器是显式指定的，不是让运行时自己挑。** 仓库里 `SIMPLER_HBG_SCHEDULER_TYPE`
只认 `percounter` 和不设，不设时由 `runtime_maker.cpp` 按图自动选 legacy 或 resident；
而且 percounter 遇到自己跑不了的图会悄悄落回默认选择。两点都会让"某调度器的数字"
掺进别的调度器。测试用一个实验补丁加了 `SIMPLER_EXP_FORCE_SCHED`：
指定谁就跑谁，跑不了就让 bind 失败并报错，表格里留空，绝不静默换人。
该补丁与计数器仪表都只存在于开发机的实验副本中，不进仓库。

## 环境

a5 硬件，单设备，经 `task-submit` 取得设备独占锁。机器为多人共用，
测试期间有其他负载——交错执行正是为此设计。

## 结果

### device wall

`kernels` 是该 case 一轮中执行的 kernel 总数，按 lane 计数器统计，用于衡量规模。
最后两列是 percounter 相对该调度器的耗时降幅。

| case | kernels | legacy | resident | percounter | 比 legacy | 比 resident |
| ---- | ------: | -----: | -------: | ---------: | --------: | ----------: |
| pa_case1 | 65536 | 71691.4 | 14461.3 | 4377.4 | 93.9% | 69.7% |
| pa_b64 | 16384 | 19852.6 | 4049.0 | 1540.6 | 92.2% | 61.9% |
| pa_b16 | 4096 | 5421.5 | 1694.4 | 958.8 | 82.3% | 43.4% |
| batch_case1 | 4096 | 5483.4 | 3796.6 | 2173.3 | 60.4% | 42.8% |
| batch_case2 | 2048 | 6872.0 | 5665.1 | 4766.3 | 30.6% | 15.9% |
| unroll_case1 | 1024 | 2131.2 | 1436.8 | 1344.7 | 36.9% | 6.4% |
| mscope_case1 | 1024 | 2266.5 | 1606.8 | 1521.2 | 32.9% | 5.3% |
| bgemm_case0 | 1000 | 2152.2 | 1171.7 | 1108.8 | 48.5% | 5.4% |
| unroll_case2 | 512 | 1653.4 | 1153.6 | 1013.2 | 38.7% | 12.2% |
| mscope_case2 | 512 | 1710.3 | 1253.1 | 1081.2 | 36.8% | 13.7% |
| spmd_case1 | 72 | 2912.0 | | 2502.5 | 14.1% | |
| pa_var4 | 64 | 1102.3 | 673.3 | 536.6 | 51.3% | 20.3% |
| pa_small2 | 32 | 1073.9 | 632.0 | 590.8 | 45.0% | 6.5% |
| pa_var2 | 20 | 1032.5 | 632.8 | 568.6 | 44.9% | 10.1% |
| pa_small1 | 12 | 1022.0 | 642.7 | 566.4 | 44.6% | 11.9% |

`spmd_case1` 的 resident 一格为空：resident 跑不了 SPMD 形状的任务，
显式指定它会被拒绝，这是能力缺失，不是测量失败。详见“已知问题”。

### sys wall

| case | legacy | resident | percounter |
| ---- | -----: | -------: | ---------: |
| pa_case1 | 73583.3 | 21592.5 | 6674.2 |
| pa_b64 | 21498.9 | 6549.7 | 3463.9 |
| pa_b16 | 6912.8 | 3723.5 | 2809.4 |
| batch_case1 | 7122.4 | 5964.4 | 4051.7 |
| batch_case2 | 8762.8 | 7733.5 | 6728.0 |
| unroll_case1 | 3994.2 | 3661.6 | 2777.4 |
| mscope_case1 | 3986.3 | 3493.4 | 3507.3 |
| bgemm_case0 | 4038.8 | 3221.0 | 2981.8 |
| unroll_case2 | 3377.8 | 3050.4 | 2683.6 |
| mscope_case2 | 3299.4 | 3071.6 | 2936.8 |
| spmd_case1 | 4706.9 | | 4253.8 |
| pa_var4 | 2464.4 | 2106.5 | 2296.6 |
| pa_small2 | 2353.2 | 2064.6 | 2009.1 |
| pa_var2 | 2755.8 | 1953.0 | 1990.9 |
| pa_small1 | 2204.9 | 2104.5 | 2359.8 |

### kernel span

| case | resident | percounter |
| ---- | -------: | ---------: |
| pa_case1 | 11302.7 | 3802.4 |
| pa_b64 | 2840.0 | 972.3 |
| pa_b16 | 916.1 | 423.9 |
| batch_case1 | 3016.7 | 1597.1 |
| batch_case2 | 4970.8 | 4237.9 |
| unroll_case1 | 755.9 | 738.6 |
| mscope_case1 | 967.8 | 952.1 |
| bgemm_case0 | 533.3 | 572.4 |
| unroll_case2 | 516.5 | 400.6 |
| mscope_case2 | 620.0 | 489.4 |
| spmd_case1 | | 1926.0 |
| pa_var4 | 67.8 | 43.5 |
| pa_small2 | 64.0 | 44.4 |
| pa_var2 | 37.8 | 24.3 |
| pa_small1 | 37.8 | 23.4 |

legacy 没有这一列：这些计数器挂在 resident 与 percounter 的 AICore 调度循环里，
读出走 AICPU 遍历每个 worker 的 `SchedulerContext`。legacy 下 AICore 不跑那个循环，
没有 lane 会发布这组数——是结构使然，不是测量失败。

## 观察

### 收益集中在大规模，且只在大规模一侧随规模增长

按 kernel 数排序后，2048 以上的 5 个 case 相对 resident 的降幅是单调的：
2048 → 15.9%，4096 → 42.8% 和 43.4%，16384 → 61.9%，65536 → 69.7%。
这与 percounter 的设计相符——resident 的单个调度核随任务数线性变忙，
而每 lane 自调度不随任务数变慢，差距因此随规模拉开。

**这条规律不能外推到小规模。** 1024 个 kernel 的两个 case 降 6.4% 和 5.3%，
而只有 512 个的反而降 12.2% 和 13.7%，12 个的降 11.9%。小规模一侧的降幅
在 5%~20% 之间无序波动，与 kernel 数没有关系：调度开销此时不是瓶颈，
剩下的差异由别的因素决定。

### 小 case 的收益在端到端时间里看不见

`pa_small1`、`pa_small2`、`pa_var2`、`pa_var4` 的 device wall 都在 0.5~1.1 ms，
三种调度器相差几十微秒；而它们的 sys wall 都是 2.0~2.4 ms，host 开销占了大头，
把设备端的差异完全淹没。`pa_small1`、`pa_var4`、`mscope_case1` 三个 case 的
sys wall 甚至是 percounter 略高于 resident——那是 host 侧几百微秒的抖动，
不是调度器变慢。

要让调度器的改进传到端到端时间，设备端耗时得先大过 host 开销。
`pa_case1` 就是这样：73.6 / 21.6 / 6.7 ms，device 的差异完整地传了上去。

### kernel span 大体解释了降幅，但有一个反例

降幅大的 case，span 也大幅缩短：`pa_case1` 11302.7→3802.4，`pa_b64` 2840.0→972.3；
降幅小的，span 也几乎不动：`unroll_case1` 755.9→738.6（差 2.3%），
`mscope_case1` 967.8→952.1（差 1.6%）。省下来的时间来自 lane 之间更紧凑的排布，
而不是 kernel 本身变快。

**`bgemm_case0` 不符合这条。** 它的 span 反而变长（533.3→572.4，+7.3%），
device wall 却缩短了 5.4%（1171.7→1108.8）。也就是说 percounter 在这个 case 上
省下的时间全在 span 之外——启动或收尾阶段——而 kernel 排布本身略有退步。
这个 case 的成因尚未查明。

## 已知问题

### resident 跑不了 SPMD 形状

`spmd_paged_attention::Case1` 的全部 3 次 resident 运行都在 bind 阶段被拒绝，
日志为 `task id=0 has a SPMD or sync-start shape resident does not run`。
这是 resident 的能力边界：默认选择遇到这种图会自动落回 legacy。
本测试显式指定 resident，于是如实报错而非静默换人，表中留空。
percounter 能跑该形状，相对 legacy 降 14.1%，是 15 个 case 中降幅最小的。

### resident 在 batch_paged_attention 上频繁超时

`batch_paged_attention` 的两个 case 在 resident 下都出现失败，
报 `sched_error_code=100 SCHEDULER_TIMEOUT`，设备日志带 `site=102`：
Case1 失败 2/3 次，Case2 失败 1/3 次。

本轮之前的一次对比中（当时用的是默认选择，未显式指定调度器），
Case1 的 resident 同样失败 2/3 次，失败位置与错误码一致——
可见与显式指定无关，是既有缺陷。legacy 与 percounter 在这两个 case 上全部通过。

因此表中 `batch_case1` 的 resident 一格取自 1 个成功样本，
`batch_case2` 取自 2 个，其余各格均为 3 个。
