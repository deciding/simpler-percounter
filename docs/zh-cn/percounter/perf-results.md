# percounter 调度器性能对比

a5 `host_build_graph` 三种 AICore 调度器在同一台机器、同一时段内交错测得的延迟对比：

- **legacy** —— AICPU 逐个派发任务。
- **resident** —— 每个 cluster 留一个核做调度。
- **percounter** —— 每条 lane 自己调度自己，见[代码走读](code-walkthrough.md)。

被测代码固定在 `perf/aic-gap` 分支的
[`d0d7485`](https://github.com/deciding/simpler-percounter/commit/d0d748581e8d8e05726e83e4b8113aac2949172f)。

## 结论

三组实验。**合成用例**：15 个 scene test，三方对比，percounter 相对 legacy
降 14%~94%，相对 resident 降 5%~70%。**真实模型**：qwen3-14B decode，
percounter 相对 legacy 的 device wall 降 22.9%、端到端降 22.1%；
resident 跑不了这个图。**细粒度 kernel**：DeepSeek-V4.1-Flash 拆出的 34 个模块，
19 个跑出数，percounter 从领先 39.0% 到落后 62.8%。

**percounter 并非总是更快。** 细粒度那组里 6 个模块它落后于 legacy：
全部 prefill attention 模块，以及 `lm_head` 的 −62.8%。两件事，不是一件。
`lm_head` **已定位**：领取批量把一个 16 块的任务串成了 4 轮，把
`PERCOUNTER_CLAIM_BATCH` 改成 1，同一个模块从落后 62.8% 变成领先 9.3%。
prefill 那批**缩小到了任务宽度**：它们含有 256 块和 512 块的任务，远宽于
64 条 AIV lane，这些任务只用到 10~26 条 lane，离理想打包 2.9~4.7 倍；
而 decode 那批最宽只有 128 块，不超过机器宽度的任务都打得很满。
机制上的假说是静态分组限制了跨组取用，尚未证实。
这是前两组实验看不到的一面——它们没有一个这样的形状。

收益极不均匀。合成用例里**只有在任务数足够大时才显著**：2048 个 kernel 以上的
5 个 case 相对 resident 降 16%~70% 且随规模单调上升；其余 10 个降幅都在 5%~20%，
与规模无关。把这些数平均成一个"提升 X%"会同时高估小 case 和低估大 case。

更要紧的是，**合成用例的降幅不能用来预期真实模型上的收益**：在几乎相同的
规模上（约 1.6 万个 kernel），合成的 `pa_b64` 降 92.2%，qwen3 只降 22.9%。
详见"观察"。

## 测试方法

### 共通

a5 硬件，单设备，经 `task-submit` 取得设备独占锁。机器为多人共用，
测试期间有其他负载——交错执行正是为此设计。全部数值单位为微秒。

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

### 合成用例

每个 case 在三种调度器下各跑 3 遍，三种调度器**交错**执行，且每遍轮换顺序
（`legacy→resident→percounter`、`resident→percounter→legacy`、
`percounter→legacy→resident`），使三者落在同一时段，不被机器负载的漂移拉开。

每次运行跑 4 轮，丢掉第一轮（含预热），取后 3 轮的中位数；再取 3 遍的中位数作为该格的值。

### qwen3

两种调度器各跑 3 遍，同样交错执行（`legacy→percounter`、`percounter→legacy`、
`legacy→percounter`）。每次运行跑 30 轮，丢掉第一轮，取后 29 轮的中位数；
再取 3 遍的中位数。正确性单独跑一个波次，见下。

## 合成用例：15 个 scene test

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

## 真实模型：qwen3-14B decode

40 层、batch 16、seq 3500、bf16，单设备。图的规模：47 个顶层任务，展开后
15000 条 AIC、1809 条 AIV、40 条 MIX 的 lane order list 条目（合计 16849 个
kernel），40 个 sync cohort，204 条边，40 次 Graph 调用、1 个 Definition、
11080 个 body task。

### 时延

| 调度器 | device wall | sys wall | 三遍分别 |
| ------ | ----------: | -------: | -------- |
| legacy | 29538.2 | 30395.4 | 29538.2 / 29684.3 / 29503.6 |
| percounter | 22781.6 | 23677.6 | 22804.3 / 22775.8 / 22781.6 |
| resident | | | 不适用 |

**percounter 相对 legacy：device wall 降 22.9%，sys wall 降 22.1%。**
三遍之间 percounter 相差 29 µs（0.13%）、legacy 相差 181 µs（0.6%），
差距远超出运行间的漂移。6 次运行全部 `rc=0`。

resident 一格为不适用：显式指定它时 bind 被拒，理由是 graph-execution
的图需要 AICPU 调度。详见"已知问题"。

与合成用例在同一规模上的对照：

| case | kernel 数 | legacy | percounter | 降幅 |
| ---- | --------: | -----: | ---------: | ---: |
| pa_b64（合成） | 16384 | 19852.6 | 1540.6 | 92.2% |
| qwen3（真模型） | 16849 | 29538.2 | 22781.6 | 22.9% |

### 正确性

以同一波次里一次 legacy 运行的设备输出为基准，比较其余运行，
容差 RTOL 5e-2、ATOL 1e-1。

| 对比对象 | out 超差 | out NaN 不一致 | out 平均绝对差 | k/v 写入超差 |
| -------- | -------: | -------------: | -------------: | -----------: |
| legacy 再跑一次（对照） | 0 | 0 | 0.00126 | 0 / 0 |
| percounter | 0 | 0 | 0.00128 | 0 / 0 |
| 负对照（读回前写 NaN） | — | 81920 | — | 0 / 0 |

percounter 与基准的差异，和 legacy 自己重跑一次与基准的差异同量级：两者
`out` 的最大绝对差都是 0.0625，正好是量级 11.875 处 bf16 的一个 ulp；
k 与 v 的写入行全部落在容差内。

**比对本身是有防护的。** `ChildMemoryTaskArgs` 对 `OUT` 方向的张量只分配不上传，
设备上留的是那块内存上一次的残留字节；若某次运行没有写满自己的输出，
比对就会拿残留数据和基准比而通过。测试在每次运行前把 `OUT` 张量灌成 NaN，
每次运行的日志都记录了 `poisoned=1`；再加一个负对照——读回前故意把 `out`
写成 NaN，比对必须报出全部 81920 个不一致。负对照如期报错，
说明这条检测链路是通的。该缺陷本身见"已知问题"。

另外用 torch 参考值做了一次绝对正确性锚点：legacy、1 轮、`mismatch=0`。
它比上面的跨调度器比对慢约 48 倍（1115 s 对 23 s），不适合每次都跑，
因此不计入上表。

## 细粒度 kernel：DeepSeek-V4.1-Flash

前两组一个是人造负载、一个是单个整图，都只告诉我们 percounter 能赢多少。
这一组把一个真实模型拆成单独的 kernel 逐个对比，它的价值不在又一个加速比，
而在**它找出了 percounter 会输的形状**——前两组里一个这样的例子都没有。

### 方法

pypto-lib 的模块，通过 `PassContext(runtime=HOST_BUILD_GRAPH)` 切到 HBG，
pypto-lib 本身不改。每个模块自带 golden，正确性随运行一起验，
不依赖跨调度器比对。每个模块跑 20 轮基准、预热 2 轮，丢掉冷启动那次校验派发取中位数。

**只有两方。** pypto 给 a5 发的全是 SPMD 形状，resident 在 bind 阶段逐个拒绝，
34 个模块无一例外，报 `task id=0 has a SPMD or sync-start shape resident does not run`。

### 34 个模块的分布

| 状态 | 数量 | 说明 |
| ---- | ---- | ---- |
| 能跑并出数 | 19 | 其中 12 个需要把 comm buffer 补齐到 64 字节 |
| 卡在 `TaskOutputTensors` 断言 | 12 | 第二次运行时取输出张量越界 |
| 需要 2 张卡 | 1 | `prefill_layer`，本波次只给 1 张 |
| 其他 | 2 | `hc_mixes` 报 `-5`；`prefill_attn_c2a_full` 只有 percounter 侧成功 |

**阻断与调度器无关。** 上面两类失败在 legacy 和 percounter 下逐字相同，
resident 更早就被拒；详见“已知问题”。

### 结果

直接能跑的 7 个，都是 1~3 个 task 的小模块：

| 模块 | tasks | legacy | percounter | 降幅 |
| ---- | ----: | -----: | ---------: | ---: |
| rmsnorm | 1 | 184.5 | 112.5 | 39.0% |
| hc_pre | 1 | 179.6 | 115.3 | 35.8% |
| hc_post | 1 | 188.2 | 121.6 | 35.4% |
| hc_head | 1 | 179.4 | 119.5 | 33.4% |
| boundary_fwd | 3 | 233.8 | 156.3 | 33.1% |
| input_pack | 1 | 210.1 | 163.7 | 22.1% |
| engram | 3 | 550.7 | 488.7 | 11.3% |

补齐 64 字节后又跑通 12 个，是真正的 attention kernel：

| 模块 | tasks | legacy | percounter | 降幅 |
| ---- | ----: | -----: | ---------: | ---: |
| decode_attn_swa | 30 | 4368.4 | 2764.9 | 36.7% |
| decode_attn_c2a_reuse | 32 | 4320.5 | 2943.7 | 31.9% |
| decode_attn_c2a_full | 63 | 4107.3 | 2886.3 | 29.7% |
| decode_attn_c1a_full | 67 | 3327.6 | 3194.2 | 4.0% |
| decode_attn_c1a_reindex | 46 | 3352.1 | 3291.2 | 1.8% |
| decode_attn_c1a_reuse | 33 | 2671.1 | 2663.3 | 0.3% |
| prefill_attn_c1a_reuse | 33 | 3333.9 | 3361.9 | −0.8% |
| prefill_attn_c1a_full | 68 | 3499.9 | 3683.3 | −5.2% |
| prefill_attn_c1a_reindex | 49 | 3479.0 | 3737.5 | −7.4% |
| prefill_attn_c2a_reuse | 37 | 2280.1 | 2541.7 | −11.5% |
| prefill_attn_swa | 95 | 16444.6 | 19256.5 | −17.1% |
| lm_head | 9 | 1228.3 | 1999.3 | −62.8% |

**6 个模块 percounter 比 legacy 慢**，全部 prefill attention 模块加上 `lm_head`。
这是本页三组实验里第一次出现 percounter 落后。

### 落后的是两个不同的机制

把每个模块的 `fanin_load`（lane 在生产者计数器上的轮询次数）除以 kernel 数：

| 模块 | kernels | polls/kernel | 降幅 |
| ---- | ------: | -----------: | ---: |
| decode_attn_swa | 4995 | 18 | 36.7% |
| decode_attn_c2a_full | 5896 | 20 | 29.7% |
| decode_attn_c2a_reuse | 5283 | 20 | 31.9% |
| engram | 104 | 23 | 11.3% |
| prefill_attn_c2a_reuse | 1572 | 108 | −11.5% |
| prefill_attn_swa | 6414 | 324 | −17.1% |
| decode_attn_c1a_full | 741 | 341 | 4.0% |
| prefill_attn_c1a_full | 1112 | 358 | −5.2% |
| prefill_attn_c1a_reindex | 1034 | 358 | −7.4% |
| decode_attn_c1a_reuse | 429 | 401 | 0.3% |
| prefill_attn_c1a_reuse | 848 | 420 | −0.8% |
| lm_head | 116 | 684 | −62.8% |

**轮询密度分得开两端，分不开中间。** 23 次以下的四个全是大幅领先；
100 次以上的全部落进 −17%~+4% 这个窄带，而带内部排不出顺序——
108 次的输 11.5%，420 次的只输 0.8%。所以这里不止一个机制。

把 AICore 侧的自旋退避占空比 `PERCOUNTER_SPIN_BACKOFF_CYCLES`
从仓库里的 256 向两个方向都试过，两组模块的反应完全不同：

| 模块 | polls/kernel | 64 | 256 | 1024 | 4096 |
| ---- | -----------: | -: | --: | ---: | ---: |
| decode_attn_swa | 18 | 38.6% | 36.7% | 35.0% | −2.5% |
| decode_attn_c1a_full | 341 | 1.7% | 4.0% | −5.5% | −37.4% |
| decode_attn_c1a_reuse | 401 | 18.7% | 0.3% | −3.8% | −0.9% |
| lm_head | 684 | −62.7% | −62.8% | −65.5% | −65.5% |

attention 模块对这个值**极度敏感**：`decode_attn_swa` 从领先 36.7% 崩到落后 2.5%，
`decode_attn_c1a_full` 崩到落后 37.4%。退避期内 lane 晚一点才看见生产者的发布，
这段观察延迟直接计入关键路径，且随轮询次数放大。

**但往小调也没有收益。** 64 与 256 之间四个模块两赢两输，最大差 2.9%，
都在运行间波动的量级上。这条曲线不是单调的，是一个在 256 附近触底的盆地，
两个方向都不值得动。（`decode_attn_c1a_reuse` 在 64 下那个 18.7% 不可信：
它的 legacy 对照自身是双峰的，同一次运行内在 2.65 ms 与 3.29 ms 两簇间跳，
而 legacy 不受这个常量影响，所以那是分母的噪声。）

`lm_head` 对它**完全不敏感**：四个取值下都是 −63%~−65%，跨 64 倍纹丝不动。
它落后的原因是另一回事，见下一节。

所以退避是这批模块**敏感的一个量，不是它们落后的成因**：256 已在盆底，
往哪边调都更差，残差无从归因到它。prefill 那批的成因见
"[prefill 落后：收敛到一个块](#prefill-落后收敛到一个块)"。

分得更干净的量是**平均 kernel 时长**（AIV 的 `kcycles / kernels`）：

| 模块 | 平均 kernel | 降幅 |
| ---- | ----------: | ---: |
| decode_attn_swa | 5.6 µs | 36.7% |
| decode_attn_c2a_full | 6.6 µs | 29.7% |
| decode_attn_c2a_reuse | 7.1 µs | 31.9% |
| prefill_attn_c2a_reuse | 23.9 µs | −11.5% |
| prefill_attn_swa | 38.2 µs | −17.1% |
| prefill_attn_c1a_full | 39.8 µs | −5.2% |
| prefill_attn_c1a_reindex | 42.5 µs | −7.4% |
| prefill_attn_c1a_reuse | 45.6 µs | −0.8% |
| decode_attn_c1a_full | 74.6 µs | 4.0% |
| decode_attn_c1a_reindex | 84.4 µs | 1.8% |
| decode_attn_c1a_reuse | 116.2 µs | 0.3% |

7 µs 以下的三个全是大幅领先，24 µs 以上的八个全落进 −17%~+4%，中间没有样本。
这说明 percounter 的收益是**每个任务省下的固定派发开销**：kernel 5 µs 时省掉 3~4 µs
是 40%~70%，kernel 100 µs 时同样的绝对值只剩 3%~4%。但这只解释收益为什么缩水，
**不解释为什么会变负**——prefill 那边还有一项 percounter 付而 legacy 不付的成本，
见"[prefill 落后：收敛到一个块](#prefill-落后收敛到一个块)"。

### lm_head 落后的是领取批量，不是调度延迟

给两种调度器都加上按任务 id 的设备时间戳后，`lm_head` 那条深度 9 的串行链
每一级的跨度与总忙时如下（ns，同一次运行）：

| 级 | 块 | legacy span | percounter span | legacy busy | percounter busy |
| -- | -: | ----------: | --------------: | ----------: | --------------: |
| 推 | 16 | 25,278 | 10,306 | 296,274 | 38,630 |
| 等 | 1 | 2,370 | 647 | 2,370 | 647 |
| 收 | 10 | 10,985 | 2,003 | 71,175 | 14,536 |
| matmul | 24 | 622,496 | 618,528 | 14,648,310 | 14,516,048 |
| 推 | 24 | 31,738 | 33,371 | 496,201 | 180,094 |
| 等 | 1 | 2,128 | 680 | 2,128 | 680 |
| 收 | 24 | 26,738 | 20,438 | 406,622 | 163,060 |
| 采样 | 16 | 317,942 | 1,243,069 | 4,917,094 | 4,918,177 |

全链 1072 µs 对 1952 µs，**差的 880 µs 全部落在最后一级采样上**，
其余七级 percounter 要么持平要么更快。

决定性的是最后两列：**4,917,094 对 4,918,177，相差 0.02%**。同样 16 个块、
每块同样约 307 µs——kernel 一点没有变慢，变的只有并发度：legacy 的
busy/span 是 15.5，percounter 是 **3.96**。percounter 把一个 16 块的任务跑成了 4 轮。

成因是 `PERCOUNTER_CLAIM_BATCH`：一条 lane 一次从顺序表抽走**连续 4 个位置**
并按序执行，抽走的位置在它跑完之前不能转交给空闲 lane。AIV 顺序表共 92 个位置，
采样占 76–91，按 4 个一块切正好是第 19–22 块，`c % 8` 把它们分给 4 个不同的组，
每组一条 lane 整块拿走——于是 4 路，其余 60 条 lane 空转。

代码里本有一条逃生门（`percounter_executor.h`）：

```cpp
batch = get_sys_cnt_aicore() - kernel_start >= PERCOUNTER_LONG_KERNEL_CYCLES ? 1 : PERCOUNTER_CLAIM_BATCH;
```

下一次抽多少由这条 lane **刚跑完**的 kernel 决定，所以它是反应式的，
保护不了第一次遇到长 kernel 的那一刻。采样是链条的最后一级，
抽到它的 lane 之前跑的是 1.5~2.4 µs 的块，于是照 4 抽走。

把这个常量改成 1 验证：

| 量 | batch=4 | batch=1 |
| -- | ------: | ------: |
| 采样这一级的跨度 | 1,243,069 ns | 302,767 ns |
| 这一级的并发度 | 3.96 | 15.9 |
| lm_head 中位时延 | 2003.5 µs | 1180.6 µs |
| 相对 legacy | −62.8% | +9.3% |

一个常量，72 个百分点，从大幅落后翻成领先。"percounter 跑 lm_head 时延双峰、
legacy 不双峰"也由此解释：哪几条 lane 抽到采样那段区间每次运行都不同，
并发度在 4 路附近抖。

**但 1 不是修法。** `PERCOUNTER_CLAIM_BATCH` 的注释记着它在 `mixed_*` 那批
多短任务的图上比各自最优差 3.11 倍（4 是 1.39 倍，是在最坏情况上折中选出来的）：
一次只抽一个位置时，票据本身成了瓶颈。该修的是那条规则的反应式特性——
顺序表是 host 在 bind 时建好的，每个位置属于哪个任务、那个任务有多少块都是已知的，
批量可以由**即将抽到的任务**决定，而不是上一个跑完的 kernel。

### 修法：按任务宽度限制一次抽取

`percounter_draw_width` 在抽取前先读一眼顺序表的抽取点：一次抽取对任何单个任务
最多取 ⌈block_num / lane_count⌉ 个位置，也就是该任务**自身的最优每-lane 份额**。
这个上界按定义不会拉长任何任务的完成时间。额度按任务计而不是按抽取计，所以
一串单块任务照样抽满 `PERCOUNTER_CLAIM_BATCH`，短 kernel 的图不受影响。

单这一条不够。只按宽度限制时，`pa_case1` 这类表长、kernel 短的图付出约 3%：
位置被切碎，一次只服务一个抽取的票据成了新瓶颈。所以还有第二个条件——
**本组剩余位置还够给组内每条 lane 各发一个满 range 时，不限制**。
多占同一个任务的几个位置，只有在"有空闲 lane 本来能拿走它们"时才有害，
而剩余工作充足恰恰意味着没有这样的 lane；两种代价由同一个条件分开。

百分比仍是 percounter 相对 legacy 的耗时降幅，改前改后在同一时段内换二进制测得：

| 用例 | 改前 | 改后 |
| ---- | ---: | ---: |
| lm_head | −63.5% | +10.5% |
| decode_attn_swa | +38.2% | +38.2% |
| decode_attn_c1a_full | −1.4% | +0.1% |
| prefill_attn_c1a_full | −4.7% | −7.6% |
| prefill_attn_c2a_reuse | −10.6% | −16.1% |
| prefill_attn_swa | −17.2% | −17.8% |
| pa_case1 | 93.8% | 93.8% |
| pa_b64 | 92.3% | 92.1% |
| pa_b16 | 79.4% | 78.9% |
| unroll_case1 | 37.5% | 39.6% |
| mscope_case1 | 32.0% | 33.4% |

`prefill_attn_c1a_full` 和 `prefill_attn_c2a_reuse` 退化是真的（percounter 自身耗时
分别 +3.1% 和 +5.2%），与 percounter 在 prefill 上本来就落后是两件事。

退化的成因是这条规则的额度 `ceil(block_num / lanes)` 对**块数恰好等于 lane 数**的
任务等于 1，于是把它们压成逐个领取。`prefill_attn_c2a_reuse` 里任务 11 和 24 正是
64 块，而它们的并发度本来已是 62.07/64 和 60.46/64，没有任何可改善的余地，
只多付了票据开销。当初判断"这条规则对 prefill 是空操作"的依据是平均 kernel
24~46 µs 远高于 `PERCOUNTER_LONG_KERNEL_CYCLES`，那个判断错在**平均值掩盖了
双峰分布**：同一模块里既有每块 453 µs 的任务，也有每块 1.4 µs 的任务，lane 跑完
后者之后批量照常回到 4。

**跨波次的数不能这样读。** 同一个未改动的二进制重跑，`decode_attn_c1a_full` 一次
测得 +4.0%、另一次 −1.4%。本页每张表都是单次波次的中位数，所以个位数的差
不是信号；只有同一时段内换二进制测出的差才是。

### prefill 落后：收敛到一个块

本节是调查过程和排除依据。成因本身、以及给没有背景的读者的讲法，在
[percounter 为什么在 prefill attention 上慢于 legacy](prefill-attention-deficit.md)。

**基线**（五个模块，同一波次，干净二进制，device wall 中位数 ns）：

| 模块 | legacy | percounter | percounter 比 legacy |
| ---- | -----: | ---------: | -------------------: |
| prefill_attn_c2a_reuse | 2,243,415 | 2,534,066 | −12.96% |
| prefill_attn_c1a_full | 3,504,355 | 3,676,598 | −4.92% |
| prefill_attn_swa | 16,488,668 | 19,267,867 | −16.86% |
| decode_attn_c1a_reuse | 2,980,150 | 2,653,528 | +10.96% |
| decode_attn_swa | 4,447,716 | 2,739,802 | +38.40% |

**差距在前 600 µs 里就产生完了。** 两种调度器的 `t13` 同时结束（126,741 / 119,108），
之后 legacy 在 219,328 就推进到 `t20`，percounter 要等到 594,609。这 375 µs
和 290 µs 的 wall 差是同一笔账。重叠率两边几乎相同（1.42 / 1.45），不是并行度问题。

主机侧 dump 出的真实 AIV 顺序表，前 64 个位置是：

| 位置 | 任务 | 块数 | b-level | t-level | 就绪时刻 |
| ---: | ---: | ---: | ------: | ------: | -------- |
| 0–3 | 15 | 4 | 15 | 2 | 119 µs（等 13） |
| 4 | 2 | 1 | 14 | 0 | 立即 |
| 5–36 | 16 | 32 | 14 | 3 | 等 15 |
| 37–40 | 8 | 4 | 13 | 2 | 467 µs（等 7，10 块 MIX，每块 453 µs） |
| 41–63 | 17 | 23/32 | 13 | 4 | 等 16 |

64 条 AIV lane 在 t≈3 µs 各领一个位置，**其中 63 个位置的任务当时没就绪**，而一条
lane 认领了位置就对它负责到底。看起来这就是成因——legacy 的 AICPU 只派发已就绪的
任务，从不把核占在未就绪的工作上，而 prefill 的 AIV lane 有 79~96% 的时间花在显式
依赖等待上（decode 只有 37~75%）。

**这个解释是错的，有直接实验。** 把同层并列的次序从 id 改成 t-level 升序，顺序表
正如设计那样改变——task 19 的 32 个就绪块从位置 73 移到 37（进入前 64），task 8
从 37 移到 69（移出）——开局可运行的块从 1 个变成 28 个，而耗时只动了 0.29%，
九个模块整体中性偏负。所以"前 64 个位置里可运行的块太少"不是瓶颈。

漏看的是一条边：**`task 20 的 fanin 是 [19, 0, 17]`，要同时等 19 和 17**。关键链是
`13 → 15 → 16 → 17 → 20`，task 19 不在上面，让它早跑没有价值。

**瓶颈是一个块，而它被一个 MIX 块扣住了。** task 15 占位置 0–3，开局就被 4 条 lane
领走，119 µs 就绪，其中 3 块当即跑完，第 4 块拖到 471,779——即 task 7（10 块 MIX，
每块 453 µs）结束之后。按块的时间戳（`tools/percounter-perf/patches/blocktrace_patch.py`，
记每块的领取时刻、执行时刻、执行它的 lane，以及该 lane 在这两刻的累计 kernel 数与
累计忙时）给出了答案：

| block | lane | 领取到执行 | 其间 kernel 数 | 其间 kernel 忙时 |
| ----: | ---: | ---------: | -------------: | ---------------: |
| 1 | 194 | 470 µs | 2 | 464,806 ns |
| 3 | 66 | 462 µs | 2 | 454,748 ns |
| 2 | 195 | 123 µs | 1 | 2,430 ns |
| 3 | 67 | 122 µs | 1 | 2,592 ns |

准时的块跑一个 kernel、忙 2.4 µs，就是它自己。**迟到的块跑两个、忙 455~465 µs**，
减去自己那 2.4 µs，另一个是 ~453 µs，即 task 7 的一个 MIX 块。迟到的 lane 全部是
`cluster_lane == 1`，也就是 MIX 认领者。

所以不是票据没推进到那个位置，也不是 lane 在空等：**一条已经认领了未就绪列表位置的
lane 被拉去跑了一个 453 µs 的 MIX 块，那个位置就在它手里锁了 450 µs**，而 15→16→17→20
这条链正是 legacy 领先的那 375 µs。

`percounter_lane_loop` 里 `percounter_mix_tick` 在 `pending` 检查之前，所以开局第一次
迭代里 lane 是先 arm、后抽位置。这也是"持有位置的 lane 不认领 MIX"那次实验无效的原因:
它拦的是已握位置的 lane 去 arm，而实际发生的是已 arm 的 lane 去握位置。

### 排除掉的候选

每一条都做过同时段 A/B 或等价的直接测量。**没有一个旋钮能让 makespan 变好**，
这本身说明 percounter 当前配置对这张图已是局部最优，差距在模型层面而非参数层面。

| 候选 | 处置 |
| ---- | ---- |
| MIX/sync 开销 | `other_add` 两族都非零，`cas` 全为 0 |
| kernel 时长、轮询密度 | 区间重叠，带内排不出顺序 |
| 自旋退避 | 64/256/1024/4096 四点，256 是盆底 |
| 票据压力 | `ticket_add` 相对 kernel 数微不足道 |
| AIC/AIV 偏重 | 所有 attention 模块 AIC 等待计数为 0 |
| 逐边观测延迟 | 骨干任务间隙合计占一轮的 2.2% |
| 图形状 | 两族都是长任务串成的串行骨干 |
| 提前占位 / 领取批量 | `CLAIM_BATCH=1`：prefill +5.82% / +3.70%，方向相反 |
| 按任务限宽 | 去掉短路后规则命中（task 19 跨度 424,553 → 11,238），makespan 不动 |
| 票据分组 | 单组：task 15 跨度塌 122×，停顿原样搬到 task 16，makespan 不动 |
| 每块整缓存失效 | 整行删掉只值 0.8%，`decode_attn_swa` 反而更受益（−1.23%） |
| MIX 认领间隔 | maxgap prefill 487 µs vs decode 1,089 µs，decode 更大却更快 |
| 持有位置的 lane 不认领 MIX | `c2a_reuse` 慢 5.49%，其余持平 |
| 同层并列按 t-level 排序 | 顺序表如期改变（开局可运行块 1 → 28），耗时只动 0.29%，九个模块整体中性偏负 |
| 有未结清 MIX 认领时不抽新位置 | 结果算错（golden 不匹配、一例段错误）。挂住的 `claim.position` 让该 lane 再不抽位置，它那份工作无人认领。时间数据作废 |

**一个贯穿的事实**：领取批量、按任务限宽、票据分组、缓存失效四个旋钮全转过，
单个任务的跨度动了 40~120 倍，模块 makespan 始终在 ±1% 内
（2,350,972 ~ 2,372,366）。被挪动的停顿是守恒的——修好一个任务，同样长度的
停顿就落到下一个任务头上。

### 修法方向

成因已定位，但两个最直接的修法都试过且都不行，而且它们的失败方式指出了一个真正的
环形约束：

- **arm 之前要求不持有位置**（"持有位置的 lane 不认领 MIX"）：无效。`mix_tick` 在
  `pending` 检查之前，lane 是先 arm 后抽位置的，守卫不触发。
- **持有 MIX 认领时不抽位置**：结果算错。挂住的 `claim.position` 让该 lane 再不抽
  位置，它那份列表工作无人认领。

把两条合起来（arm 之后不抽、抽了之后不 arm）会让认领者 lane 在被未就绪位置阻塞时
永不 arm，而那些位置恰恰要等 MIX 任务完成才就绪——**列表位置的就绪依赖 MIX 完成，
MIX 的推进依赖认领者空闲**，直接连成环。

所以修法不在"何时 arm / 何时抽"这条轴上，而在**让 lane 能退回一个未就绪的位置**。
票据是单调计数器，没有退回路径；要支持退回，位置的认领就得从"票据发区间"改成
"按位置 CAS"，并重新论证无死锁（现有论证靠"阻塞在自己范围内最低的位置"成立，见
`percounter_claim_range`）。代价是原子操作变多，收益未评估。

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
qwen3 也是：它的 sys wall 只比 device wall 高约 0.9 ms，设备端省下的 6.76 ms
几乎原样传到了端到端（22.9% → 22.1%）。它是目前测过的唯一一个满足这条的真实负载。

### 真实模型的降幅远小于同规模的合成用例

`pa_b64` 与 qwen3 的 kernel 数几乎一样（16384 对 16849），降幅却差了四倍：
92.2% 对 22.9%。换算成每个 kernel 省下的时间，`pa_b64` 省 1.12 µs，
qwen3 只省 0.40 µs——不只是"固定开销被摊薄"，省下的绝对量本身就少得多。

一个可能的成因：合成用例的 kernel 几乎不干活，legacy 逐个派发的开销就是
全部的关键路径；而 qwen3 的 kernel 是真实算子，执行时间足够长，
legacy 可以在当前 kernel 还在算的时候派发下一个，派发开销被计算盖住了一部分。
**这是推断，未经测量**——要坐实需要 AICore 侧的窗口分解，本轮没做。

但无论成因如何，有一条是确定的：上面 15 个合成 case 的降幅不能用来预期
真实模型上的收益。真实模型这里是 22.9%。

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

### resident 跑不了 graph-execution 的图

qwen3 在 HBG 上以 Graph 形式执行。显式指定 resident 时 bind 被拒，
报 `a graph-execution run needs AICPU scheduling`。和 SPMD 那条一样，
这是 resident 的能力边界，不是测量失败：默认选择遇到这种图会落回 legacy。

这次失败顺带给出了一个意外的佐证——该次运行的输出张量保持为运行前灌入的 NaN，
比对如实报出全部 81920 个不一致，说明灌毒防护确实在起作用。

### comm domain 的 64 字节契约，pypto 不满足

simpler 的 comm domain 把具名 buffer 按声明顺序紧挨着排进一个跨 rank 的对称窗口，
要求每个 `CommBufferSpec.nbytes` 和 `window_size` 都是 64 的倍数，自己不做任何补齐
（`python/simpler/comm_domain.py:46-55`）。64 是缓存维护块的大小：一次 `dcci`
覆盖整块，两个 buffer 共享一块时，一个核刷自己的会连带冲掉邻居里对端刚写进来的数据。

pypto 补齐到 32（它 `codegen/distributed/comm_layout.h` 的
`kCommBufferAlignmentBytes`，取自 A2/A3 的 FP16 peer MTE 粒度），于是 4 字节的
signal buffer 发出来是 32 字节宽，`allocate_domain` 直接拒绝。34 个模块里 26 个
倒在这里，**三个调度器一字不差地相同**——失败发生在 host 端 Python，
调度器那时还没被选出来。

这个数在 pypto 里只有一处使用，而且同一个表达式同时生成 `nbytes` 和 `window_size`，
所以改成 64 能保持两者相等；64 又是 32 的倍数，原有的 MTE/CCU 对齐和 FP16
remote TLOAD 尾部余量只多不少。修法在 pypto 侧，不在本仓库。
本页那 12 个模块的数据是用一个只作用于实验副本的补齐垫片拿到的。

### 重复运行时 `TaskOutputTensors` 取输出越界

补齐 64 字节后，12 个模块在**第二次运行**（`run_id=1`）失败，
报 `prepare_native_run failed with code -1000`，host 日志里是
`Assertion failed: index < output_count_`（`src/common/host_build_graph/types.h:116`）：
生成的编排向 `TaskOutputTensors` 要一个运行时没有物化的输出下标。
首次运行正常。同样与调度器无关。

### percounter 没有 deferred completion

HBG 的 AICPU 调度路径支持延迟完成：kernel 调
`save_expected_notification_counter()` 注册一个计数器条件后**直接返回让出核**，
由 AICPU 的 `async_wait_list.poll_and_complete()` 代为盯着
（`runtime/scheduler/scheduler_dispatch.cpp:1056`）。slab 指针在
`scheduler_cold_path.cpp:1003` 装进 AICPU 的 per-core payload，
`task_token` 每次派发由 `scheduler_dispatch.cpp:186` 填入。

percounter 的 lane 只写 `block_idx` / `block_num` / `sub_block_id`，
`scheduler_materialize_task_payload_resolved` 只写 `function_bin_addr` 和 `args[]`，
`async_ctx` 自始至终没人填。而 pypto 生成的延迟等待函数在
`async_ctx_is_deferred()` 为假时会 `trap()`——它假定"受支持的派发器总会提供
有效的 task token 和 slab"。

所以**用了这条路径的 pypto 模块在 percounter 下不是变慢，是 trap**。
已构建的模块里用它的恰好三个：`decode_fwd`（整个 40 层骨干）、`moe`、`layer_group`。
其余模块（含本页的 `lm_head`）用的是内联阻塞的 `pto::comm::TWAIT`，不受影响。

### OUT 方向的张量从不初始化到设备上

`ChildMemoryTaskArgs.add` 给每个张量都分配了设备内存，但方向为 `OUT` 时
跳过 H2D 拷贝（`if direction != D.OUT: self.worker.copy_to(buf, host)`），
而分配走的是 `aclrtMalloc`，不清零。于是 `OUT` 张量在运行前的设备字节是
那块内存上一次的残留，用例在 host 侧准备的初值（qwen3 decode 是
`out = torch.zeros([BATCH, HIDDEN])`）永远到不了设备。一次没有写满自己输出的
运行，会拿残留数据去和参考值比对，并有可能因此通过。`INOUT` 张量
（`k_cache`、`v_cache`）会上传，不受影响。

本次测试靠运行前灌 NaN 规避了它；尚未观察到它实际造成过误判，
这里记录的是一个正确性信号的缺口，不是已复现的故障。

## 复现

脚本在 [`tools/percounter-perf/`](../../../tools/percounter-perf/README.md)，三步：

```bash
python tools/percounter-perf/percounter_perf.py setup  --name g --source <干净检出>
python tools/percounter-perf/percounter_perf.py run    --name g --out <波次目录>
python tools/percounter-perf/percounter_perf.py report --out <波次目录>
```

`setup` 把一份干净检出 rsync 成实验副本再打补丁——**补丁只作用于那份副本，
不改源码树**。其中 `force_sched_patch.py` 是不可省的那个：仓库里的
`SIMPLER_HBG_SCHEDULER_TYPE` 只认 `percounter` 和不设，没法指定 legacy 或
resident，而且 percounter 遇到自己跑不了的图会悄悄落回默认选择。
`counter_patch.py` 与 `window_patch.py` 提供 kernel span 那一列读的 lane 计数器，
`cases/test_pa_exp.py` 提供仓库里没有的 `pa_b16` / `pa_b64` 两个大 shape。

`run` 在一次设备独占锁里跑完整个 plan，先过 `onboard-arch-precheck`，
每次运行的 `ASCEND_PROCESS_LOG_PATH` 指到它自己的目录。两个开关：

| 开关 | 取值 | 含义 |
| ---- | ---- | ---- |
| `--cases` | `synthetic` / `graph` / `deepseek` / `all` | `graph` 即整图例子 qwen3，`all` 为默认 |
| `--golden` | `skip` / `compare` | `compare` 为默认，额外跑一个正确性波次 |
| `--deepseek` | `runnable` / `all` | 默认只跑记录为能跑的那些 |
| `--comm-pad64` | 开关 | 把 pypto 的 comm buffer 补齐到 64 字节 |

`deepseek` 组还需要 `--pypto-lib` 和 `--pypto-venv` 两个路径：这组跑的是 pypto-lib
的模块，而 pypto 的编译核心是 Python 3.10 的扩展，和实验副本自己的 venv 不是同一个
解释器。该组只跑 legacy 与 percounter 两方，原因见“已知问题”。

`compare` 下**灌毒是无条件开的**，整图例子还带一个负对照。时延波次从不开 golden：
参考值的计算会落进被测区间。本页正文的正确性表就来自 `compare`。

至于本页提到的那次 torch 参考锚点，脚本没有把它做成一个模式——它比跨调度器比对
慢约 48 倍，不适合随波次跑，需要时单独跑一次即可。
