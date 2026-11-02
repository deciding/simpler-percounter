#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-task device timestamps for both schedulers, to locate a serial chain's cost.

Applies on top of `window_patch.py`. Both halves report the same four numbers per
graph-local task id -- block count, earliest start, latest end, summed busy time --
so a chain's stage spans, and the gaps between them, read the same way on either
scheduler. A stage whose `busy / span` is far below its block count ran its blocks
in waves rather than at once, which is the shape a span alone cannot show.

The two halves attach at different tiers, because the schedulers hold the task id
in different places:

  percounter : the lane has `task.local` at launch, so the AICore executor
               accumulates per task and flushes at lane exit into ticket lines
               8..15. Those lines exist (PERCOUNTER_TICKET_CAPACITY is 16) and no
               lane draws from them: a core type has lanes /
               PERCOUNTER_TICKET_GROUP_LANES live groups, at most 8 for 64 lanes.
  legacy     : the AICPU is the dispatcher, so the AICore side needs no change.
               `SchedulerContext::complete_slot_task` already receives the slot,
               the dispatch stamp and the finish stamp together; the stamps are
               only gated on a chip-swimlane level these runs do not set, so the
               gates come off and an AICPU-side table accumulates. Its readout
               belongs in `LegacyAicpuExecutor::deinit` -- legacy enters through
               `legacy_aicpu_execute`, a separate entry point that never reaches
               `AicpuExecutor`, where the percounter readout lives.

The clocks differ by tier: AICore reads get_sys_cnt_aicore(), AICPU reads
get_sys_cnt_aicpu(). A timestamp is therefore comparable within one scheduler's
own timeline, not across the two. Spans, gaps and busy totals are.

`busy` is not the same quantity on the two sides either -- percounter times the
kernel call, legacy times dispatch to FIN, which adds per-block dispatch and
completion latency. The two converge only where the kernel dwarfs that overhead,
so compare busy across schedulers for long kernels and never for short ones.

**Capacity is 8 task ids per core type**, folded `task_local & 7` into the eight
free ticket lines. Two tasks of one core type that are congruent mod 8 overwrite
each other, which the output shows as a block count that does not match the
graph. The legacy table is AICPU memory and holds 16 unfolded.

The AICPU stamps are taken on every dispatch and every completion rather than
behind their usual level gate, so a legacy run carries a few percent of overhead
it does not have in a clean build. Do not take latency figures from this build.

usage: tasktime_patch.py <checkout>
"""

import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit(__doc__.strip())

root = Path(sys.argv[1]).resolve()


def sub(rel, old, new, count=1):
    p = root / rel
    s = p.read_text()
    n = s.count(old)
    if n != count:
        raise SystemExit(f"{rel}: expected {count} of {old[:90]!r}, found {n}")
    p.write_text(s.replace(old, new))


TASKS = 16
NL = chr(10)

# --- percounter: the AICore lane, keyed by task.local ----------------------
E = "src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h"

sub(E, "    uint64_t *kx;\n};", "    uint64_t *kx;\n    uint64_t *tt;\n};")

KTIME = (
    "    {\n"
    "        const uint64_t e = get_sys_cnt_aicore();\n"
    "        cfg.kx[0] += 1;\n"
    "        cfg.kx[1] += e - *kernel_start;\n"
    "        if (cfg.kx[3] == 0) cfg.kx[3] = *kernel_start;\n"
    "        cfg.kx[4] = e;\n"
    "    }\n"
)
KTIME_TT = KTIME[:-6] + (
    f"        if (task.local >= 0 && task.local < {TASKS}) {{\n"
    "            uint64_t *s = &cfg.tt[static_cast<uint64_t>(task.local) * 4];\n"
    "            s[0] += 1;\n"
    "            if (s[1] == 0 || *kernel_start < s[1]) s[1] = *kernel_start;\n"
    "            if (e > s[2]) s[2] = e;\n"
    "            s[3] += e - *kernel_start;\n"
    "        }\n"
    "    }\n"
)
sub(E, KTIME, KTIME_TT)

sub(
    E,
    "    uint64_t kx[5] = {};\n    cfg.kx = kx;\n",
    f"    uint64_t kx[5] = {{}};\n    cfg.kx = kx;\n    uint64_t tt[{TASKS} * 4] = {{}};\n    cfg.tt = tt;\n",
)

FLUSH_TAIL = (
    "        for (uint64_t old = scheduler_gm_query(w[6]); kx[4] > old;) {\n"
    "            const uint64_t seen = scheduler_gm_compare_exchange(w[6], old, kx[4]);\n"
    "            if (seen == old) break;\n"
    "            old = seen;\n"
    "        }\n"
    "    }\n"
)
TT_FLUSH = (
    f"    for (int32_t i = 0; i < {TASKS}; ++i) {{\n"
    "        if (tt[i * 4] == 0) continue;\n"
    "        __gm__ uint64_t *s = reinterpret_cast<__gm__ uint64_t *>(&cfg.tickets[8 + (i & 7)]);\n"
    "        scheduler_gm_fetch_add(s[1], tt[i * 4]);\n"
    "        scheduler_gm_fetch_add(s[4], tt[i * 4 + 3]);\n"
    "        for (uint64_t old = scheduler_gm_query(s[2]); old == 0 || tt[i * 4 + 1] < old;) {\n"
    "            const uint64_t seen = scheduler_gm_compare_exchange(s[2], old, tt[i * 4 + 1]);\n"
    "            if (seen == old) break;\n"
    "            old = seen;\n"
    "        }\n"
    "        for (uint64_t old = scheduler_gm_query(s[3]); tt[i * 4 + 2] > old;) {\n"
    "            const uint64_t seen = scheduler_gm_compare_exchange(s[3], old, tt[i * 4 + 2]);\n"
    "            if (seen == old) break;\n"
    "            old = seen;\n"
    "        }\n"
    "    }\n"
)
sub(E, FLUSH_TAIL, FLUSH_TAIL + TT_FLUSH)

# --- legacy: uncover the AICPU stamps these runs never ask for -------------
D = "src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_dispatch.cpp"
sub(
    D,
    "    if (chip_swimlane_level_ >= ChipSwimlaneLevel::SCHEDULE_TIMING) {\n"
    "        dispatch_timestamp_slot =\n"
    "            to_pending ? &core_exec_state.pending_dispatch_timestamp : &core_exec_state.running_dispatch_timestamp;\n"
    "    }\n",
    "    dispatch_timestamp_slot =\n"
    "        to_pending ? &core_exec_state.pending_dispatch_timestamp : &core_exec_state.running_dispatch_timestamp;\n",
)
sub(
    D,
    "                if (chip_swimlane_level_ >= ChipSwimlaneLevel::SCHEDULE_TIMING) {\n"
    "                    dispatch_ts = get_sys_cnt_aicpu();\n"
    "                }\n",
    "                dispatch_ts = get_sys_cnt_aicpu();\n",
)

C = "src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion.cpp"
sub(
    C,
    "        if (chip_swimlane_level_ >= ChipSwimlaneLevel::SCHEDULE_TIMING && (t.pending_done || t.running_done)) {\n"
    "            finish_ts = get_sys_cnt_aicpu();\n"
    "        }\n",
    "        if (t.pending_done || t.running_done) {\n"
    "            finish_ts = get_sys_cnt_aicpu();\n"
    "        }\n",
)

# One table for every AICPU thread; threads complete different cores of the same
# task, so the stamps are folded with atomics rather than per-thread copies.
TABLE = f"""
uint64_t exp_legacy_task[{TASKS}][4] = {{}};

static void exp_legacy_record(uint64_t local, uint64_t dispatch_ts, uint64_t finish_ts) {{
    if (local >= {TASKS} || dispatch_ts == 0 || finish_ts == 0) return;
    uint64_t *s = exp_legacy_task[local];
    __atomic_fetch_add(&s[0], uint64_t{{1}}, __ATOMIC_RELAXED);
    __atomic_fetch_add(&s[3], finish_ts - dispatch_ts, __ATOMIC_RELAXED);
    for (uint64_t old = __atomic_load_n(&s[1], __ATOMIC_RELAXED); old == 0 || dispatch_ts < old;) {{
        if (__atomic_compare_exchange_n(&s[1], &old, dispatch_ts, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
    }}
    for (uint64_t old = __atomic_load_n(&s[2], __ATOMIC_RELAXED); finish_ts > old;) {{
        if (__atomic_compare_exchange_n(&s[2], &old, finish_ts, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
    }}
}}
"""
ANCHOR = "void SchedulerContext::complete_slot_task(\n    ChipTaskSlotState &slot_state,"
sub(C, ANCHOR, TABLE.lstrip(NL) + NL + ANCHOR)
sub(
    C,
    "#if SIMPLER_DFX\n    auto &chip_swimlane = sched_chip_swimlane_[thread_idx];\n",
    "#if SIMPLER_DFX\n"
    "    exp_legacy_record(TaskId::to_uint64(slot_state.to_descriptor().task_id) & 0xffffffffULL, dispatch_ts, finish_ts);\n"
    "    auto &chip_swimlane = sched_chip_swimlane_[thread_idx];\n",
)

# --- readouts --------------------------------------------------------------
LEG = "src/a5/runtime/host_build_graph/aicpu/aicpu_legacy_executor.cpp"
LEG_ANCHOR = "void LegacyAicpuExecutor::deinit(Runtime *runtime) {" + NL
sub(
    LEG,
    LEG_ANCHOR,
    f"extern uint64_t exp_legacy_task[{TASKS}][4];" + NL + NL
    + LEG_ANCHOR
    + f"    for (int i = 0; i < {TASKS}; ++i) {{" + NL
    + "        uint64_t *s = exp_legacy_task[i];" + NL
    + "        if (s[0] == 0) continue;" + NL
    + "        LOG_WARN(" + NL
    + "            \"EXPT legacy task=%d blocks=%\" PRIu64 \" first=%\" PRIu64 \" last=%\" PRIu64 \" busy=%\" PRIu64, i," + NL
    + "            s[0], s[1], s[2], s[3]" + NL
    + "        );" + NL
    + "        s[0] = s[1] = s[2] = s[3] = 0;" + NL
    + "    }" + NL,
)

A = "src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp"
sub(
    A,
    "            for (int i = 0; i < 7; ++i)\n                k[i] = 0;\n            cache_flush_range(k - 1, 64);\n",
    "            for (int i = 0; i < 7; ++i)\n                k[i] = 0;\n            cache_flush_range(k - 1, 64);\n"
    f"            for (int i = 0; i < {TASKS}; ++i) {{\n"
    "                uint64_t *s = reinterpret_cast<uint64_t *>(\n"
    "                    reinterpret_cast<uint8_t *>(control) + control->tickets_offset +\n"
    "                    t * simpler::hbg::percounter::PERCOUNTER_TICKET_CAPACITY * 64 + (8 + (i & 7)) * 64\n"
    "                );\n"
    "                cache_invalidate_range(s, 64);\n"
    "                if (s[1] == 0) continue;\n"
    "                LOG_WARN(\n"
    "                    \"EXPT percounter type=%d task=%d blocks=%\" PRIu64 \" first=%\" PRIu64 \" last=%\" PRIu64\n"
    "                    \" busy=%\" PRIu64,\n"
    "                    t, i, s[1], s[2], s[3], s[4]\n"
    "                );\n"
    "                s[1] = s[2] = s[3] = s[4] = 0;\n"
    "                cache_flush_range(s, 64);\n"
    "            }\n",
)

print("patched tasktime")
