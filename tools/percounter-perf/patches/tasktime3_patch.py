#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-task first/last device stamps for BOTH schedulers, 64 task ids, no folding.

Applies on top of `counter_patch.py count` and `window_patch.py`. It is
`tasktime2_patch.py`'s percounter half -- which holds 64 tasks because it writes
into each task's own counter line rather than into the eight spare ticket lines --
joined to `tasktime_patch.py`'s legacy half, raised to the same 64.

The question it answers is the one a single scheduler's numbers cannot: a wide
task that runs at 22 of 64 lanes under percounter is only evidence of a defect if
the same task packs better under the AICPU dispatcher. Comparing percounter's
prefill against percounter's decode instead confounds the scheduler with the
graph, which is how the draw-ahead hypothesis survived three measurements before
a `PERCOUNTER_CLAIM_BATCH = 1` A/B killed it.

Storage:

  percounter : words 6 and 7 of each task's `PercounterCounter`, which lie inside
               `reserved[20]` and are unused. The lane accumulates per
               `task.local` and flushes with min/max CAS at lane exit; the AICPU
               walks `counters_offset` once per run and prints `EXPS`.
  legacy     : an AICPU-side table, since the AICPU is the dispatcher there.
               `complete_slot_task` already has the slot, the dispatch stamp and
               the finish stamp together, and the stamps are gated only on a
               chip-swimlane level these runs do not set, so the gates come off.
               Readout is `LegacyAicpuExecutor::deinit` -- legacy enters through
               `legacy_aicpu_execute` and never reaches `AicpuExecutor`.

Clocks differ by tier (AICore `get_sys_cnt_aicore`, AICPU `get_sys_cnt_aicpu`), so
a stamp is comparable inside one scheduler's timeline, never across the two.
Spans and gaps are. The legacy stamps are taken unconditionally rather than behind
their level gate, which costs a legacy run a few percent it does not have in a
clean build -- that inflates legacy's spans, so it biases against finding legacy
the better packer. Take no latency figures from this build.

usage: tasktime3_patch.py <checkout>
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


TASKS = 64
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
sub(
    E,
    KTIME,
    KTIME[:-6]
    + (
        f"        if (task.local >= 0 && task.local < {TASKS}) {{\n"
        "            uint64_t *s = &cfg.tt[static_cast<uint64_t>(task.local) * 2];\n"
        "            if (s[0] == 0 || *kernel_start < s[0]) s[0] = *kernel_start;\n"
        "            if (e > s[1]) s[1] = e;\n"
        "        }\n"
        "    }\n"
    ),
)

sub(
    E,
    "    uint64_t kx[5] = {};\n    cfg.kx = kx;\n",
    f"    uint64_t kx[5] = {{}};\n    cfg.kx = kx;\n    uint64_t tt[{TASKS} * 2] = {{}};\n    cfg.tt = tt;\n",
)

FLUSH_TAIL = (
    "        for (uint64_t old = scheduler_gm_query(w[6]); kx[4] > old;) {\n"
    "            const uint64_t seen = scheduler_gm_compare_exchange(w[6], old, kx[4]);\n"
    "            if (seen == old) break;\n"
    "            old = seen;\n"
    "        }\n"
    "    }\n"
)
sub(
    E,
    FLUSH_TAIL,
    FLUSH_TAIL
    + (
        f"    for (int32_t i = 0; i < {TASKS}; ++i) {{\n"
        "        if (tt[i * 2] == 0) continue;\n"
        "        __gm__ uint64_t *s = reinterpret_cast<__gm__ uint64_t *>(&cfg.counters[i]) + 6;\n"
        "        for (uint64_t old = scheduler_gm_query(s[0]); old == 0 || tt[i * 2] < old;) {\n"
        "            const uint64_t seen = scheduler_gm_compare_exchange(s[0], old, tt[i * 2]);\n"
        "            if (seen == old) break;\n"
        "            old = seen;\n"
        "        }\n"
        "        for (uint64_t old = scheduler_gm_query(s[1]); tt[i * 2 + 1] > old;) {\n"
        "            const uint64_t seen = scheduler_gm_compare_exchange(s[1], old, tt[i * 2 + 1]);\n"
        "            if (seen == old) break;\n"
        "            old = seen;\n"
        "        }\n"
        "    }\n"
    ),
)

A = "src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp"
sub(
    A,
    "            for (int i = 0; i < 7; ++i)\n                k[i] = 0;\n            cache_flush_range(k - 1, 64);\n",
    "            for (int i = 0; i < 7; ++i)\n                k[i] = 0;\n            cache_flush_range(k - 1, 64);\n"
    "            if (t != 0) continue;\n"
    f"            for (uint64_t i = 0; i < control->task_count && i < {TASKS}; ++i) {{\n"
    "                uint64_t *line = reinterpret_cast<uint64_t *>(\n"
    "                    reinterpret_cast<uint8_t *>(control) + control->counters_offset + i * 64\n"
    "                );\n"
    "                cache_invalidate_range(line, 64);\n"
    "                if (line[6] == 0) continue;\n"
    "                LOG_WARN(\n"
    "                    \"EXPS percounter task=%\" PRIu64 \" first=%\" PRIu64 \" last=%\" PRIu64, i, line[6], line[7]\n"
    "                );\n"
    "                line[6] = 0;\n"
    "                line[7] = 0;\n"
    "                cache_flush_range(line, 64);\n"
    "            }\n",
)

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
    + "            \"EXPS legacy task=%d blocks=%\" PRIu64 \" first=%\" PRIu64 \" last=%\" PRIu64 \" busy=%\" PRIu64, i," + NL
    + "            s[0], s[1], s[2], s[3]" + NL
    + "        );" + NL
    + "        s[0] = s[1] = s[2] = s[3] = 0;" + NL
    + "    }" + NL,
)

print("patched tasktime3")
