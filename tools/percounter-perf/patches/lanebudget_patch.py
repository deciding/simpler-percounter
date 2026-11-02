#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Where a core type's lane-time goes: MIX rendezvous, list work, or blocked on a claimed position.

Applies on top of `counter_patch.py count` and `window_patch.py`.

Every other probe here is keyed by task or by block. That cannot answer why a
64-block AIV task whose blocks are each claimed promptly and run in 65 us still
takes 4,090 us end to end: the blocks are handed out slowly, so the question is
what the pool is doing instead, and the pool is the dimension none of the others
measure.

Four sums per core type, each accumulated per lane and folded at lane exit:

  mix    cycles inside a MIX rendezvous -- `percounter_run_subtask`, this lane's
         own subtask of a task that occupies two or three lanes of its cluster
  list   cycles inside ordinary list work -- `percounter_run_task`
  wait   cycles blocked in the producer-counter spin, holding a claimed position
         whose task is not ready
  life   lane lifetime, so the three above can be read as fractions and whatever
         is left is loop overhead: drawing, scanning, MIX arming

On `prefill_attn_swa` eight MIX tasks run back to back across the stretched
task's whole window, which makes `mix` the leading explanation -- a MIX block
occupies one AIC and one AIV lane of its cluster, so a MIX backbone can hold the
AIV pool away from the AIV list. `wait` tells the competing story: lanes parked on
positions of later, unready tasks. The two call for different repairs, and the
split decides which.

usage: lanebudget_patch.py <checkout>
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


E = "src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h"

# lb[0] mix, [1] list, [2] wait, [3] life, [4] 1 while serving a MIX subtask.
sub(E, "    uint64_t *kx;\n};", "    uint64_t *kx;\n    uint64_t *lb;\n};")

sub(
    E,
    "    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n",
    "    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n"
    "    uint64_t lb_slots[5] = {};\n    cfg.lb = lb_slots;\n",
)

# --- which path a kernel came from -----------------------------------------
sub(
    E,
    "    scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(cfg.payload));\n"
    "    __gm__ int32_t *kernel_ids = percounter_kernel_ids(task);\n",
    "    scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(cfg.payload));\n"
    "    cfg.lb[4] = 0;\n"
    "    __gm__ int32_t *kernel_ids = percounter_kernel_ids(task);\n",
)
sub(
    E,
    "    scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(cfg.payload));\n"
    "    const int32_t kernel_id = percounter_kernel_ids(task)[subtask_slot];\n",
    "    scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(cfg.payload));\n"
    "    cfg.lb[4] = 1;\n"
    "    const int32_t kernel_id = percounter_kernel_ids(task)[subtask_slot];\n",
)

KTIME = (
    "    {\n"
    "        const uint64_t e = get_sys_cnt_aicore();\n"
    "        cfg.kx[0] += 1;\n"
    "        cfg.kx[1] += e - *kernel_start;\n"
    "        if (cfg.kx[3] == 0) cfg.kx[3] = *kernel_start;\n"
    "        cfg.kx[4] = e;\n"
    "    }\n"
)
sub(E, KTIME, KTIME[:-6] + "        cfg.lb[cfg.lb[4] != 0 ? 0 : 1] += e - *kernel_start;\n    }\n")

# The blocked-wait total is already kept: the fanin wait closes with
# `if (waited) cfg.kx[2] += get_sys_cnt_aicore() - wait_t0;`.

# --- lifetime, and the fold ------------------------------------------------
sub(
    E,
    "    percounter_lane_loop(cfg, timeout_cycles);\n",
    "    const uint64_t lane_from = get_sys_cnt_aicore();\n"
    "    percounter_lane_loop(cfg, timeout_cycles);\n"
    "    lb_slots[3] = get_sys_cnt_aicore() - lane_from;\n    lb_slots[2] = kx[2];\n",
)

EXP_FLUSH = (
    "    for (int i = 0; i < 7; ++i) {\n"
    "        if (exp_counts[i] != 0)\n"
    "            scheduler_gm_fetch_add(reinterpret_cast<__gm__ uint64_t *>(control->resolved_reserved)[i], exp_counts[i]);\n"
    "    }\n"
)
LB_FLUSH = (
    "    {\n"
    "        __gm__ uint64_t *g = reinterpret_cast<__gm__ uint64_t *>(&cfg.tickets[8]);\n"
    "        for (int i = 0; i < 4; ++i) {\n"
    "            if (lb_slots[i] != 0) scheduler_gm_fetch_add(g[i + 1], lb_slots[i]);\n"
    "        }\n"
    "        scheduler_gm_fetch_add(g[5], uint64_t{1});\n"
    "    }\n"
)
sub(E, EXP_FLUSH, EXP_FLUSH + LB_FLUSH)

A = "src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp"
ANCHOR = (
    "            for (int i = 0; i < 7; ++i)\n"
    "                k[i] = 0;\n"
    "            cache_flush_range(k - 1, 64);\n"
)
READ = (
    "            {\n"
    "                uint64_t *g = reinterpret_cast<uint64_t *>(\n"
    "                    reinterpret_cast<uint8_t *>(control) + control->tickets_offset +\n"
    "                    t * simpler::hbg::percounter::PERCOUNTER_TICKET_CAPACITY * 64 + 8 * 64\n"
    "                );\n"
    "                cache_invalidate_range(g, 64);\n"
    "                if (g[5] != 0) {\n"
    "                    LOG_WARN(\n"
    "                        \"EXPL type=%d lanes=%\" PRIu64 \" mix=%\" PRIu64 \" list=%\" PRIu64\n"
    "                        \" wait=%\" PRIu64 \" life=%\" PRIu64,\n"
    "                        t, g[5], g[1], g[2], g[3], g[4]\n"
    "                    );\n"
    "                }\n"
    "                for (int i = 1; i < 8; ++i)\n"
    "                    g[i] = 0;\n"
    "                cache_flush_range(g, 64);\n"
    "            }\n"
)
sub(A, ANCHOR, ANCHOR + READ)

print("patched lanebudget")
