#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Find where a percounter list orders two tied tasks against their readiness, and price each one.

Reads one wave's run directories, each from a build carrying `graphdump_patch.py`
(EXPG: per-task shape and producers), `orderdump_patch.py` (EXPO: list, first
position, b-level) and `tasktime3_patch.py` (EXPS: per-task device stamps).

A task that is not ready when a lane claims its position holds that lane until it
is. That is not automatically a defect: a lane parked on a task that gates a large
subgraph is a pre-placed trigger, and moving such a task later costs more than it
saves -- measured at 7.10% on `prefill_attn_c2a_reuse`. What decides the trade is
how much work hangs off the task, which is what b-level measures, so **a pair at
different levels is b-level's call and this tool does not second-guess it**.

Tasks at one level have no edge between them, so their order is free and b-level
has no opinion; today the tie goes to the lower task id. This reports the ties
that order gets backwards -- an earlier-placed task that becomes ready later than
one behind it -- and scores each by what moving it could recover:

    lanes_held = min(blocks_early, pool, blocks_late, blocks_before_early)
    score      = lanes_held * (ready_early - ready_late)

Each factor rules out a way the recovery cannot happen, and three of the four
were learned from a pair that scored without them:

- `blocks_early` and `pool`: a task cannot hold more lanes than it has positions,
  nor more than exist.
- `blocks_late`: the lanes it holds are only worth freeing if the task behind it
  has ready work to put on them. A 192-block task parked ahead of a **3-block**
  one is worth three lanes; two such pairs measured 0.17% and 0.40%.
- `blocks_before_early`: lanes only arrive together if the list region just ahead
  released them together. The pair worth 10.3% follows a 256-block task, so its
  64 lanes free at once and take all 64 of the next task's positions. Two pairs in
  `prefill_attn_swa` follow a 16-block task, and their later task was measured
  starting at its own ready time -- never held at all -- so swapping them cost
  0.5%. A third pair, predicted negative by this factor and then measured, came
  out at 0.48%.

**A high score is a candidate, not a finding**, and this model still overstates
the swa pairs by more than an order of magnitude. Confirm a candidate before
acting on it:

1. `blocktrace_patch.py` on the early task. A real hold shows a large
   claim-to-start gap against little kernel work in the same window.
2. `orderafter_patch.py` with `SIMPLER_EXP_ORDER_AFTER=early:late`, same-session
   A/B, reading `rc=` and `passed=` before any time.

usage: order_misfit.py <wave dir> [lanes-per-list]   default 32,64,32
"""

import re
import statistics
import sys
from pathlib import Path

EXPG = re.compile(
    r"EXPG task=(\d+) blocks=(\d+) mix=(\d) sync=(\d) type=(\d) subtasks=(\d+) fanin=(\d+) \[([0-9,]*)\]"
)
EXPO = re.compile(r"EXPO list=(\d+) pos=(\d+) task=(\d+) blocks=(\d+) blevel=(-?\d+)")
EXPS = re.compile(r"EXPS percounter task=(\d+) first=(\d+) last=(\d+)")

wave = Path(sys.argv[1])
lanes = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["32", "64", "32"])]


def read(run, pattern, where):
    text = []
    for sub in where:
        d = run / sub
        if d.is_dir():
            text += [f.read_text(errors="ignore") for f in d.rglob("*.log")]
    return [m.groups() for m in (pattern.search(l) for l in "\n".join(text).splitlines()) if m]


def task_end(run):
    """Median end stamp per task, normalised to each round's earliest start."""
    rows, cur = [], {}
    for t, a, b in read(run, EXPS, ["ascend"]):
        t = int(t)
        if t in cur:
            rows.append(cur)
            cur = {}
        cur[t] = (int(a), int(b))
    if cur:
        rows.append(cur)
    acc = {}
    for r in rows[1:] or rows:
        base = min(a for a, _ in r.values())
        for t, (a, b) in r.items():
            acc.setdefault(t, []).append(b - base)
    return {t: statistics.median(v) for t, v in acc.items()}


for run in sorted(p for p in (wave / "runs").iterdir() if p.is_dir()):
    graph = {int(g[0]): {"blocks": int(g[1]), "fanin": [int(x) for x in g[7].split(",") if x]}
             for g in read(run, EXPG, ["host"])}
    seats, level, first_pos = {}, {}, {}
    for l, pos, t, blocks, bl in read(run, EXPO, ["host"]):
        t = int(t)
        if t in seats:
            continue
        seats[t], first_pos[t], level[t] = int(l), int(pos), int(bl)
    end = task_end(run)
    if not graph or not seats or not end:
        print("%-30s  (missing EXPG/EXPO/EXPS)" % run.name)
        continue
    ready = {t: max((end.get(p, 0) for p in graph.get(t, {}).get("fanin", [])), default=0) for t in seats}
    found = []
    for l in sorted(set(seats.values())):
        pool = lanes[l] if l < len(lanes) else 64
        by_level = {}
        for t in (t for t in seats if seats[t] == l):
            by_level.setdefault(level[t], []).append(t)
        seats_here = sorted((first_pos[t], t) for t in seats if seats[t] == l)
        before = {}
        for k, (_, t) in enumerate(seats_here):
            before[t] = graph.get(seats_here[k - 1][1], {}).get("blocks", 0) if k else 0
        for bl, group in by_level.items():
            group.sort(key=lambda t: first_pos[t])
            for i, early in enumerate(group):
                for late in group[i + 1:]:
                    gap = ready[early] - ready[late]
                    if gap <= 0:
                        continue
                    held = min(graph.get(early, {}).get("blocks", 0), pool,
                               graph.get(late, {}).get("blocks", 0), before.get(early, 0))
                    found.append((held * gap, held, gap, l, bl, early, late, before.get(early, 0)))
    found.sort(reverse=True)
    print("=== %s" % run.name)
    if not found:
        print("    no tie ordered against readiness")
    for score, held, gap, l, bl, early, late, prev in found[:6]:
        print("    list=%d blevel=%-3d  move %d after %d  (%d lanes x %d ns = %.1e)  blocks %d/%d  before %d  pos %d/%d"
              % (l, bl, early, late, held, gap, score,
                 graph.get(early, {}).get("blocks", 0), graph.get(late, {}).get("blocks", 0),
                 prev, first_pos[early], first_pos[late]))
