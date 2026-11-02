#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Replay one order list against a model of the lane loop, so orders can be ranked without the device.

An on-device A/B costs about ten minutes; this costs milliseconds, which is what
makes searching over orders possible at all. It is only useful if it ranks orders
the way the device does, so **it calibrates against a measured run first and
prints the error** -- the absolute makespan does not have to be right, the
ordering of two candidates does.

The property it exists to reproduce is the one that costs percounter its prefill
deficit: a lane draws the next position of its list and is bound to it until that
position's task is ready, whatever else becomes ready meanwhile.

Model, and where it departs from the device:

- Two pools, AIC and AIV, sized from `--lanes`. A MIX block takes one of each for
  its duration. The device binds them to one cluster; this does not, so the model
  is **optimistic about MIX placement**.
- Each list is sharded into `--groups` tickets exactly as
  `percounter_group_position` shards it -- blocks of `--batch` positions dealt
  round robin -- and a lane draws only from its own group until that group's
  ticket passes the end, then moves to the next and never comes back.
- A lane draws `--batch` consecutive positions of its group's share and runs
  whichever of them is ready, blocking only when none is.
- Durations are per block and constant, taken from a measured run. Per-block
  variance, cache state and bandwidth contention are not modelled.

The first version had one ticket per list and drew one position at a time. It
got the calibration backwards: it ranked a measured 7.07% regression as a 4.6%
improvement, because with a single ticket and no batch a lane always reaches a
position as soon as it frees up, so moving a task later never costs anything.
Sharding and batching are what make a late position expensive, which is the whole
effect being modelled, so they are not optional detail.

usage: order_sim.py <model.json> [--lanes AIC,AIV] [--order FILE] [--measured NS]
"""

import argparse
import json
import sys


def shard(positions, groups, batch):
    """Deal positions into `groups` shares, blocks of `batch` round robin.

    The same cut `percounter_group_position` makes: block `c` of the list belongs
    to group `c % groups`, and a group's share is its blocks concatenated.
    """
    share = [[] for _ in range(groups)]
    for i, task in enumerate(positions):
        share[(i // batch) % groups].append(task)
    return share


def simulate(graph, order, dur, aic_lanes, aiv_lanes, groups=8, batch=4):
    """Return the makespan in ns for `order`, a list id -> sequence of task ids."""
    blocks_left = {t: g["blocks"] for t, g in graph.items()}
    complete = {t: False for t in graph}
    # A task in no list is published at bind: inline-completed, or filled by the
    # AICPU. Either way the lanes never run it, so it starts complete.
    in_list = {t for seq in order.values() for t in seq}
    for t in graph:
        if t not in in_list:
            complete[t] = True
            blocks_left[t] = 0

    def ready(t):
        return all(complete[p] for p in graph[t]["fanin"] if p in complete)

    # Positions, expanded from the order and sharded the way the device shards.
    share = {}
    for l, seq in order.items():
        positions = []
        for t in seq:
            positions.extend([t] * max(graph[t]["blocks"], 1))
        share[l] = shard(positions, groups, batch)
    head = {l: [0] * groups for l in share}

    # A lane holds up to `batch` positions and runs whichever is ready. `grp` is
    # how many groups past its own it has found spent; it never comes back.
    def new_lane(index):
        return {"own": index % groups, "grp": 0, "held": [], "until": None, "task": None}

    aiv = [new_lane(i) for i in range(aiv_lanes)]
    aic = [new_lane(i) for i in range(aic_lanes)]

    def draw(lane, l):
        """Take up to `batch` positions from the first group that still has any."""
        while lane["grp"] < groups:
            g = (lane["own"] + lane["grp"]) % groups
            q = share[l][g]
            at = head[l][g]
            if at < len(q):
                take = q[at:at + batch]
                head[l][g] = at + len(take)
                lane["held"] = list(take)
                return True
            lane["grp"] += 1
        return False

    now = 0.0
    guard = 0
    while True:
        guard += 1
        if guard > 4_000_000:
            raise SystemExit("simulation did not terminate; check the fanin for a cycle")

        progress = True
        while progress:
            progress = False
            # MIX first, as `percounter_mix_tick` runs before the draw: it needs
            # one idle lane from each pool, taken together.
            if 2 in share:
                while True:
                    a = next((x for x in aiv if x["until"] is None and not x["held"]), None)
                    c = next((x for x in aic if x["until"] is None and not x["held"]), None)
                    if a is None or c is None or not draw(a, 2):
                        break
                    c["held"] = list(a["held"])
                    progress = True
            for pool, l in ((aiv, 1), (aic, 0)):
                for lane in pool:
                    if lane["until"] is not None:
                        continue
                    if not lane["held"] and l in share:
                        if draw(lane, l):
                            progress = True
                    hit = next((t for t in lane["held"] if ready(t)), None)
                    if hit is not None:
                        lane["held"].remove(hit)
                        lane["task"] = hit
                        lane["until"] = now + dur.get(hit, 0.0)
                        progress = True

        running = [x["until"] for pool in (aiv, aic) for x in pool if x["until"] is not None]
        if not running:
            if all(not x["held"] for x in aiv) and all(not x["held"] for x in aic):
                return now
            raise SystemExit("deadlock in the model at t=%.0f; a held task never became ready" % now)
        now = min(running)
        for pool in (aiv, aic):
            for lane in pool:
                if lane["until"] is not None and lane["until"] <= now:
                    t = lane["task"]
                    # A MIX block occupies both pools, so it retires on both and
                    # must only be counted once.
                    if pool is aiv or not graph[t]["mix"]:
                        blocks_left[t] -= 1
                        if blocks_left[t] <= 0:
                            complete[t] = True
                    lane["until"] = None
                    lane["task"] = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--lanes", default="32,64")
    p.add_argument("--order", help="a file of 'list taskid taskid ...' lines; default is the measured one")
    p.add_argument("--measured", type=float, help="measured makespan in ns, to print the calibration error")
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()

    m = json.loads(open(args.model).read())
    graph = {int(k): v for k, v in m["graph"].items()}
    dur = {int(k): float(v) for k, v in m["dur"].items()}
    placed = {int(k): tuple(v) for k, v in m["order"].items()}

    if args.order:
        order = {}
        for line in open(args.order):
            parts = line.split()
            if parts:
                order[int(parts[0])] = [int(x) for x in parts[1:]]
    else:
        order = {}
        for t, (l, pos, _bl) in placed.items():
            order.setdefault(l, []).append((pos, t))
        order = {l: [t for _, t in sorted(v)] for l, v in order.items()}

    missing = [t for seq in order.values() for t in seq if t not in dur]
    if missing:
        print("warning: %d placed tasks have no duration and are modelled as free: %s"
              % (len(missing), missing[:8]), file=sys.stderr)

    aic_lanes, aiv_lanes = (int(x) for x in args.lanes.split(","))
    span = simulate(graph, order, dur, aic_lanes, aiv_lanes, args.groups, args.batch)
    print("simulated makespan = %.0f ns" % span)
    if args.measured:
        print("measured           = %.0f ns" % args.measured)
        print("error              = %+.1f%%" % (100 * (span - args.measured) / args.measured))


if __name__ == "__main__":
    main()
