"""Print each percounter list as (list, first position, task id, blocks, b-level, t-level).

Host-side only. The sort key decides which tasks a lane pool meets first, and a
hand-computed level is not evidence about it: this prints the list the device
actually gets, so a claim about position 37 or position 64 can be read off
rather than derived.

usage: orderdump_patch.py <checkout>
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
S = "src/a5/runtime/host_build_graph/host/percounter/percounter_plan.h"
p = root / S
s = p.read_text()

OLD = """        uint64_t cursor = 0;
        for (int64_t id : eligible[l]) {
            const TaskInput &placed_task = tasks[static_cast<size_t>(id)];
"""
if s.count(OLD) != 1:
    raise SystemExit("anchor count %d" % s.count(OLD))

NEW = """        uint64_t cursor = 0;
        for (int64_t id : eligible[l]) {
            const TaskInput &placed_task = tasks[static_cast<size_t>(id)];
            LOG_WARN(
                "EXPO list=%u pos=%llu task=%lld blocks=%d blevel=%d", l,
                static_cast<unsigned long long>(cursor), static_cast<long long>(id),
                static_cast<int>(placed_task.block_num), blevel[static_cast<size_t>(id)]
            );
"""
p.write_text(s.replace(OLD, NEW))
print("patched orderdump")
