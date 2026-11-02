"""A lane with an outstanding MIX claim does not draw a list position.

Diagnostic, and the mirror of `mixhold_patch.py`. `percounter_lane_loop` calls
`percounter_mix_tick` before it tests `pending`, so on its first iteration a
claimer lane arms its cluster and *then* draws -- which is why guarding the arm
on `pending == 0` changed nothing. The guard that matches the order is on the
draw.

What it is for: on `prefill_attn_c2a_reuse`, per-block stamps show the lane
holding one of task 15's four positions claims it at +1.7 us, runs two kernels
totalling 464,806 ns, and only then runs its own 2.4 us block at +470 us. The
other kernel is a 453 us MIX block of task 7. Its three sibling blocks run at
+123 us on lanes that ran one kernel of 2.4 us.

The cost to weigh against it: claimer lanes are one per cluster, so while MIX
work remains this keeps a whole class of lanes off the per-core-type lists.

usage: mixdraw_patch.py <checkout>
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
E = "src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h"
p = root / E
s = p.read_text()
OLD = "        if (pending == 0) {\n"
if s.count(OLD) != 1:
    raise SystemExit("expected 1 draw guard, found %d" % s.count(OLD))
NEW = "        if (pending == 0 && state.claim.task_id < 0 && state.claim.position < 0) {\n"
p.write_text(s.replace(OLD, NEW))
print("patched mixdraw")
