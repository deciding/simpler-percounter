"""A lane holding a list position does not arm a MIX rendezvous.

Diagnostic. `percounter_lane_loop` calls `percounter_mix_tick` before it tests
`pending`, so a lane that already holds an unready position can take a MIX block
and spend that task's whole kernel inside it. The position it holds becomes
runnable meanwhile and no other lane can take it, because it is claimed.

In `prefill_attn_c2a_reuse` that is the whole of the deficit's first 350 us:
every AIV lane draws at once, 63 of the first 64 positions belong to tasks that
are not ready, and the lane holding position 3 (task 15, 4 blocks) disappears
into task 7's 453 us MIX kernel.

Deadlock risk is real and this is why it is a diagnostic: a MIX rendezvous whose
claimer never arms leaves its peer waiting. Read the run's pass/fail, not only
its time.

usage: mixhold_patch.py <checkout>
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
E = "src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h"
p = root / E
s = p.read_text()
OLD = "        percounter_mix_tick(cfg, &state);\n"
if s.count(OLD) != 1:
    raise SystemExit("expected 1 mix_tick call site, found %d" % s.count(OLD))
p.write_text(s.replace(OLD, "        if (pending == 0) percounter_mix_tick(cfg, &state);\n"))
print("patched mixhold")
