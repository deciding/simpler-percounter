"""A lane whose cluster has a MIX rendezvous in flight does not draw a list position.

Diagnostic. Two earlier forms of this idea missed because they keyed on the wrong
state:

  mixhold_patch.py   guards the arm on `pending == 0`. `percounter_mix_tick` runs
                     before the draw in the same iteration, so `pending` is still
                     zero when the lane arms. Measured: blocks held behind a long
                     MIX kernel go from 28% to 24% -- the guard barely fires --
                     while the heavy MIX task slips 132 us and the makespan 168 us.
  mixdraw_patch.py   guards the draw on `claim.task_id` / `claim.position`. Those
                     name a MIX position this lane holds whose task is not ready;
                     `percounter_mix_arm` clears task_id the moment it arms. The
                     state persists while the MIX task is unready, so the lane
                     stops drawing list positions for that whole stretch and its
                     share of the list goes unclaimed -- wrong results.

The state that does mean "this cluster is committed to a rendezvous" is the
signal's armed generation, which is what the serve path waits on. A lane that
sees it armed is about to spend that kernel inside the rendezvous, whether it is
the claimer or its peer, so that is when it must not take list work. The hold
lasts one rendezvous and ends when the rendezvous does.

usage: mixarmed_patch.py <checkout>
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
NEW = (
    "        SCHEDULER_SSBUF PercounterMixSignal *mix_line = cfg.mix_signal;\n"
    "        const bool mix_in_flight =\n"
    "            cfg.mix_order_count != 0 &&\n"
    "            percounter_mix_armed(scheduler_ssbuf_load_acquire(&mix_line->armed_generation));\n"
    "        if (pending == 0 && !mix_in_flight) {\n"
)
p.write_text(s.replace(OLD, NEW))
print("patched mixarmed")
