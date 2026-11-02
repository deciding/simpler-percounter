"""A lane with an outstanding MIX claim does not draw a list position.

Diagnostic, and the mirror of `mixhold_patch.py`. `percounter_lane_loop` calls
`percounter_mix_tick` before it tests `pending`, so on its first iteration a
claimer lane arms its cluster and *then* draws -- which is why guarding the arm
on `pending == 0` changed nothing. The guard that matches the order is on the
draw.

What it is for: on `prefill_attn_c2a_reuse`, per-block stamps show the lane
holding one of the head task's four positions claims it at +1.7 us, runs two
kernels totalling 464,806 ns, and only then runs its own 2.4 us block at +470 us.
The other kernel is a 453 us MIX block. Its three sibling blocks run at +123 us
on lanes that ran one kernel of 2.4 us.

Two forms, because the first one tried was wrong and the way it was wrong is the
point:

  narrow  guard on `claim.task_id` alone -- the lane has armed a rendezvous and
          has not served it yet. Serving clears it, so the hold is bounded by one
          MIX kernel and the lane always comes back to the list.
  wide    also guard on `claim.position`, which names a MIX position drawn whose
          entry the AICPU has not published yet. That state can persist, and a
          lane in it never draws a list position again -- its share of the list
          goes unclaimed and the run computes the wrong answer. Kept so the
          failure stays reproducible: it reported 13-20% faster while failing its
          golden, because it had skipped work.

Neither form is a candidate repair. A claimer is one lane per cluster, so while
MIX work remains this keeps that lane off the per-core-type lists for the
rendezvous it has armed; whether that costs more than the hold it removes is what
the A/B measures. Read `rc=` and `passed=` before reading any time.

usage: mixdraw_patch.py <checkout> [narrow|wide]
"""
import sys
from pathlib import Path

if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] not in ("narrow", "wide")):
    raise SystemExit(__doc__.strip())

root = Path(sys.argv[1]).resolve()
mode = sys.argv[2] if len(sys.argv) == 3 else "narrow"

E = "src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h"
p = root / E
s = p.read_text()
OLD = "        if (pending == 0) {\n"
if s.count(OLD) != 1:
    raise SystemExit("expected 1 draw guard, found %d" % s.count(OLD))
GUARD = "state.claim.task_id < 0" if mode == "narrow" else "state.claim.task_id < 0 && state.claim.position < 0"
p.write_text(s.replace(OLD, "        if (pending == 0 && %s) {\n" % GUARD))
print("patched mixdraw %s" % mode)
