#!/usr/bin/env python3
"""Make an OUT-direction child-memory tensor's device bytes defined before a run.

ChildMemoryTaskArgs.add allocates an OUT tensor's device buffer and skips the
H2D copy, and the allocator does not zero, so an OUT tensor starts a run
holding whatever that memory last held. A run that does not write its whole
output then compares bytes an earlier run left behind.

SIMPLER_EXP_POISON_OUT=1 fills every OUT tensor with NaN before the run, so an
unwritten output fails the compare instead of agreeing with an earlier run's
leftovers. The comparison modes turn it on unconditionally.

The negative control that proves the compare is live lives in
helpers/qwen_dump.py under SIMPLER_EXP_POISON_DUMP, which clobbers `out` after
the run and before readback.

usage: poison_patch.py <checkout>
"""

import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()


def sub(rel, old, new, count=1):
    path = root / rel
    text = path.read_text()
    found = text.count(old)
    if found != count:
        raise SystemExit(f"{rel}: expected {count} occurrence(s) of the anchor, found {found}")
    path.write_text(text.replace(old, new))
    print(f"patched {rel}")


sub(
    "simpler_setup/scene_test.py",
    "class ChildMemoryTaskArgs:\n",
    "def _exp_poison_out():\n"
    '    return os.environ.get("SIMPLER_EXP_POISON_OUT", "") not in ("", "0")\n'
    "\n"
    "\n"
    "def _exp_poison_tensor(host):\n"
    "    import torch\n"
    "\n"
    "    poison = torch.empty_like(host)\n"
    "    if poison.is_floating_point():\n"
    '        poison.fill_(float("nan"))\n'
    "    else:\n"
    "        poison.view(torch.uint8).fill_(0xA5)\n"
    "    return poison.contiguous()\n"
    "\n"
    "\n"
    "class ChildMemoryTaskArgs:\n",
)

sub(
    "simpler_setup/scene_test.py",
    "        buf = self.worker.malloc(size)\n"
    "        try:\n"
    "            if direction != D.OUT:\n"
    "                self.worker.copy_to(buf, host)\n",
    "        buf = self.worker.malloc(size)\n"
    "        try:\n"
    "            if direction != D.OUT:\n"
    "                self.worker.copy_to(buf, host)\n"
    "            elif _exp_poison_out():\n"
    "                self.worker.copy_to(buf, _exp_poison_tensor(host))\n"
    '                logger.warning("EXP: poisoned OUT tensor %r with NaN before the run", name)\n',
)

