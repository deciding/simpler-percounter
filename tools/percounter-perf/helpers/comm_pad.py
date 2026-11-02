#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Round a pypto-generated comm domain request up to simpler's 64-byte contract.

simpler carves a domain's named buffers consecutively out of one symmetric
window and rounds nothing: `comm_domain.py` requires every `CommBufferSpec`
nbytes and the window itself to be a multiple of 64, so that named buffers never
share a cache-maintenance block. A `dcci` on one buffer covers the whole 64-byte
block, so a neighbour sharing that block loses a peer's DMA or reads back stale
bytes.

pypto rounds to 32 instead -- `kCommBufferAlignmentBytes` in
`codegen/distributed/comm_layout.h`, a figure taken from the A2/A3 FP16 peer MTE
transfer granularity rather than from a5's cache line. Every a5 module that
carves a sub-64-byte signal buffer is rejected before a scheduler is ever
chosen.

Wrapping the allocation re-rounds to 64 without touching either tree. 64 is a
multiple of 32, so pypto's MTE/CCU alignment and its FP16 remote-TLOAD tail room
both still hold, with strictly more room. `count` stays the logical byte length,
so no tensor view changes; only the physical span grows.

This is a measurement vehicle for whether the 64-byte contract is the only thing
standing between these modules and a scheduler comparison. The fix belongs in
pypto's constant, not here.
"""

from __future__ import annotations

ALIGN = 64


def _align_up(n: int) -> int:
    return ((int(n) + ALIGN - 1) // ALIGN) * ALIGN


def install(align: int = ALIGN) -> None:
    """Make `Orchestrator.allocate_domain` and its siblings pad to `align`."""
    from simpler import orchestrator as _orch
    from simpler.task_interface import CommBufferSpec

    target = _orch.Orchestrator
    if getattr(target, "_comm_pad_installed", False):
        return

    def repad(kwargs):
        buffers = kwargs.get("buffers") or ()
        if not buffers:
            return kwargs
        padded, grew = [], False
        for b in buffers:
            nbytes = _align_up(b.nbytes)
            grew = grew or nbytes != int(b.nbytes)
            padded.append(
                CommBufferSpec(
                    str(b.name), str(b.dtype), int(b.count), nbytes,
                    bool(getattr(b, "load_from_host", False)),
                    bool(getattr(b, "store_to_host", False)),
                )
            )
        if not grew and int(kwargs.get("window_size", 0)) % align == 0:
            return kwargs
        kwargs = dict(kwargs)
        kwargs["buffers"] = padded
        # The window is exactly the sum of the physical spans in pypto's own
        # emitter, so recomputing it here keeps that identity rather than
        # imposing a new one.
        kwargs["window_size"] = max(_align_up(kwargs.get("window_size", 0)),
                                    sum(b.nbytes for b in padded))
        print(f"[comm_pad] repadded {len(padded)} buffer(s) to {align}B, "
              f"window_size={kwargs['window_size']}", flush=True)
        return kwargs

    for name in ("allocate_domain", "get_or_allocate_domain", "allocate_global_domain"):
        original = getattr(target, name, None)
        if original is None:
            continue

        def make(original):
            def wrapper(self, *args, **kwargs):
                return original(self, *args, **repad(kwargs))
            return wrapper

        setattr(target, name, make(original))

    target._comm_pad_installed = True
