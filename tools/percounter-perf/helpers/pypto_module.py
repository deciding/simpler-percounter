#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run one pypto-lib model module's `validate()` against a chosen simpler runtime.

usage: pypto_module.py <trb|hbg> <models/.../module.py> [the module's own args...]

pypto picks the runtime from the ambient PassContext and nothing else --
`_resolve_runtime()` reads `PassContext.current()` and falls back to
TENSORMAP_AND_RINGBUFFER -- so establishing the context around `validate()` is
the whole switch. No pypto-lib file needs an edit for it, and the jit cache
keys on the runtime, so an artifact compiled for one is never handed to the
other.

The module's own flags pass straight through. They have to reach `sys.argv`
before the import, because `config.py` parses `--tp` / `--ep` / `--experts` /
`--max-batch` at import time rather than from the parser.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

RUNTIMES = ("trb", "hbg")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[0] not in RUNTIMES:
        sys.exit(__doc__.strip())
    runtime, module_path, *rest = argv

    # config.py reads these out of sys.argv while being imported, so the module
    # args must be in place before the import below, not just passed to validate().
    sys.argv = [module_path] + rest

    if os.environ.get("SIMPLER_EXP_COMM_PAD64") == "1":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import comm_pad

        comm_pad.install()

    from pypto.ir import PassContext
    from pypto.pypto_core import passes as _passes

    kind = (_passes.RuntimeKind.HOST_BUILD_GRAPH if runtime == "hbg"
            else _passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER)

    path = Path(module_path).resolve()
    spec = importlib.util.spec_from_file_location(f"_pypto_module_{path.stem}", path)
    if spec is None or spec.loader is None:
        sys.exit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with PassContext([], runtime=kind):
        spec.loader.exec_module(module)
        if not hasattr(module, "validate"):
            sys.exit(f"{path.name} has no validate() entry")
        result = module.validate(rest)

    passed = getattr(result, "passed", None)
    error = getattr(result, "error", None)
    print(f"[pypto_module] {path.name} runtime={runtime} passed={passed} error={error}", flush=True)
    return 0 if passed is not False else 1


if __name__ == "__main__":
    sys.exit(main())
