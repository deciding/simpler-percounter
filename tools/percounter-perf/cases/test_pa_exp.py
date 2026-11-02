"""Experiment-only paged attention sizes for the scalability study. Never committed."""

import importlib.util
import sys
from pathlib import Path

from simpler_setup import scene_test

_spec = importlib.util.spec_from_file_location("_pa_base", Path(__file__).with_name("test_paged_attention.py"))
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)

_BASE = {
    "num_heads": 16,
    "kv_head_num": 1,
    "head_dim": 128,
    "block_size": 128,
    "max_model_len": 32768,
    "dtype": "bfloat16",
}


def _case(name, batch, context_len):
    tasks = batch * (context_len // 128 * 4 + 1)
    window = 1 << max(17, (tasks - 1).bit_length())
    return {
        "name": name,
        "platforms": ["a5"],
        "manual": True,
        "config": {"runtime_env": {"ring_task_window": window}},
        "params": {**_BASE, "batch": batch, "context_len": context_len},
    }


@scene_test(level=2, runtime="host_build_graph")
class TestPaExp(_base.TestPagedAttentionHostBuildGraphA5):
    CASES = [_case(f"B{b}", b, 8192) for b in (16, 32, 64, 128, 256, 512)] + [
        _case(f"L{c}", 64, c) for c in (1024, 2048, 4096, 16384)
    ]


if __name__ == "__main__":
    from simpler_setup import SceneTestCase

    SceneTestCase.run_module(__name__)
