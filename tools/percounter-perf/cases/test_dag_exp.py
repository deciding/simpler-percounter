"""Experiment-only multi_core_dag sizes for the scalability study. Never committed."""

import importlib.util
import sys
from pathlib import Path

from simpler_setup import scene_test

_spec = importlib.util.spec_from_file_location("_dag_base", Path(__file__).with_name("test_multi_core_dag.py"))
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)


def _case(name, graph_case, task_count):
    return {
        "name": name,
        "platforms": ["a5"],
        "manual": True,
        "config": {"runtime_env": {"ring_task_window": 1 << max(17, task_count.bit_length())}},
        "params": {"graph_case": graph_case, "task_count": task_count},
    }


@scene_test(level=2, runtime="host_build_graph")
class TestDagExp(_base.TestHbgMultiCoreDag):
    CASES = [_case(f"random_{n}", 3, n) for n in (1024, 4096, 16384, 65536)] + [
        _case(f"fanin32_{n}", 1, n) for n in (1024, 4096, 16384)
    ] + [_case(f"fk{k}_4096", 100 + k, 4096) for k in (4, 8, 16, 32)]


if __name__ == "__main__":
    from simpler_setup import SceneTestCase

    SceneTestCase.run_module(__name__)
