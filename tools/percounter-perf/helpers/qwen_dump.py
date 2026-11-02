"""Run the qwen3 decode driver and save its outputs before the device buffers are freed.

usage: python qwen_dump.py <out_dir> <main.py> [main.py args...]

Saves under <out_dir>:
  out.pt                      the decode output, whole
  k_written.pt / v_written.pt the KV rows this step writes, [layers, BATCH, KV_HIDDEN]
  k_sums.pt / v_sums.pt       per-layer float64 sums of the whole KV pool, which
                              catch a write anywhere else in it
The KV pools are about 7 GB each, too large to keep whole per run.
"""
import os
import runpy
import sys

import torch
import importlib
from simpler_setup.goldens.qwen3_14b_decode import (
    BATCH,
    BLOCK_SIZE,
    CACHE_ROWS,
    HIDDEN,
    KV_HIDDEN,
    N_LAYERS,
    NUM_PAGES,
    param_specs,
)

scene_test = importlib.import_module("simpler_setup.scene_test")
OUT_DIR = sys.argv[1]
SPECS = {spec.name: spec for spec in param_specs()}
_release = scene_test.ChildMemoryTaskArgs.release


def _host(self, name, dtype):
    host = torch.empty(tuple(SPECS[name].shape), dtype=dtype)
    self.worker.copy_from(host, self.buffers[name])
    return host


def _dump_then_release(self):
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.environ.get("SIMPLER_EXP_POISON_DUMP", "") not in ("", "0"):
        clobber = torch.full((BATCH, HIDDEN), float("nan"), dtype=torch.bfloat16).contiguous()
        self.worker.copy_to(self.buffers["out"], clobber)
        print("[dump] EXP: clobbered out before readback (negative control)", flush=True)
    out = _host(self, "out", torch.bfloat16)
    assert tuple(out.shape) == (BATCH, HIDDEN)
    torch.save(out, os.path.join(OUT_DIR, "out.pt"))
    slots = _host(self, "slot_mapping", torch.int32).long()
    torch.save(slots, os.path.join(OUT_DIR, "slot_mapping.pt"))
    for name, tag in (("k_cache", "k"), ("v_cache", "v")):
        cache = _host(self, name, torch.bfloat16)
        layers = cache.view(N_LAYERS, NUM_PAGES * BLOCK_SIZE, KV_HIDDEN)
        assert layers.shape[1] * KV_HIDDEN == CACHE_ROWS * cache.shape[1]
        torch.save(layers[:, slots, :].clone(), os.path.join(OUT_DIR, f"{tag}_written.pt"))
        torch.save(layers.double().sum(dim=(1, 2)), os.path.join(OUT_DIR, f"{tag}_sums.pt"))
        del cache, layers
    print(f"[dump] saved to {OUT_DIR}", flush=True)
    return _release(self)


scene_test.ChildMemoryTaskArgs.release = _dump_then_release
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
