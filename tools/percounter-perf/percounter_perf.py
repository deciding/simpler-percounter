#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reproduce docs/zh-cn/percounter/perf-results.md: a5 host_build_graph scheduler comparison.

Three subcommands, run in order:

    percounter_perf.py setup  --name g --source <clean checkout>
    percounter_perf.py run    --name g --out <dir> [--cases all] [--golden compare]
    percounter_perf.py report --out <dir>

`setup` rsyncs a clean checkout into ~/pcexp_<name>, applies the experiment
patches, and installs it editable. The patches are never applied to the source
tree: everything they add exists only in that copy.

`run` submits the waves through `task-submit`, one device, holding the lock for
the whole plan. Every run names its scheduler through SIMPLER_EXP_FORCE_SCHED,
so a graph the named scheduler cannot run fails the bind instead of being
quietly measured as another scheduler.

`report` prints the markdown tables.

Two case groups, selected with --cases:

    synthetic  15 scene tests. Small graphs, kernels that barely compute.
    graph      qwen3-14B decode: 40 real layers recorded as one Graph
               Definition. This is the whole-graph example.
    all        both (default).

Two correctness modes, selected with --golden:

    skip       timing only.
    compare    adds a correctness wave (default). Every run in it fills its
               OUT tensors with NaN first, because an OUT tensor's device bytes
               are otherwise whatever that memory last held and an unwritten
               output would agree with an earlier run's leftovers. The graph
               wave compares device outputs across schedulers and carries a
               negative control that must fail.

The timing wave never enables golden: the reference computation would land
inside the measured region.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Applied in order to the experiment copy. counter/window carry the lane
# instrumentation the kernel-span column reads; force_sched makes the scheduler
# selectable; poison defines an OUT tensor's bytes before a run.
PATCHES = ("counter_patch.py", "window_patch.py", "force_sched_patch.py", "poison_patch.py",
           "launchspec_core_num_patch.py")

# Experiment-only scene tests, copied into the experiment copy. TestPaExp
# supplies the two large paged-attention shapes (B16, B64) that no committed
# case covers.
EXTRA_CASES = (
    ("test_pa_exp.py", "tests/st/a5/host_build_graph/paged_attention"),
    ("test_dag_exp.py", "tests/st/a5/host_build_graph/multi_core_dag"),
)

SCHED_NAME = {"lg": "legacy", "res": "resident", "pc": "percounter"}

SYNTHETIC = [
    ("pa_case1", "tests/st/a5/host_build_graph/paged_attention", "TestPagedAttentionHostBuildGraphA5::Case1"),
    ("pa_b64", "tests/st/a5/host_build_graph/paged_attention", "TestPaExp::B64"),
    ("pa_b16", "tests/st/a5/host_build_graph/paged_attention", "TestPaExp::B16"),
    ("batch_case1", "tests/st/a5/host_build_graph/batch_paged_attention", "TestBatchPagedAttentionHostBuildGraph::Case1"),
    ("batch_case2", "tests/st/a5/host_build_graph/batch_paged_attention", "TestBatchPagedAttentionHostBuildGraph::Case2"),
    ("unroll_case1", "tests/st/a5/host_build_graph/paged_attention_unroll", "TestPagedAttentionUnrollHostBuildGraph::Case1"),
    ("unroll_case2", "tests/st/a5/host_build_graph/paged_attention_unroll", "TestPagedAttentionUnrollHostBuildGraph::Case2"),
    ("mscope_case1", "examples/a5/host_build_graph/paged_attention_unroll_manual_scope", "TestPagedAttentionUnrollManualScopeHostBuildGraph::Case1"),
    ("mscope_case2", "examples/a5/host_build_graph/paged_attention_unroll_manual_scope", "TestPagedAttentionUnrollManualScopeHostBuildGraph::Case2"),
    ("bgemm_case0", "examples/a5/host_build_graph/benchmark_bgemm", "TestBenchmarkBgemmHostBuildGraph::Case0"),
    ("spmd_case1", "tests/st/a5/host_build_graph/spmd_paged_attention", "TestSpmdPagedAttentionHbgA5::Case1"),
    ("pa_var4", "tests/st/a5/host_build_graph/paged_attention", "TestPagedAttentionHostBuildGraphA5::SmallCaseVarSeq4"),
    ("pa_small2", "tests/st/a5/host_build_graph/paged_attention", "TestPagedAttentionHostBuildGraphA5::SmallCase2"),
    ("pa_var2", "tests/st/a5/host_build_graph/paged_attention", "TestPagedAttentionHostBuildGraphA5::SmallCaseVarSeq2"),
    ("pa_small1", "tests/st/a5/host_build_graph/paged_attention", "TestPagedAttentionHostBuildGraphA5::SmallCase1"),
]

GRAPH_CASE = ("qwen3", "examples/a5/host_build_graph/qwen3_14b_decode/main.py")

# pypto-lib DeepSeek-V4.1-Flash modules, run one at a time against a5 HBG.
# Each carries its own golden, so correctness comes with the run rather than
# from a cross-scheduler comparison. Every module listed here was checked for
# `predicate=` and compiles under both runtimes.
#
# The second element is the module's own extra flags. config.py parses --tp at
# import, from a default of 4, so a single-card run has to say --tp 1 or the
# parser rejects the device count.
#
# The third is what a run of it is known to produce:
#
#   "run"     -- runs on a5 under both legacy and percounter, carries a golden
#   "comm64"  -- rejected in `orch.allocate_domain` before any task is
#                submitted. simpler's comm domain requires every named buffer
#                and the window itself to be a multiple of 64 bytes so named
#                buffers never share a cache-maintenance block; pypto rounds
#                to 32 (`kCommBufferAlignmentBytes` in its
#                codegen/distributed/comm_layout.h, a figure taken from the
#                A2/A3 FP16 peer MTE granularity), so a 4-byte signal buffer
#                arrives 32 bytes wide. Scheduler-independent: all three fail
#                identically, and `--comm-pad64` re-rounds the request to
#                measure what lies behind it.
#   "crash"    -- reaches the device and fails there for its own reason, the
#                same way under every scheduler.
#
# `--deepseek all` attempts every module regardless; the default takes the
# "run" group, which is what perf-results.md reports.
#
# Two members of the model are absent entirely. `decode_fwd` is the whole
# 40-layer backbone and `moe` reaches the same MoE transport: both predicate
# multi-block SPMD tasks, which `create_scheduler_state` refuses at bind.
DEEPSEEK_MODULES = [
    ("rmsnorm.py", [], "run"),
    ("hc_pre.py", [], "run"),
    ("hc_post.py", [], "run"),
    ("hc_head.py", [], "run"),
    ("input_pack.py", [], "run"),
    ("engram.py", [], "run"),
    ("boundary_fwd.py", [], "run"),
    # Its boundary case validates and benchmarks; its lm_head case then takes
    # the chip worker down with SIGSEGV during init, identically under both
    # schedulers. The reported figures are the boundary case's.
    ("hc_mixes.py", [], "crash"),
    ("lm_head.py", ["--tp", "1"], "comm64"),
    ("decode_swa.py", ["--tp", "1"], "comm64"),
    ("decode_attn_swa.py", ["--tp", "1"], "comm64"),
    ("decode_attn_c1a_full.py", ["--tp", "1"], "comm64"),
    ("decode_attn_c1a_reindex.py", ["--tp", "1"], "comm64"),
    ("decode_attn_c1a_reuse.py", ["--tp", "1"], "comm64"),
    ("decode_attn_c2a_full.py", ["--tp", "1"], "comm64"),
    ("decode_attn_c2a_reuse.py", ["--tp", "1"], "comm64"),
    ("decode_c1a_full.py", ["--tp", "1"], "comm64"),
    ("decode_c1a_reindex.py", ["--tp", "1"], "comm64"),
    ("decode_c1a_reuse.py", ["--tp", "1"], "comm64"),
    ("decode_c2a_full.py", ["--tp", "1"], "comm64"),
    ("decode_c2a_reuse.py", ["--tp", "1"], "comm64"),
    ("prefill_swa.py", ["--tp", "1"], "comm64"),
    ("prefill_attn_swa.py", ["--tp", "1"], "comm64"),
    ("prefill_attn_c1a_full.py", ["--tp", "1"], "comm64"),
    ("prefill_attn_c1a_reindex.py", ["--tp", "1"], "comm64"),
    ("prefill_attn_c1a_reuse.py", ["--tp", "1"], "comm64"),
    ("prefill_attn_c2a_full.py", ["--tp", "1"], "comm64"),
    ("prefill_attn_c2a_reuse.py", ["--tp", "1"], "comm64"),
    ("prefill_c1a_full.py", ["--tp", "1"], "comm64"),
    ("prefill_c1a_reindex.py", ["--tp", "1"], "comm64"),
    ("prefill_c1a_reuse.py", ["--tp", "1"], "comm64"),
    ("prefill_c2a_full.py", ["--tp", "1"], "comm64"),
    ("prefill_c2a_reuse.py", ["--tp", "1"], "comm64"),
    ("prefill_layer.py", ["--tp", "1"], "comm64"),
]

DEEPSEEK_DIR = "models/deepseek_v4_1_flash"

SYNTHETIC_ROUNDS = 4
GRAPH_ROUNDS = 30
# pypto's own benchmark loop, read from the environment per run. The runner
# docstring is explicit that a comparison must pin both rather than take the
# 100-round default, which is minutes for the larger modules.
DEEPSEEK_ROUNDS = 20
DEEPSEEK_WARMUP = 2

# Schedulers per group. resident refuses a graph-execution bind, so the graph
# timing wave does not spend runs on it; the correctness wave still names it
# once, which is what turns "resident does not run this" from an assumption
# into a recorded refusal.
SYNTHETIC_SCHEDS = ("lg", "res", "pc")
GRAPH_TIMING_SCHEDS = ("lg", "pc")
# Every pypto-generated a5 module carries SPMD shapes, which resident refuses at
# bind, so a deepseek wave that named it would spend its runs on recording the
# same refusal 34 times. `--deepseek all` with a single rep is how that refusal
# gets re-established rather than assumed.
DEEPSEEK_SCHEDS = ("lg", "pc")


# --------------------------------------------------------------------------
# plan construction


def rotate(seq, n):
    seq = list(seq)
    n %= len(seq)
    return seq[n:] + seq[:n]


def build_plan(cases, reps):
    """Interleave the schedulers and rotate their order once per repetition.

    Interleaving puts every scheduler's runs in the same stretch of wall clock,
    and rotating stops any one of them from always following the same
    predecessor, so neither a drifting machine load nor a warm-up effect can
    land on one scheduler.
    """
    lines = []
    for rep in range(reps):
        for name, kind, path, case, rounds, scheds in cases:
            for sched in rotate(scheds, rep):
                tag = f"s{rep + 1}_{name}_{sched}"
                lines.append(f"{tag} {sched} {kind} {path} {case} {rounds}")
    return lines


def selected_cases(group, deepseek="runnable"):
    cases = []
    if group in ("synthetic", "all"):
        for name, path, case in SYNTHETIC:
            cases.append((name, "st", path, case, SYNTHETIC_ROUNDS, SYNTHETIC_SCHEDS))
    if group in ("graph", "all"):
        name, path = GRAPH_CASE
        cases.append((name, "graph", path, "-", GRAPH_ROUNDS, GRAPH_TIMING_SCHEDS))
    if group in ("deepseek", "all"):
        for module, extra, status in DEEPSEEK_MODULES:
            if status != "run" and deepseek != "all":
                continue
            name = "ds_" + Path(module).stem
            # The plan line is whitespace-separated, so a module's own flags
            # ride in the otherwise unused case column.
            cases.append((name, "pypto", f"{DEEPSEEK_DIR}/{module}",
                          ",".join(extra) or "-", DEEPSEEK_ROUNDS, DEEPSEEK_SCHEDS))
    return cases


# --------------------------------------------------------------------------
# the worker: runs on the box, inside the device lock


SPAN_DEV = r"name=chip\.run\.runner_run\.device_wall ts=0 dur=([0-9]+)"
SPAN_SYS = r"name=chip\.run\.runner_run ts=[0-9]+ dur=([0-9]+)"


def read_spans(host_dir):
    dev, sysw = [], []
    for path in glob.glob(f"{host_dir}/**/*", recursive=True):
        if not os.path.isfile(path):
            continue
        text = open(path, errors="replace").read()
        dev += [int(v) for v in re.findall(SPAN_DEV, text)]
        sysw += [int(v) for v in re.findall(SPAN_SYS, text)]
    return dev, sysw


def run_one(checkout, out, tag, sched, kind, path, case, rounds, device, poison, golden,
            pypto_lib=None, pypto_venv=None, comm_pad64=False):
    """One measured run. Returns the shell exit status."""
    run_dir = Path(out) / "runs" / tag
    (run_dir / "ascend").mkdir(parents=True, exist_ok=True)
    (run_dir / "host").mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    for stale in ("SIMPLER_HBG_SCHEDULER_TYPE", "SIMPLER_EXP_MAX_BLOCK_DIM",
                  "SIMPLER_EXP_FORCE_SCHED", "SIMPLER_EXP_POISON_OUT", "SIMPLER_EXP_POISON_DUMP",
                  "SIMPLER_EXP_COMM_PAD64"):
        env.pop(stale, None)
    env["SIMPLER_EXP_FORCE_SCHED"] = SCHED_NAME[sched]
    env["ASCEND_PROCESS_LOG_PATH"] = str(run_dir / "ascend")
    env["KEEP_HOST_LOG_DIR"] = str(run_dir / "host")
    if poison:
        env["SIMPLER_EXP_POISON_OUT"] = "1"

    keeplog = str(HERE / "helpers" / "keeplog.py")
    python, cwd = sys.executable, checkout
    if kind == "st":
        cmd = [python, keeplog, "-m", "pytest", path, "--platform", "a5",
               "--device", str(device), "--case", case, "--manual", "include",
               "--rounds", str(rounds), "-s", "-q", "--log-level", "info"]
        if not golden:
            cmd.append("--skip-golden")
    elif kind == "graph":
        cmd = [python, keeplog, path, "-p", "a5", "-d", str(device),
               "--rounds", str(rounds), "--skip-golden", "--log-level", "timing"]
    elif kind == "pypto":
        # A pypto-lib module needs the 3.10 interpreter that owns pypto's
        # compiled core, and it carries its own golden, so `golden` does not
        # select anything here -- the module always checks itself.
        python, cwd = str(Path(pypto_venv) / "bin" / "python"), pypto_lib
        purelib = subprocess.run([python, "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        env["PYTHONPATH"] = os.pathsep.join([purelib, str(pypto_lib)])
        # The rounds/warmup pair only sizes the loop; PYPTO_BENCH is what makes
        # `golden/runner.py::_bench_enabled` run one at all. Setting the sizes
        # without it measures a single launch and says nothing.
        env["PYPTO_BENCH"] = "1"
        env["PYPTO_BENCH_ROUNDS"] = str(rounds)
        env["PYPTO_BENCH_WARMUP"] = str(DEEPSEEK_WARMUP)
        env.setdefault("PYPTO_CODEGEN_MAX_WORKERS", "16")
        if comm_pad64:
            env["SIMPLER_EXP_COMM_PAD64"] = "1"
        cmd = [python, keeplog, str(HERE / "helpers" / "pypto_module.py"), "hbg", path,
               "-p", "a5", "-d", str(device)]
        if case != "-":
            cmd += case.split(",")
    else:
        raise ValueError(f"unknown case kind {kind!r}")

    with open(run_dir / "run.log", "w") as log:
        rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log,
                            stderr=subprocess.STDOUT, check=False).returncode

    dev, sysw = read_spans(run_dir / "host")
    with open(Path(out) / "times", "a") as fh:
        fh.write(f"{tag} {sched} {kind} {case} rc={rc} "
                 f"wall={','.join(map(str, dev))} sys={','.join(map(str, sysw))}\n")
    return rc


def correctness_graph(checkout, out, device):
    """Cross-scheduler output comparison for the whole-graph case.

    One legacy run is the reference and a second legacy run bounds what
    run-to-run noise looks like, so percounter's difference from the reference
    is read against legacy's difference from itself rather than against zero.
    `neg` clobbers `out` after the run: the compare must report it, and a wave
    where it does not is not evidence of anything.
    """
    path = GRAPH_CASE[1]
    dump_root = Path(out) / "dump"
    runs = [("lg1", "lg", False), ("res", "res", False), ("pc", "pc", False),
            ("lg2", "lg", False), ("neg", "pc", True)]

    for tag, sched, negctl in runs:
        run_dir = Path(out) / "correctness" / tag
        (run_dir / "ascend").mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        for stale in ("SIMPLER_HBG_SCHEDULER_TYPE", "SIMPLER_EXP_FORCE_SCHED",
                      "SIMPLER_EXP_POISON_OUT", "SIMPLER_EXP_POISON_DUMP"):
            env.pop(stale, None)
        env["SIMPLER_EXP_FORCE_SCHED"] = SCHED_NAME[sched]
        env["SIMPLER_EXP_POISON_OUT"] = "1"
        env["ASCEND_PROCESS_LOG_PATH"] = str(run_dir / "ascend")
        if negctl:
            env["SIMPLER_EXP_POISON_DUMP"] = "1"
        cmd = [sys.executable, str(HERE / "helpers" / "qwen_dump.py"), str(dump_root / tag), path,
               "-p", "a5", "-d", str(device), "--rounds", "1", "--skip-golden", "--log-level", "info"]
        with open(run_dir / "run.log", "w") as log:
            rc = subprocess.run(cmd, cwd=checkout, env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False).returncode
        text = open(run_dir / "run.log", errors="replace").read()
        poisoned = text.count("poisoned OUT tensor")
        sel = re.search(r"SIMPLER_EXP_FORCE_SCHED[^\"\n]{0,60}", text)
        with open(Path(out) / "correctness.txt", "a") as fh:
            fh.write(f"{tag} {SCHED_NAME[sched]} negctl={negctl} rc={rc} "
                     f"poisoned={poisoned} sel={sel.group(0) if sel else ''}\n")

    for tag in ("lg2", "res", "pc", "neg"):
        with open(Path(out) / "correctness.txt", "a") as fh:
            fh.write(f"== lg1 vs {tag}\n")
            fh.flush()
            subprocess.run([sys.executable, str(HERE / "helpers" / "qwen_compare.py"),
                            str(dump_root / "lg1"), str(dump_root / tag)],
                           stdout=fh, stderr=subprocess.STDOUT, check=False)


def correctness_synthetic(checkout, out, device):
    """Run every scene test once per scheduler with its own golden enabled.

    Poisoned, so a case that leaves part of its output unwritten fails here
    instead of passing on an earlier run's bytes.
    """
    for name, path, case in SYNTHETIC:
        for sched in SYNTHETIC_SCHEDS:
            tag = f"gold_{name}_{sched}"
            rc = run_one(checkout, out, tag, sched, "st", path, case, 1, device,
                         poison=True, golden=True)
            with open(Path(out) / "correctness.txt", "a") as fh:
                fh.write(f"{tag} rc={rc}\n")


def cmd_worker(args):
    device = os.environ["TASK_DEVICE"]
    checkout = args.checkout
    with open(Path(args.out) / "times", "a") as fh:
        fh.write(f"start device={device}\n")
    for line in open(Path(args.out) / "plan.txt"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tag, sched, kind, path, case, rounds = line.split()
        run_one(checkout, args.out, tag, sched, kind, path, case, int(rounds), device,
                poison=False, golden=False,
                pypto_lib=args.pypto_lib, pypto_venv=args.pypto_venv,
                comm_pad64=args.comm_pad64)
    if args.golden == "compare":
        if args.cases in ("synthetic", "all"):
            correctness_synthetic(checkout, args.out, device)
        if args.cases in ("graph", "all"):
            correctness_graph(checkout, args.out, device)
    with open(Path(args.out) / "times", "a") as fh:
        fh.write("done\n")
    return 0


# --------------------------------------------------------------------------
# setup and submit


def cmd_setup(args):
    dest = Path(os.path.expanduser(f"~/pcexp_{args.name}"))
    subprocess.run(["rsync", "-a", "--delete", "--exclude", "build", "--exclude", ".venv",
                    "--exclude", "outputs", "--exclude", "*.egg-info", "--exclude", "_assets",
                    f"{args.source.rstrip('/')}/", f"{dest}/"], check=True)
    for patch in PATCHES:
        subprocess.run([sys.executable, str(HERE / "patches" / patch), str(dest)], check=True)
    for fname, target in EXTRA_CASES:
        subprocess.run(["cp", str(HERE / "cases" / fname), str(dest / target)], check=True)
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(dest / ".venv")], check=True)
    pip = str(dest / ".venv" / "bin" / "pip")
    subprocess.run([pip, "install", "--no-build-isolation", "-e", "."], cwd=dest, check=True)
    print(f"experiment copy ready at {dest}")
    return 0


def cmd_run(args):
    checkout = Path(os.path.expanduser(f"~/pcexp_{args.name}"))
    if not (checkout / ".venv").exists():
        sys.exit(f"{checkout} has no .venv; run `setup` first")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    plan = build_plan(selected_cases(args.cases, args.deepseek), args.reps)
    (out / "plan.txt").write_text("\n".join(plan) + "\n")

    precheck = checkout / ".claude/skills/onboard-arch-precheck/check.sh"
    subprocess.run([str(precheck), "a5"], check=True)

    python = str(checkout / ".venv" / "bin" / "python")
    inner = (f"{python} {__file__} _worker --out {out} --checkout {checkout} "
             f"--cases {args.cases} --golden {args.golden}")
    if args.pypto_lib:
        inner += f" --pypto-lib {args.pypto_lib} --pypto-venv {args.pypto_venv}"
    if args.comm_pad64:
        inner += " --comm-pad64"
    cmd = ["task-submit", "--timeout", str(args.timeout), "--max-time", str(args.timeout),
           "--device", "auto", "--device-num", "1", "--run", inner]
    print(" ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


# --------------------------------------------------------------------------
# report


def median_of(text):
    vals = [int(x) / 1000 for x in text.split(",") if x]
    if not vals:
        return None
    return statistics.median(vals[1:]) if len(vals) > 1 else vals[0]


def window_us(out, tag):
    """Kernel span from the EXPW lane counters, in microseconds.

    Only the resident and percounter AICore schedulers run the loop that
    publishes these, so a legacy run has no span by construction.
    """
    recs = []
    for f in sorted(glob.glob(f"{out}/runs/{tag}/ascend/debug/device-*/*.log")):
        text = open(f, errors="replace").read()
        for m in re.finditer(r"EXPW (percounter|resident) type=(\d) ([^\"\n]*)", text):
            recs.append((int(m.group(2)), {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", m.group(3))}))
    rounds = [recs[i:i + 2] for i in range(0, len(recs) - 1, 2) if recs[i][0] == 0 and recs[i + 1][0] == 1]
    spans = []
    for rd in (rounds[1:] or rounds):
        ts = [d for _, d in rd if d["lanes"]]
        if ts:
            spans.append((max(d["max_last"] for d in ts) - min(d["min_first"] for d in ts)) / 1000)
    return statistics.median(spans) if spans else None


def collect(out):
    cells, failed, order = {}, [], []
    for line in open(Path(out) / "times"):
        parts = line.split()
        if len(parts) < 7 or not parts[4].startswith("rc="):
            continue
        tag, sched, kind, rc = parts[0], parts[1], parts[2], parts[4][3:]
        name = tag.split("_", 1)[1].rsplit("_", 1)[0]
        if name not in order:
            order.append(name)
        if rc != "0":
            failed.append(f"{tag} rc={rc}")
            continue
        cell = cells.setdefault((name, sched), {"sys": [], "dev": [], "span": [], "kind": kind})
        for key, text in (("dev", parts[5][5:]), ("sys", parts[6][4:])):
            value = median_of(text)
            if value is not None:
                cell[key].append(value)
        span = window_us(out, tag)
        if span is not None:
            cell["span"].append(span)
    return cells, failed, order


def agg(cells, name, sched, key):
    cell = cells.get((name, sched))
    if not cell or not cell[key]:
        return None
    return statistics.median(cell[key])


def fmt(v):
    return "" if v is None else f"{v:.1f}"


def drop(base, new):
    if base is None or new is None or base == 0:
        return ""
    return f"{(base - new) / base * 100:.1f}%"


def markdown_table(header, rows, aligns):
    """Emit a table whose delimiter row matches the header's display width.

    markdownlint MD060 measures that width with East Asian characters counted
    as two columns, so a CJK header needs a delimiter sized in display columns
    rather than in characters.
    """
    def width(text):
        return sum(2 if ord(c) > 0x2E80 else 1 for c in text)

    bar = []
    for text, align in zip(header, aligns):
        n = width(text)
        bar.append("-" * (n - 1) + ":" if align == "r" else "-" * n)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(bar) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def cmd_report(args):
    cells, failed, order = collect(args.out)
    synth = [n for n in order if n != GRAPH_CASE[0]]
    graph = [n for n in order if n == GRAPH_CASE[0]]

    if synth:
        print("### device wall\n")
        rows = []
        for name in synth:
            d = {s: agg(cells, name, s, "dev") for s in SYNTHETIC_SCHEDS}
            rows.append([name, fmt(d["lg"]), fmt(d["res"]), fmt(d["pc"]),
                         drop(d["lg"], d["pc"]), drop(d["res"], d["pc"])])
        print(markdown_table(["case", "legacy", "resident", "percounter", "比 legacy", "比 resident"],
                             rows, ["l", "r", "r", "r", "r", "r"]))
        print("\n### sys wall\n")
        rows = [[name] + [fmt(agg(cells, name, s, "sys")) for s in SYNTHETIC_SCHEDS] for name in synth]
        print(markdown_table(["case", "legacy", "resident", "percounter"], rows, ["l", "r", "r", "r"]))
        print("\n### kernel span\n")
        rows = [[name] + [fmt(agg(cells, name, s, "span")) for s in ("res", "pc")] for name in synth]
        print(markdown_table(["case", "resident", "percounter"], rows, ["l", "r", "r"]))

    for name in graph:
        print(f"\n### {name}\n")
        rows = []
        for sched in SYNTHETIC_SCHEDS:
            cell = cells.get((name, sched))
            reps = " / ".join(fmt(v) for v in cell["dev"]) if cell else "不适用"
            rows.append([SCHED_NAME[sched], fmt(agg(cells, name, sched, "dev")),
                         fmt(agg(cells, name, sched, "sys")), reps])
        print(markdown_table(["调度器", "device wall", "sys wall", "三遍分别"],
                             rows, ["l", "r", "r", "l"]))
        lg = agg(cells, name, "lg", "dev")
        pc = agg(cells, name, "pc", "dev")
        if lg and pc:
            print(f"\npercounter 比 legacy: device wall -{drop(lg, pc)}, "
                  f"sys wall -{drop(agg(cells, name, 'lg', 'sys'), agg(cells, name, 'pc', 'sys'))}")

    if failed:
        print("\nno number (the named scheduler refused the graph, or the run failed):")
        for item in failed:
            print(" ", item)

    correctness = Path(args.out) / "correctness.txt"
    if correctness.exists():
        print("\n### correctness\n```")
        print(correctness.read_text().rstrip())
        print("```")
    return 0


# --------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="build the patched experiment copy")
    p.add_argument("--name", required=True, help="experiment copy suffix: ~/pcexp_<name>")
    p.add_argument("--source", required=True, help="clean checkout to copy from")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("run", help="submit the waves under the device lock")
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cases", choices=("synthetic", "graph", "deepseek", "all"), default="all")
    p.add_argument("--golden", choices=("skip", "compare"), default="compare")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--timeout", type=int, default=10800)
    p.add_argument("--pypto-lib", help="pypto-lib checkout; required for --cases deepseek")
    p.add_argument("--pypto-venv", help="venv holding pypto and the patched simpler")
    p.add_argument("--deepseek", choices=("runnable", "all"), default="runnable",
                   help="'all' also attempts the modules recorded as blocked")
    p.add_argument("--comm-pad64", action="store_true",
                   help="re-round pypto's comm buffers to simpler's 64-byte contract")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="print the markdown tables")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("_worker", help=argparse.SUPPRESS)
    p.add_argument("--out", required=True)
    p.add_argument("--checkout", required=True)
    p.add_argument("--cases", default="all")
    p.add_argument("--golden", default="compare")
    p.add_argument("--pypto-lib")
    p.add_argument("--pypto-venv")
    p.add_argument("--comm-pad64", action="store_true")
    # task-submit appends the device it allocated to the --run command. The
    # worker takes the device from TASK_DEVICE, so this only keeps argparse from
    # refusing an argument a shell script would have ignored.
    p.add_argument("--device", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_worker)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
