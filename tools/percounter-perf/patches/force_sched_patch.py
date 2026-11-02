"""Experiment-only patch: pick the AICore scheduler by hand. Never committed.

usage: exp_force_sched_patch.py <checkout>

Adds SIMPLER_EXP_FORCE_SCHED=legacy|resident|percounter to a5 host_build_graph.
The named scheduler runs or the bind fails. A graph the named scheduler cannot
run is reported and refused, never quietly handed to another one, so a run
either measures the scheduler it names or does not produce a number at all.
An unrecognised value also fails the bind. Unset leaves the stock selection
untouched.

Stock selection, for reference: percounter when SIMPLER_HBG_SCHEDULER_TYPE asks
for it and every task has a shape it runs; otherwise legacy for a
graph-execution run or a SPMD / sync-start shape; otherwise resident.
"""

import sys
from pathlib import Path

root = Path(sys.argv[1])


def sub(rel, old, new, count=1):
    p = root / rel
    s = p.read_text()
    n = s.count(old)
    if n != count:
        sys.exit(f"{rel}: expected {count} of {old[:70]!r}, found {n}")
    p.write_text(s.replace(old, new))


HBG = "src/a5/runtime/host_build_graph/host"

# --- the knob also carries the percounter request ---------------------------
sub(
    f"{HBG}/percounter/percounter_state.h",
    "inline bool percounter_requested() {\n"
    "    SchedulerRequest request = SchedulerRequest::DEFAULT;\n",
    "inline bool percounter_requested() {\n"
    '    if (const char *forced = std::getenv("SIMPLER_EXP_FORCE_SCHED"))\n'
    "        if (forced[0] != '\\0') return std::strcmp(forced, \"percounter\") == 0;\n"
    "    SchedulerRequest request = SchedulerRequest::DEFAULT;\n",
)

# --- the knob itself --------------------------------------------------------
sub(
    f"{HBG}/runtime_maker.cpp",
    "bool create_scheduler_state(\n",
    "enum class ExpForceSched : uint8_t { NONE, LEGACY, RESIDENT, PERCOUNTER };\n"
    "\n"
    "// Reads SIMPLER_EXP_FORCE_SCHED. False, failing the bind, for an unrecognised\n"
    "// value: a misspelled knob that ran the stock selection instead would be read\n"
    "// as a measurement of the scheduler it names.\n"
    "static bool exp_force_sched(ExpForceSched *out) {\n"
    '    const char *text = std::getenv("SIMPLER_EXP_FORCE_SCHED");\n'
    "    if (text == nullptr || text[0] == '\\0') {\n"
    "        *out = ExpForceSched::NONE;\n"
    "        return true;\n"
    "    }\n"
    '    if (std::strcmp(text, "legacy") == 0) {\n'
    "        *out = ExpForceSched::LEGACY;\n"
    "        return true;\n"
    "    }\n"
    '    if (std::strcmp(text, "resident") == 0) {\n'
    "        *out = ExpForceSched::RESIDENT;\n"
    "        return true;\n"
    "    }\n"
    '    if (std::strcmp(text, "percounter") == 0) {\n'
    "        *out = ExpForceSched::PERCOUNTER;\n"
    "        return true;\n"
    "    }\n"
    '    LOG_ERROR("EXP SIMPLER_EXP_FORCE_SCHED=%s is not legacy, resident or percounter", text);\n'
    "    return false;\n"
    "}\n"
    "\n"
    "bool create_scheduler_state(\n",
)

# --- forced legacy settles the bind before any shape walk -------------------
sub(
    f"{HBG}/runtime_maker.cpp",
    "    if (!simpler::hbg::percounter::check_scheduler_request()) return false;\n",
    "    if (!simpler::hbg::percounter::check_scheduler_request()) return false;\n"
    "    ExpForceSched forced = ExpForceSched::NONE;\n"
    "    if (!exp_force_sched(&forced)) return false;\n"
    "    if (forced == ExpForceSched::LEGACY) {\n"
    "        select_legacy_scheduler(runtime, SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH);\n"
    '        LOG_WARN("EXP SIMPLER_EXP_FORCE_SCHED=legacy: AICPU scheduling for %d tasks", total_tasks);\n'
    "        return true;\n"
    "    }\n",
)

# --- forced resident refuses the graph-execution fallback -------------------
sub(
    f"{HBG}/runtime_maker.cpp",
    "        if (slot.task_kind == TaskKind::GRAPH && !percounter_graph_tasks) {\n"
    "            select_legacy_scheduler(runtime, SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH);\n",
    "        if (slot.task_kind == TaskKind::GRAPH && !percounter_graph_tasks) {\n"
    "            if (forced == ExpForceSched::RESIDENT) {\n"
    '                LOG_ERROR("EXP SIMPLER_EXP_FORCE_SCHED=resident: a graph-execution run needs AICPU scheduling");\n'
    "                return false;\n"
    "            }\n"
    "            select_legacy_scheduler(runtime, SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH);\n",
)

# --- forced percounter and forced resident refuse the remaining fallbacks ---
sub(
    f"{HBG}/runtime_maker.cpp",
    "    if (graph_task_count != 0) {\n"
    "        select_legacy_scheduler(runtime, SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH);\n",
    "    if (forced == ExpForceSched::PERCOUNTER) {\n"
    '        LOG_ERROR("EXP SIMPLER_EXP_FORCE_SCHED=percounter: this graph is not one percounter runs");\n'
    "        return false;\n"
    "    }\n"
    "    if (graph_task_count != 0) {\n"
    "        if (forced == ExpForceSched::RESIDENT) {\n"
    '            LOG_ERROR("EXP SIMPLER_EXP_FORCE_SCHED=resident: a graph-execution run needs AICPU scheduling");\n'
    "            return false;\n"
    "        }\n"
    "        select_legacy_scheduler(runtime, SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH);\n",
)

sub(
    f"{HBG}/runtime_maker.cpp",
    "    if (legacy_shape_task_id >= 0) {\n"
    "        select_legacy_scheduler(runtime, SCHEDULER_RUNTIME_MODE_LEGACY_UNSUPPORTED_SHAPE);\n",
    "    if (legacy_shape_task_id >= 0) {\n"
    "        if (forced == ExpForceSched::RESIDENT) {\n"
    "            LOG_ERROR(\n"
    '                "EXP SIMPLER_EXP_FORCE_SCHED=resident: task id=%" PRId64 " has a SPMD or sync-start shape "\n'
    '                "resident does not run",\n'
    "                legacy_shape_task_id\n"
    "            );\n"
    "            return false;\n"
    "        }\n"
    "        select_legacy_scheduler(runtime, SCHEDULER_RUNTIME_MODE_LEGACY_UNSUPPORTED_SHAPE);\n",
)

print("patched force-sched")
