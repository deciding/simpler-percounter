"""Experiment-only patches for the scalability study. Never committed.

usage: exp_patch.py <checkout> cap|count
  cap   : SIMPLER_EXP_MAX_BLOCK_DIM caps the launch width
  count : cap + percounter lane counters + resident stats always on + AICPU readout
"""

import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]


def sub(rel, old, new, count=1):
    p = root / rel
    s = p.read_text()
    n = s.count(old)
    if n != count:
        sys.exit(f"{rel}: expected {count} of {old[:70]!r}, found {n}")
    p.write_text(s.replace(old, new))


# --- cap -------------------------------------------------------------------
sub(
    "src/common/platform/onboard/host/device_runner_base.cpp",
    '    LOG_INFO("block_dim resolved to %d (cube=%u, vector=%u)", max_block_dim_, max_cube_cores_, max_vector_cores_);\n'
    "    return max_block_dim_;\n",
    '    LOG_INFO("block_dim resolved to %d (cube=%u, vector=%u)", max_block_dim_, max_cube_cores_, max_vector_cores_);\n'
    '    if (const char *cap = std::getenv("SIMPLER_EXP_MAX_BLOCK_DIM")) {\n'
    "        const int v = std::atoi(cap);\n"
    "        if (v > 0 && v < max_block_dim_) {\n"
    '            LOG_WARN("EXP block_dim capped to %d of %d", v, max_block_dim_);\n'
    "            return v;\n"
    "        }\n"
    "    }\n"
    "    return max_block_dim_;\n",
)
if mode == "cap":
    sys.exit(0)

# --- percounter lane counters ----------------------------------------------
# exp[0] ticket fetch_add, [1] completion fetch_add, [2] progress fetch_add,
# [3] MIX/sync fetch_add, [4] CAS, [5] producer-counter loads, [6] wait iterations
E = "src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h"
_last = "    uint64_t aic_order_count;\n};"
if (root / E).read_text().count(_last) != 1:
    _last = "    __gm__ uint64_t *sync_pending;\n};"
sub(E, _last, _last[:-3] + "\n    uint64_t *exp;\n};")
sub(
    E,
    "        if (!percounter_counter_published(percounter_observe(percounter_producer(cfg, task, i)))) return false;",
    "        ++cfg.exp[5];\n"
    "        if (!percounter_counter_published(percounter_observe(percounter_producer(cfg, task, i)))) return false;",
)
sub(
    E,
    "    scheduler_gm_fetch_add(cfg.counters[task.id].value, count);\n",
    "    cfg.exp[1] += task.sink ? 2 : 1;\n    scheduler_gm_fetch_add(cfg.counters[task.id].value, count);\n",
)
sub(
    E,
    "        while (!percounter_counter_published(percounter_observe(address))) {\n",
    "        ++cfg.exp[5];\n"
    "        while (!percounter_counter_published(percounter_observe(address))) {\n"
    "            ++cfg.exp[5];\n            ++cfg.exp[6];\n",
)
sub(
    E,
    "        const uint64_t value = scheduler_gm_query(*slot);\n",
    "        ++cfg.exp[6];\n        const uint64_t value = scheduler_gm_query(*slot);\n",
)
_draw = "            const uint64_t base = scheduler_gm_fetch_add(cfg.ticket->next, batch);\n"
if (root / E).read_text().count(_draw) != 1:
    _draw = "                    const uint64_t base = scheduler_gm_fetch_add(cfg.tickets[range_group].next, batch);\n"
sub(E, _draw, _draw[: len(_draw) - len(_draw.lstrip())] + "++cfg.exp[0];\n" + _draw)
sub(
    E,
    "    scheduler_gm_fetch_add(cfg.control->resolved_task_count, state->local_done);\n",
    "    ++cfg.exp[2];\n    scheduler_gm_fetch_add(cfg.control->resolved_task_count, state->local_done);\n",
)
sub(
    E,
    "    if (scheduler_gm_compare_exchange(q->open_generation, generation, generation + 1) != generation) return;\n",
    "    ++cfg.exp[4];\n"
    "    if (scheduler_gm_compare_exchange(q->open_generation, generation, generation + 1) != generation) return;\n",
)
for line in [
    "    const uint64_t seq = scheduler_gm_fetch_add(cfg.sync_queue->tail, uint64_t{1});\n",
    "    const uint64_t prev = scheduler_gm_fetch_add(*claims, uint64_t{1});\n",
    "        scheduler_gm_fetch_add(cfg.sync_queue->completed, uint64_t{1});\n",
    "            const uint64_t draw = scheduler_gm_fetch_add(cfg.mix_tickets[PERCOUNTER_MIX_TICKET_DRAW].next, uint64_t{1});\n",
]:
    indent = line[: len(line) - len(line.lstrip())]
    sub(E, line, f"{indent}++cfg.exp[3];\n{line}")
sub(
    E,
    "        if (cohort_block) scheduler_gm_fetch_add(cfg.sync_queue->completed, uint64_t{1});\n",
    "        ++cfg.exp[3];\n        if (cohort_block) scheduler_gm_fetch_add(cfg.sync_queue->completed, uint64_t{1});\n",
)
sub(
    E,
    "    LaneConfig cfg{};\n    cfg.control = control;\n",
    "    LaneConfig cfg{};\n    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n    cfg.control = control;\n",
)
sub(
    E,
    "    percounter_lane_loop(cfg, timeout_cycles);\n}\n",
    "    percounter_lane_loop(cfg, timeout_cycles);\n"
    "    for (int i = 0; i < 7; ++i) {\n"
    "        if (exp_counts[i] != 0)\n"
    "            scheduler_gm_fetch_add(reinterpret_cast<__gm__ uint64_t *>(control->resolved_reserved)[i], exp_counts[i]);\n"
    "    }\n"
    "}\n",
)

# --- resident: stats always collected and published ------------------------
X = root / "src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp"
s = X.read_text()
s, n = re.subn(r"phase_timing_enabled \? (&stats->\w+) : nullptr", r"\1", s)
if n < 10:
    sys.exit(f"aicore_executor.cpp: only {n} stats pointers rewritten")
X.write_text(s)
sub(
    "src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp",
    "    wait_for_aicpu_exit_signal();\n    // Complete preceding GM writes before ACK.",
    "    publish_worker_stats(context, stats);\n    wait_for_aicpu_exit_signal();\n    // Complete preceding GM writes before ACK.",
)

# --- AICPU readout after every core proved retirement ------------------------
A = "src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp"
readout = r'''
static void exp_log_scheduler_counters(Runtime *runtime) {
    if (simpler::hbg::percounter::percounter_mode_enabled(runtime)) {
        auto *control = simpler::hbg::percounter::percounter_control(runtime);
        if (control == nullptr) return;
        cache_invalidate_range(&control->resolved_task_count, 64);
        uint64_t *v = reinterpret_cast<uint64_t *>(control->resolved_reserved);
        LOG_WARN(
            "EXP percounter tasks=%" PRIu64 " ticket_add=%" PRIu64 " completion_add=%" PRIu64 " progress_add=%" PRIu64
            " other_add=%" PRIu64 " cas=%" PRIu64 " fanin_load=%" PRIu64 " wait_iter=%" PRIu64,
            control->resolved_task_count, v[0], v[1], v[2], v[3], v[4], v[5], v[6]
        );
        for (int i = 0; i < 7; ++i)
            v[i] = 0;
        cache_flush_range(&control->resolved_task_count, 64);
        return;
    }
    SchedulerWorkerContext *base = aicore_scheduler_bootstrap_context(runtime);
    if (base == nullptr) return;
    const int32_t workers = runtime->get_worker_count();
    uint64_t t[17] = {};
    for (int32_t w = 0; w < workers; ++w) {
        SchedulerWorkerContext *c = base + w;
        cache_invalidate_range(c, sizeof(*c));
        const uint64_t f[17] = {
            c->executed_task_count, c->ready_enqueue_count, c->ready_batch_count, c->ready_pop_count,
            c->ready_steal_count, c->ready_cas_retry_count, c->ready_contention_giveup_count,
            c->fanin_state_load_count, c->wake_register_count, c->wake_cas_retry_count, c->wake_closed_retry_count,
            c->wake_migrate_count, c->wake_close_count, c->completion_enqueue_count, c->completion_resolve_count,
            c->idle_iteration_count, c->task_state_poll_count,
        };
        for (int i = 0; i < 17; ++i)
            t[i] += f[i];
    }
    LOG_WARN(
        "EXP resident workers=%d executed=%" PRIu64 " enqueue=%" PRIu64 " batch=%" PRIu64 " pop=%" PRIu64
        " steal=%" PRIu64 " ready_cas_retry=%" PRIu64 " giveup=%" PRIu64 " fanin_load=%" PRIu64
        " wake_register=%" PRIu64 " wake_cas_retry=%" PRIu64 " wake_closed_retry=%" PRIu64 " wake_migrate=%" PRIu64
        " wake_close=%" PRIu64 " completion_enqueue=%" PRIu64 " completion_resolve=%" PRIu64 " idle_iter=%" PRIu64
        " state_poll=%" PRIu64,
        workers, t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15],
        t[16]
    );
}
'''
sub(A, "static void record_resident_timeout(Runtime *runtime, SchedulerErrorSite error_site) {",
    readout.lstrip("\n").replace("static void exp_log", "static void exp_log", 1)
    + "\nstatic void record_resident_timeout(Runtime *runtime, SchedulerErrorSite error_site) {")
sub(
    A,
    "        const bool cores_retired = aicore_lifecycle_.cores_retired();\n",
    "        const bool cores_retired = aicore_lifecycle_.cores_retired();\n"
    "        if (cores_retired) exp_log_scheduler_counters(runtime);\n",
)
print("patched", mode)
