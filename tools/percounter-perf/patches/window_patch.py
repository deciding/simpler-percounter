"""Experiment-only AICore window breakdown, on top of `exp_patch.py count`. Never committed.

usage: exp_window_patch.py <checkout>
Per core type: active lanes, kernels, kernel cycles, dependency-wait cycles (percounter),
sum of per-lane first kernel start / last kernel end, global min start / max end.
"""

import sys
from pathlib import Path

root = Path(sys.argv[1])


def sub(rel, old, new, count=1):
    p = root / rel
    s = p.read_text()
    n = s.count(old)
    if n != count:
        sys.exit(f"{rel}: expected {count} of {old[:80]!r}, found {n}")
    p.write_text(s.replace(old, new))


E = "src/a5/runtime/host_build_graph/aicore/percounter/percounter_executor.h"
# kx[0] kernels, [1] kernel cycles, [2] dependency-wait cycles, [3] first kernel start, [4] last kernel end
sub(E, "    uint64_t *exp;\n};", "    uint64_t *exp;\n    uint64_t *kx;\n};")
KSTART = (
    "    *kernel_start = get_sys_cnt_aicore();\n"
    "    kernel(reinterpret_cast<__gm__ int64_t *>(cfg.payload->args));\n"
    "    OUT_OF_ORDER_STORE_BARRIER();\n"
)
KTIME = (
    "    {\n"
    "        const uint64_t e = get_sys_cnt_aicore();\n"
    "        cfg.kx[0] += 1;\n"
    "        cfg.kx[1] += e - *kernel_start;\n"
    "        if (cfg.kx[3] == 0) cfg.kx[3] = *kernel_start;\n"
    "        cfg.kx[4] = e;\n"
    "    }\n"
)
# The kernel call either ends launch_kernel or sits in percounter_execute_payload.
KCALL = "    kernel(reinterpret_cast<__gm__ int64_t *>(cfg.payload->args));\n    OUT_OF_ORDER_STORE_BARRIER();\n"
if (root / E).read_text().count(KSTART + "    return true;\n") == 1:
    sub(E, KSTART + "    return true;\n", KSTART + KTIME + "    return true;\n")
else:
    sub(E, KCALL, KCALL + KTIME)
sub(
    E,
    "        ++cfg.exp[5];\n"
    "        while (!percounter_counter_published(percounter_observe(address))) {\n",
    "        ++cfg.exp[5];\n"
    "        const uint64_t wait_t0 = get_sys_cnt_aicore();\n"
    "        bool waited = false;\n"
    "        while (!percounter_counter_published(percounter_observe(address))) {\n"
    "            waited = true;\n",
)
sub(
    E,
    "            SPIN_WAIT_HINT();\n        }\n    }\n    return true;\n}\n",
    "            SPIN_WAIT_HINT();\n        }\n"
    "        if (waited) cfg.kx[2] += get_sys_cnt_aicore() - wait_t0;\n"
    "    }\n    return true;\n}\n",
)
sub(
    E,
    "    uint64_t start = get_sys_cnt_aicore();\n    uint32_t checks = 0;\n    uint64_t misses = 0;\n    while (true) {\n",
    "    uint64_t start = get_sys_cnt_aicore();\n    const uint64_t entry_t0 = start;\n"
    "    uint32_t checks = 0;\n    uint64_t misses = 0;\n    while (true) {\n",
)
sub(
    E,
    "            *entry = value;\n            return true;\n",
    "            *entry = value;\n            cfg.kx[2] += get_sys_cnt_aicore() - entry_t0;\n            return true;\n",
)
sub(
    E,
    "    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n",
    "    uint64_t exp_counts[7] = {};\n    cfg.exp = exp_counts;\n    uint64_t kx[5] = {};\n    cfg.kx = kx;\n",
)
sub(
    E,
    "            scheduler_gm_fetch_add(reinterpret_cast<__gm__ uint64_t *>(control->resolved_reserved)[i], exp_counts[i]);\n"
    "    }\n",
    "            scheduler_gm_fetch_add(reinterpret_cast<__gm__ uint64_t *>(control->resolved_reserved)[i], exp_counts[i]);\n"
    "    }\n"
    "    if (kx[0] != 0) {\n"
    "        __gm__ uint64_t *w = reinterpret_cast<__gm__ uint64_t *>(TICKET_EXPR->reserved);\n"
    "        scheduler_gm_fetch_add(w[0], (uint64_t{1} << 40) + kx[0]);\n"
    "        scheduler_gm_fetch_add(w[1], kx[1]);\n"
    "        scheduler_gm_fetch_add(w[2], kx[2]);\n"
    "        scheduler_gm_fetch_add(w[3], kx[3]);\n"
    "        scheduler_gm_fetch_add(w[4], kx[4]);\n"
    "        for (uint64_t old = scheduler_gm_query(w[5]); old == 0 || kx[3] < old;) {\n"
    "            const uint64_t seen = scheduler_gm_compare_exchange(w[5], old, kx[3]);\n"
    "            if (seen == old) break;\n"
    "            old = seen;\n"
    "        }\n"
    "        for (uint64_t old = scheduler_gm_query(w[6]); kx[4] > old;) {\n"
    "            const uint64_t seen = scheduler_gm_compare_exchange(w[6], old, kx[4]);\n"
    "            if (seen == old) break;\n"
    "            old = seen;\n"
    "        }\n"
    "    }\n",
)

X = "src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp"
sub(X, "    uint64_t kernel_cycles{0};\n", "    uint64_t kernel_cycles{0};\n    uint64_t first_kernel_start{0};\n    uint64_t last_kernel_end{0};\n")
sub(X, "            uint64_t kernel_start = commit_executor_trace || phase_timing_enabled ? get_sys_cnt_aicore() : 0;\n",
    "            uint64_t kernel_start = get_sys_cnt_aicore();\n")
sub(X, "            uint64_t kernel_end = commit_executor_trace || phase_timing_enabled ? get_sys_cnt_aicore() : 0;\n",
    "            uint64_t kernel_end = get_sys_cnt_aicore();\n"
    "            if (stats->first_kernel_start == 0) stats->first_kernel_start = kernel_start;\n"
    "            stats->last_kernel_end = kernel_end;\n")
sub(X, "            if (phase_timing_enabled) {\n                ++stats->executed_task_count;\n",
    "            {\n                ++stats->executed_task_count;\n")
sub(X, "    context->kernel_cycles = stats.kernel_cycles;\n",
    "    context->kernel_cycles = stats.kernel_cycles;\n"
    "    context->topology_reserved[0] = stats.first_kernel_start;\n"
    "    context->topology_reserved[1] = stats.last_kernel_end;\n"
    "    scheduler_publish_cache_line(&context->topology_reserved[0]);\n"
    "    scheduler_publish_cache_line(&context->topology_reserved[1]);\n")

A = "src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp"
sub(
    A,
    "        for (int i = 0; i < 7; ++i)\n            v[i] = 0;\n",
    "        for (int i = 0; i < 7; ++i)\n            v[i] = 0;\n"
    "        for (int t = 0; t < 2; ++t) {\n"
    "            uint64_t *k = reinterpret_cast<uint64_t *>(reinterpret_cast<uint8_t *>(control) + control->tickets_offset +\n"
    "                                                       t * simpler::hbg::percounter::PERCOUNTER_TICKET_CAPACITY * 64 + 8);\n"
    "            cache_invalidate_range(k - 1, 64);\n"
    "            LOG_WARN(\n"
    "                \"EXPW percounter type=%d lanes=%\" PRIu64 \" kernels=%\" PRIu64 \" kcycles=%\" PRIu64 \" wait=%\" PRIu64\n"
    "                \" sum_first=%\" PRIu64 \" sum_last=%\" PRIu64 \" min_first=%\" PRIu64 \" max_last=%\" PRIu64,\n"
    "                t, k[0] >> 40, k[0] & ((uint64_t{1} << 40) - 1), k[1], k[2], k[3], k[4], k[5], k[6]\n"
    "            );\n"
    "            for (int i = 0; i < 7; ++i)\n"
    "                k[i] = 0;\n"
    "            cache_flush_range(k - 1, 64);\n"
    "        }\n",
)
sub(A, "    uint64_t t[17] = {};\n",
    "    uint64_t t[17] = {};\n    uint64_t kw[2][7] = {};\n")
sub(
    A,
    "        for (int i = 0; i < 17; ++i)\n            t[i] += f[i];\n    }\n",
    "        for (int i = 0; i < 17; ++i)\n            t[i] += f[i];\n"
    "        const int type = c->cluster_worker_ids[0] == static_cast<uint64_t>(w) ? 0 : 1;\n"
    "        const uint64_t first = c->topology_reserved[0];\n"
    "        const uint64_t last = c->topology_reserved[1];\n"
    "        if (first != 0) {\n"
    "            kw[type][0] += 1;\n"
    "            kw[type][1] += c->executed_task_count;\n"
    "            kw[type][2] += c->kernel_cycles;\n"
    "            kw[type][3] += first;\n"
    "            kw[type][4] += last;\n"
    "            if (kw[type][5] == 0 || first < kw[type][5]) kw[type][5] = first;\n"
    "            if (last > kw[type][6]) kw[type][6] = last;\n"
    "        }\n"
    "    }\n"
    "    for (int type = 0; type < 2; ++type)\n"
    "        LOG_WARN(\n"
    "            \"EXPW resident type=%d lanes=%\" PRIu64 \" kernels=%\" PRIu64 \" kcycles=%\" PRIu64 \" wait=0 sum_first=%\" PRIu64\n"
    "            \" sum_last=%\" PRIu64 \" min_first=%\" PRIu64 \" max_last=%\" PRIu64,\n"
    "            type, kw[type][0], kw[type][1], kw[type][2], kw[type][3], kw[type][4], kw[type][5], kw[type][6]\n"
    "        );\n",
)
_p = root / E
_t = _p.read_text()
_p.write_text(_t.replace("TICKET_EXPR", "cfg.tickets" if "PercounterTicket *tickets;" in _t else "cfg.ticket"))
print("patched window")
