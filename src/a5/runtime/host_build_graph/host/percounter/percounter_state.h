/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * The bind's percounter decision and, when percounter takes the graph, its
 * scheduler state.
 *
 * Two entry points, both called from create_scheduler_state:
 *
 *  - check_scheduler_request, before anything else: rejects an unrecognised
 *    SIMPLER_HBG_SCHEDULER_TYPE, and logs the fallback for a graph-execution
 *    run, which the default selection hands to legacy before the per-task walk.
 *  - select_percounter_state, after the default selection's per-task walk:
 *    reuses that walk's validation and metadata, and either builds the
 *    percounter region or logs why the graph falls back.
 *
 * Header-only because the UT targets that compile runtime_maker.cpp list their
 * sources by name.
 */

#pragma once

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "common/host_api.h"
#include "common/host_phase_kind.h"
#include "common/unified_log.h"
#include "host_build_graph/runtime.h"
#include "host_build_graph/shared_memory.h"
#include "intrinsic.h"
#include "percounter_plan.h"
#include "percounter_selection.h"
#include "scheduler/scheduler_layout.h"
#include "scheduler/scheduler_types.h"

namespace simpler::hbg::percounter {

inline bool percounter_requested() {
    SchedulerRequest request = SchedulerRequest::DEFAULT;
    return parse_scheduler_request(std::getenv(SCHEDULER_TYPE_ENV), &request) &&
           request == SchedulerRequest::PERCOUNTER;
}

inline void log_fallback(int32_t total_tasks, const char *reason, int64_t task_id) {
    LOG_INFO(
        "A5 HBG: %s=percounter for %d tasks: task id=%" PRId64 " is %s, which percounter does not run; keeping the "
        "default scheduler selection",
        SCHEDULER_TYPE_ENV, total_tasks, task_id, reason
    );
}

/**
 * Settle this bind's scheduler request before the default selection runs.
 * Returns false, failing the bind, only for an unrecognised value.
 */
inline bool check_scheduler_request(SharedMemoryHandle &host_sm_handle, int32_t total_tasks) {
    SchedulerRequest request = SchedulerRequest::DEFAULT;
    if (!resolve_scheduler_request(&request)) return false;
    if (request != SchedulerRequest::PERCOUNTER) return true;
    for (int64_t task_id = 0; task_id < total_tasks; ++task_id) {
        if (host_sm_handle.header->tasks.get_slot_state_by_task_id(task_id).task_kind == TaskKind::GRAPH) {
            log_fallback(total_tasks, "a graph-execution task", task_id);
            break;
        }
    }
    return true;
}

enum class StateSelection : uint8_t {
    DEFAULT,     // percounter was not requested, or does not run this graph
    PERCOUNTER,  // the percounter region is built and published; the bind is done
    FAILED,      // percounter was requested and accepted the graph, but building it failed
};

/**
 * Take the graph for percounter when it was requested and every task has a
 * shape percounter runs: one active subtask (AIC, or AIV0), one block, no
 * sync start and no dispatch predicate.
 *
 * `task_metadata` and `inline_completed_task_ids` are the default selection's
 * per-task walk, which has already rejected any malformed task and checked
 * every kernel id against the callable table.
 */
inline StateSelection select_percounter_state(
    Runtime *runtime, const HostApi *api, SharedMemoryHandle &host_sm_handle, int32_t total_tasks,
    const std::vector<SchedulerTaskMetadata> &task_metadata, const std::vector<int64_t> &inline_completed_task_ids,
    const sm_layout::SegmentOffsets &device_segments
) {
    if (!percounter_requested()) return StateSelection::DEFAULT;

    const size_t task_count = static_cast<size_t>(total_tasks);
    std::vector<bool> inline_completed(task_count, false);
    for (int64_t id : inline_completed_task_ids)
        inline_completed[static_cast<size_t>(id)] = true;

    std::vector<TaskInput> inputs(task_count);
    std::vector<int32_t> edges;
    std::vector<size_t> edge_begin(task_count, 0);
    uint64_t order_count[PERCOUNTER_CORE_TYPE_COUNT] = {0, 0};
    for (size_t t = 0; t < task_count; ++t) {
        const int64_t task_id = static_cast<int64_t>(t);
        ChipTaskSlotState &slot = host_sm_handle.header->tasks.get_slot_state_by_task_id(task_id);
        TaskInput &input = inputs[t];
        input.inline_completed = inline_completed[t];
        if (!input.inline_completed) {
            if (slot.logical_block_num != 1) {
                log_fallback(total_tasks, "an SPMD task", task_id);
                return StateSelection::DEFAULT;
            }
            if (slot.task_attrs.requires_sync_start()) {
                log_fallback(total_tasks, "a sync-start task", task_id);
                return StateSelection::DEFAULT;
            }
            if (slot.task_attrs.has_predicate()) {
                log_fallback(total_tasks, "a predicated task", task_id);
                return StateSelection::DEFAULT;
            }
            const TaskPlacement placement =
                task_placement_from_active_mask(task_metadata[t].active_mask, &input.core_type_index);
            if (placement != TaskPlacement::AIC && placement != TaskPlacement::AIV) {
                log_fallback(total_tasks, "a MIX task", task_id);
                return StateSelection::DEFAULT;
            }
            ++order_count[input.core_type_index];
        }
        const TaskPayload &payload = slot.to_payload();
        input.fanin_count = payload.fanin_count;
        edge_begin[t] = edges.size();
        for (int32_t e = 0; e < payload.fanin_count; ++e)
            edges.push_back(payload.fanin_data()[e]);
    }
    for (size_t t = 0; t < task_count; ++t)
        inputs[t].fanin_ids = inputs[t].fanin_count > 0 ? edges.data() + edge_begin[t] : nullptr;

    PercounterLayout layout{};
    if (!plan_layout(task_count, task_count, edges.size(), order_count, &layout)) {
        LOG_ERROR("A5 HBG percounter: state layout overflow");
        return StateSelection::FAILED;
    }
    void *device_state = nullptr;
    void *host_base = nullptr;
    if (api->acquire_scheduler_state_storage(
            static_cast<size_t>(layout.total_size), static_cast<size_t>(SCHEDULER_STATE_ALIGNMENT), &device_state,
            &host_base
        ) != 0 ||
        device_state == nullptr || host_base == nullptr) {
        LOG_ERROR("A5 HBG percounter: failed to obtain %" PRIu64 " scheduler state bytes", layout.total_size);
        return StateSelection::FAILED;
    }
    const uint64_t device_base = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_state));

    ExternalAddresses external{};
    external.graph_storage = reinterpret_cast<uint64_t>(runtime->get_gm_sm_ptr()) + device_segments.storage;
    external.graph_storage_count = task_count;
    external.callable_table = runtime->callable_entry_table_addr();
    external.callable_count = static_cast<uint64_t>(runtime->callable_table_len());
    BuildResult build{};
    if (!build_tables(inputs.data(), task_count, layout, host_base, device_base, external, &build)) {
        LOG_ERROR(
            "A5 HBG percounter: %s (task id=%" PRId64 ", fanin index=%d)", build_status_name(build.status),
            build.task_id, build.fanin_index
        );
        return StateSelection::FAILED;
    }

    // Each lane's payload keeps these fields for the whole run: its argument
    // vector points at its own contexts, and a lane runs one block of a
    // non-deferred task.
    for (uint64_t lane = 0; lane < SCHEDULER_WORKER_CAPACITY; ++lane) {
        const uint64_t offset = layout.dispatch_payloads_offset + lane * sizeof(DispatchPayload);
        auto *payload = region_at<DispatchPayload>(host_base, offset);
        const uint64_t device_payload = device_base + offset;
        payload->args[PAYLOAD_LOCAL_CONTEXT_INDEX] = device_payload + offsetof(DispatchPayload, local_context);
        payload->args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = device_payload + offsetof(DispatchPayload, global_context);
        payload->local_context.block_idx = 0;
        payload->local_context.block_num = 1;
        payload->local_context.async_ctx.task_token = TaskId::invalid();
        payload->src_payload = 0;
    }

    runtime->publish_scheduler_bootstrap(SCHEDULER_RUNTIME_MODE_PERCOUNTER, device_base + layout.control_offset);
    runtime->add_pending_metadata(
        device_state, host_base, static_cast<size_t>(layout.total_size), HostPhaseKind::Count, {}
    );
    LOG_INFO(
        "A5 HBG: selected percounter for %d tasks (%" PRIu64 " AIC, %" PRIu64 " AIV, %zu edges)", total_tasks,
        order_count[0], order_count[1], edges.size()
    );
    return StateSelection::PERCOUNTER;
}

}  // namespace simpler::hbg::percounter
