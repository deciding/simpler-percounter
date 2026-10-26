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
 * The percounter AICore executor: every lane schedules itself.
 *
 * A lane draws a range of positions from its core type's ticket, spins on each
 * task's producer counters until they read published, runs the kernel, and
 * publishes by adding to its own counter. There is no scheduler lane, no ready
 * queue and no AICPU dispatch; the AICPU only watches the progress count.
 *
 * Startup and retirement are the runtime's: the lane reports through the
 * handshake, waits for the AICPU's READY and the DATA_MAIN_BASE release, and
 * after EXIT answers the COND nonce exactly as the resident executor does.
 *
 * Header-only because aicore_executor.cpp is compiled by name into unit tests.
 */

#pragma once

#include "aicore/aicore.h"
#include "common/platform_config.h"
#include "dispatch_payload.h"
#include "runtime.h"
#include "scheduler/percounter/percounter_types.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_layout.h"
#include "scheduler/scheduler_memory.h"
#include "scheduler/scheduler_ready.h"
#include "scheduler/scheduler_topology.h"
#include "scheduler/scheduler_watchdog.h"

namespace simpler::hbg::percounter {

typedef void (*PercounterKernelFunc)(__gm__ int64_t *);

/**
 * Read a counter the way the runtime reads any cross-core word.
 *
 * Not a volatile dereference: on silicon a volatile load never observed a
 * remote atomic across two million polls with the producer confirmed to have
 * published. `ld_dev` reaches a coherent point (~98 cycles at realistic
 * fanout). Not `AtomicAdd(ptr, 0)` either: that is a read-modify-write and
 * would queue every poll at the atomic unit.
 */
inline __aicore__ uint64_t percounter_observe(uint64_t counter_address) {
    __gm__ uint64_t *word = reinterpret_cast<__gm__ uint64_t *>(counter_address);
    return scheduler_gm_query(*word);
}

/**
 * How many finished tasks a lane accumulates before folding them into the
 * shared progress count.
 *
 * Not per task: every lane adding to one address would queue at the atomic
 * unit. Not once at the end either: the AICPU progress watchdog would see no
 * progress on a healthy run. It must stay below the tasks a lane runs, or the
 * fold never fires before the end; 8 clears `mixed_fanin32_1024`, the smallest
 * measured shape at about 9 tasks per lane.
 */
inline constexpr uint64_t PERCOUNTER_COMPLETION_BATCH = 8;

/** How often a spinning lane looks at the exit register and the error word. */
inline constexpr uint32_t PERCOUNTER_SPIN_CHECK_INTERVAL = 64;

/**
 * Consecutive missed polls at full rate before a lane spaces them out, and how
 * long it then spins on the cycle counter between polls.
 *
 * `SPIN_WAIT_HINT()` is a no-op on this hardware, so without this every lane
 * waiting on a wide-fanout counter issues back-to-back loads to one line, and
 * the producer's publishing atomic has to win that line against all of them.
 * The pause reads the cycle counter, not the contended line, and is a bounded
 * spin, not a sleep.
 *
 * `AFTER` only decides when spacing starts (16 and 32 measured the same).
 * `CYCLES` is the duty cycle while spaced: 1024 -> 256 took `mixed_chain_1024`
 * from 3182 to 3029 us, while `mixed_fanin32_1024` (52 lanes on one counter)
 * stayed at 295 us. Both were measured on one a5 part; re-validating them
 * elsewhere needs the wide-fanout shape, the only one that shows the failure.
 */
inline constexpr uint64_t PERCOUNTER_SPIN_BACKOFF_AFTER = 16;
inline constexpr uint64_t PERCOUNTER_SPIN_BACKOFF_CYCLES = 256;

struct LaneConfig {
    __gm__ PercounterControl *control;
    SchedulerGraphView graph;
    __gm__ uint64_t *callable_addresses;
    uint32_t callable_count;
    __gm__ DispatchPayload *payload;
    __gm__ PercounterTicket *ticket;
    __gm__ uint64_t *order;
    // Resolved once: the host writes these offsets at bind and never changes
    // them, so re-deriving them per probe would spend device loads on constants.
    __gm__ PercounterFanin *fanin;
    __gm__ uint64_t *fanin_addresses;
    __gm__ PercounterCounter *counters;
    uint64_t task_count;
    uint64_t order_count;
};

/** Where a task's producer counters are: `addresses[0 .. count)`. */
struct PercounterFaninView {
    __gm__ uint64_t *addresses;
    int32_t count;
};

/**
 * Locate `task_id`'s producers.
 *
 * A device load, not a dereference. The table reaches the device by DMA, which
 * does not invalidate this core's scalar data cache, so a plain load can return
 * whatever the line held before the image landed -- on silicon a 64-task chain
 * read a producer address of 0x120000000000 for its 60th task. Invisible in
 * simulation, where every load is coherent.
 */
inline __aicore__ PercounterFaninView percounter_fanin_view(const LaneConfig &cfg, int64_t task_id) {
    const uint64_t record = scheduler_gm_query_u32_pair(reinterpret_cast<__gm__ uint32_t *>(&cfg.fanin[task_id]));
    PercounterFaninView view{};
    view.count = static_cast<int32_t>(record & 0xffffffffULL);
    view.addresses = cfg.fanin_addresses + static_cast<int32_t>(record >> 32);
    return view;
}

/**
 * One pass over `task_id`'s producers: true when every one has published.
 *
 * Never blocks, which is what lets a lane hold a range without serialising on
 * its head. It stops at the first unpublished producer, so an unready task
 * costs as little as one load and a ready one costs one per producer.
 */
inline __aicore__ bool percounter_fanin_ready(const LaneConfig &cfg, int64_t task_id) {
    const PercounterFaninView view = percounter_fanin_view(cfg, task_id);
    for (int32_t i = 0; i < view.count; ++i) {
        if (!percounter_counter_published(percounter_observe(scheduler_gm_query(view.addresses[i])))) return false;
    }
    return true;
}

/**
 * Run one task drawn from this lane's list. Its single active subtask is
 * whichever kernel slot is set; a task with none (a dependency-only task)
 * runs nothing and is published like any other. `kernel_start` receives the
 * timestamp before the kernel call, or stays untouched when there is none.
 *
 * Returns false when the callable or the payload cannot be resolved.
 */
inline __aicore__ bool percounter_run_task(const LaneConfig &cfg, int64_t task_id, uint64_t *kernel_start) {
    // The whole data cache, before anything of this task is read. Kernels load
    // GM with plain scalar reads and never invalidate first, so a line this
    // core cached during an earlier task would survive another core rewriting
    // it. The legacy executor invalidates the same way on every dispatch.
    scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(cfg.payload));

    __gm__ int32_t *kernel_ids = reinterpret_cast<__gm__ int32_t *>(
        scheduler_graph_descriptor(cfg.graph, task_id) + SCHEDULER_GRAPH_KERNEL_IDS_OFFSET
    );
    int32_t subtask_slot = -1;
    int32_t kernel_id = SCHEDULER_GRAPH_INVALID_KERNEL_ID;
    for (int32_t slot = 0; slot < PLATFORM_CORES_PER_BLOCKDIM; ++slot) {
        if (kernel_ids[slot] >= 0) {
            subtask_slot = slot;
            kernel_id = kernel_ids[slot];
            break;
        }
    }
    if (subtask_slot < 0) return true;

    uint64_t function_address = 0;
    if (!scheduler_lookup_callable_address(
            cfg.callable_addresses, cfg.callable_count, static_cast<uint16_t>(kernel_id), &function_address
        )) {
        return false;
    }
    const SchedulerTaskInfo info{task_id, kernel_id, subtask_slot, subtask_slot == 0 ? CoreType::AIC : CoreType::AIV};
    if (scheduler_materialize_task_payload_resolved(cfg.graph, info, function_address, cfg.payload) !=
        SchedulerGraphResult::OK) {
        return false;
    }
    cfg.payload->global_context.sub_block_id = subtask_slot == 2 ? 1 : 0;
    scheduler_writeback_dispatch_payload(cfg.payload);

    // The barrier is the publication fence a consumer's spin depends on: the
    // kernel's outputs are in memory before the atomic that announces them.
    PercounterKernelFunc kernel = reinterpret_cast<PercounterKernelFunc>(cfg.payload->function_bin_addr);
    *kernel_start = get_sys_cnt_aicore();
    kernel(reinterpret_cast<__gm__ int64_t *>(cfg.payload->args));
    OUT_OF_ORDER_STORE_BARRIER();
    return true;
}

/** Record one finished subtask of `task_id`; called after its store barrier. */
inline __aicore__ void percounter_publish_completion(const LaneConfig &cfg, int64_t task_id) {
    scheduler_gm_fetch_add(cfg.counters[task_id].value, uint64_t{1});
}

inline __aicore__ void percounter_fail(const LaneConfig &cfg, SchedulerGraphResult result) {
    // First error wins.
    scheduler_gm_compare_exchange(cfg.control->lane_error, uint64_t{0}, static_cast<uint64_t>(result));
}

/**
 * Block until every producer of `task_id` has published.
 *
 * Returns false on EXIT, on another lane's error, or when this producer has
 * been late past the watchdog, which it records as this lane's error.
 */
inline __aicore__ bool percounter_wait_for_fanin(const LaneConfig &cfg, int64_t task_id, uint64_t timeout_cycles) {
    const PercounterFaninView view = percounter_fanin_view(cfg, task_id);
    for (int32_t i = 0; i < view.count; ++i) {
        const uint64_t address = scheduler_gm_query(view.addresses[i]);
        const uint64_t start = get_sys_cnt_aicore();
        uint32_t checks = 0;
        uint64_t misses = 0;
        while (!percounter_counter_published(percounter_observe(address))) {
            if (++misses > PERCOUNTER_SPIN_BACKOFF_AFTER) {
                const uint64_t until = get_sys_cnt_aicore() + PERCOUNTER_SPIN_BACKOFF_CYCLES;
                while (get_sys_cnt_aicore() < until) {}
            }
            if (++checks == PERCOUNTER_SPIN_CHECK_INTERVAL) {
                checks = 0;
                if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL ||
                    scheduler_gm_query(cfg.control->lane_error) != 0) {
                    return false;
                }
                if (scheduler_watchdog_expired(start, get_sys_cnt_aicore(), timeout_cycles)) {
                    percounter_fail(cfg, SchedulerGraphResult::TIMEOUT);
                    return false;
                }
            }
            SPIN_WAIT_HINT();
        }
    }
    return true;
}

/**
 * The lane loop.
 *
 * Every lane of a core type draws from that type's single ticket, and a draw
 * yields a range of consecutive positions in the list. Whichever lane is free
 * takes the next range, and nothing is bound to a lane in advance. Deadlock
 * freedom follows from the list order alone -- see percounter_claim_range --
 * so the lane needs no topology.
 *
 * Within its range the lane takes whichever position is ready rather than the
 * next in order, and blocks only when none is. A range is otherwise
 * head-of-line blocked by its own first task: `mixed_random_1024`, whose tasks
 * each wait on up to four producers, lost 53% going from a two-wide range to a
 * four-wide one for exactly that reason.
 *
 * `pending` is which positions are left to run, bit 0 the lowest. Readiness is
 * never cached in it: it belongs to other cores and is re-read every round.
 * With `BATCH = 2`, a range whose first task A waits and whose second B is ready:
 *
 *   pending=0b11  probe A: no, probe B: yes   -> run B, pending=0b01
 *   pending=0b01  probe A: no, scan empty     -> block on A (lowest)
 *   pending=0b01  A published                 -> run A, pending=0b00
 *   pending=0b00                              -> draw the next range
 *
 * **The fallback takes the lowest pending position.** A higher one can have a
 * producer that is a lower position in this same range, which only this lane
 * can run. The lowest cannot: every position before it in the list is complete
 * or held by another lane, so it is the one the deadlock argument names.
 */
inline __aicore__ void percounter_lane_loop(const LaneConfig &cfg, uint64_t timeout_cycles) {
    uint64_t local_done = 0;
    uint64_t range_first = 0;
    uint32_t pending = 0;
    // A range withholds its positions from every other lane, which pays with
    // short kernels (fewer ticket atomics) and costs with long ones: on qwen3
    // decode a lane drew ahead into a 110-task fan-out and the wave ran on 24 of
    // 32 AIC lanes. So a lane whose last kernel was long draws one at a time.
    uint64_t batch = PERCOUNTER_CLAIM_BATCH;
    static_assert(
        PERCOUNTER_CLAIM_BATCH <= 32, "the pending set is a uint32_t bitmask, one bit per position in the range"
    );

    while (true) {
        if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL) break;
        if (scheduler_gm_query(cfg.control->lane_error) != 0) break;

        if (pending == 0) {
            const uint64_t base = scheduler_gm_fetch_add(cfg.ticket->next, batch);
            uint64_t range_end = 0;
            // False means the list is spent: draws keep rising past its end, so
            // every lane arriving after the last task leaves here.
            if (!percounter_claim_range(base, batch, cfg.order_count, &range_first, &range_end)) break;
            pending = static_cast<uint32_t>((uint64_t{1} << (range_end - range_first)) - 1);
        }

        uint32_t slot = 0;
        bool found = false;
        int64_t task_id = -1;
        for (uint32_t probe = pending; probe != 0; probe &= probe - 1) {
            const uint32_t candidate = static_cast<uint32_t>(__builtin_ctz(probe));
            const int64_t id = percounter_order_task_id(scheduler_gm_query(cfg.order[range_first + candidate]));
            if (id < 0 || static_cast<uint64_t>(id) >= cfg.task_count) {
                percounter_fail(cfg, SchedulerGraphResult::INVALID_TASK_ID);
                return;
            }
            if (percounter_fanin_ready(cfg, id)) {
                slot = candidate;
                task_id = id;
                found = true;
                break;
            }
        }
        if (!found) {
            slot = static_cast<uint32_t>(__builtin_ctz(pending));
            task_id = percounter_order_task_id(scheduler_gm_query(cfg.order[range_first + slot]));
        }
        pending &= ~(uint32_t{1} << slot);
        // Only the blocking path waits: the scan above already read a ready
        // task's producers to the end.
        if (!found && !percounter_wait_for_fanin(cfg, task_id, timeout_cycles)) break;

        uint64_t kernel_start = 0;
        if (!percounter_run_task(cfg, task_id, &kernel_start)) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_CALLABLE);
            break;
        }
        if (kernel_start != 0) {
            batch = get_sys_cnt_aicore() - kernel_start >= PERCOUNTER_LONG_KERNEL_CYCLES ? 1 : PERCOUNTER_CLAIM_BATCH;
        }

        percounter_publish_completion(cfg, task_id);
        if (++local_done >= PERCOUNTER_COMPLETION_BATCH) {
            scheduler_gm_fetch_add(cfg.control->resolved_task_count, local_done);
            local_done = 0;
        }
    }
    if (local_done != 0) scheduler_gm_fetch_add(cfg.control->resolved_task_count, local_done);
}

/**
 * Resolve a lane's view of the region and run its loop.
 *
 * Every field goes through a device load: the block is DMA-written, and a lane
 * that once read `dispatch_payloads_offset` as 0 placed its payload inside the
 * ticket array.
 */
inline __aicore__ void run_percounter_lane(__gm__ PercounterControl *control, int block_idx, CoreType core_type) {
    const bool is_aic = core_type == CoreType::AIC;
    const uint32_t type_index = is_aic ? 0U : 1U;
    SchedulerClusterCoordinate coordinate{-1, -1};
    if (!scheduler_cluster_coordinate_from_worker(
            block_idx, is_aic, static_cast<int32_t>(scheduler_gm_query(control->cluster_count)),
            static_cast<int32_t>(scheduler_gm_query(control->aiv_per_cluster)), &coordinate
        )) {
        // A lane outside the topology the AICPU published has no payload slot.
        scheduler_gm_compare_exchange(
            control->lane_error, uint64_t{0}, static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS)
        );
        return;
    }

    LaneConfig cfg{};
    cfg.control = control;
    cfg.graph = SchedulerGraphView{
        scheduler_gm_query(control->graph_storage_address), 0, scheduler_gm_query(control->graph_storage_count), 0
    };
    cfg.callable_addresses =
        reinterpret_cast<__gm__ uint64_t *>(scheduler_gm_query(control->callable_addresses_address));
    cfg.callable_count = static_cast<uint32_t>(scheduler_gm_query(control->callable_addresses_count));
    cfg.payload = scheduler_state_at<DispatchPayload>(
        control, scheduler_gm_query(control->dispatch_payloads_offset) +
                     static_cast<uint64_t>(block_idx) * sizeof(DispatchPayload)
    );
    cfg.ticket = scheduler_state_at<PercounterTicket>(
        control,
        scheduler_gm_query(control->tickets_offset) + static_cast<uint64_t>(type_index) * sizeof(PercounterTicket)
    );
    cfg.order = scheduler_state_at<uint64_t>(control, scheduler_gm_query(control->orders_offset[type_index]));
    cfg.fanin = scheduler_state_at<PercounterFanin>(control, scheduler_gm_query(control->fanin_offset));
    cfg.fanin_addresses = scheduler_state_at<uint64_t>(control, scheduler_gm_query(control->fanin_addr_offset));
    cfg.counters = scheduler_state_at<PercounterCounter>(control, scheduler_gm_query(control->counters_offset));
    cfg.task_count = scheduler_gm_query(control->task_count);
    cfg.order_count = scheduler_gm_query(control->order_count[type_index]);

    percounter_lane_loop(cfg, scheduler_gm_query(control->scheduler_timeout_cycles));
}

/**
 * A percounter AICore, from its report to the COND nonce acknowledgement.
 *
 * The same startup and retirement as the resident executor, minus the
 * scheduler bootstrap: the host already built every table, so a lane goes from
 * READY straight to the register release and into its loop.
 */
inline __aicore__ void percounter_aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake *handshake = (__gm__ Handshake *)(&runtime->dev.workers[block_idx]);
    __gm__ PercounterControl *control =
        reinterpret_cast<__gm__ PercounterControl *>(runtime->dev.scheduler_bootstrap.worker_context_base);
    scheduler_observe_cache_line(handshake);

    const uint64_t report_epoch = get_aicore_report_epoch();
    if (report_epoch != 0) {
        aicore_stage_native_report(
            handshake, get_physical_core_id(), core_type, static_cast<uint32_t>(block_idx) + 1, 0
        );
        OUT_OF_ORDER_STORE_BARRIER();
        handshake->report_epoch = report_epoch;
    } else {
        handshake->physical_core_id = get_physical_core_id();
        handshake->core_type = core_type;
        OUT_OF_ORDER_STORE_BARRIER();
        handshake->aicore_done = block_idx + 1;
    }
    dcci(handshake, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);

    // The host prefills the timeout, so it bounds this wait before the AICPU
    // has written anything.
    scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(control));
    const uint64_t timeout_cycles = scheduler_gm_query(control->scheduler_timeout_cycles);
    bool released = false;
    uint32_t signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
    const uint64_t ready_wait_start = get_sys_cnt_aicore();
    while (handshake->aicpu_ready != SCHEDULER_RUNTIME_MODE_PERCOUNTER && signal != AICORE_EXIT_SIGNAL) {
        scheduler_observe_cache_line(handshake);
        if (scheduler_watchdog_expired(ready_wait_start, get_sys_cnt_aicore(), timeout_cycles)) {
            scheduler_gm_compare_exchange(
                control->lane_error, uint64_t{0}, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)
            );
            signal = AICORE_EXIT_SIGNAL;
            break;
        }
        signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        SPIN_WAIT_HINT();
    }
    if (signal != AICORE_EXIT_SIGNAL) {
        // The AICPU published the topology before READY; re-observe the block.
        scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(control));
        const uint64_t release_wait_start = get_sys_cnt_aicore();
        while (signal != AICPU_IDLE_TASK_ID && signal != AICORE_EXIT_SIGNAL) {
            if (scheduler_watchdog_expired(release_wait_start, get_sys_cnt_aicore(), timeout_cycles)) {
                scheduler_gm_compare_exchange(
                    control->lane_error, uint64_t{0}, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)
                );
                break;
            }
            signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
            SPIN_WAIT_HINT();
        }
        released = signal == AICPU_IDLE_TASK_ID;
    }
    if (released) {
        write_reg(RegId::COND, AICORE_IDLE_VALUE);
        run_percounter_lane(control, block_idx, core_type);
        const uint64_t exit_wait_start = get_sys_cnt_aicore();
        while (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) != AICORE_EXIT_SIGNAL) {
            if (scheduler_watchdog_expired(exit_wait_start, get_sys_cnt_aicore(), timeout_cycles)) {
                scheduler_gm_compare_exchange(
                    control->lane_error, uint64_t{0}, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)
                );
                break;
            }
            SPIN_WAIT_HINT();
        }
    }
    // A local watchdog may have ended any wait above before the AICPU signalled
    // EXIT. The AICPU publishes this run's EXIT only after resetting the return
    // gate, so wait for it before acknowledging, and complete the preceding GM
    // writes first.
    wait_for_aicpu_exit_signal();
    OUT_OF_ORDER_STORE_BARRIER();
    acknowledge_aicore_cond_nonce_and_wait(&runtime->dev.teardown_gates[block_idx]);
}

}  // namespace simpler::hbg::percounter
