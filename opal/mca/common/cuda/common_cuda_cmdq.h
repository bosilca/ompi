/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/**
 * @file
 *
 * CUDA command queue active-message API.
 *
 * This interface describes a host/device command queue used by CPU threads to
 * enqueue active messages that are consumed by a resident CUDA kernel.  The
 * queue also returns every command record to the host once the CUDA kernel is
 * done with it.
 *
 * Each command queue owns one resident CUDA kernel.  opal_common_cuda_cmdq_open()
 * creates a disabled queue.  opal_common_cuda_cmdq_enable() asynchronously
 * enqueues that queue's kernel on the queue stream and moves the queue to
 * ENABLE_IN_PROGRESS.  The kernel confirms that it is polling the queue by
 * storing ENABLED in the device-visible state field.  A later call to
 * opal_common_cuda_cmdq_disable() asynchronously sends the reserved STOP AM to
 * the kernel and moves the queue to DISABLE_IN_PROGRESS.  Once the kernel has
 * completed, progress moves the queue to DISABLED.
 *
 * The queue is intentionally modeled after Open MPI's active-message style:
 * each message carries an AM id, a header, and an optional payload.  Unlike the
 * normal host-side BTL AM path, every AM id is also registered in a
 * device-visible table.  The CUDA kernel is the authority that validates and
 * accepts commands.
 *
 * Command records are bidirectional.  The host writes an AM id in the slot and
 * publishes the slot to the FIFO.  When the CUDA kernel consumes the slot, it
 * may replace the AM id with OPAL_COMMON_CUDA_CMDQ_AM_ID_RETURN as the default
 * "no response, reuse this record" tag.  If the user portion of the kernel later
 * produces an answer, it replaces that tag with the response AM id, fills the
 * response memory reserved in the same record, and then returns the record to
 * the host by setting a terminal slot state.
 *
 * Expected ownership model:
 *
 * - Many CPU threads may enqueue command AMs concurrently.
 * - One CUDA kernel, or one cooperative device-side progress context, consumes
 *   the host-to-device FIFO.
 * - The CUDA kernel always returns consumed command records to the host.
 * - Host callbacks are invoked only from opal_common_cuda_cmdq_progress().
 *
 * The FIFO backend should follow the same broad shape as the sm BTL FIFO:
 * producers publish already-filled entries with an atomic tail exchange, and
 * the single consumer owns head advancement.  Device-visible entries should be
 * referenced by byte offset, not process-relative host pointers.
 */

#ifndef OPAL_MCA_COMMON_CUDA_CMDQ_H
#define OPAL_MCA_COMMON_CUDA_CMDQ_H

#if !defined(OPAL_COMMON_CUDA_CMDQ_STANDALONE)
#    include "opal_config.h"
#else
#    if !defined(OPAL_DECLSPEC)
#        define OPAL_DECLSPEC
#    endif
#    if !defined(BEGIN_C_DECLS)
#        if defined(__cplusplus)
#            define BEGIN_C_DECLS extern "C" {
#            define END_C_DECLS   }
#        else
#            define BEGIN_C_DECLS
#            define END_C_DECLS
#        endif
#    endif
#endif

#include <stddef.h>
#include <stdint.h>

BEGIN_C_DECLS

struct opal_accelerator_stream_t;
typedef struct opal_accelerator_stream_t opal_accelerator_stream_t;

/** Current device ABI version for opal_common_cuda_cmdq_device_t. */
#define OPAL_COMMON_CUDA_CMDQ_ABI_VERSION 1

/**
 * Reserved command AM id used by opal_common_cuda_cmdq_disable().
 *
 * STOP is consumed by the queue kernel's internal STOP handler and is not
 * available for user registration.  A kernel must recognize STOP before normal
 * AM table dispatch, return the STOP record, and then complete the queue kernel.
 */
#define OPAL_COMMON_CUDA_CMDQ_AM_ID_STOP ((opal_common_cuda_cmdq_am_id_t) 1)

/** Reserved device handler selector for the queue-internal STOP handler. */
#define OPAL_COMMON_CUDA_CMDQ_DEVICE_HANDLER_STOP UINT16_MAX

/**
 * Reserved AM id used by the CUDA kernel when returning a command record that
 * has no response AM.
 */
#define OPAL_COMMON_CUDA_CMDQ_AM_ID_RETURN ((opal_common_cuda_cmdq_am_id_t) 0)

/** Registered AM requires a response AM from the queue kernel. */
#define OPAL_COMMON_CUDA_CMDQ_AM_FLAG_RESPONSE_REQUIRED 0x00000001u

/** This send requests a response AM from the queue kernel. */
#define OPAL_COMMON_CUDA_CMDQ_SEND_FLAG_RESPONSE_REQUIRED 0x00000001u

/** Invalid device-visible slot offset. */
#define OPAL_COMMON_CUDA_CMDQ_SLOT_NONE UINT32_MAX

/**
 * Default AM ids used by the CUDA datatype pack/unpack command queue.
 *
 * AM ids are queue-local, so an integration can register different ids if it
 * wants.  These defaults use the same terminology as opal_datatype_pack.h and
 * opal_datatype_unpack.h.
 */
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_PACK_PARTIAL_BLOCKLEN \
    ((opal_common_cuda_cmdq_am_id_t) 2)
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_PACK_PREDEFINED_DATATYPE \
    ((opal_common_cuda_cmdq_am_id_t) 3)
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_PACK_CONTIGUOUS_LOOP \
    ((opal_common_cuda_cmdq_am_id_t) 4)
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_UNPACK_PARTIAL_BLOCKLEN \
    ((opal_common_cuda_cmdq_am_id_t) 5)
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_UNPACK_PREDEFINED_DATATYPE \
    ((opal_common_cuda_cmdq_am_id_t) 6)
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_UNPACK_CONTIGUOUS_LOOP \
    ((opal_common_cuda_cmdq_am_id_t) 7)
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_FRAGMENT_DONE \
    ((opal_common_cuda_cmdq_am_id_t) 8)

/** Device handler selectors for datatype pack/unpack AMs. */
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_PARTIAL_BLOCKLEN 1
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_PREDEFINED_DATATYPE 2
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_CONTIGUOUS_LOOP 3
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_UNPACK_PARTIAL_BLOCKLEN 4
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_UNPACK_PREDEFINED_DATATYPE 5
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_UNPACK_CONTIGUOUS_LOOP 6

/** Maximum number of datatype copy operations in one datatype command AM. */
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_MAX_OP_COUNT 255u

/** Return the datatype operation count stored in a datatype command header. */
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_OP_COUNT(_header) \
    ((uint32_t) (_header)->op_count)

/** Check whether a datatype operation count is valid; 0 is reserved/invalid. */
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_OP_COUNT_VALID(_count) \
    (0 < (_count) && OPAL_COMMON_CUDA_CMDQ_DATATYPE_MAX_OP_COUNT >= (_count))

/** Encode a valid datatype operation count. */
#define OPAL_COMMON_CUDA_CMDQ_DATATYPE_ENCODE_OP_COUNT(_count) \
    ((uint8_t) (_count))

/** Opaque host-side command queue handle. */
typedef struct opal_common_cuda_cmdq_t opal_common_cuda_cmdq_t;

/** Opaque host-side request handle returned for an enqueued command AM. */
typedef struct opal_common_cuda_cmdq_request_t opal_common_cuda_cmdq_request_t;

/** Device-visible queue handle passed to the queue-owned CUDA kernel. */
typedef struct opal_common_cuda_cmdq_device_t opal_common_cuda_cmdq_device_t;

/**
 * Active-message id.
 *
 * AM ids are local to a queue.  The CPU registration table and the
 * device-visible registration table must agree on the meaning of every id.
 * IDs 0 and 1 are reserved by the command queue; user AM ids start at 2.
 */
typedef uint8_t opal_common_cuda_cmdq_am_id_t;

/**
 * Device-visible slot offset.
 *
 * The queue uses byte offsets rather than host pointers so variable-sized FIFO
 * slots can be understood by host code and CUDA device code.
 */
typedef uint32_t opal_common_cuda_cmdq_slot_offset_t;

/**
 * Queue-owned kernel launch callback.
 *
 * opal_common_cuda_cmdq_enable() calls this callback after changing the queue
 * state to ENABLE_IN_PROGRESS.  The callback must enqueue the queue-owned
 * kernel on the supplied stream and return once the launch has been submitted.
 */
typedef int (*opal_common_cuda_cmdq_launch_fn_t)(opal_common_cuda_cmdq_device_t *device,
                                                 opal_accelerator_stream_t *stream,
                                                 void *cbdata);

/**
 * Queue kernel lifecycle state.
 *
 * The state is host-observed and mirrored into the device-visible queue handle.
 * ENABLE and DISABLE are asynchronous operations, so callers should use
 * opal_common_cuda_cmdq_get_state() or opal_common_cuda_cmdq_progress() to
 * observe transitions out of the in-progress states.
 */
typedef enum {
    OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED = 0,
    OPAL_COMMON_CUDA_CMDQ_STATE_ENABLE_IN_PROGRESS = 1,
    OPAL_COMMON_CUDA_CMDQ_STATE_ENABLED = 2,
    OPAL_COMMON_CUDA_CMDQ_STATE_DISABLE_IN_PROGRESS = 3,
} opal_common_cuda_cmdq_state_t;

/**
 * Host-observed request status.
 *
 * QUEUED means the host has published the command.  ACCEPTED means the CUDA
 * kernel validated the AM and accepted ownership.  COMPLETED means the command
 * reached its terminal success state.  REJECTED means device validation failed
 * or the backend could not execute the command.
 */
typedef enum {
    OPAL_COMMON_CUDA_CMDQ_STATUS_QUEUED = 1,
    OPAL_COMMON_CUDA_CMDQ_STATUS_ACCEPTED = 2,
    OPAL_COMMON_CUDA_CMDQ_STATUS_COMPLETED = 3,
    OPAL_COMMON_CUDA_CMDQ_STATUS_REJECTED = 4,
} opal_common_cuda_cmdq_status_t;

/**
 * Reason for a rejected command.
 *
 * These values are intentionally generic.  AM-specific device handlers may add
 * detail in their response header or payload.
 */
typedef enum {
    OPAL_COMMON_CUDA_CMDQ_REJECT_NONE = 0,
    OPAL_COMMON_CUDA_CMDQ_REJECT_BAD_AM = 1,
    OPAL_COMMON_CUDA_CMDQ_REJECT_TOO_LARGE = 2,
    OPAL_COMMON_CUDA_CMDQ_REJECT_DEVICE_VALIDATION = 3,
    OPAL_COMMON_CUDA_CMDQ_REJECT_BACKEND_ERROR = 4,
} opal_common_cuda_cmdq_reject_reason_t;

/**
 * Device-visible slot state.
 *
 * Slot states are part of the host/device ABI.  Producers fill all slot fields,
 * issue the appropriate system-scope release operation, then publish a slot as
 * READY by linking it into a FIFO.  Consumers acquire the slot before reading
 * message fields.
 */
typedef enum {
    OPAL_COMMON_CUDA_CMDQ_SLOT_FREE = 0,
    OPAL_COMMON_CUDA_CMDQ_SLOT_READY = 1,
    OPAL_COMMON_CUDA_CMDQ_SLOT_ACCEPTED = 2,
    OPAL_COMMON_CUDA_CMDQ_SLOT_COMPLETED = 3,
    OPAL_COMMON_CUDA_CMDQ_SLOT_REJECTED = 4,
} opal_common_cuda_cmdq_slot_state_t;

/**
 * Attributes used to create a command queue.
 */
typedef struct {
    /** CUDA device id associated with the resident kernel and queue memory. */
    int device_id;

    /** Number of bytes in the device-visible FIFO. */
    uint32_t fifo_size;

    /** Maximum header + payload bytes copied inline per slot, or 0 for no extra limit. */
    uint32_t max_inline;

    /** Reserved queue creation flags.  Must be 0 for the initial backend. */
    uint32_t flags;

    /** Stream used to enqueue the queue-owned kernel and backend work. */
    opal_accelerator_stream_t *stream;

    /** Callback that enqueues the queue-owned CUDA kernel. */
    opal_common_cuda_cmdq_launch_fn_t launch;

    /** Opaque data passed to launch. */
    void *launch_cbdata;
} opal_common_cuda_cmdq_attr_t;

/**
 * Attributes for one AM id.
 *
 * Registration defines both host-side dispatch and the device-side validation
 * contract.  For the first backend, registrations should be completed before
 * the device handle is passed to a running CUDA kernel.
 */
typedef struct {
    /** AM flags, including OPAL_COMMON_CUDA_CMDQ_AM_FLAG_RESPONSE_REQUIRED. */
    uint32_t flags;

    /** Maximum accepted AM header size in bytes. */
    size_t max_header;

    /** Maximum accepted AM payload size in bytes. */
    size_t max_payload;

    /**
     * Device-side handler selector.
     *
     * The CUDA kernel interprets this value.  It can index a kernel-local
     * dispatch table or select a switch case without exposing function pointers
     * in host memory.
     */
    uint16_t device_handler_id;

    /**
     * Expected response AM id when the registration or a send requests a
     * response AM.
     *
     * If OPAL_COMMON_CUDA_CMDQ_AM_FLAG_RESPONSE_REQUIRED is set, every send for
     * this AM reserves response space.  If the AM flag is clear, individual
     * sends may still request a response with
     * OPAL_COMMON_CUDA_CMDQ_SEND_FLAG_RESPONSE_REQUIRED.
     */
    opal_common_cuda_cmdq_am_id_t response_am_id;
} opal_common_cuda_cmdq_am_attr_t;

/**
 * Common header for datatype pack/unpack command AMs.
 *
 * The CPU convertor driver walks the datatype stack and emits one or more AMs
 * whose payloads are arrays of descriptors matching one helper from
 * opal_datatype_pack.h or opal_datatype_unpack.h.  The CUDA kernel executes
 * only the copy body; the CPU keeps responsibility for convertor preamble,
 * epilog, stack updates, partial_length, bConverted, checksum, and completion.
 */
typedef struct {
    /**
     * Total bytes completed by the fragment.
     *
     * The CPU driver sets this on the AM that requests the fragment-done
     * response.  If it is 0, the CUDA kernel reports only the bytes executed by
     * that AM.
     */
    uint32_t fragment_bytes;

    /** Fragment identifier copied into the optional fragment-done response. */
    uint16_t fragment_id;

    /** Number of descriptors in the AM payload; 0 is reserved/invalid. */
    uint8_t op_count;

    /** Reserved datatype command flags.  Must be 0 for now. */
    uint8_t flags;
} opal_common_cuda_cmdq_datatype_header_t;

/** Payload descriptor for pack_partial_blocklen() / unpack_partial_blocklen(). */
typedef struct {
    /** Address of the datatype memory side. */
    uint64_t memory;

    /** Address of the packed-buffer side. */
    uint64_t packed;

    /** Number of bytes copied for this partial blocklen operation. */
    uint64_t do_now_bytes;
} opal_common_cuda_cmdq_datatype_partial_blocklen_t;

/** Payload descriptor for pack_predefined_data() / unpack_predefined_data(). */
typedef struct {
    /** Address of the first block on the datatype memory side. */
    uint64_t memory;

    /** Address of the first byte on the packed-buffer side. */
    uint64_t packed;

    /** Number of bytes in one full blocklen copy. */
    uint64_t blocklen_bytes;

    /** Extent, in bytes, between consecutive datatype memory blocks. */
    int64_t extent;

    /** Number of full blocklen copies to execute. */
    uint64_t count;
} opal_common_cuda_cmdq_datatype_predefined_datatype_t;

/** Payload descriptor for pack_contiguous_loop() / unpack_contiguous_loop(). */
typedef struct {
    /** Address of the first loop instance on the datatype memory side. */
    uint64_t memory;

    /** Address of the first byte on the packed-buffer side. */
    uint64_t packed;

    /** Number of bytes copied from each contiguous loop instance. */
    uint64_t loop_size;

    /** Extent, in bytes, between consecutive loop instances in datatype memory. */
    int64_t loop_extent;

    /** Number of loop instances copied by this descriptor. */
    uint64_t copy_loops;
} opal_common_cuda_cmdq_datatype_contiguous_loop_t;

/** Response header used once per datatype pack/unpack fragment. */
typedef struct {
    /** Total fragment bytes completed, or the bytes completed by this AM. */
    uint32_t bytes_done;

    /** Fragment identifier copied from the command AM. */
    uint16_t fragment_id;

    /** Number of descriptors completed; 0 is reserved/invalid. */
    uint8_t op_count_done;

    /** Command status, using opal_common_cuda_cmdq_status_t values. */
    uint8_t status;
} opal_common_cuda_cmdq_datatype_fragment_done_t;

/**
 * Device-visible AM registration entry.
 *
 * The host backend mirrors opal_common_cuda_cmdq_am_attr_t into this compact
 * representation so the CUDA kernel can validate AM flags and sizes without
 * calling back into host code.
 */
typedef struct {
    uint16_t device_handler_id;
    opal_common_cuda_cmdq_am_id_t response_am_id;
    uint8_t reserved8;
    uint32_t flags;
    uint64_t max_header;
    uint64_t max_payload;
} opal_common_cuda_cmdq_device_am_entry_t;

/**
 * Device-visible FIFO descriptor.
 *
 * The FIFO stores byte offsets to variable-sized command records.  Host
 * producers publish a record by atomically exchanging tail with the record
 * offset and linking the previous tail's next field.  The CUDA kernel owns head
 * while consuming records.  The host owns reclaim and can reuse returned records
 * after observing a terminal slot state.
 */
typedef struct {
    volatile opal_common_cuda_cmdq_slot_offset_t head;
    volatile opal_common_cuda_cmdq_slot_offset_t tail;
    volatile opal_common_cuda_cmdq_slot_offset_t reclaim;
    /** Base of the device-visible storage addressed by this FIFO's offsets. */
    uint8_t *base;
    /** Number of bytes available from base. */
    uint32_t size;
} opal_common_cuda_cmdq_fifo_t;

/**
 * Device-visible command slot.
 *
 * All offsets are relative to the containing FIFO's base.  The host publishes
 * the command AM id in am_id.  The CUDA kernel may overwrite am_id with
 * OPAL_COMMON_CUDA_CMDQ_AM_ID_RETURN after accepting the command.  If a response
 * is produced, the kernel overwrites am_id with the response AM id, writes the
 * response bytes starting at response_offset, updates response_header_len and
 * response_payload_len, and then sets a terminal slot state.  If am_id is
 * OPAL_COMMON_CUDA_CMDQ_AM_ID_STOP, the kernel must run the queue-internal STOP
 * handler instead of looking up the AM table.
 */
typedef struct {
    volatile opal_common_cuda_cmdq_slot_offset_t next;
    volatile uint32_t state;
    volatile uint32_t status;
    volatile uint32_t reject_reason;
    uint32_t flags;
    uint32_t record_bytes;
    opal_common_cuda_cmdq_am_id_t am_id;
    uint8_t reserved8[3];
    uint32_t header_offset;
    uint32_t header_len;
    uint32_t payload_offset;
    uint32_t payload_len;
    uint32_t response_offset;
    uint32_t response_capacity;
    uint32_t response_header_len;
    uint32_t response_payload_len;
} opal_common_cuda_cmdq_slot_t;

/**
 * Device-visible queue handle passed to the CUDA kernel.
 *
 * The handle remains valid until opal_common_cuda_cmdq_close().  Host code owns
 * allocation and teardown.  Device code owns validation and consumption of the
 * FIFO and returns consumed records in place.  The state field is the
 * device-visible lifecycle flag: the
 * host sets in-progress states, the kernel sets ENABLED once it is running, and
 * host progress sets DISABLED after kernel completion.
 */
struct opal_common_cuda_cmdq_device_t {
    uint32_t abi_version;
    uint32_t flags;
    int32_t device_id;
    uint32_t max_inline;
    volatile uint32_t state;
    opal_common_cuda_cmdq_fifo_t fifo;
    opal_common_cuda_cmdq_device_am_entry_t *am_table;
    uint32_t am_table_size;
    uint32_t reserved;
};

/**
 * Descriptor passed to response AM callbacks.
 *
 * The descriptor and the memory it points to are valid only for the duration of
 * the callback.  Callers must copy out any data needed after callback return.
 */
typedef struct {
    opal_common_cuda_cmdq_t *queue;
    opal_common_cuda_cmdq_am_id_t am_id;
    const void *header;
    size_t header_len;
    const void *payload;
    size_t payload_len;
    void *cbdata;
} opal_common_cuda_cmdq_am_desc_t;

/**
 * Completion callback for command requests.
 *
 * The callback is invoked from opal_common_cuda_cmdq_progress().  A request may
 * receive ACCEPTED before its terminal COMPLETED or REJECTED status.
 */
typedef void (*opal_common_cuda_cmdq_completion_fn_t)(
    opal_common_cuda_cmdq_request_t *request, opal_common_cuda_cmdq_status_t status,
    opal_common_cuda_cmdq_reject_reason_t reason, void *cbdata);

/**
 * Response AM callback.
 *
 * The callback is invoked when opal_common_cuda_cmdq_progress() consumes a
 * device-to-host response AM whose id was registered with a response callback.
 */
typedef void (*opal_common_cuda_cmdq_response_cb_fn_t)(
    const opal_common_cuda_cmdq_am_desc_t *descriptor);

/**
 * Per-send parameters for command AM enqueue.
 */
typedef struct {
    /** Send flags, including OPAL_COMMON_CUDA_CMDQ_SEND_FLAG_RESPONSE_REQUIRED. */
    uint32_t flags;

    /** Optional request completion callback. */
    opal_common_cuda_cmdq_completion_fn_t cbfunc;

    /** Opaque callback data passed to cbfunc. */
    void *cbdata;
} opal_common_cuda_cmdq_send_param_t;

/**
 * Open a CUDA command queue.
 *
 * @param[in]  attr   Queue creation attributes.
 * @param[out] queue  New host-side queue handle.
 *
 * @retval OPAL_SUCCESS Queue opened successfully.
 * @retval OPAL_ERR_BAD_PARAM Invalid attributes or output pointer.
 * @retval OPAL_ERROR Backend allocation or CUDA setup failed.
 */
OPAL_DECLSPEC int opal_common_cuda_cmdq_open(const opal_common_cuda_cmdq_attr_t *attr,
                                             opal_common_cuda_cmdq_t **queue);

/**
 * Close a CUDA command queue.
 *
 * The queue must be disabled before close.  All commands must have reached a
 * terminal state before the queue is disabled.
 */
OPAL_DECLSPEC int opal_common_cuda_cmdq_close(opal_common_cuda_cmdq_t *queue);

/**
 * Asynchronously enable a command queue.
 *
 * This call enqueues the queue-owned CUDA kernel on the stream stored in the
 * queue attributes and changes the lifecycle state to
 * OPAL_COMMON_CUDA_CMDQ_STATE_ENABLE_IN_PROGRESS.  The call returns before the
 * kernel is necessarily running.  Once the kernel has started polling the
 * queue, it stores OPAL_COMMON_CUDA_CMDQ_STATE_ENABLED in the device-visible
 * queue handle; host progress mirrors that state for callers.
 */
OPAL_DECLSPEC int opal_common_cuda_cmdq_enable(opal_common_cuda_cmdq_t *queue);

/**
 * Asynchronously disable a command queue.
 *
 * This call changes the lifecycle state to
 * OPAL_COMMON_CUDA_CMDQ_STATE_DISABLE_IN_PROGRESS and enqueues the reserved
 * OPAL_COMMON_CUDA_CMDQ_AM_ID_STOP command AM to the queue kernel.  The kernel's
 * queue-internal STOP handler returns the STOP record like any other command
 * record and then completes the kernel.  The call returns once STOP has been
 * published, not when the kernel has completed.  Host progress observes kernel
 * completion and changes the state to
 * OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED.
 */
OPAL_DECLSPEC int opal_common_cuda_cmdq_disable(opal_common_cuda_cmdq_t *queue);

/**
 * Return the latest host-observed lifecycle state.
 *
 * This function does not itself guarantee accelerator progress.  Call
 * opal_common_cuda_cmdq_progress() to advance state transitions that depend on
 * device-to-host records or kernel completion events.
 */
OPAL_DECLSPEC int opal_common_cuda_cmdq_get_state(opal_common_cuda_cmdq_t *queue,
                                                  opal_common_cuda_cmdq_state_t *state);

/**
 * Return the device-visible queue handle.
 *
 * The returned handle is passed to the CUDA kernel.  It remains owned by the
 * queue and must not be freed by the caller.
 */
OPAL_DECLSPEC int opal_common_cuda_cmdq_get_device_handle(opal_common_cuda_cmdq_t *queue,
                                                          opal_common_cuda_cmdq_device_t **device);

/**
 * Register one active-message id.
 *
 * Registration binds an AM id to validation limits, optional response
 * requirements, device handler selector, and host-side response callback.
 */
OPAL_DECLSPEC int opal_common_cuda_cmdq_am_register(
    opal_common_cuda_cmdq_t *queue, opal_common_cuda_cmdq_am_id_t am_id,
    const opal_common_cuda_cmdq_am_attr_t *attr,
    opal_common_cuda_cmdq_response_cb_fn_t cbfunc, void *cbdata);

/**
 * Enqueue a command active message from a CPU thread.
 *
 * This function is thread-safe with respect to other command enqueue calls on
 * the same queue.  It publishes a command to the host-to-device FIFO and returns
 * before the CUDA kernel validates the command.  Use request_test, a completion
 * callback, or queue progress to observe ACCEPTED, COMPLETED, or REJECTED.
 * User command AMs may be sent only while the queue is ENABLED.
 *
 * Header bytes are always logically part of the AM.  The backend copies
 * header/payload bytes into the variable-sized FIFO record.  If the registered
 * AM requires a response, or if this send sets
 * OPAL_COMMON_CUDA_CMDQ_SEND_FLAG_RESPONSE_REQUIRED, the backend reserves
 * response space in the same record before publishing it to the CUDA kernel.
 */
OPAL_DECLSPEC int opal_common_cuda_cmdq_am_send_nbx(
    opal_common_cuda_cmdq_t *queue, opal_common_cuda_cmdq_am_id_t am_id, const void *header,
    size_t header_len, const void *payload, size_t payload_len,
    const opal_common_cuda_cmdq_send_param_t *param,
    opal_common_cuda_cmdq_request_t **request);

/**
 * Progress returned command records and response AMs.
 *
 * This function observes returned command records, updates request state,
 * invokes request completion callbacks, dispatches response AM callbacks when
 * the returned tag is not OPAL_COMMON_CUDA_CMDQ_AM_ID_RETURN, and then makes the
 * record space reusable.
 *
 * @return Number of records progressed, or an OPAL error code.
 */
OPAL_DECLSPEC int opal_common_cuda_cmdq_progress(opal_common_cuda_cmdq_t *queue);

/**
 * Test a request for its latest host-observed status.
 *
 * A terminal status is COMPLETED or REJECTED.  The request remains owned by the
 * caller until opal_common_cuda_cmdq_request_release().
 */
OPAL_DECLSPEC int opal_common_cuda_cmdq_request_test(
    opal_common_cuda_cmdq_request_t *request, opal_common_cuda_cmdq_status_t *status,
    opal_common_cuda_cmdq_reject_reason_t *reason);

/**
 * Release a request handle returned by opal_common_cuda_cmdq_am_send_nbx().
 *
 * Releasing a non-terminal request is invalid.
 */
OPAL_DECLSPEC void opal_common_cuda_cmdq_request_release(
    opal_common_cuda_cmdq_request_t *request);

#if defined(__CUDACC__)
static __device__ __forceinline__ void opal_common_cuda_cmdq_device_yield(void)
{
    __nanosleep(100);
}

static __device__ __forceinline__ opal_common_cuda_cmdq_slot_t *
opal_common_cuda_cmdq_device_slot(opal_common_cuda_cmdq_device_t *device,
                                  opal_common_cuda_cmdq_slot_offset_t offset)
{
    return (opal_common_cuda_cmdq_slot_t *) (device->fifo.base + offset);
}

static __device__ inline opal_common_cuda_cmdq_slot_t *
opal_common_cuda_cmdq_device_pop(opal_common_cuda_cmdq_device_t *device)
{
    opal_common_cuda_cmdq_slot_offset_t offset = device->fifo.head;
    opal_common_cuda_cmdq_slot_t *slot;

    if (OPAL_COMMON_CUDA_CMDQ_SLOT_NONE == offset) {
        return NULL;
    }

    __threadfence_system();
    slot = opal_common_cuda_cmdq_device_slot(device, offset);
    while (OPAL_COMMON_CUDA_CMDQ_SLOT_READY != slot->state) {
        opal_common_cuda_cmdq_device_yield();
        __threadfence_system();
    }

    if (OPAL_COMMON_CUDA_CMDQ_SLOT_NONE == slot->next) {
        if (offset
            == atomicCAS((unsigned int *) &device->fifo.tail, offset,
                         OPAL_COMMON_CUDA_CMDQ_SLOT_NONE)) {
            device->fifo.head = OPAL_COMMON_CUDA_CMDQ_SLOT_NONE;
        } else {
            while (OPAL_COMMON_CUDA_CMDQ_SLOT_NONE == slot->next) {
                opal_common_cuda_cmdq_device_yield();
                __threadfence_system();
            }
            device->fifo.head = slot->next;
        }
    } else {
        device->fifo.head = slot->next;
    }

    __threadfence_system();
    return slot;
}

static __device__ inline opal_common_cuda_cmdq_slot_t *
opal_common_cuda_cmdq_device_wait(opal_common_cuda_cmdq_device_t *device)
{
    opal_common_cuda_cmdq_slot_t *slot;

    while (NULL == (slot = opal_common_cuda_cmdq_device_pop(device))) {
        opal_common_cuda_cmdq_device_yield();
    }

    return slot;
}

static __device__ inline void
opal_common_cuda_cmdq_device_accept(opal_common_cuda_cmdq_slot_t *slot)
{
    slot->status = OPAL_COMMON_CUDA_CMDQ_STATUS_ACCEPTED;
    __threadfence_system();
    slot->state = OPAL_COMMON_CUDA_CMDQ_SLOT_ACCEPTED;
    __threadfence_system();
}

static __device__ inline void
opal_common_cuda_cmdq_device_complete(opal_common_cuda_cmdq_slot_t *slot)
{
    slot->reject_reason = OPAL_COMMON_CUDA_CMDQ_REJECT_NONE;
    slot->status = OPAL_COMMON_CUDA_CMDQ_STATUS_COMPLETED;
    __threadfence_system();
    slot->state = OPAL_COMMON_CUDA_CMDQ_SLOT_COMPLETED;
    __threadfence_system();
}

static __device__ inline void
opal_common_cuda_cmdq_device_reject(opal_common_cuda_cmdq_slot_t *slot,
                                    opal_common_cuda_cmdq_reject_reason_t reason)
{
    slot->reject_reason = (uint32_t) reason;
    slot->status = OPAL_COMMON_CUDA_CMDQ_STATUS_REJECTED;
    __threadfence_system();
    slot->state = OPAL_COMMON_CUDA_CMDQ_SLOT_REJECTED;
    __threadfence_system();
}

static __device__ __forceinline__ int
opal_common_cuda_cmdq_device_datatype_is_pack(uint16_t handler)
{
    return OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_PARTIAL_BLOCKLEN == handler
           || OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_PREDEFINED_DATATYPE == handler
           || OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_CONTIGUOUS_LOOP == handler;
}

static __device__ __forceinline__ void
opal_common_cuda_cmdq_device_datatype_memcpy(uint8_t *dst, const uint8_t *src, uint64_t bytes)
{
    for (uint64_t offset = threadIdx.x; offset < bytes; offset += blockDim.x) {
        dst[offset] = src[offset];
    }
}

static __device__ __forceinline__ int
opal_common_cuda_cmdq_device_datatype_payload_size(uint16_t handler, uint64_t *payload_size)
{
    switch (handler) {
    case OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_PARTIAL_BLOCKLEN:
    case OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_UNPACK_PARTIAL_BLOCKLEN:
        *payload_size = sizeof(opal_common_cuda_cmdq_datatype_partial_blocklen_t);
        return 1;
    case OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_PREDEFINED_DATATYPE:
    case OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_UNPACK_PREDEFINED_DATATYPE:
        *payload_size = sizeof(opal_common_cuda_cmdq_datatype_predefined_datatype_t);
        return 1;
    case OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_CONTIGUOUS_LOOP:
    case OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_UNPACK_CONTIGUOUS_LOOP:
        *payload_size = sizeof(opal_common_cuda_cmdq_datatype_contiguous_loop_t);
        return 1;
    default:
        *payload_size = 0;
        return 0;
    }
}

static __device__ inline int
opal_common_cuda_cmdq_device_datatype_execute(
    uint16_t handler, const opal_common_cuda_cmdq_datatype_header_t *header, const void *payload,
    uint64_t *bytes_done, uint32_t *op_count_done)
{
    uint64_t total_bytes = 0;
    uint32_t op_count = OPAL_COMMON_CUDA_CMDQ_DATATYPE_OP_COUNT(header);
    uint32_t total_count = 0;
    int is_pack = opal_common_cuda_cmdq_device_datatype_is_pack(handler);

    if (!OPAL_COMMON_CUDA_CMDQ_DATATYPE_OP_COUNT_VALID(op_count)) {
        return 0;
    }

    switch (handler) {
    case OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_PARTIAL_BLOCKLEN:
    case OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_UNPACK_PARTIAL_BLOCKLEN: {
        const opal_common_cuda_cmdq_datatype_partial_blocklen_t *items =
            (const opal_common_cuda_cmdq_datatype_partial_blocklen_t *) payload;

        for (uint32_t item = 0; item < op_count; ++item) {
            uint8_t *memory = (uint8_t *) (uintptr_t) items[item].memory;
            uint8_t *packed = (uint8_t *) (uintptr_t) items[item].packed;
            uint8_t *dst = is_pack ? packed : memory;
            const uint8_t *src = is_pack ? memory : packed;

            opal_common_cuda_cmdq_device_datatype_memcpy(dst, src, items[item].do_now_bytes);
            total_bytes += items[item].do_now_bytes;
        }
        total_count = op_count;
        break;
    }
    case OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_PREDEFINED_DATATYPE:
    case OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_UNPACK_PREDEFINED_DATATYPE: {
        const opal_common_cuda_cmdq_datatype_predefined_datatype_t *items =
            (const opal_common_cuda_cmdq_datatype_predefined_datatype_t *) payload;

        for (uint32_t item = 0; item < op_count; ++item) {
            for (uint64_t block = 0; block < items[item].count; ++block) {
                uint8_t *memory = (uint8_t *) (uintptr_t) items[item].memory
                                  + (int64_t) block * items[item].extent;
                uint8_t *packed = (uint8_t *) (uintptr_t) items[item].packed
                                  + block * items[item].blocklen_bytes;
                uint8_t *dst = is_pack ? packed : memory;
                const uint8_t *src = is_pack ? memory : packed;

                opal_common_cuda_cmdq_device_datatype_memcpy(dst, src,
                                                             items[item].blocklen_bytes);
                total_bytes += items[item].blocklen_bytes;
            }
        }
        total_count = op_count;
        break;
    }
    case OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_CONTIGUOUS_LOOP:
    case OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_UNPACK_CONTIGUOUS_LOOP: {
        const opal_common_cuda_cmdq_datatype_contiguous_loop_t *items =
            (const opal_common_cuda_cmdq_datatype_contiguous_loop_t *) payload;

        for (uint32_t item = 0; item < op_count; ++item) {
            for (uint64_t loop = 0; loop < items[item].copy_loops; ++loop) {
                uint8_t *memory = (uint8_t *) (uintptr_t) items[item].memory
                                  + (int64_t) loop * items[item].loop_extent;
                uint8_t *packed = (uint8_t *) (uintptr_t) items[item].packed
                                  + loop * items[item].loop_size;
                uint8_t *dst = is_pack ? packed : memory;
                const uint8_t *src = is_pack ? memory : packed;

                opal_common_cuda_cmdq_device_datatype_memcpy(dst, src, items[item].loop_size);
                total_bytes += items[item].loop_size;
            }
        }
        total_count = op_count;
        break;
    }
    default:
        return 0;
    }

    *bytes_done = total_bytes;
    *op_count_done = total_count;
    return 1;
}
#endif

END_C_DECLS

#endif /* OPAL_MCA_COMMON_CUDA_CMDQ_H */
