/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#if !defined(OPAL_COMMON_CUDA_CMDQ_STANDALONE)
#    include "opal_config.h"
#endif

#include "opal/mca/common/cuda/common_cuda_cmdq.h"

#include <cuda.h>

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(OPAL_COMMON_CUDA_CMDQ_STANDALONE)
#    define OPAL_SUCCESS             0
#    define OPAL_ERROR              -1
#    define OPAL_ERR_OUT_OF_RESOURCE -2
#    define OPAL_ERR_RESOURCE_BUSY   -4
#    define OPAL_ERR_BAD_PARAM       -5
#    define OPAL_ERR_NOT_FOUND      -13
#    define OPAL_ERR_NOT_AVAILABLE  -16
#    if !defined(MCA_ACCELERATOR_STREAM_DEFAULT)
#        define MCA_ACCELERATOR_STREAM_DEFAULT ((opal_accelerator_stream_t *) 0x00000002)
#    endif
struct opal_accelerator_stream_t {
    void *stream;
};
#else
#    include "opal/include/opal/constants.h"
#    include "opal/mca/accelerator/accelerator.h"
#endif

#define OPAL_COMMON_CUDA_CMDQ_AM_TABLE_SIZE 256
#define OPAL_COMMON_CUDA_CMDQ_ALIGN         8u

typedef struct {
    bool registered;
    opal_common_cuda_cmdq_am_attr_t attr;
    opal_common_cuda_cmdq_response_cb_fn_t cbfunc;
    void *cbdata;
} opal_common_cuda_cmdq_am_registration_t;

struct opal_common_cuda_cmdq_request_t {
    opal_common_cuda_cmdq_t *queue;
    opal_common_cuda_cmdq_slot_offset_t offset;
    opal_common_cuda_cmdq_slot_t *slot;
    opal_common_cuda_cmdq_status_t status;
    opal_common_cuda_cmdq_reject_reason_t reason;
    opal_common_cuda_cmdq_completion_fn_t cbfunc;
    void *cbdata;
    bool accepted_callback_done;
    bool terminal;
    bool reclaimed;
    bool auto_release;
    struct opal_common_cuda_cmdq_request_t *next_active;
};

struct opal_common_cuda_cmdq_t {
    opal_common_cuda_cmdq_attr_t attr;
    opal_common_cuda_cmdq_state_t state;
    opal_common_cuda_cmdq_device_t *device;
    uint8_t *fifo_storage;
    opal_common_cuda_cmdq_device_am_entry_t *am_table;
    opal_common_cuda_cmdq_am_registration_t registrations[OPAL_COMMON_CUDA_CMDQ_AM_TABLE_SIZE];
    pthread_mutex_t lock;
    CUcontext context;
    CUdevice cu_device;
    bool context_retained;
    CUevent kernel_done;
    bool kernel_done_created;
    bool kernel_done_recorded;
    uint32_t next_alloc;
    size_t active_count;
    opal_common_cuda_cmdq_request_t *active_head;
    opal_common_cuda_cmdq_request_t *active_tail;
    opal_common_cuda_cmdq_request_t *stop_request;
};

static inline uint32_t cmdq_align(uint32_t value)
{
    return (value + OPAL_COMMON_CUDA_CMDQ_ALIGN - 1u) & ~(OPAL_COMMON_CUDA_CMDQ_ALIGN - 1u);
}

static inline bool cmdq_align_size(size_t value, uint32_t *aligned)
{
    if (value > UINT32_MAX - (OPAL_COMMON_CUDA_CMDQ_ALIGN - 1u)) {
        return false;
    }

    *aligned = cmdq_align((uint32_t) value);
    return true;
}

static inline bool cmdq_reserved_am_id(opal_common_cuda_cmdq_am_id_t am_id)
{
    return OPAL_COMMON_CUDA_CMDQ_AM_ID_RETURN == am_id || OPAL_COMMON_CUDA_CMDQ_AM_ID_STOP == am_id;
}

static inline bool cmdq_response_requested(const opal_common_cuda_cmdq_am_registration_t *registration,
                                           const opal_common_cuda_cmdq_send_param_t *param)
{
    return (registration->attr.flags & OPAL_COMMON_CUDA_CMDQ_AM_FLAG_RESPONSE_REQUIRED)
           || (NULL != param
               && (param->flags & OPAL_COMMON_CUDA_CMDQ_SEND_FLAG_RESPONSE_REQUIRED));
}

static inline opal_common_cuda_cmdq_slot_t *cmdq_slot(opal_common_cuda_cmdq_t *queue,
                                                      opal_common_cuda_cmdq_slot_offset_t offset)
{
    return (opal_common_cuda_cmdq_slot_t *) (queue->device->fifo.base + offset);
}

static CUstream cmdq_cuda_stream(opal_common_cuda_cmdq_t *queue)
{
    opal_accelerator_stream_t *stream = queue->attr.stream;

    if (NULL == stream || MCA_ACCELERATOR_STREAM_DEFAULT == stream || NULL == stream->stream) {
        return (CUstream) 0;
    }

    return *(CUstream *) stream->stream;
}

static int cmdq_make_context_current(opal_common_cuda_cmdq_t *queue)
{
    CUresult result;

    if (NULL == queue->context) {
        return OPAL_SUCCESS;
    }

    result = cuCtxSetCurrent(queue->context);
    return CUDA_SUCCESS == result ? OPAL_SUCCESS : OPAL_ERROR;
}

static int cmdq_init_context(opal_common_cuda_cmdq_t *queue)
{
    CUcontext current;
    CUresult result;

    result = cuInit(0);
    if (CUDA_SUCCESS != result) {
        return OPAL_ERROR;
    }

    if (0 <= queue->attr.device_id) {
        result = cuDeviceGet(&queue->cu_device, queue->attr.device_id);
        if (CUDA_SUCCESS != result) {
            return OPAL_ERROR;
        }

        result = cuDevicePrimaryCtxRetain(&queue->context, queue->cu_device);
        if (CUDA_SUCCESS != result) {
            return OPAL_ERROR;
        }
        queue->context_retained = true;

        return cmdq_make_context_current(queue);
    }

    result = cuCtxGetCurrent(&current);
    if (CUDA_SUCCESS != result || NULL == current) {
        return OPAL_ERR_BAD_PARAM;
    }

    queue->context = current;
    return OPAL_SUCCESS;
}

static void cmdq_reset_fifo(opal_common_cuda_cmdq_t *queue)
{
    queue->device->fifo.head = OPAL_COMMON_CUDA_CMDQ_SLOT_NONE;
    queue->device->fifo.tail = OPAL_COMMON_CUDA_CMDQ_SLOT_NONE;
    queue->device->fifo.reclaim = OPAL_COMMON_CUDA_CMDQ_SLOT_NONE;
    queue->next_alloc = 0;
}

static void cmdq_append_active(opal_common_cuda_cmdq_t *queue,
                               opal_common_cuda_cmdq_request_t *request)
{
    request->next_active = NULL;

    if (NULL == queue->active_tail) {
        queue->active_head = request;
    } else {
        queue->active_tail->next_active = request;
    }

    queue->active_tail = request;
    queue->active_count++;
}

static void cmdq_remove_active(opal_common_cuda_cmdq_t *queue,
                               opal_common_cuda_cmdq_request_t *prev,
                               opal_common_cuda_cmdq_request_t *request)
{
    if (NULL == prev) {
        queue->active_head = request->next_active;
    } else {
        prev->next_active = request->next_active;
    }

    if (queue->active_tail == request) {
        queue->active_tail = prev;
    }

    request->next_active = NULL;
    queue->active_count--;
}

static void cmdq_publish(opal_common_cuda_cmdq_t *queue, opal_common_cuda_cmdq_slot_t *slot,
                         opal_common_cuda_cmdq_slot_offset_t offset)
{
    opal_common_cuda_cmdq_slot_offset_t prev;

    __atomic_thread_fence(__ATOMIC_RELEASE);
    slot->state = OPAL_COMMON_CUDA_CMDQ_SLOT_READY;
    __atomic_thread_fence(__ATOMIC_RELEASE);

    prev = __atomic_exchange_n((opal_common_cuda_cmdq_slot_offset_t *) &queue->device->fifo.tail,
                               offset, __ATOMIC_ACQ_REL);
    if (OPAL_COMMON_CUDA_CMDQ_SLOT_NONE != prev) {
        opal_common_cuda_cmdq_slot_t *prev_slot = cmdq_slot(queue, prev);
        prev_slot->next = offset;
    } else {
        queue->device->fifo.head = offset;
        queue->device->fifo.reclaim = offset;
    }

    __atomic_thread_fence(__ATOMIC_RELEASE);
}

static int cmdq_update_lifecycle(opal_common_cuda_cmdq_t *queue)
{
    if (OPAL_COMMON_CUDA_CMDQ_STATE_ENABLE_IN_PROGRESS == queue->state
        && OPAL_COMMON_CUDA_CMDQ_STATE_ENABLED == queue->device->state) {
        queue->state = OPAL_COMMON_CUDA_CMDQ_STATE_ENABLED;
    }

    if (OPAL_COMMON_CUDA_CMDQ_STATE_DISABLE_IN_PROGRESS == queue->state
        && NULL != queue->stop_request && queue->stop_request->terminal) {
        CUresult result;

        if (OPAL_SUCCESS != cmdq_make_context_current(queue)) {
            return OPAL_ERROR;
        }

        result = cuEventQuery(queue->kernel_done);
        if (CUDA_SUCCESS == result) {
            queue->state = OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED;
            queue->device->state = OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED;
            free(queue->stop_request);
            queue->stop_request = NULL;
            queue->kernel_done_recorded = false;
        } else if (CUDA_ERROR_NOT_READY != result) {
            return OPAL_ERROR;
        }
    }

    return OPAL_SUCCESS;
}

static int cmdq_send(opal_common_cuda_cmdq_t *queue, opal_common_cuda_cmdq_am_id_t am_id,
                     const void *header, size_t header_len, const void *payload,
                     size_t payload_len, const opal_common_cuda_cmdq_send_param_t *param,
                     opal_common_cuda_cmdq_request_t **request_out, bool internal)
{
    const opal_common_cuda_cmdq_am_registration_t *registration = NULL;
    opal_common_cuda_cmdq_request_t *request;
    opal_common_cuda_cmdq_slot_t *slot;
    uint32_t header_offset, payload_offset, response_offset;
    uint32_t aligned_slot, aligned_header, aligned_payload, response_capacity, record_bytes;
    uint64_t record_bytes64;
    opal_common_cuda_cmdq_slot_offset_t offset;

    if (NULL != request_out) {
        *request_out = NULL;
    }

    if (NULL == queue || (!internal && cmdq_reserved_am_id(am_id))
        || (NULL != param
            && 0 != (param->flags & ~OPAL_COMMON_CUDA_CMDQ_SEND_FLAG_RESPONSE_REQUIRED))) {
        return OPAL_ERR_BAD_PARAM;
    }

    pthread_mutex_lock(&queue->lock);

    if (!internal) {
        if (OPAL_COMMON_CUDA_CMDQ_STATE_ENABLED != queue->state) {
            pthread_mutex_unlock(&queue->lock);
            return OPAL_ERR_NOT_AVAILABLE;
        }

        registration = &queue->registrations[am_id];
        if (!registration->registered) {
            pthread_mutex_unlock(&queue->lock);
            return OPAL_ERR_NOT_FOUND;
        }

        if (header_len > registration->attr.max_header || payload_len > registration->attr.max_payload
            || header_len > UINT32_MAX || payload_len > UINT32_MAX
            || (0 != queue->attr.max_inline
                && (header_len > queue->attr.max_inline
                    || payload_len > queue->attr.max_inline - header_len))) {
            pthread_mutex_unlock(&queue->lock);
            return OPAL_ERR_BAD_PARAM;
        }
    }

    if (internal && OPAL_COMMON_CUDA_CMDQ_STATE_DISABLE_IN_PROGRESS != queue->state) {
        pthread_mutex_unlock(&queue->lock);
        return OPAL_ERR_BAD_PARAM;
    }

    response_capacity = 0;
    if (!internal && cmdq_response_requested(registration, param)) {
        const opal_common_cuda_cmdq_am_registration_t *response_registration;

        if (cmdq_reserved_am_id(registration->attr.response_am_id)) {
            pthread_mutex_unlock(&queue->lock);
            return OPAL_ERR_BAD_PARAM;
        }

        response_registration = &queue->registrations[registration->attr.response_am_id];
        if (!response_registration->registered) {
            pthread_mutex_unlock(&queue->lock);
            return OPAL_ERR_NOT_FOUND;
        }

        if (response_registration->attr.max_header > UINT32_MAX
            || response_registration->attr.max_payload
                   > UINT32_MAX - response_registration->attr.max_header) {
            pthread_mutex_unlock(&queue->lock);
            return OPAL_ERR_BAD_PARAM;
        }

        if (!cmdq_align_size(response_registration->attr.max_header
                                 + response_registration->attr.max_payload,
                             &response_capacity)) {
            pthread_mutex_unlock(&queue->lock);
            return OPAL_ERR_BAD_PARAM;
        }
    }

    aligned_slot = cmdq_align((uint32_t) sizeof(*slot));
    if (!cmdq_align_size(header_len, &aligned_header)
        || !cmdq_align_size(payload_len, &aligned_payload)) {
        pthread_mutex_unlock(&queue->lock);
        return OPAL_ERR_BAD_PARAM;
    }

    record_bytes64 = (uint64_t) aligned_slot + aligned_header + aligned_payload
                     + response_capacity;
    if (record_bytes64 > UINT32_MAX || record_bytes64 > queue->device->fifo.size) {
        pthread_mutex_unlock(&queue->lock);
        return OPAL_ERR_BAD_PARAM;
    }
    record_bytes = (uint32_t) record_bytes64;

    if ((uint64_t) queue->next_alloc + record_bytes > queue->device->fifo.size) {
        if (0 == queue->active_count) {
            cmdq_reset_fifo(queue);
        } else {
            pthread_mutex_unlock(&queue->lock);
            return OPAL_ERR_OUT_OF_RESOURCE;
        }
    }

    request = (opal_common_cuda_cmdq_request_t *) calloc(1, sizeof(*request));
    if (NULL == request) {
        pthread_mutex_unlock(&queue->lock);
        return OPAL_ERR_OUT_OF_RESOURCE;
    }

    offset = queue->next_alloc;
    queue->next_alloc += record_bytes;
    slot = cmdq_slot(queue, offset);
    memset(slot, 0, record_bytes);

    header_offset = offset + aligned_slot;
    payload_offset = header_offset + aligned_header;
    response_offset = payload_offset + aligned_payload;

    if (0 != header_len) {
        memcpy(queue->device->fifo.base + header_offset, header, header_len);
    }
    if (0 != payload_len) {
        memcpy(queue->device->fifo.base + payload_offset, payload, payload_len);
    }

    slot->next = OPAL_COMMON_CUDA_CMDQ_SLOT_NONE;
    slot->state = OPAL_COMMON_CUDA_CMDQ_SLOT_FREE;
    slot->status = OPAL_COMMON_CUDA_CMDQ_STATUS_QUEUED;
    slot->reject_reason = OPAL_COMMON_CUDA_CMDQ_REJECT_NONE;
    slot->flags = NULL == param ? 0 : param->flags;
    slot->record_bytes = record_bytes;
    slot->am_id = am_id;
    slot->header_offset = header_offset;
    slot->header_len = (uint32_t) header_len;
    slot->payload_offset = payload_offset;
    slot->payload_len = (uint32_t) payload_len;
    slot->response_offset = response_offset;
    slot->response_capacity = response_capacity;

    request->queue = queue;
    request->offset = offset;
    request->slot = slot;
    request->status = OPAL_COMMON_CUDA_CMDQ_STATUS_QUEUED;
    request->reason = OPAL_COMMON_CUDA_CMDQ_REJECT_NONE;
    request->cbfunc = NULL == param ? NULL : param->cbfunc;
    request->cbdata = NULL == param ? NULL : param->cbdata;
    request->auto_release = (NULL == request_out && !internal);

    cmdq_append_active(queue, request);
    cmdq_publish(queue, slot, offset);

    if (NULL != request_out) {
        *request_out = request;
    }

    pthread_mutex_unlock(&queue->lock);
    return OPAL_SUCCESS;
}

int opal_common_cuda_cmdq_open(const opal_common_cuda_cmdq_attr_t *attr,
                               opal_common_cuda_cmdq_t **queue_out)
{
    opal_common_cuda_cmdq_t *queue;
    CUdeviceptr ptr;
    CUresult result;

    if (NULL == attr || NULL == queue_out || 0 == attr->fifo_size || 0 != attr->flags
        || NULL == attr->launch) {
        return OPAL_ERR_BAD_PARAM;
    }

    *queue_out = NULL;

    queue = (opal_common_cuda_cmdq_t *) calloc(1, sizeof(*queue));
    if (NULL == queue) {
        return OPAL_ERR_OUT_OF_RESOURCE;
    }

    queue->attr = *attr;
    queue->state = OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED;
    pthread_mutex_init(&queue->lock, NULL);

    if (OPAL_SUCCESS != cmdq_init_context(queue)) {
        goto error_out;
    }

    result = cuMemAllocManaged(&ptr, sizeof(*queue->device), CU_MEM_ATTACH_GLOBAL);
    if (CUDA_SUCCESS != result) {
        goto error_out;
    }
    queue->device = (opal_common_cuda_cmdq_device_t *) (uintptr_t) ptr;

    result = cuMemAllocManaged(&ptr, attr->fifo_size, CU_MEM_ATTACH_GLOBAL);
    if (CUDA_SUCCESS != result) {
        goto error_out;
    }
    queue->fifo_storage = (uint8_t *) (uintptr_t) ptr;

    result = cuMemAllocManaged(&ptr, sizeof(*queue->am_table) * OPAL_COMMON_CUDA_CMDQ_AM_TABLE_SIZE,
                               CU_MEM_ATTACH_GLOBAL);
    if (CUDA_SUCCESS != result) {
        goto error_out;
    }
    queue->am_table = (opal_common_cuda_cmdq_device_am_entry_t *) (uintptr_t) ptr;

    result = cuEventCreate(&queue->kernel_done, CU_EVENT_DISABLE_TIMING);
    if (CUDA_SUCCESS != result) {
        goto error_out;
    }
    queue->kernel_done_created = true;

    memset(queue->device, 0, sizeof(*queue->device));
    memset(queue->fifo_storage, 0, attr->fifo_size);
    memset(queue->am_table, 0, sizeof(*queue->am_table) * OPAL_COMMON_CUDA_CMDQ_AM_TABLE_SIZE);

    queue->device->abi_version = OPAL_COMMON_CUDA_CMDQ_ABI_VERSION;
    queue->device->flags = attr->flags;
    queue->device->device_id = attr->device_id;
    queue->device->max_inline = attr->max_inline;
    queue->device->state = OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED;
    queue->device->fifo.base = queue->fifo_storage;
    queue->device->fifo.size = attr->fifo_size;
    queue->device->am_table = queue->am_table;
    queue->device->am_table_size = OPAL_COMMON_CUDA_CMDQ_AM_TABLE_SIZE;
    cmdq_reset_fifo(queue);

    result = cuCtxSynchronize();
    if (CUDA_SUCCESS != result) {
        goto error_out;
    }

    *queue_out = queue;
    return OPAL_SUCCESS;

error_out:
    opal_common_cuda_cmdq_close(queue);
    return OPAL_ERROR;
}

int opal_common_cuda_cmdq_close(opal_common_cuda_cmdq_t *queue)
{
    if (NULL == queue) {
        return OPAL_ERR_BAD_PARAM;
    }

    if (OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED != queue->state || 0 != queue->active_count) {
        return OPAL_ERR_RESOURCE_BUSY;
    }

    (void) cmdq_make_context_current(queue);

    if (queue->kernel_done_created) {
        cuEventDestroy(queue->kernel_done);
    }
    if (NULL != queue->am_table) {
        cuMemFree((CUdeviceptr) (uintptr_t) queue->am_table);
    }
    if (NULL != queue->fifo_storage) {
        cuMemFree((CUdeviceptr) (uintptr_t) queue->fifo_storage);
    }
    if (NULL != queue->device) {
        cuMemFree((CUdeviceptr) (uintptr_t) queue->device);
    }
    if (queue->context_retained) {
        cuDevicePrimaryCtxRelease(queue->cu_device);
    }

    pthread_mutex_destroy(&queue->lock);
    free(queue);
    return OPAL_SUCCESS;
}

int opal_common_cuda_cmdq_enable(opal_common_cuda_cmdq_t *queue)
{
    CUstream stream;
    CUresult result;
    int rc;

    if (NULL == queue) {
        return OPAL_ERR_BAD_PARAM;
    }

    pthread_mutex_lock(&queue->lock);
    if (OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED != queue->state) {
        pthread_mutex_unlock(&queue->lock);
        return OPAL_ERR_RESOURCE_BUSY;
    }

    queue->state = OPAL_COMMON_CUDA_CMDQ_STATE_ENABLE_IN_PROGRESS;
    queue->device->state = OPAL_COMMON_CUDA_CMDQ_STATE_ENABLE_IN_PROGRESS;
    pthread_mutex_unlock(&queue->lock);

    if (OPAL_SUCCESS != cmdq_make_context_current(queue)) {
        pthread_mutex_lock(&queue->lock);
        queue->state = OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED;
        queue->device->state = OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED;
        pthread_mutex_unlock(&queue->lock);
        return OPAL_ERROR;
    }

    stream = cmdq_cuda_stream(queue);
    rc = queue->attr.launch(queue->device, queue->attr.stream, queue->attr.launch_cbdata);
    if (OPAL_SUCCESS != rc) {
        pthread_mutex_lock(&queue->lock);
        queue->state = OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED;
        queue->device->state = OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED;
        pthread_mutex_unlock(&queue->lock);
        return rc;
    }

    result = cuEventRecord(queue->kernel_done, stream);
    if (CUDA_SUCCESS != result) {
        pthread_mutex_lock(&queue->lock);
        queue->state = OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED;
        queue->device->state = OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED;
        pthread_mutex_unlock(&queue->lock);
        return OPAL_ERROR;
    }

    queue->kernel_done_recorded = true;
    return OPAL_SUCCESS;
}

int opal_common_cuda_cmdq_disable(opal_common_cuda_cmdq_t *queue)
{
    int rc;

    if (NULL == queue) {
        return OPAL_ERR_BAD_PARAM;
    }

    pthread_mutex_lock(&queue->lock);
    if (OPAL_COMMON_CUDA_CMDQ_STATE_ENABLED != queue->state) {
        pthread_mutex_unlock(&queue->lock);
        return OPAL_ERR_RESOURCE_BUSY;
    }

    queue->state = OPAL_COMMON_CUDA_CMDQ_STATE_DISABLE_IN_PROGRESS;
    queue->device->state = OPAL_COMMON_CUDA_CMDQ_STATE_DISABLE_IN_PROGRESS;
    pthread_mutex_unlock(&queue->lock);

    rc = cmdq_send(queue, OPAL_COMMON_CUDA_CMDQ_AM_ID_STOP, NULL, 0, NULL, 0, NULL,
                   &queue->stop_request, true);
    if (OPAL_SUCCESS != rc) {
        pthread_mutex_lock(&queue->lock);
        queue->state = OPAL_COMMON_CUDA_CMDQ_STATE_ENABLED;
        queue->device->state = OPAL_COMMON_CUDA_CMDQ_STATE_ENABLED;
        pthread_mutex_unlock(&queue->lock);
    }

    return rc;
}

int opal_common_cuda_cmdq_get_state(opal_common_cuda_cmdq_t *queue,
                                    opal_common_cuda_cmdq_state_t *state)
{
    int rc;

    if (NULL == queue || NULL == state) {
        return OPAL_ERR_BAD_PARAM;
    }

    pthread_mutex_lock(&queue->lock);
    rc = cmdq_update_lifecycle(queue);
    *state = queue->state;
    pthread_mutex_unlock(&queue->lock);

    return rc;
}

int opal_common_cuda_cmdq_get_device_handle(opal_common_cuda_cmdq_t *queue,
                                            opal_common_cuda_cmdq_device_t **device)
{
    if (NULL == queue || NULL == device) {
        return OPAL_ERR_BAD_PARAM;
    }

    *device = queue->device;
    return OPAL_SUCCESS;
}

int opal_common_cuda_cmdq_am_register(opal_common_cuda_cmdq_t *queue,
                                      opal_common_cuda_cmdq_am_id_t am_id,
                                      const opal_common_cuda_cmdq_am_attr_t *attr,
                                      opal_common_cuda_cmdq_response_cb_fn_t cbfunc,
                                      void *cbdata)
{
    if (NULL == queue || NULL == attr || cmdq_reserved_am_id(am_id)
        || OPAL_COMMON_CUDA_CMDQ_DEVICE_HANDLER_STOP == attr->device_handler_id
        || 0 != (attr->flags & ~OPAL_COMMON_CUDA_CMDQ_AM_FLAG_RESPONSE_REQUIRED)
        || attr->max_header > UINT32_MAX || attr->max_payload > UINT32_MAX
        || ((attr->flags & OPAL_COMMON_CUDA_CMDQ_AM_FLAG_RESPONSE_REQUIRED)
            && cmdq_reserved_am_id(attr->response_am_id))) {
        return OPAL_ERR_BAD_PARAM;
    }

    pthread_mutex_lock(&queue->lock);
    if (OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED != queue->state) {
        pthread_mutex_unlock(&queue->lock);
        return OPAL_ERR_RESOURCE_BUSY;
    }

    queue->registrations[am_id].registered = true;
    queue->registrations[am_id].attr = *attr;
    queue->registrations[am_id].cbfunc = cbfunc;
    queue->registrations[am_id].cbdata = cbdata;

    queue->am_table[am_id].device_handler_id = attr->device_handler_id;
    queue->am_table[am_id].response_am_id = attr->response_am_id;
    queue->am_table[am_id].flags = attr->flags;
    queue->am_table[am_id].max_header = attr->max_header;
    queue->am_table[am_id].max_payload = attr->max_payload;
    __atomic_thread_fence(__ATOMIC_RELEASE);

    pthread_mutex_unlock(&queue->lock);
    return OPAL_SUCCESS;
}

int opal_common_cuda_cmdq_am_send_nbx(opal_common_cuda_cmdq_t *queue,
                                      opal_common_cuda_cmdq_am_id_t am_id, const void *header,
                                      size_t header_len, const void *payload, size_t payload_len,
                                      const opal_common_cuda_cmdq_send_param_t *param,
                                      opal_common_cuda_cmdq_request_t **request)
{
    return cmdq_send(queue, am_id, header, header_len, payload, payload_len, param, request, false);
}

int opal_common_cuda_cmdq_progress(opal_common_cuda_cmdq_t *queue)
{
    opal_common_cuda_cmdq_request_t *request, *prev = NULL;
    int progressed = 0;

    if (NULL == queue) {
        return OPAL_ERR_BAD_PARAM;
    }

    pthread_mutex_lock(&queue->lock);

    request = queue->active_head;
    while (NULL != request) {
        opal_common_cuda_cmdq_request_t *next = request->next_active;
        opal_common_cuda_cmdq_slot_t *slot = request->slot;
        uint32_t slot_state = __atomic_load_n((uint32_t *) &slot->state, __ATOMIC_ACQUIRE);

        if (OPAL_COMMON_CUDA_CMDQ_SLOT_ACCEPTED == slot_state
            && OPAL_COMMON_CUDA_CMDQ_STATUS_ACCEPTED != request->status) {
            request->status = OPAL_COMMON_CUDA_CMDQ_STATUS_ACCEPTED;
            if (NULL != request->cbfunc && !request->accepted_callback_done) {
                request->accepted_callback_done = true;
                request->cbfunc(request, request->status, request->reason, request->cbdata);
            }
        }

        if (OPAL_COMMON_CUDA_CMDQ_SLOT_COMPLETED == slot_state
            || OPAL_COMMON_CUDA_CMDQ_SLOT_REJECTED == slot_state) {
            opal_common_cuda_cmdq_am_id_t returned_am = slot->am_id;
            bool response_ok;

            request->status = (opal_common_cuda_cmdq_status_t) slot->status;
            request->reason = (opal_common_cuda_cmdq_reject_reason_t) slot->reject_reason;
            response_ok = slot->response_header_len <= slot->response_capacity
                          && slot->response_payload_len
                                 <= slot->response_capacity - slot->response_header_len;
            if (!response_ok) {
                request->status = OPAL_COMMON_CUDA_CMDQ_STATUS_REJECTED;
                request->reason = OPAL_COMMON_CUDA_CMDQ_REJECT_BACKEND_ERROR;
            }
            request->terminal = true;

            if (OPAL_COMMON_CUDA_CMDQ_STATUS_COMPLETED == request->status
                && response_ok && !cmdq_reserved_am_id(returned_am)
                && queue->registrations[returned_am].registered
                && NULL != queue->registrations[returned_am].cbfunc) {
                const uint8_t *response = queue->device->fifo.base + slot->response_offset;
                opal_common_cuda_cmdq_am_desc_t desc;

                desc.queue = queue;
                desc.am_id = returned_am;
                desc.header = response;
                desc.header_len = slot->response_header_len;
                desc.payload = response + slot->response_header_len;
                desc.payload_len = slot->response_payload_len;
                desc.cbdata = queue->registrations[returned_am].cbdata;

                queue->registrations[returned_am].cbfunc(&desc);
            }

            if (NULL != request->cbfunc) {
                request->cbfunc(request, request->status, request->reason, request->cbdata);
            }

            request->reclaimed = true;
            queue->device->fifo.reclaim = slot->next;
            cmdq_remove_active(queue, prev, request);
            progressed++;
            if (request->auto_release) {
                free(request);
            }
        } else {
            prev = request;
        }

        request = next;
    }

    if (0 == queue->active_count) {
        cmdq_reset_fifo(queue);
    }

    if (OPAL_SUCCESS != cmdq_update_lifecycle(queue)) {
        pthread_mutex_unlock(&queue->lock);
        return OPAL_ERROR;
    }

    pthread_mutex_unlock(&queue->lock);
    return progressed;
}

int opal_common_cuda_cmdq_request_test(opal_common_cuda_cmdq_request_t *request,
                                       opal_common_cuda_cmdq_status_t *status,
                                       opal_common_cuda_cmdq_reject_reason_t *reason)
{
    if (NULL == request || NULL == status || NULL == reason) {
        return OPAL_ERR_BAD_PARAM;
    }

    pthread_mutex_lock(&request->queue->lock);
    *status = request->status;
    *reason = request->reason;
    pthread_mutex_unlock(&request->queue->lock);
    return OPAL_SUCCESS;
}

void opal_common_cuda_cmdq_request_release(opal_common_cuda_cmdq_request_t *request)
{
    if (NULL == request || !request->terminal) {
        return;
    }

    free(request);
}
