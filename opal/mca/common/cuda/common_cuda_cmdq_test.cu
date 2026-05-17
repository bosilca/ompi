/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 * Standalone build from the source-tree root:
 *
 *   nvcc -DOPAL_COMMON_CUDA_CMDQ_STANDALONE -I. -x cu \
 *        opal/mca/common/cuda/common_cuda_cmdq.c \
 *        opal/mca/common/cuda/common_cuda_cmdq_test.cu \
 *        -o /tmp/common_cuda_cmdq_test -lcuda -lpthread
 */

#if !defined(OPAL_COMMON_CUDA_CMDQ_STANDALONE)
#    define OPAL_COMMON_CUDA_CMDQ_STANDALONE 1
#endif

#include "opal/mca/common/cuda/common_cuda_cmdq.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_OPAL_SUCCESS 0
#define TEST_OPAL_ERROR  -1

struct opal_accelerator_stream_t {
    void *stream;
};

typedef struct {
    int seen;
    opal_common_cuda_cmdq_datatype_fragment_done_t done[2];
} test_context_t;

static void check_cuda(cudaError_t error, const char *expr, int line)
{
    if (cudaSuccess != error) {
        fprintf(stderr, "%s:%d: CUDA failure in %s: %s\n", __FILE__, line, expr,
                cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __LINE__)

static void check_opal(int rc, const char *expr, int line)
{
    if (TEST_OPAL_SUCCESS != rc) {
        fprintf(stderr, "%s:%d: command queue failure in %s: %d\n", __FILE__, line, expr, rc);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_OPAL(expr) check_opal((expr), #expr, __LINE__)

static void fragment_done_cb(const opal_common_cuda_cmdq_am_desc_t *desc)
{
    test_context_t *ctx = (test_context_t *) desc->cbdata;

    if (sizeof(ctx->done[0]) != desc->header_len) {
        fprintf(stderr, "unexpected fragment response header length: %zu\n", desc->header_len);
        exit(EXIT_FAILURE);
    }
    if (2 <= ctx->seen) {
        fprintf(stderr, "too many fragment responses\n");
        exit(EXIT_FAILURE);
    }

    ctx->done[ctx->seen++] =
        *(const opal_common_cuda_cmdq_datatype_fragment_done_t *) desc->header;
}

__global__ static void test_datatype_cmdq_kernel(opal_common_cuda_cmdq_device_t *device)
{
    __shared__ opal_common_cuda_cmdq_slot_t *slot;
    __shared__ opal_common_cuda_cmdq_am_id_t am_id;
    __shared__ uint16_t handler;
    __shared__ int reject;

    if (0 == threadIdx.x) {
        device->state = OPAL_COMMON_CUDA_CMDQ_STATE_ENABLED;
        __threadfence_system();
    }
    __syncthreads();

    for (;;) {
        if (0 == threadIdx.x) {
            slot = opal_common_cuda_cmdq_device_wait(device);
            am_id = slot->am_id;
            reject = 0;
            handler = 0;

            opal_common_cuda_cmdq_device_accept(slot);

            if (OPAL_COMMON_CUDA_CMDQ_AM_ID_STOP != am_id) {
                uint64_t item_size = 0;
                const opal_common_cuda_cmdq_datatype_header_t *header =
                    (const opal_common_cuda_cmdq_datatype_header_t *) (device->fifo.base
                                                                       + slot->header_offset);

                if (am_id >= device->am_table_size || 0 == device->am_table[am_id].device_handler_id
                    || sizeof(*header) != slot->header_len || 0 != header->flags) {
                    reject = OPAL_COMMON_CUDA_CMDQ_REJECT_BAD_AM;
                } else {
                    uint32_t op_count = OPAL_COMMON_CUDA_CMDQ_DATATYPE_OP_COUNT(header);

                    handler = device->am_table[am_id].device_handler_id;
                    if (!opal_common_cuda_cmdq_device_datatype_payload_size(handler, &item_size)
                        || !OPAL_COMMON_CUDA_CMDQ_DATATYPE_OP_COUNT_VALID(op_count)
                        || slot->payload_len != op_count * item_size) {
                        reject = OPAL_COMMON_CUDA_CMDQ_REJECT_BAD_AM;
                    } else if ((slot->flags & OPAL_COMMON_CUDA_CMDQ_SEND_FLAG_RESPONSE_REQUIRED)
                               && slot->response_capacity
                                      < sizeof(opal_common_cuda_cmdq_datatype_fragment_done_t)) {
                        reject = OPAL_COMMON_CUDA_CMDQ_REJECT_TOO_LARGE;
                    }
                }
            }

            slot->am_id = OPAL_COMMON_CUDA_CMDQ_AM_ID_RETURN;
        }
        __syncthreads();

        if (OPAL_COMMON_CUDA_CMDQ_AM_ID_STOP == am_id) {
            if (0 == threadIdx.x) {
                opal_common_cuda_cmdq_device_complete(slot);
            }
            break;
        }

        if (reject) {
            if (0 == threadIdx.x) {
                opal_common_cuda_cmdq_device_reject(
                    slot, (opal_common_cuda_cmdq_reject_reason_t) reject);
            }
            __syncthreads();
            continue;
        }

        const opal_common_cuda_cmdq_datatype_header_t *header =
            (const opal_common_cuda_cmdq_datatype_header_t *) (device->fifo.base
                                                               + slot->header_offset);
        const void *payload = device->fifo.base + slot->payload_offset;
        uint64_t bytes_done = 0;
        uint32_t op_count_done = 0;

        if (!opal_common_cuda_cmdq_device_datatype_execute(handler, header, payload, &bytes_done,
                                                           &op_count_done)) {
            if (0 == threadIdx.x) {
                opal_common_cuda_cmdq_device_reject(slot,
                                                    OPAL_COMMON_CUDA_CMDQ_REJECT_BAD_AM);
            }
            __syncthreads();
            continue;
        }
        __syncthreads();

        if (0 == threadIdx.x) {
            if (slot->flags & OPAL_COMMON_CUDA_CMDQ_SEND_FLAG_RESPONSE_REQUIRED) {
                opal_common_cuda_cmdq_datatype_fragment_done_t *response =
                    (opal_common_cuda_cmdq_datatype_fragment_done_t *) (device->fifo.base
                                                                       + slot->response_offset);

                response->fragment_id = header->fragment_id;
                response->bytes_done = (0 != header->fragment_bytes) ? header->fragment_bytes
                                                                      : (uint32_t) bytes_done;
                response->op_count_done =
                    OPAL_COMMON_CUDA_CMDQ_DATATYPE_ENCODE_OP_COUNT(op_count_done);
                response->status = (uint8_t) OPAL_COMMON_CUDA_CMDQ_STATUS_COMPLETED;
                slot->response_header_len = sizeof(*response);
                slot->response_payload_len = 0;
                slot->am_id = device->am_table[am_id].response_am_id;
            }

            opal_common_cuda_cmdq_device_complete(slot);
        }
        __syncthreads();
    }
}

static int test_launch(opal_common_cuda_cmdq_device_t *device, opal_accelerator_stream_t *stream,
                       void *cbdata)
{
    cudaStream_t cuda_stream = 0;
    int threads = (NULL == cbdata) ? 128 : *(int *) cbdata;

    if (NULL != stream && NULL != stream->stream) {
        cuda_stream = *(cudaStream_t *) stream->stream;
    }

    test_datatype_cmdq_kernel<<<1, threads, 0, cuda_stream>>>(device);
    return (cudaSuccess == cudaPeekAtLastError()) ? TEST_OPAL_SUCCESS : TEST_OPAL_ERROR;
}

static void wait_enabled(opal_common_cuda_cmdq_t *queue)
{
    opal_common_cuda_cmdq_state_t state;

    for (int i = 0; i < 1000000; ++i) {
        CHECK_OPAL(opal_common_cuda_cmdq_progress(queue));
        CHECK_OPAL(opal_common_cuda_cmdq_get_state(queue, &state));
        if (OPAL_COMMON_CUDA_CMDQ_STATE_ENABLED == state) {
            return;
        }
    }

    fprintf(stderr, "queue did not become enabled\n");
    exit(EXIT_FAILURE);
}

static void wait_request(opal_common_cuda_cmdq_t *queue, opal_common_cuda_cmdq_request_t *request)
{
    opal_common_cuda_cmdq_status_t status;
    opal_common_cuda_cmdq_reject_reason_t reason;

    for (int i = 0; i < 1000000; ++i) {
        CHECK_OPAL(opal_common_cuda_cmdq_progress(queue));
        CHECK_OPAL(opal_common_cuda_cmdq_request_test(request, &status, &reason));
        if (OPAL_COMMON_CUDA_CMDQ_STATUS_COMPLETED == status) {
            return;
        }
        if (OPAL_COMMON_CUDA_CMDQ_STATUS_REJECTED == status) {
            fprintf(stderr, "request rejected: %d\n", (int) reason);
            exit(EXIT_FAILURE);
        }
    }

    fprintf(stderr, "request did not complete\n");
    exit(EXIT_FAILURE);
}

static void wait_disabled(opal_common_cuda_cmdq_t *queue)
{
    opal_common_cuda_cmdq_state_t state;

    for (int i = 0; i < 1000000; ++i) {
        CHECK_OPAL(opal_common_cuda_cmdq_progress(queue));
        CHECK_OPAL(opal_common_cuda_cmdq_get_state(queue, &state));
        if (OPAL_COMMON_CUDA_CMDQ_STATE_DISABLED == state) {
            return;
        }
    }

    fprintf(stderr, "queue did not become disabled\n");
    exit(EXIT_FAILURE);
}

static void register_datatype_am(opal_common_cuda_cmdq_t *queue,
                                 opal_common_cuda_cmdq_am_id_t am_id,
                                 uint16_t handler)
{
    opal_common_cuda_cmdq_am_attr_t attr;

    attr.flags = 0;
    attr.max_header = sizeof(opal_common_cuda_cmdq_datatype_header_t);
    attr.max_payload = 8 * sizeof(opal_common_cuda_cmdq_datatype_contiguous_loop_t);
    attr.device_handler_id = handler;
    attr.response_am_id = OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_FRAGMENT_DONE;

    CHECK_OPAL(opal_common_cuda_cmdq_am_register(queue, am_id, &attr, NULL, NULL));
}

static int check_range(const uint8_t *dst, const uint8_t *src, uint64_t bytes)
{
    for (uint64_t i = 0; i < bytes; ++i) {
        if (dst[i] != src[i]) {
            return (int) i + 1;
        }
    }

    return 0;
}

int main(void)
{
    enum {
        PACK_FRAGMENT_ID = 100,
        UNPACK_FRAGMENT_ID = 101,
    };
    const uint64_t memory_size = 2048;
    const uint64_t packed_size = 256;
    const uint64_t partial_bytes = 13;
    const uint64_t predefined_blocklen_bytes = 16;
    const uint64_t predefined_count = 4;
    const uint64_t predefined_extent = 40;
    const uint64_t contiguous_loop_size = 24;
    const uint64_t contiguous_copy_loops = 3;
    const uint64_t contiguous_loop_extent = 64;
    const uint64_t partial_memory_disp = 11;
    const uint64_t predefined_memory_disp = 128;
    const uint64_t contiguous_memory_disp = 512;
    const uint64_t partial_packed_disp = 0;
    const uint64_t predefined_packed_disp = partial_packed_disp + partial_bytes;
    const uint64_t contiguous_packed_disp =
        predefined_packed_disp + predefined_blocklen_bytes * predefined_count;
    const uint32_t fragment_bytes = (uint32_t) (partial_bytes
                                               + predefined_blocklen_bytes * predefined_count
                                               + contiguous_loop_size * contiguous_copy_loops);
    opal_common_cuda_cmdq_t *queue = NULL;
    opal_common_cuda_cmdq_request_t *requests[6] = {NULL};
    test_context_t context = {0};
    opal_common_cuda_cmdq_attr_t attr;
    opal_common_cuda_cmdq_am_attr_t fragment_done_attr;
    opal_common_cuda_cmdq_send_param_t response_param;
    int threads = 128;
    uint8_t *memory = NULL, *packed = NULL, *unpacked = NULL;

    CHECK_CUDA(cudaSetDevice(0));

    attr.device_id = 0;
    attr.fifo_size = 1u << 20;
    attr.max_inline = 64u * 1024u;
    attr.flags = 0;
    attr.stream = NULL;
    attr.launch = test_launch;
    attr.launch_cbdata = &threads;

    CHECK_OPAL(opal_common_cuda_cmdq_open(&attr, &queue));

    fragment_done_attr.flags = 0;
    fragment_done_attr.max_header = sizeof(opal_common_cuda_cmdq_datatype_fragment_done_t);
    fragment_done_attr.max_payload = 0;
    fragment_done_attr.device_handler_id = 0;
    fragment_done_attr.response_am_id = 0;
    CHECK_OPAL(opal_common_cuda_cmdq_am_register(
        queue, OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_FRAGMENT_DONE, &fragment_done_attr,
        fragment_done_cb, &context));

    register_datatype_am(queue, OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_PACK_PARTIAL_BLOCKLEN,
                         OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_PARTIAL_BLOCKLEN);
    register_datatype_am(queue, OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_PACK_PREDEFINED_DATATYPE,
                         OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_PREDEFINED_DATATYPE);
    register_datatype_am(queue, OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_PACK_CONTIGUOUS_LOOP,
                         OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_PACK_CONTIGUOUS_LOOP);
    register_datatype_am(queue, OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_UNPACK_PARTIAL_BLOCKLEN,
                         OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_UNPACK_PARTIAL_BLOCKLEN);
    register_datatype_am(queue, OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_UNPACK_PREDEFINED_DATATYPE,
                         OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_UNPACK_PREDEFINED_DATATYPE);
    register_datatype_am(queue, OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_UNPACK_CONTIGUOUS_LOOP,
                         OPAL_COMMON_CUDA_CMDQ_DATATYPE_HANDLER_UNPACK_CONTIGUOUS_LOOP);

    CHECK_OPAL(opal_common_cuda_cmdq_enable(queue));
    wait_enabled(queue);

    CHECK_CUDA(cudaMallocManaged((void **) &memory, memory_size));
    CHECK_CUDA(cudaMallocManaged((void **) &packed, packed_size));
    CHECK_CUDA(cudaMallocManaged((void **) &unpacked, memory_size));

    for (uint64_t i = 0; i < memory_size; ++i) {
        memory[i] = (uint8_t) ((i * 17u + 3u) & 0xff);
        unpacked[i] = 0;
    }
    for (uint64_t i = 0; i < packed_size; ++i) {
        packed[i] = 0;
    }

    opal_common_cuda_cmdq_datatype_header_t pack_partial_header;
    opal_common_cuda_cmdq_datatype_partial_blocklen_t pack_partial;
    opal_common_cuda_cmdq_datatype_header_t pack_predefined_header;
    opal_common_cuda_cmdq_datatype_predefined_datatype_t pack_predefined;
    opal_common_cuda_cmdq_datatype_header_t pack_contiguous_header;
    opal_common_cuda_cmdq_datatype_contiguous_loop_t pack_contiguous;
    opal_common_cuda_cmdq_datatype_header_t unpack_partial_header;
    opal_common_cuda_cmdq_datatype_partial_blocklen_t unpack_partial;
    opal_common_cuda_cmdq_datatype_header_t unpack_predefined_header;
    opal_common_cuda_cmdq_datatype_predefined_datatype_t unpack_predefined;
    opal_common_cuda_cmdq_datatype_header_t unpack_contiguous_header;
    opal_common_cuda_cmdq_datatype_contiguous_loop_t unpack_contiguous;

    pack_partial_header.fragment_id = PACK_FRAGMENT_ID;
    pack_partial_header.fragment_bytes = 0;
    pack_partial_header.op_count = OPAL_COMMON_CUDA_CMDQ_DATATYPE_ENCODE_OP_COUNT(1);
    pack_partial_header.flags = 0;
    pack_partial.memory = (uint64_t) (uintptr_t) (memory + partial_memory_disp);
    pack_partial.packed = (uint64_t) (uintptr_t) (packed + partial_packed_disp);
    pack_partial.do_now_bytes = partial_bytes;

    pack_predefined_header.fragment_id = PACK_FRAGMENT_ID;
    pack_predefined_header.fragment_bytes = 0;
    pack_predefined_header.op_count = OPAL_COMMON_CUDA_CMDQ_DATATYPE_ENCODE_OP_COUNT(1);
    pack_predefined_header.flags = 0;
    pack_predefined.memory = (uint64_t) (uintptr_t) (memory + predefined_memory_disp);
    pack_predefined.packed = (uint64_t) (uintptr_t) (packed + predefined_packed_disp);
    pack_predefined.blocklen_bytes = predefined_blocklen_bytes;
    pack_predefined.extent = predefined_extent;
    pack_predefined.count = predefined_count;

    pack_contiguous_header.fragment_id = PACK_FRAGMENT_ID;
    pack_contiguous_header.fragment_bytes = fragment_bytes;
    pack_contiguous_header.op_count = OPAL_COMMON_CUDA_CMDQ_DATATYPE_ENCODE_OP_COUNT(1);
    pack_contiguous_header.flags = 0;
    pack_contiguous.memory = (uint64_t) (uintptr_t) (memory + contiguous_memory_disp);
    pack_contiguous.packed = (uint64_t) (uintptr_t) (packed + contiguous_packed_disp);
    pack_contiguous.loop_size = contiguous_loop_size;
    pack_contiguous.loop_extent = contiguous_loop_extent;
    pack_contiguous.copy_loops = contiguous_copy_loops;

    unpack_partial_header.fragment_id = UNPACK_FRAGMENT_ID;
    unpack_partial_header.fragment_bytes = 0;
    unpack_partial_header.op_count = OPAL_COMMON_CUDA_CMDQ_DATATYPE_ENCODE_OP_COUNT(1);
    unpack_partial_header.flags = 0;
    unpack_partial.memory = (uint64_t) (uintptr_t) (unpacked + partial_memory_disp);
    unpack_partial.packed = (uint64_t) (uintptr_t) (packed + partial_packed_disp);
    unpack_partial.do_now_bytes = partial_bytes;

    unpack_predefined_header.fragment_id = UNPACK_FRAGMENT_ID;
    unpack_predefined_header.fragment_bytes = 0;
    unpack_predefined_header.op_count = OPAL_COMMON_CUDA_CMDQ_DATATYPE_ENCODE_OP_COUNT(1);
    unpack_predefined_header.flags = 0;
    unpack_predefined.memory = (uint64_t) (uintptr_t) (unpacked + predefined_memory_disp);
    unpack_predefined.packed = (uint64_t) (uintptr_t) (packed + predefined_packed_disp);
    unpack_predefined.blocklen_bytes = predefined_blocklen_bytes;
    unpack_predefined.extent = predefined_extent;
    unpack_predefined.count = predefined_count;

    unpack_contiguous_header.fragment_id = UNPACK_FRAGMENT_ID;
    unpack_contiguous_header.fragment_bytes = fragment_bytes;
    unpack_contiguous_header.op_count = OPAL_COMMON_CUDA_CMDQ_DATATYPE_ENCODE_OP_COUNT(1);
    unpack_contiguous_header.flags = 0;
    unpack_contiguous.memory = (uint64_t) (uintptr_t) (unpacked + contiguous_memory_disp);
    unpack_contiguous.packed = (uint64_t) (uintptr_t) (packed + contiguous_packed_disp);
    unpack_contiguous.loop_size = contiguous_loop_size;
    unpack_contiguous.loop_extent = contiguous_loop_extent;
    unpack_contiguous.copy_loops = contiguous_copy_loops;

    response_param.flags = OPAL_COMMON_CUDA_CMDQ_SEND_FLAG_RESPONSE_REQUIRED;
    response_param.cbfunc = NULL;
    response_param.cbdata = NULL;

    CHECK_OPAL(opal_common_cuda_cmdq_am_send_nbx(
        queue, OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_PACK_PARTIAL_BLOCKLEN, &pack_partial_header,
        sizeof(pack_partial_header), &pack_partial, sizeof(pack_partial), NULL, &requests[0]));
    CHECK_OPAL(opal_common_cuda_cmdq_am_send_nbx(
        queue, OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_PACK_PREDEFINED_DATATYPE,
        &pack_predefined_header, sizeof(pack_predefined_header), &pack_predefined,
        sizeof(pack_predefined), NULL, &requests[1]));
    CHECK_OPAL(opal_common_cuda_cmdq_am_send_nbx(
        queue, OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_PACK_CONTIGUOUS_LOOP, &pack_contiguous_header,
        sizeof(pack_contiguous_header), &pack_contiguous, sizeof(pack_contiguous),
        &response_param, &requests[2]));
    CHECK_OPAL(opal_common_cuda_cmdq_am_send_nbx(
        queue, OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_UNPACK_PARTIAL_BLOCKLEN,
        &unpack_partial_header, sizeof(unpack_partial_header), &unpack_partial,
        sizeof(unpack_partial), NULL, &requests[3]));
    CHECK_OPAL(opal_common_cuda_cmdq_am_send_nbx(
        queue, OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_UNPACK_PREDEFINED_DATATYPE,
        &unpack_predefined_header, sizeof(unpack_predefined_header), &unpack_predefined,
        sizeof(unpack_predefined), NULL, &requests[4]));
    CHECK_OPAL(opal_common_cuda_cmdq_am_send_nbx(
        queue, OPAL_COMMON_CUDA_CMDQ_DATATYPE_AM_ID_UNPACK_CONTIGUOUS_LOOP,
        &unpack_contiguous_header, sizeof(unpack_contiguous_header), &unpack_contiguous,
        sizeof(unpack_contiguous), &response_param, &requests[5]));

    for (int i = 0; i < 6; ++i) {
        wait_request(queue, requests[i]);
    }

    CHECK_OPAL(opal_common_cuda_cmdq_disable(queue));
    wait_disabled(queue);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (2 != context.seen || PACK_FRAGMENT_ID != context.done[0].fragment_id
        || UNPACK_FRAGMENT_ID != context.done[1].fragment_id
        || fragment_bytes != context.done[0].bytes_done
        || fragment_bytes != context.done[1].bytes_done) {
        fprintf(stderr,
                "unexpected fragment responses: seen=%d pack=(%llu,%llu) unpack=(%llu,%llu)\n",
                context.seen, (unsigned long long) context.done[0].fragment_id,
                (unsigned long long) context.done[0].bytes_done,
                (unsigned long long) context.done[1].fragment_id,
                (unsigned long long) context.done[1].bytes_done);
        return EXIT_FAILURE;
    }

    if (check_range(packed + partial_packed_disp, memory + partial_memory_disp, partial_bytes)) {
        fprintf(stderr, "pack_partial_blocklen validation failed\n");
        return EXIT_FAILURE;
    }
    for (uint64_t block = 0; block < predefined_count; ++block) {
        if (check_range(packed + predefined_packed_disp + block * predefined_blocklen_bytes,
                        memory + predefined_memory_disp + block * predefined_extent,
                        predefined_blocklen_bytes)) {
            fprintf(stderr, "pack_predefined_datatype validation failed at block %llu\n",
                    (unsigned long long) block);
            return EXIT_FAILURE;
        }
    }
    for (uint64_t loop = 0; loop < contiguous_copy_loops; ++loop) {
        if (check_range(packed + contiguous_packed_disp + loop * contiguous_loop_size,
                        memory + contiguous_memory_disp + loop * contiguous_loop_extent,
                        contiguous_loop_size)) {
            fprintf(stderr, "pack_contiguous_loop validation failed at loop %llu\n",
                    (unsigned long long) loop);
            return EXIT_FAILURE;
        }
    }

    if (check_range(unpacked + partial_memory_disp, memory + partial_memory_disp, partial_bytes)) {
        fprintf(stderr, "unpack_partial_blocklen validation failed\n");
        return EXIT_FAILURE;
    }
    for (uint64_t block = 0; block < predefined_count; ++block) {
        if (check_range(unpacked + predefined_memory_disp + block * predefined_extent,
                        memory + predefined_memory_disp + block * predefined_extent,
                        predefined_blocklen_bytes)) {
            fprintf(stderr, "unpack_predefined_datatype validation failed at block %llu\n",
                    (unsigned long long) block);
            return EXIT_FAILURE;
        }
    }
    for (uint64_t loop = 0; loop < contiguous_copy_loops; ++loop) {
        if (check_range(unpacked + contiguous_memory_disp + loop * contiguous_loop_extent,
                        memory + contiguous_memory_disp + loop * contiguous_loop_extent,
                        contiguous_loop_size)) {
            fprintf(stderr, "unpack_contiguous_loop validation failed at loop %llu\n",
                    (unsigned long long) loop);
            return EXIT_FAILURE;
        }
    }

    for (int i = 0; i < 6; ++i) {
        opal_common_cuda_cmdq_request_release(requests[i]);
    }

    CHECK_CUDA(cudaFree(memory));
    CHECK_CUDA(cudaFree(packed));
    CHECK_CUDA(cudaFree(unpacked));
    CHECK_OPAL(opal_common_cuda_cmdq_close(queue));

    printf("common_cuda_cmdq_test passed\n");
    return EXIT_SUCCESS;
}
