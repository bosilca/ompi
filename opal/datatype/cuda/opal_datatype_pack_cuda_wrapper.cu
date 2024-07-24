#include "opal/datatype/opal_convertor_internal.h"
#include "opal/datatype/opal_datatype_internal.h"
#include "opal/util/output.h"

#include "opal_datatype_cuda_internal.cuh"
#include "opal_datatype_cuda.cuh"

#include <stdio.h>
#include <assert.h>

int32_t opal_datatype_cuda_generic_simple_pack_function_iov( opal_convertor_t* pConvertor,
                                                             struct iovec* iov,
                                                             uint32_t* out_size,
                                                             size_t* max_data )
{
    size_t buffer_size;
    unsigned char *destination;
    size_t total_packed;
    uint8_t transfer_required, free_required;
    cudaStream_t working_stream = NULL; 
    cudaError_t cuda_err;

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end, start_total, end_total;
    long total_time, move_time;
#endif

    if ((iov[0].iov_base == NULL) || opal_datatype_cuda_is_gpu_buffer(iov[0].iov_base)) {
        assert (iov[0].iov_len != 0);
        buffer_size = iov[0].iov_len;
        
        if (iov[0].iov_base == NULL) {
            iov[0].iov_base = (unsigned char *)opal_datatype_cuda_malloc_gpu_buffer(buffer_size, 0);
            destination = (unsigned char *)iov[0].iov_base;
            pConvertor->gpu_buffer_ptr = destination;
            pConvertor->gpu_buffer_size = buffer_size;
            free_required = 1;
        } else {
            destination = (unsigned char *)iov[0].iov_base;
            free_required = 0;
        }
        transfer_required = 0;
    } else {
        buffer_size = iov[0].iov_len;
        if (OPAL_DATATYPE_USE_ZEROCPY) {
            pConvertor->gpu_buffer_ptr = NULL;
            transfer_required = 0;
            free_required = 0;
            cuda_err = cudaHostGetDevicePointer((void **)&destination, (void *)iov[0].iov_base, 0);
            if (cuda_err != cudaSuccess) {
                OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "Zero copy is not supported\n"));
                return 0;
            }
        } else {
            if (pConvertor->gpu_buffer_ptr == NULL) {
                pConvertor->gpu_buffer_ptr = (unsigned char*)opal_datatype_cuda_malloc_gpu_buffer(buffer_size, 0);
                pConvertor->gpu_buffer_size = buffer_size;
            }
            transfer_required = 1;
            free_required = 1;
            destination = pConvertor->gpu_buffer_ptr + pConvertor->pipeline_size * pConvertor->pipeline_seq;
        }
    }   

    total_packed = 0;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start_total);
#endif
    
    /* start pack */
    if (cuda_iov_cache_enabled) {
        opal_datatype_cuda_generic_simple_pack_function_iov_cached(pConvertor, destination, buffer_size, &total_packed);
    } else {
        opal_datatype_cuda_generic_simple_pack_function_iov_non_cached(pConvertor, destination, buffer_size, &total_packed);
    }

    pConvertor->bConverted += total_packed;
    OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "Pack total packed %ld\n", total_packed));
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    if (transfer_required) {
        if (pConvertor->stream == NULL) {
            ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
            working_stream = cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id];
        } else {
            working_stream = (cudaStream_t)pConvertor->stream;
        }
        cuda_err = cudaMemcpyAsync(iov[0].iov_base, destination, total_packed, cudaMemcpyDeviceToHost, working_stream);
        CUDA_ERROR_CHECK(cuda_err);
        if (!(pConvertor->flags & CONVERTOR_ACCELERATOR_ASYNC)) {
            cuda_err = cudaStreamSynchronize(working_stream);
            CUDA_ERROR_CHECK(cuda_err);
        }
    } 
#if defined(OPAL_DATATYPE_CUDA_TIMING) 
    GET_TIME( end );
    move_time = ELAPSED_TIME( start, end );
    OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "[Timing]: DtoH memcpy in %ld microsec, transfer required %d, pipeline_size %lu, pipeline_seq %lu\n", move_time, transfer_required, pConvertor->pipeline_size, pConvertor->pipeline_seq ));
#endif

    iov[0].iov_len = total_packed;
    *max_data = total_packed;
    *out_size = 1;
    
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    GET_TIME( end_total );
    total_time = ELAPSED_TIME( start_total, end_total );
    OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "[Timing]: total packing in %ld microsec, kernel %ld microsec\n", total_time, total_time-move_time ));
#endif
    
    if( pConvertor->bConverted == pConvertor->local_size ) {
        pConvertor->flags |= CONVERTOR_COMPLETED;
        if (pConvertor->gpu_buffer_ptr != NULL && free_required && !(pConvertor->flags & CONVERTOR_ACCELERATOR_ASYNC)) {
           opal_datatype_cuda_free_gpu_buffer(pConvertor->gpu_buffer_ptr, 0);
           pConvertor->gpu_buffer_ptr = NULL;
        }
        return 1;
    }        
    return 0; 
}

int32_t opal_datatype_cuda_generic_simple_pack_function_iov_non_cached( opal_convertor_t* pConvertor, unsigned char *destination, size_t buffer_size, size_t *total_packed)
{
    uint32_t nb_blocks, thread_per_block, nb_blocks_used;
    unsigned char *destination_base, *source_base;
    uint8_t buffer_isfull = 0;
    cudaError_t cuda_err;
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    ddt_cuda_iov_dist_cached_t* cuda_iov_dist_h_current;
    ddt_cuda_iov_dist_cached_t* cuda_iov_dist_d_current;
    ddt_cuda_iov_pipeline_block_non_cached_t *cuda_iov_pipeline_block_non_cached;
    cudaStream_t cuda_stream_iov = NULL;
    const struct iovec *ddt_iov = NULL;
    uint32_t ddt_iov_count = 0;
    size_t contig_disp = 0;
    uint32_t ddt_iov_start_pos, ddt_iov_end_pos, current_ddt_iov_pos;
    OPAL_PTRDIFF_TYPE ddt_extent;

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end;
    long total_time;
#endif

    OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "Pack using IOV non cached, convertor %p, GPU base %p, pack to buffer %p\n", pConvertor, pConvertor->pBaseBuf, destination));
    
    opal_convertor_raw_cached( pConvertor, &ddt_iov, &ddt_iov_count);
    if (ddt_iov == NULL) {
        OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "Can not get ddt iov\n"));
        return OPAL_ERROR;
    }
    
    thread_per_block = CUDA_WARP_SIZE * 5;
    nb_blocks = 256;
    opal_datatype_type_extent(pConvertor->pDesc, &ddt_extent);
    source_base = (unsigned char*)pConvertor->pBaseBuf + pConvertor->current_count * ddt_extent; 
    destination_base = destination;
    
    while( pConvertor->current_count < pConvertor->count && !buffer_isfull) {
        
        nb_blocks_used = 0;
        ddt_iov_start_pos = pConvertor->current_iov_pos;
        ddt_iov_end_pos = ddt_iov_start_pos + IOV_PIPELINE_SIZE;
        if (ddt_iov_end_pos > ddt_iov_count) {
            ddt_iov_end_pos = ddt_iov_count;
        }
        cuda_iov_pipeline_block_non_cached = current_cuda_device->cuda_iov_pipeline_block_non_cached[current_cuda_device->cuda_iov_pipeline_block_non_cached_first_avail];
        if (pConvertor->stream == NULL) {
            cuda_iov_pipeline_block_non_cached->cuda_stream = cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id];
        } else {
            cuda_iov_pipeline_block_non_cached->cuda_stream = (cudaStream_t)pConvertor->stream;
        }
        cuda_iov_dist_h_current = cuda_iov_pipeline_block_non_cached->cuda_iov_dist_non_cached_h;
        cuda_iov_dist_d_current = cuda_iov_pipeline_block_non_cached->cuda_iov_dist_non_cached_d;
        cuda_stream_iov = cuda_iov_pipeline_block_non_cached->cuda_stream;
        cuda_err = cudaEventSynchronize(cuda_iov_pipeline_block_non_cached->cuda_event);
        CUDA_ERROR_CHECK(cuda_err);

#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif

        buffer_isfull = opal_datatype_cuda_iov_to_cuda_iov(pConvertor, ddt_iov, cuda_iov_dist_h_current, ddt_iov_start_pos, ddt_iov_end_pos, &buffer_size, &nb_blocks_used, total_packed, &contig_disp, &current_ddt_iov_pos);

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "[Timing]: Pack src %p to dest %p, iov is prepared in %ld microsec, kernel submitted to CUDA stream %d, nb_blocks %d\n", source_base, destination_base, total_time,  cuda_streams->current_stream_id, nb_blocks_used));
#endif

        cudaMemcpyAsync(cuda_iov_dist_d_current, cuda_iov_dist_h_current, sizeof(ddt_cuda_iov_dist_cached_t)*(nb_blocks_used+1), cudaMemcpyHostToDevice, cuda_stream_iov);
        opal_generic_simple_pack_cuda_iov_cached_kernel<<<nb_blocks, thread_per_block, 0, cuda_stream_iov>>>(cuda_iov_dist_d_current, 0, nb_blocks_used, 0, 0, nb_blocks_used, source_base, destination_base);
        cuda_err = cudaEventRecord(cuda_iov_pipeline_block_non_cached->cuda_event, cuda_stream_iov);
        CUDA_ERROR_CHECK(cuda_err);
        current_cuda_device->cuda_iov_pipeline_block_non_cached_first_avail ++;
        if (current_cuda_device->cuda_iov_pipeline_block_non_cached_first_avail >= NB_PIPELINE_NON_CACHED_BLOCKS) {
            current_cuda_device->cuda_iov_pipeline_block_non_cached_first_avail = 0;
        }
        destination_base += contig_disp;
        
        if (!buffer_isfull) {
            pConvertor->current_iov_pos = current_ddt_iov_pos;
            if (current_ddt_iov_pos == ddt_iov_count) {
                pConvertor->current_count ++;
                pConvertor->current_iov_pos = 0;
                source_base += ddt_extent;
            }
        }
        
    }
        
    return OPAL_SUCCESS;
}

int32_t opal_datatype_cuda_generic_simple_pack_function_iov_cached( opal_convertor_t* pConvertor, unsigned char *destination, size_t buffer_size, size_t *total_packed)
{
    uint32_t i;
    uint32_t nb_blocks, thread_per_block, nb_blocks_used;
    unsigned char *destination_base, *source_base;
    uint8_t buffer_isfull = 0;
    ddt_cuda_stream_t *cuda_streams = current_cuda_device->cuda_streams;
    cudaStream_t cuda_stream_iov = NULL;
    uint32_t cuda_iov_start_pos, cuda_iov_end_pos;
    ddt_cuda_iov_total_cached_t* cached_cuda_iov = NULL;
    ddt_cuda_iov_dist_cached_t* cached_cuda_iov_dist_d = NULL;
    uint32_t *cached_cuda_iov_nb_bytes_list_h = NULL;
    uint32_t cached_cuda_iov_count = 0;
    opal_datatype_count_t convertor_current_count;
    OPAL_PTRDIFF_TYPE ddt_extent;

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    TIMER_DATA_TYPE start, end;
    long total_time;
#endif
    
    OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "Pack using IOV cached, convertor %p, GPU base %p, pack to buffer %p\n", pConvertor, pConvertor->pBaseBuf, destination));

    destination_base = destination;
    thread_per_block = CUDA_WARP_SIZE * 8;
    nb_blocks = 64;
    source_base = (unsigned char*)pConvertor->pBaseBuf; 
    
    /* cuda iov is not cached, start to cache iov */
    if(opal_datatype_cuda_cuda_iov_is_cached(pConvertor) == 0) {
#if defined(OPAL_DATATYPE_CUDA_TIMING)
        GET_TIME(start);
#endif
        if (opal_datatype_cuda_cache_cuda_iov(pConvertor, &nb_blocks_used) == OPAL_SUCCESS) {
            OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "Pack cuda iov is cached, count %d\n", nb_blocks_used));
        } else {
            OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "Pack cache cuda iov is failed\n"));
            return OPAL_ERROR;
        }
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
        GET_TIME( end );
        total_time = ELAPSED_TIME( start, end );
        OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "[Timing]: Pack cuda iov is cached in %ld microsec, nb_blocks %d\n", total_time, nb_blocks_used));
#endif
    }
    
    /* now we use cached cuda iov */
    opal_datatype_cuda_get_cached_cuda_iov(pConvertor, &cached_cuda_iov);
    cached_cuda_iov_dist_d = cached_cuda_iov->cuda_iov_dist_d;
    assert(cached_cuda_iov_dist_d != NULL);
    cached_cuda_iov_nb_bytes_list_h = cached_cuda_iov->nb_bytes_h;
    assert(cached_cuda_iov_nb_bytes_list_h != NULL);
    
    cached_cuda_iov_count = cached_cuda_iov->cuda_iov_count;
    cuda_iov_start_pos = pConvertor->current_cuda_iov_pos;
    cuda_iov_end_pos = cached_cuda_iov_count;
    nb_blocks_used = 0;
    if (pConvertor->stream == NULL) {
        cuda_stream_iov = cuda_streams->ddt_cuda_stream[cuda_streams->current_stream_id];
    } else {
        cuda_stream_iov = (cudaStream_t)pConvertor->stream;
    }
    convertor_current_count = pConvertor->current_count;
   
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif
    while( pConvertor->current_count < pConvertor->count && !buffer_isfull) {
        for (i = cuda_iov_start_pos; i < cuda_iov_end_pos && !buffer_isfull; i++) {
            if (buffer_size >= cached_cuda_iov_nb_bytes_list_h[i]) {
                *total_packed += cached_cuda_iov_nb_bytes_list_h[i];
                buffer_size -= cached_cuda_iov_nb_bytes_list_h[i];
                nb_blocks_used++;
            } else {
                buffer_isfull = 1;
                break;
            }
        }
        if (!buffer_isfull) {
            pConvertor->current_count ++;
            cuda_iov_start_pos = 0;
            cuda_iov_end_pos = cached_cuda_iov->cuda_iov_count;
        }
    }
#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "[Timing]: Pack to dest %p, cached cuda iov is prepared in %ld microsec, kernel submitted to CUDA stream %d, nb_blocks %d\n", destination_base, total_time,  cuda_streams->current_stream_id, nb_blocks_used));
#endif
    opal_datatype_type_extent(pConvertor->pDesc, &ddt_extent);
    OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "Pack kernel launched src_base %p, dst_base %p, nb_blocks %d, extent %ld\n", source_base, destination_base, nb_blocks_used, ddt_extent));
#if defined(OPAL_DATATYPE_CUDA_TIMING)
    GET_TIME(start);
#endif    
    opal_generic_simple_pack_cuda_iov_cached_kernel<<<nb_blocks, thread_per_block, 0, cuda_stream_iov>>>(cached_cuda_iov_dist_d, pConvertor->current_cuda_iov_pos, cached_cuda_iov_count, ddt_extent, convertor_current_count, nb_blocks_used, source_base, destination_base);
    pConvertor->current_cuda_iov_pos += nb_blocks_used;
    pConvertor->current_cuda_iov_pos = pConvertor->current_cuda_iov_pos % cached_cuda_iov->cuda_iov_count;

#if defined(OPAL_DATATYPE_CUDA_TIMING)    
    GET_TIME( end );
    total_time = ELAPSED_TIME( start, end );
    OPAL_OUTPUT_VERBOSE((2, opal_datatype_cuda_output, "[Timing]: Pack kernel %ld microsec\n", total_time));
#endif    
    return OPAL_SUCCESS;
}

