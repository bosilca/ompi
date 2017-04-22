#include "ompi_config.h"
#include "coll_adapt_cuda.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <stdarg.h>

static int coll_adapt_cuda_kernel_enabled = 0;

int coll_adapt_cuda_init(void)
{
    int device;
    cudaError cuda_err;

    cuda_err = cudaGetDevice(&device);
    if( cudaSuccess != cuda_err ) {
       // OPAL_OUTPUT_VERBOSE((0, opal_datatype_cuda_output, "Cannot retrieve the device being used. Drop CUDA support!\n"));
        return -1;
    }
    
    coll_adapt_cuda_kernel_enabled = 1;
    return 0;
}

int coll_adapt_cuda_fini(void)
{
    coll_adapt_cuda_kernel_enabled = 0;
    return 0;
}

int coll_adapt_cuda_is_gpu_buffer(const void *ptr)
{
    CUmemorytype memType;
    CUdeviceptr dbuf = (CUdeviceptr)ptr;
    int res;

    res = cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dbuf);
    if (res != CUDA_SUCCESS) {
        /* If we cannot determine it is device pointer,
         * just assume it is not. */
      //  OPAL_OUTPUT_VERBOSE((1, opal_datatype_cuda_output, "!!!!!!! %p is not a gpu buffer. Take no-CUDA path!\n", ptr));
        return 0;
    }
    /* Anything but CU_MEMORYTYPE_DEVICE is not a GPU memory */
    return (memType == CU_MEMORYTYPE_DEVICE) ? 1 : 0;
}