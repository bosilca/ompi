#ifndef COLL_ADAPT_CUDA_H
#define COLL_ADAPT_CUDA_H

#include "ompi_config.h"

#if OPAL_CUDA_SUPPORT
#include "ompi/mca/coll/base/coll_base_topo.h"

struct coll_adapt_cuda_function_table_s {
    int (*coll_adapt_cuda_init_p)(void);
    int (*coll_adapt_cuda_fini_p)(void);
    int (*coll_adapt_cuda_is_gpu_buffer_p)(const void *ptr);
};

typedef struct coll_adapt_cuda_function_table_s coll_adapt_cuda_function_table_t;

int coll_adapt_cuda_init(void);

int coll_adapt_cuda_fini(void);

int coll_adapt_cuda_is_gpu_buffer(const void *ptr);

int coll_adapt_cuda_get_gpu_topo(ompi_coll_topo_gpu_t *gpu_topo);

int coll_adapt_cuda_free_gpu_topo(ompi_coll_topo_gpu_t *gpu_topo);

#endif

#endif