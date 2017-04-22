#ifndef COLL_ADAPT_CUDA_CUH
#define COLL_ADAPT_CUDA_CUH

BEGIN_C_DECLS

/* init cuda collective kernel lib */    
int coll_adapt_cuda_init(void);

/* fini cuda collective kernel lib */
int coll_adapt_cuda_fini(void);

/* check if a pointer is GPU or CPU */
int coll_adapt_cuda_is_gpu_buffer(const void *ptr);
        
END_C_DECLS
#endif