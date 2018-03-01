#include "coll_shared.h"

int mca_coll_shared_reduce_intra(const void *sbuf, void* rbuf, int count,
                                 struct ompi_datatype_t *dtype,
                                 struct ompi_op_t *op,
                                 int root,
                                 struct ompi_communicator_t *comm,
                                 mca_coll_base_module_t *module){
    printf("In shared reduce\n");
    return 1;
}
