/*
 * Copyright (c) 2012-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "opal/util/bit_ops.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/base/base.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/coll/base/coll_base_functions.h"

int
ompi_coll_base_agree_noft(ompi_coll_args_t *args,
                         struct ompi_communicator_t *comm,
                         mca_coll_base_module_t *module)
{
    void *sendbuf = OMPI_COMM_IS_INTER(comm) ? args->src.info.buffer : MPI_IN_PLACE;
    ompi_coll_args_t _ar;
    ompi_coll_args_allreduce(&_ar, sendbuf, args->src.info.buffer, args->src.info.count,
                             args->src.info.datatype, args->op);
    return comm->c_coll->coll_allreduce(&_ar, comm,
                                       comm->c_coll->coll_allreduce_module);
}

int
ompi_coll_base_iagree_noft(ompi_coll_args_t *args,
                          struct ompi_communicator_t *comm,
                          ompi_request_t **request,
                          mca_coll_base_module_t *module)
{
    void *sendbuf = OMPI_COMM_IS_INTER(comm) ? args->src.info.buffer : MPI_IN_PLACE;
    ompi_coll_args_t _ar;
    ompi_coll_args_allreduce(&_ar, sendbuf, args->src.info.buffer, args->src.info.count,
                             args->src.info.datatype, args->op);
    return comm->c_coll->coll_iallreduce(&_ar, comm, request,
                                        comm->c_coll->coll_iallreduce_module);
}
