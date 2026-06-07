/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006-2007 University of Houston. All rights reserved.
 * Copyright (c) 2013      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2023      Jeffrey M. Squyres.  All rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "coll_inter.h"

#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/op/op.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/mca/pml/pml.h"

/*
 *	allreduce_inter
 *
 *	Function:	- allreduce using other MPI collectives
 *	Accepts:	- same as MPI_Allreduce()
 *	Returns:	- MPI_SUCCESS or error code
 */
int
mca_coll_inter_allreduce_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    void *rbuf = args->dst.info.buffer;
    size_t count = args->dst.info.count;
    struct ompi_datatype_t *dtype = args->dst.info.datatype;
    int err, rank, root = 0;
    char *tmpbuf = NULL, *pml_buffer = NULL;
    const char *source;
    ptrdiff_t gap, span;

    rank = ompi_comm_rank(comm);

    /* Perform the reduction locally */
    span = opal_datatype_span(&dtype->super, count, &gap);

    tmpbuf = (char *) malloc(span);
    if (NULL == tmpbuf) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    pml_buffer = tmpbuf - gap;
    source = (MPI_IN_PLACE == args->src.info.buffer) ? rbuf : args->src.info.buffer;

    ompi_coll_args_t _r;
    ompi_coll_args_reduce(&_r, source, pml_buffer, count, dtype, args->op, root);
    err = comm->c_local_comm->c_coll->coll_reduce(&_r, comm->c_local_comm,
                                                  comm->c_local_comm->c_coll->coll_reduce_module);
    if (OMPI_SUCCESS != err) {
        goto exit;
    }

    if (rank == root) {
        /* Do a send-recv between the two root procs. to avoid deadlock */
        err = ompi_coll_base_sendrecv_actual(pml_buffer, count, dtype, 0,
                                             MCA_COLL_BASE_TAG_ALLREDUCE,
                                             rbuf, count, dtype, 0,
                                             MCA_COLL_BASE_TAG_ALLREDUCE,
                                             comm, MPI_STATUS_IGNORE);
        if (OMPI_SUCCESS != err) {
            goto exit;
        }
    }

    /* bcast the message to all the local processes */
    ompi_coll_args_t _b;
    ompi_coll_args_bcast(&_b, rbuf, count, dtype, root);
    err = comm->c_local_comm->c_coll->coll_bcast(&_b, comm->c_local_comm,
                                                 comm->c_local_comm->c_coll->coll_bcast_module);
    if (OMPI_SUCCESS != err) {
            goto exit;
    }

  exit:
    if (NULL != tmpbuf) {
        free(tmpbuf);
    }

    return err;
}
