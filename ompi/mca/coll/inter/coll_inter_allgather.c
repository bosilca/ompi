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
 * Copyright (c) 2006-2010 University of Houston. All rights reserved.
 * Copyright (c) 2015-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2022      IBM Corporation.  All rights reserved.
 * Copyright (c) 2024      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "coll_inter.h"

#include <stdlib.h>

#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/coll/base/coll_base_util.h"

/*
 *	allgather_inter
 *
 *	Function:	- allgather using other MPI collections
 *	Accepts:	- same as MPI_Allgather()
 *	Returns:	- MPI_SUCCESS or error code
 */
int
mca_coll_inter_allgather_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    size_t scount = args->src.info.count;
    struct ompi_datatype_t *sdtype = args->src.info.datatype;
    void *rbuf = args->dst.info.buffer;
    size_t rcount = args->dst.info.count;
    struct ompi_datatype_t *rdtype = args->dst.info.datatype;
    int rank, root = 0, size, rsize, err = OMPI_SUCCESS, i;
    char *ptmp_free = NULL, *ptmp = NULL;
    ptrdiff_t gap, span;
    void *rbuf_ptr;

    rank = ompi_comm_rank(comm);
    size = ompi_comm_size(comm->c_local_comm);
    rsize = ompi_comm_remote_size(comm);

    /* Perform the gather locally at the root */
    if ( scount > 0 ) {
        span = opal_datatype_span(&sdtype->super, (int64_t)scount*(int64_t)size, &gap);
	    ptmp_free = (char*)malloc(span);
	    if (NULL == ptmp_free) {
	        return OMPI_ERR_OUT_OF_RESOURCE;
	    }
        ptmp = ptmp_free - gap;

	    ompi_coll_args_t _g;
	    ompi_coll_args_gather(&_g, args->src.info.buffer, scount, sdtype, ptmp, scount, sdtype, 0);
	    err = comm->c_local_comm->c_coll->coll_gather(&_g, comm->c_local_comm,
						     comm->c_local_comm->c_coll->coll_gather_module);
	    if (OMPI_SUCCESS != err) {
	        goto exit;
	    }
    }

    if (rank == root) {
	    /* Do a send-recv between the two root procs. to avoid deadlock */
        err = ompi_coll_base_sendrecv_actual(ptmp, scount*(size_t)size, sdtype, 0,
                                             MCA_COLL_BASE_TAG_ALLGATHER,
                                             rbuf, rcount*(size_t)rsize, rdtype, 0,
                                             MCA_COLL_BASE_TAG_ALLGATHER,
                                             comm, MPI_STATUS_IGNORE);
        if (OMPI_SUCCESS != err) {
            goto exit;
        }
    }
    /* bcast the message to all the local processes */
    if ( rcount > 0 ) {
        if ( OPAL_UNLIKELY(rcount*(size_t)rsize > INT_MAX) ) {
            // Sending the message in the coll_bcast as "rcount*rsize" would exceed
            // the 'int count' parameter in the coll_bcast() function. Instead broadcast
            // the result in "rcount" chunks to the local group.
            span = opal_datatype_span(&rdtype->super, rcount, &gap);
            for( i = 0; i < rsize; ++i) {
                rbuf_ptr = (char*)rbuf + span * (size_t)i;
                ompi_coll_args_t _b;
                ompi_coll_args_bcast(&_b, rbuf_ptr, rcount, rdtype, root);
                err = comm->c_local_comm->c_coll->coll_bcast(&_b, comm->c_local_comm,
                                                             comm->c_local_comm->c_coll->coll_bcast_module);
                if (OMPI_SUCCESS != err) {
                    goto exit;
                }
            }
        } else {
            ompi_coll_args_t _b;
            ompi_coll_args_bcast(&_b, rbuf, rcount*rsize, rdtype, root);
            err = comm->c_local_comm->c_coll->coll_bcast(&_b, comm->c_local_comm,
                                                         comm->c_local_comm->c_coll->coll_bcast_module);
            if (OMPI_SUCCESS != err) {
                goto exit;
            }
        }
    }

 exit:
    if (NULL != ptmp_free) {
        free(ptmp_free);
    }

    return err;
}
