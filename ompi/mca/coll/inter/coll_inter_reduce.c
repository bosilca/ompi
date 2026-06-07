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
 * Copyright (c) 2015-2016 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "coll_inter.h"

#include <stdio.h>

#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/op/op.h"
#include "ompi/mca/pml/pml.h"

/*
 *	reduce_inter
 *
 *	Function:	- reduction using the local_comm
 *	Accepts:	- same as MPI_Reduce()
 *	Returns:	- MPI_SUCCESS or error code
 */
int
mca_coll_inter_reduce_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    size_t count = args->dst.info.count;
    struct ompi_datatype_t *dtype = args->dst.info.datatype;
    int root = args->root;
    int rank, err;

    /* Initialize */
    rank = ompi_comm_rank(comm);

    if (MPI_PROC_NULL == root) {
        /* do nothing */
        err = OMPI_SUCCESS;
    } else if (MPI_ROOT != root) {
        ptrdiff_t gap, span;
        char *free_buffer = NULL;
        char *pml_buffer = NULL;

	/* Perform the reduce locally with the first process as root */
        span = opal_datatype_span(&dtype->super, count, &gap);

	free_buffer = (char*)malloc(span);
	if (NULL == free_buffer) {
	    return OMPI_ERR_OUT_OF_RESOURCE;
	}
	pml_buffer = free_buffer - gap;

	ompi_coll_args_t _r;
	ompi_coll_args_reduce(&_r, args->src.info.buffer, pml_buffer, count, dtype, args->op, 0);
	err = comm->c_local_comm->c_coll->coll_reduce(&_r, comm->c_local_comm,
                                                     comm->c_local_comm->c_coll->coll_reduce_module);
	if (0 == rank) {
	    /* First process sends the result to the root */
	    err = MCA_PML_CALL(send(pml_buffer, count, dtype, root,
				    MCA_COLL_BASE_TAG_REDUCE,
				    MCA_PML_BASE_SEND_STANDARD, comm));
	    if (OMPI_SUCCESS != err) {
                return err;
            }
	}

	if (NULL != free_buffer) {
	    free(free_buffer);
	}
    } else {
        /* Root receives the reduced message from the first process  */
	err = MCA_PML_CALL(recv(args->dst.info.buffer, count, dtype, 0,
				MCA_COLL_BASE_TAG_REDUCE, comm,
				MPI_STATUS_IGNORE));
	if (OMPI_SUCCESS != err) {
	    return err;
	}
    }
    /* All done */
    return err;
}
