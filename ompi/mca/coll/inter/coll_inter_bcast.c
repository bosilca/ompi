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
 * Copyright (c) 2013 Cisco Systems, Inc.  All rights reserved.
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
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"


/*
 *	bcast_inter
 *
 *	Function:	- broadcast using the local_comm
 *	Accepts:	- same arguments as MPI_Bcast()
 *	Returns:	- MPI_SUCCESS or error code
 */
int
mca_coll_inter_bcast_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    void *buff = args->src.info.buffer;
    size_t count = args->src.info.count;
    struct ompi_datatype_t *datatype = args->src.info.datatype;
    int root = args->root;
    int rank;
    int err;

    rank = ompi_comm_rank(comm);

    if (MPI_PROC_NULL == root) {
        /* do nothing */
        err = OMPI_SUCCESS;
    } else if (MPI_ROOT != root) {
        /* Non-root, first process receives the data and bcast to others */
	if ( 0 == rank ) {
	    err = MCA_PML_CALL(recv(buff, count, datatype, root,
				    MCA_COLL_BASE_TAG_BCAST, comm,
				    MPI_STATUS_IGNORE));
	    if (OMPI_SUCCESS != err) {
                return err;
            }
	}
	ompi_coll_args_t _b;
	ompi_coll_args_bcast(&_b, buff, count, datatype, 0);
	err = comm->c_local_comm->c_coll->coll_bcast(&_b, comm->c_local_comm,
                                                    comm->c_local_comm->c_coll->coll_bcast_module);
    } else {
        /* root section, send to the first process of the remote group */
	err = MCA_PML_CALL(send(buff, count, datatype, 0,
				MCA_COLL_BASE_TAG_BCAST,
				MCA_PML_BASE_SEND_STANDARD,
				comm));
	if (OMPI_SUCCESS != err) {
	    return err;
	}
    }

    /* All done */
    return err;
}

