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
 * Copyright (c) 2012      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "coll_basic.h"

#include "mpi.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/constants.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "coll_basic.h"


/*
 *	allgatherv_inter
 *
 *	Function:	- allgatherv using other MPI collectives
 *	Accepts:	- same as MPI_Allgatherv()
 *	Returns:	- MPI_SUCCESS or error code
 */
int
mca_coll_basic_allgatherv_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    int rsize, err, i;
    size_t *scounts;
    ptrdiff_t *sdisps;
    ompi_count_array_t scounts_desc;
    ompi_disp_array_t sdisps_desc;

    rsize = ompi_comm_remote_size(comm);

    scounts = (size_t *) malloc(rsize * sizeof(size_t) + rsize * sizeof(ptrdiff_t));
    sdisps = (ptrdiff_t *) (scounts + rsize);
    if (NULL == scounts) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    for (i = 0; i < rsize; i++) {
        scounts[i] = args->src.info.count;
        sdisps[i] = 0;
    }

    OMPI_COUNT_ARRAY_INIT(&scounts_desc, scounts);
    OMPI_DISP_ARRAY_INIT(&sdisps_desc, sdisps);
    ompi_coll_args_t _ata;
    ompi_coll_args_alltoallv(&_ata, args->src.info.buffer, scounts_desc, sdisps_desc,
                             args->src.info.datatype, args->dst.info_v.buffer,
                             args->dst.info_v.counts, args->dst.info_v.displacements,
                             args->dst.info_v.datatype);
    err = comm->c_coll->coll_alltoallv(&_ata, comm,
                                       comm->c_coll->coll_alltoallv_module);

    if (NULL != scounts) {
        free(scounts);
    }

    return err;
}
