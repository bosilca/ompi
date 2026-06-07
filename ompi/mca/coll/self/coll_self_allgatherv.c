/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
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

#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "coll_self.h"


/*
 *	allgatherv_intra
 *
 *	Function:	- allgather
 *	Accepts:	- same as MPI_Allgatherv()
 *	Returns:	- MPI_SUCCESS or error code
 */
int mca_coll_self_allgatherv_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    if (MPI_IN_PLACE == args->src.info.buffer) {
        return MPI_SUCCESS;
    } else {
        int err;
        ptrdiff_t lb, extent;
        err = ompi_datatype_get_extent(args->dst.info_v.datatype, &lb, &extent);
        if (OMPI_SUCCESS != err) {
            return OMPI_ERROR;
        }
        return ompi_datatype_sndrcv(args->src.info.buffer, args->src.info.count, args->src.info.datatype,
                               ((char *) args->dst.info_v.buffer) + ompi_disp_array_get(args->dst.info_v.displacements, 0) * extent, ompi_count_array_get(args->dst.info_v.counts, 0), args->dst.info_v.datatype);
    }
}
