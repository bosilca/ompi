/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2015-2021 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2017-2022 IBM Corporation.  All rights reserved.
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
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "coll_basic.h"


/*
 *	scatterv_intra
 *
 *	Function:	- scatterv operation
 *	Accepts:	- same arguments as MPI_Scatterv()
 *	Returns:	- MPI_SUCCESS or error code
 */
int
mca_coll_basic_scatterv_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    ompi_count_array_t scounts = args->src.info_v.counts;
    struct ompi_datatype_t *sdtype = args->src.info_v.datatype;
    void *rbuf = args->dst.info.buffer;
    size_t rcount = args->dst.info.count;
    struct ompi_datatype_t *rdtype = args->dst.info.datatype;
    int i, rank, size, err;
    char *ptmp;
    ptrdiff_t lb, extent;
    size_t sdsize;

    /* Initialize */

    rank = ompi_comm_rank(comm);
    size = ompi_comm_size(comm);

    /* If not root, receive data. */

    if (rank != args->root) {
        size_t rdsize;
        ompi_datatype_type_size(rdtype, &rdsize);
        /* Only receive if there is something to receive */
        if (rcount > 0 && rdsize > 0) {
            return MCA_PML_CALL(recv(rbuf, rcount, rdtype,
                                     args->root, MCA_COLL_BASE_TAG_SCATTERV,
                                     comm, MPI_STATUS_IGNORE));
        }
        return MPI_SUCCESS;
    }

    ompi_datatype_type_size(sdtype, &sdsize);
    if (OPAL_UNLIKELY(0 == sdsize)) {
        /* bozzo case */
        return MPI_SUCCESS;
    }

    /* I am the root, loop sending data. */

    err = ompi_datatype_get_extent(sdtype, &lb, &extent);
    if (OMPI_SUCCESS != err) {
        return OMPI_ERROR;
    }

    for (i = 0; i < size; ++i) {
        ptmp = ((char *) args->src.info_v.buffer) + (extent * ompi_disp_array_get(args->src.info_v.displacements, i));

        /* simple optimization */

        if (i == rank) {
            /* simple optimization or a local operation */
            if (ompi_count_array_get(scounts, i) > 0 && MPI_IN_PLACE != rbuf) {
                err = ompi_datatype_sndrcv(ptmp, ompi_count_array_get(scounts, i), sdtype, rbuf, rcount,
                                      rdtype);
                if (MPI_SUCCESS != err) {
                    return err;
                }
            }
        } else {
            /* Only send if there is something to send */
            if (ompi_count_array_get(scounts, i) > 0) {
                err = MCA_PML_CALL(send(ptmp, ompi_count_array_get(scounts, i), sdtype, i,
                                        MCA_COLL_BASE_TAG_SCATTERV,
                                        MCA_PML_BASE_SEND_STANDARD, comm));
                if (MPI_SUCCESS != err) {
                    return err;
                }
            }
        }
    }

    /* All done */

    return MPI_SUCCESS;
}


/*
 *	scatterv_inter
 *
 *	Function:	- scatterv operation
 *	Accepts:	- same arguments as MPI_Scatterv()
 *	Returns:	- MPI_SUCCESS or error code
 */
int
mca_coll_basic_scatterv_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    int root = args->root;
    int i, size, err;
    char *ptmp;
    ptrdiff_t lb, extent;
    ompi_request_t **reqs;

    /* Initialize */
    size = ompi_comm_remote_size(comm);

    /* If not root, receive data.  Note that we will only get here if
     * rcount > 0 or rank == root. */

    if (MPI_PROC_NULL == root) {
        /* do nothing */
        err = OMPI_SUCCESS;
    } else if (MPI_ROOT != root) {
        /* If not root, receive data. */
        err = MCA_PML_CALL(recv(args->dst.info.buffer, args->dst.info.count,
                                args->dst.info.datatype,
                                root, MCA_COLL_BASE_TAG_SCATTERV,
                                comm, MPI_STATUS_IGNORE));
    } else {
        /* I am the root, loop sending data. */
        err = ompi_datatype_get_extent(args->src.info_v.datatype, &lb, &extent);
        if (OMPI_SUCCESS != err) {
            return OMPI_ERROR;
        }

        reqs = ompi_coll_base_comm_get_reqs(module->base_data, size);
        if( NULL == reqs ) { return OMPI_ERR_OUT_OF_RESOURCE; }

        for (i = 0; i < size; ++i) {
            ptmp = ((char *) args->src.info_v.buffer) + (extent * ompi_disp_array_get(args->src.info_v.displacements, i));
            err = MCA_PML_CALL(isend(ptmp, ompi_count_array_get(args->src.info_v.counts, i), args->src.info_v.datatype, i,
                                     MCA_COLL_BASE_TAG_SCATTERV,
                                     MCA_PML_BASE_SEND_STANDARD, comm,
                                     &(reqs[i])));
            if (OMPI_SUCCESS != err) {
                ompi_coll_base_free_reqs(reqs, i + 1);
                return err;
            }
        }

        err = ompi_request_wait_all(size, reqs, MPI_STATUSES_IGNORE);
        if (OMPI_SUCCESS != err) {
            ompi_coll_base_free_reqs(reqs, size);
        }
    }

    /* All done */
    return err;
}
