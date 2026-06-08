/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2011-2012 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2015-2021 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2025-2026 Triad National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "ompi/mpi/fortran/mpif-h/bindings.h"
#include "ompi/mpi/fortran/base/constants.h"
#include "ompi/mpi/fortran/base/fortran_base_topo_neighbors.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mpi/c/coll_w_dispatch.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak PMPI_NEIGHBOR_ALLTOALLW_INIT = ompi_neighbor_alltoallw_init_f
#pragma weak pmpi_neighbor_alltoallw_init = ompi_neighbor_alltoallw_init_f
#pragma weak pmpi_neighbor_alltoallw_init_ = ompi_neighbor_alltoallw_init_f
#pragma weak pmpi_neighbor_alltoallw_init__ = ompi_neighbor_alltoallw_init_f

#pragma weak PMPI_Neighbor_alltoallw_init_f = ompi_neighbor_alltoallw_init_f
#pragma weak PMPI_Neighbor_alltoallw_init_f08 = ompi_neighbor_alltoallw_init_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPI_NEIGHBOR_ALLTOALLW_INIT,
                            pmpi_neighbor_alltoallw_init,
                            pmpi_neighbor_alltoallw_init_,
                            pmpi_neighbor_alltoallw_init__,
                            pompi_neighbor_alltoallw_init_f,
                            (char *sendbuf, MPI_Fint *sendcounts, MPI_Aint *sdispls, MPI_Fint *sendtypes, char *recvbuf, MPI_Fint *recvcounts, MPI_Aint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *info, MPI_Fint *request, MPI_Fint *ierr),
                            (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, info, request, ierr) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_NEIGHBOR_ALLTOALLW_INIT = ompi_neighbor_alltoallw_init_f
#pragma weak mpi_neighbor_alltoallw_init = ompi_neighbor_alltoallw_init_f
#pragma weak mpi_neighbor_alltoallw_init_ = ompi_neighbor_alltoallw_init_f
#pragma weak mpi_neighbor_alltoallw_init__ = ompi_neighbor_alltoallw_init_f

#pragma weak MPI_Neighbor_alltoallw_init_f = ompi_neighbor_alltoallw_init_f
#pragma weak MPI_Neighbor_alltoallw_init_f08 = ompi_neighbor_alltoallw_init_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPI_NEIGHBOR_ALLTOALLW_INIT,
                            mpi_neighbor_alltoallw_init,
                            mpi_neighbor_alltoallw_init_,
                            mpi_neighbor_alltoallw_init__,
                            ompi_neighbor_alltoallw_init_f,
                            (char *sendbuf, MPI_Fint *sendcounts, MPI_Aint *sdispls, MPI_Fint *sendtypes, char *recvbuf, MPI_Fint *recvcounts, MPI_Aint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *info, MPI_Fint *request, MPI_Fint *ierr),
                            (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, info, request, ierr) )
#else
#define ompi_neighbor_alltoallw_init_f pompi_neighbor_alltoallw_init_f
#endif
#endif


void ompi_neighbor_alltoallw_init_f(char *sendbuf, MPI_Fint *sendcounts,
                                    MPI_Aint *sdispls, MPI_Fint *sendtypes,
                                    char *recvbuf, MPI_Fint *recvcounts,
                                    MPI_Aint *rdispls, MPI_Fint *recvtypes,
                                    MPI_Fint *comm, MPI_Fint *info, MPI_Fint *request, MPI_Fint *ierr)
{
    MPI_Comm c_comm;
    MPI_Info c_info;
    MPI_Request c_request;
    int indegree, outdegree, c_ierr;
    ompi_count_array_t sendcounts_desc, recvcounts_desc;
    ompi_disp_array_t sdispls_desc, rdispls_desc;
    OMPI_ARRAY_NAME_DECL(sendcounts);
    OMPI_ARRAY_NAME_DECL(recvcounts);

    c_comm = PMPI_Comm_f2c(*comm);
    c_info = PMPI_Info_f2c(*info);

    c_ierr = ompi_fortran_neighbor_count(c_comm, &indegree, &outdegree);
    if (MPI_SUCCESS != c_ierr) {
        if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);
        return;
    }

    OMPI_ARRAY_FINT_2_INT(sendcounts, outdegree);
    OMPI_ARRAY_FINT_2_INT(recvcounts, indegree);
    OMPI_COUNT_ARRAY_INIT(&sendcounts_desc, OMPI_ARRAY_NAME_CONVERT(sendcounts));
    OMPI_COUNT_ARRAY_INIT(&recvcounts_desc, OMPI_ARRAY_NAME_CONVERT(recvcounts));
    OMPI_DISP_ARRAY_INIT(&sdispls_desc, sdispls);
    OMPI_DISP_ARRAY_INIT(&rdispls_desc, rdispls);

    /* Neighbor_alltoallw_init does not support MPI_IN_PLACE; datatype handles
     * are passed straight through and retained on the persistent request. */
    sendbuf = (char *) OMPI_F2C_BOTTOM(sendbuf);
    recvbuf = (char *) OMPI_F2C_BOTTOM(recvbuf);

    c_ierr = ompi_neighbor_alltoallw_init_dispatch(sendbuf, sendcounts_desc, sdispls_desc,
                                                   ompi_datatype_array_create_f(sendtypes),
                                                   recvbuf, recvcounts_desc, rdispls_desc,
                                                   ompi_datatype_array_create_f(recvtypes),
                                                   c_comm, c_info, &c_request,
                                                   "MPI_Neighbor_alltoallw_init");
    if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);
    if (MPI_SUCCESS == c_ierr) {
        *request = PMPI_Request_c2f(c_request);
        if ((void *)sendcounts != (void *)OMPI_ARRAY_NAME_CONVERT(sendcounts)) {
            ((ompi_coll_base_nbc_request_t *) c_request)->args.mask |= OMPI_COLL_ARGS_FREE_SRC_COUNTS;
            ((ompi_coll_base_nbc_request_t *) c_request)->args.mask |= OMPI_COLL_ARGS_FREE_DST_COUNTS;
        }
        ompi_coll_base_add_release_arrays_cb(c_request);
    } else {
        OMPI_ARRAY_FINT_2_INT_CLEANUP(sendcounts);
        OMPI_ARRAY_FINT_2_INT_CLEANUP(recvcounts);
    }
}
