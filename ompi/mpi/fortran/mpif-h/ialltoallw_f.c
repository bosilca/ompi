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
 * Copyright (c) 2015-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2025      Triad National Security, LLC. All rights
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
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/mpi/c/coll_w_dispatch.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak PMPI_IALLTOALLW = ompi_ialltoallw_f
#pragma weak pmpi_ialltoallw = ompi_ialltoallw_f
#pragma weak pmpi_ialltoallw_ = ompi_ialltoallw_f
#pragma weak pmpi_ialltoallw__ = ompi_ialltoallw_f

#pragma weak PMPI_Ialltoallw_f = ompi_ialltoallw_f
#pragma weak PMPI_Ialltoallw_f08 = ompi_ialltoallw_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPI_IALLTOALLW,
                            pmpi_ialltoallw,
                            pmpi_ialltoallw_,
                            pmpi_ialltoallw__,
                            pompi_ialltoallw_f,
                            (char *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtypes, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr),
                            (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request, ierr) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_IALLTOALLW = ompi_ialltoallw_f
#pragma weak mpi_ialltoallw = ompi_ialltoallw_f
#pragma weak mpi_ialltoallw_ = ompi_ialltoallw_f
#pragma weak mpi_ialltoallw__ = ompi_ialltoallw_f

#pragma weak MPI_Ialltoallw_f = ompi_ialltoallw_f
#pragma weak MPI_Ialltoallw_f08 = ompi_ialltoallw_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPI_IALLTOALLW,
                            mpi_ialltoallw,
                            mpi_ialltoallw_,
                            mpi_ialltoallw__,
                            ompi_ialltoallw_f,
                            (char *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtypes, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr),
                            (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, request, ierr) )
#else
#define ompi_ialltoallw_f pompi_ialltoallw_f
#endif
#endif


void ompi_ialltoallw_f(char *sendbuf, MPI_Fint *sendcounts,
                       MPI_Fint *sdispls, MPI_Fint *sendtypes,
                       char *recvbuf, MPI_Fint *recvcounts,
                       MPI_Fint *rdispls, MPI_Fint *recvtypes,
                       MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr)
{
    MPI_Comm c_comm;
    MPI_Request c_request;
    int size, c_ierr;
    ompi_count_array_t sendcounts_desc, recvcounts_desc;
    ompi_disp_array_t sdispls_desc, rdispls_desc;
    ompi_datatype_array_t sendtypes_desc = OMPI_DATATYPE_ARRAY_NULL;
    OMPI_ARRAY_NAME_DECL(sendcounts);
    OMPI_ARRAY_NAME_DECL(sdispls);
    OMPI_ARRAY_NAME_DECL(recvcounts);
    OMPI_ARRAY_NAME_DECL(rdispls);

    c_comm = PMPI_Comm_f2c(*comm);
    size = OMPI_COMM_IS_INTER(c_comm)?ompi_comm_remote_size(c_comm):ompi_comm_size(c_comm);

    /* Pass the Fortran datatype handles straight through; the dispatch retains
     * stable object pointers on the request, so no temporary MPI_Datatype
     * array (and no MPI_Datatype-array free bits) are needed here.  The count
     * and displacement arrays still transfer ownership to the request when a
     * Fortran-to-int conversion allocated a temporary. */
    if (!OMPI_IS_FORTRAN_IN_PLACE(sendbuf)) {
        OMPI_ARRAY_FINT_2_INT(sendcounts, size);
        OMPI_ARRAY_FINT_2_INT(sdispls, size);
        sendtypes_desc = ompi_datatype_array_create_f(sendtypes);
    }
    OMPI_COUNT_ARRAY_INIT(&sendcounts_desc, OMPI_ARRAY_NAME_CONVERT(sendcounts));
    OMPI_DISP_ARRAY_INIT(&sdispls_desc, OMPI_ARRAY_NAME_CONVERT(sdispls));

    OMPI_ARRAY_FINT_2_INT(recvcounts, size);
    OMPI_ARRAY_FINT_2_INT(rdispls, size);
    OMPI_COUNT_ARRAY_INIT(&recvcounts_desc, OMPI_ARRAY_NAME_CONVERT(recvcounts));
    OMPI_DISP_ARRAY_INIT(&rdispls_desc, OMPI_ARRAY_NAME_CONVERT(rdispls));

    sendbuf = (char *) OMPI_F2C_IN_PLACE(sendbuf);
    sendbuf = (char *) OMPI_F2C_BOTTOM(sendbuf);
    recvbuf = (char *) OMPI_F2C_BOTTOM(recvbuf);

    c_ierr = ompi_ialltoallw_dispatch(sendbuf, sendcounts_desc, sdispls_desc, sendtypes_desc,
                                      recvbuf, recvcounts_desc, rdispls_desc,
                                      ompi_datatype_array_create_f(recvtypes),
                                      c_comm, &c_request, "MPI_Ialltoallw");
    if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);
    if (MPI_SUCCESS == c_ierr) *request = PMPI_Request_c2f(c_request);

    if ( REQUEST_COMPLETE(c_request)) {
        OMPI_ARRAY_FINT_2_INT_CLEANUP(sendcounts);
        OMPI_ARRAY_FINT_2_INT_CLEANUP(sdispls);
        OMPI_ARRAY_FINT_2_INT_CLEANUP(recvcounts);
        OMPI_ARRAY_FINT_2_INT_CLEANUP(rdispls);
    } else {
        if ((void *)sendcounts != (void *)OMPI_ARRAY_NAME_CONVERT(sendcounts)) {
            ((ompi_coll_base_nbc_request_t *) c_request)->args.mask |= OMPI_COLL_ARGS_FREE_SRC_COUNTS;
            ((ompi_coll_base_nbc_request_t *) c_request)->args.mask |= OMPI_COLL_ARGS_FREE_DST_COUNTS;
        }
        if ((void *)sdispls != (void *)OMPI_ARRAY_NAME_CONVERT(sdispls)) {
            ((ompi_coll_base_nbc_request_t *) c_request)->args.mask |= OMPI_COLL_ARGS_FREE_SRC_DISPS;
            ((ompi_coll_base_nbc_request_t *) c_request)->args.mask |= OMPI_COLL_ARGS_FREE_DST_DISPS;
        }
        ompi_coll_base_add_release_arrays_cb(c_request);
    }
}
