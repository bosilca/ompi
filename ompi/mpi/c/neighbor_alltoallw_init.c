/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2008 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2007      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2012-2017 Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2014-2023 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2017      IBM Corporation.  All rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include <stdio.h>

#include "ompi/mpi/c/bindings.h"
#include "ompi/mpi/c/coll_w_dispatch.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Neighbor_alltoallw_init = PMPI_Neighbor_alltoallw_init
#endif
#define MPI_Neighbor_alltoallw_init PMPI_Neighbor_alltoallw_init
#endif

static const char FUNC_NAME[] = "MPI_Neighbor_alltoallw_init";


int MPI_Neighbor_alltoallw_init(const void *sendbuf, const int sendcounts[], const MPI_Aint sdispls[],
                                const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
                                const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm,
                                MPI_Info info, MPI_Request *request)
{
    ompi_count_array_t sendcounts_desc, recvcounts_desc;
    ompi_disp_array_t sdispls_desc, rdispls_desc;

    OMPI_COUNT_ARRAY_INIT(&sendcounts_desc, sendcounts);
    OMPI_DISP_ARRAY_INIT(&sdispls_desc, sdispls);
    OMPI_COUNT_ARRAY_INIT(&recvcounts_desc, recvcounts);
    OMPI_DISP_ARRAY_INIT(&rdispls_desc, rdispls);

    return ompi_neighbor_alltoallw_init_dispatch(sendbuf, sendcounts_desc, sdispls_desc,
                                                 ompi_datatype_array_create((struct ompi_datatype_t * const *) sendtypes),
                                                 recvbuf, recvcounts_desc, rdispls_desc,
                                                 ompi_datatype_array_create((struct ompi_datatype_t * const *) recvtypes),
                                                 comm, info, request, FUNC_NAME);
}

