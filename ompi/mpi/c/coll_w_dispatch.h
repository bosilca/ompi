/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/*
 * Shared back-end dispatch for the alltoallw / neighbor_alltoallw family.
 *
 * The per-peer datatype arguments of these operations are passed as tagged
 * ompi_datatype_array_t values, so a single implementation serves both the C
 * bindings (which wrap their "MPI_Datatype []" arrays) and the Fortran
 * bindings (which wrap their "MPI_Fint []" handle arrays without allocating or
 * converting a temporary C array).  Each function performs the MPI parameter
 * checks, builds the ompi_coll_args_t descriptor, invokes the coll component,
 * and (for the non-blocking and persistent variants) retains the datatypes on
 * the request.  The caller is responsible for any language-specific buffer
 * fixups (e.g. Fortran MPI_BOTTOM / MPI_IN_PLACE translation) before calling.
 */

#ifndef OMPI_MPI_C_COLL_W_DISPATCH_H
#define OMPI_MPI_C_COLL_W_DISPATCH_H

#include "ompi_config.h"

#include "ompi/communicator/communicator.h"
#include "ompi/info/info.h"
#include "ompi/request/request.h"
#include "ompi/util/count_disp_array.h"
#include "ompi/util/datatype_array.h"

BEGIN_C_DECLS

int ompi_alltoallw_dispatch(const void *sendbuf, ompi_count_array_t scounts,
                            ompi_disp_array_t sdisps, ompi_datatype_array_t stypes,
                            void *recvbuf, ompi_count_array_t rcounts,
                            ompi_disp_array_t rdisps, ompi_datatype_array_t rtypes,
                            ompi_communicator_t *comm, const char *func_name);

int ompi_ialltoallw_dispatch(const void *sendbuf, ompi_count_array_t scounts,
                             ompi_disp_array_t sdisps, ompi_datatype_array_t stypes,
                             void *recvbuf, ompi_count_array_t rcounts,
                             ompi_disp_array_t rdisps, ompi_datatype_array_t rtypes,
                             ompi_communicator_t *comm, ompi_request_t **request,
                             const char *func_name);

int ompi_alltoallw_init_dispatch(const void *sendbuf, ompi_count_array_t scounts,
                                 ompi_disp_array_t sdisps, ompi_datatype_array_t stypes,
                                 void *recvbuf, ompi_count_array_t rcounts,
                                 ompi_disp_array_t rdisps, ompi_datatype_array_t rtypes,
                                 ompi_communicator_t *comm, ompi_info_t *info,
                                 ompi_request_t **request, const char *func_name);

int ompi_neighbor_alltoallw_dispatch(const void *sendbuf, ompi_count_array_t scounts,
                                     ompi_disp_array_t sdisps, ompi_datatype_array_t stypes,
                                     void *recvbuf, ompi_count_array_t rcounts,
                                     ompi_disp_array_t rdisps, ompi_datatype_array_t rtypes,
                                     ompi_communicator_t *comm, const char *func_name);

int ompi_ineighbor_alltoallw_dispatch(const void *sendbuf, ompi_count_array_t scounts,
                                      ompi_disp_array_t sdisps, ompi_datatype_array_t stypes,
                                      void *recvbuf, ompi_count_array_t rcounts,
                                      ompi_disp_array_t rdisps, ompi_datatype_array_t rtypes,
                                      ompi_communicator_t *comm, ompi_request_t **request,
                                      const char *func_name);

int ompi_neighbor_alltoallw_init_dispatch(const void *sendbuf, ompi_count_array_t scounts,
                                          ompi_disp_array_t sdisps, ompi_datatype_array_t stypes,
                                          void *recvbuf, ompi_count_array_t rcounts,
                                          ompi_disp_array_t rdisps, ompi_datatype_array_t rtypes,
                                          ompi_communicator_t *comm, ompi_info_t *info,
                                          ompi_request_t **request, const char *func_name);

END_C_DECLS

#endif /* OMPI_MPI_C_COLL_W_DISPATCH_H */
