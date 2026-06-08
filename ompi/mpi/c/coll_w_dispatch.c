/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 * Shared back-end dispatch for the alltoallw / neighbor_alltoallw family.
 * See coll_w_dispatch.h for the rationale.
 */

#include "ompi_config.h"

#include "ompi/mpi/c/bindings.h"
#include "ompi/mpi/c/coll_w_dispatch.h"
#include "ompi/runtime/params.h"
#include "ompi/communicator/communicator.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/memchecker.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/mca/topo/topo.h"
#include "ompi/mca/topo/base/base.h"
#include "ompi/runtime/ompi_spc.h"

/*
 * Parameter checks shared by alltoallw, ialltoallw and alltoallw_init.
 *
 * Operates on the tagged arrays through the accessors so it serves both the C
 * and Fortran callers.  On MPI_IN_PLACE the send-side descriptors are aliased
 * to the receive side (matching the historical bindings).  Returns MPI_SUCCESS
 * when all checks pass; otherwise the relevant error handler has already been
 * invoked and its MPI error code is returned.
 */
static int alltoallw_param_check(ompi_communicator_t *comm,
                                 const void *sendbuf, void *recvbuf,
                                 ompi_count_array_t *scounts, ompi_disp_array_t *sdisps,
                                 ompi_datatype_array_t *stypes,
                                 ompi_count_array_t rcounts, ompi_disp_array_t rdisps,
                                 ompi_datatype_array_t rtypes,
                                 const char *func_name)
{
    int i, size, err = MPI_SUCCESS;

    OMPI_ERR_INIT_FINALIZE(func_name);
    if (ompi_comm_invalid(comm)) {
        return OMPI_ERRHANDLER_NOHANDLE_INVOKE(MPI_ERR_COMM, func_name);
    }

    if (MPI_IN_PLACE == sendbuf) {
        *scounts = rcounts;
        *sdisps  = rdisps;
        *stypes  = rtypes;
    }

    if ((NULL == ompi_count_array_ptr(*scounts)) || (NULL == ompi_disp_array_ptr(*sdisps)) ||
        (OMPI_DATATYPE_ARRAY_NULL == *stypes) ||
        (NULL == ompi_count_array_ptr(rcounts)) || (NULL == ompi_disp_array_ptr(rdisps)) ||
        (OMPI_DATATYPE_ARRAY_NULL == rtypes) ||
        (MPI_IN_PLACE == sendbuf && OMPI_COMM_IS_INTER(comm)) ||
        MPI_IN_PLACE == recvbuf) {
        return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ARG, func_name);
    }

    size = OMPI_COMM_IS_INTER(comm) ? ompi_comm_remote_size(comm) : ompi_comm_size(comm);
    for (i = 0; i < size; ++i) {
        struct ompi_datatype_t *stype = ompi_datatype_array_get(*stypes, i);
        struct ompi_datatype_t *rtype = ompi_datatype_array_get(rtypes, i);
        OMPI_CHECK_DATATYPE_FOR_SEND(err, stype, (ptrdiff_t) ompi_count_array_get(*scounts, i));
        OMPI_ERRHANDLER_CHECK(err, comm, err, func_name);
        OMPI_CHECK_DATATYPE_FOR_RECV(err, rtype, (ptrdiff_t) ompi_count_array_get(rcounts, i));
        OMPI_ERRHANDLER_CHECK(err, comm, err, func_name);
    }

    if (MPI_IN_PLACE != sendbuf && !OMPI_COMM_IS_INTER(comm)) {
        size_t sendtype_size, recvtype_size;
        int me = ompi_comm_rank(comm);
        ompi_datatype_type_size(ompi_datatype_array_get(*stypes, me), &sendtype_size);
        ompi_datatype_type_size(ompi_datatype_array_get(rtypes, me), &recvtype_size);
        if ((sendtype_size * ompi_count_array_get(*scounts, me)) !=
            (recvtype_size * ompi_count_array_get(rcounts, me))) {
            return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_TRUNCATE, func_name);
        }
    }

    return MPI_SUCCESS;
}

/*
 * Parameter checks shared by neighbor_alltoallw, ineighbor_alltoallw and
 * neighbor_alltoallw_init.  Same conventions as alltoallw_param_check(), but
 * MPI_IN_PLACE is not permitted and the loop bounds come from the neighbor
 * topology.
 */
static int neighbor_alltoallw_param_check(ompi_communicator_t *comm,
                                          const void *sendbuf, void *recvbuf,
                                          ompi_count_array_t scounts, ompi_disp_array_t sdisps,
                                          ompi_datatype_array_t stypes,
                                          ompi_count_array_t rcounts, ompi_disp_array_t rdisps,
                                          ompi_datatype_array_t rtypes,
                                          const char *func_name)
{
    int i, err = MPI_SUCCESS, indegree, outdegree;

    OMPI_ERR_INIT_FINALIZE(func_name);
    if (ompi_comm_invalid(comm) || OMPI_COMM_IS_INTER(comm)) {
        return OMPI_ERRHANDLER_NOHANDLE_INVOKE(MPI_ERR_COMM, func_name);
    } else if (! OMPI_COMM_IS_TOPO(comm)) {
        return OMPI_ERRHANDLER_NOHANDLE_INVOKE(MPI_ERR_TOPOLOGY, func_name);
    }

    err = mca_topo_base_neighbor_count(comm, &indegree, &outdegree);
    OMPI_ERRHANDLER_CHECK(err, comm, err, func_name);
    if (((0 < outdegree) && ((NULL == ompi_count_array_ptr(scounts)) ||
                             (NULL == ompi_disp_array_ptr(sdisps)) ||
                             (OMPI_DATATYPE_ARRAY_NULL == stypes))) ||
        ((0 < indegree) && ((NULL == ompi_count_array_ptr(rcounts)) ||
                            (NULL == ompi_disp_array_ptr(rdisps)) ||
                            (OMPI_DATATYPE_ARRAY_NULL == rtypes))) ||
        MPI_IN_PLACE == sendbuf || MPI_IN_PLACE == recvbuf) {
        return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ARG, func_name);
    }
    for (i = 0; i < outdegree; ++i) {
        struct ompi_datatype_t *stype = ompi_datatype_array_get(stypes, i);
        OMPI_CHECK_DATATYPE_FOR_SEND(err, stype, (ptrdiff_t) ompi_count_array_get(scounts, i));
        OMPI_ERRHANDLER_CHECK(err, comm, err, func_name);
    }
    for (i = 0; i < indegree; ++i) {
        struct ompi_datatype_t *rtype = ompi_datatype_array_get(rtypes, i);
        OMPI_CHECK_DATATYPE_FOR_RECV(err, rtype, (ptrdiff_t) ompi_count_array_get(rcounts, i));
        OMPI_ERRHANDLER_CHECK(err, comm, err, func_name);
    }

    if (OMPI_COMM_IS_CART(comm)) {
        const mca_topo_base_comm_cart_2_2_0_t *cart = comm->c_topo->mtc.cart;
        if (0 > cart->ndims) {
            return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ARG, func_name);
        }
    } else if (OMPI_COMM_IS_GRAPH(comm)) {
        int degree;
        mca_topo_base_graph_neighbors_count(comm, ompi_comm_rank(comm), &degree);
        if (0 > degree) {
            return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ARG, func_name);
        }
    } else if (OMPI_COMM_IS_DIST_GRAPH(comm)) {
        const mca_topo_base_comm_dist_graph_2_2_0_t *dist_graph = comm->c_topo->mtc.dist_graph;
        if (dist_graph->indegree < 0 || dist_graph->outdegree < 0) {
            return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ARG, func_name);
        }
    }

    return MPI_SUCCESS;
}

int ompi_alltoallw_dispatch(const void *sendbuf, ompi_count_array_t scounts,
                            ompi_disp_array_t sdisps, ompi_datatype_array_t stypes,
                            void *recvbuf, ompi_count_array_t rcounts,
                            ompi_disp_array_t rdisps, ompi_datatype_array_t rtypes,
                            ompi_communicator_t *comm, const char *func_name)
{
    int err;

    SPC_RECORD(OMPI_SPC_ALLTOALLW, 1);

    MEMCHECKER(
        int i;
        int size;
        memchecker_comm(comm);
        size = OMPI_COMM_IS_INTER(comm) ? ompi_comm_remote_size(comm) : ompi_comm_size(comm);
        for (i = 0; i < size; i++) {
            if (MPI_IN_PLACE != sendbuf) {
                struct ompi_datatype_t *stype = ompi_datatype_array_get(stypes, i);
                memchecker_datatype(stype);
                memchecker_call(&opal_memchecker_base_isdefined,
                                (char *) sendbuf + ompi_disp_array_get(sdisps, i),
                                ompi_count_array_get(scounts, i), stype);
            }
            struct ompi_datatype_t *rtype = ompi_datatype_array_get(rtypes, i);
            memchecker_datatype(rtype);
            memchecker_call(&opal_memchecker_base_isaddressable,
                            (char *) recvbuf + ompi_disp_array_get(rdisps, i),
                            ompi_count_array_get(rcounts, i), rtype);
        }
    );

    if (MPI_PARAM_CHECK) {
        err = alltoallw_param_check(comm, sendbuf, recvbuf, &scounts, &sdisps, &stypes,
                                    rcounts, rdisps, rtypes, func_name);
        if (OPAL_UNLIKELY(MPI_SUCCESS != err)) {
            return err;
        }
    }

#if OPAL_ENABLE_FT_MPI
    /*
     * An early check, so as to return early if we are using a broken
     * communicator. This is not absolutely necessary since we will
     * check for this, and other, error conditions during the operation.
     */
    if (OPAL_UNLIKELY(!ompi_comm_iface_coll_check(comm, &err))) {
        OMPI_ERRHANDLER_RETURN(err, comm, err, func_name);
    }
#endif

    ompi_coll_args_t coll_args;
    ompi_coll_args_alltoallw(&coll_args, sendbuf, scounts, sdisps, stypes,
                             recvbuf, rcounts, rdisps, rtypes);
    err = comm->c_coll->coll_alltoallw(&coll_args, comm,
                                       comm->c_coll->coll_alltoallw_module);
    OMPI_ERRHANDLER_RETURN(err, comm, err, func_name);
}

int ompi_ialltoallw_dispatch(const void *sendbuf, ompi_count_array_t scounts,
                             ompi_disp_array_t sdisps, ompi_datatype_array_t stypes,
                             void *recvbuf, ompi_count_array_t rcounts,
                             ompi_disp_array_t rdisps, ompi_datatype_array_t rtypes,
                             ompi_communicator_t *comm, ompi_request_t **request,
                             const char *func_name)
{
    int err;

    SPC_RECORD(OMPI_SPC_IALLTOALLW, 1);

    MEMCHECKER(
        int i;
        int size;
        memchecker_comm(comm);
        size = OMPI_COMM_IS_INTER(comm) ? ompi_comm_remote_size(comm) : ompi_comm_size(comm);
        for (i = 0; i < size; i++) {
            if (MPI_IN_PLACE != sendbuf) {
                struct ompi_datatype_t *stype = ompi_datatype_array_get(stypes, i);
                memchecker_datatype(stype);
                memchecker_call(&opal_memchecker_base_isdefined,
                                (char *) sendbuf + ompi_disp_array_get(sdisps, i),
                                ompi_count_array_get(scounts, i), stype);
            }
            struct ompi_datatype_t *rtype = ompi_datatype_array_get(rtypes, i);
            memchecker_datatype(rtype);
            memchecker_call(&opal_memchecker_base_isaddressable,
                            (char *) recvbuf + ompi_disp_array_get(rdisps, i),
                            ompi_count_array_get(rcounts, i), rtype);
        }
    );

    if (MPI_PARAM_CHECK) {
        err = alltoallw_param_check(comm, sendbuf, recvbuf, &scounts, &sdisps, &stypes,
                                    rcounts, rdisps, rtypes, func_name);
        if (OPAL_UNLIKELY(MPI_SUCCESS != err)) {
            return err;
        }
    }

    ompi_coll_args_t coll_args;
    ompi_coll_args_alltoallw(&coll_args, sendbuf, scounts, sdisps, stypes,
                             recvbuf, rcounts, rdisps, rtypes);
    err = comm->c_coll->coll_ialltoallw(&coll_args, comm, request,
                                        comm->c_coll->coll_ialltoallw_module);
    if (OPAL_LIKELY(OMPI_SUCCESS == err)) {
        ((ompi_coll_base_nbc_request_t *) *request)->args = coll_args;
        ompi_coll_base_retain_datatypes_w(*request,
                                          (MPI_IN_PLACE == sendbuf) ? OMPI_DATATYPE_ARRAY_NULL : stypes,
                                          rtypes, false);
    }
    OMPI_ERRHANDLER_RETURN(err, comm, err, func_name);
}

int ompi_alltoallw_init_dispatch(const void *sendbuf, ompi_count_array_t scounts,
                                 ompi_disp_array_t sdisps, ompi_datatype_array_t stypes,
                                 void *recvbuf, ompi_count_array_t rcounts,
                                 ompi_disp_array_t rdisps, ompi_datatype_array_t rtypes,
                                 ompi_communicator_t *comm, ompi_info_t *info,
                                 ompi_request_t **request, const char *func_name)
{
    int err;

    SPC_RECORD(OMPI_SPC_ALLTOALLW_INIT, 1);

    MEMCHECKER(
        ptrdiff_t recv_ext;
        ptrdiff_t send_ext;
        int i;
        int size;
        memchecker_comm(comm);
        size = OMPI_COMM_IS_INTER(comm) ? ompi_comm_remote_size(comm) : ompi_comm_size(comm);
        for (i = 0; i < size; i++) {
            if (MPI_IN_PLACE != sendbuf) {
                struct ompi_datatype_t *stype = ompi_datatype_array_get(stypes, i);
                memchecker_datatype(stype);
                ompi_datatype_type_extent(stype, &send_ext);
                memchecker_call(&opal_memchecker_base_isdefined,
                                (char *) sendbuf + ompi_disp_array_get(sdisps, i) * send_ext,
                                ompi_count_array_get(scounts, i), stype);
            }
            struct ompi_datatype_t *rtype = ompi_datatype_array_get(rtypes, i);
            memchecker_datatype(rtype);
            ompi_datatype_type_extent(rtype, &recv_ext);
            memchecker_call(&opal_memchecker_base_isaddressable,
                            (char *) recvbuf + ompi_disp_array_get(rdisps, i) * recv_ext,
                            ompi_count_array_get(rcounts, i), rtype);
        }
    );

    if (MPI_PARAM_CHECK) {
        err = alltoallw_param_check(comm, sendbuf, recvbuf, &scounts, &sdisps, &stypes,
                                    rcounts, rdisps, rtypes, func_name);
        if (OPAL_UNLIKELY(MPI_SUCCESS != err)) {
            return err;
        }
    }

    ompi_coll_args_t coll_args;
    ompi_coll_args_alltoallw(&coll_args, sendbuf, scounts, sdisps, stypes,
                             recvbuf, rcounts, rdisps, rtypes);
    err = comm->c_coll->coll_alltoallw_init(&coll_args, comm, info, request,
                                            comm->c_coll->coll_alltoallw_init_module);
    if (OPAL_LIKELY(OMPI_SUCCESS == err)) {
        ((ompi_coll_base_nbc_request_t *) *request)->args = coll_args;
        ompi_coll_base_retain_datatypes_w(*request,
                                          (MPI_IN_PLACE == sendbuf) ? OMPI_DATATYPE_ARRAY_NULL : stypes,
                                          rtypes, false);
    }
    OMPI_ERRHANDLER_RETURN(err, comm, err, func_name);
}

int ompi_neighbor_alltoallw_dispatch(const void *sendbuf, ompi_count_array_t scounts,
                                     ompi_disp_array_t sdisps, ompi_datatype_array_t stypes,
                                     void *recvbuf, ompi_count_array_t rcounts,
                                     ompi_disp_array_t rdisps, ompi_datatype_array_t rtypes,
                                     ompi_communicator_t *comm, const char *func_name)
{
    int err;

    SPC_RECORD(OMPI_SPC_NEIGHBOR_ALLTOALLW, 1);

    MEMCHECKER(
        ptrdiff_t recv_ext;
        ptrdiff_t send_ext;
        int i;
        int indegree;
        int outdegree;
        memchecker_comm(comm);
        err = mca_topo_base_neighbor_count(comm, &indegree, &outdegree);
        if (MPI_SUCCESS == err) {
            if (MPI_IN_PLACE != sendbuf) {
                for (i = 0; i < outdegree; i++) {
                    struct ompi_datatype_t *stype = ompi_datatype_array_get(stypes, i);
                    memchecker_datatype(stype);
                    ompi_datatype_type_extent(stype, &send_ext);
                    memchecker_call(&opal_memchecker_base_isdefined,
                                    (char *) sendbuf + ompi_disp_array_get(sdisps, i) * send_ext,
                                    ompi_count_array_get(scounts, i), stype);
                }
            }
            for (i = 0; i < indegree; i++) {
                struct ompi_datatype_t *rtype = ompi_datatype_array_get(rtypes, i);
                memchecker_datatype(rtype);
                ompi_datatype_type_extent(rtype, &recv_ext);
                memchecker_call(&opal_memchecker_base_isaddressable,
                                (char *) recvbuf + ompi_disp_array_get(sdisps, i) * recv_ext,
                                ompi_count_array_get(rcounts, i), rtype);
            }
        }
    );

    if (MPI_PARAM_CHECK) {
        err = neighbor_alltoallw_param_check(comm, sendbuf, recvbuf, scounts, sdisps, stypes,
                                             rcounts, rdisps, rtypes, func_name);
        if (OPAL_UNLIKELY(MPI_SUCCESS != err)) {
            return err;
        }
    }

#if OPAL_ENABLE_FT_MPI
    if (OPAL_UNLIKELY(!ompi_comm_iface_coll_check(comm, &err))) {
        OMPI_ERRHANDLER_RETURN(err, comm, err, func_name);
    }
#endif

    ompi_coll_args_t coll_args;
    ompi_coll_args_neighbor_alltoallw(&coll_args, sendbuf, scounts, sdisps, stypes,
                                      recvbuf, rcounts, rdisps, rtypes);
    err = comm->c_coll->coll_neighbor_alltoallw(&coll_args, comm,
                                                comm->c_coll->coll_neighbor_alltoallw_module);
    OMPI_ERRHANDLER_RETURN(err, comm, err, func_name);
}

int ompi_ineighbor_alltoallw_dispatch(const void *sendbuf, ompi_count_array_t scounts,
                                      ompi_disp_array_t sdisps, ompi_datatype_array_t stypes,
                                      void *recvbuf, ompi_count_array_t rcounts,
                                      ompi_disp_array_t rdisps, ompi_datatype_array_t rtypes,
                                      ompi_communicator_t *comm, ompi_request_t **request,
                                      const char *func_name)
{
    int err;

    SPC_RECORD(OMPI_SPC_INEIGHBOR_ALLTOALLW, 1);

    MEMCHECKER(
        ptrdiff_t recv_ext;
        ptrdiff_t send_ext;
        int i;
        int indegree;
        int outdegree;
        memchecker_comm(comm);
        err = mca_topo_base_neighbor_count(comm, &indegree, &outdegree);
        if (MPI_SUCCESS == err) {
            if (MPI_IN_PLACE != sendbuf) {
                for (i = 0; i < outdegree; i++) {
                    struct ompi_datatype_t *stype = ompi_datatype_array_get(stypes, i);
                    memchecker_datatype(stype);
                    ompi_datatype_type_extent(stype, &send_ext);
                    memchecker_call(&opal_memchecker_base_isdefined,
                                    (char *) sendbuf + ompi_disp_array_get(sdisps, i) * send_ext,
                                    ompi_count_array_get(scounts, i), stype);
                }
            }
            for (i = 0; i < indegree; i++) {
                struct ompi_datatype_t *rtype = ompi_datatype_array_get(rtypes, i);
                memchecker_datatype(rtype);
                ompi_datatype_type_extent(rtype, &recv_ext);
                memchecker_call(&opal_memchecker_base_isaddressable,
                                (char *) recvbuf + ompi_disp_array_get(sdisps, i) * recv_ext,
                                ompi_count_array_get(rcounts, i), rtype);
            }
        }
    );

    if (MPI_PARAM_CHECK) {
        err = neighbor_alltoallw_param_check(comm, sendbuf, recvbuf, scounts, sdisps, stypes,
                                             rcounts, rdisps, rtypes, func_name);
        if (OPAL_UNLIKELY(MPI_SUCCESS != err)) {
            return err;
        }
    }

    ompi_coll_args_t coll_args;
    ompi_coll_args_neighbor_alltoallw(&coll_args, sendbuf, scounts, sdisps, stypes,
                                      recvbuf, rcounts, rdisps, rtypes);
    err = comm->c_coll->coll_ineighbor_alltoallw(&coll_args, comm, request,
                                                 comm->c_coll->coll_ineighbor_alltoallw_module);
    if (OPAL_LIKELY(OMPI_SUCCESS == err)) {
        ((ompi_coll_base_nbc_request_t *) *request)->args = coll_args;
        ompi_coll_base_retain_datatypes_w(*request, stypes, rtypes, true);
    }
    OMPI_ERRHANDLER_RETURN(err, comm, err, func_name);
}

int ompi_neighbor_alltoallw_init_dispatch(const void *sendbuf, ompi_count_array_t scounts,
                                          ompi_disp_array_t sdisps, ompi_datatype_array_t stypes,
                                          void *recvbuf, ompi_count_array_t rcounts,
                                          ompi_disp_array_t rdisps, ompi_datatype_array_t rtypes,
                                          ompi_communicator_t *comm, ompi_info_t *info,
                                          ompi_request_t **request, const char *func_name)
{
    int err;

    SPC_RECORD(OMPI_SPC_NEIGHBOR_ALLTOALLW_INIT, 1);

    MEMCHECKER(
        ptrdiff_t recv_ext;
        ptrdiff_t send_ext;
        int i;
        int indegree;
        int outdegree;
        memchecker_comm(comm);
        err = mca_topo_base_neighbor_count(comm, &indegree, &outdegree);
        if (MPI_SUCCESS == err) {
            if (MPI_IN_PLACE != sendbuf) {
                for (i = 0; i < outdegree; i++) {
                    struct ompi_datatype_t *stype = ompi_datatype_array_get(stypes, i);
                    memchecker_datatype(stype);
                    ompi_datatype_type_extent(stype, &send_ext);
                    memchecker_call(&opal_memchecker_base_isdefined,
                                    (char *) sendbuf + ompi_disp_array_get(sdisps, i) * send_ext,
                                    ompi_count_array_get(scounts, i), stype);
                }
            }
            for (i = 0; i < indegree; i++) {
                struct ompi_datatype_t *rtype = ompi_datatype_array_get(rtypes, i);
                memchecker_datatype(rtype);
                ompi_datatype_type_extent(rtype, &recv_ext);
                memchecker_call(&opal_memchecker_base_isaddressable,
                                (char *) recvbuf + ompi_disp_array_get(sdisps, i) * recv_ext,
                                ompi_count_array_get(rcounts, i), rtype);
            }
        }
    );

    if (MPI_PARAM_CHECK) {
        err = neighbor_alltoallw_param_check(comm, sendbuf, recvbuf, scounts, sdisps, stypes,
                                             rcounts, rdisps, rtypes, func_name);
        if (OPAL_UNLIKELY(MPI_SUCCESS != err)) {
            return err;
        }
    }

    ompi_coll_args_t coll_args;
    ompi_coll_args_neighbor_alltoallw(&coll_args, sendbuf, scounts, sdisps, stypes,
                                      recvbuf, rcounts, rdisps, rtypes);
    err = comm->c_coll->coll_neighbor_alltoallw_init(&coll_args, comm, info, request,
                                                     comm->c_coll->coll_neighbor_alltoallw_init_module);
    if (OPAL_LIKELY(OMPI_SUCCESS == err)) {
        ((ompi_coll_base_nbc_request_t *) *request)->args = coll_args;
        ompi_coll_base_retain_datatypes_w(*request, stypes, rtypes, true);
    }
    OMPI_ERRHANDLER_RETURN(err, comm, err, func_name);
}
