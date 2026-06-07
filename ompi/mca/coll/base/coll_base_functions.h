/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Sun Microsystems, Inc.  All rights reserved.
 * Copyright (c) 2008      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013-2016 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2015-2018 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2016-2017 IBM Corporation.  All rights reserved.
 * Copyright (c) 2017      FUJITSU LIMITED.  All rights reserved.
 * Copyright (c) 2019      Mellanox Technologies. All rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_BASE_EXPORT_H
#define MCA_COLL_BASE_EXPORT_H

#include "ompi_config.h"

#include "ompi/mca/coll/base/base.h"
#include "ompi/mca/mca.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/info/info.h"
#include "ompi/request/request.h"

/* need to include our own topo prototypes so we can malloc data on the comm correctly */
#include "coll_base_topo.h"

/* some fixed value index vars to simplify certain operations */
typedef enum COLLTYPE {
    ALLGATHER = 0,       /*  0 */
    ALLGATHERV,          /*  1 */
    ALLREDUCE,           /*  2 */
    ALLTOALL,            /*  3 */
    ALLTOALLV,           /*  4 */
    ALLTOALLW,           /*  5 */
    BARRIER,             /*  6 */
    BCAST,               /*  7 */
    EXSCAN,              /*  8 */
    GATHER,              /*  9 */
    GATHERV,             /* 10 */
    REDUCE,              /* 11 */
    REDUCESCATTER,       /* 12 */
    REDUCESCATTERBLOCK,  /* 13 */
    SCAN,                /* 14 */
    SCATTER,             /* 15 */
    SCATTERV,            /* 16 */
    NEIGHBOR_ALLGATHER,  /* 17 */
    NEIGHBOR_ALLGATHERV, /* 18 */
    NEIGHBOR_ALLTOALL,   /* 19 */
    NEIGHBOR_ALLTOALLV,  /* 20 */
    NEIGHBOR_ALLTOALLW,  /* 21 */
    COLLCOUNT            /* 22 end counter keep it as last element */
} COLLTYPE_T;


BEGIN_C_DECLS

/* All Gather */
int ompi_coll_base_allgather_intra_recursivedoubling(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allgather_intra_sparbit(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allgather_intra_ring(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allgather_intra_neighborexchange(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allgather_intra_basic_linear(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allgather_intra_two_procs(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allgather_intra_k_bruck(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, int radix);
int ompi_coll_base_allgather_direct_messaging(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

/* All GatherV */
int ompi_coll_base_allgatherv_intra_bruck(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allgatherv_intra_sparbit(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allgatherv_intra_ring(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allgatherv_intra_neighborexchange(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allgatherv_intra_basic_default(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allgatherv_intra_two_procs(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

/* All Reduce */
int ompi_coll_base_allreduce_intra_nonoverlapping(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allreduce_intra_recursivedoubling(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allreduce_intra_ring(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allreduce_intra_ring_segmented(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize);
int ompi_coll_base_allreduce_intra_basic_linear(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allreduce_intra_redscat_allgather(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_allreduce_intra_allgather_reduce(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

/* AlltoAll */
int ompi_coll_base_alltoall_intra_pairwise(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_alltoall_intra_bruck(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_alltoall_intra_basic_linear(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_alltoall_intra_linear_sync(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, int max_requests);
int ompi_coll_base_alltoall_intra_two_procs(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_base_alltoall_intra_basic_inplace(const void *rbuf, size_t rcount,
                                               struct ompi_datatype_t *rdtype,
                                               struct ompi_communicator_t *comm,
                                               mca_coll_base_module_t *module);  /* special version for INPLACE */

/* AlltoAllV */
int ompi_coll_base_alltoallv_intra_pairwise(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_alltoallv_intra_basic_linear(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_base_alltoallv_intra_basic_inplace(const void *rbuf, ompi_count_array_t rcounts, ompi_disp_array_t rdisps,
                                                struct ompi_datatype_t *rdtype,
                                                struct ompi_communicator_t *comm,
                                                mca_coll_base_module_t *module);  /* special version for INPLACE */

/* AlltoAllW */

/* Barrier */
int ompi_coll_base_barrier_intra_doublering(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_barrier_intra_recursivedoubling(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_barrier_intra_bruck(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_barrier_intra_two_procs(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_barrier_intra_tree(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_barrier_intra_basic_linear(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

/* Bcast */
int ompi_coll_base_bcast_intra_generic(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t count_by_segment, ompi_coll_tree_t* tree);
int ompi_coll_base_bcast_intra_basic_linear(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_bcast_intra_chain(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize, int32_t chains);
int ompi_coll_base_bcast_intra_pipeline(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize);
int ompi_coll_base_bcast_intra_binomial(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize);
int ompi_coll_base_bcast_intra_bintree(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize);
int ompi_coll_base_bcast_intra_split_bintree(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize);
int ompi_coll_base_bcast_intra_knomial(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize, int radix);
int ompi_coll_base_bcast_intra_scatter_allgather(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize);
int ompi_coll_base_bcast_intra_scatter_allgather_ring(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize);

/* Exscan */
int ompi_coll_base_exscan_intra_recursivedoubling(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_exscan_intra_linear(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_exscan_intra_recursivedoubling(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

/* Gather */
int ompi_coll_base_gather_intra_basic_linear(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_gather_intra_binomial(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_gather_intra_linear_sync(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, int first_segment_size);

/* GatherV */

/* Reduce */
int ompi_coll_base_reduce_generic(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, ompi_coll_tree_t* tree, size_t count_by_segment, int max_outstanding_reqs);
int ompi_coll_base_reduce_intra_basic_linear(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_reduce_intra_chain(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize, int fanout, int max_outstanding_reqs );
int ompi_coll_base_reduce_intra_pipeline(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize, int max_outstanding_reqs );
int ompi_coll_base_reduce_intra_binary(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize, int max_outstanding_reqs );
int ompi_coll_base_reduce_intra_binomial(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize, int max_outstanding_reqs );
int ompi_coll_base_reduce_intra_in_order_binary(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize, int max_outstanding_reqs );
int ompi_coll_base_reduce_intra_redscat_gather(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_reduce_intra_knomial(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize, int max_outstanding_reqs, int radix);

/* Reduce_scatter */
int ompi_coll_base_reduce_scatter_intra_nonoverlapping(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_reduce_scatter_intra_basic_recursivehalving(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_reduce_scatter_intra_ring(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_reduce_scatter_intra_butterfly(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

/* Reduce_scatter_block */
int ompi_coll_base_reduce_scatter_block_basic_linear(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_reduce_scatter_block_intra_recursivedoubling(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_reduce_scatter_block_intra_recursivehalving(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_reduce_scatter_block_intra_butterfly(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

/* Scan */
int ompi_coll_base_scan_intra_recursivedoubling(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_scan_intra_linear(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_scan_intra_recursivedoubling(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

/* Scatter */
int ompi_coll_base_scatter_intra_basic_linear(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_scatter_intra_binomial(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_scatter_intra_linear_nb(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, int max_reqs);

/* ScatterV */

/* Reduce_local: inbuf/inoutbuf/count/dtype in args->src/dst.info, op in
 * args->op. comm is unused (callers pass comm_self). */
int mca_coll_base_reduce_local(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_base_revoke_local(struct ompi_communicator_t *comm);

#if OPAL_ENABLE_FT_MPI
/* Agreement: contrib/dt_count/dt live in args->src.info, op in args->op,
 * the failed group in args->failedgroup, and update_grp is conveyed by
 * OMPI_COLL_ARGS_FLAG_UPDATE_FAILEDGROUP in args->flags. */
int ompi_coll_base_agree_noft(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_base_iagree_noft(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request,
                               mca_coll_base_module_t *module);
#endif /* OPAL_ENABLE_FT_MPI */

END_C_DECLS

#define COLL_BASE_UPDATE_BINTREE( OMPI_COMM, BASE_MODULE, ROOT )	\
do {                                                                                       \
    mca_coll_base_comm_t* coll_comm = (BASE_MODULE)->base_data;                        \
    if( !( (coll_comm->cached_bintree)                                                     \
           && (coll_comm->cached_bintree_root == (ROOT)) ) ) {                             \
        if( coll_comm->cached_bintree ) { /* destroy previous binomial if defined */       \
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_bintree) );             \
        }                                                                                  \
        coll_comm->cached_bintree = ompi_coll_base_topo_build_tree(2,(OMPI_COMM),(ROOT)); \
        coll_comm->cached_bintree_root = (ROOT);                                           \
    }                                                                                      \
} while (0)

#define COLL_BASE_UPDATE_BMTREE( OMPI_COMM, BASE_MODULE, ROOT )	\
do {                                                                                         \
    mca_coll_base_comm_t* coll_comm = (BASE_MODULE)->base_data;                           \
    if( !( (coll_comm->cached_bmtree)                                                        \
           && (coll_comm->cached_bmtree_root == (ROOT)) ) ) {                                \
        if( coll_comm->cached_bmtree ) { /* destroy previous binomial if defined */          \
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_bmtree) );                \
        }                                                                                    \
        coll_comm->cached_bmtree = ompi_coll_base_topo_build_bmtree( (OMPI_COMM), (ROOT) ); \
        coll_comm->cached_bmtree_root = (ROOT);                                              \
    }                                                                                        \
} while (0)

#define COLL_BASE_UPDATE_IN_ORDER_BMTREE( OMPI_COMM, BASE_MODULE, ROOT ) \
do {                                                                                         \
    mca_coll_base_comm_t* coll_comm = (BASE_MODULE)->base_data;                           \
    if( !( (coll_comm->cached_in_order_bmtree)                                               \
           && (coll_comm->cached_in_order_bmtree_root == (ROOT)) ) ) {                       \
        if( coll_comm->cached_in_order_bmtree ) { /* destroy previous binomial if defined */ \
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_in_order_bmtree) );       \
        }                                                                                    \
        coll_comm->cached_in_order_bmtree = ompi_coll_base_topo_build_in_order_bmtree( (OMPI_COMM), (ROOT) ); \
        coll_comm->cached_in_order_bmtree_root = (ROOT);                                     \
    }                                                                                        \
} while (0)

#define COLL_BASE_UPDATE_KMTREE(OMPI_COMM, BASE_MODULE, ROOT, RADIX)	\
do {                                                                                         \
    mca_coll_base_comm_t* coll_comm = (BASE_MODULE)->base_data;                           \
    if (!((coll_comm->cached_kmtree)                                                       \
           && (coll_comm->cached_kmtree_root == (ROOT))                                     \
           && (coll_comm->cached_kmtree_radix == (RADIX))))                                   \
    {                                                                                        \
        if (coll_comm->cached_kmtree ) { /* destroy previous k-nomial tree if defined */     \
            ompi_coll_base_topo_destroy_tree(&(coll_comm->cached_kmtree));                  \
        }                                                                                    \
        coll_comm->cached_kmtree = ompi_coll_base_topo_build_kmtree((OMPI_COMM), (ROOT), (RADIX)); \
        coll_comm->cached_kmtree_root = (ROOT);                                              \
        coll_comm->cached_kmtree_radix = (RADIX);                                              \
    }                                                                                        \
} while (0)

#define COLL_BASE_UPDATE_PIPELINE( OMPI_COMM, BASE_MODULE, ROOT )	\
do {                                                                                             \
    mca_coll_base_comm_t* coll_comm = (BASE_MODULE)->base_data;                               \
    if( !( (coll_comm->cached_pipeline)                                                          \
           && (coll_comm->cached_pipeline_root == (ROOT)) ) ) {                                  \
        if (coll_comm->cached_pipeline) { /* destroy previous pipeline if defined */             \
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_pipeline) );                  \
        }                                                                                        \
        coll_comm->cached_pipeline = ompi_coll_base_topo_build_chain( 1, (OMPI_COMM), (ROOT) ); \
        coll_comm->cached_pipeline_root = (ROOT);                                                \
    }                                                                                            \
} while (0)

#define COLL_BASE_UPDATE_CHAIN( OMPI_COMM, BASE_MODULE, ROOT, FANOUT )	\
do {                                                                                             \
    mca_coll_base_comm_t* coll_comm = (BASE_MODULE)->base_data;                               \
    if( !( (coll_comm->cached_chain)                                                             \
           && (coll_comm->cached_chain_root == (ROOT))                                           \
           && (coll_comm->cached_chain_fanout == (FANOUT)) ) ) {                                 \
        if( coll_comm->cached_chain) { /* destroy previous chain if defined */                   \
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_chain) );                     \
        }                                                                                        \
        coll_comm->cached_chain = ompi_coll_base_topo_build_chain((FANOUT), (OMPI_COMM), (ROOT)); \
        coll_comm->cached_chain_root = (ROOT);                                                   \
        coll_comm->cached_chain_fanout = (FANOUT);                                               \
    }                                                                                            \
} while (0)

#define COLL_BASE_UPDATE_IN_ORDER_BINTREE( OMPI_COMM, BASE_MODULE )	\
do {                                                                           \
    mca_coll_base_comm_t* coll_comm = (BASE_MODULE)->base_data;             \
    if( !(coll_comm->cached_in_order_bintree) ) {                              \
        /* In-order binary tree topology is defined by communicator size */    \
        /* Thus, there is no need to destroy anything */                       \
        coll_comm->cached_in_order_bintree =                                   \
        ompi_coll_base_topo_build_in_order_bintree((OMPI_COMM)); \
    }                                                                          \
} while (0)

/**
 * This macro gives a generic way to compute the best count of
 * the segment (i.e. the number of complete datatypes that
 * can fit in the specified SEGSIZE). Beware, when this macro
 * is called, the SEGCOUNT should be initialized to the count as
 * expected by the collective call.
 */
#define COLL_BASE_COMPUTED_SEGCOUNT(SEGSIZE, TYPELNG, SEGCOUNT)        \
    if( ((SEGSIZE) >= (TYPELNG)) &&                                     \
        ((SEGSIZE) < ((TYPELNG) * (SEGCOUNT))) ) {                      \
        size_t residual;                                                \
        (SEGCOUNT) = (int)((SEGSIZE) / (TYPELNG));                      \
        residual = (SEGSIZE) - (SEGCOUNT) * (TYPELNG);                  \
        if( residual > ((TYPELNG) >> 1) )                               \
            (SEGCOUNT)++;                                               \
    }                                                                   \

/**
 * This macro gives a generic way to compute the well distributed block counts
 * when the count and number of blocks are fixed.
 * Macro returns "early-block" count, "late-block" count, and "split-index"
 * which is the block at which we switch from "early-block" count to
 * the "late-block" count.
 * count = split_index * early_block_count +
 *         (block_count - split_index) * late_block_count
 * We do not perform ANY error checks - make sure that the input values
 * make sense (eg. count > num_blocks).
 */
#define COLL_BASE_COMPUTE_BLOCKCOUNT( COUNT, NUM_BLOCKS, SPLIT_INDEX,       \
                                       EARLY_BLOCK_COUNT, LATE_BLOCK_COUNT ) \
    EARLY_BLOCK_COUNT = LATE_BLOCK_COUNT = COUNT / NUM_BLOCKS;               \
    SPLIT_INDEX = COUNT % NUM_BLOCKS;                                        \
    if (0 != SPLIT_INDEX) {                                                  \
        EARLY_BLOCK_COUNT = EARLY_BLOCK_COUNT + 1;                           \
    }                                                                        \

/*
 * Data structure for hanging data off the communicator
 * i.e. per module instance
 */
struct mca_coll_base_comm_t {
    opal_object_t super;

    /* standard data for requests and PML usage */

    /* Precreate space for requests
     * Note this does not effect basic,
     * but if in wrong context can confuse a debugger
     * this is controlled by an MCA param
     */

    ompi_request_t **mcct_reqs;
    int mcct_num_reqs;

    /*
     * base topo information caching per communicator
     *
     * for each communicator we cache the topo information so we can
     * reuse without regenerating if we change the root, [or fanout]
     * then regenerate and recache this information
     */

    /* general tree with n fan out */
    ompi_coll_tree_t *cached_ntree;
    int cached_ntree_root;
    int cached_ntree_fanout;

    /* binary tree */
    ompi_coll_tree_t *cached_bintree;
    int cached_bintree_root;

    /* binomial tree */
    ompi_coll_tree_t *cached_bmtree;
    int cached_bmtree_root;

    /* binomial tree */
    ompi_coll_tree_t *cached_in_order_bmtree;
    int cached_in_order_bmtree_root;

    /* k-nomial tree */
    ompi_coll_tree_t *cached_kmtree;
    int cached_kmtree_root;
    int cached_kmtree_radix;

    /* chained tree (fanout followed by pipelines) */
    ompi_coll_tree_t *cached_chain;
    int cached_chain_root;
    int cached_chain_fanout;

    /* pipeline */
    ompi_coll_tree_t *cached_pipeline;
    int cached_pipeline_root;

    /* in-order binary tree (root of the in-order binary tree is rank 0) */
    ompi_coll_tree_t *cached_in_order_bintree;
};
typedef struct mca_coll_base_comm_t mca_coll_base_comm_t;
OMPI_DECLSPEC OBJ_CLASS_DECLARATION(mca_coll_base_comm_t);

/**
 * Free all requests in an array. As these requests are usually used during
 * collective communications, and as on a successful collective they are
 * expected to be released during the corresponding wait, the array should
 * generally be empty. However, this function might be used on error conditions
 * where it will allow a correct cleanup.
 */
static inline void ompi_coll_base_free_reqs(ompi_request_t **reqs, int count)
{
    if (OPAL_UNLIKELY(NULL == reqs)) {
        return;
    }

    for (int i = 0; i < count; ++i) {
        if( MPI_REQUEST_NULL != reqs[i] ) {
#if OPAL_ENABLE_FT_MPI
            if( MPI_ERR_PROC_FAILED == reqs[i]->req_status.MPI_ERROR
             || MPI_ERR_PROC_FAILED_PENDING == reqs[i]->req_status.MPI_ERROR
             || MPI_ERR_REVOKED == reqs[i]->req_status.MPI_ERROR ) {
                /* We cannot just 'free' and forget, as the PML/BTLS would still
                 * be updating the request buffer after we return from the MPI
                 * call!
                 * For other errors that do not have a well defined post-error
                 * behavior, calling the cancel/wait could deadlock, so we just
                 * free, as this is the best that can be done in this case. */
                ompi_request_cancel(reqs[i]);
                ompi_request_wait(&reqs[i], MPI_STATUS_IGNORE);
            } else /* this 'else' intentionally spills outside the ifdef */
#endif /* OPAL_ENABLE_FT_MPI */
            ompi_request_free(&reqs[i]);
        }
    }
}

/**
 * Return the array of requests on the data. If the array was not initialized
 * or if its size was too small, allocate it to fit the requested size.
 */
ompi_request_t** ompi_coll_base_comm_get_reqs(mca_coll_base_comm_t* data, int nreqs);

#endif /* MCA_COLL_BASE_EXPORT_H */
