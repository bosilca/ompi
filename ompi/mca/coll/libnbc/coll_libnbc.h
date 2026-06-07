/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013-2015 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2014-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2016-2017 IBM Corporation.  All rights reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_LIBNBC_EXPORT_H
#define MCA_COLL_LIBNBC_EXPORT_H

#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "opal/sys/atomic.h"

BEGIN_C_DECLS

/*********************** LibNBC tuning parameters ************************/

/* the debug level */
#define NBC_DLEVEL 0

/* enable schedule caching - undef NBC_CACHE_SCHEDULE to deactivate it */
/* TODO: this whole schedule cache stuff does not work with the tmbuf
 * :-( - first, the tmpbuf must not be freed if a schedule using it is
 * still in the cache and second, the tmpbuf used by the schedule must
 * be attached to the handle that uses this schedule !!!!
 * I.E., THIS IS EXPERIMENTAL AND MIGHT NOT WORK */
/* It also leaks memory because the schedule is never cleaned up when
   the communicator is destroyed, so don't use it for now */
#ifdef NBC_CACHE_SCHEDULE
#undef NBC_CACHE_SCHEDULE
#endif
#define NBC_SCHED_DICT_UPPER 1024 /* max. number of dict entries */
#define NBC_SCHED_DICT_LOWER 512  /* number of dict entries after wipe, if SCHED_DICT_UPPER is reached */

/********************* end of LibNBC tuning parameters ************************/

/* Function return codes  */
#define NBC_OK 0 /* everything went fine */
#define NBC_SUCCESS 0 /* everything went fine (MPI compliant :) */
#define NBC_OOR 1 /* out of resources */
#define NBC_BAD_SCHED 2 /* bad schedule */
#define NBC_CONTINUE 3 /* progress not done */
#define NBC_DATATYPE_NOT_SUPPORTED 4 /* datatype not supported or not valid */
#define NBC_OP_NOT_SUPPORTED 5 /* operation not supported or not valid */
#define NBC_NOT_IMPLEMENTED 6
#define NBC_INVALID_PARAM 7 /* invalid parameters */
#define NBC_INVALID_TOPOLOGY_COMM 8 /* invalid topology attached to communicator */

/* number of implemented collective functions */
#define NBC_NUM_COLL 17

extern bool libnbc_ibcast_skip_dt_decision;
extern int libnbc_iallgather_algorithm;
extern int libnbc_iallreduce_algorithm;
extern int libnbc_ibcast_algorithm;
extern int libnbc_ibcast_knomial_radix;
extern int libnbc_iexscan_algorithm;
extern int libnbc_ireduce_algorithm;
extern int libnbc_iscan_algorithm;

struct ompi_coll_libnbc_component_t {
    mca_coll_base_component_4_0_0_t super;
    opal_free_list_t requests;
    opal_list_t active_requests;
    opal_atomic_int32_t active_comms;
    opal_mutex_t lock;                /* protect access to the active_requests list */
};
typedef struct ompi_coll_libnbc_component_t ompi_coll_libnbc_component_t;

/* Globally exported variables */
OMPI_DECLSPEC extern ompi_coll_libnbc_component_t mca_coll_libnbc_component;

struct ompi_coll_libnbc_module_t {
    mca_coll_base_module_t super;
    opal_mutex_t mutex;
    bool comm_registered;
#ifdef NBC_CACHE_SCHEDULE
  void *NBC_Dict[NBC_NUM_COLL]; /* this should point to a struct
                                      hb_tree, but since this is a
                                      public header-file, this would be
                                      an include mess :-(. So let's void
                                      it ...*/
  int NBC_Dict_size[NBC_NUM_COLL];
#endif
};
typedef struct ompi_coll_libnbc_module_t ompi_coll_libnbc_module_t;
OBJ_CLASS_DECLARATION(ompi_coll_libnbc_module_t);

typedef ompi_coll_libnbc_module_t NBC_Comminfo;

struct NBC_Schedule {
    opal_object_t super;
    volatile int size;
    volatile int current_round_offset;
    char *data;
};

typedef struct NBC_Schedule NBC_Schedule;

OBJ_CLASS_DECLARATION(NBC_Schedule);

struct ompi_coll_libnbc_request_t {
    ompi_coll_base_nbc_request_t super;
    MPI_Comm comm;
    long row_offset;
    bool nbc_complete; /* status in libnbc level */
    int tag;
    volatile int req_count;
    ompi_request_t **req_array;
    NBC_Comminfo *comminfo;
    NBC_Schedule *schedule;
    void *tmpbuf; /* temporary buffer e.g. used for Reduce */
    /* TODO: we should make a handle pointer to a state later (that the user
     * can move request handles) */
};
typedef struct ompi_coll_libnbc_request_t ompi_coll_libnbc_request_t;
OBJ_CLASS_DECLARATION(ompi_coll_libnbc_request_t);

typedef ompi_coll_libnbc_request_t NBC_Handle;


#define OMPI_COLL_LIBNBC_REQUEST_ALLOC(comm, persistent, req)           \
    do {                                                                \
        opal_free_list_item_t *item;                                    \
        item = opal_free_list_wait (&mca_coll_libnbc_component.requests); \
        req = (ompi_coll_libnbc_request_t*) item;                       \
        OMPI_REQUEST_INIT(&req->super.super, persistent);               \
        req->super.super.req_mpi_object.comm = comm;                    \
    } while (0)

#define OMPI_COLL_LIBNBC_REQUEST_RETURN(req)                            \
    do {                                                                \
        OMPI_REQUEST_FINI(&(req)->super.super);                         \
        opal_free_list_return (&mca_coll_libnbc_component.requests,     \
                               (opal_free_list_item_t*) (req));         \
    } while (0)

int ompi_coll_libnbc_progress(void);

int NBC_Init_comm(MPI_Comm comm, ompi_coll_libnbc_module_t *module);
int NBC_Progress(NBC_Handle *handle);


int ompi_coll_libnbc_iallgather(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_iallgatherv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_iallreduce(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ialltoall(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ialltoallv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ialltoallw(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ibarrier(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ibcast(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_iexscan(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_igather(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_igatherv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ireduce(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ireduce_scatter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ireduce_scatter_block(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_iscan(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_iscatter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_iscatterv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);


int ompi_coll_libnbc_iallgather_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_iallgatherv_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_iallreduce_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ialltoall_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ialltoallv_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ialltoallw_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ibarrier_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ibcast_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_igather_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_igatherv_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ireduce_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ireduce_scatter_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ireduce_scatter_block_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_iscatter_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_iscatterv_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);


int ompi_coll_libnbc_ineighbor_allgather(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ineighbor_allgatherv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ineighbor_alltoall(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ineighbor_alltoallv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_ineighbor_alltoallw(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

int ompi_coll_libnbc_allgather_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_allgatherv_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_allreduce_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_alltoall_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_alltoallv_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_alltoallw_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_barrier_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_bcast_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_exscan_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_gather_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_gatherv_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_reduce_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_reduce_scatter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_reduce_scatter_block_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_scan_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_scatter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_scatterv_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);

int ompi_coll_libnbc_allgather_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_allgatherv_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_allreduce_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_alltoall_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_alltoallv_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_alltoallw_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_barrier_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_bcast_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_gather_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_gatherv_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_reduce_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_reduce_scatter_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_reduce_scatter_block_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_scatter_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_scatterv_inter_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);

int ompi_coll_libnbc_neighbor_allgather_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_neighbor_allgatherv_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_neighbor_alltoall_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_neighbor_alltoallv_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);
int ompi_coll_libnbc_neighbor_alltoallw_init(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_info_t *info, ompi_request_t **request, mca_coll_base_module_t *module);


END_C_DECLS

#endif /* MCA_COLL_LIBNBC_EXPORT_H */
