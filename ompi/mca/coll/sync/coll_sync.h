/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2006 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008-2009 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2024      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_SYNC_EXPORT_H
#define MCA_COLL_SYNC_EXPORT_H

#include "ompi_config.h"

#include "mpi.h"

#include "opal/class/opal_object.h"
#include "opal/mca/mca.h"
#include "opal/util/output.h"

#include "ompi/constants.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/communicator/communicator.h"

BEGIN_C_DECLS

/* API functions */

int mca_coll_sync_init_query(bool enable_progress_threads,
                             bool enable_mpi_threads);
mca_coll_base_module_t
*mca_coll_sync_comm_query(struct ompi_communicator_t *comm,
                          int *priority);

int mca_coll_sync_barrier(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_sync_bcast(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_sync_exscan(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_sync_gather(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_sync_gatherv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_sync_reduce(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_sync_reduce_scatter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_sync_scan(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_sync_scatter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_sync_scatterv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

/* Types */
/* Module */

typedef struct mca_coll_sync_module_t {
    mca_coll_base_module_t super;

    /* Pointers to all the "real" collective functions */
    mca_coll_base_comm_coll_t c_coll;

    /* How many ops we've executed */
    int before_num_operations;

    /* How many ops we've executed (it's easier to have 2) */
    int after_num_operations;

    /* Avoid recursion of syncs */
    bool in_operation;
} mca_coll_sync_module_t;

OBJ_CLASS_DECLARATION(mca_coll_sync_module_t);

/* Component */

typedef struct mca_coll_sync_component_t {
    mca_coll_base_component_4_0_0_t super;

    /* Priority of this component */
    int priority;

    /* Do a sync *before* each Nth collective */
    int barrier_before_nops;

    /* Do a sync *after* each Nth collective */
    int barrier_after_nops;
} mca_coll_sync_component_t;

/* Globally exported variables */

OMPI_DECLSPEC extern mca_coll_sync_component_t mca_coll_sync_component;

/* Macro used in most of the collectives */

#define COLL_SYNC(m, op) \
do { \
    int err = MPI_SUCCESS; \
    ompi_coll_args_t _sync_barrier_args;                                        \
    ompi_coll_args_barrier(&_sync_barrier_args);                                \
    (m)->in_operation = true; \
    if (OPAL_UNLIKELY(++((m)->before_num_operations) ==                         \
                      mca_coll_sync_component.barrier_before_nops)) {           \
        (m)->before_num_operations = 0;                                         \
        err = (m)->c_coll.coll_barrier(&_sync_barrier_args, comm,               \
                                       (m)->c_coll.coll_barrier_module);        \
    }                                                                           \
    if (OPAL_LIKELY(MPI_SUCCESS == err)) {                                      \
        err = op;                                                               \
    }                                                                           \
    if (OPAL_UNLIKELY(++((m)->after_num_operations) ==                          \
                      mca_coll_sync_component.barrier_after_nops) &&            \
        OPAL_LIKELY(MPI_SUCCESS == err)) {                                      \
        (m)->after_num_operations = 0;                                          \
        err = (m)->c_coll.coll_barrier(&_sync_barrier_args, comm,               \
                                       (m)->c_coll.coll_barrier_module);        \
    }                                                                           \
    (m)->in_operation = false;                                                  \
    return err;                                                                 \
} while(0)

END_C_DECLS

#endif /* MCA_COLL_SYNC_EXPORT_H */
