/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2024      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_SELF_EXPORT_H
#define MCA_COLL_SELF_EXPORT_H

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/mca/mca.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/request/request.h"

BEGIN_C_DECLS

/*
 * Globally exported variable
 */

OMPI_DECLSPEC extern const mca_coll_base_component_4_0_0_t mca_coll_self_component;
extern int ompi_coll_self_priority;

/*
 * coll API functions
 */


  /* API functions */

int mca_coll_self_init_query(bool enable_progress_threads,
                             bool enable_mpi_threads);
mca_coll_base_module_t *
mca_coll_self_comm_query(struct ompi_communicator_t *comm, int *priority);

int mca_coll_self_allgather_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_allgatherv_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_allreduce_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_alltoall_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_alltoallv_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_alltoallw_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_barrier_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_bcast_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_exscan_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_gather_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_gatherv_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_reduce_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_reduce_scatter_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_scan_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_scatter_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int mca_coll_self_scatterv_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);


struct mca_coll_self_module_t {
    mca_coll_base_module_t super;
};
typedef struct mca_coll_self_module_t mca_coll_self_module_t;
OBJ_CLASS_DECLARATION(mca_coll_self_module_t);


END_C_DECLS

#endif /* MCA_COLL_SELF_EXPORT_H */
