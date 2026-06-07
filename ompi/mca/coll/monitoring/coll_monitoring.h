/*
 * Copyright (c) 2016      Inria.  All rights reserved.
 * Copyright (c) 2017-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2017      Amazon.com, Inc. or its affiliates.  All Rights
 *                         reserved.
 * Copyright (c) 2024      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_MONITORING_H
#define MCA_COLL_MONITORING_H

BEGIN_C_DECLS

#include "ompi_config.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/op/op.h"
#include "ompi/request/request.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/common/monitoring/common_monitoring.h"

struct mca_coll_monitoring_component_t {
    mca_coll_base_component_t super;
    int priority;
};
typedef struct mca_coll_monitoring_component_t mca_coll_monitoring_component_t;

OMPI_DECLSPEC extern mca_coll_monitoring_component_t mca_coll_monitoring_component;

struct mca_coll_monitoring_module_t {
    mca_coll_base_module_t super;
    mca_coll_base_comm_coll_t real;
    mca_monitoring_coll_data_t*data;
};
typedef struct mca_coll_monitoring_module_t mca_coll_monitoring_module_t;
OMPI_DECLSPEC OBJ_CLASS_DECLARATION(mca_coll_monitoring_module_t);

/* 
 * Coll interface functions
 */

/* Blocking */
extern int mca_coll_monitoring_allgather(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_allgatherv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_allreduce(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_alltoall(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_alltoallv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_alltoallw(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_barrier(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_bcast(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_exscan(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_gather(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_gatherv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_reduce(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_reduce_scatter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_reduce_scatter_block(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_scan(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_scatter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_scatterv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

/* Nonblocking */
extern int mca_coll_monitoring_iallgather(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_iallgatherv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_iallreduce(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_ialltoall(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_ialltoallv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_ialltoallw(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_ibarrier(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_ibcast(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_iexscan(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_igather(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_igatherv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_ireduce(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_ireduce_scatter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_ireduce_scatter_block(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_iscan(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_iscatter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_iscatterv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

/* Neighbor */
extern int mca_coll_monitoring_neighbor_allgather(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_neighbor_allgatherv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_neighbor_alltoall(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_neighbor_alltoallv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_neighbor_alltoallw(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_ineighbor_allgather(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_ineighbor_allgatherv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_ineighbor_alltoall(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_ineighbor_alltoallv(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

extern int mca_coll_monitoring_ineighbor_alltoallw(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

END_C_DECLS

#endif  /* MCA_COLL_MONITORING_H */
