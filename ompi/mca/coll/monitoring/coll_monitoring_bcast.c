/*
 * Copyright (c) 2016-2018 Inria. All rights reserved.
 * Copyright (c) 2017-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "ompi/request/request.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "coll_monitoring.h"

int mca_coll_monitoring_bcast(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_monitoring_module_t*monitoring_module = (mca_coll_monitoring_module_t*) module;
    size_t type_size, data_size;
    const int comm_size = ompi_comm_size(comm);
    ompi_datatype_type_size(args->src.info.datatype, &type_size);
    data_size = args->src.info.count * type_size;
    if( args->root == ompi_comm_rank(comm) ) {
        int i, rank;
        mca_common_monitoring_coll_o2a(data_size * (comm_size - 1), monitoring_module->data);
        for( i = 0; i < comm_size; ++i ) {
            if( i == args->root ) continue; /* No self sending */
            /**
             * If this fails the destination is not part of my MPI_COM_WORLD
             * Lookup its name in the rank hashtable to get its MPI_COMM_WORLD rank
             */
            if( OPAL_SUCCESS == mca_common_monitoring_get_world_rank(i, comm->c_remote_group, &rank) ) {
                mca_common_monitoring_record_coll(rank, data_size);
            }
        }
    }
    return monitoring_module->real.coll_bcast(args, comm, monitoring_module->real.coll_bcast_module);
}

int mca_coll_monitoring_ibcast(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module)
{
    mca_coll_monitoring_module_t*monitoring_module = (mca_coll_monitoring_module_t*) module;
    size_t type_size, data_size;
    const int comm_size = ompi_comm_size(comm);
    ompi_datatype_type_size(args->src.info.datatype, &type_size);
    data_size = args->src.info.count * type_size;
    if( args->root == ompi_comm_rank(comm) ) {
        int i, rank;
        mca_common_monitoring_coll_o2a(data_size * (comm_size - 1), monitoring_module->data);
        for( i = 0; i < comm_size; ++i ) {
            if( i == args->root ) continue; /* No self sending */
            /**
             * If this fails the destination is not part of my MPI_COM_WORLD
             * Lookup its name in the rank hashtable to get its MPI_COMM_WORLD rank
             */
            if( OPAL_SUCCESS == mca_common_monitoring_get_world_rank(i, comm->c_remote_group, &rank) ) {
                mca_common_monitoring_record_coll(rank, data_size);
            }
        }
    }
    return monitoring_module->real.coll_ibcast(args, comm, request, monitoring_module->real.coll_ibcast_module);
}
