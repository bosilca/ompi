/*
 * Copyright (c) 2014-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 */

#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_base_topo.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include <math.h>

/* Bcast */
int ompi_coll_adapt_ibcast_register(void);
int ompi_coll_adapt_ibcast_fini(void);
int ompi_coll_adapt_bcast(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_adapt_ibcast(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

/* Reduce */
int ompi_coll_adapt_ireduce_register(void);
int ompi_coll_adapt_ireduce_fini(void);
int ompi_coll_adapt_reduce(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module);
int ompi_coll_adapt_ireduce(ompi_coll_args_t *args, struct ompi_communicator_t *comm, ompi_request_t **request, mca_coll_base_module_t *module);

