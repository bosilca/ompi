/*
 * Copyright (c) 2014-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2014      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2019      Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include <string.h>
#include <stdio.h>

#include "coll_cuda.h"

#include "mpi.h"

#include "opal/util/show_help.h"
#include "ompi/mca/rte/rte.h"

#include "ompi/constants.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"
#include "coll_cuda.h"

static int
mca_coll_cuda_module_enable(mca_coll_base_module_t *module,
                            struct ompi_communicator_t *comm);
static int
mca_coll_cuda_module_disable(mca_coll_base_module_t *module,
                             struct ompi_communicator_t *comm);

static void mca_coll_cuda_module_construct(mca_coll_cuda_module_t *module)
{
    memset(&(module->c_coll), 0, sizeof(module->c_coll));
}

OBJ_CLASS_INSTANCE(mca_coll_cuda_module_t, mca_coll_base_module_t,
                   mca_coll_cuda_module_construct,
                   NULL);


/*
 * Initial query function that is invoked during MPI_INIT, allowing
 * this component to disqualify itself if it doesn't support the
 * required level of thread support.
 */
int mca_coll_cuda_init_query(bool enable_progress_threads,
                             bool enable_mpi_threads)
{
    /* Nothing to do */

    return OMPI_SUCCESS;
}


/*
 * Invoked when there's a new communicator that has been created.
 * Look at the communicator and decide which set of functions and
 * priority we want to return.
 */
mca_coll_base_module_t *
mca_coll_cuda_comm_query(struct ompi_communicator_t *comm,
                         int *priority)
{
    mca_coll_cuda_module_t *cuda_module;

    cuda_module = OBJ_NEW(mca_coll_cuda_module_t);
    if (NULL == cuda_module) {
        return NULL;
    }

    *priority = mca_coll_cuda_component.priority;

    /* Choose whether to use [intra|inter] */
    cuda_module->super.coll_module_enable  = mca_coll_cuda_module_enable;
    cuda_module->super.coll_module_disable = mca_coll_cuda_module_disable;
    cuda_module->super.ft_event = NULL;

    cuda_module->super.coll_allreduce = mca_coll_cuda_allreduce;
    cuda_module->super.coll_reduce = mca_coll_cuda_reduce;
    cuda_module->super.coll_reduce_scatter_block = mca_coll_cuda_reduce_scatter_block;
    if (!OMPI_COMM_IS_INTER(comm)) {
        cuda_module->super.coll_exscan = mca_coll_cuda_exscan;
        cuda_module->super.coll_scan = mca_coll_cuda_scan;
    }

    return &(cuda_module->super);
}

#define CUDA_INSTALL_COLL_API(__comm, __module, __api)                                                                           \
    do                                                                                                                           \
    {                                                                                                                            \
        if ((__comm)->c_coll->coll_##__api)                                                                                      \
        {                                                                                                                        \
            MCA_COLL_SAVE_API(__comm, __api, (__module)->c_coll.coll_##__api, (__module)->c_coll.coll_##__api##_module, "cuda"); \
            MCA_COLL_INSTALL_API(__comm, __api, mca_coll_cuda_##__api, &__module->super, "cuda");                                \
        }                                                                                                                        \
        else                                                                                                                     \
        {                                                                                                                        \
            opal_show_help("help-mca-coll-base.txt", "comm-select:missing collective", true,                                     \
                           "cuda", #__api, ompi_process_info.nodename,                                                           \
                           mca_coll_cuda_component.priority);                                                                    \
        }                                                                                                                        \
    } while (0)

#define CUDA_UNINSTALL_COLL_API(__comm, __module, __api)                                                                            \
    do                                                                                                                              \
    {                                                                                                                               \
        if (&(__module)->super == (__comm)->c_coll->coll_##__api##_module)                                                          \
        {                                                                                                                           \
            MCA_COLL_INSTALL_API(__comm, __api, (__module)->c_coll.coll_##__api, (__module)->c_coll.coll_##__api##_module, "cuda"); \
            (__module)->c_coll.coll_##__api##_module = NULL;                                                                        \
            (__module)->c_coll.coll_##__api = NULL;                                                                                 \
        }                                                                                                                           \
    } while (0)
/*
 * Init module on the communicator
 */
int
mca_coll_cuda_module_enable(mca_coll_base_module_t *module,
                            struct ompi_communicator_t *comm)
{
    mca_coll_cuda_module_t *s = (mca_coll_cuda_module_t*) module;

    CUDA_INSTALL_COLL_API(comm, s, allreduce);
    CUDA_INSTALL_COLL_API(comm, s, reduce);
    CUDA_INSTALL_COLL_API(comm, s, reduce_scatter_block);
    if (!OMPI_COMM_IS_INTER(comm)) {
        /* MPI does not define scan/exscan on intercommunicators */
        CUDA_INSTALL_COLL_API(comm, s, exscan);
        CUDA_INSTALL_COLL_API(comm, s, scan);
    }

    return OMPI_SUCCESS;
}

int
mca_coll_cuda_module_disable(mca_coll_base_module_t *module,
                             struct ompi_communicator_t *comm)
{
    mca_coll_cuda_module_t *s = (mca_coll_cuda_module_t*) module;

    CUDA_UNINSTALL_COLL_API(comm, s, allreduce);
    CUDA_UNINSTALL_COLL_API(comm, s, reduce);
    CUDA_UNINSTALL_COLL_API(comm, s, reduce_scatter_block);
    if (!OMPI_COMM_IS_INTER(comm))
    {
        /* MPI does not define scan/exscan on intercommunicators */
        CUDA_UNINSTALL_COLL_API(comm, s, exscan);
        CUDA_UNINSTALL_COLL_API(comm, s, scan);
    }

    return OMPI_SUCCESS;
}
