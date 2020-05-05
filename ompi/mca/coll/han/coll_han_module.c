/*
 * Copyright (c) 2018-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2020      Bull S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "mpi.h"
#include "coll_han.h"
#include "coll_han_dynamic.h"


/*
 * Local functions
 */
static int han_module_enable(mca_coll_base_module_t * module,
                             struct ompi_communicator_t *comm);
static int mca_coll_han_module_disable(mca_coll_base_module_t * module,
                                       struct ompi_communicator_t *comm);

/*
 * Module constructor
 */
static void han_module_clear(mca_coll_han_module_t *han_module)
{
    int i;

    for (i = 0; i < COLLCOUNT; i++) {
        /*
         * Since the previous routines function pointers are declared as
         * a union, initializing the dummy routineis enough
         */
        han_module->previous_routines[i].previous_routine.dummy = NULL;
        han_module->previous_routines[i].previous_module = NULL;
    }
}

static void mca_coll_han_module_construct(mca_coll_han_module_t * module)
{
    int i;

    module->enabled = false;
    module->super.coll_module_disable = mca_coll_han_module_disable;
    module->cached_comm = NULL;
    module->cached_low_comms = NULL;
    module->cached_up_comms = NULL;
    module->cached_vranks = NULL;
    module->cached_topo = NULL;
    module->is_mapbycore = false;
    module->storage_initialized = false;
    for (i = 0 ; i < NB_TOPO_LVL ; i++) {
        module->sub_comm[i] = NULL;
    }
    for (i=SELF ; i<COMPONENTS_COUNT ; i++) {
        module->modules_storage.modules[i].module_handler = NULL;
    }

    han_module_clear(module);
}


#define OBJ_RELEASE_IF_NOT_NULL(obj) do {   \
    if (NULL != (obj)) {                    \
        OBJ_RELEASE(obj);                   \
    }                                       \
} while (0)

/*
 * Module destructor
 */
static void mca_coll_han_module_destruct(mca_coll_han_module_t * module)
{
    int i;

    module->enabled = false;
    if (module->cached_low_comms != NULL) {
        for (i = 0; i < COLL_HAN_LOW_MODULES; i++) {
            ompi_comm_free(&(module->cached_low_comms[i]));
            module->cached_low_comms[i] = NULL;
        }
        free(module->cached_low_comms);
        module->cached_low_comms = NULL;
    }
    if (module->cached_up_comms != NULL) {
        for (i = 0; i < COLL_HAN_UP_MODULES; i++) {
            ompi_comm_free(&(module->cached_up_comms[i]));
            module->cached_up_comms[i] = NULL;
        }
        free(module->cached_up_comms);
        module->cached_up_comms = NULL;
    }
    if (module->cached_vranks != NULL) {
        free(module->cached_vranks);
        module->cached_vranks = NULL;
    }
    if (module->cached_topo != NULL) {
        free(module->cached_topo);
        module->cached_topo = NULL;
    }
    for(i=0 ; i<NB_TOPO_LVL ; i++) {
        if(NULL != module->sub_comm[i]) {
            ompi_comm_free(&(module->sub_comm[i]));
        }
    }

    OBJ_RELEASE_IF_NOT_NULL(module->previous_allgather_module);
    OBJ_RELEASE_IF_NOT_NULL(module->previous_allreduce_module);
    OBJ_RELEASE_IF_NOT_NULL(module->previous_bcast_module);
    OBJ_RELEASE_IF_NOT_NULL(module->previous_gather_module);
    OBJ_RELEASE_IF_NOT_NULL(module->previous_reduce_module);
    OBJ_RELEASE_IF_NOT_NULL(module->previous_scatter_module);

    han_module_clear(module);
}


OBJ_CLASS_INSTANCE(mca_coll_han_module_t,
                   mca_coll_base_module_t,
                   mca_coll_han_module_construct,
                   mca_coll_han_module_destruct);

/*
 * Initial query function that is invoked during MPI_INIT, allowing
 * this component to disqualify itself if it doesn't support the
 * required level of thread support.  This function is invoked exactly
 * once.
 */
int mca_coll_han_init_query(bool enable_progress_threads,
                            bool enable_mpi_threads)
{
    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:han:init_query: pick me! pick me!");
    return OMPI_SUCCESS;
}


/*
 * Invoked when there's a new communicator that has been created.
 * Look at the communicator and decide which set of functions and
 * priority we want to return.
 */
mca_coll_base_module_t *
mca_coll_han_comm_query(struct ompi_communicator_t * comm, int *priority)
{
    mca_coll_han_module_t *han_module;

    /*
     * If we're intercomm, or if there's only one process in the communicator
     */
    if (OMPI_COMM_IS_INTER(comm)) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:han:comm_query (%d/%s): intercomm; disqualifying myself",
                            comm->c_contextid, comm->c_name);
        return NULL;
    }
    if (1 == ompi_comm_size(comm)) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:han:comm_query (%d/%s): comm is too small; disqualifying myself",
                            comm->c_contextid, comm->c_name);
        return NULL;
    }

    /* Get the priority level attached to this module. If priority is less
     * than or equal to 0, then the module is unavailable. */
    *priority = mca_coll_han_component.han_priority;
    if (mca_coll_han_component.han_priority <= 0) {
        opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                            "coll:han:comm_query (%d/%s): priority too low; disqualifying myself",
                            comm->c_contextid, comm->c_name);
        return NULL;
    }

    han_module = OBJ_NEW(mca_coll_han_module_t);
    if (NULL == han_module) {
        return NULL;
    }

    /* All is good -- return a module */
    han_module->topologic_level = mca_coll_han_component.topo_level;

    /*
     * TODO: When the selector is fully implemented,
     *       this if will be meaningless
     */
    if (GLOBAL_COMMUNICATOR == han_module->topologic_level) {
        /* We are on the global communicator, return topological algorithms */
        han_module->super.coll_module_enable = han_module_enable;
        han_module->super.ft_event        = NULL;
        han_module->super.coll_allgather  = NULL;
        han_module->super.coll_allgatherv = NULL;
        if(mca_coll_han_component.use_simple_algorithm[ALLREDUCE]){
            OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                "Using simplified allreduce"));
            han_module->super.coll_allreduce  = mca_coll_han_allreduce_intra_simple;
        } else {
            han_module->super.coll_allreduce  = mca_coll_han_allreduce_intra;
        }
        han_module->super.coll_alltoall   = NULL;
        han_module->super.coll_alltoallv  = NULL;
        han_module->super.coll_alltoallw  = NULL;
        han_module->super.coll_barrier    = NULL;
        han_module->super.coll_bcast      = mca_coll_han_bcast_intra_dynamic;
        han_module->super.coll_exscan     = NULL;
        han_module->super.coll_gather     = mca_coll_han_gather_intra;
        han_module->super.coll_gatherv    = NULL;
        if(mca_coll_han_component.use_simple_algorithm[REDUCE]){
            OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                "Using simplified reduce"));
            han_module->super.coll_reduce  = mca_coll_han_reduce_intra_simple;
        } else {
            han_module->super.coll_reduce  = mca_coll_han_reduce_intra;
        }
        han_module->super.coll_reduce_scatter = NULL;
        han_module->super.coll_scan       = NULL;
        han_module->super.coll_scatter    = NULL;
        han_module->super.coll_scatterv   = NULL;
    } else {
        /* We are on a topologic sub-communicator, return only the selector */
        han_module->super.coll_module_enable = han_module_enable;
        han_module->super.ft_event        = NULL;
        han_module->super.coll_allgather  = NULL;
        han_module->super.coll_allgatherv = NULL;
        han_module->super.coll_allreduce  = NULL;
        han_module->super.coll_alltoall   = NULL;
        han_module->super.coll_alltoallv  = NULL;
        han_module->super.coll_alltoallw  = NULL;
        han_module->super.coll_barrier    = NULL;
        han_module->super.coll_bcast      = mca_coll_han_bcast_intra_dynamic;
        han_module->super.coll_exscan     = NULL;
        han_module->super.coll_gather     = NULL;
        han_module->super.coll_gatherv    = NULL;
        han_module->super.coll_reduce     = NULL;
        han_module->super.coll_reduce_scatter = NULL;
        han_module->super.coll_scan       = NULL;
        han_module->super.coll_scatter    = NULL;
        han_module->super.coll_scatterv   = NULL;
    }

    opal_output_verbose(10, ompi_coll_base_framework.framework_output,
                        "coll:han:comm_query (%d/%s): pick me! pick me!",
                        comm->c_contextid, comm->c_name);
    return &(han_module->super);
}


/*
 * In this macro, the following variables are supposed to have been declared
 * in the caller:
 * . ompi_communicator_t *comm 
 * . mca_coll_han_module_t *han_module
 */ 
#define HAN_SAVE_PREV_COLL_API(__api)   do {                                             \
        han_module->previous_ ## __api            = comm->c_coll->coll_ ## __api;           \
        han_module->previous_ ## __api ## _module = comm->c_coll->coll_ ## __api ## _module;\
        if (!comm->c_coll->coll_ ## __api || !comm->c_coll->coll_ ## __api ## _module) {    \
            opal_output_verbose(1, ompi_coll_base_framework.framework_output,               \
                    "(%d/%s): no underlying " # __api"; disqualifying myself",              \
                    comm->c_contextid, comm->c_name);                                       \
                    return OMPI_ERROR;                                                      \
        }                                                                                   \
        /* TODO add a OBJ_RELEASE at module disabling */                                    \
        /*   + FIXME find why releasing generates memory corruption */                      \
        OBJ_RETAIN(han_module->previous_ ## __api ## _module);                           \
    } while(0)

/*
 * Init module on the communicator
 */
static int han_module_enable(mca_coll_base_module_t * module, struct ompi_communicator_t *comm)
{
    mca_coll_han_module_t * han_module = (mca_coll_han_module_t*) module;

    HAN_SAVE_PREV_COLL_API(allgather);
    HAN_SAVE_PREV_COLL_API(allreduce);
    HAN_SAVE_PREV_COLL_API(bcast);
    HAN_SAVE_PREV_COLL_API(gather);
    HAN_SAVE_PREV_COLL_API(reduce);
    HAN_SAVE_PREV_COLL_API(scatter);

    return OMPI_SUCCESS;
}


/*
 * Module disable
 */
static int mca_coll_han_module_disable(mca_coll_base_module_t * module,
                                       struct ompi_communicator_t *comm)
{
    mca_coll_han_module_t * han_module = (mca_coll_han_module_t *) module;

    OBJ_RELEASE_IF_NOT_NULL(han_module->previous_allgather_module);
    OBJ_RELEASE_IF_NOT_NULL(han_module->previous_allreduce_module);
    OBJ_RELEASE_IF_NOT_NULL(han_module->previous_bcast_module);
    OBJ_RELEASE_IF_NOT_NULL(han_module->previous_gather_module);
    OBJ_RELEASE_IF_NOT_NULL(han_module->previous_reduce_module);
    OBJ_RELEASE_IF_NOT_NULL(han_module->previous_scatter_module);

    han_module_clear(han_module);

    return OMPI_SUCCESS;
}


/*
 * Free the han request
 */
int han_request_free(ompi_request_t ** request)
{
    (*request)->req_state = OMPI_REQUEST_INVALID;
    OBJ_RELEASE(*request);
    *request = MPI_REQUEST_NULL;
    return OMPI_SUCCESS;
}

/* Create the communicators used in the HAN module */
void mca_coll_han_comm_create(struct ompi_communicator_t *comm, mca_coll_han_module_t * han_module)
{
    /* Use cached communicators if possible */
    if (han_module->cached_comm == comm && han_module->cached_low_comms != NULL
        && han_module->cached_up_comms != NULL && han_module->cached_vranks != NULL) {
        return;
    }
    /* Create communicators if there is no cached communicator */
    else {
        int low_rank, low_size;
        int up_rank;
        int w_rank = ompi_comm_rank(comm);
        int w_size = ompi_comm_size(comm);
        ompi_communicator_t **low_comms =
            (struct ompi_communicator_t **) malloc(sizeof(struct ompi_communicator_t *) * 2);
        ompi_communicator_t **up_comms =
            (struct ompi_communicator_t **) malloc(sizeof(struct ompi_communicator_t *) * 2);
        /* Create low_comms which contain all the process on a node */
        const int *origin_priority = NULL;
        /* Lower the priority of HAN module */
        int han_var_id;
        int tmp_han_priority = 0;
        int tmp_han_origin = 0;
        mca_base_var_find_by_name("coll_han_priority", &han_var_id);
        mca_base_var_get_value(han_var_id, &origin_priority, NULL, NULL);
        tmp_han_origin = *origin_priority;
        mca_base_var_set_flag(han_var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(han_var_id, &tmp_han_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET,
                               NULL);
        comm->c_coll->coll_allreduce = ompi_coll_base_allreduce_intra_recursivedoubling;
        comm->c_coll->coll_allgather = ompi_coll_base_allgather_intra_bruck;

        int var_id;
        int tmp_priority = 100;
        int tmp_origin = 0;
        /* Set up low_comms[0] with sm module */
        mca_base_var_find_by_name("coll_sm_priority", &var_id);
        mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
        tmp_origin = *origin_priority;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "[%d] sm_priority origin %d %d\n", w_rank, *origin_priority,
                             tmp_origin));
        mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        ompi_comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, (opal_info_t *) (&ompi_mpi_info_null),
                             &(low_comms[0]));
        mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        low_size = ompi_comm_size(low_comms[0]);
        low_rank = ompi_comm_rank(low_comms[0]);

        /* Set up low_comms[1] with solo module */
        mca_base_var_find_by_name("coll_solo_priority", &var_id);
        mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
        tmp_origin = *origin_priority;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "[%d] solo_priority origin %d %d\n", w_rank, *origin_priority,
                             tmp_origin));
        mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        ompi_comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, (opal_info_t *) (&ompi_mpi_info_null),
                             &(low_comms[1]));
        mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);

        /* Create up_comms[0] with libnbc which contain one process per node (across nodes) */
        mca_base_var_find_by_name("coll_libnbc_priority", &var_id);
        mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
        tmp_origin = *origin_priority;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "[%d] libnbc_priority origin %d %d\n", w_rank, *origin_priority,
                             tmp_origin));
        mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        ompi_comm_split(comm, low_rank, w_rank, &(up_comms[0]), false);
        mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        up_rank = ompi_comm_rank(up_comms[0]);

        /* Create up_comms[1] with adapt which contain one process per node (across nodes) */
        mca_base_var_find_by_name("coll_adapt_priority", &var_id);
        mca_base_var_get_value(var_id, &origin_priority, NULL, NULL);
        tmp_origin = *origin_priority;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "[%d] adapt_priority origin %d %d\n", w_rank, *origin_priority,
                             tmp_origin));
        mca_base_var_set_flag(var_id, MCA_BASE_VAR_FLAG_SETTABLE, true);
        mca_base_var_set_value(var_id, &tmp_priority, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);
        ompi_comm_split(comm, low_rank, w_rank, &(up_comms[1]), false);
        mca_base_var_set_value(var_id, &tmp_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET, NULL);

        int *vranks = malloc(sizeof(int) * w_size);
        /* Do allgather to gather vrank from each process so every process knows other processes' vrank */
        int vrank = low_size * up_rank + low_rank;
        ompi_coll_base_allgather_intra_bruck(&vrank, 1, MPI_INT, vranks, 1, MPI_INT, comm,
                                             comm->c_coll->coll_allgather_module);
        han_module->cached_comm = comm;
        han_module->cached_low_comms = low_comms;
        han_module->cached_up_comms = up_comms;
        han_module->cached_vranks = vranks;

        mca_base_var_set_value(han_var_id, &tmp_han_origin, sizeof(int), MCA_BASE_VAR_SOURCE_SET,
                               NULL);
        comm->c_coll->coll_allreduce = mca_coll_han_allreduce_intra;
        comm->c_coll->coll_allgather = mca_coll_han_allgather_intra;
    }
}
