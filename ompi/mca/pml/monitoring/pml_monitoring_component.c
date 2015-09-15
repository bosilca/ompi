/*
 * Copyright (c) 2013-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013-2015 Inria.  All rights reserved.
 * Copyright (c) 2015      Bull SAS.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include <ompi_config.h>
#include <pml_monitoring.h>
#include <ompi/constants.h>
#include <ompi/mca/pml/base/base.h>
#include <opal/mca/base/mca_base_component_repository.h>

static int mca_pml_monitoring_enabled = 0;
static int mca_pml_monitoring_output_enabled = 0;
static int mca_pml_monitoring_active = 0;
mca_pml_base_component_t pml_selected_component;
mca_pml_base_module_t pml_selected_module;

extern void finalize_monitoring( void );
extern int  ompi_mca_pml_monitoring_flush(char* filename);
int filter_monitoring( void );

/* Return 1 if the the seperation between internal tags and external tags is enabled */
int filter_monitoring( void )
{
    return (mca_pml_monitoring_enabled == 2) ? 1 : 0;
}

static int
mca_pml_monitoring_set_flush(struct mca_base_pvar_t *pvar, const void *value, void *obj)
{
    char* filename = (char*)value;
    int err = ompi_mca_pml_monitoring_flush(filename);
    if( 0 == err )
        return OMPI_SUCCESS;
    return OMPI_ERROR;
}

static int
mca_pml_monitoring_get_flush(const struct mca_base_pvar_t *pvar, void *value, void *obj)
{
    return OMPI_SUCCESS;
}

static int
mca_pml_monitoring_notify_flush(struct mca_base_pvar_t *pvar, mca_base_pvar_event_t event,
                                void *obj, int *count)
{
    switch event {
        case MCA_BASE_PVAR_HANDLE_BIND:
        case MCA_BASE_PVAR_HANDLE_UNBIND:
        case MCA_BASE_PVAR_HANDLE_START:
        case MCA_BASE_PVAR_HANDLE_STOP:
            return OMPI_SUCCESS;
        }
    return OMPI_ERROR;
}

int mca_pml_monitoring_enable(bool enable)
{
    /* If we reach this point we were succesful at hijacking the interface of
     * the real PML, and we are now correctly interleaved between the upper
     * layer and the real PML.
     */
    (void)mca_base_pvar_register("ompi", "pml", "monitoring", "flush", "Flush the monitoring information"
                                 "in the provided file", OPAL_INFO_LVL_1, MCA_BASE_PVAR_CLASS_GENERIC,
                                 MCA_BASE_VAR_TYPE_STRING, NULL, MPI_T_BIND_NO_OBJECT,
                                 0,
                                 mca_pml_monitoring_get_flush, mca_pml_monitoring_set_flush,
                                 mca_pml_monitoring_notify_flush, &mca_pml_monitoring_component);

    return pml_selected_module.pml_enable(enable);
}

static int mca_pml_monitoring_component_open(void)
{
    if( mca_pml_monitoring_enabled ) {
        opal_pointer_array_add(&mca_pml_base_pml,
                               strdup(mca_pml_monitoring_component.pmlm_version.mca_component_name));
    }
    return OMPI_SUCCESS;
}

static int
mca_pml_monitoring_comm_size_notify(mca_base_pvar_t *pvar,
                                    mca_base_pvar_event_t event,
                                    void *obj_handle,
                                    int *count)
{
    if (MCA_BASE_PVAR_HANDLE_BIND == event) {
        /* Return the size of the communicator as the number of values */
        *count = ompi_comm_size ((ompi_communicator_t *) obj_handle);
    }

    return OMPI_SUCCESS;
}

static int mca_pml_monitoring_component_close(void)
{
    if( !mca_pml_monitoring_enabled )
        return OMPI_SUCCESS;

    if( mca_pml_monitoring_active ) {  /* Already active, turn off */
        pml_selected_component.pmlm_version.mca_close_component();
        memset(&pml_selected_component, 0, sizeof(mca_base_component_t));
        memset(&pml_selected_module, 0, sizeof(mca_base_component_t));
        mca_base_component_repository_release((mca_base_component_t*)&mca_pml_monitoring_component);
        mca_pml_monitoring_active = 0;
        return OMPI_SUCCESS;
    }

    /* Bump my ref count up to avoid getting released too early */
    if( OPAL_SUCCESS != mca_base_component_repository_retain_component(mca_pml_monitoring_component.pmlm_version.mca_type_name,
                                                                       mca_pml_monitoring_component.pmlm_version.mca_component_name) ) {
        return OMPI_ERROR;
    }

    /* Save a copy of the selected PML */
    pml_selected_component = mca_pml_base_selected_component;
    pml_selected_module = mca_pml;
    /* And now install the interception layer */
    mca_pml_base_selected_component = mca_pml_monitoring_component;
    mca_pml = mca_pml_monitoring;
    mca_pml.pml_progress = pml_selected_module.pml_progress;

    mca_pml_monitoring_active = 1;

    return OMPI_SUCCESS;
}

static mca_pml_base_module_t*
mca_pml_monitoring_component_init(int* priority,
                                  bool enable_progress_threads,
                                  bool enable_mpi_threads)
{
    if( mca_pml_monitoring_enabled ) {
        *priority = 0;  /* I'm up but don't select me */
        return &mca_pml_monitoring;
    }
    return NULL;
}

static int mca_pml_monitoring_component_finish(void)
{
    if( mca_pml_monitoring_enabled && mca_pml_monitoring_active ) {
        /* It is over... Output what has been monitored*/
        if ( mca_pml_monitoring_output_enabled != 0) {
            ompi_mca_pml_monitoring_flush(NULL);
        }
        /* Free internal data structure */
        finalize_monitoring();
        /* Call the original PML and then close */
        mca_pml_monitoring_active = 0;
        mca_pml_monitoring_enabled = 0;
        /* Restore the original PML */
        mca_pml_base_selected_component = pml_selected_component;
        mca_pml = pml_selected_module;
        /* Redirect the close call to the original PML */
        pml_selected_component.pmlm_finalize();
        /**
         * We should never release the last ref on the current component or face forever punishement.
         */
        /* mca_base_component_repository_release(&mca_pml_monitoring_component.pmlm_version); */
    }
    return OMPI_SUCCESS;
}

static int mca_pml_monitoring_component_register(void)
{
    (void)mca_base_component_var_register(&mca_pml_monitoring_component.pmlm_version, "enable",
                                          "Enable the monitoring at the PML level. This value should be different than 0 in order for the monitoring to be enabled (default disable)", MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                          OPAL_INFO_LVL_9,
                                          MCA_BASE_VAR_SCOPE_READONLY, &mca_pml_monitoring_enabled);

    (void)mca_base_component_var_register(&mca_pml_monitoring_component.pmlm_version, "enable_output",
                                          "Enable the PML monitoring textual output at MPI_Finalize. This value should be different than 0 in order for the output to be enabled (default disable)", MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                          OPAL_INFO_LVL_9,
                                          MCA_BASE_VAR_SCOPE_READONLY, &mca_pml_monitoring_output_enabled);

    (void)mca_base_pvar_register("ompi", "pml", "monitoring", "messages_count", "Number of messages "
                                 "sent to each peer in a communicator", OPAL_INFO_LVL_4, MPI_T_PVAR_CLASS_SIZE,
                                  MCA_BASE_VAR_TYPE_UNSIGNED_INT, NULL, MPI_T_BIND_MPI_COMM,
                                  MCA_BASE_PVAR_FLAG_READONLY | MCA_BASE_PVAR_FLAG_CONTINUOUS,
                                  mca_pml_monitoring_get_messages_count, NULL, mca_pml_monitoring_comm_size_notify, NULL);

    (void)mca_base_pvar_register("ompi", "pml", "monitoring", "messages_size", "Size of messages "
                                 "sent to each peer in a communicator", OPAL_INFO_LVL_4, MPI_T_PVAR_CLASS_SIZE,
                                 MCA_BASE_VAR_TYPE_UNSIGNED_INT, NULL, MPI_T_BIND_MPI_COMM,
                                 MCA_BASE_PVAR_FLAG_READONLY | MCA_BASE_PVAR_FLAG_CONTINUOUS,
                                 mca_pml_monitoring_get_messages_size, NULL, mca_pml_monitoring_comm_size_notify, NULL);
    return OMPI_SUCCESS;
}

mca_pml_base_component_2_0_0_t mca_pml_monitoring_component = {

    /* First, the mca_base_component_t struct containing meta
       information about the component itself */

    {
        MCA_PML_BASE_VERSION_2_0_0,

        "monitoring", /* MCA component name */
        OMPI_MAJOR_VERSION,  /* MCA component major version */
        OMPI_MINOR_VERSION,  /* MCA component minor version */
        OMPI_RELEASE_VERSION,  /* MCA component release version */
        mca_pml_monitoring_component_open,  /* component open */
        mca_pml_monitoring_component_close, /* component close */
        NULL,
        mca_pml_monitoring_component_register
    },
    {
        /* The component is checkpoint ready */
        MCA_BASE_METADATA_PARAM_CHECKPOINT
    },

    mca_pml_monitoring_component_init,  /* component init */
    mca_pml_monitoring_component_finish   /* component finalize */

};

