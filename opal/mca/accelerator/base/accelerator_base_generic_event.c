/*
 * Copyright (c) 2024      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"

#include <string.h>

#include "opal/class/opal_list.h"
#include "opal/constants.h"
#include "opal/mca/accelerator/accelerator.h"
#include "opal/mca/accelerator/base/base.h"
#include "opal/mca/base/base.h"
#include "opal/mca/mca.h"
#include "opal/util/output.h"
#include "opal/util/proc.h" 
#include "opal/util/show_help.h"
#include "opal/datatype/opal_convertor.h"

static void
opal_accelerator_stream_trackable_constructor(opal_accelerator_stream_trackable_t *stream)
{
    stream->super.type = MCA_ACCELERATOR_STREAM_TYPE_TRACKABLE;
    OBJ_CONSTRUCT(&stream->stream_lock, opal_mutex_t);
    stream->stream_name = NULL;
    stream->stream_first_avail = 0;
    stream->stream_first_used = 0;
    stream->stream_num_events = 0;
    stream->stream_cb_info = NULL;
}

static void
opal_accelerator_stream_trackable_destructor(opal_accelerator_stream_trackable_t *stream)
{
    if( stream->stream_first_avail != stream->stream_first_used ) {
        OPAL_OUTPUT_VERBOSE((0, 0, "Stream %s freed with %d pending events",
                             stream->stream_name,
                             (stream->stream_num_events + 
                              (stream->stream_first_avail - stream->stream_first_used)) % stream->stream_num_events));
    }
    if( NULL != stream->stream_cb_info ) {
        assert(0 != stream->stream_num_events);
        free(stream->stream_cb_info);
        stream->stream_cb_info = NULL;
    } else {
        assert(0 == stream->stream_num_events);
    }
    stream->stream_num_events = 0;
    stream->stream_first_avail = 0;
    stream->stream_first_used = 0;
    if( NULL != stream->stream_name ) {
        free(stream->stream_name);
        stream->stream_name = NULL;
    }
}

OBJ_CLASS_INSTANCE(opal_accelerator_stream_trackable_t, opal_accelerator_stream_t,
                   opal_accelerator_stream_trackable_constructor,
                   opal_accelerator_stream_trackable_destructor);

/**
 *  Create a trackable stream, where the events are automatically tracked by OPAL, and the
 *  associated callbacks are triggered from opal_progress.
 */
int mca_base_accelerator_trackable_stream_create(int dev_id, opal_accelerator_stream_t **stream,
                                                 uint16_t num_events, char *name)
{
    opal_accelerator_stream_trackable_t *acc_stream = OBJ_NEW(opal_accelerator_stream_trackable_t);

    opal_accelerator.create_stream(dev_id, &acc_stream->super);
    acc_stream->super.type = MCA_ACCELERATOR_STREAM_TYPE_TRACKABLE;
    acc_stream->stream_num_events = num_events;
    acc_stream->stream_name = name;
}
/**
 * Add an event to the stream and trigger the cbfct(stream, cb_data, status) once the
 * event completes. The completion will happen later on during opal_progress.
 */
int mca_base_accelerator_track_event(opal_accelerator_stream_t* stream,
                                     opal_mca_accelerator_callback_fn_t* cb,
                                     void* cb_data,
                                     char *msg)
{
    return OPAL_SUCCESS;
}

/*
 * Record an event and save the frag.  This is called by the sending side and
 * is used to queue an event when a htod copy has been initiated.
 */
int mca_base_accelerator_record_dtoh_event(char *msg, struct mca_btl_base_descriptor_t *frag,
                                           opal_convertor_t *convertor, void *cuda_stream)
{
#if 0
    /* TODO GB: clean this up */
    CUresult result;

    /* First make sure there is room to store the event.  If not, then
     * return an error.  The error message will tell the user to try and
     * run again, but with a larger array for storing events. */
    OPAL_THREAD_LOCK(&common_cuda_dtoh_lock);
    if (cuda_event_dtoh_num_used == cuda_event_max) {
        opal_show_help("help-mpi-common-cuda.txt", "Out of cuEvent handles", true, cuda_event_max,
                       cuda_event_max + 100, cuda_event_max + 100);
        return OPAL_ERR_OUT_OF_RESOURCE;
    }

    if (cuda_event_dtoh_num_used > cuda_event_dtoh_most) {
        cuda_event_dtoh_most = cuda_event_dtoh_num_used;
        /* Just print multiples of 10 */
        if (0 == (cuda_event_dtoh_most % 10)) {
            opal_output_verbose(20, mca_common_cuda_output, "Maximum DtoH events used is now %d",
                                cuda_event_dtoh_most);
        }
    }

    if (cuda_stream == NULL) {
        result = cuFunc.cuEventRecord(cuda_event_dtoh_array[cuda_event_dtoh_first_avail],
                                      dtohStream);
    } else {
        result = cuFunc.cuEventRecord(cuda_event_dtoh_array[cuda_event_dtoh_first_avail],
                                      (CUstream) cuda_stream);
    }
    if (OPAL_UNLIKELY(CUDA_SUCCESS != result)) {
        opal_show_help("help-mpi-common-cuda.txt", "cuEventRecord failed", true,
                       OPAL_PROC_MY_HOSTNAME, result);
        OPAL_THREAD_UNLOCK(&common_cuda_dtoh_lock);
        return OPAL_ERROR;
    }
    cuda_event_dtoh_frag_array[cuda_event_dtoh_first_avail] = frag;
    cuda_event_dtoh_convertor_array[cuda_event_dtoh_first_avail] = convertor;

    /* Bump up the first available slot and number used by 1 */
    cuda_event_dtoh_first_avail++;
    if (cuda_event_dtoh_first_avail >= cuda_event_max) {
        cuda_event_dtoh_first_avail = 0;
    }
    cuda_event_dtoh_num_used++;

    OPAL_THREAD_UNLOCK(&common_cuda_dtoh_lock);
#endif
    return OPAL_SUCCESS;
}

/*
 * Record an event and save the frag.  This is called by the receiving side and
 * is used to queue an event when a dtoh copy has been initiated.
 */
int mca_base_accelerator_record_htod_event(char *msg, struct mca_btl_base_descriptor_t *frag,
                                           void *cuda_stream)
{
    return OPAL_SUCCESS;
}
