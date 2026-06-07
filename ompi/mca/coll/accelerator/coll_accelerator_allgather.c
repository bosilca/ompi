/*
 * Copyright (c) 2014-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2014-2015 NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2022      Amazon.com, Inc. or its affiliates.  All Rights reserved.
 * Copyright (c) 2024      Triad National Security, LLC. All rights reserved.
 * Copyright (c) 2024      Advanced Micro Devices, Inc. All Rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "coll_accelerator.h"

#include <stdio.h>

#include "ompi/op/op.h"
#include "opal/datatype/opal_convertor.h"

/*
 *	Function:	- allgather for device buffers through temp CPU buffer
 *	Accepts:	- same as MPI_Allgather()
 *	Returns:	- MPI_SUCCESS or error code
 */
int
mca_coll_accelerator_allgather(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_accelerator_module_t *s = (mca_coll_accelerator_module_t*) module;
    const void *sbuf = (const void *) args->src.info.buffer;
    void *rbuf = args->dst.info.buffer;
    ptrdiff_t sgap, rgap;
    char *rbuf1 = NULL, *sbuf1 = NULL, *rbuf2 = NULL;
    int sbuf_dev, rbuf_dev;
    size_t sbufsize, rbufsize;
    int rc;
    int comm_size = ompi_comm_size(comm);
    
    sbufsize = opal_datatype_span(&args->src.info.datatype->super, args->src.info.count, &sgap);
    rc = mca_coll_accelerator_check_buf((void *)sbuf, &sbuf_dev);
    if (rc < 0) {
        return rc;
    }
    if ((MPI_IN_PLACE != sbuf) && (rc > 0) &&
        (sbufsize <= (size_t)mca_coll_accelerator_allgather_thresh)) {
        sbuf1 = (char*)malloc(sbufsize);
        if (NULL == sbuf1) {
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        mca_coll_accelerator_memcpy(sbuf1, MCA_ACCELERATOR_NO_DEVICE_ID, sbuf, sbuf_dev,
                                    sbufsize, MCA_ACCELERATOR_TRANSFER_DTOH);
        sbuf = sbuf1 - sgap;
    }

    rbufsize = opal_datatype_span(&args->dst.info.datatype->super, args->dst.info.count, &rgap);
    rc = mca_coll_accelerator_check_buf(rbuf, &rbuf_dev);
    if (rc < 0) {
        goto exit;
    }
    /* Using sbufsize here on purpose to ensure symmetric decision for handling of GPU vs
       CPU buffers. The two buffer sizes are expected to be the same for pre-defined datatypes,
       but could vary due to layout issues/gaps for derived datatypes */
    if ((rc > 0) && (sbufsize <= (size_t)mca_coll_accelerator_allgather_thresh)) {
        rbuf1 = (char*)malloc(rbufsize * comm_size);
        if (NULL == rbuf1) {
            rc = OMPI_ERR_OUT_OF_RESOURCE;
            goto exit;
        }
        mca_coll_accelerator_memcpy(rbuf1, MCA_ACCELERATOR_NO_DEVICE_ID, rbuf, rbuf_dev,
                                    rbufsize * comm_size, MCA_ACCELERATOR_TRANSFER_DTOH);
        rbuf2 = rbuf; /* save original buffer */
        rbuf = rbuf1 - rgap;
    }
    ompi_coll_args_t _fwd;
    ompi_coll_args_allgather(&_fwd, sbuf, args->src.info.count, args->src.info.datatype, rbuf, args->dst.info.count, args->dst.info.datatype);
    rc = s->c_coll.coll_allgather(&_fwd, comm, s->c_coll.coll_allgather_module);
    if (rc < 0) {
        goto exit;
    }

    if (NULL != rbuf1) {
        mca_coll_accelerator_memcpy(rbuf2, rbuf_dev, rbuf1, MCA_ACCELERATOR_NO_DEVICE_ID, rbufsize * comm_size,
                                    MCA_ACCELERATOR_TRANSFER_HTOD);
    }

 exit:
    if (NULL != sbuf1) {
        free(sbuf1);
    }
    if (NULL != rbuf1) {
        free(rbuf1);
    }

    return rc;
}
