/*
 * Copyright (c) 2024      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2004-2023 The University of Tennessee and The University
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
 *
 *	Function:	- Bcast for device buffers through temp CPU buffer. 
 *	Accepts:	- same as MPI_Bcast()
 *	Returns:	- MPI_SUCCESS or error code
 */
int
mca_coll_accelerator_bcast(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_accelerator_module_t *s = (mca_coll_accelerator_module_t*) module;
    void *orig_buf = args->src.info.buffer;
    ptrdiff_t gap;
    char *buf1 = NULL;
    char *sbuf = (char*) orig_buf;
    int buf_dev;
    size_t bufsize;
    int rc;

    bufsize = opal_datatype_span(&args->src.info.datatype->super, args->src.info.count, &gap);

    rc = mca_coll_accelerator_check_buf((void *)orig_buf, &buf_dev);
    if (rc < 0) {
        return rc;
    }
    if ((rc > 0) && (bufsize <= (size_t)mca_coll_accelerator_bcast_thresh)) {
        buf1 = (char*)malloc(bufsize);
        if (NULL == buf1) {
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        mca_coll_accelerator_memcpy(buf1, MCA_ACCELERATOR_NO_DEVICE_ID, orig_buf, buf_dev, bufsize,
                                    MCA_ACCELERATOR_TRANSFER_DTOH);
        sbuf = buf1 - gap;
    }

    ompi_coll_args_t _fwd;
    ompi_coll_args_bcast(&_fwd, (void *) sbuf, args->src.info.count, args->src.info.datatype, args->root);
    rc = s->c_coll.coll_bcast(&_fwd, comm, s->c_coll.coll_bcast_module);
    if (rc < 0) {
        goto exit;
    }
    if (NULL != buf1) {
        mca_coll_accelerator_memcpy((void*)orig_buf, buf_dev, buf1, MCA_ACCELERATOR_NO_DEVICE_ID, bufsize,
                                    MCA_ACCELERATOR_TRANSFER_HTOD);
    }

 exit:
    if (NULL != buf1) {
        free(buf1);
    }

    return rc;
}
