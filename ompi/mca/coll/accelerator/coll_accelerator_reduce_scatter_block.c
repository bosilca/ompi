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
 *	reduce_scatter_block
 *
 *	Function:	- reduce then scatter
 *	Accepts:	- same as MPI_Reduce_scatter_block()
 *	Returns:	- MPI_SUCCESS or error code
 *
 * Algorithm:
 *     reduce and scatter (needs to be cleaned
 *     up at some point)
 */
int
mca_coll_accelerator_reduce_scatter_block(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_accelerator_module_t *s = (mca_coll_accelerator_module_t*) module;
    const void *sbuf = (const void *) args->src.info.buffer;
    void *rbuf = args->dst.info.buffer;
    ptrdiff_t gap;
    char *rbuf1 = NULL, *sbuf1 = NULL, *rbuf2 = NULL;
    int sbuf_dev, rbuf_dev;
    size_t sbufsize, rbufsize, rbuf_in_size;
    int rc;

    rbufsize = opal_datatype_span(&args->dst.info.datatype->super, args->dst.info.count, &gap);

    sbufsize = rbufsize * ompi_comm_size(comm);
    /* With MPI_IN_PLACE the input lives entirely in rbuf and spans
     * comm_size * rcount elements; stage the full span device->host so the
     * fallback collective doesn't read past the host allocation. */
    rbuf_in_size = (MPI_IN_PLACE == sbuf) ? sbufsize : rbufsize;
    rc = mca_coll_accelerator_check_buf((void *)sbuf, &sbuf_dev);
    if (rc < 0) {
        return rc;
    }
    if ((MPI_IN_PLACE != sbuf) && (rc > 0)) {
        sbuf1 = (char*)malloc(sbufsize);
        if (NULL == sbuf1) {
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        mca_coll_accelerator_memcpy(sbuf1, MCA_ACCELERATOR_NO_DEVICE_ID, sbuf, sbuf_dev, sbufsize,
                                    MCA_ACCELERATOR_TRANSFER_DTOH);
        sbuf = sbuf1 - gap;
    }
    rc = mca_coll_accelerator_check_buf(rbuf, &rbuf_dev);
    if (rc < 0) {
        return rc;
    }
    if (rc > 0) {
        rbuf1 = (char*)malloc(rbuf_in_size);
        if (NULL == rbuf1) {
            if (NULL != sbuf1) free(sbuf1);
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        mca_coll_accelerator_memcpy(rbuf1, MCA_ACCELERATOR_NO_DEVICE_ID, rbuf, rbuf_dev, rbuf_in_size,
                                    MCA_ACCELERATOR_TRANSFER_DTOH);
        rbuf2 = rbuf; /* save away original buffer */
        rbuf = rbuf1 - gap;
    }
    ompi_coll_args_t _fwd;
    ompi_coll_args_reduce_scatter_block(&_fwd, sbuf, rbuf, args->dst.info.count, args->dst.info.datatype, args->op);
    rc = s->c_coll.coll_reduce_scatter_block(&_fwd, comm,
                                             s->c_coll.coll_reduce_scatter_block_module);
    if (NULL != sbuf1) {
        free(sbuf1);
    }
    if (NULL != rbuf1) {
        rbuf = rbuf2;
        mca_coll_accelerator_memcpy(rbuf, rbuf_dev, rbuf1, MCA_ACCELERATOR_NO_DEVICE_ID, rbufsize,
                                    MCA_ACCELERATOR_TRANSFER_HTOD);
        free(rbuf1);
    }
    return rc;
}

