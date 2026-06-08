/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2012      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2013      FUJITSU LIMITED.  All rights reserved.
 * Copyright (c) 2014-2021 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2014      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2017      IBM Corporation. All rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "coll_basic.h"

#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "opal/datatype/opal_convertor_internal.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"

/*
 * We want to minimize the amount of temporary memory needed while allowing as many ranks
 * to exchange data simultaneously. We use a variation of the ring algorithm, where in a
 * single step a process echange the data with both neighbors at distance k (on the left
 * and the right on a logical ring topology). With this approach we need to pack the data
 * for a single of the two neighbors, as we can then use the original buffer (and datatype
 * and count) to send the data to the other.
 */
static int
mca_coll_basic_alltoallw_intra_inplace(const void *rbuf, ompi_count_array_t rcounts, ompi_disp_array_t rdisps,
                                       ompi_datatype_array_t rdtypes,
                                       struct ompi_communicator_t *comm,
                                       mca_coll_base_module_t *module)
{
    int i, size, rank, left, right, err = MPI_SUCCESS;
    ompi_request_t *req = MPI_REQUEST_NULL;
    char *tmp_buffer = NULL;
    size_t max_size = 0, packed_size, msg_size_left, msg_size_right;
    opal_convertor_t convertor;

    size = ompi_comm_size(comm);
    if (1 == size) {  /* If only one process, we're done. */
        return MPI_SUCCESS;
    }
    rank = ompi_comm_rank(comm);

    /* Find the largest amount of packed send/recv data among all peers where
     * we need to pack before the send.
     */
    for (i = 1 ; i <= (size >> 1) ; ++i) {
        struct ompi_datatype_t *rdtype_right;
        right = (rank + i) % size;
        rdtype_right = ompi_datatype_array_get(rdtypes, right);
#if OPAL_ENABLE_HETEROGENEOUS_SUPPORT
        ompi_proc_t *ompi_proc = ompi_comm_peer_lookup(comm, right);

        if( OPAL_LIKELY(opal_local_arch == ompi_proc->super.proc_convertor->master->remote_arch))  {
            opal_datatype_type_size(&rdtype_right->super, &packed_size);
        } else {
            packed_size = opal_datatype_compute_remote_size(&rdtype_right->super,
                                                            ompi_proc->super.proc_convertor->master->remote_sizes);
        }
#else
        opal_datatype_type_size(&rdtype_right->super, &packed_size);
#endif  /* OPAL_ENABLE_HETEROGENEOUS_SUPPORT */
        packed_size *= ompi_count_array_get(rcounts, right);
        max_size = packed_size > max_size ? packed_size : max_size;
    }

    /* Allocate a temporary buffer */
    tmp_buffer = calloc (max_size, 1);
    if (NULL == tmp_buffer) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    for (i = 1 ; i <= (size >> 1) ; ++i) {
        struct iovec iov = {.iov_base = tmp_buffer, .iov_len = max_size};
        uint32_t iov_count = 1;
        struct ompi_datatype_t *rdtype_right, *rdtype_left;

        right = (rank + i) % size;
        left  = (rank + size - i) % size;
        rdtype_right = ompi_datatype_array_get(rdtypes, right);
        rdtype_left  = ompi_datatype_array_get(rdtypes, left);

        ompi_datatype_type_size(rdtype_right, &msg_size_right);
        msg_size_right *= ompi_count_array_get(rcounts, right);

        ompi_datatype_type_size(rdtype_left, &msg_size_left);
        msg_size_left *= ompi_count_array_get(rcounts, left);

        if( 0 != msg_size_right ) {  /* nothing to exchange with the peer on the right */
            ompi_proc_t *right_proc = ompi_comm_peer_lookup(comm, right);
            opal_convertor_clone(right_proc->super.proc_convertor, &convertor, 0);
            opal_convertor_prepare_for_send(&convertor, &rdtype_right->super, ompi_count_array_get(rcounts, right),
                                            (char *) rbuf + ompi_disp_array_get(rdisps, right));
            packed_size = max_size;
            err = opal_convertor_pack(&convertor, &iov, &iov_count, &packed_size);
            if (1 != err) { goto error_hndl; }

            /* Receive data from the right */
            err = MCA_PML_CALL(irecv ((char *) rbuf + ompi_disp_array_get(rdisps, right),
                                      ompi_count_array_get(rcounts, right), rdtype_right,
                                      right, MCA_COLL_BASE_TAG_ALLTOALLW, comm, &req));
            if (MPI_SUCCESS != err) { goto error_hndl; }
        }

        if( (left != right) && (0 != msg_size_left) ) {
            /* Send data to the left */
            err = MCA_PML_CALL(send ((char *) rbuf + ompi_disp_array_get(rdisps, left),
                                     ompi_count_array_get(rcounts, left), rdtype_left,
                                     left, MCA_COLL_BASE_TAG_ALLTOALLW, MCA_PML_BASE_SEND_STANDARD,
                                     comm));
            if (MPI_SUCCESS != err) { goto error_hndl; }

            err = ompi_request_wait (&req, MPI_STATUSES_IGNORE);
            if (MPI_SUCCESS != err) { goto error_hndl; }

            /* Receive data from the left */
            err = MCA_PML_CALL(irecv ((char *) rbuf + ompi_disp_array_get(rdisps, left),
                                      ompi_count_array_get(rcounts, left), rdtype_left,
                                      left, MCA_COLL_BASE_TAG_ALLTOALLW, comm, &req));
            if (MPI_SUCCESS != err) { goto error_hndl; }
        }

        if( 0 != msg_size_right ) {  /* nothing to exchange with the peer on the right */
            /* Send data to the right */
            err = MCA_PML_CALL(send ((char *) tmp_buffer,  packed_size, MPI_PACKED,
                                     right, MCA_COLL_BASE_TAG_ALLTOALLW, MCA_PML_BASE_SEND_STANDARD,
                                     comm));
            if (MPI_SUCCESS != err) { goto error_hndl; }
        }

        err = ompi_request_wait (&req, MPI_STATUSES_IGNORE);
        if (MPI_SUCCESS != err) { goto error_hndl; }
    }

 error_hndl:
    /* Free the temporary buffer */
    free (tmp_buffer);

    /* All done */

    return err;
}


/*
 *	alltoallw_intra
 *
 *	Function:	- MPI_Alltoallw
 *	Accepts:	- same as MPI_Alltoallw()
 *	Returns:	- MPI_SUCCESS or an MPI error code
 */
int
mca_coll_basic_alltoallw_intra(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    const void *sbuf = (const void *) args->src.info_v.buffer;
    ompi_count_array_t scounts = args->src.info_v.counts;
    ompi_datatype_array_t sdtypes = args->src.info_v.datatypes;
    void *rbuf = args->dst.info_v.buffer;
    ompi_count_array_t rcounts = args->dst.info_v.counts;
    ompi_disp_array_t rdisps = args->dst.info_v.displacements;
    ompi_datatype_array_t rdtypes = args->dst.info_v.datatypes;
    int i, size, rank, err, nreqs;
    char *psnd, *prcv;
    ompi_request_t **preq, **reqs;

    /* Initialize. */
    if (MPI_IN_PLACE == sbuf) {
        return mca_coll_basic_alltoallw_intra_inplace (rbuf, rcounts, rdisps,
                                                       rdtypes, comm, module);
    }

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    /* simple optimization */

    psnd = ((char *) sbuf) + ompi_disp_array_get(args->src.info_v.displacements, rank);
    prcv = ((char *) rbuf) + ompi_disp_array_get(rdisps, rank);

    err = ompi_datatype_sndrcv(psnd, ompi_count_array_get(scounts, rank), ompi_datatype_array_get(sdtypes, rank),
                               prcv, ompi_count_array_get(rcounts, rank), ompi_datatype_array_get(rdtypes, rank));
    if (MPI_SUCCESS != err) {
        return err;
    }

    /* If only one process, we're done. */

    if (1 == size) {
        return MPI_SUCCESS;
    }

    /* Initiate all send/recv to/from others. */

    nreqs = 0;
    reqs = preq = ompi_coll_base_comm_get_reqs(module->base_data, 2 * size);
    if( NULL == reqs ) { return OMPI_ERR_OUT_OF_RESOURCE; }

    /* Post all receives first -- a simple optimization */

    for (i = 0; i < size; ++i) {
        size_t msg_size;
        struct ompi_datatype_t *rdtype_i = ompi_datatype_array_get(rdtypes, i);
        ompi_datatype_type_size(rdtype_i, &msg_size);
        msg_size *= ompi_count_array_get(rcounts, i);

        if (i == rank || 0 == msg_size)
            continue;

        prcv = ((char *) rbuf) + ompi_disp_array_get(rdisps, i);
        err = MCA_PML_CALL(irecv_init(prcv, ompi_count_array_get(rcounts, i), rdtype_i,
                                      i, MCA_COLL_BASE_TAG_ALLTOALLW, comm,
                                      preq++));
        ++nreqs;
        if (MPI_SUCCESS != err) {
            ompi_coll_base_free_reqs(reqs, nreqs);
            return err;
        }
    }

    /* Now post all sends */

    for (i = 0; i < size; ++i) {
        size_t msg_size;
        struct ompi_datatype_t *sdtype_i = ompi_datatype_array_get(sdtypes, i);
        ompi_datatype_type_size(sdtype_i, &msg_size);
        msg_size *= ompi_count_array_get(scounts, i);

        if (i == rank || 0 == msg_size)
            continue;

        psnd = ((char *) sbuf) + ompi_disp_array_get(args->src.info_v.displacements, i);
        err = MCA_PML_CALL(isend_init(psnd, ompi_count_array_get(scounts, i), sdtype_i,
                                      i, MCA_COLL_BASE_TAG_ALLTOALLW,
                                      MCA_PML_BASE_SEND_STANDARD, comm,
                                      preq++));
        ++nreqs;
        if (MPI_SUCCESS != err) {
            ompi_coll_base_free_reqs(reqs, nreqs);
            return err;
        }
    }

    /* Start your engines.  This will never return an error. */

    MCA_PML_CALL(start(nreqs, reqs));

    /* Wait for them all.  If there's an error, note that we don't care
     * what the error was -- just that there *was* an error.  The PML
     * will finish all requests, even if one or more of them fail.
     * i.e., by the end of this call, all the requests are free-able.
     * So free them anyway -- even if there was an error, and return the
     * error after we free everything. */

    err = ompi_request_wait_all(nreqs, reqs, MPI_STATUSES_IGNORE);
    /* Free the requests in all cases as they are persistent */
    ompi_coll_base_free_reqs(reqs, nreqs);

    /* All done */
    return err;
}


/*
 *	alltoallw_inter
 *
 *	Function:	- MPI_Alltoallw
 *	Accepts:	- same as MPI_Alltoallw()
 *	Returns:	- MPI_SUCCESS or an MPI error code
 */
int
mca_coll_basic_alltoallw_inter(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    int i, size, err, nreqs;
    char *psnd, *prcv;
    ompi_request_t **preq, **reqs;

    /* Initialize. */
    size = ompi_comm_remote_size(comm);

    /* Initiate all send/recv to/from others. */
    nreqs = 0;
    reqs = preq = ompi_coll_base_comm_get_reqs(module->base_data, 2 * size);
    if( NULL == reqs ) { return OMPI_ERR_OUT_OF_RESOURCE; }

    /* Post all receives first -- a simple optimization */
    for (i = 0; i < size; ++i) {
        size_t msg_size;
        struct ompi_datatype_t *rdtype_i = ompi_datatype_array_get(args->dst.info_v.datatypes, i);
        ompi_datatype_type_size(rdtype_i, &msg_size);
        msg_size *= ompi_count_array_get(args->dst.info_v.counts, i);

        if (0 == msg_size)
            continue;

        prcv = ((char *) args->dst.info_v.buffer) + ompi_disp_array_get(args->dst.info_v.displacements, i);
        err = MCA_PML_CALL(irecv_init(prcv, ompi_count_array_get(args->dst.info_v.counts, i), rdtype_i,
                                      i, MCA_COLL_BASE_TAG_ALLTOALLW,
                                      comm, preq++));
        ++nreqs;
        if (OMPI_SUCCESS != err) {
            ompi_coll_base_free_reqs(reqs, nreqs);
            return err;
        }
    }

    /* Now post all sends */
    for (i = 0; i < size; ++i) {
        size_t msg_size;
        struct ompi_datatype_t *sdtype_i = ompi_datatype_array_get(args->src.info_v.datatypes, i);
        ompi_datatype_type_size(sdtype_i, &msg_size);
        msg_size *= ompi_count_array_get(args->src.info_v.counts, i);

        if (0 == msg_size)
            continue;

        psnd = ((char *) args->src.info_v.buffer) + ompi_disp_array_get(args->src.info_v.displacements, i);
        err = MCA_PML_CALL(isend_init(psnd, ompi_count_array_get(args->src.info_v.counts, i), sdtype_i,
                                      i, MCA_COLL_BASE_TAG_ALLTOALLW,
                                      MCA_PML_BASE_SEND_STANDARD, comm,
                                      preq++));
        ++nreqs;
        if (OMPI_SUCCESS != err) {
            ompi_coll_base_free_reqs(reqs, nreqs);
            return err;
        }
    }

    /* Start your engines.  This will never return an error. */
    MCA_PML_CALL(start(nreqs, reqs));

    /* Wait for them all.  If there's an error, note that we don't care
     * what the error was -- just that there *was* an error.  The PML
     * will finish all requests, even if one or more of them fail.
     * i.e., by the end of this call, all the requests are free-able.
     * So free them anyway -- even if there was an error, and return the
     * error after we free everything. */
    err = ompi_request_wait_all(nreqs, reqs, MPI_STATUSES_IGNORE);

    /* Free the requests in all cases as they are persistent */
    ompi_coll_base_free_reqs(reqs, nreqs);

    /* All done */
    return err;
}
