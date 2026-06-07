/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/**
 * @file
 *
 * Unified argument descriptor for the collective (coll) framework.
 *
 * Every collective module function receives a single ompi_coll_args_t
 * describing the operation payload, instead of a long per-collective
 * argument list. The framework context (communicator, info, request and
 * module) is passed as explicit function parameters and is NOT part of
 * this structure.
 *
 * The descriptor is self-describing: coll_type names the operation,
 * the mask records which optional fields are populated (including which
 * input arrays the coll layer owns and must free on request completion),
 * and flags carries behavioral hints (vector buffers, per-peer datatypes,
 * neighborhood, in-place, ...).
 *
 * The send (src) and receive (dst) sides are each described by either a
 * scalar ompi_coll_buffer_info_t (single count + datatype) or a vector
 * ompi_coll_buffer_info_v_t (count/displacement arrays, optionally a
 * per-peer datatype array). The active union member is selected by the
 * SRC_VECTOR / DST_VECTOR flags.
 */

#ifndef OMPI_MCA_COLL_COLL_ARGS_H
#define OMPI_MCA_COLL_COLL_ARGS_H

#include "ompi_config.h"

#include "ompi/request/request.h"
#include "ompi/util/count_disp_array.h"

BEGIN_C_DECLS

struct ompi_datatype_t;
struct ompi_op_t;
struct ompi_group_t;

/**
 * Collective operation discriminator.
 *
 * Distinct from the unprefixed COLLTYPE_T enum in coll_base_functions.h
 * (which indexes the tuned/base decision tables): this one is the payload
 * discriminator carried in ompi_coll_args_t and also covers agree,
 * reduce_local and revoke_local.
 */
typedef enum ompi_coll_type_t {
    OMPI_COLL_TYPE_BARRIER,
    OMPI_COLL_TYPE_BCAST,
    OMPI_COLL_TYPE_GATHER,
    OMPI_COLL_TYPE_GATHERV,
    OMPI_COLL_TYPE_SCATTER,
    OMPI_COLL_TYPE_SCATTERV,
    OMPI_COLL_TYPE_ALLGATHER,
    OMPI_COLL_TYPE_ALLGATHERV,
    OMPI_COLL_TYPE_ALLTOALL,
    OMPI_COLL_TYPE_ALLTOALLV,
    OMPI_COLL_TYPE_ALLTOALLW,
    OMPI_COLL_TYPE_REDUCE,
    OMPI_COLL_TYPE_ALLREDUCE,
    OMPI_COLL_TYPE_REDUCE_SCATTER,
    OMPI_COLL_TYPE_REDUCE_SCATTER_BLOCK,
    OMPI_COLL_TYPE_SCAN,
    OMPI_COLL_TYPE_EXSCAN,
    OMPI_COLL_TYPE_NEIGHBOR_ALLGATHER,
    OMPI_COLL_TYPE_NEIGHBOR_ALLGATHERV,
    OMPI_COLL_TYPE_NEIGHBOR_ALLTOALL,
    OMPI_COLL_TYPE_NEIGHBOR_ALLTOALLV,
    OMPI_COLL_TYPE_NEIGHBOR_ALLTOALLW,
    OMPI_COLL_TYPE_AGREE,
    OMPI_COLL_TYPE_REDUCE_LOCAL,
    OMPI_COLL_TYPE_REVOKE_LOCAL,
    OMPI_COLL_TYPE_MAX
} ompi_coll_type_t;

/*
 * mask bits: which optional fields of ompi_coll_args_t are populated.
 * The FIELD_* bits describe presence of optional members; the FREE_*
 * bits mark input arrays that the coll layer owns and must free when a
 * nonblocking/persistent request completes (clear for blocking and for
 * caller-owned arrays). All share the single args->mask field.
 */
enum {
    OMPI_COLL_ARGS_FIELD_OP            = 1ull << 0,
    OMPI_COLL_ARGS_FIELD_ROOT          = 1ull << 1,
    OMPI_COLL_ARGS_FIELD_SRC           = 1ull << 2,
    OMPI_COLL_ARGS_FIELD_DST           = 1ull << 3,
    OMPI_COLL_ARGS_FIELD_FAILEDGROUP   = 1ull << 4,  /* FT agree */
    OMPI_COLL_ARGS_FIELD_TAG           = 1ull << 5,  /* explicit ordering tag */
    OMPI_COLL_ARGS_FIELD_COMPLETE_CB   = 1ull << 6,  /* caller completion cb */

    /* Arrays the coll layer must free on request completion. For *_DTYPES
     * this also releases each datatype object before freeing the array. */
    OMPI_COLL_ARGS_FREE_SRC_COUNTS     = 1ull << 8,
    OMPI_COLL_ARGS_FREE_SRC_DISPS      = 1ull << 9,
    OMPI_COLL_ARGS_FREE_SRC_DTYPES     = 1ull << 10,
    OMPI_COLL_ARGS_FREE_DST_COUNTS     = 1ull << 11,
    OMPI_COLL_ARGS_FREE_DST_DISPS      = 1ull << 12,
    OMPI_COLL_ARGS_FREE_DST_DTYPES     = 1ull << 13,
};

/* flags: behavioral hints. */
enum {
    OMPI_COLL_ARGS_FLAG_SRC_VECTOR     = 1ull << 0,  /* read src.info_v not src.info */
    OMPI_COLL_ARGS_FLAG_DST_VECTOR     = 1ull << 1,  /* read dst.info_v not dst.info */
    OMPI_COLL_ARGS_FLAG_SRC_DTYPE_VEC  = 1ull << 2,  /* src.info_v.datatypes valid */
    OMPI_COLL_ARGS_FLAG_DST_DTYPE_VEC  = 1ull << 3,  /* dst.info_v.datatypes valid */
    OMPI_COLL_ARGS_FLAG_NEIGHBOR       = 1ull << 4,
    OMPI_COLL_ARGS_FLAG_IN_PLACE       = 1ull << 5,
    OMPI_COLL_ARGS_FLAG_UPDATE_FAILEDGROUP = 1ull << 6,
};

/*
 * Memory placement of a buffer: CPU vs GPU/VMM, and the device index when
 * on GPU. The real encoding is to be defined later; a size_t placeholder
 * for now. Left zero by builders/call sites and populated, at most once,
 * inside a collective module that needs it.
 */
typedef size_t ompi_memory_type_t;

/** Scalar buffer descriptor: one count and one datatype. */
typedef struct ompi_coll_buffer_info_t {
    void                   *buffer;     /**< sbuf/rbuf/buff; src cast from const */
    ompi_memory_type_t      mem_type;   /**< CPU/GPU/VMM (+device idx if GPU) */
    size_t                  count;
    struct ompi_datatype_t *datatype;
} ompi_coll_buffer_info_t;

/** Vector buffer descriptor: count/displacement arrays and one datatype form. */
typedef struct ompi_coll_buffer_info_v_t {
    void                   *buffer;
    ompi_memory_type_t      mem_type;        /**< CPU/GPU/VMM (+device idx if GPU) */
    ompi_count_array_t      counts;          /**< intptr_t by value */
    ompi_disp_array_t       displacements;   /**< unused for reduce_scatter */
    union {                                  /**< anonymous: only one is set */
        struct ompi_datatype_t *datatype;          /**< single datatype */
        struct ompi_datatype_t * const *datatypes; /**< per-peer (alltoallw) */
    };
} ompi_coll_buffer_info_v_t;

/**
 * Unified collective argument descriptor.
 *
 * comm / info / request / module are NOT here -- they are explicit
 * function parameters of the collective module functions.
 */
typedef struct ompi_coll_args_t {
    uint64_t          mask;        /**< OMPI_COLL_ARGS_FIELD_* + FREE_* present */
    ompi_coll_type_t  coll_type;
    uint64_t          flags;       /**< OMPI_COLL_ARGS_FLAG_* hints */
    union {
        ompi_coll_buffer_info_t   info;
        ompi_coll_buffer_info_v_t info_v;
    } src;                          /**< send side (sbuf/inbuf/contrib) */
    union {
        ompi_coll_buffer_info_t   info;
        ompi_coll_buffer_info_v_t info_v;
    } dst;                          /**< recv side (rbuf/inoutbuf) */
    struct ompi_op_t *op;           /**< reductions; valid iff FIELD_OP */
    int               root;         /**< rooted ops; valid iff FIELD_ROOT */
    int               tag;          /**< ordering tag; valid iff FIELD_TAG */

    /* Optional caller completion callback, valid iff FIELD_COMPLETE_CB.
     * Invoked when the nonblocking/persistent request completes; ignored by
     * blocking collectives (no request to hand it). */
    ompi_request_complete_fn_t req_complete_cb;
    void                      *req_complete_cb_data;

    /* FT agree extras (valid iff FIELD_FAILEDGROUP). */
    struct ompi_group_t **failedgroup;
} ompi_coll_args_t;


/* ******************************************************************** */
/*  Builder helpers.                                                    */
/*                                                                      */
/*  One builder per collective family fills coll_type/mask/flags and    */
/*  the src/dst descriptors. Nonblocking (i*) and persistent (*_init)   */
/*  callers use the same builder and pass request/info as parameters.   */
/*  Compound-literal assignment zero-fills omitted members.             */
/* ******************************************************************** */

static inline void
ompi_coll_args_set_complete_cb(ompi_coll_args_t *a, ompi_request_complete_fn_t cb,
                               void *cbdata)
{
    a->req_complete_cb = cb;
    a->req_complete_cb_data = cbdata;
    a->mask |= OMPI_COLL_ARGS_FIELD_COMPLETE_CB;
}

static inline void
ompi_coll_args_barrier(ompi_coll_args_t *a)
{
    *a = (ompi_coll_args_t) { .coll_type = OMPI_COLL_TYPE_BARRIER };
}

static inline void
ompi_coll_args_bcast(ompi_coll_args_t *a, void *buff, size_t count,
                     struct ompi_datatype_t *dtype, int root)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_BCAST,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_ROOT,
        .src.info = { .buffer = buff, .count = count, .datatype = dtype },
        .root = root,
    };
}

static inline void
ompi_coll_args_gather(ompi_coll_args_t *a, const void *sbuf, size_t scount,
                      struct ompi_datatype_t *sdtype, void *rbuf, size_t rcount,
                      struct ompi_datatype_t *rdtype, int root)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_GATHER,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST | OMPI_COLL_ARGS_FIELD_ROOT,
        .src.info = { .buffer = (void *) sbuf, .count = scount, .datatype = sdtype },
        .dst.info = { .buffer = rbuf, .count = rcount, .datatype = rdtype },
        .root = root,
    };
}

static inline void
ompi_coll_args_gatherv(ompi_coll_args_t *a, const void *sbuf, size_t scount,
                       struct ompi_datatype_t *sdtype, void *rbuf,
                       ompi_count_array_t rcounts, ompi_disp_array_t disps,
                       struct ompi_datatype_t *rdtype, int root)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_GATHERV,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST | OMPI_COLL_ARGS_FIELD_ROOT,
        .flags = OMPI_COLL_ARGS_FLAG_DST_VECTOR,
        .src.info = { .buffer = (void *) sbuf, .count = scount, .datatype = sdtype },
        .dst.info_v = { .buffer = rbuf, .counts = rcounts, .displacements = disps,
                        .datatype = rdtype },
        .root = root,
    };
}

static inline void
ompi_coll_args_scatter(ompi_coll_args_t *a, const void *sbuf, size_t scount,
                       struct ompi_datatype_t *sdtype, void *rbuf, size_t rcount,
                       struct ompi_datatype_t *rdtype, int root)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_SCATTER,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST | OMPI_COLL_ARGS_FIELD_ROOT,
        .src.info = { .buffer = (void *) sbuf, .count = scount, .datatype = sdtype },
        .dst.info = { .buffer = rbuf, .count = rcount, .datatype = rdtype },
        .root = root,
    };
}

static inline void
ompi_coll_args_scatterv(ompi_coll_args_t *a, const void *sbuf,
                        ompi_count_array_t scounts, ompi_disp_array_t disps,
                        struct ompi_datatype_t *sdtype, void *rbuf, size_t rcount,
                        struct ompi_datatype_t *rdtype, int root)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_SCATTERV,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST | OMPI_COLL_ARGS_FIELD_ROOT,
        .flags = OMPI_COLL_ARGS_FLAG_SRC_VECTOR,
        .src.info_v = { .buffer = (void *) sbuf, .counts = scounts, .displacements = disps,
                        .datatype = sdtype },
        .dst.info = { .buffer = rbuf, .count = rcount, .datatype = rdtype },
        .root = root,
    };
}

static inline void
ompi_coll_args_allgather(ompi_coll_args_t *a, const void *sbuf, size_t scount,
                         struct ompi_datatype_t *sdtype, void *rbuf, size_t rcount,
                         struct ompi_datatype_t *rdtype)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_ALLGATHER,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST,
        .src.info = { .buffer = (void *) sbuf, .count = scount, .datatype = sdtype },
        .dst.info = { .buffer = rbuf, .count = rcount, .datatype = rdtype },
    };
}

static inline void
ompi_coll_args_allgatherv(ompi_coll_args_t *a, const void *sbuf, size_t scount,
                          struct ompi_datatype_t *sdtype, void *rbuf,
                          ompi_count_array_t rcounts, ompi_disp_array_t disps,
                          struct ompi_datatype_t *rdtype)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_ALLGATHERV,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST,
        .flags = OMPI_COLL_ARGS_FLAG_DST_VECTOR,
        .src.info = { .buffer = (void *) sbuf, .count = scount, .datatype = sdtype },
        .dst.info_v = { .buffer = rbuf, .counts = rcounts, .displacements = disps,
                        .datatype = rdtype },
    };
}

static inline void
ompi_coll_args_alltoall(ompi_coll_args_t *a, const void *sbuf, size_t scount,
                        struct ompi_datatype_t *sdtype, void *rbuf, size_t rcount,
                        struct ompi_datatype_t *rdtype)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_ALLTOALL,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST,
        .src.info = { .buffer = (void *) sbuf, .count = scount, .datatype = sdtype },
        .dst.info = { .buffer = rbuf, .count = rcount, .datatype = rdtype },
    };
}

static inline void
ompi_coll_args_alltoallv(ompi_coll_args_t *a, const void *sbuf,
                         ompi_count_array_t scounts, ompi_disp_array_t sdisps,
                         struct ompi_datatype_t *sdtype, void *rbuf,
                         ompi_count_array_t rcounts, ompi_disp_array_t rdisps,
                         struct ompi_datatype_t *rdtype)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_ALLTOALLV,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST,
        .flags = OMPI_COLL_ARGS_FLAG_SRC_VECTOR | OMPI_COLL_ARGS_FLAG_DST_VECTOR,
        .src.info_v = { .buffer = (void *) sbuf, .counts = scounts, .displacements = sdisps,
                        .datatype = sdtype },
        .dst.info_v = { .buffer = rbuf, .counts = rcounts, .displacements = rdisps,
                        .datatype = rdtype },
    };
}

static inline void
ompi_coll_args_alltoallw(ompi_coll_args_t *a, const void *sbuf,
                         ompi_count_array_t scounts, ompi_disp_array_t sdisps,
                         struct ompi_datatype_t * const *sdtypes, void *rbuf,
                         ompi_count_array_t rcounts, ompi_disp_array_t rdisps,
                         struct ompi_datatype_t * const *rdtypes)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_ALLTOALLW,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST,
        .flags = OMPI_COLL_ARGS_FLAG_SRC_VECTOR | OMPI_COLL_ARGS_FLAG_DST_VECTOR
               | OMPI_COLL_ARGS_FLAG_SRC_DTYPE_VEC | OMPI_COLL_ARGS_FLAG_DST_DTYPE_VEC,
        .src.info_v = { .buffer = (void *) sbuf, .counts = scounts, .displacements = sdisps,
                        .datatypes = sdtypes },
        .dst.info_v = { .buffer = rbuf, .counts = rcounts, .displacements = rdisps,
                        .datatypes = rdtypes },
    };
}

static inline void
ompi_coll_args_reduce(ompi_coll_args_t *a, const void *sbuf, void *rbuf, size_t count,
                      struct ompi_datatype_t *dtype, struct ompi_op_t *op, int root)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_REDUCE,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST
              | OMPI_COLL_ARGS_FIELD_OP | OMPI_COLL_ARGS_FIELD_ROOT,
        .src.info = { .buffer = (void *) sbuf, .count = count, .datatype = dtype },
        .dst.info = { .buffer = rbuf, .count = count, .datatype = dtype },
        .op = op,
        .root = root,
    };
}

static inline void
ompi_coll_args_allreduce(ompi_coll_args_t *a, const void *sbuf, void *rbuf, size_t count,
                         struct ompi_datatype_t *dtype, struct ompi_op_t *op)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_ALLREDUCE,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST | OMPI_COLL_ARGS_FIELD_OP,
        .src.info = { .buffer = (void *) sbuf, .count = count, .datatype = dtype },
        .dst.info = { .buffer = rbuf, .count = count, .datatype = dtype },
        .op = op,
    };
}

static inline void
ompi_coll_args_reduce_scatter(ompi_coll_args_t *a, const void *sbuf, void *rbuf,
                              ompi_count_array_t rcounts, struct ompi_datatype_t *dtype,
                              struct ompi_op_t *op)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_REDUCE_SCATTER,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST | OMPI_COLL_ARGS_FIELD_OP,
        .flags = OMPI_COLL_ARGS_FLAG_DST_VECTOR,
        .src.info = { .buffer = (void *) sbuf, .datatype = dtype },
        .dst.info_v = { .buffer = rbuf, .counts = rcounts, .displacements = OMPI_DISP_ARRAY_NULL,
                        .datatype = dtype },
        .op = op,
    };
}

static inline void
ompi_coll_args_reduce_scatter_block(ompi_coll_args_t *a, const void *sbuf, void *rbuf,
                                    size_t rcount, struct ompi_datatype_t *dtype,
                                    struct ompi_op_t *op)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_REDUCE_SCATTER_BLOCK,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST | OMPI_COLL_ARGS_FIELD_OP,
        .src.info = { .buffer = (void *) sbuf, .count = rcount, .datatype = dtype },
        .dst.info = { .buffer = rbuf, .count = rcount, .datatype = dtype },
        .op = op,
    };
}

static inline void
ompi_coll_args_scan(ompi_coll_args_t *a, const void *sbuf, void *rbuf, size_t count,
                    struct ompi_datatype_t *dtype, struct ompi_op_t *op)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_SCAN,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST | OMPI_COLL_ARGS_FIELD_OP,
        .src.info = { .buffer = (void *) sbuf, .count = count, .datatype = dtype },
        .dst.info = { .buffer = rbuf, .count = count, .datatype = dtype },
        .op = op,
    };
}

static inline void
ompi_coll_args_exscan(ompi_coll_args_t *a, const void *sbuf, void *rbuf, size_t count,
                      struct ompi_datatype_t *dtype, struct ompi_op_t *op)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_EXSCAN,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST | OMPI_COLL_ARGS_FIELD_OP,
        .src.info = { .buffer = (void *) sbuf, .count = count, .datatype = dtype },
        .dst.info = { .buffer = rbuf, .count = count, .datatype = dtype },
        .op = op,
    };
}

/* Neighborhood collectives reuse the base-shape builders and then mark the
 * neighborhood type/flag. */
static inline void
ompi_coll_args_neighbor_allgather(ompi_coll_args_t *a, const void *sbuf, size_t scount,
                                  struct ompi_datatype_t *sdtype, void *rbuf, size_t rcount,
                                  struct ompi_datatype_t *rdtype)
{
    ompi_coll_args_allgather(a, sbuf, scount, sdtype, rbuf, rcount, rdtype);
    a->coll_type = OMPI_COLL_TYPE_NEIGHBOR_ALLGATHER;
    a->flags |= OMPI_COLL_ARGS_FLAG_NEIGHBOR;
}

static inline void
ompi_coll_args_neighbor_allgatherv(ompi_coll_args_t *a, const void *sbuf, size_t scount,
                                   struct ompi_datatype_t *sdtype, void *rbuf,
                                   ompi_count_array_t rcounts, ompi_disp_array_t disps,
                                   struct ompi_datatype_t *rdtype)
{
    ompi_coll_args_allgatherv(a, sbuf, scount, sdtype, rbuf, rcounts, disps, rdtype);
    a->coll_type = OMPI_COLL_TYPE_NEIGHBOR_ALLGATHERV;
    a->flags |= OMPI_COLL_ARGS_FLAG_NEIGHBOR;
}

static inline void
ompi_coll_args_neighbor_alltoall(ompi_coll_args_t *a, const void *sbuf, size_t scount,
                                 struct ompi_datatype_t *sdtype, void *rbuf, size_t rcount,
                                 struct ompi_datatype_t *rdtype)
{
    ompi_coll_args_alltoall(a, sbuf, scount, sdtype, rbuf, rcount, rdtype);
    a->coll_type = OMPI_COLL_TYPE_NEIGHBOR_ALLTOALL;
    a->flags |= OMPI_COLL_ARGS_FLAG_NEIGHBOR;
}

static inline void
ompi_coll_args_neighbor_alltoallv(ompi_coll_args_t *a, const void *sbuf,
                                  ompi_count_array_t scounts, ompi_disp_array_t sdisps,
                                  struct ompi_datatype_t *sdtype, void *rbuf,
                                  ompi_count_array_t rcounts, ompi_disp_array_t rdisps,
                                  struct ompi_datatype_t *rdtype)
{
    ompi_coll_args_alltoallv(a, sbuf, scounts, sdisps, sdtype, rbuf, rcounts, rdisps, rdtype);
    a->coll_type = OMPI_COLL_TYPE_NEIGHBOR_ALLTOALLV;
    a->flags |= OMPI_COLL_ARGS_FLAG_NEIGHBOR;
}

static inline void
ompi_coll_args_neighbor_alltoallw(ompi_coll_args_t *a, const void *sbuf,
                                  ompi_count_array_t scounts, ompi_disp_array_t sdisps,
                                  struct ompi_datatype_t * const *sdtypes, void *rbuf,
                                  ompi_count_array_t rcounts, ompi_disp_array_t rdisps,
                                  struct ompi_datatype_t * const *rdtypes)
{
    ompi_coll_args_alltoallw(a, sbuf, scounts, sdisps, sdtypes, rbuf, rcounts, rdisps, rdtypes);
    a->coll_type = OMPI_COLL_TYPE_NEIGHBOR_ALLTOALLW;
    a->flags |= OMPI_COLL_ARGS_FLAG_NEIGHBOR;
}

static inline void
ompi_coll_args_reduce_local(ompi_coll_args_t *a, const void *inbuf, void *inoutbuf,
                            size_t count, struct ompi_datatype_t *dtype,
                            struct ompi_op_t *op)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_REDUCE_LOCAL,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_DST | OMPI_COLL_ARGS_FIELD_OP,
        .src.info = { .buffer = (void *) inbuf, .count = count, .datatype = dtype },
        .dst.info = { .buffer = inoutbuf, .count = count, .datatype = dtype },
        .op = op,
    };
}

static inline void
ompi_coll_args_agree(ompi_coll_args_t *a, void *contrib, size_t dt_count,
                     struct ompi_datatype_t *dtype, struct ompi_op_t *op,
                     struct ompi_group_t **failedgroup, bool update_failedgroup)
{
    *a = (ompi_coll_args_t) {
        .coll_type = OMPI_COLL_TYPE_AGREE,
        .mask = OMPI_COLL_ARGS_FIELD_SRC | OMPI_COLL_ARGS_FIELD_OP
              | OMPI_COLL_ARGS_FIELD_FAILEDGROUP,
        .flags = update_failedgroup ? OMPI_COLL_ARGS_FLAG_UPDATE_FAILEDGROUP : 0,
        .src.info = { .buffer = contrib, .count = dt_count, .datatype = dtype },
        .op = op,
        .failedgroup = failedgroup,
    };
}

static inline void
ompi_coll_args_revoke_local(ompi_coll_args_t *a)
{
    *a = (ompi_coll_args_t) { .coll_type = OMPI_COLL_TYPE_REVOKE_LOCAL };
}

END_C_DECLS

#endif /* OMPI_MCA_COLL_COLL_ARGS_H */
