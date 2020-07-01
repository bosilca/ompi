/*
 * Copyright (c) 2019      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * Copyright (c) 2019      Arm Ltd.  All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#include "opal/util/output.h"

#include "ompi/op/op.h"
#include "ompi/mca/op/op.h"
#include "ompi/mca/op/base/base.h"
#include "ompi/mca/op/sve/op_sve.h"
#include "ompi/mca/op/sve/op_sve_functions.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif /* __ARM_FEATURE_SVE */

/*
 * Since all the functions in this file are essentially identical, we
 * use a macro to substitute in names and types.  The core operation
 * in all functions that use this macro is the same.
 *
 * This macro is for (out op in).
 *
 */
#define OP_SVE_FUNC(name, type_name, type_size, type, op) \
    static void ompi_op_sve_2buff_##name##_##type(void *_in, void *_out, int *count, \
            struct ompi_datatype_t **dtype, \
            struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int types_per_step = svcnt##type_name();                           \
    int left_over = *count; \
    type* in = (type*)_in; \
    type* out = (type*)_out; \
    svbool_t Pg = svptrue_b##type_size(); \
    for (; left_over >= types_per_step; left_over -= types_per_step) { \
        sv##type  vsrc = svld1(Pg, in);                                \
        sv##type  vdst = svld1(Pg, out);                               \
        in += types_per_step;                                          \
        vdst=sv##op##_z(Pg,vdst,vsrc);                                 \
        svst1(Pg, out,vdst);                                           \
        out += types_per_step; \
    }                                                                  \
    \
    while( left_over > 0 ) {                        \
        int how_much = (left_over > 8) ? 8 : left_over; \
        switch(left_over) {                         \
            case 8: out[7] = current_func(out[7],in[7]) ;                        \
            case 7: out[6] = current_func(out[6],in[6]) ;                        \
            case 6: out[5] = current_func(out[5],in[5]) ;                        \
            case 5: out[4] = current_func(out[4],in[4]) ;                        \
            case 4: out[3] = current_func(out[3],in[3]) ;                        \
            case 3: out[2] = current_func(out[2],in[2]) ;                        \
            case 2: out[1] = current_func(out[1],in[1]) ;                        \
            case 1: out[0] = current_func(out[0],in[0]) ;                        \
        }\
        left_over -= how_much;                 \
        out += how_much;                       \
        in += how_much;                        \
    } \
}

/*************************************************************************
 * Max
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) > (b) ? (a) : (b))
    OP_SVE_FUNC(max, b,  8,   int8_t, max)
    OP_SVE_FUNC(max, b,   8,  uint8_t, max)
    OP_SVE_FUNC(max, h,  16,  int16_t, max)
    OP_SVE_FUNC(max, h, 16, uint16_t, max)
    OP_SVE_FUNC(max, w,  32,  int32_t, max)
    OP_SVE_FUNC(max, w, 32, uint32_t, max)
    OP_SVE_FUNC(max, d,  64,  int64_t, max)
    OP_SVE_FUNC(max, d, 64, uint64_t, max)

    OP_SVE_FUNC(max, w, 32, float32_t, max)
    OP_SVE_FUNC(max, d, 64, float64_t, max)

/*************************************************************************
 * Min
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) < (b) ? (a) : (b))
    OP_SVE_FUNC(min, b,  8,   int8_t, min)
    OP_SVE_FUNC(min, b,   8,  uint8_t, min)
    OP_SVE_FUNC(min, h,  16,  int16_t, min)
    OP_SVE_FUNC(min, h, 16, uint16_t, min)
    OP_SVE_FUNC(min, w,  32,  int32_t, min)
    OP_SVE_FUNC(min, w, 32, uint32_t, min)
    OP_SVE_FUNC(min, d,  64,  int64_t, min)
    OP_SVE_FUNC(min, d, 64, uint64_t, min)

    OP_SVE_FUNC(min, w, 32, float32_t, min)
    OP_SVE_FUNC(min, d, 64, float64_t, min)

 /*************************************************************************
 * Sum
 ************************************************************************/
#undef current_func
#define current_func(a, b) ((a) + (b))
    OP_SVE_FUNC(sum, b,  8,   int8_t, add)
    OP_SVE_FUNC(sum, b,   8,  uint8_t, add)
    OP_SVE_FUNC(sum, h,  16,  int16_t, add)
    OP_SVE_FUNC(sum, h, 16, uint16_t, add)
    OP_SVE_FUNC(sum, w,  32,  int32_t, add)
    OP_SVE_FUNC(sum, w, 32, uint32_t, add)
    OP_SVE_FUNC(sum, d,  64,  int64_t, add)
    OP_SVE_FUNC(sum, d, 64, uint64_t, add)

    OP_SVE_FUNC(sum, w, 32, float32_t, add)
    OP_SVE_FUNC(sum, d, 64, float64_t, add)

/*************************************************************************
 * Product
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) * (b))
    OP_SVE_FUNC(prod, b,  8,   int8_t, mul)
    OP_SVE_FUNC(prod, b,   8,  uint8_t, mul)
    OP_SVE_FUNC(prod, h,  16,  int16_t, mul)
    OP_SVE_FUNC(prod, h, 16, uint16_t, mul)
    OP_SVE_FUNC(prod, w,  32,  int32_t, mul)
    OP_SVE_FUNC(prod, w, 32, uint32_t, mul)
    OP_SVE_FUNC(prod, d,  64,  int64_t, mul)
    OP_SVE_FUNC(prod, d, 64, uint64_t, mul)

    OP_SVE_FUNC(prod, w, 32, float32_t, mul)
    OP_SVE_FUNC(prod, d, 64, float64_t, mul)

/*************************************************************************
 * Bitwise AND
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) & (b))
    OP_SVE_FUNC(band, b,  8,   int8_t, and)
    OP_SVE_FUNC(band, b,   8,  uint8_t, and)
    OP_SVE_FUNC(band, h,  16,  int16_t, and)
    OP_SVE_FUNC(band, h, 16, uint16_t, and)
    OP_SVE_FUNC(band, w,  32,  int32_t, and)
    OP_SVE_FUNC(band, w, 32, uint32_t, and)
    OP_SVE_FUNC(band, d,  64,  int64_t, and)
OP_SVE_FUNC(band, d, 64, uint64_t, and)

 /*************************************************************************
 * Bitwise OR
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) | (b))
    OP_SVE_FUNC(bor, b,  8,   int8_t, orr)
    OP_SVE_FUNC(bor, b,   8,  uint8_t, orr)
    OP_SVE_FUNC(bor, h,  16,  int16_t, orr)
    OP_SVE_FUNC(bor, h, 16, uint16_t, orr)
    OP_SVE_FUNC(bor, w,  32,  int32_t, orr)
    OP_SVE_FUNC(bor, w, 32, uint32_t, orr)
    OP_SVE_FUNC(bor, d,  64,  int64_t, orr)
OP_SVE_FUNC(bor, d, 64, uint64_t, orr)

/*************************************************************************
 * Bitwise XOR
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) ^ (b))
    OP_SVE_FUNC(bxor, b,  8,   int8_t, eor)
    OP_SVE_FUNC(bxor, b,   8,  uint8_t, eor)
    OP_SVE_FUNC(bxor, h,  16,  int16_t, eor)
    OP_SVE_FUNC(bxor, h, 16, uint16_t, eor)
    OP_SVE_FUNC(bxor, w,  32,  int32_t, eor)
    OP_SVE_FUNC(bxor, w, 32, uint32_t, eor)
    OP_SVE_FUNC(bxor, d,  64,  int64_t, eor)
OP_SVE_FUNC(bxor, d, 64, uint64_t, eor)

/*
 *  This is a three buffer (2 input and 1 output) version of the reduction
 *  routines, needed for some optimizations.
 */
#define OP_SVE_FUNC_3BUFF(name, type_name, type_size, type, op) \
        static void ompi_op_sve_3buff_##name##_##type(void * restrict _in1,   \
                void * restrict _in2, void * restrict _out, int *count, \
                struct ompi_datatype_t **dtype, \
                struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int types_per_step = svcnt##type_name();                           \
    int left_over = *count; \
    type* in1 = (type*)_in1; \
    type* in2 = (type*)_in2; \
    type* out = (type*)_out; \
    svbool_t Pg = svptrue_b##type_size(); \
    for (; left_over >= types_per_step; left_over -= types_per_step) { \
        sv##type  vsrc = svld1(Pg, in1);                               \
        sv##type  vdst = svld1(Pg, in2);                               \
        in1 += types_per_step; \
        in2 += types_per_step; \
        vdst=sv##op##_z(Pg,vdst,vsrc);                                 \
        svst1(Pg, out,vdst);                                           \
        out += types_per_step; \
    }                                                                  \
    if (left_over !=0){                                                \
        Pg = svwhilelt_b##type_size##_u64(0, left_over);               \
        sv##type  vsrc = svld1(Pg, in1);                               \
        sv##type  vdst = svld1(Pg, in2);                               \
        vdst=sv##op##_z(Pg,vdst,vsrc);                                 \
        svst1(Pg, out,vdst);                                           \
    } \
}

/*************************************************************************
 * Max
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(max, b,  8,   int8_t, max)
    OP_SVE_FUNC_3BUFF(max, b,   8,  uint8_t, max)
    OP_SVE_FUNC_3BUFF(max, h,  16,  int16_t, max)
    OP_SVE_FUNC_3BUFF(max, h, 16, uint16_t, max)
    OP_SVE_FUNC_3BUFF(max, w,  32,  int32_t, max)
    OP_SVE_FUNC_3BUFF(max, w, 32, uint32_t, max)
    OP_SVE_FUNC_3BUFF(max, d,  64,  int64_t, max)
    OP_SVE_FUNC_3BUFF(max, d, 64, uint64_t, max)

    OP_SVE_FUNC_3BUFF(max, w, 32, float32_t, max)
    OP_SVE_FUNC_3BUFF(max, d, 64, float64_t, max)

/*************************************************************************
 * Min
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(min, b,  8,   int8_t, min)
    OP_SVE_FUNC_3BUFF(min, b,   8,  uint8_t, min)
    OP_SVE_FUNC_3BUFF(min, h,  16,  int16_t, min)
    OP_SVE_FUNC_3BUFF(min, h, 16, uint16_t, min)
    OP_SVE_FUNC_3BUFF(min, w,  32,  int32_t, min)
    OP_SVE_FUNC_3BUFF(min, w, 32, uint32_t, min)
    OP_SVE_FUNC_3BUFF(min, d,  64,  int64_t, min)
    OP_SVE_FUNC_3BUFF(min, d, 64, uint64_t, min)

    OP_SVE_FUNC_3BUFF(min, w, 32, float32_t, min)
    OP_SVE_FUNC_3BUFF(min, d, 64, float64_t, min)

 /*************************************************************************
 * Sum
 ************************************************************************/
    OP_SVE_FUNC_3BUFF(sum, b,  8,   int8_t, add)
    OP_SVE_FUNC_3BUFF(sum, b,   8,  uint8_t, add)
    OP_SVE_FUNC_3BUFF(sum, h,  16,  int16_t, add)
    OP_SVE_FUNC_3BUFF(sum, h, 16, uint16_t, add)
    OP_SVE_FUNC_3BUFF(sum, w,  32,  int32_t, add)
    OP_SVE_FUNC_3BUFF(sum, w, 32, uint32_t, add)
    OP_SVE_FUNC_3BUFF(sum, d,  64,  int64_t, add)
    OP_SVE_FUNC_3BUFF(sum, d, 64, uint64_t, add)

    OP_SVE_FUNC_3BUFF(sum, w, 32, float32_t, add)
    OP_SVE_FUNC_3BUFF(sum, d, 64, float64_t, add)

/*************************************************************************
 * Product
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(prod, b,  8,   int8_t, mul)
    OP_SVE_FUNC_3BUFF(prod, b,   8,  uint8_t, mul)
    OP_SVE_FUNC_3BUFF(prod, h,  16,  int16_t, mul)
    OP_SVE_FUNC_3BUFF(prod, h, 16, uint16_t, mul)
    OP_SVE_FUNC_3BUFF(prod, w,  32,  int32_t, mul)
    OP_SVE_FUNC_3BUFF(prod, w, 32, uint32_t, mul)
    OP_SVE_FUNC_3BUFF(prod, d,  64,  int64_t, mul)
    OP_SVE_FUNC_3BUFF(prod, d, 64, uint64_t, mul)

    OP_SVE_FUNC_3BUFF(prod, w, 32, float32_t, mul)
    OP_SVE_FUNC_3BUFF(prod, d, 64, float64_t, mul)

/*************************************************************************
 * Bitwise AND
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(band, b,  8,   int8_t, and)
    OP_SVE_FUNC_3BUFF(band, b,   8,  uint8_t, and)
    OP_SVE_FUNC_3BUFF(band, h,  16,  int16_t, and)
    OP_SVE_FUNC_3BUFF(band, h, 16, uint16_t, and)
    OP_SVE_FUNC_3BUFF(band, w,  32,  int32_t, and)
    OP_SVE_FUNC_3BUFF(band, w, 32, uint32_t, and)
    OP_SVE_FUNC_3BUFF(band, d,  64,  int64_t, and)
    OP_SVE_FUNC_3BUFF(band, d, 64, uint64_t, and)

 /*************************************************************************
 * Bitwise OR
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(bor, b,  8,   int8_t, orr)
    OP_SVE_FUNC_3BUFF(bor, b,   8,  uint8_t, orr)
    OP_SVE_FUNC_3BUFF(bor, h,  16,  int16_t, orr)
    OP_SVE_FUNC_3BUFF(bor, h, 16, uint16_t, orr)
    OP_SVE_FUNC_3BUFF(bor, w,  32,  int32_t, orr)
    OP_SVE_FUNC_3BUFF(bor, w, 32, uint32_t, orr)
    OP_SVE_FUNC_3BUFF(bor, d,  64,  int64_t, orr)
    OP_SVE_FUNC_3BUFF(bor, d, 64, uint64_t, orr)

/*************************************************************************
 * Bitwise XOR
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(bxor, b,  8,   int8_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, b,   8,  uint8_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, h,  16,  int16_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, h, 16, uint16_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, w,  32,  int32_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, w, 32, uint32_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, d,  64,  int64_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, d, 64, uint64_t, eor)

/** C integer ***********************************************************/
#define C_INTEGER(name, ftype)                                              \
    [OMPI_OP_BASE_TYPE_INT8_T] = ompi_op_sve_##ftype##_##name##_int8_t,     \
    [OMPI_OP_BASE_TYPE_UINT8_T] = ompi_op_sve_##ftype##_##name##_uint8_t,   \
    [OMPI_OP_BASE_TYPE_INT16_T] = ompi_op_sve_##ftype##_##name##_int16_t,   \
    [OMPI_OP_BASE_TYPE_UINT16_T] = ompi_op_sve_##ftype##_##name##_uint16_t, \
    [OMPI_OP_BASE_TYPE_INT32_T] = ompi_op_sve_##ftype##_##name##_int32_t,   \
    [OMPI_OP_BASE_TYPE_UINT32_T] = ompi_op_sve_##ftype##_##name##_uint32_t, \
    [OMPI_OP_BASE_TYPE_INT64_T] = ompi_op_sve_##ftype##_##name##_int64_t,   \
    [OMPI_OP_BASE_TYPE_UINT64_T] = ompi_op_sve_##ftype##_##name##_uint64_t


/** Floating point, including all the Fortran reals *********************/
#define FLOAT(name, ftype) ompi_op_sve_##ftype##_##name##_float32_t
#define DOUBLE(name, ftype) ompi_op_sve_##ftype##_##name##_float64_t

#define FLOATING_POINT(name, ftype)                                        \
    [OMPI_OP_BASE_TYPE_FLOAT] = FLOAT(name, ftype),                        \
    [OMPI_OP_BASE_TYPE_DOUBLE] = DOUBLE(name, ftype)

/*
 * MPI_OP_NULL
 * All types
 */
#define FLAGS_NO_FLOAT \
        (OMPI_OP_FLAGS_INTRINSIC | OMPI_OP_FLAGS_ASSOC | OMPI_OP_FLAGS_COMMUTE)
#define FLAGS \
        (OMPI_OP_FLAGS_INTRINSIC | OMPI_OP_FLAGS_ASSOC | \
         OMPI_OP_FLAGS_FLOAT_ASSOC | OMPI_OP_FLAGS_COMMUTE)

ompi_op_base_handler_fn_t ompi_op_sve_functions[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX] =
{
    /* Corresponds to MPI_OP_NULL */
    [OMPI_OP_BASE_FORTRAN_NULL] = {
        /* Leaving this empty puts in NULL for all entries */
        NULL,
    },
    /* Corresponds to MPI_MAX */
    [OMPI_OP_BASE_FORTRAN_MAX] = {
        C_INTEGER(max, 2buff),
        FLOATING_POINT(max, 2buff),
    },
    /* Corresponds to MPI_MIN */
    [OMPI_OP_BASE_FORTRAN_MIN] = {
        C_INTEGER(min, 2buff),
        FLOATING_POINT(min, 2buff),
    },
    /* Corresponds to MPI_SUM */
    [OMPI_OP_BASE_FORTRAN_SUM] = {
        C_INTEGER(sum, 2buff),
        FLOATING_POINT(sum, 2buff),
    },
    /* Corresponds to MPI_PROD */
    [OMPI_OP_BASE_FORTRAN_PROD] = {
        C_INTEGER(prod, 2buff),
        FLOATING_POINT(prod, 2buff),
    },
    /* Corresponds to MPI_LAND */
    [OMPI_OP_BASE_FORTRAN_LAND] = {
        NULL,
    },
    /* Corresponds to MPI_BAND */
    [OMPI_OP_BASE_FORTRAN_BAND] = {
        C_INTEGER(band, 2buff),
    },
    /* Corresponds to MPI_LOR */
    [OMPI_OP_BASE_FORTRAN_LOR] = {
        NULL,
    },
    /* Corresponds to MPI_BOR */
    [OMPI_OP_BASE_FORTRAN_BOR] = {
        C_INTEGER(bor, 2buff),
    },
    /* Corresponds to MPI_LXOR */
    [OMPI_OP_BASE_FORTRAN_LXOR] = {
        NULL,
    },
    /* Corresponds to MPI_BXOR */
    [OMPI_OP_BASE_FORTRAN_BXOR] = {
        C_INTEGER(bxor, 2buff),
    },
    /* Corresponds to MPI_REPLACE */
    [OMPI_OP_BASE_FORTRAN_REPLACE] = {
        /* (MPI_ACCUMULATE is handled differently than the other
           reductions, so just zero out its function
           implementations here to ensure that users don't invoke
           MPI_REPLACE with any reduction operations other than
           ACCUMULATE) */
        NULL,
    },

};

ompi_op_base_3buff_handler_fn_t ompi_op_sve_3buff_functions[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX] =
{
    /* Corresponds to MPI_OP_NULL */
    [OMPI_OP_BASE_FORTRAN_NULL] = {
        /* Leaving this empty puts in NULL for all entries */
        NULL,
    },
    /* Corresponds to MPI_MAX */
    [OMPI_OP_BASE_FORTRAN_MAX] = {
        C_INTEGER(max, 3buff),
        FLOATING_POINT(max, 3buff),
    },
    /* Corresponds to MPI_MIN */
    [OMPI_OP_BASE_FORTRAN_MIN] = {
        C_INTEGER(min, 3buff),
        FLOATING_POINT(min, 3buff),
    },
    /* Corresponds to MPI_SUM */
    [OMPI_OP_BASE_FORTRAN_SUM] = {
        C_INTEGER(sum, 3buff),
        FLOATING_POINT(sum, 3buff),
    },
    /* Corresponds to MPI_PROD */
    [OMPI_OP_BASE_FORTRAN_PROD] = {
        C_INTEGER(prod, 3buff),
        FLOATING_POINT(prod, 3buff),
    },
    /* Corresponds to MPI_LAND */
    [OMPI_OP_BASE_FORTRAN_LAND] ={
        NULL,
    },
    /* Corresponds to MPI_BAND */
    [OMPI_OP_BASE_FORTRAN_BAND] = {
        C_INTEGER(band, 3buff),
    },
    /* Corresponds to MPI_LOR */
    [OMPI_OP_BASE_FORTRAN_LOR] = {
        NULL,
    },
    /* Corresponds to MPI_BOR */
    [OMPI_OP_BASE_FORTRAN_BOR] = {
        C_INTEGER(bor, 3buff),
    },
    /* Corresponds to MPI_LXOR */
    [OMPI_OP_BASE_FORTRAN_LXOR] = {
        NULL,
    },
    /* Corresponds to MPI_BXOR */
    [OMPI_OP_BASE_FORTRAN_BXOR] = {
        C_INTEGER(bxor, 3buff),
    },
    /* Corresponds to MPI_REPLACE */
    [OMPI_OP_BASE_FORTRAN_REPLACE] = {
        /* MPI_ACCUMULATE is handled differently than the other
           reductions, so just zero out its function
           implementations here to ensure that users don't invoke
           MPI_REPLACE with any reduction operations other than
           ACCUMULATE */
        NULL,
    },
};
