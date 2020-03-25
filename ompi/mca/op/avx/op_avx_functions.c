/*
 * Copyright (c) 2019-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2020      Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
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
#include "ompi/mca/op/avx/op_avx.h"
#include "ompi/mca/op/avx/op_avx_functions.h"

#include <immintrin.h>
/*
 * Since all the functions in this file are essentially identical, we
 * use a macro to substitute in names and types.  The core operation
 * in all functions that use this macro is the same.
 *
 * This macro is for (out op in).
 *
 * Support ops: max, min, for signed/unsigned 8,16,32,64
 *              sum, for integer 8,16,32,64
 *
 */

#define OMPI_OP_AVX_HAS_FLAGS(_flag) \
  (((_flag) & mca_op_avx_component.flags) == (_flag))

#if defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512)
#define OP_AVX_AVX512_FUNC(name, type_sign, type_size, type, op)               \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX512F_FLAG|OMPI_OP_AVX_HAS_AVX512BW_FLAG) ) { \
        types_per_step = (512 / 8) / sizeof(type);                             \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) {    \
            __m512i vecA =  _mm512_loadu_si512((__m512*)in);                   \
            in += types_per_step;                                              \
            __m512i vecB =  _mm512_loadu_si512((__m512*)out);                  \
            __m512i res = _mm512_##op##_ep##type_sign##type_size(vecA, vecB);  \
            _mm512_storeu_si512((__m512*)out, res);                            \
            out += types_per_step;                                             \
        }                                                                      \
        if( 0 == left_over ) return;                                           \
    }
#else
#define OP_AVX_AVX512_FUNC(name, type_sign, type_size, type, op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512) */

#if defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2)
#define OP_AVX_AVX2_FUNC(name, type_sign, type_size, type, op)                 \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX2_FLAG | OMPI_OP_AVX_HAS_AVX_FLAG) ) {  \
        types_per_step = (256 / 8) / sizeof(type);  /* AVX2 */                 \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) {    \
            __m256i vecA = _mm256_loadu_si256((__m256i*)in);                   \
            in += types_per_step;                                              \
            __m256i vecB = _mm256_loadu_si256((__m256i*)out);                  \
            __m256i res =  _mm256_##op##_ep##type_sign##type_size(vecA, vecB); \
            _mm256_storeu_si256((__m256i*)out, res);                           \
            out += types_per_step;                                             \
        }                                                                      \
        if( 0 == left_over ) return;                                           \
    }
#else
#define OP_AVX_AVX2_FUNC(name, type_sign, type_size, type, op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2) */

#if defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX)
#define OP_AVX_SSE4_1_FUNC(name, type_sign, type_size, type, op)               \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_SSE3_FLAG | OMPI_OP_AVX_HAS_SSE4_1_FLAG) ) { \
        types_per_step = (128 / 8) / sizeof(type);  /* AVX */                  \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) {    \
            __m128i vecA = _mm_lddqu_si128((__m128i*)in);                      \
            in += types_per_step;                                              \
            __m128i vecB = _mm_lddqu_si128((__m128i*)out);                     \
            __m128i res =  _mm_##op##_ep##type_sign##type_size(vecA, vecB);    \
            _mm_storeu_si128((__m128i*)out, res);                              \
            out += types_per_step;                                             \
        }                                                                      \
    }
#else
#define OP_AVX_SSE4_1_FUNC(name, type_sign, type_size, type, op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX) */

#define OP_AVX_FUNC(name, type_sign, type_size, type, op)                      \
static void ompi_op_avx_2buff_##name##_##type(const void *_in, void *_out, int *count, \
                                              struct ompi_datatype_t **dtype,  \
                                              struct ompi_op_base_module_1_0_0_t *module) \
{                                                                              \
    int types_per_step, left_over = *count;                                    \
    type *in = (type*)_in, *out = (type*)_out;                                 \
    OP_AVX_AVX512_FUNC(name, type_sign, type_size, type, op);                  \
    OP_AVX_AVX2_FUNC(name, type_sign, type_size, type, op);                    \
    OP_AVX_SSE4_1_FUNC(name, type_sign, type_size, type, op);                  \
    while( left_over > 0 ) {                                                   \
        int how_much = (left_over > 8) ? 8 : left_over;                        \
        switch(how_much) {                                                     \
        case 8: out[7] = current_func(out[7], in[7]);                          \
        case 7: out[6] = current_func(out[6], in[6]);                          \
        case 6: out[5] = current_func(out[5], in[5]);                          \
        case 5: out[4] = current_func(out[4], in[4]);                          \
        case 4: out[3] = current_func(out[3], in[3]);                          \
        case 3: out[2] = current_func(out[2], in[2]);                          \
        case 2: out[1] = current_func(out[1], in[1]);                          \
        case 1: out[0] = current_func(out[0], in[0]);                          \
        }                                                                      \
        left_over -= how_much;                                                 \
        out += how_much;                                                       \
        in += how_much;                                                        \
    }                                                                          \
}

#if defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512)
#define OP_AVX512_FUNC(name, type_sign, type_size, type, op)                   \
        OP_AVX_FUNC(name, type_sign, type_size, type, op)
#else
#define OP_AVX512_FUNC(name, type_sign, type_size, type, op)
#endif

#if defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512)
#define OP_AVX_AVX512_MUL(name, type_sign, type_size, type, op)         \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX512F_FLAG | OMPI_OP_AVX_HAS_AVX512BW_FLAG) ) {  \
        int types_per_step = (256 / 8) / sizeof(type);                  \
        for (; left_over >= types_per_step; left_over -= types_per_step) { \
            __m256i vecA_tmp =  _mm256_loadu_si256((__m256i*)in);       \
            __m256i vecB_tmp =  _mm256_loadu_si256((__m256i*)out);      \
            in += types_per_step;                                       \
            __m512i vecA = _mm512_cvtepi8_epi16(vecA_tmp);              \
            __m512i vecB = _mm512_cvtepi8_epi16(vecB_tmp);              \
            __m512i res = _mm512_##op##_ep##type_sign##16(vecA, vecB);  \
            vecB_tmp = _mm512_cvtepi16_epi8(res);                       \
            _mm256_storeu_si256((__m256i*)out, vecB_tmp);               \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
    }
#else
#define OP_AVX_AVX512_MUL(name, type_sign, type_size, type, op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512) */
/**
 * There is no support for 16 to 8 conversion without AVX512BW and AVX512VL, so
 * there is no AVX-only optimized function posible for OP_AVX_AVX2_MUL.
 */

/* special case for int8 mul */
#define OP_AVX_MUL(name, type_sign, type_size, type, op)                \
static void ompi_op_avx_2buff_##name##_##type(const void *_in, void *_out, int *count, \
                                              struct ompi_datatype_t **dtype, \
                                              struct ompi_op_base_module_1_0_0_t *module) \
{                                                                       \
    int left_over = *count;                                             \
    type *in = (type*)_in, *out = (type*)_out;                          \
    OP_AVX_AVX512_MUL(name, type_sign, type_size, type, op);            \
    while( left_over > 0 ) {                                            \
        int how_much = (left_over > 8) ? 8 : left_over;                 \
        switch(how_much) {                                              \
        case 8: out[7] = current_func(out[7], in[7]);                   \
        case 7: out[6] = current_func(out[6], in[6]);                   \
        case 6: out[5] = current_func(out[5], in[5]);                   \
        case 5: out[4] = current_func(out[4], in[4]);                   \
        case 4: out[3] = current_func(out[3], in[3]);                   \
        case 3: out[2] = current_func(out[2], in[2]);                   \
        case 2: out[1] = current_func(out[1], in[1]);                   \
        case 1: out[0] = current_func(out[0], in[0]);                   \
        }                                                               \
        left_over -= how_much;                                          \
        out += how_much;                                                \
        in += how_much;                                                 \
    }                                                                   \
}

/*
 *  This macro is for bit-wise operations (out op in).
 *
 *  Support ops: or, xor, and of 512 bits (representing integer data)
 *
 */
#if defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512)
#define OP_AVX_AVX512_BIT_FUNC(name, type_size, type, op)               \
    if( OMPI_OP_AVX_HAS_FLAGS( OMPI_OP_AVX_HAS_AVX512F_FLAG) ) {        \
        types_per_step = (512 / 8) / sizeof(type);                      \
        for (; left_over >= types_per_step; left_over -= types_per_step) { \
            __m512i vecA =  _mm512_loadu_si512((__m512i*)in);           \
            in += types_per_step;                                       \
            __m512i vecB =  _mm512_loadu_si512((__m512i*)out);          \
            __m512i res = _mm512_##op##_si512(vecA, vecB);              \
            _mm512_storeu_si512((__m512i*)out, res);                    \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
    }
#else
#define OP_AVX_AVX512_BIT_FUNC(name, type_size, type, op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512) */

#if defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2)
#define OP_AVX_AVX2_BIT_FUNC(name, type_size, type, op)                 \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX2_FLAG | OMPI_OP_AVX_HAS_AVX_FLAG) ) { \
        types_per_step = (256 / 8) / sizeof(type);                      \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) { \
            __m256i vecA = _mm256_loadu_si256((__m256i*)in);            \
            in += types_per_step;                                       \
            __m256i vecB = _mm256_loadu_si256((__m256i*)out);           \
            __m256i res =  _mm256_##op##_si256(vecA, vecB);             \
            _mm256_storeu_si256((__m256i*)out, res);                    \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
    }
#else
#define OP_AVX_AVX2_BIT_FUNC(name, type_size, type, op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2) */

#if defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX)
#define OP_AVX_SSE3_BIT_FUNC(name, type_size, type, op)                 \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_SSE3_FLAG) ) {            \
        types_per_step = (128 / 8) / sizeof(type);                      \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) { \
            __m128i vecA = _mm_lddqu_si128((__m128i*)in);               \
            in += types_per_step;                                       \
            __m128i vecB = _mm_lddqu_si128((__m128i*)out);              \
            __m128i res =  _mm_##op##_si128(vecA, vecB);                \
            _mm_storeu_si128((__m128i*)out, res);                       \
            out += types_per_step;                                      \
        }                                                               \
    }
#else
#define OP_AVX_SSE3_BIT_FUNC(name, type_size, type, op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX) */

#define OP_AVX_BIT_FUNC(name, type_size, type, op)                      \
static void ompi_op_avx_2buff_##name##_##type(const void *_in, void *_out, int *count, \
                                              struct ompi_datatype_t **dtype, \
                                              struct ompi_op_base_module_1_0_0_t *module) \
{                                                                       \
    int types_per_step, left_over = *count;                             \
    type *in = (type*)_in, *out = (type*)_out;                          \
    OP_AVX_AVX512_BIT_FUNC(name, type_size, type, op);                  \
    OP_AVX_AVX2_BIT_FUNC(name, type_size, type, op);                    \
    OP_AVX_SSE3_BIT_FUNC(name, type_size, type, op);                    \
    while( left_over > 0 ) {                                            \
        int how_much = (left_over > 8) ? 8 : left_over;                 \
        switch(how_much) {                                              \
        case 8: out[7] = current_func(out[7], in[7]);                   \
        case 7: out[6] = current_func(out[6], in[6]);                   \
        case 6: out[5] = current_func(out[5], in[5]);                   \
        case 5: out[4] = current_func(out[4], in[4]);                   \
        case 4: out[3] = current_func(out[3], in[3]);                   \
        case 3: out[2] = current_func(out[2], in[2]);                   \
        case 2: out[1] = current_func(out[1], in[1]);                   \
        case 1: out[0] = current_func(out[0], in[0]);                   \
        }                                                               \
        left_over -= how_much;                                          \
        out += how_much;                                                \
        in += how_much;                                                 \
    }                                                                   \
}

#if defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512)
#define OP_AVX_AVX512_FLOAT_FUNC(op)                                    \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX512F_FLAG) ) {         \
        types_per_step = (512 / 8) / sizeof(float);                     \
        for (; left_over >= types_per_step; left_over -= types_per_step) { \
            __m512 vecA =  _mm512_load_ps((__m512*)in);                 \
            __m512 vecB =  _mm512_load_ps((__m512*)out);                \
            in += types_per_step;                                       \
            __m512 res = _mm512_##op##_ps(vecA, vecB);                  \
            _mm512_store_ps((__m512*)out, res);                         \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
    }
#else
#define OP_AVX_AVX512_FLOAT_FUNC(op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512) */

#if defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2)
#define OP_AVX_AVX_FLOAT_FUNC(op)                                       \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX_FLAG) ) {             \
        types_per_step = (256 / 8) / sizeof(float);                     \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) { \
            __m256 vecA =  _mm256_load_ps(in);                          \
            in += types_per_step;                                       \
            __m256 vecB =  _mm256_load_ps(out);                         \
            __m256 res = _mm256_##op##_ps(vecA, vecB);                  \
            _mm256_store_ps(out, res);                                  \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
    }
#else
#define OP_AVX_AVX_FLOAT_FUNC(op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2) */

#if defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX)
#define OP_AVX_SSE_FLOAT_FUNC(op)                                       \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_SSE_FLAG) ) {             \
        types_per_step = (128 / 8) / sizeof(float);                     \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) { \
            __m128 vecA = _mm_load_ps(in);                              \
            in += types_per_step;                                       \
            __m128 vecB = _mm_load_ps(out);                             \
            __m128 res = _mm_##op##_ps(vecA, vecB);                     \
            _mm_store_ps(out, res);                                     \
            out += types_per_step;                                      \
        }                                                               \
    }
#else
#define OP_AVX_SSE_FLOAT_FUNC(op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX) */

#define OP_AVX_FLOAT_FUNC(op) \
static void ompi_op_avx_2buff_##op##_float(const void *_in, void *_out, int *count, \
                                           struct ompi_datatype_t **dtype, \
                                           struct ompi_op_base_module_1_0_0_t *module) \
{                                                                       \
    int types_per_step, left_over = *count;                             \
    float *in = (float*)_in, *out = (float*)_out;                       \
    OP_AVX_AVX512_FLOAT_FUNC(op);                                       \
    OP_AVX_AVX_FLOAT_FUNC(op);                                          \
    OP_AVX_SSE_FLOAT_FUNC(op);                                          \
    while( left_over > 0 ) {                                            \
        int how_much = (left_over > 8) ? 8 : left_over;                 \
        switch(how_much) {                                              \
        case 8: out[7] = current_func(out[7], in[7]);                   \
        case 7: out[6] = current_func(out[6], in[6]);                   \
        case 6: out[5] = current_func(out[5], in[5]);                   \
        case 5: out[4] = current_func(out[4], in[4]);                   \
        case 4: out[3] = current_func(out[3], in[3]);                   \
        case 3: out[2] = current_func(out[2], in[2]);                   \
        case 2: out[1] = current_func(out[1], in[1]);                   \
        case 1: out[0] = current_func(out[0], in[0]);                   \
        }                                                               \
        left_over -= how_much;                                          \
        out += how_much;                                                \
        in += how_much;                                                 \
    }                                                                   \
}

#if defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512)
#define OP_AVX_AVX512_DOUBLE_FUNC(op)                                   \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX512F_FLAG) ) {         \
        types_per_step = (512 / 8)  / sizeof(double);                   \
        for (; left_over >= types_per_step; left_over -= types_per_step) { \
            __m512d vecA =  _mm512_load_pd(in);                         \
            in += types_per_step;                                       \
            __m512d vecB =  _mm512_load_pd(out);                        \
            __m512d res = _mm512_##op##_pd(vecA, vecB);                 \
            _mm512_store_pd((out), res);                                \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
    }
#else
#define OP_AVX_AVX512_DOUBLE_FUNC(op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512) */

#if defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2)
#define OP_AVX_AVX_DOUBLE_FUNC(op)                                      \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX_FLAG) ) {             \
        types_per_step = (256 / 8)  / sizeof(double);                   \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) { \
            __m256d vecA =  _mm256_load_pd(in);                         \
            in += types_per_step;                                       \
            __m256d vecB =  _mm256_load_pd(out);                        \
            __m256d res = _mm256_##op##_pd(vecA, vecB);                 \
            _mm256_store_pd(out, res);                                  \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
      }
#else
#define OP_AVX_AVX_DOUBLE_FUNC(op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2) */

#if defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX)
#define OP_AVX_SSE2_DOUBLE_FUNC(op)                                     \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_SSE2_FLAG) ) {            \
        types_per_step = (128 / 8)  / sizeof(double);                   \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) { \
            __m128d vecA = _mm_load_pd(in);                             \
            in += types_per_step;                                       \
            __m128d vecB = _mm_load_pd(out);                            \
            __m128d res = _mm_##op##_pd(vecA, vecB);                    \
            _mm_store_pd(out, res);                                     \
            out += types_per_step;                                      \
        }                                                               \
    }
#else
#define OP_AVX_SSE2_DOUBLE_FUNC(op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX) */

#define OP_AVX_DOUBLE_FUNC(op) \
static void ompi_op_avx_2buff_##op##_double(const void *_in, void *_out, int *count, \
                                            struct ompi_datatype_t **dtype, \
                                            struct ompi_op_base_module_1_0_0_t *module) \
{                                                                       \
    int types_per_step = (512 / 8)  / sizeof(double);                   \
    int left_over = *count;                                             \
    double* in = (double*)_in;                                          \
    double* out = (double*)_out;                                        \
    OP_AVX_AVX512_DOUBLE_FUNC(op);                                      \
    OP_AVX_AVX_DOUBLE_FUNC(op);                                         \
    OP_AVX_SSE2_DOUBLE_FUNC(op);                                        \
    while( left_over > 0 ) {                                            \
        int how_much = (left_over > 8) ? 8 : left_over;                 \
        switch(how_much) {                                              \
        case 8: out[7] = current_func(out[7], in[7]);                   \
        case 7: out[6] = current_func(out[6], in[6]);                   \
        case 6: out[5] = current_func(out[5], in[5]);                   \
        case 5: out[4] = current_func(out[4], in[4]);                   \
        case 4: out[3] = current_func(out[3], in[3]);                   \
        case 3: out[2] = current_func(out[2], in[2]);                   \
        case 2: out[1] = current_func(out[1], in[1]);                   \
        case 1: out[0] = current_func(out[0], in[0]);                   \
        }                                                               \
        left_over -= how_much;                                          \
        out += how_much;                                                \
        in += how_much;                                                 \
    }                                                                   \
}


/*************************************************************************
 * Max
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) > (b) ? (a) : (b))
    OP_AVX_FUNC(max, i, 8,    int8_t, max)
    OP_AVX_FUNC(max, u, 8,   uint8_t, max)
    OP_AVX_FUNC(max, i, 16,  int16_t, max)
    OP_AVX_FUNC(max, u, 16, uint16_t, max)
    OP_AVX_FUNC(max, i, 32,  int32_t, max)
    OP_AVX_FUNC(max, u, 32, uint32_t, max)
    OP_AVX512_FUNC(max, i, 64,  int64_t, max)
    OP_AVX512_FUNC(max, u, 64, uint64_t, max)

    /* Floating point */
    OP_AVX_FLOAT_FUNC(max)
    OP_AVX_DOUBLE_FUNC(max)

/*************************************************************************
 * Min
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) < (b) ? (a) : (b))
    OP_AVX_FUNC(min, i, 8,    int8_t, min)
    OP_AVX_FUNC(min, u, 8,   uint8_t, min)
    OP_AVX_FUNC(min, i, 16,  int16_t, min)
    OP_AVX_FUNC(min, u, 16, uint16_t, min)
    OP_AVX_FUNC(min, i, 32,  int32_t, min)
    OP_AVX_FUNC(min, u, 32, uint32_t, min)
    OP_AVX512_FUNC(min, i, 64,  int64_t, min)
    OP_AVX512_FUNC(min, u, 64, uint64_t, min)

    /* Floating point */
    OP_AVX_FLOAT_FUNC(min)
    OP_AVX_DOUBLE_FUNC(min)

/*************************************************************************
 * Sum
 ************************************************************************/
#undef current_func
#define current_func(a, b) ((a) + (b))
    OP_AVX_FUNC(sum, i, 8,    int8_t, adds)
    OP_AVX_FUNC(sum, u, 8,   uint8_t, adds)
    OP_AVX_FUNC(sum, i, 16,  int16_t, adds)
    OP_AVX_FUNC(sum, u, 16, uint16_t, adds)
    OP_AVX_FUNC(sum, i, 32,  int32_t, add)
    OP_AVX_FUNC(sum, i, 32, uint32_t, add)
    OP_AVX512_FUNC(sum, i, 64,  int64_t, add)
    OP_AVX512_FUNC(sum, i, 64, uint64_t, add)

    /* Floating point */
    OP_AVX_FLOAT_FUNC(add)
    OP_AVX_DOUBLE_FUNC(add)

/*************************************************************************
 * Product
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) * (b))
    OP_AVX_MUL(prod, i, 8, int8_t, mullo)
    OP_AVX_MUL(prod, i, 8, uint8_t, mullo)
    OP_AVX_FUNC(prod, i, 16,  int16_t, mullo)
    OP_AVX_FUNC(prod, i, 16, uint16_t, mullo)
    OP_AVX_FUNC(prod, i, 32,  int32_t, mullo)
    OP_AVX_FUNC(prod, i ,32, uint32_t, mullo)
    OP_AVX512_FUNC(prod, i, 64,  int64_t, mullo)
    OP_AVX512_FUNC(prod, i, 64, uint64_t, mullo)

    /* Floating point */
    OP_AVX_FLOAT_FUNC(mul)
    OP_AVX_DOUBLE_FUNC(mul)

/*************************************************************************
 * Bitwise AND
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) & (b))
    OP_AVX_BIT_FUNC(band, 8,    int8_t, and)
    OP_AVX_BIT_FUNC(band, 8,   uint8_t, and)
    OP_AVX_BIT_FUNC(band, 16,  int16_t, and)
    OP_AVX_BIT_FUNC(band, 16, uint16_t, and)
    OP_AVX_BIT_FUNC(band, 32,  int32_t, and)
    OP_AVX_BIT_FUNC(band, 32, uint32_t, and)
    OP_AVX_BIT_FUNC(band, 64,  int64_t, and)
    OP_AVX_BIT_FUNC(band, 64, uint64_t, and)

    // not defined - OP_AVX_FLOAT_FUNC(and)
    // not defined - OP_AVX_DOUBLE_FUNC(and)

/*************************************************************************
 * Bitwise OR
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) | (b))
    OP_AVX_BIT_FUNC(bor, 8,    int8_t, or)
    OP_AVX_BIT_FUNC(bor, 8,   uint8_t, or)
    OP_AVX_BIT_FUNC(bor, 16,  int16_t, or)
    OP_AVX_BIT_FUNC(bor, 16, uint16_t, or)
    OP_AVX_BIT_FUNC(bor, 32,  int32_t, or)
    OP_AVX_BIT_FUNC(bor, 32, uint32_t, or)
    OP_AVX_BIT_FUNC(bor, 64,  int64_t, or)
    OP_AVX_BIT_FUNC(bor, 64, uint64_t, or)

    // not defined - OP_AVX_FLOAT_FUNC(or)
    // not defined - OP_AVX_DOUBLE_FUNC(or)

/*************************************************************************
 * Bitwise XOR
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) ^ (b))
    OP_AVX_BIT_FUNC(bxor, 8,    int8_t, xor)
    OP_AVX_BIT_FUNC(bxor, 8,   uint8_t, xor)
    OP_AVX_BIT_FUNC(bxor, 16,  int16_t, xor)
    OP_AVX_BIT_FUNC(bxor, 16, uint16_t, xor)
    OP_AVX_BIT_FUNC(bxor, 32,  int32_t, xor)
    OP_AVX_BIT_FUNC(bxor, 32, uint32_t, xor)
    OP_AVX_BIT_FUNC(bxor, 64,  int64_t, xor)
    OP_AVX_BIT_FUNC(bxor, 64, uint64_t, xor)

    // not defined - OP_AVX_FLOAT_FUNC(xor)
    // not defined - OP_AVX_DOUBLE_FUNC(xor)

/*
 *  This is a three buffer (2 input and 1 output) version of the reduction
 *  routines, needed for some optimizations.
 */
#if defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512)
#define OP_AVX_AVX512_FUNC_3(name, type_sign, type_size, type, op)      \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX512F_FLAG|OMPI_OP_AVX_HAS_AVX512BW_FLAG) ) {   \
        types_per_step = (512 / 8) / sizeof(type);                      \
        for (; left_over >= types_per_step; left_over -= types_per_step) { \
            __m512i vecA =  _mm512_loadu_si512(in1);                    \
            __m512i vecB =  _mm512_loadu_si512(in2);                    \
            in1 += types_per_step;                                      \
            in2 += types_per_step;                                      \
            __m512i res = _mm512_##op##_ep##type_sign##type_size(vecA, vecB); \
            _mm512_storeu_si512((out), res);                            \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
    }
#else
#define OP_AVX_AVX512_FUNC_3(name, type_sign, type_size, type, op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512) */

#if defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2)
#define OP_AVX_AVX2_FUNC_3(name, type_sign, type_size, type, op)        \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX2_FLAG | OMPI_OP_AVX_HAS_AVX_FLAG) ) { \
        types_per_step = (256 / 8) / sizeof(type);                      \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) { \
            __m256i vecA = _mm256_loadu_si256((__m256i*)in1);           \
            __m256i vecB = _mm256_loadu_si256((__m256i*)in2);           \
            in1 += types_per_step;                                      \
            in2 += types_per_step;                                      \
            __m256i res =  _mm256_##op##_ep##type_sign##type_size(vecA, vecB); \
            _mm256_storeu_si256((__m256i*)out, res);                    \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
    }
#else
#define OP_AVX_AVX2_FUNC_3(name, type_sign, type_size, type, op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2) */

#if defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX)
#define OP_AVX_SSE4_1_FUNC_3(name, type_sign, type_size, type, op)      \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_SSE3_FLAG | OMPI_OP_AVX_HAS_SSE4_1_FLAG) ) {       \
        types_per_step = (128 / 8) / sizeof(type);                      \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) { \
            __m128i vecA = _mm_lddqu_si128((__m128i*)in1);              \
            __m128i vecB = _mm_lddqu_si128((__m128i*)in2);              \
            in1 += types_per_step;                                      \
            in2 += types_per_step;                                      \
            __m128i res =  _mm_##op##_ep##type_sign##type_size(vecA, vecB); \
            _mm_storeu_si128((__m128i*)out, res);                       \
            out += types_per_step;                                      \
        }                                                               \
    }
#else
#define OP_AVX_SSE4_1_FUNC_3(name, type_sign, type_size, type, op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX) */

#define OP_AVX_FUNC_3(name, type_sign, type_size, type, op)               \
static void ompi_op_avx_3buff_##name##_##type(const void * restrict _in1, \
                                              const void * restrict _in2, \
                                              void * restrict _out, int *count, \
                                              struct ompi_datatype_t **dtype, \
                                              struct ompi_op_base_module_1_0_0_t *module) \
{                                                                       \
    type *in1 = (type*)_in1, *in2 = (type*)_in2, *out = (type*)_out;    \
    int types_per_step, left_over = *count;                             \
    OP_AVX_AVX512_FUNC_3(name, type_sign, type_size, type, op);         \
    OP_AVX_AVX2_FUNC_3(name, type_sign, type_size, type, op);           \
    OP_AVX_SSE4_1_FUNC_3(name, type_sign, type_size, type, op);         \
    while( left_over > 0 ) {                                            \
        int how_much = (left_over > 8) ? 8 : left_over;                 \
        switch(how_much) {                                              \
        case 8: out[7] = current_func(in1[7], in2[7]);                  \
        case 7: out[6] = current_func(in1[6], in2[6]);                  \
        case 6: out[5] = current_func(in1[5], in2[5]);                  \
        case 5: out[4] = current_func(in1[4], in2[4]);                  \
        case 4: out[3] = current_func(in1[3], in2[3]);                  \
        case 3: out[2] = current_func(in1[2], in2[2]);                  \
        case 2: out[1] = current_func(in1[1], in2[1]);                  \
        case 1: out[0] = current_func(in1[0], in2[0]);                  \
        }                                                               \
        left_over -= how_much;                                          \
        out += how_much;                                                \
        in1 += how_much;                                                \
        in2 += how_much;                                                \
    }                                                                   \
}

#if defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512)
#define OP_AVX512_FUNC_3(name, type_sign, type_size, type, op)            \
        OP_AVX_FUNC_3(name, type_sign, type_size, type, op)
#else
#define OP_AVX512_FUNC_3(name, type_sign, type_size, type, op)
#endif

#if defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512)
#define OP_AVX_AVX512_MUL_3(name, type_sign, type_size, type, op)       \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX512F_FLAG | OMPI_OP_AVX_HAS_AVX512BW_FLAG) ) { \
        int types_per_step = (256 / 8) / sizeof(type);                  \
        for (; left_over >= types_per_step; left_over -= types_per_step) { \
            __m256i vecA_tmp =  _mm256_loadu_si256((__m256i*)in1);      \
            __m256i vecB_tmp =  _mm256_loadu_si256((__m256i*)in2);      \
            in1 += types_per_step;                                      \
            in2 += types_per_step;                                      \
            __m512i vecA = _mm512_cvtepi8_epi16(vecA_tmp);              \
            __m512i vecB = _mm512_cvtepi8_epi16(vecB_tmp);              \
            __m512i res = _mm512_##op##_ep##type_sign##16(vecA, vecB);  \
            vecB_tmp = _mm512_cvtepi16_epi8(res);                       \
            _mm256_storeu_si256((__m256i*)out, vecB_tmp);               \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
  }
#else
#define OP_AVX_AVX512_MUL_3(name, type_sign, type_size, type, op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512) */
/**
 * There is no support for 16 to 8 conversion without AVX512BW and AVX512VL, so
 * there is no AVX-only optimized function posible for OP_AVX_AVX2_MUL.
 */

/* special case for int8 mul */
#define OP_AVX_MUL_3(name, type_sign, type_size, type, op)              \
static void ompi_op_avx_3buff_##name##_##type(const void * restrict _in1, \
                                              const void * restrict _in2, \
                                              void * restrict _out, int *count, \
                                              struct ompi_datatype_t **dtype, \
                                              struct ompi_op_base_module_1_0_0_t *module) \
{                                                                       \
    type *in1 = (type*)_in1, *in2 = (type*)_in2, *out = (type*)_out;    \
    int left_over = *count;                                             \
    OP_AVX_AVX512_MUL_3(name, type_sign, type_size, type, op);          \
    while( left_over > 0 ) {                                            \
        int how_much = (left_over > 8) ? 8 : left_over;                 \
        switch(how_much) {                                              \
        case 8: out[7] = current_func(in1[7], in2[7]);                  \
        case 7: out[6] = current_func(in1[6], in2[6]);                  \
        case 6: out[5] = current_func(in1[5], in2[5]);                  \
        case 5: out[4] = current_func(in1[4], in2[4]);                  \
        case 4: out[3] = current_func(in1[3], in2[3]);                  \
        case 3: out[2] = current_func(in1[2], in2[2]);                  \
        case 2: out[1] = current_func(in1[1], in2[1]);                  \
        case 1: out[0] = current_func(in1[0], in2[0]);                  \
        }                                                               \
        left_over -= how_much;                                          \
        out += how_much;                                                \
        in1 += how_much;                                                \
        in2 += how_much;                                                \
    }                                                                   \
}

#if defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512)
#define OP_AVX_AVX512_BIT_FUNC_3(name, type_size, type, op)             \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX512F_FLAG) ) {         \
        types_per_step = (512 / 8) / sizeof(type);                      \
        for (; left_over >= types_per_step; left_over -= types_per_step) {  \
            __m512i vecA =  _mm512_loadu_si512(in1);                    \
            __m512i vecB =  _mm512_loadu_si512(in2);                    \
            in1 += types_per_step;                                      \
            in2 += types_per_step;                                      \
            __m512i res = _mm512_##op##_si512(vecA, vecB);              \
            _mm512_storeu_si512(out, res);                              \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
    }
#else
#define OP_AVX_AVX512_BIT_FUNC_3(name, type_size, type, op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512) */

#if defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2)
#define OP_AVX_AVX2_BIT_FUNC_3(name, type_size, type, op)               \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX2_FLAG | OMPI_OP_AVX_HAS_AVX_FLAG) ) { \
        types_per_step = (256 / 8) / sizeof(type);                      \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) {     \
            __m256i vecA = _mm256_loadu_si256((__m256i*)in1);           \
            __m256i vecB = _mm256_loadu_si256((__m256i*)in2);           \
            in1 += types_per_step;                                      \
            in2 += types_per_step;                                      \
            __m256i res =  _mm256_##op##_si256(vecA, vecB);             \
            _mm256_storeu_si256((__m256i*)out, res);                    \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
    }
#else
#define OP_AVX_AVX2_BIT_FUNC_3(name, type_size, type, op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2) */

#if defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX)
#define OP_AVX_SSE3_BIT_FUNC_3(name, type_size, type, op)               \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_SSE3_FLAG) ) {            \
        types_per_step = (128 / 8) / sizeof(type);                      \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) {     \
            __m128i vecA = _mm_lddqu_si128((__m128i*)in1);              \
            __m128i vecB = _mm_lddqu_si128((__m128i*)in2);              \
            in1 += types_per_step;                                      \
            in2 += types_per_step;                                      \
            __m128i res =  _mm_##op##_si128(vecA, vecB);                \
            _mm_storeu_si128((__m128i*)out, res);                       \
            out += types_per_step;                                      \
        }                                                               \
    }
#else
#define OP_AVX_SSE3_BIT_FUNC_3(name, type_size, type, op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX) */

#define OP_AVX_BIT_FUNC_3(name, type_size, type, op)                    \
static void ompi_op_avx_3buff_##op##_##type(const void *_in1, const void *_in2, \
                                            void *_out, int *count,     \
                                            struct ompi_datatype_t **dtype, \
                                            struct ompi_op_base_module_1_0_0_t *module) \
{                                                                       \
    int types_per_step, left_over = *count;                             \
    type *in1 = (type*)_in1, *in2 = (type*)_in2, *out = (type*)_out;    \
    OP_AVX_AVX512_BIT_FUNC_3(name, type_size, type, op);                \
    OP_AVX_AVX2_BIT_FUNC_3(name, type_size, type, op);                  \
    OP_AVX_SSE3_BIT_FUNC_3(name, type_size, type, op);                  \
    while( left_over > 0 ) {                                            \
        int how_much = (left_over > 8) ? 8 : left_over;                 \
        switch(how_much) {                                              \
        case 8: out[7] = current_func(in1[7], in2[7]);                  \
        case 7: out[6] = current_func(in1[6], in2[6]);                  \
        case 6: out[5] = current_func(in1[5], in2[5]);                  \
        case 5: out[4] = current_func(in1[4], in2[4]);                  \
        case 4: out[3] = current_func(in1[3], in2[3]);                  \
        case 3: out[2] = current_func(in1[2], in2[2]);                  \
        case 2: out[1] = current_func(in1[1], in2[1]);                  \
        case 1: out[0] = current_func(in1[0], in2[0]);                  \
        }                                                               \
        left_over -= how_much;                                          \
        out += how_much;                                                \
        in1 += how_much;                                                \
        in2 += how_much;                                                \
    }                                                                   \
}

#if defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512)
#define OP_AVX_AVX512_FLOAT_FUNC_3(op)                                  \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX512F_FLAG) ) {         \
        types_per_step = (512 / 8) / sizeof(float);                     \
        for (; left_over >= types_per_step; left_over -= types_per_step) { \
            __m512 vecA =  _mm512_load_ps(in1);                         \
            __m512 vecB =  _mm512_load_ps(in2);                         \
            in1 += types_per_step;                                      \
            in2 += types_per_step;                                      \
            __m512 res = _mm512_##op##_ps(vecA, vecB);                  \
            _mm512_store_ps(out, res);                                  \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
    }
#else
#define OP_AVX_AVX512_FLOAT_FUNC_3(op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512) */

#if defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2)
#define OP_AVX_AVX_FLOAT_FUNC_3(op)                                     \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX_FLAG) ) {             \
        types_per_step = (256 / 8) / sizeof(float);                     \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) { \
            __m256 vecA =  _mm256_load_ps(in1);                         \
            __m256 vecB =  _mm256_load_ps(in2);                         \
            in1 += types_per_step;                                      \
            in2 += types_per_step;                                      \
            __m256 res = _mm256_##op##_ps(vecA, vecB);                  \
            _mm256_store_ps(out, res);                                  \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
    }
#else
#define OP_AVX_AVX_FLOAT_FUNC_3(op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2) */

#if defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX)
#define OP_AVX_SSE_FLOAT_FUNC_3(op)                  \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_SSE_FLAG) ) {             \
        types_per_step = (128 / 8) / sizeof(float);                     \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) { \
            __m128 vecA = _mm_load_ps(in1);                             \
            __m128 vecB = _mm_load_ps(in2);                             \
            in1 += types_per_step;                                      \
            in2 += types_per_step;                                      \
            __m128 res = _mm_##op##_ps(vecA, vecB);                     \
            _mm_store_ps(out, res);                                     \
            out += types_per_step;                                      \
        }                                                               \
    }
#else
#define OP_AVX_SSE_FLOAT_FUNC_3(op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX) */

#define OP_AVX_FLOAT_FUNC_3(op)                                         \
static void ompi_op_avx_3buff_##op##_float(const void *_in1, const void *_in2, \
                                           void *_out, int *count,             \
                                           struct ompi_datatype_t **dtype, \
                                           struct ompi_op_base_module_1_0_0_t *module) \
{                                                                       \
    int types_per_step, left_over = *count;                             \
    float *in1 = (float*)_in1, *in2 = (float*)_in2, *out = (float*)_out; \
    OP_AVX_AVX512_FLOAT_FUNC_3(op);                                     \
    OP_AVX_AVX_FLOAT_FUNC_3(op);                                        \
    OP_AVX_SSE_FLOAT_FUNC_3(op);                                        \
    while( left_over > 0 ) {                                            \
        int how_much = (left_over > 8) ? 8 : left_over;                 \
        switch(how_much) {                                              \
        case 8: out[7] = current_func(in1[7], in2[7]);                  \
        case 7: out[6] = current_func(in1[6], in2[6]);                  \
        case 6: out[5] = current_func(in1[5], in2[5]);                  \
        case 5: out[4] = current_func(in1[4], in2[4]);                  \
        case 4: out[3] = current_func(in1[3], in2[3]);                  \
        case 3: out[2] = current_func(in1[2], in2[2]);                  \
        case 2: out[1] = current_func(in1[1], in2[1]);                  \
        case 1: out[0] = current_func(in1[0], in2[0]);                  \
        }                                                               \
        left_over -= how_much;                                          \
        out += how_much;                                                \
        in1 += how_much;                                                \
        in2 += how_much;                                                \
    }                                                                   \
}

#if defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512)
#define OP_AVX_AVX512_DOUBLE_FUNC_3(op)                                 \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX512F_FLAG) ) {         \
        types_per_step = (512 / 8) / sizeof(double);                    \
        for (; left_over >= types_per_step; left_over -= types_per_step) { \
            __m512d vecA =  _mm512_load_pd((in1));                      \
            __m512d vecB =  _mm512_load_pd((in2));                      \
            in1 += types_per_step;                                      \
            in2 += types_per_step;                                      \
            __m512d res = _mm512_##op##_pd(vecA, vecB);                 \
            _mm512_store_pd((out), res);                                \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
    }
#else
#define OP_AVX_AVX512_DOUBLE_FUNC_3(op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX512) && (1 == OMPI_MCA_OP_HAVE_AVX512) */

#if defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2)
#define OP_AVX_AVX_DOUBLE_FUNC_3(op)                                    \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_AVX_FLAG) ) {             \
        types_per_step = (256 / 8) / sizeof(double);                    \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) { \
            __m256d vecA =  _mm256_load_pd(in1);                        \
            __m256d vecB =  _mm256_load_pd(in2);                        \
            in1 += types_per_step;                                      \
            in2 += types_per_step;                                      \
            __m256d res = _mm256_##op##_pd(vecA, vecB);                 \
            _mm256_store_pd(out, res);                                  \
            out += types_per_step;                                      \
        }                                                               \
        if( 0 == left_over ) return;                                    \
    }
#else
#define OP_AVX_AVX_DOUBLE_FUNC_3(op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX2) && (1 == OMPI_MCA_OP_HAVE_AVX2) */

#if defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX)
#define OP_AVX_SSE2_DOUBLE_FUNC_3(op)                                   \
    if( OMPI_OP_AVX_HAS_FLAGS(OMPI_OP_AVX_HAS_SSE2_FLAG) ) {            \
        types_per_step = (128 / 8) / sizeof(double);                    \
        for( ; left_over >= types_per_step; left_over -= types_per_step ) { \
            __m128d vecA = _mm_load_pd(in1);                            \
            __m128d vecB = _mm_load_pd(in2);                            \
            in1 += types_per_step;                                      \
            in2 += types_per_step;                                      \
            __m128d res = _mm_##op##_pd(vecA, vecB);                    \
            _mm_store_pd(out, res);                                     \
            out += types_per_step;                                      \
        }                                                               \
    }
#else
#define OP_AVX_SSE2_DOUBLE_FUNC_3(op) {}
#endif  /* defined(OMPI_MCA_OP_HAVE_AVX) && (1 == OMPI_MCA_OP_HAVE_AVX) */

#define OP_AVX_DOUBLE_FUNC_3(op)                                        \
static void ompi_op_avx_3buff_##op##_double(const void *_in1, const void *_in2, \
                                            void *_out, int *count,     \
                                            struct ompi_datatype_t **dtype, \
                                            struct ompi_op_base_module_1_0_0_t *module) \
{                                                                       \
    int types_per_step, left_over = *count;                             \
    double *in1 = (double*)_in1, *in2 = (double*)_in2, *out = (double*)_out; \
    OP_AVX_AVX512_DOUBLE_FUNC_3(op);                                    \
    OP_AVX_AVX_DOUBLE_FUNC_3(op);                                       \
    OP_AVX_SSE2_DOUBLE_FUNC_3(op);                                      \
    while( left_over > 0 ) {                                            \
        int how_much = (left_over > 8) ? 8 : left_over;                 \
        switch(how_much) {                                              \
        case 8: out[7] = current_func(in1[7], in2[7]);                  \
        case 7: out[6] = current_func(in1[6], in2[6]);                  \
        case 6: out[5] = current_func(in1[5], in2[5]);                  \
        case 5: out[4] = current_func(in1[4], in2[4]);                  \
        case 4: out[3] = current_func(in1[3], in2[3]);                  \
        case 3: out[2] = current_func(in1[2], in2[2]);                  \
        case 2: out[1] = current_func(in1[1], in2[1]);                  \
        case 1: out[0] = current_func(in1[0], in2[0]);                  \
        }                                                               \
        left_over -= how_much;                                          \
        out += how_much;                                                \
        in1 += how_much;                                                \
        in2 += how_much;                                                \
    }                                                                   \
}

/*************************************************************************
 * Max
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) > (b) ? (a) : (b))

    OP_AVX_FUNC_3(max, i, 8,    int8_t, max)
    OP_AVX_FUNC_3(max, u, 8,   uint8_t, max)
    OP_AVX_FUNC_3(max, i, 16,  int16_t, max)
    OP_AVX_FUNC_3(max, u, 16, uint16_t, max)
    OP_AVX_FUNC_3(max, i, 32,  int32_t, max)
    OP_AVX_FUNC_3(max, u, 32, uint32_t, max)
    OP_AVX512_FUNC_3(max, i, 64,  int64_t, max)
    OP_AVX512_FUNC_3(max, u, 64, uint64_t, max)

    /* Floating point */
    OP_AVX_FLOAT_FUNC_3(max)
    OP_AVX_DOUBLE_FUNC_3(max)

/*************************************************************************
 * Min
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) < (b) ? (a) : (b))
    OP_AVX_FUNC_3(min, i, 8,    int8_t, min)
    OP_AVX_FUNC_3(min, u, 8,   uint8_t, min)
    OP_AVX_FUNC_3(min, i, 16,  int16_t, min)
    OP_AVX_FUNC_3(min, u, 16, uint16_t, min)
    OP_AVX_FUNC_3(min, i, 32,  int32_t, min)
    OP_AVX_FUNC_3(min, u, 32, uint32_t, min)
    OP_AVX512_FUNC_3(min, i, 64,  int64_t, min)
    OP_AVX512_FUNC_3(min, u, 64, uint64_t, min)

    /* Floating point */
    OP_AVX_FLOAT_FUNC_3(min)
    OP_AVX_DOUBLE_FUNC_3(min)

/*************************************************************************
 * Sum
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) + (b))

    OP_AVX_FUNC_3(sum, i, 8,    int8_t, add)
    OP_AVX_FUNC_3(sum, i, 8,   uint8_t, add)
    OP_AVX_FUNC_3(sum, i, 16,  int16_t, add)
    OP_AVX_FUNC_3(sum, i, 16, uint16_t, add)
    OP_AVX_FUNC_3(sum, i, 32,  int32_t, add)
    OP_AVX_FUNC_3(sum, i, 32, uint32_t, add)
    OP_AVX512_FUNC_3(sum, i, 64,  int64_t, add)
    OP_AVX512_FUNC_3(sum, i, 64, uint64_t, add)

    /* Floating point */
    OP_AVX_FLOAT_FUNC_3(add)
    OP_AVX_DOUBLE_FUNC_3(add)

/*************************************************************************
 * Product
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) * (b))
    OP_AVX_MUL_3(prod, i, 8, int8_t, mullo)
    OP_AVX_MUL_3(prod, i, 8, uint8_t, mullo)
    OP_AVX_FUNC_3(prod, i, 16,  int16_t, mullo)
    OP_AVX_FUNC_3(prod, i, 16, uint16_t, mullo)
    OP_AVX_FUNC_3(prod, i, 32,  int32_t, mullo)
    OP_AVX_FUNC_3(prod, i ,32, uint32_t, mullo)
    OP_AVX512_FUNC_3(prod, i, 64,  int64_t, mullo)
    OP_AVX512_FUNC_3(prod, i, 64, uint64_t, mullo)

    /* Floating point */
    OP_AVX_FLOAT_FUNC_3(mul)
    OP_AVX_DOUBLE_FUNC_3(mul)

/*************************************************************************
 * Bitwise AND
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) & (b))
    OP_AVX_BIT_FUNC_3(band, 8,    int8_t, and)
    OP_AVX_BIT_FUNC_3(band, 8,   uint8_t, and)
    OP_AVX_BIT_FUNC_3(band, 16,  int16_t, and)
    OP_AVX_BIT_FUNC_3(band, 16, uint16_t, and)
    OP_AVX_BIT_FUNC_3(band, 32,  int32_t, and)
    OP_AVX_BIT_FUNC_3(band, 32, uint32_t, and)
    OP_AVX_BIT_FUNC_3(band, 64,  int64_t, and)
    OP_AVX_BIT_FUNC_3(band, 64, uint64_t, and)

    // not defined - OP_AVX_FLOAT_FUNC_3(and)
    // not defined - OP_AVX_DOUBLE_FUNC_3(and)

/*************************************************************************
 * Bitwise OR
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) | (b))
    OP_AVX_BIT_FUNC_3(bor, 8,    int8_t, or)
    OP_AVX_BIT_FUNC_3(bor, 8,   uint8_t, or)
    OP_AVX_BIT_FUNC_3(bor, 16,  int16_t, or)
    OP_AVX_BIT_FUNC_3(bor, 16, uint16_t, or)
    OP_AVX_BIT_FUNC_3(bor, 32,  int32_t, or)
    OP_AVX_BIT_FUNC_3(bor, 32, uint32_t, or)
    OP_AVX_BIT_FUNC_3(bor, 64,  int64_t, or)
    OP_AVX_BIT_FUNC_3(bor, 64, uint64_t, or)

    // not defined - OP_AVX_FLOAT_FUNC_3(or)
    // not defined - OP_AVX_DOUBLE_FUNC_3(or)

/*************************************************************************
 * Bitwise XOR
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) ^ (b))
    OP_AVX_BIT_FUNC_3(bxor, 8,    int8_t, xor)
    OP_AVX_BIT_FUNC_3(bxor, 8,   uint8_t, xor)
    OP_AVX_BIT_FUNC_3(bxor, 16,  int16_t, xor)
    OP_AVX_BIT_FUNC_3(bxor, 16, uint16_t, xor)
    OP_AVX_BIT_FUNC_3(bxor, 32,  int32_t, xor)
    OP_AVX_BIT_FUNC_3(bxor, 32, uint32_t, xor)
    OP_AVX_BIT_FUNC_3(bxor, 64,  int64_t, xor)
    OP_AVX_BIT_FUNC_3(bxor, 64, uint64_t, xor)

    // not defined - OP_AVX_FLOAT_FUNC_3(xor)
    // not defined - OP_AVX_DOUBLE_FUNC_3(xor)


/** C integer ***********************************************************/
#define C_INTEGER(name, ftype)                                              \
    [OMPI_OP_BASE_TYPE_INT8_T] = ompi_op_avx_##ftype##_##name##_int8_t,     \
    [OMPI_OP_BASE_TYPE_UINT8_T] = ompi_op_avx_##ftype##_##name##_uint8_t,   \
    [OMPI_OP_BASE_TYPE_INT16_T] = ompi_op_avx_##ftype##_##name##_int16_t,   \
    [OMPI_OP_BASE_TYPE_UINT16_T] = ompi_op_avx_##ftype##_##name##_uint16_t, \
    [OMPI_OP_BASE_TYPE_INT32_T] = ompi_op_avx_##ftype##_##name##_int32_t,   \
    [OMPI_OP_BASE_TYPE_UINT32_T] = ompi_op_avx_##ftype##_##name##_uint32_t

#define C_INTEGER64(name, ftype)                                            \
    [OMPI_OP_BASE_TYPE_INT64_T] = ompi_op_avx_##ftype##_##name##_int64_t,   \
    [OMPI_OP_BASE_TYPE_UINT64_T] = ompi_op_avx_##ftype##_##name##_uint64_t


/** Floating point, including all the Fortran reals *********************/
#define FLOAT(name, ftype) ompi_op_avx_##ftype##_##name##_float
#define DOUBLE(name, ftype) ompi_op_avx_##ftype##_##name##_double

#define FLOATING_POINT(name, ftype)                                         \
    [OMPI_OP_BASE_TYPE_SHORT_FLOAT] = NULL,                                 \
    [OMPI_OP_BASE_TYPE_FLOAT] = FLOAT(name, ftype),                         \
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

ompi_op_base_handler_fn_t ompi_op_avx_functions[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX] =
{
    /* Corresponds to MPI_OP_NULL */
    [OMPI_OP_BASE_FORTRAN_NULL] = {
        /* Leaving this empty puts in NULL for all entries */
        NULL,
    },
    /* Corresponds to MPI_MAX */
    [OMPI_OP_BASE_FORTRAN_MAX] = {
        C_INTEGER(max, 2buff),
        C_INTEGER64(max, 2buff),
        FLOATING_POINT(max, 2buff),
    },
    /* Corresponds to MPI_MIN */
    [OMPI_OP_BASE_FORTRAN_MIN] = {
        C_INTEGER(min, 2buff),
        C_INTEGER64(min, 2buff),
        FLOATING_POINT(min, 2buff),
    },
    /* Corresponds to MPI_SUM */
    [OMPI_OP_BASE_FORTRAN_SUM] = {
        C_INTEGER(sum, 2buff),
        C_INTEGER64(sum, 2buff),
        FLOATING_POINT(add, 2buff),
    },
    /* Corresponds to MPI_PROD */
    [OMPI_OP_BASE_FORTRAN_PROD] = {
        C_INTEGER(prod, 2buff),
        C_INTEGER64(prod, 2buff),
        FLOATING_POINT(mul, 2buff),
    },
    /* Corresponds to MPI_LAND */
    [OMPI_OP_BASE_FORTRAN_LAND] = {
        NULL,
    },
    /* Corresponds to MPI_BAND */
    [OMPI_OP_BASE_FORTRAN_BAND] = {
        C_INTEGER  (band, 2buff),
        C_INTEGER64(band, 2buff),
    },
    /* Corresponds to MPI_LOR */
    [OMPI_OP_BASE_FORTRAN_LOR] = {
        NULL,
    },
    /* Corresponds to MPI_BOR */
    [OMPI_OP_BASE_FORTRAN_BOR] = {
        C_INTEGER  (bor, 2buff),
        C_INTEGER64(bor, 2buff),
    },
    /* Corresponds to MPI_LXOR */
    [OMPI_OP_BASE_FORTRAN_LXOR] = {
        NULL,
    },
    /* Corresponds to MPI_BXOR */
    [OMPI_OP_BASE_FORTRAN_BXOR] = {
        C_INTEGER  (bxor, 2buff),
        C_INTEGER64(bxor, 2buff),
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

ompi_op_base_3buff_handler_fn_t ompi_op_avx_3buff_functions[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX] =
{
    /* Corresponds to MPI_OP_NULL */
    [OMPI_OP_BASE_FORTRAN_NULL] = {
        /* Leaving this empty puts in NULL for all entries */
        NULL,
    },
    /* Corresponds to MPI_MAX */
    [OMPI_OP_BASE_FORTRAN_MAX] = {
        C_INTEGER(max, 3buff),
        C_INTEGER64(max, 3buff),
        FLOATING_POINT(max, 3buff),
    },
    /* Corresponds to MPI_MIN */
    [OMPI_OP_BASE_FORTRAN_MIN] = {
        C_INTEGER(min, 3buff),
        C_INTEGER64(min, 3buff),
        FLOATING_POINT(min, 3buff),
    },
    /* Corresponds to MPI_SUM */
    [OMPI_OP_BASE_FORTRAN_SUM] = {
        C_INTEGER(sum, 3buff),
        C_INTEGER64(sum, 3buff),
        FLOATING_POINT(add, 3buff),
    },
    /* Corresponds to MPI_PROD */
    [OMPI_OP_BASE_FORTRAN_PROD] = {
        C_INTEGER(prod, 3buff),
        C_INTEGER64(prod, 3buff),
        FLOATING_POINT(mul, 3buff),
    },
    /* Corresponds to MPI_LAND */
    [OMPI_OP_BASE_FORTRAN_LAND] ={
        NULL,
    },
    /* Corresponds to MPI_BAND */
    [OMPI_OP_BASE_FORTRAN_BAND] = {
        C_INTEGER  (and, 3buff),
        C_INTEGER64(and, 3buff),
    },
    /* Corresponds to MPI_LOR */
    [OMPI_OP_BASE_FORTRAN_LOR] = {
        NULL,
    },
    /* Corresponds to MPI_BOR */
    [OMPI_OP_BASE_FORTRAN_BOR] = {
        C_INTEGER  (or, 3buff),
        C_INTEGER64(or, 3buff),
    },
    /* Corresponds to MPI_LXOR */
    [OMPI_OP_BASE_FORTRAN_LXOR] = {
        NULL,
    },
    /* Corresponds to MPI_BXOR */
    [OMPI_OP_BASE_FORTRAN_BXOR] = {
        C_INTEGER  (xor, 3buff),
        C_INTEGER64(xor, 3buff),
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
