/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2006 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2014      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"

#include <stddef.h>
#include <stdlib.h>
#include <libgccjit.h>

#include "opal/datatype/opal_convertor.h"
#include "opal/datatype/opal_datatype.h"
#include "opal/datatype/opal_datatype_internal.h"

static int32_t opal_datatype_optimize_short(opal_datatype_t *pData, size_t count,
                                            dt_type_desc_t *pTypeDesc)
{
    dt_elem_desc_t *pElemDesc;
    dt_stack_t *pOrigStack, *pStack; /* pointer to the position on the stack */
    int32_t pos_desc = 0; /* actual position in the description of the derived datatype */
    int32_t stack_pos = 0;
    int32_t nbElems = 0;
    ptrdiff_t total_disp = 0;
    ddt_elem_desc_t last = {.common.flags = 0xFFFF /* all on */, .count = 0, .disp = 0}, compress;
    ddt_elem_desc_t *current;

    pOrigStack = pStack = (dt_stack_t *) malloc(sizeof(dt_stack_t) * (pData->loops + 2));
    SAVE_STACK(pStack, -1, 0, count, 0);

    pTypeDesc->length = 2 * pData->desc.used
                        + 1 /* for the fake OPAL_DATATYPE_END_LOOP at the end */;
    pTypeDesc->desc = pElemDesc = (dt_elem_desc_t *) malloc(sizeof(dt_elem_desc_t)
                                                            * pTypeDesc->length);
    pTypeDesc->used = 0;

    assert(OPAL_DATATYPE_END_LOOP == pData->desc.desc[pData->desc.used].elem.common.type);

    while (stack_pos >= 0) {
        if (OPAL_DATATYPE_END_LOOP
            == pData->desc.desc[pos_desc].elem.common.type) { /* end of the current loop */
            ddt_endloop_desc_t *end_loop = &(pData->desc.desc[pos_desc].end_loop);
            if (0 != last.count) {
                CREATE_ELEM(pElemDesc, last.common.type, OPAL_DATATYPE_FLAG_BASIC, last.blocklen,
                            last.count, last.disp, last.extent);
                pElemDesc++;
                nbElems++;
                last.count = 0;
            }
            CREATE_LOOP_END(pElemDesc, nbElems - pStack->index + 1, /* # of elems in this loop */
                            end_loop->first_elem_disp, end_loop->size, end_loop->common.flags);
            if (--stack_pos >= 0) { /* still something to do ? */
                ddt_loop_desc_t *pStartLoop = &(pTypeDesc->desc[pStack->index - 1].loop);
                pStartLoop->items = pElemDesc->end_loop.items;
                total_disp = pStack->disp; /* update the displacement position */
            }
            pElemDesc++;
            nbElems++;
            pStack--; /* go down one position on the stack */
            pos_desc++;
            continue;
        }
        if (OPAL_DATATYPE_LOOP == pData->desc.desc[pos_desc].elem.common.type) {
            ddt_loop_desc_t *loop = (ddt_loop_desc_t *) &(pData->desc.desc[pos_desc]);
            int index = GET_FIRST_NON_LOOP(&(pData->desc.desc[pos_desc]));

            if (loop->common.flags & OPAL_DATATYPE_FLAG_CONTIGUOUS) {
                ddt_endloop_desc_t *end_loop = (ddt_endloop_desc_t *) &(
                    pData->desc.desc[pos_desc + loop->items]);

                assert(pData->desc.desc[pos_desc + index].elem.disp == end_loop->first_elem_disp);
                compress.common.flags = loop->common.flags;
                compress.common.type = pData->desc.desc[pos_desc + index].elem.common.type;
                compress.blocklen = pData->desc.desc[pos_desc + index].elem.blocklen;
                for (uint32_t i = index + 1; i < loop->items; i++) {
                    current = &pData->desc.desc[pos_desc + i].elem;
                    assert(1 == current->count);
                    if ((current->common.type == OPAL_DATATYPE_LOOP)
                        || compress.common.type != current->common.type) {
                        compress.common.type   = OPAL_DATATYPE_UINT1;
                        compress.common.flags |= OPAL_DATATYPE_OPTIMIZED_RESTRICTED;
                        pData->flags          |= OPAL_DATATYPE_OPTIMIZED_RESTRICTED;
                        compress.blocklen = end_loop->size;
                        break;
                    }
                    compress.blocklen += current->blocklen;
                }
                compress.count = loop->loops;
                compress.extent = loop->extent;
                compress.disp = end_loop->first_elem_disp;
                if (compress.extent
                    == (ptrdiff_t)(compress.blocklen
                                   * opal_datatype_basicDatatypes[compress.common.type]->size)) {
                    /* The compressed element is contiguous: collapse it into a single large
                     * blocklen */
                    compress.blocklen *= compress.count;
                    compress.extent *= compress.count;
                    compress.count = 1;
                }
                /**
                 * The current loop has been compressed and can now be treated as if it
                 * was a data element. We can now look if it can be fused with last,
                 * as done in the fusion of 2 elements below. Let's use the same code.
                 */
                pos_desc += loop->items + 1;
                current = &compress;
                goto fuse_loops;
            }

            /**
             * If the content of the loop is not contiguous there is little we can do
             * that would not incur significant optimization cost and still be beneficial
             * in reducing the number of memcpy during pack/unpack.
             */

            if (0 != last.count) { /* Generate the pending element */
                CREATE_ELEM(pElemDesc, last.common.type, OPAL_DATATYPE_FLAG_BASIC, last.blocklen,
                            last.count, last.disp, last.extent);
                pElemDesc++;
                nbElems++;
                last.count = 0;
                last.common.type = OPAL_DATATYPE_LOOP;
            }

            /* Can we unroll the loop? */
            if ((loop->items <= 3) && (loop->loops <= 2)) {
                ptrdiff_t elem_displ = 0;
                for (uint32_t i = 0; i < loop->loops; i++) {
                    for (uint32_t j = 0; j < (loop->items - 1); j++) {
                        current = &pData->desc.desc[pos_desc + index + j].elem;
                        CREATE_ELEM(pElemDesc, current->common.type, current->common.flags,
                                    current->blocklen, current->count, current->disp + elem_displ,
                                    current->extent);
                        pElemDesc++;
                        nbElems++;
                    }
                    elem_displ += loop->extent;
                }
                pos_desc += loop->items + 1;
                goto complete_loop;
            }

            CREATE_LOOP_START(pElemDesc, loop->loops, loop->items, loop->extent,
                              loop->common.flags);
            pElemDesc++;
            nbElems++;
            PUSH_STACK(pStack, stack_pos, nbElems, OPAL_DATATYPE_LOOP, loop->loops, total_disp);
            pos_desc++;
            DDT_DUMP_STACK(pStack, stack_pos, pData->desc.desc, "advance loops");

        complete_loop:
            total_disp = pStack->disp; /* update the displacement */
            continue;
        }
        while (pData->desc.desc[pos_desc].elem.common.flags
               & OPAL_DATATYPE_FLAG_DATA) { /* go over all basic datatype elements */
            current = &pData->desc.desc[pos_desc].elem;
            pos_desc++; /* point to the next element as current points to the current one */

        fuse_loops:
            if (0 == last.count) { /* first data of the datatype */
                last = *current;
                continue; /* next data */
            } else {      /* can we merge it in order to decrease count */
                if ((ptrdiff_t) last.blocklen
                        * (ptrdiff_t) opal_datatype_basicDatatypes[last.common.type]->size
                    == last.extent) {
                    last.extent *= last.count;
                    last.blocklen *= last.count;
                    last.count = 1;
                }
            }

            /* are the two elements compatible: aka they have very similar values and they
             * can be merged together by increasing the count, and/or changing the extent.
             */
            if ((last.blocklen * opal_datatype_basicDatatypes[last.common.type]->size)
                == (current->blocklen * opal_datatype_basicDatatypes[current->common.type]->size)) {
                ddt_elem_desc_t save = last; /* safekeep the type and blocklen */
                if (last.common.type != current->common.type) {
                    last.blocklen *= opal_datatype_basicDatatypes[last.common.type]->size;
                    last.common.type   = OPAL_DATATYPE_UINT1;
                    last.common.flags |= OPAL_DATATYPE_OPTIMIZED_RESTRICTED;
                    pData->flags      |= OPAL_DATATYPE_OPTIMIZED_RESTRICTED;
                }

                if ((last.extent * (ptrdiff_t) last.count + last.disp) == current->disp) {
                    if (1 == current->count) {
                        last.count++;
                        continue;
                    }
                    if (last.extent == current->extent) {
                        last.count += current->count;
                        continue;
                    }
                }
                if (1 == last.count) {
                    /* we can ignore the extent of the element with count == 1 and merge them
                     * together if their displacements match */
                    if (1 == current->count) {
                        last.extent = current->disp - last.disp;
                        last.count++;
                        continue;
                    }
                    /* can we compute a matching displacement ? */
                    if ((last.disp + current->extent) == current->disp) {
                        last.extent = current->extent;
                        last.count = current->count + last.count;
                        continue;
                    }
                }
                last.blocklen = save.blocklen;
                last.common.type = save.common.type;
                /* try other optimizations */
            }
            /* are the elements fusionable such that we can fusion the last blocklen of one with the
             * first blocklen of the other.
             */
            if ((ptrdiff_t)(last.disp + (last.count - 1) * last.extent
                            + last.blocklen * opal_datatype_basicDatatypes[last.common.type]->size)
                == current->disp) {
                if (last.count != 1) {
                    CREATE_ELEM(pElemDesc, last.common.type, OPAL_DATATYPE_FLAG_BASIC,
                                last.blocklen, last.count - 1, last.disp, last.extent);
                    pElemDesc++;
                    nbElems++;
                    last.disp += (last.count - 1) * last.extent;
                    last.count = 1;
                }
                if (last.common.type == current->common.type) {
                    last.blocklen += current->blocklen;
                } else {
                    last.blocklen = ((last.blocklen
                                      * opal_datatype_basicDatatypes[last.common.type]->size)
                                     + (current->blocklen
                                        * opal_datatype_basicDatatypes[current->common.type]
                                              ->size));
                    last.common.type   = OPAL_DATATYPE_UINT1;
                    last.common.flags |= OPAL_DATATYPE_OPTIMIZED_RESTRICTED;
                    pData->flags      |= OPAL_DATATYPE_OPTIMIZED_RESTRICTED;
                }
                last.extent += current->extent;
                if (current->count != 1) {
                    CREATE_ELEM(pElemDesc, last.common.type, OPAL_DATATYPE_FLAG_BASIC,
                                last.blocklen, last.count, last.disp, last.extent);
                    pElemDesc++;
                    nbElems++;
                    last = *current;
                    last.count -= 1;
                    last.disp += last.extent;
                }
                continue;
            }
            CREATE_ELEM(pElemDesc, last.common.type, OPAL_DATATYPE_FLAG_BASIC, last.blocklen,
                        last.count, last.disp, last.extent);
            pElemDesc++;
            nbElems++;
            last = *current;
        }
    }

    if (0 != last.count) {
        CREATE_ELEM(pElemDesc, last.common.type, OPAL_DATATYPE_FLAG_BASIC, last.blocklen,
                    last.count, last.disp, last.extent);
        pElemDesc++;
        nbElems++;
    }
    /* cleanup the stack */
    pTypeDesc->used = nbElems - 1; /* except the last fake END_LOOP */
    free(pOrigStack);
    return OPAL_SUCCESS;
}

int32_t opal_datatype_commit(opal_datatype_t *pData)
{
    ddt_endloop_desc_t *pLast = &(pData->desc.desc[pData->desc.used].end_loop);
    ptrdiff_t first_elem_disp = 0;

    if (pData->flags & OPAL_DATATYPE_FLAG_COMMITTED) {
        return OPAL_SUCCESS;
    }
    pData->flags |= OPAL_DATATYPE_FLAG_COMMITTED;

    /* We have to compute the displacement of the first non loop item in the
     * description.
     */
    if (0 != pData->size) {
        int index;
        dt_elem_desc_t *pElem = pData->desc.desc;

        index = GET_FIRST_NON_LOOP(pElem);
        assert(pElem[index].elem.common.flags & OPAL_DATATYPE_FLAG_DATA);
        first_elem_disp = pElem[index].elem.disp;
    }

    /* let's add a fake element at the end just to avoid useless comparaisons
     * in pack/unpack functions.
     */
    pLast->common.type = OPAL_DATATYPE_END_LOOP;
    pLast->common.flags = 0;
    pLast->items = pData->desc.used;
    pLast->first_elem_disp = first_elem_disp;
    pLast->size = pData->size;

    /* If there is no datatype description how can we have an optimized description ? */
    if (0 == pData->desc.used) {
        pData->opt_desc.length = 0;
        pData->opt_desc.desc = NULL;
        pData->opt_desc.used = 0;
        return OPAL_SUCCESS;
    }

    /* If the data is contiguous is useless to generate an optimized version. */
    /*if( pData->size == (pData->true_ub - pData->true_lb) ) return OPAL_SUCCESS; */

    (void) opal_datatype_optimize_short(pData, 1, &(pData->opt_desc));
    if (0 != pData->opt_desc.used) {
        /* let's add a fake element at the end just to avoid useless comparaisons
         * in pack/unpack functions.
         */
        pLast = &(pData->opt_desc.desc[pData->opt_desc.used].end_loop);
        pLast->common.type = OPAL_DATATYPE_END_LOOP;
        pLast->common.flags = 0;
        pLast->items = pData->opt_desc.used;
        pLast->first_elem_disp = first_elem_disp;
        pLast->size = pData->size;
    }

    if( pData->iov == NULL ){
        opal_generate_iovec( pData );
        opal_datatype_create_jit_pack( pData );
        opal_datatype_create_jit_partial_pack( pData );
    }

    return OPAL_SUCCESS;
}

int32_t
opal_generate_iovec( opal_datatype_t *pData )
{
    opal_convertor_t *local_convertor = opal_convertor_create( opal_local_arch, 0 );

    size_t max = SIZE_MAX;
    int rc;

    opal_convertor_prepare_for_send( local_convertor, pData, 1, (void*)0 );

    pData->iovcnt = 512;
    uint32_t save_iov = 0;
    uint32_t leftover_iovec = 0;
    do {
        pData->iovcnt *= 2;
        pData->iov = realloc( pData->iov, sizeof(struct iovec) * pData->iovcnt );
        leftover_iovec = pData->iovcnt - save_iov;
        rc = opal_convertor_raw( local_convertor, pData->iov + save_iov, &leftover_iovec, &max );
        leftover_iovec += save_iov; 
        save_iov = pData->iovcnt;

    } while (0 == rc);

    pData->iov = realloc( pData->iov, sizeof(struct iovec) * leftover_iovec );
    pData->iovcnt = leftover_iovec;

    return 1;
}

void opal_datatype_create_jit_pack( opal_datatype_t *pData )
{
    gcc_jit_context *ctxt;
    gcc_jit_type *char_type, *void_type, *char_ptr_type, *void_ptr_type, *sizet_type,
                 *char_ptr_type_const;

    ctxt = gcc_jit_context_acquire();
    char_type = gcc_jit_context_get_type( ctxt, GCC_JIT_TYPE_CHAR );
    void_type = gcc_jit_context_get_type( ctxt, GCC_JIT_TYPE_VOID );
    sizet_type = gcc_jit_context_get_type( ctxt, GCC_JIT_TYPE_SIZE_T );

    char_ptr_type = gcc_jit_type_get_pointer( char_type );
    void_ptr_type = gcc_jit_type_get_pointer( void_type );
    char_ptr_type_const = gcc_jit_type_get_const( char_ptr_type );

    gcc_jit_param *param[2] = {
        gcc_jit_context_new_param( ctxt, NULL, char_ptr_type_const, "dst" ),
        gcc_jit_context_new_param( ctxt, NULL, char_ptr_type_const, "src" )
    };

    gcc_jit_function *func;
    func = gcc_jit_context_new_function( ctxt, NULL,
		    GCC_JIT_FUNCTION_EXPORTED,
		    void_type,
		    "pack_function",
		    2, param,
		    0 );

    gcc_jit_block *block = gcc_jit_function_new_block( func, "initial" );

    size_t total_disp = 0;
    ptrdiff_t diff = 0;

    gcc_jit_lvalue *dst_val = gcc_jit_function_new_local( func, NULL, char_ptr_type, "dst_val" ),
                   *src_val = gcc_jit_function_new_local( func, NULL, char_ptr_type, "src_val" );
    gcc_jit_block_add_assignment( block, NULL, dst_val, gcc_jit_param_as_rvalue( param[0] ) );
    gcc_jit_block_add_assignment( block, NULL, src_val, gcc_jit_param_as_rvalue( param[1] ) );

    for( int i = 0; i < pData->iovcnt; i++ ){

        if( i == 0 ){
            total_disp = 0;
        } else {
            total_disp += pData->iov[i-1].iov_len;
        }

	gcc_jit_lvalue *dst_input,
		       *src_input;
        
        dst_input = gcc_jit_lvalue_get_address(
                        gcc_jit_context_new_array_access( ctxt, NULL,
                            gcc_jit_lvalue_as_rvalue( dst_val ),
                            gcc_jit_context_new_rvalue_from_long( ctxt, sizet_type, total_disp ) ),
                        NULL );

        
        src_input = gcc_jit_lvalue_get_address(
                        gcc_jit_context_new_array_access( ctxt, NULL,
                            gcc_jit_lvalue_as_rvalue( src_val ),
                            gcc_jit_context_new_rvalue_from_long( ctxt, sizet_type, (ptrdiff_t)(pData->iov[i].iov_base) ) ),
                        NULL );

        gcc_jit_rvalue *args[3] = {
            gcc_jit_context_new_cast( ctxt, NULL,
                                      gcc_jit_lvalue_as_rvalue( dst_input ),
                                      void_ptr_type ),
            gcc_jit_context_new_cast( ctxt, NULL,
                                      gcc_jit_lvalue_as_rvalue( src_input ),
                                      void_ptr_type ),
            gcc_jit_context_new_rvalue_from_long( ctxt, sizet_type, pData->iov[i].iov_len )
        };

        gcc_jit_function *builtin_memcpy = gcc_jit_context_get_builtin_function( ctxt, "__builtin_memcpy" );

        gcc_jit_rvalue *memcpy_call = gcc_jit_context_new_call(
            ctxt, NULL,
            builtin_memcpy,
            3,
            args );
        gcc_jit_block_add_eval( block, NULL, memcpy_call );
    }

    gcc_jit_block_end_with_void_return( block, NULL );

    
    gcc_jit_result *result = NULL;

    result = gcc_jit_context_compile( ctxt );
    gcc_jit_context_release( ctxt );

    void *pack_func = gcc_jit_result_get_code( result, "pack_function" );
    pData->jit_pack = (pack_type)pack_func;
 
    return;
}

static char* itoa(int value, char* result, int base) {
	if (base < 2 || base > 36) { *result = '\0'; return result; }

	char* ptr = result, *ptr1 = result, tmp_char;
	int tmp_value;

	do {
		tmp_value = value;
		value /= base;
		*ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz" [35 + (tmp_value - value * base)];
	} while ( value );

	if (tmp_value < 0) *ptr++ = '-';
	*ptr-- = '\0';
	while(ptr1 < ptr) {
		tmp_char = *ptr;
		*ptr--= *ptr1;
		*ptr1++ = tmp_char;
	}
	return result;
}

void opal_datatype_create_jit_partial_pack( opal_datatype_t *pData )
{
    gcc_jit_context *ctxt;
    gcc_jit_type *char_type, *void_type, *char_ptr_type, *void_ptr_type,
                 *char_ptr_ptr_type,
                 *int_type, *int_ptr_type,
                 *sizet_type, *sizet_ptr_type,
                 *bool_type;

    ctxt = gcc_jit_context_acquire();
    char_type = gcc_jit_context_get_type( ctxt, GCC_JIT_TYPE_CHAR );
    void_type = gcc_jit_context_get_type( ctxt, GCC_JIT_TYPE_VOID );
    sizet_type = gcc_jit_context_get_type( ctxt, GCC_JIT_TYPE_SIZE_T );
    int_type = gcc_jit_context_get_type( ctxt, GCC_JIT_TYPE_INT );
    bool_type = gcc_jit_context_get_type( ctxt, GCC_JIT_TYPE_BOOL );

    char_ptr_type = gcc_jit_type_get_pointer( char_type );
    char_ptr_ptr_type = gcc_jit_type_get_pointer( char_ptr_type );
    void_ptr_type = gcc_jit_type_get_pointer( void_type );
    int_ptr_type = gcc_jit_type_get_pointer( int_type );
    sizet_ptr_type = gcc_jit_type_get_pointer( sizet_type );

    gcc_jit_param *param[7] = {
        gcc_jit_context_new_param( ctxt, NULL, char_ptr_ptr_type, "dst" ),
        gcc_jit_context_new_param( ctxt, NULL, char_ptr_type, "src" ),
        gcc_jit_context_new_param( ctxt, NULL, sizet_ptr_type, "count" ),
        gcc_jit_context_new_param( ctxt, NULL, int_ptr_type, "index" ),
        gcc_jit_context_new_param( ctxt, NULL, sizet_ptr_type, "totdisp" ),
        gcc_jit_context_new_param( ctxt, NULL, sizet_ptr_type, "disp" ),
        gcc_jit_context_new_param( ctxt, NULL, sizet_ptr_type, "max_data" )
    };

    gcc_jit_function *func;
    func = gcc_jit_context_new_function( ctxt, NULL,
		    GCC_JIT_FUNCTION_EXPORTED,
		    int_type,
		    "pack_partial_function",
		    7, param,
		    0 );

    gcc_jit_block *block = gcc_jit_function_new_block( func, "partial_initial" ),
                  *end_block = gcc_jit_function_new_block( func, "termination_block" );

    gcc_jit_lvalue *dst_input = gcc_jit_function_new_local( func, NULL, char_ptr_ptr_type, "dst_input" ),
                   *src_input = gcc_jit_function_new_local( func, NULL, char_ptr_type, "src_input" ),
                   *full_src_arg = gcc_jit_function_new_local( func, NULL, char_ptr_type, "full_src_input" ),
                   *full_src_disp_arg = gcc_jit_function_new_local( func, NULL, char_ptr_type, "full_src_disp_input" ),
                   *full_src_len_arg = gcc_jit_function_new_local( func, NULL, char_ptr_type, "full_src_len_input" ),
                   *partial_src_arg = gcc_jit_function_new_local( func, NULL, char_ptr_type, "partial_src_input" ),
                   *count_val = gcc_jit_function_new_local( func, NULL, sizet_ptr_type, "count_val" ),
                   *index_val = gcc_jit_function_new_local( func, NULL, int_ptr_type, "index_val" ),
                   *totdisp_val = gcc_jit_function_new_local( func, NULL, sizet_ptr_type, "totdisp_val" ),
                   *disp_val = gcc_jit_function_new_local( func, NULL, sizet_ptr_type, "disp_val" ),
                   *maxdata_val = gcc_jit_function_new_local( func, NULL, sizet_ptr_type, "maxdata_val" ),
                   *track = gcc_jit_function_new_local( func, NULL, sizet_type, "track_val" ),
                   *offset = gcc_jit_function_new_local( func, NULL, sizet_type, "offset" );

    gcc_jit_block_add_assignment( block, NULL, dst_input, gcc_jit_param_as_rvalue( param[0] ) );
    gcc_jit_block_add_assignment( block, NULL, src_input, gcc_jit_param_as_rvalue( param[1] ) );
    gcc_jit_block_add_assignment( block, NULL, count_val, gcc_jit_param_as_rvalue( param[2] ) );
    gcc_jit_block_add_assignment( block, NULL, index_val, gcc_jit_param_as_rvalue( param[3] ) );
    gcc_jit_block_add_assignment( block, NULL, totdisp_val, gcc_jit_param_as_rvalue( param[4] ) );
    gcc_jit_block_add_assignment( block, NULL, disp_val, gcc_jit_param_as_rvalue( param[5] ) );
    gcc_jit_block_add_assignment( block, NULL, maxdata_val, gcc_jit_param_as_rvalue( param[6] ) );

    gcc_jit_block_add_assignment( block, NULL, track, gcc_jit_context_zero( ctxt, sizet_type ) );

    gcc_jit_lvalue *dst_val = gcc_jit_context_new_array_access( ctxt, NULL, dst_input,
                                  gcc_jit_context_new_rvalue_from_int( ctxt, int_type, 0 ) ),
                   *count = gcc_jit_context_new_array_access( ctxt, NULL, count_val,
                                gcc_jit_context_new_rvalue_from_int( ctxt, int_type, 0 ) ),
                   *index = gcc_jit_context_new_array_access( ctxt, NULL, index_val,
                                gcc_jit_context_new_rvalue_from_int( ctxt, int_type, 0 ) ),
                   *totdisp = gcc_jit_context_new_array_access( ctxt, NULL, totdisp_val,
                                  gcc_jit_context_new_rvalue_from_int( ctxt, int_type, 0 ) ),
                   *disp = gcc_jit_context_new_array_access( ctxt, NULL, disp_val,
                               gcc_jit_context_new_rvalue_from_int( ctxt, int_type, 0 ) ),
                   *maxdata = gcc_jit_context_new_array_access( ctxt, NULL, maxdata_val,
                                  gcc_jit_context_new_rvalue_from_int( ctxt, int_type, 0 ) );
    gcc_jit_function *builtin_memcpy = gcc_jit_context_get_builtin_function( ctxt, "__builtin_memcpy" );

    //if( *max_data == 0 || convertor->pStack[0].count == 0 )
    //    return 0;
    gcc_jit_rvalue *check_maxdata = gcc_jit_context_new_comparison( ctxt, NULL, GCC_JIT_COMPARISON_EQ,
                                        maxdata, gcc_jit_context_zero( ctxt, sizet_type ) ),
                   *check_count = gcc_jit_context_new_comparison( ctxt, NULL, GCC_JIT_COMPARISON_EQ,
                                      count, gcc_jit_context_zero( ctxt, sizet_type ) );

    gcc_jit_rvalue *check_or = gcc_jit_context_new_binary_op( ctxt, NULL, GCC_JIT_BINARY_OP_LOGICAL_OR,
                                   bool_type, check_maxdata, check_count );
    
    int blockn = 0;
    char block_yes[10], block_no[10];
    gcc_jit_block *check_true = gcc_jit_function_new_block( func, itoa( blockn++, block_yes, 10 ) ),
                  *check_false = gcc_jit_function_new_block( func, itoa( blockn++, block_no, 10 ) );
    gcc_jit_block_end_with_conditional( block, NULL, check_or, check_true, check_false );

    // return 0 in check no
    gcc_jit_block_end_with_return( check_true, NULL, gcc_jit_context_zero( ctxt, int_type ) );

    // const opal_datatype_t *pData = convertor->pDesc;
    // struct iovec *iov = pData->iov;
    // size_t track = *max_data;
    // uint32_t i;
    // *src = convertor->pBaseBuf + convertor->pStack[0].disp;
    gcc_jit_lvalue *src_val = gcc_jit_lvalue_get_address( 
                                  gcc_jit_context_new_array_access( ctxt, NULL, 
                                      gcc_jit_lvalue_as_rvalue( src_input ),
                                      totdisp ), NULL );
    // for( i = convertor->pStack[1].index; i < pData->iovcnt; i++ ) {
    gcc_jit_block *elem_block[ pData->iovcnt ];
    for( int i = 0; i < pData->iovcnt; i++ ){
        char block_name[10];
        elem_block[i] = gcc_jit_function_new_block( func, itoa( blockn++, block_name, 10 ) );
    }

    gcc_jit_block_end_with_jump( check_false, NULL, elem_block[0] );
    for( int i = 0; i < pData->iovcnt; i++ ){
        /* first check if index is correct */
        gcc_jit_rvalue *index_check = gcc_jit_context_new_comparison( ctxt, NULL,
                                          GCC_JIT_COMPARISON_EQ,
                                          gcc_jit_lvalue_as_rvalue( index ), 
                                          gcc_jit_context_new_rvalue_from_int( ctxt, int_type, i ) );
        /* stay at current block or goto next block */
        gcc_jit_block *right_index, *wrong_index;
        char block_name1[10], block_name2[10];
        right_index = gcc_jit_function_new_block( func, itoa( blockn++, block_name1, 10 ) );
        wrong_index = gcc_jit_function_new_block( func, itoa( blockn++, block_name2, 10 ) );
        
        gcc_jit_block_end_with_conditional( elem_block[i], NULL, index_check,
                                            right_index, wrong_index );

        if( i != pData->iovcnt - 1 )
            gcc_jit_block_end_with_jump( wrong_index, NULL, elem_block[ i+1 ] );
        else 
            gcc_jit_block_end_with_jump( wrong_index, NULL, end_block );

        // if( track < (iov[i].iov_len - convertor->pStack[1].disp) || track == 0 ) {
        /* get iov[i].iov_len - disp into a variable, len_diff */
        gcc_jit_lvalue *len_diff = gcc_jit_function_new_local( func, NULL, sizet_type, "length diff" );
        gcc_jit_block_add_assignment( right_index, NULL, len_diff, 
                                      gcc_jit_context_new_rvalue_from_long( ctxt, sizet_type,
                                                                            pData->iov[i].iov_len ) );
        gcc_jit_block_add_assignment_op( right_index, NULL, len_diff, GCC_JIT_BINARY_OP_MINUS, disp );
    
        /* check expr track < (iov[i].iov_len - convertor->pStack[1].disp */
        gcc_jit_rvalue *len_diff_expr = gcc_jit_context_new_comparison( ctxt, NULL,
                                            GCC_JIT_COMPARISON_LT,
                                            maxdata, len_diff );

        gcc_jit_block *partial_copy, *full_copy;
        char block_name3[10], block_name4[10];
        partial_copy = gcc_jit_function_new_block( func, itoa( blockn++, block_name3, 10 ) );
        full_copy = gcc_jit_function_new_block( func, itoa( blockn++, block_name4, 10 ) );

        gcc_jit_block_end_with_conditional( right_index, NULL, len_diff_expr,
                                            partial_copy, full_copy );

        /* partial copy */
        //        memcpy( *dst,
        //                *src + (ptrdiff_t)(iov[i].iov_base) + convertor->pStack[1].disp,
        //                 track);

        gcc_jit_block_add_assignment( partial_copy, NULL, offset,
                                      gcc_jit_context_new_binary_op( ctxt, NULL, GCC_JIT_BINARY_OP_PLUS,
                                          sizet_type,
                                          gcc_jit_lvalue_as_rvalue( disp ),
                                          gcc_jit_context_new_rvalue_from_long( ctxt, sizet_type, (ptrdiff_t)(pData->iov[i].iov_base) ) ) );
        partial_src_arg = gcc_jit_lvalue_get_address( 
                              gcc_jit_context_new_array_access( ctxt, NULL,
                                  gcc_jit_lvalue_as_rvalue( src_val ),
                                  gcc_jit_lvalue_as_rvalue( offset ) ), NULL );

        gcc_jit_rvalue *partial_args[3] = {
            gcc_jit_context_new_cast( ctxt, NULL,
                                      gcc_jit_lvalue_as_rvalue( dst_val ),
                                      void_ptr_type ),
            gcc_jit_context_new_cast( ctxt, NULL,
                                      gcc_jit_lvalue_as_rvalue( partial_src_arg ),
                                      void_ptr_type ),
            gcc_jit_lvalue_as_rvalue( maxdata )
        };

        gcc_jit_rvalue *partial_memcpy_call = gcc_jit_context_new_call(
            ctxt, NULL,
            builtin_memcpy,
            3,
            partial_args );
        gcc_jit_block_add_eval( partial_copy, NULL, partial_memcpy_call );

        //       *dst += track;
        gcc_jit_block_add_assignment( partial_copy, NULL, dst_val,
            gcc_jit_lvalue_get_address( 
                gcc_jit_context_new_array_access( ctxt, NULL, 
                    gcc_jit_lvalue_as_rvalue( dst_val ), 
                    gcc_jit_lvalue_as_rvalue( maxdata ) ), NULL ) );
        
        //       convertor->pStack[1].disp += track;
        gcc_jit_block_add_assignment_op( partial_copy, NULL, disp, GCC_JIT_BINARY_OP_PLUS, maxdata );
        //       convertor->pStack[1].index = i;
        gcc_jit_block_add_assignment( partial_copy, NULL, index, 
            gcc_jit_context_new_rvalue_from_int( ctxt, int_type, i ) );
        //       *max_data = 0;
        gcc_jit_block_add_assignment( partial_copy, NULL, maxdata, gcc_jit_context_zero( ctxt, sizet_type ) );
        //       return 0;
        gcc_jit_block_end_with_return( partial_copy, NULL, gcc_jit_context_zero( ctxt, int_type ) );


        /* full copy */
        //    memcpy( *dst,
        //            *src + (ptrdiff_t)(iov[i].iov_base) + convertor->pStack[1].disp,
        //            iov[i].iov_len - convertor->pStack[1].disp );
        gcc_jit_block *full_len_copy, *full_disp_copy;
        char block_name5[10], block_name6[10];
        full_len_copy = gcc_jit_function_new_block( func, itoa( blockn++, block_name5, 10 ) );
        full_disp_copy = gcc_jit_function_new_block( func, itoa( blockn++, block_name6, 10 ) );

        gcc_jit_rvalue *disp_eq = gcc_jit_context_new_comparison( ctxt, NULL,
                                      GCC_JIT_COMPARISON_EQ,
                                      disp,
                                      gcc_jit_context_zero( ctxt, sizet_type ) );

        gcc_jit_block_end_with_conditional( full_copy, NULL, disp_eq, full_len_copy, full_disp_copy );

      
        /* full disp copy */
        gcc_jit_block_add_assignment( full_disp_copy, NULL, offset,
                                      gcc_jit_context_new_binary_op( ctxt, NULL, GCC_JIT_BINARY_OP_PLUS,
                                          sizet_type,
                                          gcc_jit_lvalue_as_rvalue( disp ),
                                          gcc_jit_context_new_rvalue_from_long( ctxt, sizet_type, (ptrdiff_t)(pData->iov[i].iov_base) ) ) );

        gcc_jit_block_add_assignment( full_disp_copy, NULL, len_diff,
                                      gcc_jit_context_new_rvalue_from_long( ctxt, sizet_type,
                                                                            pData->iov[i].iov_len ) );
        gcc_jit_block_add_assignment_op( full_disp_copy, NULL, len_diff, GCC_JIT_BINARY_OP_MINUS, disp );

        full_src_disp_arg = gcc_jit_lvalue_get_address(
                           gcc_jit_context_new_array_access( ctxt, NULL,
                               gcc_jit_lvalue_as_rvalue( src_val ), 
                               gcc_jit_lvalue_as_rvalue( offset ) ), NULL );

        gcc_jit_rvalue *full_disp_args[3] = {
            gcc_jit_context_new_cast( ctxt, NULL,
                                      gcc_jit_lvalue_as_rvalue( dst_val ),
                                      void_ptr_type ),
            gcc_jit_context_new_cast( ctxt, NULL,
                                      gcc_jit_lvalue_as_rvalue( full_src_disp_arg ),
                                      void_ptr_type ),
            gcc_jit_lvalue_as_rvalue( len_diff )
        };

        gcc_jit_rvalue *full_disp_memcpy_call = gcc_jit_context_new_call(
            ctxt, NULL,
            builtin_memcpy,
            3,
            full_disp_args );
        gcc_jit_block_add_eval( full_disp_copy, NULL, full_disp_memcpy_call ); 


        //    *dst += iov[i].iov_len - convertor->pStack[1].disp;
        gcc_jit_block_add_assignment( full_disp_copy, NULL, dst_val,
            gcc_jit_lvalue_get_address( 
                gcc_jit_context_new_array_access( ctxt, NULL,
                    gcc_jit_lvalue_as_rvalue( dst_val ),
                    gcc_jit_lvalue_as_rvalue( len_diff) ), NULL ) );
        
        //    track -= iov[i].iov_len - convertor->pStack[1].disp;
        gcc_jit_block_add_assignment_op( full_disp_copy, NULL, maxdata, GCC_JIT_BINARY_OP_MINUS,
            gcc_jit_lvalue_as_rvalue( len_diff ) );
        //    *max_data -= iov[i].iov_len - convertor->pStack[1].disp;
        //    convertor->pStack[1].disp = 0;
        gcc_jit_block_add_assignment( full_disp_copy, NULL, disp, gcc_jit_context_zero( ctxt, sizet_type ) );
        //    index++
        gcc_jit_block_add_assignment_op( full_disp_copy, NULL, index, GCC_JIT_BINARY_OP_PLUS,
            gcc_jit_context_one( ctxt, int_type ) );

        /* jumping to next block or to end block */
        if( i != pData->iovcnt - 1 )
            gcc_jit_block_end_with_jump( full_disp_copy, NULL, elem_block[ i+1 ] );
        else 
            gcc_jit_block_end_with_jump( full_disp_copy, NULL, end_block );


        /* full length copy */
        gcc_jit_block_add_assignment( full_len_copy, NULL, offset,
                                      gcc_jit_context_new_binary_op( ctxt, NULL, GCC_JIT_BINARY_OP_PLUS,
                                          sizet_type,
                                          gcc_jit_lvalue_as_rvalue( disp ),
                                          gcc_jit_context_new_rvalue_from_long( ctxt, sizet_type, (ptrdiff_t)(pData->iov[i].iov_base) ) ) );

        full_src_len_arg = gcc_jit_lvalue_get_address(
                           gcc_jit_context_new_array_access( ctxt, NULL,
                               gcc_jit_lvalue_as_rvalue( src_val ), 
                               gcc_jit_lvalue_as_rvalue( offset ) ), NULL );

        gcc_jit_rvalue *full_len_args[3] = {
            gcc_jit_context_new_cast( ctxt, NULL,
                                      gcc_jit_lvalue_as_rvalue( dst_val ),
                                      void_ptr_type ),
            gcc_jit_context_new_cast( ctxt, NULL,
                                      gcc_jit_lvalue_as_rvalue( full_src_len_arg ),
                                      void_ptr_type ),
            gcc_jit_context_new_rvalue_from_long( ctxt, sizet_type, pData->iov[i].iov_len )
        };

        gcc_jit_rvalue *full_len_memcpy_call = gcc_jit_context_new_call(
            ctxt, NULL,
            builtin_memcpy,
            3,
            full_len_args );
        gcc_jit_block_add_eval( full_len_copy, NULL, full_len_memcpy_call ); 


        //    *dst += iov[i].iov_len - convertor->pStack[1].disp;
        gcc_jit_block_add_assignment( full_len_copy, NULL, dst_val,
            gcc_jit_lvalue_get_address( 
                gcc_jit_context_new_array_access( ctxt, NULL,
                    gcc_jit_lvalue_as_rvalue( dst_val ),
                    gcc_jit_context_new_rvalue_from_long( ctxt, sizet_type, pData->iov[i].iov_len ) ), NULL ) );
        
        //    track -= iov[i].iov_len - convertor->pStack[1].disp;
        gcc_jit_block_add_assignment_op( full_len_copy, NULL, maxdata, GCC_JIT_BINARY_OP_MINUS,
            gcc_jit_context_new_rvalue_from_long( ctxt, sizet_type, pData->iov[i].iov_len ) );
        //    *max_data -= iov[i].iov_len - convertor->pStack[1].disp;
        //    convertor->pStack[1].disp = 0;
        gcc_jit_block_add_assignment( full_len_copy, NULL, disp, gcc_jit_context_zero( ctxt, sizet_type ) );
        //    index++
        gcc_jit_block_add_assignment_op( full_len_copy, NULL, index, GCC_JIT_BINARY_OP_PLUS,
            gcc_jit_context_one( ctxt, int_type ) );

        /* jumping to next block or to end block */
        if( i != pData->iovcnt - 1 )
            gcc_jit_block_end_with_jump( full_len_copy, NULL, elem_block[ i+1 ] );
        else 
            gcc_jit_block_end_with_jump( full_len_copy, NULL, end_block );

    }

    // convertor->pStack[0].disp += pData->ub - pData->lb;
    gcc_jit_block_add_assignment_op( end_block, NULL, totdisp, GCC_JIT_BINARY_OP_PLUS,
        gcc_jit_context_new_rvalue_from_long( ctxt, sizet_type, pData->ub - pData->lb ) );
    // convertor->pStack[1].index = 0;
    gcc_jit_block_add_assignment( end_block, NULL, index, gcc_jit_context_zero( ctxt, int_type ) );
    // convertor->pStack[0].count--;
    gcc_jit_block_add_assignment_op( end_block, NULL, count, GCC_JIT_BINARY_OP_MINUS,
                                     gcc_jit_context_one( ctxt, sizet_type ) );
    // return 1
    gcc_jit_block_end_with_return( end_block, NULL, gcc_jit_context_one( ctxt, int_type ) );
    
    gcc_jit_result *result = NULL;
    result = gcc_jit_context_compile( ctxt );
    gcc_jit_context_release( ctxt );

    void *pack_func = gcc_jit_result_get_code( result, "pack_partial_function" );
    pData->jit_partial_pack = (pack_partial_type)pack_func;
 
    return;

}
