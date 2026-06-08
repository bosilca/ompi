/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#ifndef OMPI_UTIL_DATATYPE_ARRAY_H
#define OMPI_UTIL_DATATYPE_ARRAY_H

#include "ompi_config.h"

#include "mpi.h"
#include "opal/class/opal_pointer_array.h"
#include "ompi/datatype/ompi_datatype.h"

/*
 * Per-peer datatype array (the alltoallw / neighbor_alltoallw family).
 *
 * Like ompi_count_array_t / ompi_disp_array_t, this is a tagged pointer that
 * lets the collective back-end read a datatype array element-by-element
 * without copying it into an intermediate buffer. Two element representations
 * are supported, selected by the least significant bit of the stored pointer
 * (which is always free because both representations are at least 2-byte
 * aligned):
 *
 *   - C form (LSB == 0): the pointer is a plain "struct ompi_datatype_t *
 *     const *" array, exactly as the C MPI bindings receive it. This is the
 *     common/hot path; accessing an element is a single load.
 *
 *   - Fortran form (LSB == 1): the pointer is an "MPI_Fint *" array of Fortran
 *     datatype handles. Accessing an element resolves the handle through the
 *     same f-to-c table that MPI_Type_f2c() uses, so the Fortran bindings can
 *     pass their handle array straight through instead of allocating and
 *     converting a temporary "MPI_Datatype" array.
 *
 * NOTE: resolving a Fortran handle on access is only safe while the underlying
 *       handle is still valid (i.e. for the duration of a blocking call). A
 *       non-blocking request that must outlive the call has to capture stable
 *       object pointers at initiation time (see ompi_coll_base_retain_datatypes_w).
 */
typedef intptr_t ompi_datatype_array_t;

_Static_assert(_Alignof(MPI_Fint) >= 2, "MPI_Fint alignment assumption violated");

#define OMPI_DATATYPE_ARRAY_NULL ((ompi_datatype_array_t) 0)

/* Initialize a C-pointer-array variant (the representation used by the C
 * bindings and stored by the retain helpers). */
static inline void ompi_datatype_array_init(ompi_datatype_array_t *array,
                                            struct ompi_datatype_t * const *data)
{
    assert(((intptr_t) data & 0x1L) == 0);
    *array = (intptr_t) data;
}

/* Initialize a Fortran-handle-array variant (no conversion, no copy). */
static inline void ompi_datatype_array_init_f(ompi_datatype_array_t *array,
                                              const MPI_Fint *data)
{
    assert(((intptr_t) data & 0x1L) == 0);
    *array = (intptr_t) data | 0x1L;
}

static inline ompi_datatype_array_t
ompi_datatype_array_create(struct ompi_datatype_t * const *data)
{
    ompi_datatype_array_t array;
    ompi_datatype_array_init(&array, data);
    return array;
}

static inline ompi_datatype_array_t
ompi_datatype_array_create_f(const MPI_Fint *data)
{
    ompi_datatype_array_t array;
    ompi_datatype_array_init_f(&array, data);
    return array;
}

/* Return true if the array holds Fortran handles (resolved on access). */
static inline bool ompi_datatype_array_is_fortran(ompi_datatype_array_t array)
{
    return (array & 0x1L) != 0;
}

/* Return the underlying (untagged) data pointer. */
static inline const void *ompi_datatype_array_ptr(ompi_datatype_array_t array)
{
    return (const void *) (array & ~0x1L);
}

/* Get the resolved datatype at index i. For a Fortran-handle array this
 * performs the same lookup as MPI_Type_f2c(), returning NULL for an invalid
 * handle (per MPI-2:4.12.4 semantics). */
static inline struct ompi_datatype_t *ompi_datatype_array_get(ompi_datatype_array_t array,
                                                              size_t i)
{
    if (OPAL_LIKELY(0 == (array & 0x1L))) {
        return ((struct ompi_datatype_t * const *) array)[i];
    }
    const MPI_Fint *handles = (const MPI_Fint *) (array & ~0x1L);
    int index = (int) handles[i];
    if (index < 0 || index >= opal_pointer_array_get_size(&ompi_datatype_f_to_c_table)) {
        return NULL;
    }
    return (struct ompi_datatype_t *) opal_pointer_array_get_item(&ompi_datatype_f_to_c_table,
                                                                  index);
}

#endif /* OMPI_UTIL_DATATYPE_ARRAY_H */
