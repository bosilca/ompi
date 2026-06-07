/*
 * Copyright (c) 2024      Triad National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/*
 * Bigcount array conversion macros for Fortran templates.
 */

#define OMPI_FORTRAN_BIGCOUNT_ARRAY_SET(array, tmp_array, n) \
    do { \
        if (sizeof(*(array)) == sizeof(*(tmp_array))) { \
            (tmp_array) = (void *)(array); \
        } else { \
            (tmp_array) = malloc(sizeof(*tmp_array) * n); \
            for (int bigcount_array_i = 0; bigcount_array_i < n; ++bigcount_array_i) { \
                (tmp_array)[bigcount_array_i] = (array)[bigcount_array_i]; \
            } \
        } \
    } while (0)

#define OMPI_FORTRAN_BIGCOUNT_ARRAY_COPYOUT(array, tmp_array, n) \
    do { \
        if ((array) != (tmp_array) && NULL != (tmp_array)) { \
            for (int bigcount_array_i = 0; bigcount_array_i < n; ++bigcount_array_i) { \
                (array)[bigcount_array_i] = (tmp_array)[bigcount_array_i]; \
            } \
        } \
    } while (0)


#define OMPI_FORTRAN_BIGCOUNT_ARRAY_CLEANUP(array, tmp_array) \
    do { \
        if ((void *)(array) != (void *)(tmp_array) && NULL != (tmp_array)) { \
            free(tmp_array); \
            tmp_array = NULL; \
        } \
    } while (0)

/*
 * Mark a converted (temporary) array to be freed when the non-blocking /
 * persistent collective request completes. The array is the same pointer the
 * back-end C binding stored in the request's coll args descriptor, so we just
 * set the matching OMPI_COLL_ARGS_FREE_* bit and let the coll layer free it.
 * free_bit identifies which slot of the descriptor (src/dst counts /
 * displacements / datatypes) the array occupies. idx counts how many arrays
 * were flagged so the caller knows whether to install the release callback.
 */
#define OMPI_FORTRAN_BIGCOUNT_ARRAY_CLEANUP_NONBLOCKING(array, tmp_array, c_request, c_ierr, idx, free_bit) \
    do { \
        if (MPI_SUCCESS == (c_ierr)) { \
            if ((void *)(array) != (void *)(tmp_array) && (tmp_array) != NULL) { \
                ((ompi_coll_base_nbc_request_t *) (c_request))->args.mask |= (free_bit); \
                (idx)++;                                                         \
            } \
        } else { \
            OMPI_FORTRAN_BIGCOUNT_ARRAY_CLEANUP((array), (tmp_array)); \
        } \
    } while (0)
