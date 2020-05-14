/*
 * Copyright (c) 2018-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "coll_han.h"

/* Get root's low_rank and up_rank from vranks array */
void mca_coll_han_get_ranks(int *vranks, int root, int low_size, int *root_low_rank,
                            int *root_up_rank)
{
    *root_up_rank = vranks[root] / low_size;
    *root_low_rank = vranks[root] % low_size;
}
