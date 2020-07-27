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

#ifndef MCA_COLL_HAN_TRIGGER_EXPORT_H
#define MCA_COLL_HAN_TRIGGER_EXPORT_H

#include "ompi/communicator/communicator.h"
#include "ompi/op/op.h"
#include "ompi/datatype/ompi_datatype.h"


typedef int (*task_func_ptr) (void *);

struct mca_coll_task_s {
    opal_object_t super;
    task_func_ptr func_ptr;
    void *func_args;
};

typedef struct mca_coll_task_s mca_coll_task_t;

OBJ_CLASS_DECLARATION(mca_coll_task_t);

/* Init task */
int init_task(mca_coll_task_t * t, task_func_ptr func_ptr, void *func_args);

/* Issue the task */
int issue_task(mca_coll_task_t * t);

#endif  /* MCA_COLL_HAN_TRIGGER_EXPORT_H */
