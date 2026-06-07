/*
 * Copyright (c) 2021-2024 Computer Architecture and VLSI Systems (CARV)
 *                         Laboratory, ICS Forth. All rights reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "mpi.h"

#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/op/op.h"

#include "opal/mca/rcache/base/base.h"
#include "opal/util/show_help.h"
#include "opal/util/minmax.h"

#include "coll_xhc.h"

int mca_coll_xhc_reduce(ompi_coll_args_t *args, struct ompi_communicator_t *comm, mca_coll_base_module_t *module) {

    ompi_communicator_t *ompi_comm = comm;

    xhc_module_t *xhc_module = (xhc_module_t *) module;

    /* Currently, XHC's reduce only supports the top-level
     * owner as the root (typically rank 0). */
    if(0 == args->root) {
        return mca_coll_xhc_allreduce_internal(args->src.info.buffer, args->dst.info.buffer, args->dst.info.count,
            args->dst.info.datatype, args->op, ompi_comm, module, false);
    } else {
        WARN_ONCE("coll:xhc: Warning: XHC does not currently support "
            "non-zero-root reduce; utilizing fallback component");
        return XHC_CALL_FALLBACK(xhc_module->prev_colls, XHC_REDUCE,
            reduce, args, ompi_comm);
    }
}
