/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 * SPDX-License-Identifier: BSD-3-Clause-Open-MPI
 */

#ifndef _COMMON_UCX_MD_H_
#define _COMMON_UCX_MD_H_

#include "opal_config.h"

#include <uct/api/uct.h>

#include "opal/class/opal_list.h"
#include "opal/mca/mca.h"

BEGIN_C_DECLS

/*
 * The node's UCT memory domains, opened at most once each per process.
 *
 * Opening a memory domain is not free: it opens the device and builds a
 * registration cache.  More than one UCX user in a process tends to want the
 * same domains -- btl/uct opens the ones in its include list, and
 * opal_common_ucx_support_level() has to look at every domain on the node to
 * decide whether UCX is worth using here -- so left to themselves they open
 * the same devices twice.
 *
 * Users take a reference on a domain by name; it is opened on the first
 * reference and closed after the last.  A domain's transport list is queried
 * once and kept for the life of the process, so a caller that only wants to
 * know what a domain offers does not keep the device open, and does not
 * reopen it if someone asked before.
 *
 * This layer deliberately knows nothing about UCP.  A UCP context opens its
 * own memory domains internally and does not export them, so nothing here
 * can be shared with a UCP user; these are the domains of UCT consumers and
 * of the node inventory only.
 */
struct opal_common_ucx_md_t {
    opal_list_item_t super;

    /** name of this memory domain, e.g. "mlx5_0" */
    char *md_name;

    /** component this domain belongs to */
    uct_component_h uct_component;

    /** open while refcnt is non-zero, NULL otherwise */
    uct_md_h uct_md;

    /** valid while uct_md is open; re-queried on each open */
    uct_md_attr_t md_attr;

    /** number of outstanding opal_common_ucx_md_acquire() calls */
    int refcnt;

    /** our copy of the transport list, queried on the first open */
    uct_tl_resource_desc_t *tl_resources;
    unsigned num_tl_resources;
    bool tl_resources_valid;
};

typedef struct opal_common_ucx_md_t opal_common_ucx_md_t;

OBJ_CLASS_DECLARATION(opal_common_ucx_md_t);

/**
 * @brief Announce that this component will use the registry.
 *
 * Refcounted, and must be paired with opal_common_ucx_md_finalize().  The
 * registry outlives the last caller's domains this way: btl/uct keeps its
 * domains for as long as its modules exist, which can be longer than any
 * one other UCX component is around for.
 *
 * Components do not normally call this: opal_common_ucx_mca_register() and
 * opal_common_ucx_mca_deregister() do it for everyone who uses them.
 */
OPAL_DECLSPEC int opal_common_ucx_md_init(void);

/**
 * @brief Drop a reference taken by opal_common_ucx_md_init().
 *
 * The registry is torn down once the last user is gone.  Any domain still
 * referenced at that point is a leak on the user's side; it is left open
 * rather than closed out from under them.
 */
OPAL_DECLSPEC void opal_common_ucx_md_finalize(void);

/**
 * @brief Every memory domain on this node.
 *
 * Built on the first call from the UCT component list and owned by this
 * layer; the caller must not modify or free it.  No domain is opened, so
 * this is cheap enough to call just to iterate names.
 *
 * @param[out] md_list list of opal_common_ucx_md_t
 *
 * @return OPAL_SUCCESS, or OPAL_ERR_NOT_AVAILABLE if UCT cannot be queried
 */
OPAL_DECLSPEC int opal_common_ucx_md_list(opal_list_t **md_list);

/**
 * @brief Find a memory domain without opening it.
 *
 * Domain names are only unique within a component, so both are needed.
 *
 * @return the domain, or NULL if this component offers no such domain
 */
OPAL_DECLSPEC opal_common_ucx_md_t *opal_common_ucx_md_lookup(uct_component_h component,
                                                              const char *md_name);

/**
 * @brief Take a reference on a memory domain, opening it if needed.
 *
 * On success md->uct_md and md->md_attr are valid until the matching
 * opal_common_ucx_md_release().
 */
OPAL_DECLSPEC int opal_common_ucx_md_acquire(opal_common_ucx_md_t *md);

/**
 * @brief Drop a reference taken by opal_common_ucx_md_acquire().
 *
 * The domain is closed once the last reference goes away.  Its transport
 * list stays cached.
 */
OPAL_DECLSPEC void opal_common_ucx_md_release(opal_common_ucx_md_t *md);

/**
 * @brief The transports a memory domain offers.
 *
 * Queried on the first call, which opens the domain if no one holds a
 * reference and closes it again afterwards.  The returned array belongs to
 * this layer and stays valid for the life of the process.
 */
OPAL_DECLSPEC int opal_common_ucx_md_tl_resources(opal_common_ucx_md_t *md,
                                                  const uct_tl_resource_desc_t **tl_resources,
                                                  unsigned *num_tl_resources);

END_C_DECLS

#endif /* _COMMON_UCX_MD_H_ */
