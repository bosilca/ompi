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

#ifndef _COMMON_UCX_CONTEXT_H_
#define _COMMON_UCX_CONTEXT_H_

#include "opal_config.h"

#include <ucp/api/ucp.h>

#include "opal/mca/common/ucx/common_ucx.h"

BEGIN_C_DECLS

/*
 * One UCP context per process, shared by every UCX user in it.
 *
 * A UCP context is not a cheap object: creating one walks every device and
 * transport on the node, opens a memory domain per device and builds a
 * registration cache.  A process holding several of them pays all of that
 * once per context, pins the same buffer once per context, and -- because
 * workers and endpoints belong to a context -- opens one endpoint per
 * context per peer.  Nothing in Open MPI needs a second one.
 *
 * What makes a single context awkward is that ucp_init() takes one feature
 * set, one request layout and one tag_sender_mask, while each user wants
 * something different, and the users do not all show up at the same time:
 * pml/ucx is selected early in MPI_Init, osc/ucx only needs a context when
 * the first window is created, and spml/ucx appears only if the application
 * calls shmem_init(), long after MPI_Init returned.  UCX cannot add a
 * feature to a live context, so the requirements have to be in hand before
 * the first user is served.
 *
 * Hence two phases.  A user *declares* what it needs as early as it can --
 * component open is the natural place, and the attributes may still be
 * amended afterwards, which is how values that are only known later (world
 * size, the thread level) get in -- and *gets* the context when it has real
 * work to do.  The first get() builds one context out of the union of
 * everything declared by then; later gets share it.  A user whose
 * requirements that context cannot meet, because it declared too late or
 * wants a feature nobody else asked for, is handed a private context
 * instead.  Sharing is therefore an optimization and never a correctness
 * requirement, which is also what makes it safe for any one component to
 * switch off with its own share_context MCA parameter.
 *
 * Declarations are scoped to the declaring user: undeclaring withdraws the
 * requirement.  That is what keeps UCP_FEATURE_TAG out of the context of a
 * job where pml/ucx was built and opened but lost the selection to ob1 --
 * closing the component withdraws TAG, and the context osc/ucx creates at
 * the first window is built without it.
 */

/* Enough slots for the UCX users that exist (pml/ucx, osc/ucx, spml/ucx)
 * plus room to grow.  A fixed array rather than a list keeps declarations
 * in caller-owned storage, so a component can declare from a static
 * structure without allocating. */
#define OPAL_COMMON_UCX_CONTEXT_MAX_USERS 8

/* An uninitialized request offset, so that a user that asked for no private
 * request area trips over its offset rather than silently writing over the
 * common header at offset zero. */
#define OPAL_COMMON_UCX_CONTEXT_NO_REQUEST ((size_t) -1)

/* Room for a ucp_config_read() prefix.  The context keeps its own copy
 * rather than the declaring user's pointer: a component may be closed, and
 * its DSO unloaded, while another user still holds the context. */
#define OPAL_COMMON_UCX_CONTEXT_PREFIX_MAX 32

/**
 * What one UCX user needs from a UCP context.
 */
typedef struct opal_common_ucx_context_attr_t {
    /** Name of the declaring user, for verbose output, e.g. "pml/ucx". */
    const char *name;

    /** Prefix passed to ucp_config_read().  Users that disagree about it
     *  cannot share a context: the prefix decides which UCX_* environment
     *  variables apply, so sharing would silently drop someone's tuning. */
    const char *config_prefix;

    /** Whether this user takes part in sharing at all.  A user that opts
     *  out gets a context of its own *and* keeps its requirements out of
     *  the shared one, so that switching it off really does leave the
     *  other users with the context they would have had on their own.
     *  Normally comes straight from this component's share_context MCA
     *  parameter -- see opal_common_ucx_context_var_register(). */
    bool share_context;

    /** UCP_FEATURE_* bits this user cannot work without. */
    uint64_t features;

    /** Tag bits that identify the sender, or zero if this user does not do
     *  tag matching.  Two users that both want one and disagree cannot
     *  share a context. */
    uint64_t tag_sender_mask;

    /** True if this user may drive the context from several threads at
     *  once, i.e. with more than one worker in flight concurrently. */
    bool mt_workers_shared;

    /** Sizing hints; the context uses the largest value declared.  Zero
     *  means "no opinion". */
    int estimated_num_eps;
    int estimated_num_ppn;

    /** Bytes of private space this user wants inside every UCP request of
     *  the context, with the callbacks that construct and destroy it.  UCX
     *  runs these once per pooled request rather than once per operation,
     *  so a constructor here is amortized across every operation that
     *  reuses the request -- which is why pml/ucx can afford to build a
     *  whole ompi_request_t in it.  Zero size means this user needs no
     *  private area beyond the common header. */
    size_t request_size;
    size_t request_align;
    ucp_request_init_callback_t request_init;
    ucp_request_cleanup_callback_t request_cleanup;
} opal_common_ucx_context_attr_t;

struct opal_common_ucx_context_user_t;

/**
 * A UCP context and the terms it was created under.
 */
typedef struct opal_common_ucx_context_t {
    ucp_context_h ucp_context;

    /** Number of users currently holding this context. */
    int refcnt;

    /** True for the context that users share, false for one created
     *  privately for a single user. */
    bool shared;

    /** Terms this context was created with, kept so that a later user can
     *  be told whether it fits. */
    uint64_t features;
    uint64_t tag_sender_mask;
    bool mt_workers_shared;
    char config_prefix[OPAL_COMMON_UCX_CONTEXT_PREFIX_MAX];

    /** Total size of the user area of a UCP request of this context,
     *  common header included. */
    size_t request_size;

    /** Where each user's private request area lives, snapshotted at
     *  creation so that the layout of a live context never moves when a
     *  declaration comes or goes.  A part whose declaring user goes away
     *  keeps its offset -- moving it would corrupt requests already in
     *  flight -- but loses its callbacks, which may live in a DSO that is
     *  about to be unloaded. */
    int nparts;
    struct {
        const struct opal_common_ucx_context_user_t *user;
        size_t offset;
        ucp_request_init_callback_t init;
        ucp_request_cleanup_callback_t cleanup;
    } parts[OPAL_COMMON_UCX_CONTEXT_MAX_USERS];
} opal_common_ucx_context_t;

/**
 * One UCX user's declaration, and its handle on a context.
 *
 * The storage belongs to the caller and is normally a static member of the
 * component structure.  Everything in it other than \c attr is maintained
 * by the functions below.
 */
typedef struct opal_common_ucx_context_user_t {
    /** What this user needs.  May be amended freely between declare() and
     *  the first get() anywhere in the process; afterwards an amendment
     *  only affects whether this user is told it fits. */
    opal_common_ucx_context_attr_t attr;

    /** The context this user holds, between get() and put(). */
    opal_common_ucx_context_t *context;

    /** Offset of this user's private area inside a UCP request of the held
     *  context, or OPAL_COMMON_UCX_CONTEXT_NO_REQUEST if it asked for
     *  none. */
    size_t request_offset;

    bool declared;
} opal_common_ucx_context_user_t;

/**
 * Register this component's \c share_context MCA parameter, defaulting it
 * to \c true, so that every UCX component spells the knob the same way and
 * describes it the same way.  Call it from the component's register(), with
 * the storage that later feeds
 * opal_common_ucx_context_attr_t::share_context.
 *
 * Sharing is deliberately a per-component decision: the components differ
 * in what they ask of UCX, so the one whose demands are unwelcome in a
 * given job is the one that should step out.
 */
OPAL_DECLSPEC int opal_common_ucx_context_var_register(const mca_base_component_t *component,
                                                       bool *share_context);

/**
 * Declare what a UCX user needs from a UCP context.
 *
 * Cheap, and in particular it does not touch UCX, so it is safe to call
 * from a component's open() whether or not the component ends up being
 * used.  \c attr is copied; amend \c user->attr afterwards to refine it.
 */
OPAL_DECLSPEC int opal_common_ucx_context_declare(opal_common_ucx_context_user_t *user,
                                                  const opal_common_ucx_context_attr_t *attr);

/**
 * Withdraw a declaration.  Call it from component close(), so that a
 * component which was opened but never used stops imposing its features on
 * the context other users get.  Must not be called while the user still
 * holds a context.
 */
OPAL_DECLSPEC void opal_common_ucx_context_undeclare(opal_common_ucx_context_user_t *user);

/**
 * Acquire a UCP context satisfying \c user->attr, creating one if this is
 * the first user to ask.
 *
 * On success \c user->context and \c user->request_offset are set and the
 * context handle is available from opal_common_ucx_context_handle().  The
 * context returned is the shared one when it satisfies this user, and a
 * private one otherwise.
 *
 * A user handle holds at most one reference, so calling this twice is a
 * no-op rather than a second reference.  Components reach the point of
 * needing a context from more than one path -- selection, the first
 * window, an explicit init -- and this spares them from tracking which
 * path got there first.
 */
OPAL_DECLSPEC int opal_common_ucx_context_get(opal_common_ucx_context_user_t *user);

/**
 * Release a context acquired with opal_common_ucx_context_get().  The last
 * user to let go of a context destroys it.  Harmless on a user that holds
 * none.
 */
OPAL_DECLSPEC void opal_common_ucx_context_put(opal_common_ucx_context_user_t *user);

/**
 * The UCP context a user is holding, or NULL if it holds none.
 */
static inline ucp_context_h
opal_common_ucx_context_handle(const opal_common_ucx_context_user_t *user)
{
    return (NULL == user->context) ? NULL : user->context->ucp_context;
}

/**
 * This user's private area inside a UCP request of the context it holds,
 * and the inverse.  On the hot path, so pointer arithmetic only: the
 * offset is fixed for the lifetime of the held context.
 */
static inline void *opal_common_ucx_context_request_priv(const opal_common_ucx_context_user_t *user,
                                                         void *request)
{
    return (void *) ((char *) request + user->request_offset);
}

static inline void *opal_common_ucx_context_request_base(const opal_common_ucx_context_user_t *user,
                                                         void *priv)
{
    return (void *) ((char *) priv - user->request_offset);
}

END_C_DECLS

#endif /* _COMMON_UCX_CONTEXT_H_ */
