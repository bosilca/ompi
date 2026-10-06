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

#ifndef _COMMON_UCX_WORKER_H_
#define _COMMON_UCX_WORKER_H_

#include "opal_config.h"

#include <ucp/api/ucp.h>

#include "opal/class/opal_hash_table.h"
#include "opal/mca/common/ucx/common_ucx.h"
#include "opal/mca/pmix/pmix-internal.h"
#include "opal/mca/threads/mutex.h"

BEGIN_C_DECLS

/*
 * One UCP worker per process, and the endpoints opened on it.
 *
 * Endpoints belong to a worker, so two UCX users driving two workers hold two
 * endpoints to every peer -- two sets of queue pairs, two sets of per-endpoint
 * buffers.  That is the real cost of not sharing: on a job of any size the
 * endpoints dominate, and a process with both the ucx PML and the ucx
 * one-sided component in use has exactly twice as many as it needs.
 *
 * A worker is cheap next to the UCP context it comes from, so sharing one is
 * worth doing for the endpoints alone.
 *
 * Thread mode cannot be renegotiated once a worker exists, so each user says
 * what it needs when it asks, and a user whose requirement the existing
 * shared worker does not meet quietly gets a private one instead of a worker
 * that is unsafe for it.  In practice every user derives its requirement from
 * the same job-wide thread level and they all ask for the same thing, so this
 * is a safety net rather than a normal path.
 *
 * Endpoints are refcounted, because two users that both connect to the same
 * peer get the same handle back and either may finish with it first.
 *
 * Not every worker can be the shared one.  A user that wants a worker to
 * itself -- a thread that would rather not contend for the shared one, or an
 * OpenSHMEM context, which is a user-visible object carrying its own thread
 * mode -- asks opal_common_ucx_worker_acquire() instead, and gets a worker
 * that nobody else will be handed.  Those are client-side only: they publish
 * no address, and peers reach this process through the shared worker however
 * many private ones it is driving.
 *
 * TODO: the ucx PML drives exactly one worker, because MPI requires messages
 * between a pair of processes to be matched in the order they were sent and
 * two workers means two endpoints with no ordering between them.  That is
 * only true within one matching scope, though: distinct communicators match
 * independently, and a communicator that asserts mpi_assert_allow_overtaking
 * gives up ordering outright.  Either would let the PML spread its traffic
 * over several workers and endpoints, which is what this machinery is here
 * to make possible.
 */

/* Enough slots for the UCX users that exist (pml/ucx, osc/ucx, spml/ucx)
 * plus room to grow. */
#define OPAL_COMMON_UCX_WORKER_MAX_USERS 8

struct opal_common_ucx_worker_t {
    ucp_worker_h ucp_worker;

    /** Context this worker came from.  A user holding a different context
     *  cannot share this worker, since a worker belongs to its context. */
    ucp_context_h ucp_context;

    /** Number of users currently holding this worker. */
    int refcnt;

    /** True for the worker that users share, false for one created
     *  privately for a single user. */
    bool shared;

    /** Thread mode UCX actually granted, which is not always the one that
     *  was asked for: a UCX built without multithreading support hands back
     *  a weaker worker.  A user that cannot work with what it got says so
     *  and steps aside -- which is how pml/ucx declines a
     *  MPI_THREAD_MULTIPLE job and lets ob1 have it. */
    ucs_thread_mode_t thread_mode;

    /** True once this worker's address has gone into the modex.  False
     *  means peers cannot look us up and a caller that wants an endpoint
     *  has to supply the peer's address itself -- which is the case for a
     *  worker created after the modex closed, e.g. by the first
     *  MPI_Win_create of a job running some other PML. */
    bool published;

    /** proc name (as a uint64 key) -> opal_common_ucx_worker_ep_t. */
    opal_hash_table_t endpoints;

    opal_mutex_t mutex;

    /** Link for the idle pool a released private worker waits in.  Unused
     *  while the worker is held. */
    struct opal_common_ucx_worker_t *next;
};
typedef struct opal_common_ucx_worker_t opal_common_ucx_worker_t;

/**
 * One UCX user's handle on a worker.
 *
 * The storage belongs to the caller and is normally a static member of the
 * component structure.
 */
typedef struct opal_common_ucx_worker_user_t {
    /** Name of this user, for verbose output, e.g. "pml/ucx". */
    const char *name;

    /** Whether this user takes part in sharing at all.  Normally comes
     *  straight from this component's share_worker MCA parameter. */
    bool share_worker;

    /** The worker this user holds, between get() and put(). */
    opal_common_ucx_worker_t *worker;
} opal_common_ucx_worker_user_t;

/**
 * Register this component's \c share_worker MCA parameter, defaulting it to
 * \c true, so that every UCX component spells the knob the same way.  Call
 * it from the component's register().
 */
OPAL_DECLSPEC int opal_common_ucx_worker_var_register(const mca_base_component_t *component,
                                                      bool *share_worker);

/**
 * Acquire a worker on \c ucp_context, creating one if this is the first user
 * to ask for it.
 *
 * \c thread_mode is the weakest mode this user can work with.  The shared
 * worker is reused only if it is at least that strong; otherwise this user
 * gets a private worker, since UCX cannot strengthen a worker after the
 * fact.  Note that UCX may grant less than was asked for, so check
 * opal_common_ucx_worker_thread_mode() if the requirement is a hard one.
 *
 * \c worker_flags are UCP_WORKER_FLAG_* bits, applied only if this call is
 * the one that creates the worker; a later user's flags are ignored, which
 * is harmless for the one flag in play
 * (UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK) because it only suppresses a
 * warning.
 *
 * A user handle holds at most one reference, so calling this twice is a
 * no-op rather than a second reference.
 */
OPAL_DECLSPEC int opal_common_ucx_worker_get(opal_common_ucx_worker_user_t *user,
                                             ucp_context_h ucp_context,
                                             ucs_thread_mode_t thread_mode,
                                             uint64_t worker_flags);

/**
 * Release a worker acquired with opal_common_ucx_worker_get().  The last
 * user to let go destroys it, along with any endpoints still open on it.
 * Harmless on a user that holds none.
 */
OPAL_DECLSPEC void opal_common_ucx_worker_put(opal_common_ucx_worker_user_t *user);

/**
 * Take a worker nobody else will be given, at least \c thread_mode strong.
 *
 * Unlike opal_common_ucx_worker_get() this never hands back the shared
 * worker; the caller wanted one to itself and gets one.  Creating a UCP
 * worker is not free, so a worker released earlier is reused when it came
 * from the same context and is strong enough, which matters for a caller
 * that makes and destroys them repeatedly -- OpenSHMEM contexts, say.
 *
 * The caller drives it: this layer registers no progress callback for a
 * private worker, and publishes no address for it.
 */
OPAL_DECLSPEC int opal_common_ucx_worker_acquire(ucp_context_h ucp_context,
                                                 ucs_thread_mode_t thread_mode,
                                                 opal_common_ucx_worker_t **worker_ptr);

/**
 * Give back a worker from opal_common_ucx_worker_acquire().  It goes into
 * the idle pool rather than being destroyed, so that the next caller wanting
 * one like it does not pay to create it again.
 */
OPAL_DECLSPEC void opal_common_ucx_worker_release(opal_common_ucx_worker_t *worker);

/**
 * Destroy the idle workers belonging to \c ucp_context, or all of them when
 * it is NULL.
 *
 * A worker holds its context alive, so this has to happen before the context
 * is let go -- there is nothing else that will come along and clear the pool
 * out.
 */
OPAL_DECLSPEC void opal_common_ucx_worker_drain_idle(ucp_context_h ucp_context);

/**
 * Whether a thread that would otherwise share should take a private worker.
 *
 * The thread_workers MCA parameter, which is a real trade rather than a
 * tuning detail: a private worker per thread takes the contention off the
 * shared one, and costs an endpoint per peer per thread plus a progress call
 * per worker per cycle.  Only meaningful in a MPI_THREAD_MULTIPLE job, since
 * below that there are no concurrent callers to separate.
 */
OPAL_DECLSPEC bool opal_common_ucx_worker_per_thread(void);

/**
 * Fetch a peer's published worker address out of the modex.
 *
 * For callers that drive a worker of their own -- the wpool's per-thread
 * workers, which are client-side only and never publish an address -- and so
 * need the bytes rather than an endpoint on the shared worker.  Sets
 * \c *address to memory the caller must free().  Returns
 * OPAL_ERR_NOT_FOUND if the peer never published.
 *
 * \c user selects which of the peer's workers to ask about: the shared one,
 * or the private one belonging to the same user, matching how this process
 * publishes its own.
 */
OPAL_DECLSPEC int opal_common_ucx_worker_lookup_addr(const opal_common_ucx_worker_user_t *user,
                                                      const opal_process_name_t *peer,
                                                      ucp_address_t **address, size_t *addrlen);

/**
 * Publish this process's worker address, so that peers can connect to it
 * without being told the address by some other means.
 *
 * Goes into the modex, so it only has an effect before the modex closes --
 * PMIx_Commit() during MPI_Init.  Calling it later is not detectable from
 * here and will appear to succeed while reaching nobody, so it is the
 * caller's business to call it in time or not at all; a user that cannot
 * must arrange its own exchange, and
 * opal_common_ucx_worker_is_published() tells it which case it is in.
 *
 * Idempotent: the first user to call it publishes, and the rest are satisfied
 * by what it published, which is what lets several users share one address
 * rather than each exchanging its own.
 */
OPAL_DECLSPEC int opal_common_ucx_worker_publish(opal_common_ucx_worker_user_t *user);

/**
 * An endpoint to \c peer on this user's worker, opened if it does not exist
 * yet.
 *
 * Slow path only: both the PML and the one-sided component keep their own
 * array-indexed cache of the endpoints they use, and come here only on a
 * miss.  Nothing in here belongs on a send or an RMA fast path.
 *
 * The peer's address comes from the modex, so this fails if nobody published
 * before it closed; opal_common_ucx_worker_connect_addr() is the way in when
 * the caller has the address from somewhere else.
 */
OPAL_DECLSPEC int opal_common_ucx_worker_connect(opal_common_ucx_worker_user_t *user,
                                                 const opal_process_name_t *peer, ucp_ep_h *ep);

/**
 * As opal_common_ucx_worker_connect(), with the peer's address supplied by
 * the caller rather than looked up in the modex.  For a worker created too
 * late to use the modex, whose users exchange addresses themselves.
 */
OPAL_DECLSPEC int opal_common_ucx_worker_connect_addr(opal_common_ucx_worker_user_t *user,
                                                      const opal_process_name_t *peer,
                                                      ucp_address_t *address, ucp_ep_h *ep);

/**
 * Drop a reference on the endpoint to \c peer.  The endpoint is closed once
 * no user holds it, so a user tearing down its own connections cannot pull
 * an endpoint out from under another one.
 *
 * \c ep_to_close is set to the handle when this was the last reference and
 * the caller is therefore the one that has to close it -- closing is a
 * staged, fenced operation the caller drives (see
 * opal_common_ucx_del_procs()), not something to do while holding a lock.
 * It is set to NULL when another user still holds the endpoint.
 */
OPAL_DECLSPEC int opal_common_ucx_worker_disconnect(opal_common_ucx_worker_user_t *user,
                                                    const opal_process_name_t *peer,
                                                    ucp_ep_h *ep_to_close);

/**
 * The worker a user is holding, or NULL if it holds none.
 */
static inline ucp_worker_h
opal_common_ucx_worker_handle(const opal_common_ucx_worker_user_t *user)
{
    return (NULL == user->worker) ? NULL : user->worker->ucp_worker;
}

/**
 * Whether this user's worker is the shared one.
 *
 * A user holding the shared worker must not register a progress callback of
 * its own: this layer registers one for it, so that the worker is progressed
 * once per cycle however many users it has.  A user holding a private worker
 * is on its own as before.
 */
static inline bool opal_common_ucx_worker_is_shared(const opal_common_ucx_worker_user_t *user)
{
    return (NULL != user->worker) && user->worker->shared;
}

/**
 * Thread mode UCX granted this user's worker.  UCS_THREAD_MODE_SINGLE when
 * the user holds no worker.
 */
static inline ucs_thread_mode_t
opal_common_ucx_worker_thread_mode(const opal_common_ucx_worker_user_t *user)
{
    return (NULL == user->worker) ? UCS_THREAD_MODE_SINGLE : user->worker->thread_mode;
}

/**
 * Whether this user's worker address went into the modex, and so whether
 * opal_common_ucx_worker_connect() can be expected to work.
 *
 * False says this process's worker came up too late to be found there, which
 * for a symmetric job means the peers' did too.
 */
static inline bool
opal_common_ucx_worker_is_published(const opal_common_ucx_worker_user_t *user)
{
    return (NULL != user->worker) && user->worker->published;
}

END_C_DECLS

#endif /* _COMMON_UCX_WORKER_H_ */
