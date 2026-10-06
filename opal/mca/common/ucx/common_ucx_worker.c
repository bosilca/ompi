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

#include "opal_config.h"

#include <stdio.h>
#include <stdlib.h>

#include "common_ucx.h"
#include "common_ucx_worker.h"

#include "opal/mca/threads/mutex.h"
#include "opal/mca/threads/thread_usage.h"
#include "opal/runtime/opal_progress.h"
#include "opal/util/error.h"
#include "opal/util/proc.h"

/* The key the shared worker's address goes out under.  One key, because the
 * point of sharing is that there is one address: a peer looking us up gets
 * the worker all of our UCX users are on.
 *
 * A user that opted out of sharing has a worker of its own, which still has
 * to be reachable, so its address goes out under this key suffixed with the
 * user's name.  Peers find it because they look under the key matching their
 * own worker, and the parameter that decides this is set job-wide. */
#define OPAL_COMMON_UCX_WORKER_MODEX_KEY "opal.common.ucx.worker"

/* Long enough for the key above plus any user name we give ourselves. */
#define OPAL_COMMON_UCX_WORKER_KEY_MAX 64

/* One endpoint of the registry.  Refcounted because two users that connect
 * to the same peer get the same handle, and either may finish first. */
typedef struct {
    ucp_ep_h ep;
    int refcnt;
} opal_common_ucx_worker_ep_t;

static opal_common_ucx_worker_t *opal_common_ucx_worker_shared = NULL;

/* Private workers that have been released and not yet reclaimed, newest
 * first.  Kept rather than destroyed because creating a UCP worker is not
 * free and a caller that makes and destroys them repeatedly would pay for it
 * every time. */
static opal_common_ucx_worker_t *opal_common_ucx_worker_idle = NULL;

static opal_mutex_t opal_common_ucx_worker_mutex = OPAL_MUTEX_STATIC_INIT;

/*
 * A process name as a hash key.  Both halves are 32 bits wide, so this is
 * lossless and two distinct names cannot collide.
 */
static inline uint64_t opal_common_ucx_worker_ep_key(const opal_process_name_t *name)
{
    return ((uint64_t) name->jobid << 32) | (uint64_t) name->vpid;
}

/*
 * The modex key this user's worker address belongs under.
 *
 * Writes into the caller's buffer and returns it, so that the common case
 * costs no allocation.
 */
static const char *opal_common_ucx_worker_modex_key(const opal_common_ucx_worker_user_t *user,
                                                    char *buf, size_t buflen)
{
    if (opal_common_ucx_worker_is_shared(user)) {
        return OPAL_COMMON_UCX_WORKER_MODEX_KEY;
    }

    snprintf(buf, buflen, "%s.%s", OPAL_COMMON_UCX_WORKER_MODEX_KEY,
             (NULL == user->name) ? "private" : user->name);
    return buf;
}

/*
 * Progress the shared worker.
 *
 * Registered once, by whoever creates the shared worker, so that a process
 * with several UCX users calls ucp_worker_progress() on it once per progress
 * cycle instead of once per user.  Users that hold a private worker still
 * progress it themselves.
 */
static int opal_common_ucx_worker_progress(void)
{
    opal_common_ucx_worker_t *worker = opal_common_ucx_worker_shared;

    /* Unregistration and the last put() are not atomic with respect to each
     * other, so this can be reached just after the worker went away. */
    if (OPAL_UNLIKELY(NULL == worker)) {
        return 0;
    }

    return ucp_worker_progress(worker->ucp_worker);
}

/*
 * Create a worker on a context.  The lock is held by the caller.
 */
static opal_common_ucx_worker_t *opal_common_ucx_worker_create(ucp_context_h ucp_context,
                                                               ucs_thread_mode_t thread_mode,
                                                               uint64_t worker_flags, bool shared)
{
    opal_common_ucx_worker_t *worker;
    ucp_worker_params_t params;
    ucp_worker_attr_t attr;
    ucs_status_t status;
    int rc;

    worker = calloc(1, sizeof(*worker));
    if (NULL == worker) {
        return NULL;
    }

    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    params.thread_mode = thread_mode;

    if (0 != worker_flags) {
        params.field_mask |= UCP_WORKER_PARAM_FIELD_FLAGS;
        params.flags = worker_flags;
    }

    status = ucp_worker_create(ucp_context, &params, &worker->ucp_worker);
    if (UCS_OK != status) {
        MCA_COMMON_UCX_ERROR("ucp_worker_create failed: %s", ucs_status_string(status));
        free(worker);
        return NULL;
    }

    /* Report what UCX actually granted rather than what we asked for: a UCX
     * built without multithreading support hands back a weaker worker, and
     * a caller that needs MULTI has to be able to see that and step aside. */
    attr.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE;
    status = ucp_worker_query(worker->ucp_worker, &attr);
    if (UCS_OK != status) {
        MCA_COMMON_UCX_ERROR("ucp_worker_query failed: %s", ucs_status_string(status));
        ucp_worker_destroy(worker->ucp_worker);
        free(worker);
        return NULL;
    }
    worker->thread_mode = attr.thread_mode;

    OBJ_CONSTRUCT(&worker->endpoints, opal_hash_table_t);
    rc = opal_hash_table_init(&worker->endpoints, 128);
    if (OPAL_SUCCESS != rc) {
        OBJ_DESTRUCT(&worker->endpoints);
        ucp_worker_destroy(worker->ucp_worker);
        free(worker);
        return NULL;
    }

    OBJ_CONSTRUCT(&worker->mutex, opal_mutex_t);

    worker->ucp_context = ucp_context;
    worker->refcnt = 0;
    worker->shared = shared;
    worker->published = false;

    MCA_COMMON_UCX_VERBOSE(1, "created %s ucp worker %p, thread mode %d",
                           shared ? "shared" : "private", (void *) worker->ucp_worker,
                           (int) worker->thread_mode);
    return worker;
}

/*
 * Destroy a worker nobody holds any more.  The lock is held by the caller.
 */
static void opal_common_ucx_worker_destroy(opal_common_ucx_worker_t *worker)
{
    opal_common_ucx_worker_ep_t *entry;
    uint64_t key;
    void *node;

    /* Any endpoint still here is one a user never disconnected.  The worker
     * is going away underneath it either way, so close it the abrupt way
     * rather than leak it -- there is no one left to drive a fenced
     * disconnect for us. */
    for (int rc = opal_hash_table_get_first_key_uint64(&worker->endpoints, &key, (void **) &entry,
                                                       &node);
         OPAL_SUCCESS == rc;
         rc = opal_hash_table_get_next_key_uint64(&worker->endpoints, &key, (void **) &entry, node,
                                                  &node)) {
        MCA_COMMON_UCX_VERBOSE(1, "worker %p torn down with an endpoint still held",
                               (void *) worker->ucp_worker);
        ucp_ep_destroy(entry->ep);
        free(entry);
    }

    OBJ_DESTRUCT(&worker->endpoints);
    OBJ_DESTRUCT(&worker->mutex);
    ucp_worker_destroy(worker->ucp_worker);
    free(worker);
}

int opal_common_ucx_worker_var_register(const mca_base_component_t *component, bool *share_worker)
{
    *share_worker = true;

    return mca_base_component_var_register(component, "share_worker",
                                           "Share one UCP worker, and therefore one endpoint "
                                           "per peer, with the other UCX components of this "
                                           "process, instead of creating a worker of our own.  "
                                           "A component that turns this off gets its own "
                                           "worker and its own endpoints, as it had before "
                                           "sharing was possible",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0, OPAL_INFO_LVL_5,
                                           MCA_BASE_VAR_SCOPE_GROUP, share_worker);
}

int opal_common_ucx_worker_get(opal_common_ucx_worker_user_t *user, ucp_context_h ucp_context,
                               ucs_thread_mode_t thread_mode, uint64_t worker_flags)
{
    opal_common_ucx_worker_t *worker;
    bool shared;

    if (NULL == ucp_context) {
        return OPAL_ERR_BAD_PARAM;
    }

    /* One reference per user handle, so a component that reaches this from
     * more than one path does not have to track which got here first. */
    if (NULL != user->worker) {
        return OPAL_SUCCESS;
    }

    OPAL_THREAD_LOCK(&opal_common_ucx_worker_mutex);

    /* A worker belongs to its context, so a user holding a different one --
     * because it opted out of context sharing, or wanted features nobody
     * else asked for -- cannot share this worker either.  Nor can one that
     * needs more concurrency than the shared worker was built for: the modes
     * are ordered by strength and UCX will not strengthen one after the
     * fact. */
    if (user->share_worker && (NULL != opal_common_ucx_worker_shared)
        && (ucp_context == opal_common_ucx_worker_shared->ucp_context)
        && (opal_common_ucx_worker_shared->thread_mode >= thread_mode)) {
        worker = opal_common_ucx_worker_shared;
        worker->refcnt++;
        user->worker = worker;
        MCA_COMMON_UCX_VERBOSE(1, "%s joined the shared ucp worker %p (%d users)",
                               user->name ? user->name : "ucx user", (void *) worker->ucp_worker,
                               worker->refcnt);
        OPAL_THREAD_UNLOCK(&opal_common_ucx_worker_mutex);
        return OPAL_SUCCESS;
    }

    shared = user->share_worker && (NULL == opal_common_ucx_worker_shared);

    worker = opal_common_ucx_worker_create(ucp_context, thread_mode, worker_flags, shared);
    if (NULL == worker) {
        OPAL_THREAD_UNLOCK(&opal_common_ucx_worker_mutex);
        return OPAL_ERROR;
    }

    worker->refcnt = 1;
    if (worker->shared) {
        opal_common_ucx_worker_shared = worker;
        /* One driver for the worker all of us share, rather than one per
         * user all calling into the same worker. */
        opal_progress_register(opal_common_ucx_worker_progress);
    }
    user->worker = worker;

    OPAL_THREAD_UNLOCK(&opal_common_ucx_worker_mutex);
    return OPAL_SUCCESS;
}

void opal_common_ucx_worker_put(opal_common_ucx_worker_user_t *user)
{
    opal_common_ucx_worker_t *worker = user->worker;

    if (NULL == worker) {
        return;
    }

    OPAL_THREAD_LOCK(&opal_common_ucx_worker_mutex);

    user->worker = NULL;
    assert(worker->refcnt > 0);
    if (0 != --worker->refcnt) {
        OPAL_THREAD_UNLOCK(&opal_common_ucx_worker_mutex);
        return;
    }

    if (worker == opal_common_ucx_worker_shared) {
        /* Unregister before the worker becomes unreachable, so that no
         * progress cycle can be looking at it while it is torn down. */
        opal_progress_unregister(opal_common_ucx_worker_progress);
        opal_common_ucx_worker_shared = NULL;
    }
    opal_common_ucx_worker_destroy(worker);

    OPAL_THREAD_UNLOCK(&opal_common_ucx_worker_mutex);
}

int opal_common_ucx_worker_acquire(ucp_context_h ucp_context, ucs_thread_mode_t thread_mode,
                                   opal_common_ucx_worker_t **worker_ptr)
{
    opal_common_ucx_worker_t *worker, **prev;

    *worker_ptr = NULL;

    if (NULL == ucp_context) {
        return OPAL_ERR_BAD_PARAM;
    }

    OPAL_THREAD_LOCK(&opal_common_ucx_worker_mutex);

    /* A worker belongs to its context and its mode cannot be strengthened
     * after the fact, so an idle one is only of use if it came from the same
     * context and is at least as strong as what is being asked for. */
    for (prev = &opal_common_ucx_worker_idle; NULL != *prev; prev = &(*prev)->next) {
        worker = *prev;
        if ((ucp_context != worker->ucp_context) || (worker->thread_mode < thread_mode)) {
            continue;
        }

        *prev = worker->next;
        worker->next = NULL;
        worker->refcnt = 1;
        OPAL_THREAD_UNLOCK(&opal_common_ucx_worker_mutex);

        MCA_COMMON_UCX_VERBOSE(1, "reclaimed idle ucp worker %p", (void *) worker->ucp_worker);
        *worker_ptr = worker;
        return OPAL_SUCCESS;
    }

    worker = opal_common_ucx_worker_create(ucp_context, thread_mode, 0, false);
    if (NULL == worker) {
        OPAL_THREAD_UNLOCK(&opal_common_ucx_worker_mutex);
        return OPAL_ERROR;
    }
    worker->refcnt = 1;

    OPAL_THREAD_UNLOCK(&opal_common_ucx_worker_mutex);

    *worker_ptr = worker;
    return OPAL_SUCCESS;
}

void opal_common_ucx_worker_release(opal_common_ucx_worker_t *worker)
{
    if (NULL == worker) {
        return;
    }

    /* The shared worker is reference counted among its users and goes back
     * through put(); this one was nobody else's to begin with. */
    assert(!worker->shared);
    assert(1 == worker->refcnt);

    OPAL_THREAD_LOCK(&opal_common_ucx_worker_mutex);
    worker->refcnt = 0;
    worker->next = opal_common_ucx_worker_idle;
    opal_common_ucx_worker_idle = worker;
    OPAL_THREAD_UNLOCK(&opal_common_ucx_worker_mutex);
}

void opal_common_ucx_worker_drain_idle(ucp_context_h ucp_context)
{
    opal_common_ucx_worker_t *worker, **prev;

    OPAL_THREAD_LOCK(&opal_common_ucx_worker_mutex);

    prev = &opal_common_ucx_worker_idle;
    while (NULL != *prev) {
        worker = *prev;
        if ((NULL != ucp_context) && (ucp_context != worker->ucp_context)) {
            prev = &worker->next;
            continue;
        }
        *prev = worker->next;
        opal_common_ucx_worker_destroy(worker);
    }

    OPAL_THREAD_UNLOCK(&opal_common_ucx_worker_mutex);
}

bool opal_common_ucx_worker_per_thread(void)
{
    /* opal_using_threads() is true for MPI_THREAD_MULTIPLE and nothing
     * weaker, which is the only level at which separating threads onto
     * workers of their own buys anything. */
    return opal_common_ucx.thread_workers && opal_using_threads();
}

#if HAVE_UCP_WORKER_ADDRESS_FLAGS
/*
 * Publish one flavour of this worker's address.
 *
 * Both flavours go under the same key at different PMIx scopes, so a peer
 * gets whichever applies to it: remote peers need only the network device
 * addresses, and sending them the shared-memory transports as well would
 * put bytes in the global modex that nobody off-node can use.
 */
static int opal_common_ucx_worker_publish_type(opal_common_ucx_worker_t *worker, const char *key,
                                               int addr_flags, int modex_scope)
{
    ucp_worker_attr_t attrs;
    ucs_status_t status;
    int rc;

    attrs.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS | UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
    attrs.address_flags = addr_flags;

    status = ucp_worker_query(worker->ucp_worker, &attrs);
    if (UCS_OK != status) {
        MCA_COMMON_UCX_ERROR("failed to query UCP worker address: %s",
                             ucs_status_string(status));
        return OPAL_ERROR;
    }

    OPAL_MODEX_SEND_STRING(rc, modex_scope, key, (void *) attrs.address, attrs.address_length);

    ucp_worker_release_address(worker->ucp_worker, attrs.address);

    if (OPAL_SUCCESS != rc) {
        return OPAL_ERROR;
    }

    MCA_COMMON_UCX_VERBOSE(2, "published %s worker address, size %zu",
                           (PMIX_LOCAL == modex_scope) ? "local" : "remote",
                           attrs.address_length);
    return OPAL_SUCCESS;
}
#endif

int opal_common_ucx_worker_publish(opal_common_ucx_worker_user_t *user)
{
    opal_common_ucx_worker_t *worker = user->worker;
    char keybuf[OPAL_COMMON_UCX_WORKER_KEY_MAX];
    const char *key;
    int rc;

    if (NULL == worker) {
        return OPAL_ERR_BAD_PARAM;
    }

    key = opal_common_ucx_worker_modex_key(user, keybuf, sizeof(keybuf));

    OPAL_THREAD_LOCK(&worker->mutex);

    /* One address for the worker, however many users it has: whoever got
     * here first published it, and the rest are already reachable. */
    if (worker->published) {
        OPAL_THREAD_UNLOCK(&worker->mutex);
        return OPAL_SUCCESS;
    }

#if HAVE_UCP_WORKER_ADDRESS_FLAGS
    rc = opal_common_ucx_worker_publish_type(worker, key, UCP_WORKER_ADDRESS_FLAG_NET_ONLY,
                                             PMIX_REMOTE);
    if (OPAL_SUCCESS == rc) {
        rc = opal_common_ucx_worker_publish_type(worker, key, 0, PMIX_LOCAL);
    }
#else
    ucp_address_t *address;
    size_t addrlen;
    ucs_status_t status;

    status = ucp_worker_get_address(worker->ucp_worker, &address, &addrlen);
    if (UCS_OK != status) {
        MCA_COMMON_UCX_ERROR("failed to get worker address: %s", ucs_status_string(status));
        OPAL_THREAD_UNLOCK(&worker->mutex);
        return OPAL_ERROR;
    }

    OPAL_MODEX_SEND_STRING(rc, PMIX_GLOBAL, key, (void *) address, addrlen);
    ucp_worker_release_address(worker->ucp_worker, address);
    if (OPAL_SUCCESS == rc) {
        MCA_COMMON_UCX_VERBOSE(2, "published worker address, size %zu", addrlen);
    }
#endif

    if (OPAL_SUCCESS == rc) {
        worker->published = true;
    } else {
        MCA_COMMON_UCX_ERROR("%s", "could not distribute UCX endpoint connection details");
    }

    OPAL_THREAD_UNLOCK(&worker->mutex);
    return rc;
}

int opal_common_ucx_worker_connect_addr(opal_common_ucx_worker_user_t *user,
                                        const opal_process_name_t *peer, ucp_address_t *address,
                                        ucp_ep_h *ep)
{
    opal_common_ucx_worker_t *worker = user->worker;
    opal_common_ucx_worker_ep_t *entry;
    uint64_t key = opal_common_ucx_worker_ep_key(peer);
    ucp_ep_params_t ep_params;
    ucs_status_t status;
    int rc;

    if ((NULL == worker) || (NULL == address)) {
        return OPAL_ERR_BAD_PARAM;
    }

    OPAL_THREAD_LOCK(&worker->mutex);

    rc = opal_hash_table_get_value_uint64(&worker->endpoints, key, (void **) &entry);
    if (OPAL_SUCCESS == rc) {
        entry->refcnt++;
        *ep = entry->ep;
        /* The whole point of the registry: the endpoint to this peer already
         * exists (the ucx PML or another window opened it), so we hand the
         * same ucp_ep_h back rather than opening a second one. */
        MCA_COMMON_UCX_VERBOSE(10, "%s reused shared endpoint %p to proc %u (refcnt %d)",
                               user->name ? user->name : "ucx user", (void *) entry->ep,
                               peer->vpid, entry->refcnt);
        OPAL_THREAD_UNLOCK(&worker->mutex);
        return OPAL_SUCCESS;
    }

    entry = malloc(sizeof(*entry));
    if (NULL == entry) {
        OPAL_THREAD_UNLOCK(&worker->mutex);
        return OPAL_ERR_OUT_OF_RESOURCE;
    }

    memset(&ep_params, 0, sizeof(ep_params));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = address;

    status = ucp_ep_create(worker->ucp_worker, &ep_params, &entry->ep);
    if (UCS_OK != status) {
        MCA_COMMON_UCX_ERROR("ucp_ep_create(proc=%u) failed: %s", peer->vpid,
                             ucs_status_string(status));
        free(entry);
        OPAL_THREAD_UNLOCK(&worker->mutex);
        return OPAL_ERROR;
    }

    entry->refcnt = 1;
    rc = opal_hash_table_set_value_uint64(&worker->endpoints, key, entry);
    if (OPAL_SUCCESS != rc) {
        ucp_ep_destroy(entry->ep);
        free(entry);
        OPAL_THREAD_UNLOCK(&worker->mutex);
        return rc;
    }

    *ep = entry->ep;

    /* First reference to this peer: a new endpoint was physically opened and
     * recorded in the registry, where later users (this component or another)
     * will find and reuse it. */
    MCA_COMMON_UCX_VERBOSE(10, "%s created shared endpoint %p to proc %u (refcnt %d)",
                           user->name ? user->name : "ucx user", (void *) entry->ep,
                           peer->vpid, entry->refcnt);

    OPAL_THREAD_UNLOCK(&worker->mutex);
    return OPAL_SUCCESS;
}

int opal_common_ucx_worker_lookup_addr(const opal_common_ucx_worker_user_t *user,
                                       const opal_process_name_t *peer, ucp_address_t **address,
                                       size_t *addrlen)
{
    char keybuf[OPAL_COMMON_UCX_WORKER_KEY_MAX];
    const char *key;
    int rc;

    *address = NULL;
    *addrlen = 0;

    key = opal_common_ucx_worker_modex_key(user, keybuf, sizeof(keybuf));

    /* No gate on whether we published our own address: the question here is
     * whether the peer published theirs, and the modex is the only one that
     * can answer it.  A caller that gets OPAL_ERR_NOT_FOUND back is in a job
     * whose workers came up after the modex closed, and has to come by the
     * address some other way. */
    OPAL_MODEX_RECV_STRING(rc, key, peer, (void **) address, addrlen);
    if (OPAL_SUCCESS != rc) {
        MCA_COMMON_UCX_VERBOSE(2, "no published UCX worker address for proc %u: %s", peer->vpid,
                               opal_strerror(rc));
        return rc;
    }

    MCA_COMMON_UCX_VERBOSE(2, "got proc %u worker address, size %zu", peer->vpid, *addrlen);
    return OPAL_SUCCESS;
}

int opal_common_ucx_worker_connect(opal_common_ucx_worker_user_t *user,
                                   const opal_process_name_t *peer, ucp_ep_h *ep)
{
    ucp_address_t *address;
    size_t addrlen;
    int rc;

    rc = opal_common_ucx_worker_lookup_addr(user, peer, &address, &addrlen);
    if (OPAL_SUCCESS != rc) {
        return rc;
    }

    rc = opal_common_ucx_worker_connect_addr(user, peer, address, ep);
    free(address);
    return rc;
}

int opal_common_ucx_worker_disconnect(opal_common_ucx_worker_user_t *user,
                                      const opal_process_name_t *peer, ucp_ep_h *ep_to_close)
{
    opal_common_ucx_worker_t *worker = user->worker;
    opal_common_ucx_worker_ep_t *entry;
    uint64_t key = opal_common_ucx_worker_ep_key(peer);
    int rc;

    *ep_to_close = NULL;

    if (NULL == worker) {
        return OPAL_ERR_BAD_PARAM;
    }

    OPAL_THREAD_LOCK(&worker->mutex);

    rc = opal_hash_table_get_value_uint64(&worker->endpoints, key, (void **) &entry);
    if (OPAL_SUCCESS != rc) {
        OPAL_THREAD_UNLOCK(&worker->mutex);
        return OPAL_ERR_NOT_FOUND;
    }

    assert(entry->refcnt > 0);
    if (0 == --entry->refcnt) {
        /* Handed back rather than closed here: closing is a staged, fenced
         * operation the caller drives, and doing it under this lock would
         * serialize every other user's connects behind it. */
        *ep_to_close = entry->ep;
        MCA_COMMON_UCX_VERBOSE(10,
                               "%s released last ref on shared endpoint %p to proc %u; "
                               "handing it back to close",
                               user->name ? user->name : "ucx user", (void *) entry->ep,
                               peer->vpid);
        (void) opal_hash_table_remove_value_uint64(&worker->endpoints, key);
        free(entry);
    } else {
        /* Still shared by at least one other user, so the endpoint stays open
         * and only our reference is dropped. */
        MCA_COMMON_UCX_VERBOSE(10, "%s released shared endpoint %p to proc %u (refcnt %d)",
                               user->name ? user->name : "ucx user", (void *) entry->ep,
                               peer->vpid, entry->refcnt);
    }

    OPAL_THREAD_UNLOCK(&worker->mutex);
    return OPAL_SUCCESS;
}
