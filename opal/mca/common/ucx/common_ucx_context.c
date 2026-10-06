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

#include <assert.h>
#include <inttypes.h>
#include <string.h>

#include "common_ucx_context.h"
#include "common_ucx_wpool.h"
#include "opal/align.h"
#include "opal/mca/base/mca_base_var.h"
#include "opal/mca/threads/mutex.h"
#include "opal/util/proc.h"
#include "opal/util/string_copy.h"

/* Declarations, in declaration order.  The array is append-only while a
 * context is alive, so the request layout of a live context never has to
 * move; undeclare() leaves a hole that the next declare() may reuse. */
static opal_common_ucx_context_user_t *opal_common_ucx_context_users[OPAL_COMMON_UCX_CONTEXT_MAX_USERS];

/* Context storage.  UCX's request_init callback is handed nothing but the
 * request, so each context needs its own C function to find its layout;
 * the trampolines below index this array, which is why the contexts live in
 * fixed storage rather than being allocated. */
static opal_common_ucx_context_t opal_common_ucx_contexts[OPAL_COMMON_UCX_CONTEXT_MAX_USERS];

/* The context users share, while anyone holds it. */
static opal_common_ucx_context_t *opal_common_ucx_shared_context = NULL;

static opal_mutex_t opal_common_ucx_context_mutex = OPAL_MUTEX_STATIC_INIT;

static void opal_common_ucx_context_req_init(opal_common_ucx_context_t *context, void *request)
{
    int i;

    /* The common header sits at offset zero of every request of every
     * context handed out here, because that is where the wpool paths and
     * opal_common_ucx_req_completion() expect to find it. */
    opal_common_ucx_req_init(request);

    for (i = 0; i < context->nparts; ++i) {
        if (NULL != context->parts[i].init) {
            context->parts[i].init((char *) request + context->parts[i].offset);
        }
    }
}

static void opal_common_ucx_context_req_cleanup(opal_common_ucx_context_t *context, void *request)
{
    int i;

    for (i = 0; i < context->nparts; ++i) {
        if (NULL != context->parts[i].cleanup) {
            context->parts[i].cleanup((char *) request + context->parts[i].offset);
        }
    }
}

#define OPAL_COMMON_UCX_CONTEXT_TRAMPOLINE(_i)                                              \
    static void opal_common_ucx_context_req_init_##_i(void *request)                        \
    {                                                                                       \
        opal_common_ucx_context_req_init(&opal_common_ucx_contexts[_i], request);            \
    }                                                                                       \
    static void opal_common_ucx_context_req_cleanup_##_i(void *request)                     \
    {                                                                                       \
        opal_common_ucx_context_req_cleanup(&opal_common_ucx_contexts[_i], request);         \
    }

OPAL_COMMON_UCX_CONTEXT_TRAMPOLINE(0)
OPAL_COMMON_UCX_CONTEXT_TRAMPOLINE(1)
OPAL_COMMON_UCX_CONTEXT_TRAMPOLINE(2)
OPAL_COMMON_UCX_CONTEXT_TRAMPOLINE(3)
OPAL_COMMON_UCX_CONTEXT_TRAMPOLINE(4)
OPAL_COMMON_UCX_CONTEXT_TRAMPOLINE(5)
OPAL_COMMON_UCX_CONTEXT_TRAMPOLINE(6)
OPAL_COMMON_UCX_CONTEXT_TRAMPOLINE(7)

static const ucp_request_init_callback_t
    opal_common_ucx_context_req_init_fns[OPAL_COMMON_UCX_CONTEXT_MAX_USERS]
    = {opal_common_ucx_context_req_init_0, opal_common_ucx_context_req_init_1,
       opal_common_ucx_context_req_init_2, opal_common_ucx_context_req_init_3,
       opal_common_ucx_context_req_init_4, opal_common_ucx_context_req_init_5,
       opal_common_ucx_context_req_init_6, opal_common_ucx_context_req_init_7};

static const ucp_request_cleanup_callback_t
    opal_common_ucx_context_req_cleanup_fns[OPAL_COMMON_UCX_CONTEXT_MAX_USERS]
    = {opal_common_ucx_context_req_cleanup_0, opal_common_ucx_context_req_cleanup_1,
       opal_common_ucx_context_req_cleanup_2, opal_common_ucx_context_req_cleanup_3,
       opal_common_ucx_context_req_cleanup_4, opal_common_ucx_context_req_cleanup_5,
       opal_common_ucx_context_req_cleanup_6, opal_common_ucx_context_req_cleanup_7};

static const char *opal_common_ucx_context_prefix(const opal_common_ucx_context_user_t *user)
{
    return (NULL == user->attr.config_prefix) ? "" : user->attr.config_prefix;
}

static const char *opal_common_ucx_context_name(const opal_common_ucx_context_user_t *user)
{
    return (NULL == user->attr.name) ? "ucx" : user->attr.name;
}

/* Where this user's private request area lives in this context, or
 * OPAL_COMMON_UCX_CONTEXT_NO_REQUEST if the context has no room for it --
 * which is how a user that declared after the context was created learns
 * that it does not fit. */
static size_t opal_common_ucx_context_offset(const opal_common_ucx_context_t *context,
                                             const opal_common_ucx_context_user_t *user)
{
    int i;

    for (i = 0; i < context->nparts; ++i) {
        if (context->parts[i].user == user) {
            return context->parts[i].offset;
        }
    }

    return OPAL_COMMON_UCX_CONTEXT_NO_REQUEST;
}

static bool opal_common_ucx_context_fits(const opal_common_ucx_context_t *context,
                                         const opal_common_ucx_context_user_t *user)
{
    const opal_common_ucx_context_attr_t *attr = &user->attr;

    /* The prefix decides which UCX_* environment variables apply, so two
     * users that disagree must not be given the same context. */
    if (0 != strcmp(context->config_prefix, opal_common_ucx_context_prefix(user))) {
        return false;
    }

    if (attr->features != (context->features & attr->features)) {
        return false;
    }

    if ((0 != attr->tag_sender_mask) && (context->tag_sender_mask != attr->tag_sender_mask)) {
        return false;
    }

    /* Driving a context that was not built for concurrent workers from
     * several threads is not safe, so this one is a hard requirement
     * rather than a hint. */
    if (attr->mt_workers_shared && !context->mt_workers_shared) {
        return false;
    }

    if ((0 != attr->request_size)
        && (OPAL_COMMON_UCX_CONTEXT_NO_REQUEST == opal_common_ucx_context_offset(context, user))) {
        return false;
    }

    return true;
}

static opal_common_ucx_context_t *opal_common_ucx_context_alloc(int *slot)
{
    int i;

    for (i = 0; i < OPAL_COMMON_UCX_CONTEXT_MAX_USERS; ++i) {
        if (NULL == opal_common_ucx_contexts[i].ucp_context) {
            memset(&opal_common_ucx_contexts[i], 0, sizeof(opal_common_ucx_contexts[i]));
            *slot = i;
            return &opal_common_ucx_contexts[i];
        }
    }

    return NULL;
}

/*
 * Build a context for \c users[0..nusers-1].  Shared contexts take every
 * declared user, a private one takes only its owner.
 */
static opal_common_ucx_context_t *
opal_common_ucx_context_create(opal_common_ucx_context_user_t **users, int nusers, bool shared)
{
    opal_common_ucx_context_t *context;
    ucp_params_t params;
    ucp_config_t *config;
    ucs_status_t status;
    size_t num_eps = 0;
    size_t num_ppn = 0;
    size_t offset;
    int slot = 0;
    int i;

    context = opal_common_ucx_context_alloc(&slot);
    if (NULL == context) {
        MCA_COMMON_UCX_ERROR("no free UCP context slot (%d in use)",
                             OPAL_COMMON_UCX_CONTEXT_MAX_USERS);
        return NULL;
    }

    context->shared = shared;
    opal_string_copy(context->config_prefix, opal_common_ucx_context_prefix(users[0]),
                     sizeof(context->config_prefix));

    /* The common header is always there; private areas follow it in
     * declaration order. */
    offset = sizeof(opal_common_ucx_request_t);

    for (i = 0; i < nusers; ++i) {
        const opal_common_ucx_context_attr_t *attr = &users[i]->attr;

        context->features |= attr->features;

        if (0 != attr->tag_sender_mask) {
            if ((0 != context->tag_sender_mask)
                && (context->tag_sender_mask != attr->tag_sender_mask)) {
                /* Nobody declares a conflicting mask today.  Keep the
                 * first one and let the loser fall back to a private
                 * context through opal_common_ucx_context_fits(). */
                MCA_COMMON_UCX_VERBOSE(1,
                                       "%s wants tag_sender_mask 0x%" PRIx64 ", context already "
                                       "has 0x%" PRIx64 "; it will not share this context",
                                       opal_common_ucx_context_name(users[i]),
                                       attr->tag_sender_mask, context->tag_sender_mask);
            } else {
                context->tag_sender_mask = attr->tag_sender_mask;
            }
        }

        context->mt_workers_shared = context->mt_workers_shared || attr->mt_workers_shared;

        if ((attr->estimated_num_eps > 0) && ((size_t) attr->estimated_num_eps > num_eps)) {
            num_eps = (size_t) attr->estimated_num_eps;
        }

        if ((attr->estimated_num_ppn > 0) && ((size_t) attr->estimated_num_ppn > num_ppn)) {
            num_ppn = (size_t) attr->estimated_num_ppn;
        }

        if (0 == attr->request_size) {
            continue;
        }

        if (context->nparts == OPAL_COMMON_UCX_CONTEXT_MAX_USERS) {
            MCA_COMMON_UCX_ERROR("too many UCP request reservations");
            goto err;
        }

        offset = OPAL_ALIGN(offset, (0 == attr->request_align) ? OPAL_ALIGN_MIN
                                                               : attr->request_align,
                            size_t);
        context->parts[context->nparts].user = users[i];
        context->parts[context->nparts].offset = offset;
        context->parts[context->nparts].init = attr->request_init;
        context->parts[context->nparts].cleanup = attr->request_cleanup;
        ++context->nparts;
        offset += attr->request_size;
    }

    context->request_size = offset;

    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE
                        | UCP_PARAM_FIELD_REQUEST_INIT | UCP_PARAM_FIELD_REQUEST_CLEANUP
                        | UCP_PARAM_FIELD_MT_WORKERS_SHARED;
    params.features = context->features;
    params.request_size = context->request_size;
    params.request_init = opal_common_ucx_context_req_init_fns[slot];
    params.request_cleanup = opal_common_ucx_context_req_cleanup_fns[slot];
    params.mt_workers_shared = context->mt_workers_shared ? 1 : 0;

    if (0 != context->tag_sender_mask) {
        params.field_mask |= UCP_PARAM_FIELD_TAG_SENDER_MASK;
        params.tag_sender_mask = context->tag_sender_mask;
    }

    if (0 != num_eps) {
        params.field_mask |= UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
        params.estimated_num_eps = num_eps;
    }

#if HAVE_DECL_UCP_PARAM_FIELD_ESTIMATED_NUM_PPN
    if (num_ppn < (size_t) opal_process_info.num_local_peers + 1) {
        num_ppn = (size_t) opal_process_info.num_local_peers + 1;
    }
    params.estimated_num_ppn = num_ppn;
    params.field_mask |= UCP_PARAM_FIELD_ESTIMATED_NUM_PPN;
#else
    (void) num_ppn;
#endif

#if HAVE_DECL_UCP_PARAM_FIELD_NODE_LOCAL_ID
    params.node_local_id = opal_process_info.my_local_rank;
    params.field_mask |= UCP_PARAM_FIELD_NODE_LOCAL_ID;
#endif

    status = ucp_config_read(('\0' == context->config_prefix[0]) ? NULL : context->config_prefix,
                             NULL, &config);
    if (UCS_OK != status) {
        MCA_COMMON_UCX_ERROR("ucp_config_read failed: %s", ucs_status_string(status));
        goto err;
    }

    status = ucp_init(&params, config, &context->ucp_context);
    ucp_config_release(config);
    if (UCS_OK != status) {
        MCA_COMMON_UCX_ERROR("ucp_init failed: %s", ucs_status_string(status));
        context->ucp_context = NULL;
        goto err;
    }

    MCA_COMMON_UCX_VERBOSE(1,
                           "created %s UCP context: features 0x%" PRIx64 ", request size %zu, "
                           "mt_workers_shared %d, config prefix \"%s\"",
                           shared ? "shared" : "private", context->features,
                           context->request_size, (int) context->mt_workers_shared,
                           context->config_prefix);

    return context;

err:
    /* ucp_context stays NULL, which is what marks the slot free again. */
    memset(context, 0, sizeof(*context));
    return NULL;
}

int opal_common_ucx_context_var_register(const mca_base_component_t *component,
                                         bool *share_context)
{
    *share_context = true;

    return mca_base_component_var_register(component, "share_context",
                                           "Share one UCP context with the other UCX "
                                           "components of this process, when their "
                                           "requirements are compatible, instead of "
                                           "initializing a context of our own.  A component "
                                           "that turns this off neither uses the shared "
                                           "context nor contributes its requirements to it",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0, OPAL_INFO_LVL_5,
                                           MCA_BASE_VAR_SCOPE_GROUP, share_context);
}

int opal_common_ucx_context_declare(opal_common_ucx_context_user_t *user,
                                    const opal_common_ucx_context_attr_t *attr)
{
    int rc = OPAL_ERR_OUT_OF_RESOURCE;
    int i;

    OPAL_THREAD_LOCK(&opal_common_ucx_context_mutex);

    if (user->declared) {
        rc = OPAL_SUCCESS;
        goto out;
    }

    for (i = 0; i < OPAL_COMMON_UCX_CONTEXT_MAX_USERS; ++i) {
        if (NULL == opal_common_ucx_context_users[i]) {
            user->attr = *attr;
            user->context = NULL;
            user->request_offset = OPAL_COMMON_UCX_CONTEXT_NO_REQUEST;
            user->declared = true;
            opal_common_ucx_context_users[i] = user;
            rc = OPAL_SUCCESS;
            goto out;
        }
    }

    MCA_COMMON_UCX_ERROR("too many UCX users (%d) to declare %s",
                         OPAL_COMMON_UCX_CONTEXT_MAX_USERS,
                         (NULL == attr->name) ? "ucx" : attr->name);

out:
    OPAL_THREAD_UNLOCK(&opal_common_ucx_context_mutex);
    return rc;
}

void opal_common_ucx_context_undeclare(opal_common_ucx_context_user_t *user)
{
    int i;

    OPAL_THREAD_LOCK(&opal_common_ucx_context_mutex);

    if (!user->declared) {
        OPAL_THREAD_UNLOCK(&opal_common_ucx_context_mutex);
        return;
    }

    /* Undeclaring while still holding a context would leave the context's
     * request layout pointing at a user that no longer exists. */
    assert(NULL == user->context);

    for (i = 0; i < OPAL_COMMON_UCX_CONTEXT_MAX_USERS; ++i) {
        if (user == opal_common_ucx_context_users[i]) {
            opal_common_ucx_context_users[i] = NULL;
            break;
        }
    }

    /* A context built while this user was declared reserved space for it
     * and recorded its callbacks.  The space has to stay where it is --
     * requests are in flight through it -- but the callbacks must go: this
     * is usually a component being closed, and calling into it after its
     * DSO is unloaded would be fatal.  Dropping the user also means that if
     * it ever declares again it will be told it does not fit, and get a
     * context of its own rather than a stale offset. */
    for (i = 0; i < OPAL_COMMON_UCX_CONTEXT_MAX_USERS; ++i) {
        opal_common_ucx_context_t *context = &opal_common_ucx_contexts[i];
        int part;

        for (part = 0; part < context->nparts; ++part) {
            if (user == context->parts[part].user) {
                context->parts[part].user = NULL;
                context->parts[part].init = NULL;
                context->parts[part].cleanup = NULL;
            }
        }
    }

    user->declared = false;

    OPAL_THREAD_UNLOCK(&opal_common_ucx_context_mutex);
}

int opal_common_ucx_context_get(opal_common_ucx_context_user_t *user)
{
    opal_common_ucx_context_user_t *participants[OPAL_COMMON_UCX_CONTEXT_MAX_USERS];
    opal_common_ucx_context_t *context = NULL;
    int nparticipants = 0;
    int rc = OPAL_SUCCESS;
    int i;

    OPAL_THREAD_LOCK(&opal_common_ucx_context_mutex);

    if (!user->declared) {
        MCA_COMMON_UCX_ERROR("%s asked for a UCP context without declaring one",
                             opal_common_ucx_context_name(user));
        rc = OPAL_ERR_NOT_INITIALIZED;
        goto out;
    }

    /* A user handle holds at most one reference, so a repeated get() is a
     * no-op and an unmatched put() is harmless.  Components release the
     * context from more than one error path, and this keeps them from
     * having to track whether they already did. */
    if (NULL != user->context) {
        goto out;
    }

    if (user->attr.share_context) {
        bool created = false;

        if (NULL == opal_common_ucx_shared_context) {
            /* Build it from every user that declared so far and is willing
             * to share.  Users that declare later may still share it if
             * they fit.  A user that opted out is left out of the union
             * too: turning sharing off for one component must not leave
             * its features imposed on everybody else's context. */
            for (i = 0; i < OPAL_COMMON_UCX_CONTEXT_MAX_USERS; ++i) {
                if ((NULL != opal_common_ucx_context_users[i])
                    && opal_common_ucx_context_users[i]->attr.share_context) {
                    participants[nparticipants++] = opal_common_ucx_context_users[i];
                }
            }
            opal_common_ucx_shared_context
                = opal_common_ucx_context_create(participants, nparticipants, true);
            created = (NULL != opal_common_ucx_shared_context);
        }

        if (NULL != opal_common_ucx_shared_context) {
            if (opal_common_ucx_context_fits(opal_common_ucx_shared_context, user)) {
                context = opal_common_ucx_shared_context;
            } else {
                MCA_COMMON_UCX_VERBOSE(1,
                                       "%s cannot use the shared UCP context, creating a "
                                       "private one",
                                       opal_common_ucx_context_name(user));
                if (created) {
                    /* Built for this user and still rejected by it -- a
                     * conflicting tag_sender_mask does that.  Nobody holds
                     * it, so take it back down rather than leave a context
                     * around that no put() will ever reach. */
                    ucp_cleanup(opal_common_ucx_shared_context->ucp_context);
                    memset(opal_common_ucx_shared_context, 0,
                           sizeof(*opal_common_ucx_shared_context));
                    opal_common_ucx_shared_context = NULL;
                }
            }
        }
    }

    if (NULL == context) {
        context = opal_common_ucx_context_create(&user, 1, false);
        if (NULL == context) {
            rc = OPAL_ERROR;
            goto out;
        }
    }

    ++context->refcnt;
    user->context = context;
    user->request_offset = opal_common_ucx_context_offset(context, user);

out:
    OPAL_THREAD_UNLOCK(&opal_common_ucx_context_mutex);
    return rc;
}

void opal_common_ucx_context_put(opal_common_ucx_context_user_t *user)
{
    opal_common_ucx_context_t *context;

    OPAL_THREAD_LOCK(&opal_common_ucx_context_mutex);

    context = user->context;
    if (NULL == context) {
        OPAL_THREAD_UNLOCK(&opal_common_ucx_context_mutex);
        return;
    }

    user->context = NULL;
    user->request_offset = OPAL_COMMON_UCX_CONTEXT_NO_REQUEST;

    if (0 != --context->refcnt) {
        OPAL_THREAD_UNLOCK(&opal_common_ucx_context_mutex);
        return;
    }

    MCA_COMMON_UCX_VERBOSE(1, "destroying the %s UCP context, last user was %s",
                           context->shared ? "shared" : "private",
                           opal_common_ucx_context_name(user));

    ucp_cleanup(context->ucp_context);

    if (context == opal_common_ucx_shared_context) {
        opal_common_ucx_shared_context = NULL;
    }

    /* Frees the slot, and with it the trampoline that pointed here. */
    memset(context, 0, sizeof(*context));

    OPAL_THREAD_UNLOCK(&opal_common_ucx_context_mutex);
}
