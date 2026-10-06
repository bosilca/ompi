/* SPDX-License-Identifier: BSD-3-Clause-Open-MPI */
#include "opal_config.h"

#include "common_ucx.h"
#include "common_ucx_wpool.h"
#include "common_ucx_wpool_int.h"
#include "opal/mca/base/mca_base_framework.h"
#include "opal/mca/base/mca_base_var.h"
#include "opal/mca/pmix/pmix-internal.h"
#include "opal/memoryhooks/memory.h"
#include "opal/util/proc.h"
#include "opal/util/sys_limits.h"
#include "opal/util/sys_limits.h"
#include <ucm/api/ucm.h>

/*******************************************************************************
 *******************************************************************************
 *
 * Worker Pool (wpool) framework
 * Used to manage multi-threaded implementation of UCX for ompi/OSC & OSHMEM
 *
 *******************************************************************************
 ******************************************************************************/

OBJ_CLASS_INSTANCE(opal_common_ucx_winfo_t, opal_list_item_t, NULL, _winfo_destructor);
OBJ_CLASS_INSTANCE(_ctx_record_t, opal_list_item_t, NULL, NULL);
OBJ_CLASS_INSTANCE(_mem_record_t, opal_list_item_t, NULL, NULL);

// TODO: Remove once debug is completed
#ifdef OPAL_COMMON_UCX_WPOOL_DBG
__thread FILE *tls_pf = NULL;
__thread int initialized = 0;
#endif

bool opal_common_ucx_single_threaded = true;
opal_atomic_int64_t opal_common_ucx_ep_counts = 0;
opal_atomic_int64_t opal_common_ucx_unpacked_rkey_counts = 0;

static _ctx_record_t *_tlocal_add_ctx_rec(opal_common_ucx_ctx_t *ctx);
static inline _ctx_record_t *_tlocal_get_ctx_rec(opal_tsd_tracked_key_t tls_key);
static void _tlocal_ctx_rec_cleanup(_ctx_record_t *ctx_rec);
static void _tlocal_mem_rec_cleanup(_mem_record_t *mem_rec);
static void _ctx_rec_destructor(void *arg);
static void _mem_rec_destructor(void *arg);

/* -----------------------------------------------------------------------------
 * Worker information (winfo) management functionality
 *----------------------------------------------------------------------------*/
/*
 * Create a worker info.
 *
 * is_dflt says whether this is the pool's default worker -- the one whose
 * address goes out in the modex -- rather than one handed to a thread.  The
 * caller knows which it is asking for, so it is told rather than inferred
 * from the state of the pool.
 *
 * The default winfo runs on the process-wide shared worker, which the pool
 * already holds a reference on; so does every winfo in a job that is not
 * threaded, since there is then no reason for a thread to have one of its
 * own.  Only a thread in a threaded job gets a private worker, and only if
 * the thread_workers parameter says it should -- the shared worker is built
 * for concurrency in such a job, so this is a trade between contention on
 * one worker and an endpoint per peer on each of several.  A private worker
 * is purely client-side either way: it publishes no address, and peers reach
 * this process through the shared worker regardless.
 */
static opal_common_ucx_winfo_t *_winfo_create(opal_common_ucx_wpool_t *wpool,
                                              bool is_dflt)
{
    bool shared_worker = is_dflt || !opal_common_ucx_worker_per_thread();
    opal_common_ucx_worker_t *private_worker = NULL;
    opal_common_ucx_winfo_t *winfo = NULL;
    ucs_thread_mode_t thread_mode;
    ucp_worker_h worker;
    int rc;

    if (shared_worker) {
        worker = opal_common_ucx_worker_handle(&wpool->worker_user);
        assert(NULL != worker);
    } else {
        /* One thread drives this one, so it needs no more than serialized
         * access, whatever the job's thread level. */
        thread_mode = opal_common_ucx_single_threaded ? UCS_THREAD_MODE_SINGLE
                                                      : UCS_THREAD_MODE_SERIALIZED;
        rc = opal_common_ucx_worker_acquire(wpool->ucp_ctx, thread_mode, &private_worker);
        if (OPAL_SUCCESS != rc) {
            goto exit;
        }
        worker = private_worker->ucp_worker;
    }

    winfo = OBJ_NEW(opal_common_ucx_winfo_t);
    if (NULL == winfo) {
        MCA_COMMON_UCX_ERROR("Cannot allocate memory for worker info");
        goto release_worker;
    }

    OBJ_CONSTRUCT(&winfo->mutex, opal_recursive_mutex_t);
    winfo->worker = worker;
    winfo->endpoints = NULL;
    winfo->comm_size = 0;
    winfo->inflight_ops = NULL;
    winfo->global_inflight_ops = 0;
    winfo->inflight_req = UCS_OK;
    winfo->is_dflt_winfo = is_dflt;
    winfo->shared_worker = shared_worker;
    winfo->private_worker = private_worker;

    return winfo;

release_worker:
    opal_common_ucx_worker_release(private_worker);
exit:
    return winfo;
}

static void _winfo_destructor(opal_common_ucx_winfo_t *winfo)
{
    if (winfo->inflight_req != UCS_OK) {
        opal_common_ucx_wait_request_mt(winfo->inflight_req, "opal_common_ucx_flush");
        winfo->inflight_req = UCS_OK;
    }

    assert(winfo->global_inflight_ops == 0);

    if (winfo->comm_size != 0) {
        size_t i;
        /* On the shared worker the endpoints are the context's, borrowed
         * from its registry references, and it releases them.  On a private
         * worker they are ours alone. */
        if (!winfo->shared_worker) {
            for (i = 0; i < winfo->comm_size; i++) {
                if (NULL != winfo->endpoints[i]) {
                    ucp_ep_destroy(winfo->endpoints[i]);
                    OPAL_COMMON_UCX_DEBUG_ATOMIC_ADD(opal_common_ucx_ep_counts, -1);
                }
                assert(winfo->inflight_ops[i] == 0);
            }
        }
        free(winfo->endpoints);
        free(winfo->inflight_ops);
    }
    winfo->endpoints = NULL;
    winfo->comm_size = 0;

    OBJ_DESTRUCT(&winfo->mutex);
    if (!winfo->shared_worker) {
        opal_common_ucx_worker_release(winfo->private_worker);
        winfo->private_worker = NULL;
    }

}

/* -----------------------------------------------------------------------------
 * Worker Pool management functionality
 *----------------------------------------------------------------------------*/

OPAL_DECLSPEC opal_common_ucx_wpool_t *opal_common_ucx_wpool_allocate(void)
{
    opal_common_ucx_wpool_t *ptr = calloc(1, sizeof(opal_common_ucx_wpool_t));
    ptr->refcnt = 0;

    return ptr;
}

OPAL_DECLSPEC void opal_common_ucx_wpool_free(opal_common_ucx_wpool_t *wpool)
{
    assert(wpool->refcnt == 0);
    free(wpool);
}

static int _wpool_list_put(opal_common_ucx_wpool_t *wpool, opal_list_t *list,
                           opal_common_ucx_winfo_t *winfo);

OPAL_DECLSPEC int opal_common_ucx_wpool_init(opal_common_ucx_wpool_t *wpool)
{
    opal_common_ucx_winfo_t *winfo;
    ucs_thread_mode_t thread_mode;
    int rc = OPAL_SUCCESS;

    wpool->refcnt++;

    if (1 < wpool->refcnt) {
        return rc;
    }

    OBJ_CONSTRUCT(&wpool->mutex, opal_mutex_t);

    /* create recv worker and add to idle pool */
    OBJ_CONSTRUCT(&wpool->idle_workers, opal_list_t);
    OBJ_CONSTRUCT(&wpool->active_workers, opal_list_t);

    wpool->dflt_winfo = NULL;

    /* The same derivation every other UCX user of this process makes, so
     * that we all ask for the same thing and can therefore share. */
    thread_mode = opal_common_ucx_job_thread_mode();

    /* Deliberately no opal_common_ucx_worker_publish() here.  A pool is only
     * ever built on the first use of a window, which is long after MPI_Init
     * closed the business card, so there is nothing we could put in it:
     * either the ucx PML already published this worker for us, or nobody
     * did and our users have to exchange addresses among themselves.  Which
     * of the two it is is exactly what opal_common_ucx_worker_is_published()
     * reports, and the caller tells wpctx_create() how to cope. */
    rc = opal_common_ucx_worker_get(&wpool->worker_user, wpool->ucp_ctx, thread_mode, 0);
    if (OPAL_SUCCESS != rc) {
        MCA_COMMON_UCX_ERROR("Failed to acquire a UCP worker");
        goto err_worker_get;
    }

    winfo = _winfo_create(wpool, true);
    if (NULL == winfo) {
        MCA_COMMON_UCX_ERROR("Failed to create receive worker");
        rc = OPAL_ERROR;
        goto err_worker_create;
    }
    wpool->dflt_winfo = winfo;
    OBJ_RETAIN(wpool->dflt_winfo);

    rc = _wpool_list_put(wpool, &wpool->idle_workers, winfo);
    if (rc) {
        goto err_wpool_add;
    }

    return rc;

err_wpool_add:
    OBJ_RELEASE(winfo);
    wpool->dflt_winfo = NULL;
err_worker_create:
    opal_common_ucx_worker_put(&wpool->worker_user);
err_worker_get:
    OBJ_DESTRUCT(&wpool->idle_workers);
    OBJ_DESTRUCT(&wpool->active_workers);
    if (wpool->ucp_ctx_owned) {
        ucp_cleanup(wpool->ucp_ctx);
        wpool->ucp_ctx = NULL;
    }
    return rc;
}

OPAL_DECLSPEC
void opal_common_ucx_wpool_finalize(opal_common_ucx_wpool_t *wpool)
{
    wpool->refcnt--;
    if (wpool->refcnt > 0) {
        return;
    }

    /* Go over the list, free idle list items */
    if (!opal_list_is_empty(&wpool->idle_workers)) {
        opal_common_ucx_winfo_t *winfo, *next;
        OPAL_LIST_FOREACH_SAFE (winfo, next, &wpool->idle_workers, opal_common_ucx_winfo_t) {
            opal_list_remove_item(&wpool->idle_workers, &winfo->super);
            OBJ_RELEASE(winfo);
        }
    }
    OBJ_DESTRUCT(&wpool->idle_workers);

    /* Release active workers. They are no longer active actually
     * because opal_common_ucx_wpool_finalize is being called. */
    if (!opal_list_is_empty(&wpool->active_workers)) {
        opal_common_ucx_winfo_t *winfo, *next;
        OPAL_LIST_FOREACH_SAFE (winfo, next, &wpool->active_workers, opal_common_ucx_winfo_t) {
            opal_list_remove_item(&wpool->active_workers, &winfo->super);
            OBJ_RELEASE(winfo);
        }
    }
    OBJ_DESTRUCT(&wpool->active_workers);

    OBJ_RELEASE(wpool->dflt_winfo);
    wpool->dflt_winfo = NULL;

    /* The winfos above gave their private workers back to the idle pool,
     * where they would sit holding the context alive.  Nothing else will
     * come along and clear them out. */
    opal_common_ucx_worker_drain_idle(wpool->ucp_ctx);

    /* After the winfos, which were running on it. */
    opal_common_ucx_worker_put(&wpool->worker_user);

    OBJ_DESTRUCT(&wpool->mutex);
    if (wpool->ucp_ctx_owned && (NULL != wpool->ucp_ctx)) {
        ucp_cleanup(wpool->ucp_ctx);
        wpool->ucp_ctx = NULL;
    }

    return;
}

OPAL_DECLSPEC int opal_common_ucx_wpool_progress(opal_common_ucx_wpool_t *wpool)
{
    opal_common_ucx_winfo_t *winfo = NULL, *next = NULL;
    int completed = 0, progressed = 0;

    /* Go over all active workers and progress them
     * TODO: may want to have some partitioning to progress only part of
     * workers */
    if (0 != opal_mutex_trylock(&wpool->mutex)) {
        return completed;
    }

    OPAL_LIST_FOREACH_SAFE (winfo, next, &wpool->active_workers, opal_common_ucx_winfo_t) {
        /* The shared worker has one driver for the whole process, registered
         * by common/ucx; progressing it from here as well would mean one
         * call per active winfo on the same worker.  Only the private
         * per-thread workers are ours to drive. */
        if (winfo->shared_worker) {
            continue;
        }
        if (0 != opal_mutex_trylock(&winfo->mutex)) {
            continue;
        }
        do {
            progressed = ucp_worker_progress(winfo->worker);
            completed += progressed;
        } while (progressed);
        opal_mutex_unlock(&winfo->mutex);
    }
    opal_mutex_unlock(&wpool->mutex);

    return completed;
}

static int _wpool_list_put(opal_common_ucx_wpool_t *wpool, opal_list_t *list,
                           opal_common_ucx_winfo_t *winfo)
{
    opal_list_append(list, &winfo->super);
    return OPAL_SUCCESS;
}

static opal_common_ucx_winfo_t *_wpool_list_get(opal_common_ucx_wpool_t *wpool, opal_list_t *list)
{
    opal_common_ucx_winfo_t *winfo = NULL;

    if (!opal_list_is_empty(list)) {
        winfo = (opal_common_ucx_winfo_t *) opal_list_get_first(list);
        opal_list_remove_item(list, &winfo->super);
    }

    return winfo;
}

static opal_common_ucx_winfo_t *_wpool_get_winfo(opal_common_ucx_wpool_t *wpool, size_t comm_size)
{
    opal_common_ucx_winfo_t *winfo;
    opal_mutex_lock(&wpool->mutex);
    winfo = _wpool_list_get(wpool, &wpool->idle_workers);
    if (!winfo) {
        winfo = _winfo_create(wpool, false);
        if (!winfo) {
            MCA_COMMON_UCX_ERROR("Failed to allocate worker info structure");
            opal_mutex_unlock(&wpool->mutex);
            return NULL;
        }
    }

    winfo->endpoints = calloc(comm_size, sizeof(ucp_ep_h));
    winfo->inflight_ops = calloc(comm_size, sizeof(short));
    winfo->comm_size = comm_size;

    /* Put the worker on the active list */
    _wpool_list_put(wpool, &wpool->active_workers, winfo);

    opal_mutex_unlock(&wpool->mutex);

    return winfo;
}

/* Remove the winfo from active workers and add it to idle workers */
static void _wpool_put_winfo(opal_common_ucx_wpool_t *wpool, opal_common_ucx_winfo_t *winfo)
{
    opal_mutex_lock(&wpool->mutex);
    if (winfo->comm_size != 0) {
        size_t i;
        /* See _winfo_destructor(): only a private worker's endpoints are
         * ours to close. */
        if (!winfo->shared_worker) {
            for (i = 0; i < winfo->comm_size; i++) {
                if (NULL != winfo->endpoints[i]) {
                    ucp_ep_destroy(winfo->endpoints[i]);
                    OPAL_COMMON_UCX_DEBUG_ATOMIC_ADD(opal_common_ucx_ep_counts, -1);
                }
                assert(winfo->inflight_ops[i] == 0);
            }
        }
        free(winfo->endpoints);
        free(winfo->inflight_ops);
    }
    winfo->endpoints = NULL;
    winfo->comm_size = 0;
    opal_list_remove_item(&wpool->active_workers, &winfo->super);
    opal_list_prepend(&wpool->idle_workers, &winfo->super);
    opal_mutex_unlock(&wpool->mutex);

    return;
}

/* -----------------------------------------------------------------------------
 * Worker Pool Communication context management functionality
 *----------------------------------------------------------------------------*/

OPAL_DECLSPEC int opal_common_ucx_wpctx_create(opal_common_ucx_wpool_t *wpool, int comm_size,
                                               const opal_process_name_t *proc_names,
                                               opal_common_ucx_exchange_func_t exchange_func,
                                               void *exchange_metadata,
                                               opal_common_ucx_ctx_t **ctx_ptr)
{
    opal_common_ucx_ctx_t *ctx = calloc(1, sizeof(*ctx));
    int ret = OPAL_SUCCESS;

    if (NULL == ctx) {
        (*ctx_ptr) = NULL;
        return OPAL_ERR_OUT_OF_RESOURCE;
    }

    OBJ_CONSTRUCT(&ctx->mutex, opal_recursive_mutex_t);
    OBJ_CONSTRUCT(&ctx->ctx_records, opal_list_t);

    ctx->wpool = wpool;
    ctx->comm_size = comm_size;
    ctx->num_incomplete_req_ops = 0;

    ctx->proc_names = malloc(comm_size * sizeof(*ctx->proc_names));
    if (NULL == ctx->proc_names) {
        ret = OPAL_ERR_OUT_OF_RESOURCE;
        goto error;
    }
    memcpy(ctx->proc_names, proc_names, comm_size * sizeof(*ctx->proc_names));

    /* Normally nothing to exchange: the addresses are already in the
     * business card, put there once for the whole process rather than once
     * per context. */
    ctx->recv_worker_addrs = NULL;
    ctx->recv_worker_displs = NULL;
    if (NULL != exchange_func) {
        ucp_address_t *my_addr;
        size_t my_addr_len;
        ucs_status_t status;

        status = ucp_worker_get_address(opal_common_ucx_worker_handle(&wpool->worker_user),
                                        &my_addr, &my_addr_len);
        if (UCS_OK != status) {
            MCA_COMMON_UCX_VERBOSE(1, "ucp_worker_get_address failed: %d", status);
            ret = OPAL_ERROR;
            goto error;
        }

        ret = exchange_func(my_addr, my_addr_len, &ctx->recv_worker_addrs,
                            &ctx->recv_worker_displs, exchange_metadata);
        ucp_worker_release_address(opal_common_ucx_worker_handle(&wpool->worker_user), my_addr);
        if (OPAL_SUCCESS != ret) {
            goto error;
        }
    }

    OBJ_CONSTRUCT(&ctx->tls_key, opal_tsd_tracked_key_t);
    opal_tsd_tracked_key_set_destructor(&ctx->tls_key, _ctx_rec_destructor);

    (*ctx_ptr) = ctx;
    return ret;
error:
    free(ctx->recv_worker_addrs);
    free(ctx->recv_worker_displs);
    free(ctx->proc_names);
    OBJ_DESTRUCT(&ctx->mutex);
    OBJ_DESTRUCT(&ctx->ctx_records);
    free(ctx);
    (*ctx_ptr) = NULL;
    return ret;
}

OPAL_DECLSPEC void opal_common_ucx_wpctx_release(opal_common_ucx_ctx_t *ctx)
{
    _ctx_record_t *ctx_rec = NULL, *next;

    /* Application is expected to guarantee that no operation
     * is performed on the context that is being released */

    /* destroy key so that other threads don't invoke destructors */
    OBJ_DESTRUCT(&ctx->tls_key);

    /* loop through list of records */
    OPAL_LIST_FOREACH_SAFE (ctx_rec, next, &ctx->ctx_records, _ctx_record_t) {
        _tlocal_ctx_rec_cleanup(ctx_rec);
    }

    /* The shared endpoints are not ours to release: they live in the
     * caller's process-wide slots, which outlive any one context. */
    free(ctx->recv_worker_addrs);
    free(ctx->recv_worker_displs);
    free(ctx->proc_names);

    OBJ_DESTRUCT(&ctx->mutex);
    OBJ_DESTRUCT(&ctx->ctx_records);

    free(ctx);
}

/* -----------------------------------------------------------------------------
 * Worker Pool Memory management functionality
 *----------------------------------------------------------------------------*/

OPAL_DECLSPEC
int opal_common_ucx_wpmem_create(opal_common_ucx_ctx_t *ctx, void **mem_base, size_t mem_size,
                                 opal_common_ucx_mem_type_t mem_type,
                                 opal_common_ucx_exchange_func_t exchange_func,
                                 opal_common_ucx_exchange_mode_t exchange_mode,
                                 void *exchange_metadata, char **my_mem_addr, int *my_mem_addr_size,
                                 opal_common_ucx_wpmem_t **mem_ptr)
{
    opal_common_ucx_wpmem_t *mem = calloc(1, sizeof(*mem));
    void *rkey_addr = NULL;
    size_t rkey_addr_len;
    ucs_status_t status;
    int ret = OPAL_SUCCESS;

    mem->ctx = ctx;
    mem->mem_addrs = NULL;
    mem->mem_displs = NULL;
    mem->skip_periodic_flush = false;

    OBJ_CONSTRUCT(&mem->mutex, opal_mutex_t);

    ret = _comm_ucx_wpmem_map(ctx->wpool, mem_base, mem_size, &mem->memh, mem_type);
    if (ret != OPAL_SUCCESS) {
        MCA_COMMON_UCX_VERBOSE(1, "_comm_ucx_mem_map failed: %d", ret);
        goto error_mem_map;
    }

    status = ucp_rkey_pack(ctx->wpool->ucp_ctx, mem->memh, &rkey_addr, &rkey_addr_len);
    if (status != UCS_OK) {
        MCA_COMMON_UCX_VERBOSE(1, "ucp_rkey_pack failed: %d", status);
        ret = OPAL_ERROR;
        goto error_rkey_pack;
    }

    if (exchange_mode == OPAL_COMMON_UCX_WPMEM_ADDR_EXCHANGE_FULL) {
        ret = exchange_func(rkey_addr, rkey_addr_len, &mem->mem_addrs, &mem->mem_displs,
                            exchange_metadata);
        if (ret != OPAL_SUCCESS) {
            goto error_rkey_pack;
        }
    }
    OBJ_CONSTRUCT(&mem->tls_key, opal_tsd_tracked_key_t);
    opal_tsd_tracked_key_set_destructor(&mem->tls_key, _mem_rec_destructor);

    (*mem_ptr) = mem;
    (*my_mem_addr) = rkey_addr;
    (*my_mem_addr_size) = rkey_addr_len;

    return ret;

error_rkey_pack:
    ucp_mem_unmap(ctx->wpool->ucp_ctx, mem->memh);
error_mem_map:
    free(mem);
    (*mem_ptr) = NULL;
    return ret;
}

static int _comm_ucx_wpmem_map(opal_common_ucx_wpool_t *wpool, void **base, size_t size,
                               ucp_mem_h *memh_ptr, opal_common_ucx_mem_type_t mem_type)
{
    ucp_mem_map_params_t mem_params;
    ucp_mem_attr_t mem_attrs;
    ucs_status_t status;
    int ret = OPAL_SUCCESS;

    memset(&mem_params, 0, sizeof(ucp_mem_map_params_t));
    mem_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH
                            | UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    mem_params.length = size;
    if (mem_type == OPAL_COMMON_UCX_MEM_ALLOCATE_MAP) {
        mem_params.address = NULL;
        mem_params.flags = UCP_MEM_MAP_ALLOCATE;
    } else {
        mem_params.address = (*base);
    }

    status = ucp_mem_map(wpool->ucp_ctx, &mem_params, memh_ptr);
    if (status != UCS_OK) {
        MCA_COMMON_UCX_VERBOSE(1, "ucp_mem_map failed: %d", status);
        ret = OPAL_ERROR;
        return ret;
    }

    mem_attrs.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH;
    status = ucp_mem_query((*memh_ptr), &mem_attrs);
    if (status != UCS_OK) {
        MCA_COMMON_UCX_VERBOSE(1, "ucp_mem_query failed: %d", status);
        ret = OPAL_ERROR;
        goto error;
    }

    assert(mem_attrs.length >= size);
    if (mem_type != OPAL_COMMON_UCX_MEM_ALLOCATE_MAP) {
        /* Returned mapped address is aligned to ucs rcache->params.alignment.
         * Alignment is less than page size */
        assert(((mem_attrs.address <= (*base)) && ((*base) - opal_getpagesize()
                < mem_attrs.address)) || (size == 0 && mem_attrs.address == NULL));
    } else {
        (*base) = mem_attrs.address;
    }

    return ret;
error:
    ucp_mem_unmap(wpool->ucp_ctx, (*memh_ptr));
    return ret;
}

void opal_common_ucx_wpmem_free(opal_common_ucx_wpmem_t *mem)
{
    if (NULL == mem) {
        return;
    }

    OBJ_DESTRUCT(&mem->tls_key);

    free(mem->mem_addrs);
    free(mem->mem_displs);

    ucp_mem_unmap(mem->ctx->wpool->ucp_ctx, mem->memh);
    free(mem);
}

static inline _ctx_record_t *_tlocal_get_ctx_rec(opal_tsd_tracked_key_t tls_key)
{
    _ctx_record_t *ctx_rec = NULL;
    int rc = opal_tsd_tracked_key_get(&tls_key, (void **) &ctx_rec);

    if (OPAL_SUCCESS != rc) {
        return NULL;
    }

    return ctx_rec;
}

static void _ctx_rec_destructor(void *arg)
{
    _tlocal_ctx_rec_cleanup((_ctx_record_t *) arg);
    return;
}

/* Thread local storage destructor, also called from wpool_ctx release */
static void _tlocal_ctx_rec_cleanup(_ctx_record_t *ctx_rec)
{
    if (NULL == ctx_rec) {
        return;
    }

    opal_common_ucx_winfo_t *winfo = ctx_rec->winfo;
    opal_common_ucx_wpool_t *wpool = ctx_rec->gctx->wpool;

    opal_mutex_lock(&winfo->mutex);
    int rc = opal_common_ucx_winfo_flush(winfo, 0, OPAL_COMMON_UCX_FLUSH_B, OPAL_COMMON_UCX_SCOPE_WORKER, NULL);
    winfo->global_inflight_ops = 0;
    memset(winfo->inflight_ops, 0, winfo->comm_size * sizeof(short));
    opal_mutex_unlock(&winfo->mutex);
    if (rc != OPAL_SUCCESS) {
        MCA_COMMON_UCX_ERROR("opal_common_ucx_flush failed: %d", rc);
        return;
    }

    /* Remove worker from active and return to idle list. */
    _wpool_put_winfo(wpool, winfo);

    /* Remove the context record from the ctx list. */
    opal_mutex_lock(&ctx_rec->gctx->mutex);
    opal_list_remove_item(&ctx_rec->gctx->ctx_records, &ctx_rec->super);
    opal_mutex_unlock(&ctx_rec->gctx->mutex);

    OBJ_RELEASE(ctx_rec);

    return;
}

static _ctx_record_t *_tlocal_add_ctx_rec(opal_common_ucx_ctx_t *ctx)
{
    int rc;

    _ctx_record_t *ctx_rec = OBJ_NEW(_ctx_record_t);
    if (!ctx_rec) {
        MCA_COMMON_UCX_ERROR("Failed to allocate new ctx_rec");
        goto error1;
    }

    ctx_rec->gctx = ctx;
    ctx_rec->winfo = _wpool_get_winfo(ctx->wpool, ctx->comm_size);
    if (NULL == ctx_rec->winfo) {
        MCA_COMMON_UCX_ERROR("Failed to allocate new worker");
        goto error2;
    }

    /* Add ctx_rec to list */
    opal_mutex_lock(&ctx->mutex);
    opal_list_append(&ctx->ctx_records, &ctx_rec->super);
    opal_mutex_unlock(&ctx->mutex);

    /* Add tls reference to record */
    rc = opal_tsd_tracked_key_set(&ctx->tls_key, ctx_rec);
    if (OPAL_SUCCESS != rc) {
        MCA_COMMON_UCX_ERROR("Failed to set ctx_rec tls key");
        goto error3;
    }

    /* All good - return the record */
    return ctx_rec;

error3:
    opal_mutex_lock(&ctx->mutex);
    opal_list_remove_item(&ctx->ctx_records, &ctx_rec->super);
    opal_mutex_unlock(&ctx->mutex);
    _wpool_put_winfo(ctx->wpool, ctx_rec->winfo);
error2:
    OBJ_RELEASE(ctx_rec);
error1:
    return NULL;
}

/*
 * Open the endpoint this winfo needs in order to reach `target'.
 *
 * `shared_ep' is the caller's process-wide slot for this peer -- osc/ucx's
 * component-level endpoint array -- and it, not us, owns what ends up in it:
 * it outlives any one context, and the registry reference is released when
 * the caller tears the slot down.  We only fill it on a miss and borrow the
 * handle into our own per-winfo array.
 *
 * A winfo on a private per-thread worker cannot use the slot at all, since an
 * endpoint belongs to the worker it was opened on; it gets one of its own and
 * destroys it itself.
 */
static int _tlocal_ctx_connect(_ctx_record_t *ctx_rec, int target, ucp_ep_h *shared_ep)
{
    ucp_ep_params_t ep_params;
    opal_common_ucx_winfo_t *winfo = ctx_rec->winfo;
    opal_common_ucx_ctx_t *gctx = ctx_rec->gctx;
    ucp_address_t *address, *owned_address = NULL;
    ucs_status_t status;
    size_t addrlen;
    int rc;

    assert(winfo->endpoints[target] == NULL);

    /* The business card is where a peer's address normally comes from; the
     * gathered table is the fallback for a context whose members could not
     * reach it (see opal_common_ucx_wpctx_create()). */
    if (NULL != gctx->recv_worker_addrs) {
        address = (ucp_address_t *) &gctx->recv_worker_addrs[gctx->recv_worker_displs[target]];
    } else {
        rc = opal_common_ucx_worker_lookup_addr(&gctx->wpool->worker_user,
                                                &gctx->proc_names[target], &owned_address,
                                                &addrlen);
        if (OPAL_SUCCESS != rc) {
            return rc;
        }
        address = owned_address;
    }

    if (winfo->shared_worker && (NULL != shared_ep)) {
        /* One endpoint per peer for the whole process, so this may well be
         * one the ucx PML or another window opened already, in which case
         * the address goes unused. */
        opal_mutex_lock(&gctx->wpool->mutex);
        if (NULL == *shared_ep) {
            rc = opal_common_ucx_worker_connect_addr(&gctx->wpool->worker_user,
                                                     &gctx->proc_names[target], address,
                                                     shared_ep);
            if (OPAL_SUCCESS != rc) {
                opal_mutex_unlock(&gctx->wpool->mutex);
                goto out;
            }
            OPAL_COMMON_UCX_DEBUG_ATOMIC_ADD(opal_common_ucx_ep_counts, 1);
        }
        winfo->endpoints[target] = *shared_ep;
        opal_mutex_unlock(&gctx->wpool->mutex);

        rc = OPAL_SUCCESS;
        goto out;
    }

    /* A worker of this thread's own, which nobody else drives and whose
     * address was never published, so it needs an endpoint of its own. */
    memset(&ep_params, 0, sizeof(ucp_ep_params_t));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = address;

    opal_mutex_lock(&winfo->mutex);
    status = ucp_ep_create(winfo->worker, &ep_params, &winfo->endpoints[target]);
    opal_mutex_unlock(&winfo->mutex);
    if (status != UCS_OK) {
        MCA_COMMON_UCX_VERBOSE(1, "ucp_ep_create failed: %d", status);
        rc = OPAL_ERROR;
        goto out;
    }
    OPAL_COMMON_UCX_DEBUG_ATOMIC_ADD(opal_common_ucx_ep_counts, 1);
    rc = OPAL_SUCCESS;

out:
    free(owned_address);
    assert((OPAL_SUCCESS != rc) || (NULL != winfo->endpoints[target]));
    return rc;
}

static void _mem_rec_destructor(void *arg)
{
    _tlocal_mem_rec_cleanup((_mem_record_t *) arg);
    return;
}

static void _tlocal_mem_rec_cleanup(_mem_record_t *mem_rec)
{
    size_t i;
    if (NULL == mem_rec) {
        return;
    }

    opal_mutex_lock(&mem_rec->winfo->mutex);
    for (i = 0; i < mem_rec->gmem->ctx->comm_size; i++) {
        if (mem_rec->rkeys[i]) {
            ucp_rkey_destroy(mem_rec->rkeys[i]);
            OPAL_COMMON_UCX_DEBUG_ATOMIC_ADD(opal_common_ucx_unpacked_rkey_counts, -1);
        }
    }
    opal_mutex_unlock(&mem_rec->winfo->mutex);
    free(mem_rec->rkeys);

    OBJ_RELEASE(mem_rec);

    return;
}

static _mem_record_t *_tlocal_add_mem_rec(opal_common_ucx_wpmem_t *mem, _ctx_record_t *ctx_rec)
{
    int rc = OPAL_SUCCESS;
    _mem_record_t *mem_rec = OBJ_NEW(_mem_record_t);
    if (NULL == mem_rec) {
        return NULL;
    }

    mem_rec->gmem = mem;
    mem_rec->ctx_rec = ctx_rec;
    mem_rec->winfo = ctx_rec->winfo;
    mem_rec->rkeys = calloc(mem->ctx->comm_size, sizeof(*mem_rec->rkeys));

    rc = opal_tsd_tracked_key_set(&mem->tls_key, mem_rec);
    if (OPAL_SUCCESS != rc) {
        return NULL;
    }

    return mem_rec;
}

static int _tlocal_mem_create_rkey(_mem_record_t *mem_rec, ucp_ep_h ep, int target)
{
    opal_common_ucx_wpmem_t *gmem = mem_rec->gmem;
    int displ = gmem->mem_displs[target];
    ucs_status_t status;

    opal_mutex_lock(&mem_rec->winfo->mutex);
    status = ucp_ep_rkey_unpack(ep, &gmem->mem_addrs[displ], &mem_rec->rkeys[target]);
    OPAL_COMMON_UCX_DEBUG_ATOMIC_ADD(opal_common_ucx_unpacked_rkey_counts, 1);
    opal_mutex_unlock(&mem_rec->winfo->mutex);
    if (status != UCS_OK) {
        MCA_COMMON_UCX_VERBOSE(1, "ucp_ep_rkey_unpack failed: %d", status);
        return OPAL_ERROR;
    }

    return OPAL_SUCCESS;
}

/* Get the TLS in case of slow path (not everything has been yet initialized */
OPAL_DECLSPEC int opal_common_ucx_tlocal_fetch_spath(opal_common_ucx_wpmem_t *mem, int target,
                                                     ucp_ep_h *shared_ep)
{
    _ctx_record_t *ctx_rec = NULL;
    _mem_record_t *mem_rec = NULL;
    opal_common_ucx_winfo_t *winfo = NULL;
    ucp_ep_h ep;
    int rc = OPAL_SUCCESS;

    ctx_rec = _tlocal_get_ctx_rec(mem->ctx->tls_key);
    if (OPAL_UNLIKELY(!ctx_rec)) {
        ctx_rec = _tlocal_add_ctx_rec(mem->ctx);
        if (NULL == ctx_rec) {
            return OPAL_ERR_OUT_OF_RESOURCE;
        }
    }
    winfo = ctx_rec->winfo;

    /* Obtain the endpoint */
    if (OPAL_UNLIKELY(NULL == winfo->endpoints[target])) {
        rc = _tlocal_ctx_connect(ctx_rec, target, shared_ep);
        if (rc != OPAL_SUCCESS) {
            return rc;
        }
    }
    ep = winfo->endpoints[target];


    rc = opal_tsd_tracked_key_get(&mem->tls_key, (void **) &mem_rec);
    if (OPAL_SUCCESS != rc) {
        return rc;
    }

    if (NULL == mem_rec) {
        /* Allocate a memory region info */
        mem_rec = _tlocal_add_mem_rec(mem, ctx_rec);
    }

    /* Obtain the rkey */
    if (NULL == mem_rec->rkeys[target]) {
        /* Create the rkey */
        rc = _tlocal_mem_create_rkey(mem_rec, ep, target);
        if (rc) {
            return rc;
        }
    }

    return OPAL_SUCCESS;
}

OPAL_DECLSPEC int opal_common_ucx_winfo_flush(opal_common_ucx_winfo_t *winfo, int target,
                                              opal_common_ucx_flush_type_t type,
                                              opal_common_ucx_flush_scope_t scope,
                                              ucs_status_ptr_t *req_ptr)
{
    ucs_status_ptr_t req;
    ucs_status_t status = UCS_OK;
    int rc = OPAL_SUCCESS;

#if HAVE_DECL_UCP_EP_FLUSH_NB
    if (scope == OPAL_COMMON_UCX_SCOPE_EP) {
        req = ucp_ep_flush_nb(winfo->endpoints[target], 0, opal_common_ucx_empty_complete_cb);
    } else {
        req = ucp_worker_flush_nb(winfo->worker, 0, opal_common_ucx_empty_complete_cb);
    }
    if (UCS_PTR_IS_PTR(req)) {
        ((opal_common_ucx_request_t *) req)->winfo = winfo;
    }

    if (OPAL_COMMON_UCX_FLUSH_B == type) {
        rc = opal_common_ucx_wait_request_mt(req, "ucp_ep_flush_nb");
    } else {
        *req_ptr = req;
    }
    return rc;
#endif
    switch (type) {
    case OPAL_COMMON_UCX_FLUSH_NB_PREFERRED:
    case OPAL_COMMON_UCX_FLUSH_B:
        if (scope == OPAL_COMMON_UCX_SCOPE_EP) {
            status = ucp_ep_flush(winfo->endpoints[target]);
        } else {
            status = ucp_worker_flush(winfo->worker);
        }
        rc = (status == UCS_OK) ? OPAL_SUCCESS : OPAL_ERROR;
    case OPAL_COMMON_UCX_FLUSH_NB:
    default:
        rc = OPAL_ERROR;
    }
    return rc;
}

static inline int ctx_flush(opal_common_ucx_ctx_t *ctx,
                                opal_common_ucx_flush_scope_t scope, int target)
{
    _ctx_record_t *ctx_rec;
    int rc = OPAL_SUCCESS;

    if (NULL == ctx) {
        return OPAL_SUCCESS;
    }

    opal_mutex_lock(&ctx->mutex);

    OPAL_LIST_FOREACH (ctx_rec, &ctx->ctx_records, _ctx_record_t) {
        opal_common_ucx_winfo_t *winfo = ctx_rec->winfo;
        if ((scope == OPAL_COMMON_UCX_SCOPE_EP) && (NULL == winfo->endpoints[target])) {
            continue;
        }
        opal_mutex_lock(&winfo->mutex);
        rc = opal_common_ucx_winfo_flush(winfo, target, OPAL_COMMON_UCX_FLUSH_B, scope, NULL);
        switch (scope) {
        case OPAL_COMMON_UCX_SCOPE_WORKER:
            winfo->global_inflight_ops = 0;
            memset(winfo->inflight_ops, 0, winfo->comm_size * sizeof(short));
            break;
        case OPAL_COMMON_UCX_SCOPE_EP:
            winfo->global_inflight_ops -= winfo->inflight_ops[target];
            winfo->inflight_ops[target] = 0;
            break;
        }
        opal_mutex_unlock(&winfo->mutex);

        if (rc != OPAL_SUCCESS) {
            MCA_COMMON_UCX_ERROR("opal_common_ucx_flush failed: %d", rc);
            rc = OPAL_ERROR;
            break;
        }
    }

    opal_mutex_unlock(&ctx->mutex);

    return rc;
}

OPAL_DECLSPEC int opal_common_ucx_ctx_flush(opal_common_ucx_ctx_t *ctx,
                                    opal_common_ucx_flush_scope_t scope, int target)
{
    int rc = OPAL_SUCCESS;
    int spin = 0;

    if (NULL == ctx) {
        return OPAL_SUCCESS;
    }

    rc = ctx_flush(ctx, scope, target);
    if (rc != OPAL_SUCCESS) {
        return rc;
    }

    /* progress the nonblocking operations */
    while (ctx->num_incomplete_req_ops > 0) {
        spin++;
        rc = ctx_flush(ctx, OPAL_COMMON_UCX_SCOPE_WORKER, 0);
        if (rc != OPAL_SUCCESS) {
            return rc;
        }
        if (spin == opal_common_ucx.progress_iterations) {
            opal_progress();
            spin = 0;
        }
    }

    return rc;
}


OPAL_DECLSPEC int opal_common_ucx_wpmem_flush_ep_nb(opal_common_ucx_wpmem_t *mem,
                                                    int target,
                                                    opal_common_ucx_user_req_handler_t user_req_cb,
                                                    void *user_req_ptr, ucp_ep_h *shared_ep)
{
#if HAVE_DECL_UCP_EP_FLUSH_NB
    int rc = OPAL_SUCCESS;
    ucp_ep_h ep = NULL;
    ucp_rkey_h rkey = NULL;
    opal_common_ucx_winfo_t *winfo = NULL;

    if (NULL == mem) {
        return OPAL_SUCCESS;
    }

    rc = opal_common_ucx_tlocal_fetch(mem, target, &ep, &rkey, &winfo, shared_ep);
    if (OPAL_UNLIKELY(OPAL_SUCCESS != rc)) {
        MCA_COMMON_UCX_ERROR("tlocal_fetch failed: %d", rc);
        return rc;
    }

    opal_mutex_lock(&winfo->mutex);
    opal_common_ucx_request_t *req;
    req = ucp_ep_flush_nb(ep, 0, opal_common_ucx_req_completion);
    if (UCS_PTR_IS_PTR(req)) {
        req->ext_req = user_req_ptr;
        req->ext_cb = user_req_cb;
        req->winfo = winfo;
    } else {
        if (user_req_cb != NULL) {
            (*user_req_cb)(user_req_ptr);
        }
    }
    opal_mutex_unlock(&winfo->mutex);
    return rc;
#else
    return OPAL_ERR_NOT_SUPPORTED;
#endif // HAVE_DECL_UCP_EP_FLUSH_NB

}

/* TODO Replace the input with opal_common_ucx_ctx_t */
OPAL_DECLSPEC int opal_common_ucx_wpmem_fence(opal_common_ucx_wpmem_t *mem)
{
    ucs_status_t status = UCS_OK;
    _ctx_record_t *ctx_rec;
    opal_common_ucx_winfo_t *winfo;
    opal_common_ucx_ctx_t *ctx;
    int rc = OPAL_SUCCESS;

    if (NULL == mem) {
        return OPAL_SUCCESS;
    }

    ctx = mem->ctx;
    opal_mutex_lock(&ctx->mutex);

    OPAL_LIST_FOREACH (ctx_rec, &ctx->ctx_records, _ctx_record_t) {
        winfo = ctx_rec->winfo;
        opal_mutex_lock(&winfo->mutex);
        status = ucp_worker_fence(winfo->worker);
        opal_mutex_unlock(&winfo->mutex);
        if (status != UCS_OK) {
            MCA_COMMON_UCX_ERROR("opal_common_ucx_fence failed: %d", rc);
            rc = OPAL_ERROR;
            break;
        }
    }

    opal_mutex_unlock(&ctx->mutex);

    return rc;
}

OPAL_DECLSPEC void opal_common_ucx_req_init(void *request)
{
    opal_common_ucx_request_t *req = (opal_common_ucx_request_t *) request;
    req->ext_req = NULL;
    req->ext_cb = NULL;
    req->winfo = NULL;
}

OPAL_DECLSPEC void opal_common_ucx_req_completion(void *request, ucs_status_t status)
{
    opal_common_ucx_request_t *req = (opal_common_ucx_request_t *) request;
    if (req->ext_cb != NULL) {
        (*req->ext_cb)(req->ext_req);
    }
    ucp_request_release(req);
}
