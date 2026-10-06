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
#include <string.h>

#include "common_ucx.h"
#include "common_ucx_md.h"

#include "opal/mca/threads/mutex.h"

static opal_list_t opal_common_ucx_md_registry;
static bool opal_common_ucx_md_registry_valid = false;
static int opal_common_ucx_md_users = 0;
static opal_mutex_t opal_common_ucx_md_mutex = OPAL_MUTEX_STATIC_INIT;

static void opal_common_ucx_md_construct(opal_common_ucx_md_t *md)
{
    md->md_name = NULL;
    md->uct_component = NULL;
    md->uct_md = NULL;
    md->refcnt = 0;
    md->tl_resources = NULL;
    md->num_tl_resources = 0;
    md->tl_resources_valid = false;
}

static void opal_common_ucx_md_destruct(opal_common_ucx_md_t *md)
{
    /* A domain still referenced here means a user did not release it.  Its
     * handle is more likely to still be in use than not, so leave it open
     * rather than pull it out from under them. */
    if (0 == md->refcnt && NULL != md->uct_md) {
        uct_md_close(md->uct_md);
    }
    md->uct_md = NULL;

    free(md->md_name);
    md->md_name = NULL;
    free(md->tl_resources);
    md->tl_resources = NULL;
    md->num_tl_resources = 0;
    md->tl_resources_valid = false;
}

OBJ_CLASS_INSTANCE(opal_common_ucx_md_t, opal_list_item_t, opal_common_ucx_md_construct,
                   opal_common_ucx_md_destruct);

/*
 * Add every memory domain of one component to the registry.
 *
 * Only names are recorded; nothing is opened here.
 */
static int opal_common_ucx_md_add_component(uct_component_h component)
{
    uct_component_attr_t attr = {
        .field_mask = UCT_COMPONENT_ATTR_FIELD_NAME
                      | UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT,
    };
    opal_common_ucx_md_t *md;
    ucs_status_t status;

    status = uct_component_query(component, &attr);
    if (UCS_OK != status) {
        MCA_COMMON_UCX_VERBOSE(2, "uct_component_query() failed: %s",
                               ucs_status_string(status));
        return OPAL_SUCCESS; /* skip this component, not a fatal condition */
    }

    if (0 == attr.md_resource_count) {
        return OPAL_SUCCESS;
    }

    attr.md_resources = calloc(attr.md_resource_count, sizeof(*attr.md_resources));
    if (NULL == attr.md_resources) {
        return OPAL_ERR_OUT_OF_RESOURCE;
    }

    attr.field_mask |= UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
    status = uct_component_query(component, &attr);
    if (UCS_OK != status) {
        MCA_COMMON_UCX_VERBOSE(2, "uct_component_query(%s) failed: %s", attr.name,
                               ucs_status_string(status));
        free(attr.md_resources);
        return OPAL_SUCCESS;
    }

    for (unsigned i = 0; i < attr.md_resource_count; ++i) {
        md = OBJ_NEW(opal_common_ucx_md_t);
        if (NULL == md) {
            free(attr.md_resources);
            return OPAL_ERR_OUT_OF_RESOURCE;
        }

        md->md_name = strdup(attr.md_resources[i].md_name);
        if (NULL == md->md_name) {
            OBJ_RELEASE(md);
            free(attr.md_resources);
            return OPAL_ERR_OUT_OF_RESOURCE;
        }

        md->uct_component = component;
        opal_list_append(&opal_common_ucx_md_registry, &md->super);

        MCA_COMMON_UCX_VERBOSE(3, "component %s offers memory domain %s", attr.name,
                               md->md_name);
    }

    free(attr.md_resources);

    return OPAL_SUCCESS;
}

/* Caller must hold opal_common_ucx_md_mutex. */
static int opal_common_ucx_md_registry_init(void)
{
    uct_component_h *components;
    unsigned num_components;
    ucs_status_t status;
    int rc = OPAL_SUCCESS;

    if (opal_common_ucx_md_registry_valid) {
        return OPAL_SUCCESS;
    }

    status = uct_query_components(&components, &num_components);
    if (UCS_OK != status) {
        MCA_COMMON_UCX_VERBOSE(1, "uct_query_components() failed: %s",
                               ucs_status_string(status));
        return OPAL_ERR_NOT_AVAILABLE;
    }

    OBJ_CONSTRUCT(&opal_common_ucx_md_registry, opal_list_t);

    for (unsigned i = 0; i < num_components; ++i) {
        rc = opal_common_ucx_md_add_component(components[i]);
        if (OPAL_SUCCESS != rc) {
            break;
        }
    }

    /* Only the array is ours to release; the uct_component_h values in it
     * refer to UCX's own process-wide component list and stay valid, which
     * is also what lets us compare them against another caller's. */
    uct_release_component_list(components);

    if (OPAL_SUCCESS != rc) {
        OPAL_LIST_DESTRUCT(&opal_common_ucx_md_registry);
        return rc;
    }

    opal_common_ucx_md_registry_valid = true;

    return OPAL_SUCCESS;
}

int opal_common_ucx_md_init(void)
{
    int rc;

    opal_mutex_lock(&opal_common_ucx_md_mutex);
    rc = opal_common_ucx_md_registry_init();
    if (OPAL_SUCCESS == rc) {
        ++opal_common_ucx_md_users;
    }
    opal_mutex_unlock(&opal_common_ucx_md_mutex);

    return rc;
}

void opal_common_ucx_md_finalize(void)
{
    opal_mutex_lock(&opal_common_ucx_md_mutex);

    assert(opal_common_ucx_md_users > 0);

    if ((0 == --opal_common_ucx_md_users) && opal_common_ucx_md_registry_valid) {
        OPAL_LIST_DESTRUCT(&opal_common_ucx_md_registry);
        opal_common_ucx_md_registry_valid = false;
    }

    opal_mutex_unlock(&opal_common_ucx_md_mutex);
}

int opal_common_ucx_md_list(opal_list_t **md_list)
{
    bool valid;

    opal_mutex_lock(&opal_common_ucx_md_mutex);
    valid = opal_common_ucx_md_registry_valid;
    opal_mutex_unlock(&opal_common_ucx_md_mutex);

    if (!valid) {
        return OPAL_ERR_NOT_AVAILABLE;
    }

    /* The list itself is built once by opal_common_ucx_md_init() and never
     * added to or removed from afterwards, so a caller holding a reference
     * can walk it without the lock. */
    *md_list = &opal_common_ucx_md_registry;

    return OPAL_SUCCESS;
}

opal_common_ucx_md_t *opal_common_ucx_md_lookup(uct_component_h component, const char *md_name)
{
    opal_common_ucx_md_t *md;
    opal_list_t *md_list;

    if (OPAL_SUCCESS != opal_common_ucx_md_list(&md_list)) {
        return NULL;
    }

    OPAL_LIST_FOREACH (md, md_list, opal_common_ucx_md_t) {
        if ((component == md->uct_component) && (0 == strcmp(md->md_name, md_name))) {
            return md;
        }
    }

    return NULL;
}

/* Caller must hold opal_common_ucx_md_mutex. */
static int opal_common_ucx_md_open(opal_common_ucx_md_t *md)
{
    uct_md_config_t *md_config;
    ucs_status_t status;

    assert(NULL == md->uct_md);

    status = uct_md_config_read(md->uct_component, NULL, NULL, &md_config);
    if (UCS_OK != status) {
        MCA_COMMON_UCX_VERBOSE(2, "uct_md_config_read(%s) failed: %s", md->md_name,
                               ucs_status_string(status));
        return OPAL_ERR_NOT_AVAILABLE;
    }

    status = uct_md_open(md->uct_component, md->md_name, md_config, &md->uct_md);
    uct_config_release(md_config);
    if (UCS_OK != status) {
        MCA_COMMON_UCX_VERBOSE(2, "uct_md_open(%s) failed: %s", md->md_name,
                               ucs_status_string(status));
        md->uct_md = NULL;
        return OPAL_ERR_NOT_AVAILABLE;
    }

    status = uct_md_query(md->uct_md, &md->md_attr);
    if (UCS_OK != status) {
        MCA_COMMON_UCX_VERBOSE(2, "uct_md_query(%s) failed: %s", md->md_name,
                               ucs_status_string(status));
        uct_md_close(md->uct_md);
        md->uct_md = NULL;
        return OPAL_ERR_NOT_AVAILABLE;
    }

    MCA_COMMON_UCX_VERBOSE(2, "opened memory domain %s", md->md_name);

    return OPAL_SUCCESS;
}

int opal_common_ucx_md_acquire(opal_common_ucx_md_t *md)
{
    int rc = OPAL_SUCCESS;

    opal_mutex_lock(&opal_common_ucx_md_mutex);

    if (NULL == md->uct_md) {
        rc = opal_common_ucx_md_open(md);
    }

    if (OPAL_SUCCESS == rc) {
        ++md->refcnt;
    }

    opal_mutex_unlock(&opal_common_ucx_md_mutex);

    return rc;
}

void opal_common_ucx_md_release(opal_common_ucx_md_t *md)
{
    opal_mutex_lock(&opal_common_ucx_md_mutex);

    assert(md->refcnt > 0);

    if (0 == --md->refcnt) {
        MCA_COMMON_UCX_VERBOSE(2, "closing memory domain %s", md->md_name);
        uct_md_close(md->uct_md);
        md->uct_md = NULL;
    }

    opal_mutex_unlock(&opal_common_ucx_md_mutex);
}

int opal_common_ucx_md_tl_resources(opal_common_ucx_md_t *md,
                                    const uct_tl_resource_desc_t **tl_resources,
                                    unsigned *num_tl_resources)
{
    uct_tl_resource_desc_t *uct_tls;
    unsigned num_uct_tls;
    ucs_status_t status;
    bool opened = false;
    int rc = OPAL_SUCCESS;

    opal_mutex_lock(&opal_common_ucx_md_mutex);

    if (md->tl_resources_valid) {
        goto out;
    }

    /* Someone may already be working with this domain, in which case we ask
     * through their handle and leave it alone.  Otherwise open it just long
     * enough to ask. */
    if (NULL == md->uct_md) {
        rc = opal_common_ucx_md_open(md);
        if (OPAL_SUCCESS != rc) {
            goto out;
        }
        opened = true;
    }

    status = uct_md_query_tl_resources(md->uct_md, &uct_tls, &num_uct_tls);
    if (UCS_OK != status) {
        MCA_COMMON_UCX_VERBOSE(2, "uct_md_query_tl_resources(%s) failed: %s", md->md_name,
                               ucs_status_string(status));
        rc = OPAL_ERR_NOT_AVAILABLE;
        goto out_close;
    }

    /* Keep our own copy so the list outlives the domain being open. */
    if (0 != num_uct_tls) {
        md->tl_resources = malloc(num_uct_tls * sizeof(*md->tl_resources));
        if (NULL == md->tl_resources) {
            uct_release_tl_resource_list(uct_tls);
            rc = OPAL_ERR_OUT_OF_RESOURCE;
            goto out_close;
        }
        memcpy(md->tl_resources, uct_tls, num_uct_tls * sizeof(*md->tl_resources));
    }

    md->num_tl_resources = num_uct_tls;
    md->tl_resources_valid = true;
    uct_release_tl_resource_list(uct_tls);

out_close:
    if (opened) {
        uct_md_close(md->uct_md);
        md->uct_md = NULL;
    }

out:
    if (OPAL_SUCCESS == rc) {
        *tl_resources = md->tl_resources;
        *num_tl_resources = md->num_tl_resources;
    }

    opal_mutex_unlock(&opal_common_ucx_md_mutex);

    return rc;
}
