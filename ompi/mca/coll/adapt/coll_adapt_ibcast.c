#include "ompi_config.h"
#include "ompi/mca/pml/pml.h"
#include "coll_adapt.h"
#include "coll_adapt_algorithms.h"
#include "coll_adapt_context.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/coll/base/coll_base_functions.h"     //COLL_BASE_COMPUTED_SEGCOUNT
#include "opal/util/bit_ops.h"
#include "opal/sys/atomic.h"                //atomic
#include "ompi/mca/pml/ob1/pml_ob1.h"       //dump


typedef int (*mca_coll_adapt_ibcast_fn_t)(
    void *buff, 
    int count, 
    struct ompi_datatype_t *datatype, 
    int root, 
    struct ompi_communicator_t *comm, 
    ompi_request_t ** request, 
    mca_coll_base_module_t *module, 
    int ibcast_tag
);
    
static mca_coll_adapt_algorithm_index_t mca_coll_adapt_ibcast_algorithm_index[] = {
    {0, (uintptr_t)mca_coll_adapt_ibcast_tuned},
    {1, (uintptr_t)mca_coll_adapt_ibcast_binomial},
    {2, (uintptr_t)mca_coll_adapt_ibcast_in_order_binomial},
    {3, (uintptr_t)mca_coll_adapt_ibcast_binary},
    {4, (uintptr_t)mca_coll_adapt_ibcast_pipeline},
    {5, (uintptr_t)mca_coll_adapt_ibcast_chain},
    {6, (uintptr_t)mca_coll_adapt_ibcast_linear},
};

int mca_coll_adapt_ibcast_init(void)
{
    mca_base_component_t *c = &mca_coll_adapt_component.super.collm_version;
    
    mca_coll_adapt_component.adapt_ibcast_algorithm = 1;
    mca_base_component_var_register(c, "bcast_algorithm",
                                    "Algorithm of broadcast",
                                    MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_READONLY,
                                    &mca_coll_adapt_component.adapt_ibcast_algorithm);
    
    mca_coll_adapt_component.adapt_ibcast_segment_size = 0;
    mca_base_component_var_register(c, "bcast_segment_size",
                                    "Segment size in bytes used by default for bcast algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                    MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_READONLY,
                                    &mca_coll_adapt_component.adapt_ibcast_segment_size);

    mca_coll_adapt_component.adapt_ibcast_max_send_requests = 2;
    mca_base_component_var_register(c, "bcast_max_send_requests",
                                    "Maximum number of send requests",
                                    MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_READONLY,
                                    &mca_coll_adapt_component.adapt_ibcast_max_send_requests);
    
    mca_coll_adapt_component.adapt_ibcast_max_recv_requests = 3;
    mca_base_component_var_register(c, "bcast_max_recv_requests",
                                    "Maximum number of receive requests",
                                    MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_READONLY,
                                    &mca_coll_adapt_component.adapt_ibcast_max_recv_requests);
    
    mca_coll_adapt_component.adapt_ibcast_context_free_list = NULL;
    mca_coll_adapt_component.adapt_ibcast_context_free_list_enabled = 0;
    return OMPI_SUCCESS;
}

int mca_coll_adapt_ibcast_fini(void)
{
    /* release the free list */
    if (NULL != mca_coll_adapt_component.adapt_ibcast_context_free_list) {
        OBJ_RELEASE(mca_coll_adapt_component.adapt_ibcast_context_free_list);
        mca_coll_adapt_component.adapt_ibcast_context_free_list = NULL;
        mca_coll_adapt_component.adapt_ibcast_context_free_list_enabled = 0;
        OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "ibcast fini\n"));
    }
    return OMPI_SUCCESS;
}

static int ibcast_request_fini(mca_coll_adapt_bcast_context_t *context)
{
    ompi_request_t *temp_req = context->con->request;
    if (context->con->tree->tree_nextsize != 0) {
        free(context->con->send_array);
    }
    if (context->con->num_segs !=0) {
        free(context->con->recv_array);
    }
    OBJ_RELEASE(context->con->mutex);
    OBJ_RELEASE(context->con);
    OBJ_RELEASE(context->con);
    opal_free_list_return(mca_coll_adapt_component.adapt_ibcast_context_free_list, (opal_free_list_item_t*)context);
    ompi_request_complete(temp_req, 1);
    
    return OMPI_SUCCESS;
}

/* send call back */
/*
 * The req_lock is held when entering this routine
 * It is released in this routine if no error occured
 */
static int send_cb(ompi_request_t *req)
{
    mca_coll_adapt_bcast_context_t *context = (mca_coll_adapt_bcast_context_t *) req->req_complete_cb_data;
    
    //opal_output_init();
    //mca_pml_ob1_dump(context->con->comm, 0);
    //opal_output_finalize();
    
    int err;
    
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: Send(cb): segment %d to %d at buff %p root %d\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff, context->con->root));

    OPAL_THREAD_LOCK(context->con->mutex);
    int sent_id = context->con->send_array[context->child_id];
    //has fragments in recv_array can be sent
    if (sent_id < context->con->num_recv_segs) {
        ompi_request_t *send_req;
        int new_id = context->con->recv_array[sent_id];
        mca_coll_adapt_bcast_context_t * send_context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(mca_coll_adapt_component.adapt_ibcast_context_free_list);
        send_context->buff = context->buff + (new_id - context->frag_id) * context->con->real_seg_size;
        send_context->frag_id = new_id;
        send_context->child_id = context->child_id;
        send_context->peer = context->peer;
        send_context->con = context->con;
        OBJ_RETAIN(context->con);
        int send_count = send_context->con->seg_count;
        if (new_id == (send_context->con->num_segs - 1)) {
            send_count = send_context->con->count - new_id * send_context->con->seg_count;
        }
        ++(send_context->con->send_array[send_context->child_id]);
        char *send_buff = send_context->buff;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Send(start in send cb): segment %d to %d at buff %p send_count %d tag %d\n", ompi_comm_rank(send_context->con->comm), send_context->frag_id, send_context->peer, (void *)send_context->buff, send_count, (send_context->con->ibcast_tag << 16) + new_id));
        err = MCA_PML_CALL(isend(send_buff, send_count, send_context->con->datatype, send_context->peer, (send_context->con->ibcast_tag << 16) + new_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
        if (MPI_SUCCESS != err) {
            OPAL_THREAD_UNLOCK(context->con->mutex);
            /*
             * No need to unlock the req_lock in case of error:
             * this is done in the calling routine
             */
            return err;
        }
        //invoke send call back
        OPAL_THREAD_UNLOCK(context->con->mutex);
        ompi_request_set_callback(send_req, send_cb, send_context);
        OPAL_THREAD_LOCK(context->con->mutex);
    }

    int num_sent = ++(context->con->num_sent_segs);
    int num_recv_fini_t = context->con->num_recv_fini;
    int rank = ompi_comm_rank(context->con->comm);
    opal_mutex_t * mutex_temp = context->con->mutex;
    //check whether signal the condition
    if ((rank == context->con->root && num_sent == context->con->tree->tree_nextsize * context->con->num_segs) ||
        (context->con->tree->tree_nextsize > 0 && rank != context->con->root && num_sent == context->con->tree->tree_nextsize * context->con->num_segs && num_recv_fini_t == context->con->num_segs) ||
        (context->con->tree->tree_nextsize == 0 && num_recv_fini_t == context->con->num_segs)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Singal in send\n", ompi_comm_rank(context->con->comm)));
        OPAL_THREAD_UNLOCK(mutex_temp);
        ibcast_request_fini(context);
    }
    else {
        OBJ_RELEASE(context->con);
        opal_free_list_return(mca_coll_adapt_component.adapt_ibcast_context_free_list, (opal_free_list_item_t*)context);
        OPAL_THREAD_UNLOCK(mutex_temp);
    }
    OPAL_THREAD_UNLOCK(&(req->req_lock));
    req->req_free(&req);
    return 1;
}

//receive call back
/*
 * The req_lock is held when entering this routine
 * It is released in this routine if no error occured
 */
static int recv_cb(ompi_request_t *req){
    //get necessary info from request
    mca_coll_adapt_bcast_context_t *context = (mca_coll_adapt_bcast_context_t *) req->req_complete_cb_data;
    
    int err, i;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: Recv(cb): segment %d from %d at buff %p root %d\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff, context->con->root));
    
    //store the frag_id to seg array
    OPAL_THREAD_LOCK(context->con->mutex);
    int num_recv_segs_t = ++(context->con->num_recv_segs);
    context->con->recv_array[num_recv_segs_t-1] = context->frag_id;
    
    int new_id = num_recv_segs_t + mca_coll_adapt_component.adapt_ibcast_max_recv_requests - 1;
    //receive new segment
    if (new_id < context->con->num_segs) {
        ompi_request_t *recv_req;
        //get new context item from free list
        mca_coll_adapt_bcast_context_t * recv_context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(mca_coll_adapt_component.adapt_ibcast_context_free_list);
        recv_context->buff = context->buff + (new_id - context->frag_id) * context->con->real_seg_size;
        recv_context->frag_id = new_id;
        recv_context->child_id = context->child_id;
        recv_context->peer = context->peer;
        recv_context->con = context->con;
        OBJ_RETAIN(context->con);
        int recv_count = recv_context->con->seg_count;
        if (new_id == (recv_context->con->num_segs - 1)) {
            recv_count = recv_context->con->count - new_id * recv_context->con->seg_count;
        }
        char *recv_buff = recv_context->buff;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Recv(start in recv cb): segment %d from %d at buff %p recv_count %d tag %d\n", ompi_comm_rank(context->con->comm), context->frag_id, context->peer, (void *)recv_buff, recv_count, (recv_context->con->ibcast_tag << 16) + recv_context->frag_id));
        MCA_PML_CALL(irecv(recv_buff, recv_count, recv_context->con->datatype, recv_context->peer, (recv_context->con->ibcast_tag << 16) + recv_context->frag_id, recv_context->con->comm, &recv_req));

        //invoke recvive call back
        OPAL_THREAD_UNLOCK(context->con->mutex);
        ompi_request_set_callback(recv_req, recv_cb, recv_context);
        OPAL_THREAD_LOCK(context->con->mutex);
    }
    
    //send segment to its children
    for (i = 0; i < context->con->tree->tree_nextsize; i++) {
        //if can send the segment now means the only segment need to be sent is the just arrived one
        if (num_recv_segs_t-1 == context->con->send_array[i]) {
            ompi_request_t *send_req;
            int send_count = context->con->seg_count;
            if (context->frag_id == (context->con->num_segs - 1)) {
                send_count = context->con->count - context->frag_id * context->con->seg_count;
            }
            
            mca_coll_adapt_bcast_context_t * send_context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(mca_coll_adapt_component.adapt_ibcast_context_free_list);
            send_context->buff = context->buff;
            send_context->frag_id = context->frag_id;
            send_context->child_id = i;
            send_context->peer = context->con->tree->tree_next[i];
            send_context->con = context->con;
            OBJ_RETAIN(context->con);
            ++(send_context->con->send_array[i]);
            char *send_buff = send_context->buff;
            OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Send(start in recv cb): segment %d to %d at buff %p send_count %d tag %d\n", ompi_comm_rank(send_context->con->comm), send_context->frag_id, send_context->peer, (void *)send_context->buff, send_count, (send_context->con->ibcast_tag << 16) + send_context->frag_id));
            err = MCA_PML_CALL(isend(send_buff, send_count, send_context->con->datatype, send_context->peer, (send_context->con->ibcast_tag << 16) + send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
            if (MPI_SUCCESS != err) {
                OPAL_THREAD_UNLOCK(context->con->mutex);
                return err;
            }
            //invoke send call back
            OPAL_THREAD_UNLOCK(context->con->mutex);
            ompi_request_set_callback(send_req, send_cb, send_context);
            OPAL_THREAD_LOCK(context->con->mutex);
        }
    }
    
    int num_sent = context->con->num_sent_segs;
    int num_recv_fini_t = ++(context->con->num_recv_fini);
    int rank = ompi_comm_rank(context->con->comm);
    opal_mutex_t * mutex_temp = context->con->mutex;

    //if this is leaf and has received all the segments
    if ((rank == context->con->root && num_sent == context->con->tree->tree_nextsize * context->con->num_segs) ||
        (context->con->tree->tree_nextsize > 0 && rank != context->con->root && num_sent == context->con->tree->tree_nextsize * context->con->num_segs && num_recv_fini_t == context->con->num_segs) ||
        (context->con->tree->tree_nextsize == 0 && num_recv_fini_t == context->con->num_segs)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d]: Singal in recv\n", ompi_comm_rank(context->con->comm)));
        OPAL_THREAD_UNLOCK(mutex_temp);
        ibcast_request_fini(context);
    }
    else{
        OBJ_RELEASE(context->con);
        opal_free_list_return(mca_coll_adapt_component.adapt_ibcast_context_free_list, (opal_free_list_item_t*)context);
        OPAL_THREAD_UNLOCK(mutex_temp);
    }
    OPAL_THREAD_UNLOCK(&(req->req_lock));
    req->req_free(&req);
    return 1;
}

int mca_coll_adapt_ibcast(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    //printf("ADAPT\n");
    if (count == 0) {
        ompi_request_t *temp_request;
        temp_request = OBJ_NEW(ompi_request_t);
        OMPI_REQUEST_INIT(temp_request, false);
        temp_request->req_type = 0;
        temp_request->req_free = adapt_request_free;
        temp_request->req_status.MPI_SOURCE = 0;
        temp_request->req_status.MPI_TAG = 0;
        temp_request->req_status.MPI_ERROR = 0;
        temp_request->req_status._cancelled = 0;
        temp_request->req_status._ucount = 0;
        ompi_request_complete(temp_request, 1);
        *request = temp_request;
        return MPI_SUCCESS;
    }
    else {
        int rank = ompi_comm_rank(comm);
        if (rank == root) {
            OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "ibcast root %d, algorithm %d, coll_adapt_ibcast_segment_size %zu, coll_adapt_ibcast_max_send_requests %d, coll_adapt_ibcast_max_recv_requests %d\n", root, mca_coll_adapt_component.adapt_ibcast_algorithm, mca_coll_adapt_component.adapt_ibcast_segment_size, mca_coll_adapt_component.adapt_ibcast_max_send_requests, mca_coll_adapt_component.adapt_ibcast_max_recv_requests));
        }
        int ibcast_tag = opal_atomic_add_fetch_32(&(comm->c_ibcast_tag), 1);
        ibcast_tag = ibcast_tag % 4096;
        mca_coll_adapt_ibcast_fn_t bcast_func = (mca_coll_adapt_ibcast_fn_t)mca_coll_adapt_ibcast_algorithm_index[mca_coll_adapt_component.adapt_ibcast_algorithm].algorithm_fn_ptr;
        return bcast_func(buff, count, datatype, root, comm, request, module, ibcast_tag);
        //return mca_coll_adapt_ibcast_binomial(buff, count, datatype, root, comm, request, module, ibcast_tag);
    }
}

int mca_coll_adapt_ibcast_tuned(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    OPAL_OUTPUT_VERBOSE((10, mca_coll_adapt_component.adapt_output, "tuned not implemented\n"));
    return OMPI_SUCCESS;

}

int mca_coll_adapt_ibcast_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_bmtree(comm, root);
    int err = mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, mca_coll_adapt_component.adapt_ibcast_segment_size, ibcast_tag);
    return err;
}

int mca_coll_adapt_ibcast_in_order_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_in_order_bmtree(comm, root);
    int err = mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, mca_coll_adapt_component.adapt_ibcast_segment_size, ibcast_tag);
    return err;
}


int mca_coll_adapt_ibcast_binary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    ompi_coll_tree_t* tree = ompi_coll_base_topo_build_tree(2, comm, root);
    int err = mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, mca_coll_adapt_component.adapt_ibcast_segment_size, ibcast_tag);
    return err;
}

int mca_coll_adapt_ibcast_pipeline(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_chain(1, comm, root);
    int err = mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, mca_coll_adapt_component.adapt_ibcast_segment_size, ibcast_tag);
    return err;
}


int mca_coll_adapt_ibcast_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    ompi_coll_tree_t * tree = ompi_coll_base_topo_build_chain(4, comm, root);
    int err = mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, mca_coll_adapt_component.adapt_ibcast_segment_size, ibcast_tag);
    return err;
}

int mca_coll_adapt_ibcast_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, int ibcast_tag){
    int fanout = ompi_comm_size(comm) - 1;
    ompi_coll_tree_t * tree;
    if (fanout < 1) {
        tree = ompi_coll_base_topo_build_chain(1, comm, root);
    }
    else if (fanout <= MAXTREEFANOUT) {
        tree = ompi_coll_base_topo_build_tree(ompi_comm_size(comm) - 1, comm, root);
    }
    else {
        tree = ompi_coll_base_topo_build_tree(MAXTREEFANOUT, comm, root);
    }
    int err = mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, tree, mca_coll_adapt_component.adapt_ibcast_segment_size, ibcast_tag);
    return err;
}


int mca_coll_adapt_ibcast_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree, size_t seg_size, int ibcast_tag){
    int i, j;       //temp variable for iteration
    int rank;       //rank of this node
    int err;        //record return value
    int min;        //the min of num_segs and SEND_NUM or RECV_NUM, in case the num_segs is less than SEND_NUM or RECV_NUM
    
    int seg_count = count;      //number of datatype in a segment
    size_t type_size;           //the size of a datatype
    size_t real_seg_size;       //the real size of a segment
    ptrdiff_t extent, lb;
    int num_segs;               //the number of segments
    
    ompi_request_t * temp_request = NULL;  //the request be passed outside
    opal_mutex_t * mutex;
    int *recv_array = NULL;   //store those segments which are received
    int *send_array = NULL;   //record how many isend has been issued for every child
    
    //set up free list
    if (0 == mca_coll_adapt_component.adapt_ibcast_context_free_list_enabled) {
        int32_t context_free_list_enabled = opal_atomic_add_fetch_32(&(mca_coll_adapt_component.adapt_ibcast_context_free_list_enabled), 1);
        if (1 == context_free_list_enabled) {
            mca_coll_adapt_component.adapt_ibcast_context_free_list = OBJ_NEW(opal_free_list_t);
            opal_free_list_init(mca_coll_adapt_component.adapt_ibcast_context_free_list,
                                sizeof(mca_coll_adapt_bcast_context_t),
                                opal_cache_line_size,
                                OBJ_CLASS(mca_coll_adapt_bcast_context_t),
                                0,opal_cache_line_size,
                                mca_coll_adapt_component.adapt_context_free_list_min,
                                mca_coll_adapt_component.adapt_context_free_list_max,
                                mca_coll_adapt_component.adapt_context_free_list_inc,
                                NULL, 0, NULL, NULL, NULL);
        }
    }
    
    //set up request
    temp_request = OBJ_NEW(ompi_request_t);
    OMPI_REQUEST_INIT(temp_request, false);
    temp_request->req_state = OMPI_REQUEST_ACTIVE;
    temp_request->req_type = 0;
    temp_request->req_free = adapt_request_free;
    temp_request->req_status.MPI_SOURCE = 0;
    temp_request->req_status.MPI_TAG = 0;
    temp_request->req_status.MPI_ERROR = 0;
    temp_request->req_status._cancelled = 0;
    temp_request->req_status._ucount = 0;
    *request = temp_request;
    
    //set up mutex
    mutex = OBJ_NEW(opal_mutex_t);
    
    rank = ompi_comm_rank(comm);
    
    //Determine number of elements sent per operation
    ompi_datatype_type_size(datatype, &type_size);
    COLL_BASE_COMPUTED_SEGCOUNT(seg_size, type_size, seg_count);
    
    ompi_datatype_get_extent(datatype, &lb, &extent);
    num_segs = (count + seg_count - 1) / seg_count;
    real_seg_size = (ptrdiff_t)seg_count * extent;
    
    //set memory for recv_array and send_array, created on heap becasue they are needed to be accessed by other functions, callback function
    if (num_segs!=0) {
        recv_array = (int *)malloc(sizeof(int) * num_segs);
    }
    if (tree->tree_nextsize!=0) {
        send_array = (int *)malloc(sizeof(int) * tree->tree_nextsize);
    }
    
    //Set constant context for send and recv call back
    mca_coll_adapt_constant_bcast_context_t *con = OBJ_NEW(mca_coll_adapt_constant_bcast_context_t);
    con->root = root;
    con->count = count;
    con->seg_count = seg_count;
    con->datatype = datatype;
    con->comm = comm;
    con->real_seg_size = real_seg_size;
    con->num_segs = num_segs;
    con->recv_array = recv_array;
    con->num_recv_segs = 0;
    con->num_recv_fini = 0;
    con->send_array = send_array;
    con->num_sent_segs = 0;
    con->mutex = mutex;
    con->request = temp_request;
    con->tree = tree;
    con->ibcast_tag = ibcast_tag;
    
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output,"[%d, %" PRIx64 "]: Ibcast, root %d, tag %d\n", rank, gettid(), root, ibcast_tag));
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output,"[%d, %" PRIx64 "]: con->mutex = %p, num_children = %d, num_segs = %d, real_seg_size = %d, seg_count = %d, tree_adreess = %p\n", rank, gettid(), (void *)con->mutex, tree->tree_nextsize, num_segs, (int)real_seg_size, seg_count, (void *)con->tree));
    
    OPAL_THREAD_LOCK(mutex);
    
    //if root, send segment to every children.
    if (rank == root){
        //handle the situation when num_segs < SEND_NUM
        if (num_segs <= mca_coll_adapt_component.adapt_ibcast_max_send_requests) {
            min = num_segs;
        }
        else{
            min = mca_coll_adapt_component.adapt_ibcast_max_send_requests;
        }
        
        //set recv_array, root has already had all the segments
        for (i = 0; i < num_segs; i++) {
            recv_array[i] = i;
        }
        con->num_recv_segs = num_segs;
        //set send_array, has not sent any segments
        for (i = 0; i < tree->tree_nextsize; i++) {
            send_array[i] = mca_coll_adapt_component.adapt_ibcast_max_send_requests;
        }
        
        ompi_request_t *send_req;
        int send_count = seg_count;             //number of datatype in each send
        for (i = 0; i < min; i++) {
            if (i == (num_segs - 1)) {
                send_count = count - i * seg_count;
            }
            for (j=0; j<tree->tree_nextsize; j++) {
                mca_coll_adapt_bcast_context_t * context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(mca_coll_adapt_component.adapt_ibcast_context_free_list);
                context->buff = (char *)buff + i * real_seg_size;
                context->frag_id = i;
                context->child_id = j;              //the id of peer in in children_list
                context->peer = tree->tree_next[j];   //the actural rank of the peer
                context->con = con;
                OBJ_RETAIN(con);

                char* send_buff = context->buff;
                OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: Send(start in main): segment %d to %d at buff %p send_count %d tag %d\n", rank, gettid(), context->frag_id, context->peer, (void *)send_buff, send_count, (ibcast_tag << 16) + i));
                err = MCA_PML_CALL(isend(send_buff, send_count, datatype, context->peer, (ibcast_tag << 16) + i, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req));         
                if (MPI_SUCCESS != err) {
                    return err;
                }
                //invoke send call back
                OPAL_THREAD_UNLOCK(mutex);
                ompi_request_set_callback(send_req, send_cb, context);
                OPAL_THREAD_LOCK(mutex);
            }
        }
        
    }
    
    //if not root, receive data from parent in the tree.
    else{
        //handle the situation is num_segs < RECV_NUM
        if (num_segs <= mca_coll_adapt_component.adapt_ibcast_max_recv_requests) {
            min = num_segs;
        }
        else{
            min = mca_coll_adapt_component.adapt_ibcast_max_recv_requests;
        }
        
        //set recv_array, recv_array is empty
        for (i = 0; i < num_segs; i++) {
            recv_array[i] = 0;
        }
        //set send_array to empty
        for (i = 0; i < tree->tree_nextsize ; i++) {
            send_array[i] = 0;
        }
        
        //create a recv request
        ompi_request_t *recv_req;
        
        //recevice some segments from its parent
        int recv_count = seg_count;
        for (i = 0; i < min; i++) {
            if (i == (num_segs - 1)) {
                recv_count = count - i * seg_count;
            }
            mca_coll_adapt_bcast_context_t * context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(mca_coll_adapt_component.adapt_ibcast_context_free_list);
            context->buff = (char *)buff + i * real_seg_size;
            context->frag_id = i;
            context->peer = tree->tree_prev;
            context->con = con;
            OBJ_RETAIN(con);
            char* recv_buff = context->buff;
            OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: Recv(start in main): segment %d from %d at buff %p recv_count %d tag %d\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)recv_buff, recv_count, (ibcast_tag << 16) + i));
            err = MCA_PML_CALL(irecv(recv_buff, recv_count, datatype, context->peer, (ibcast_tag << 16) + i, comm, &recv_req));
            if (MPI_SUCCESS != err) {
                return err;
            }
            //invoke receive call back
            OPAL_THREAD_UNLOCK(mutex);
            ompi_request_set_callback(recv_req, recv_cb, context);
            OPAL_THREAD_LOCK(mutex);
        }
        
    }
    
    OPAL_THREAD_UNLOCK(mutex);
    
    OPAL_OUTPUT_VERBOSE((30, mca_coll_adapt_component.adapt_output, "[%d, %" PRIx64 "]: End of Ibcast\n", rank, gettid()));
    
    return MPI_SUCCESS;
}


