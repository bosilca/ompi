//TODO: change tag to 3 bits
//TODO: add sent part in root use sent array to acheive always send the next one
//TODO: move receve before send
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


#define SEND_NUM 2    //send how many fragments at once
#define RECV_NUM 3    //receive how many fragments at once
#define SEG_SIZE 163740   //size of a segment
#define FREE_LIST_NUM 10    //The start size of the free list
#define FREE_LIST_MAX 10000  //The max size of the free list
#define FREE_LIST_INC 10    //The incresment of the free list
#define TEST printfno

//send call back
static int send_cb(ompi_request_t *req)
{
    mca_coll_adapt_bcast_context_t *context = (mca_coll_adapt_bcast_context_t *) req->req_complete_cb_data;
    
    //opal_output_init();
    //mca_pml_ob1_dump(context->con->comm, 0);
    //opal_output_finalize();
    
    int err;
    
    TEST("[%d, %" PRIx64 "]: Send(cb): segment %d to %d at buff %p \n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff);

    OPAL_THREAD_LOCK(context->con->mutex);
    int sent_id = context->con->send_array[context->child_id];
    //has fragments in recv_array can be sent
    if (sent_id < context->con->num_recv_segs) {
        ompi_request_t *send_req;
        int new_id = context->con->recv_array[sent_id];
        mca_coll_adapt_bcast_context_t * send_context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(context->con->context_list);
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
        TEST("[%d]: Send(start in send cb): segment %d to %d at buff %p send_count %d\n", ompi_comm_rank(send_context->con->comm), send_context->frag_id, send_context->peer, (void *)send_context->buff, send_count);
        err = MCA_PML_CALL(isend(send_context->buff, send_count, send_context->con->datatype, send_context->peer, new_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
        //invoke send call back
        OPAL_THREAD_UNLOCK(context->con->mutex);
        ompi_request_set_callback(send_req, send_cb, send_context);
        OPAL_THREAD_LOCK(context->con->mutex);
    }
    
    int num_sent = ++(context->con->num_sent_segs);
    int num_recv_fini_t = context->con->num_recv_fini;
    int rank = ompi_comm_rank(context->con->comm);
    opal_free_list_t * temp = context->con->context_list;
    opal_mutex_t * mutex_temp = context->con->mutex;
    //check whether signal the condition
    if ((rank == context->con->root && num_sent == context->con->tree->tree_nextsize * context->con->num_segs) ||
        (context->con->tree->tree_nextsize > 0 && rank != context->con->root && num_sent == context->con->tree->tree_nextsize * context->con->num_segs && num_recv_fini_t == context->con->num_segs) ||
        (context->con->tree->tree_nextsize == 0 && num_recv_fini_t == context->con->num_segs)) {
        TEST("[%d]: Singal in send\n", ompi_comm_rank(context->con->comm));
        OPAL_THREAD_UNLOCK(mutex_temp);
        ompi_request_t *temp_req = context->con->request;
        if (context->con->tree->tree_nextsize != 0) {
            free(context->con->send_array);
        }
        if (context->con->num_segs !=0) {
            free(context->con->recv_array);
        }
        ompi_coll_base_topo_destroy_tree(&(context->con->tree));
        OBJ_RELEASE(context->con->mutex);
        OBJ_RELEASE(context->con);
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        OBJ_RELEASE(temp);
        ompi_request_complete(temp_req, 1);
    }
    else {
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        OPAL_THREAD_UNLOCK(mutex_temp);
    }
    OPAL_THREAD_UNLOCK(req->req_lock);
    req->req_free(&req);
    return 1;
}

//receive call back
static int recv_cb(ompi_request_t *req){
    //get necessary info from request
    mca_coll_adapt_bcast_context_t *context = (mca_coll_adapt_bcast_context_t *) req->req_complete_cb_data;
    
    int err, i;
    
    TEST("[%d, %" PRIx64 "]: Recv(cb): segment %d from %d at buff %p\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff);
    
    //store the frag_id to seg array
    OPAL_THREAD_LOCK(context->con->mutex);
    int num_recv_segs_t = ++(context->con->num_recv_segs);
    context->con->recv_array[num_recv_segs_t-1] = context->frag_id;
    
    int new_id = num_recv_segs_t + RECV_NUM - 1;
    //receive new segment
    if (new_id < context->con->num_segs) {
        ompi_request_t *recv_req;
        //get new context item from free list
        mca_coll_adapt_bcast_context_t * recv_context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(context->con->context_list);
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
        TEST("[%d]: Recv(start in recv cb): segment %d from %d at buff %p recv_count %d\n", ompi_comm_rank(context->con->comm), context->frag_id, context->peer, (void *)context->buff, recv_count);
        MCA_PML_CALL(irecv(recv_context->buff, recv_count, recv_context->con->datatype, recv_context->peer, recv_context->frag_id, recv_context->con->comm, &recv_req));
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
            
            mca_coll_adapt_bcast_context_t * send_context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(context->con->context_list);
            send_context->buff = context->buff;
            send_context->frag_id = context->frag_id;
            send_context->child_id = i;
            send_context->peer = context->con->tree->tree_next[i];
            send_context->con = context->con;
            OBJ_RETAIN(context->con);
            ++(send_context->con->send_array[i]);
            TEST("[%d]: Send(start in recv cb): segment %d to %d at buff %p send_count %d\n", ompi_comm_rank(send_context->con->comm), send_context->frag_id, send_context->peer, (void *)send_context->buff, send_count);
            err = MCA_PML_CALL(isend(send_context->buff, send_count, send_context->con->datatype, send_context->peer, send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
            //invoke send call back
            OPAL_THREAD_UNLOCK(context->con->mutex);
            ompi_request_set_callback(send_req, send_cb, send_context);
            OPAL_THREAD_LOCK(context->con->mutex);
        }
    }
    
    int num_sent = context->con->num_sent_segs;
    int num_recv_fini_t = ++(context->con->num_recv_fini);
    int rank = ompi_comm_rank(context->con->comm);
    opal_free_list_t * temp = context->con->context_list;
    opal_mutex_t * mutex_temp = context->con->mutex;

    //if this is leaf and has received all the segments
    if ((rank == context->con->root && num_sent == context->con->tree->tree_nextsize * context->con->num_segs) ||
        (context->con->tree->tree_nextsize > 0 && rank != context->con->root && num_sent == context->con->tree->tree_nextsize * context->con->num_segs && num_recv_fini_t == context->con->num_segs) ||
        (context->con->tree->tree_nextsize == 0 && num_recv_fini_t == context->con->num_segs)) {
        TEST("[%d]: Singal in recv\n", ompi_comm_rank(context->con->comm));
        OPAL_THREAD_UNLOCK(mutex_temp);
        ompi_request_t *temp_req = context->con->request;
        if (context->con->tree->tree_nextsize != 0) {
            free(context->con->send_array);
        }
        if (context->con->num_segs !=0) {
            free(context->con->recv_array);
        }
        ompi_coll_base_topo_destroy_tree(&(context->con->tree));
        OBJ_RELEASE(context->con->mutex);
        OBJ_RELEASE(context->con);
        OBJ_RELEASE(context->con);
        OBJ_RELEASE(temp);
        ompi_request_complete(temp_req, 1);
    }
    else{
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        OPAL_THREAD_UNLOCK(mutex_temp);
    }
    OPAL_THREAD_UNLOCK(req->req_lock);
    req->req_free(&req);
    return 1;
}

int mca_coll_adapt_ibcast(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
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
        return mca_coll_adapt_ibcast_binomial(buff, count, datatype, root, comm, request, module);
    }
}

//non-blocking broadcast using binomial tree with pipeline
int mca_coll_adapt_ibcast_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_bmtree) && (coll_comm->cached_bmtree_root == root) ) ) {
        if( coll_comm->cached_bmtree ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_bmtree) );
        }
        coll_comm->cached_bmtree = ompi_coll_base_topo_build_bmtree(comm, root);
        coll_comm->cached_bmtree_root = root;
    }
    //print_tree(coll_comm->cached_bmtree, ompi_comm_rank(comm));
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, coll_comm->cached_bmtree);

}

int mca_coll_adapt_ibcast_in_order_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_in_order_bmtree) && (coll_comm->cached_in_order_bmtree_root == root) ) ) {
        if( coll_comm->cached_in_order_bmtree ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_in_order_bmtree) );
        }
        coll_comm->cached_in_order_bmtree = ompi_coll_base_topo_build_in_order_bmtree(comm, root);
        coll_comm->cached_in_order_bmtree_root = root;
    }
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, coll_comm->cached_in_order_bmtree);
}


int mca_coll_adapt_ibcast_bininary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_bintree) && (coll_comm->cached_bintree_root == root) ) ) {
        if( coll_comm->cached_bintree ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_bintree) );
        }
        coll_comm->cached_bintree = ompi_coll_base_topo_build_tree(2, comm, root);
        coll_comm->cached_bintree_root = root;
    }
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, coll_comm->cached_bintree);
}

int mca_coll_adapt_ibcast_pipeline(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_pipeline) && (coll_comm->cached_pipeline_root == root) ) ) {
        if( coll_comm->cached_pipeline ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_pipeline) );
        }
        coll_comm->cached_pipeline = ompi_coll_base_topo_build_chain(1, comm, root);
        coll_comm->cached_pipeline_root = root;
    }
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, coll_comm->cached_pipeline);
}

int mca_coll_adapt_ibcast_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_chain) && (coll_comm->cached_chain_root == root) ) ) {
        if( coll_comm->cached_chain ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_chain) );
        }
        coll_comm->cached_chain = ompi_coll_base_topo_build_chain(4, comm, root);
        coll_comm->cached_chain_root = root;
    }
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, coll_comm->cached_chain);
}

int mca_coll_adapt_ibcast_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_linear) && (coll_comm->cached_linear_root == root) ) ) {
        if( coll_comm->cached_linear ) { /* destroy previous tree if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_linear) );
        }
        int fanout = ompi_comm_size(comm) - 1;
        ompi_coll_tree_t * tree;
        if (fanout > 1) {
            tree = ompi_coll_base_topo_build_tree(ompi_comm_size(comm) - 1, comm, root);
        }
        else{
            tree = ompi_coll_base_topo_build_chain(1, comm, root);
        }
        coll_comm->cached_linear = tree;
        coll_comm->cached_linear_root = root;
    }
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, coll_comm->cached_linear);

}

int mca_coll_adapt_ibcast_topoaware_linear(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_topolinear) && (coll_comm->cached_topolinear_root == root) ) ) {
        if( coll_comm->cached_topolinear ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_topolinear) );
        }
        coll_comm->cached_topolinear = ompi_coll_base_topo_build_topoaware_linear(comm, root, module);
        coll_comm->cached_topolinear_root = root;
    }
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, coll_comm->cached_topolinear);
}

int mca_coll_adapt_ibcast_topoaware_chain(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    mca_coll_base_comm_t *coll_comm = module->base_data;
    if( !( (coll_comm->cached_topochain) && (coll_comm->cached_topochain_root == root) ) ) {
        if( coll_comm->cached_topochain ) { /* destroy previous binomial if defined */
            ompi_coll_base_topo_destroy_tree( &(coll_comm->cached_topochain) );
        }
        coll_comm->cached_topochain = ompi_coll_base_topo_build_topoaware_chain(comm, root, module);
        coll_comm->cached_topochain_root = root;
    }
    else {
    }
    //print_tree(coll_comm->cached_topochain, ompi_comm_rank(comm));
    return mca_coll_adapt_ibcast_generic(buff, count, datatype, root, comm, request, module, coll_comm->cached_topochain);
}


int mca_coll_adapt_ibcast_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t* tree){
    int i, j;       //temp variable for iteration
    int size;       //size of the communicator
    int rank;       //rank of this node
    int err;        //record return value
    int min;        //the min of num_segs and SEND_NUM or RECV_NUM, in case the num_segs is less than SEND_NUM or RECV_NUM
    
    size_t seg_size;            //the size of a segment
    int seg_count = count;      //number of datatype in a segment
    size_t type_size;           //the size of a datatype
    size_t real_seg_size;       //the real size of a segment
    ptrdiff_t extent, lb;
    int num_segs;               //the number of segments
    
    opal_free_list_t * context_list; //a free list contain all the context of call backs
    ompi_request_t * temp_request = NULL;  //the request be passed outside
    opal_mutex_t * mutex;
    int *recv_array = NULL;   //store those segments which are received
    int *send_array = NULL;   //record how many isend has been issued for every child
    
    //set up free list
    context_list = OBJ_NEW(opal_free_list_t);
    opal_free_list_init(context_list,
                        sizeof(mca_coll_adapt_bcast_context_t),
                        opal_cache_line_size,
                        OBJ_CLASS(mca_coll_adapt_bcast_context_t),
                        0,opal_cache_line_size,
                        FREE_LIST_NUM,
                        FREE_LIST_MAX,
                        FREE_LIST_INC,
                        NULL, 0, NULL, NULL, NULL);
    
    //set up request
    temp_request = OBJ_NEW(ompi_request_t);
    OMPI_REQUEST_INIT(temp_request, false);
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
    
    seg_size = SEG_SIZE;
    size = ompi_comm_size(comm);
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
    con->context_list = context_list;
    con->recv_array = recv_array;
    con->num_recv_segs = 0;
    con->num_recv_fini = 0;
    con->send_array = send_array;
    con->num_sent_segs = 0;
    con->mutex = mutex;
    con->request = temp_request;
    con->tree = tree;
    
    TEST("[%d, %" PRIx64 "]: Ibcast, root %d\n", rank, gettid(), root);
    TEST("[%d, %" PRIx64 "]: con->mutex = %p, num_children = %d, num_segs = %d, real_seg_size = %d, seg_count = %d, tree_adreess = %p\n", rank, gettid(), (void *)con->mutex, tree->tree_nextsize, num_segs, (int)real_seg_size, seg_count, (void *)con->tree);
    
    OPAL_THREAD_LOCK(mutex);
    
    //if root, send segment to every children.
    if (rank == root){
        //handle the situation when num_segs < SEND_NUM
        if (num_segs <= SEND_NUM) {
            min = num_segs;
        }
        else{
            min = SEND_NUM;
        }
        
        //set recv_array, root has already had all the segments
        for (i = 0; i < num_segs; i++) {
            recv_array[i] = i;
        }
        con->num_recv_segs = num_segs;
        //set send_array, has not sent any segments
        for (i = 0; i < tree->tree_nextsize; i++) {
            send_array[i] = SEND_NUM;
        }
        
        ompi_request_t *send_req;
        int send_count = seg_count;             //number of datatype in each send
        for (i = 0; i < min; i++) {
            if (i == (num_segs - 1)) {
                send_count = count - i * seg_count;
            }
            for (j=0; j<tree->tree_nextsize; j++) {
                mca_coll_adapt_bcast_context_t * context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(context_list);
                context->buff = (char *)buff + i * real_seg_size;
                context->frag_id = i;
                context->child_id = j;              //the id of peer in in children_list
                context->peer = tree->tree_next[j];   //the actural rank of the peer
                context->con = con;
                OBJ_RETAIN(con);
                TEST("[%d, %" PRIx64 "]: Send(start in main): segment %d to %d at buff %p send_count %d\n", rank, gettid(), context->frag_id, context->peer, (void *)context->buff, send_count);
                err = MCA_PML_CALL(isend(context->buff, send_count, datatype, context->peer, i, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req));
                
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
        if (num_segs <= RECV_NUM) {
            min = num_segs;
        }
        else{
            min = RECV_NUM;
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
            mca_coll_adapt_bcast_context_t * context = (mca_coll_adapt_bcast_context_t *) opal_free_list_wait(context_list);
            context->buff = (char *)buff + i * real_seg_size;
            context->frag_id = i;
            context->peer = tree->tree_prev;
            context->con = con;
            OBJ_RETAIN(con);
            TEST("[%d, %" PRIx64 "]: Recv(start in main): segment %d from %d at buff %p recv_count %d\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff, recv_count);
            err = MCA_PML_CALL(irecv(context->buff, recv_count, datatype, context->peer, i, comm, &recv_req));
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
    
    TEST("[%d, %" PRIx64 "]: End of Ibcast\n", rank, gettid());
    
    return MPI_SUCCESS;
}

int mca_coll_adapt_ibcast_two_trees_binary(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    size_t type_size;                       //the size of a datatype
    size_t seg_size = SEG_SIZE;            //the size of a segment
    int seg_count = count;      //number of datatype in a segment
    ompi_datatype_type_size(datatype, &type_size);
    COLL_BASE_COMPUTED_SEGCOUNT(seg_size, type_size, seg_count);
    int total_num_segs = (count + seg_count - 1) / seg_count;
    int size = ompi_comm_size(comm);
    if (total_num_segs > 1 && size >= 3) {
        mca_coll_base_comm_t *coll_comm = module->base_data;
        if( !( (coll_comm->cached_two_trees_binary) && (coll_comm->cached_two_trees_binary_root == root) ) ) {
            if( coll_comm->cached_two_trees_binary ) { /* destroy previous binomial if defined */
                ompi_coll_base_topo_destroy_two_trees(coll_comm->cached_two_trees_binary);
            }
            coll_comm->cached_two_trees_binary = ompi_coll_base_topo_build_two_trees_binary(comm, root);
            coll_comm->cached_two_trees_binary_root = root;
            //print_tree(two_trees[0], ompi_comm_rank(comm));
            //print_tree(two_trees[1], ompi_comm_rank(comm));
        }
        return mca_coll_adapt_ibcast_two_trees_generic(buff, count, datatype, root, comm, request, module, coll_comm->cached_two_trees_binary);
    }
    else{
        return mca_coll_adapt_ibcast_binary(buff, count, datatype, root, comm, request, module);
    }
}

int mca_coll_adapt_ibcast_two_trees_binomial(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    size_t type_size;                       //the size of a datatype
    size_t seg_size = SEG_SIZE;            //the size of a segment
    int seg_count = count;      //number of datatype in a segment
    ompi_datatype_type_size(datatype, &type_size);
    COLL_BASE_COMPUTED_SEGCOUNT(seg_size, type_size, seg_count);
    int total_num_segs = (count + seg_count - 1) / seg_count;
    int size = ompi_comm_size(comm);
    if (total_num_segs > 1 && size >= 3) {
        mca_coll_base_comm_t *coll_comm = module->base_data;
        if( !( (coll_comm->cached_two_trees_binomial) && (coll_comm->cached_two_trees_binomial_root == root) ) ) {
            if( coll_comm->cached_two_trees_binomial ) { /* destroy previous binomial if defined */
                ompi_coll_base_topo_destroy_two_trees(coll_comm->cached_two_trees_binomial);
            }
            coll_comm->cached_two_trees_binomial = ompi_coll_base_topo_build_two_trees_binomial(comm, root);
            coll_comm->cached_two_trees_binomial_root = root;
            //print_tree(two_trees[0], ompi_comm_rank(comm));
            //print_tree(two_trees[1], ompi_comm_rank(comm));
        }
        return mca_coll_adapt_ibcast_two_trees_generic(buff, count, datatype, root, comm, request, module, coll_comm->cached_two_trees_binomial);
    }
    else{
        return mca_coll_adapt_ibcast_binomial(buff, count, datatype, root, comm, request, module);
    }
}

int mca_coll_adapt_ibcast_two_chains(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module){
    size_t type_size;                       //the size of a datatype
    size_t seg_size = SEG_SIZE;            //the size of a segment
    int seg_count = count;      //number of datatype in a segment
    ompi_datatype_type_size(datatype, &type_size);
    COLL_BASE_COMPUTED_SEGCOUNT(seg_size, type_size, seg_count);
    int total_num_segs = (count + seg_count - 1) / seg_count;
    int size = ompi_comm_size(comm);
    if (total_num_segs > 1 && size >= 3) {
        mca_coll_base_comm_t *coll_comm = module->base_data;
        if( !( (coll_comm->cached_two_chains) && (coll_comm->cached_two_chains_root == root) ) ) {
            if( coll_comm->cached_two_chains ) { /* destroy previous binomial if defined */
                ompi_coll_base_topo_destroy_two_trees(coll_comm->cached_two_chains);
            }
            coll_comm->cached_two_chains = ompi_coll_base_topo_build_two_chains(comm, root);
            coll_comm->cached_two_chains_root = root;
            //print_tree(two_trees[0], ompi_comm_rank(comm));
            //print_tree(two_trees[1], ompi_comm_rank(comm));
        }
        return mca_coll_adapt_ibcast_two_trees_generic(buff, count, datatype, root, comm, request, module, coll_comm->cached_two_chains);
    }
    else{
        return mca_coll_adapt_ibcast_binomial(buff, count, datatype, root, comm, request, module);
    }
}



//send call back
static int two_trees_send_cb(ompi_request_t *req)
{
    mca_coll_adapt_bcast_two_trees_context_t *context = (mca_coll_adapt_bcast_two_trees_context_t *) req->req_complete_cb_data;
    
    int err;
    
    TEST("[%d, %" PRIx64 "]: Send(cb): segment %d to %d at buff %p tree %d\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff, context->tree);
    
    OPAL_THREAD_LOCK(context->con->mutex);
    int sent_id = context->con->send_arrays[context->tree][context->child_id];  //How many sends has been issued
    int num_sent = ++(context->con->num_sent_segs[context->tree]);
    //has fragments in recv_array can be sent
    if (sent_id < context->con->num_recv_segs[context->tree]) {
        ompi_request_t *send_req;
        int new_id = context->con->recv_arrays[context->tree][sent_id];
        int send_count = context->con->seg_count;
        if (new_id == (context->con->num_segs[0] + context->con->num_segs[1]- 1)) {
            send_count = context->con->count - new_id * context->con->seg_count;
        }
        mca_coll_adapt_bcast_two_trees_context_t * send_context = (mca_coll_adapt_bcast_two_trees_context_t *) opal_free_list_wait(context->con->context_lists[context->tree]);
        send_context->buff = context->buff + (new_id - context->frag_id) * context->con->real_seg_size;
        send_context->frag_id = new_id;
        send_context->child_id = context->child_id;
        send_context->peer = context->peer;
        send_context->tree = context->tree;
        send_context->con = context->con;
        OBJ_RETAIN(context->con);
        ++(send_context->con->send_arrays[context->tree][send_context->child_id]);
        TEST("[%d]: Send(start in send cb): segment %d to %d at buff %p send_count %d dataype %p tree %d\n", ompi_comm_rank(send_context->con->comm), send_context->frag_id, send_context->peer, (void *)send_context->buff, send_count, (void *)send_context->con->datatype, context->tree);
        err = MCA_PML_CALL(isend(send_context->buff, send_count, send_context->con->datatype, send_context->peer, new_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
        //invoke send call back
        if(!ompi_request_set_callback(send_req, two_trees_send_cb, send_context)) {
            OPAL_THREAD_UNLOCK(context->con->mutex);
            int finished = two_trees_send_cb(send_req);
            if (finished) {
                return 1;
            }
            OPAL_THREAD_LOCK(context->con->mutex);
        }
    }
    OPAL_THREAD_UNLOCK(context->con->mutex);
    
    //check whether signal the condition, need to signal after return the context
    if (num_sent == context->con->trees[context->tree]->tree_nextsize * context->con->num_segs[context->tree]) {
        int complete = ++(context->con->complete);       //TODO:change to atomic add
        opal_free_list_t * temp = context->con->context_lists[context->tree];
        if (complete >= 2) {
            TEST("[%d]: Singal in send, tree %d\n", ompi_comm_rank(context->con->comm), context->tree);
            ompi_request_t *temp_req = context->con->request;
            if (context->con->num_segs[0]!=0) {
                free(context->con->recv_arrays[0]);
            }
            if (context->con->trees[0]->tree_nextsize!=0) {
                free(context->con->send_arrays[0]);
            }
            if (context->con->num_segs[1]!=0) {
                free(context->con->recv_arrays[1]);
            }
            if (context->con->trees[1]->tree_nextsize!=0) {
                free(context->con->send_arrays[1]);
            }
            free(context->con->send_arrays);
            free(context->con->recv_arrays);
            free(context->con->num_sent_segs);
            free(context->con->num_recv_segs);
            free(context->con->num_segs);
            ompi_coll_base_topo_destroy_two_trees(context->con->trees);
            OBJ_RELEASE(context->con->mutex);
            OBJ_RELEASE(context->con->context_lists[1-context->tree]);
            free(context->con->context_lists);
            OBJ_RELEASE(context->con);
            OBJ_RELEASE(context->con);
            opal_free_list_return(temp, (opal_free_list_item_t*)context);
            OBJ_RELEASE(temp);
            ompi_request_complete(temp_req, 1);
        }
        else {
            OBJ_RELEASE(context->con);
            opal_free_list_return(temp, (opal_free_list_item_t*)context);
        }
        return 1;
    }
    else{
        opal_free_list_t * temp = context->con->context_lists[context->tree];
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
    }
    return 0;
}

//receive call back
static int two_trees_recv_cb(ompi_request_t *req){
    //get necessary info from request
    mca_coll_adapt_bcast_two_trees_context_t *context = (mca_coll_adapt_bcast_two_trees_context_t *) req->req_complete_cb_data;
    
    int err, i;
    
    TEST("[%d, %" PRIx64 "]: Recv(cb): segment %d from %d at buff %p tree %d\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff, context->tree);
    
    //store the frag_id to seg array
    OPAL_THREAD_LOCK(context->con->mutex);
    int num_recv_segs_t = ++(context->con->num_recv_segs[context->tree]);
    context->con->recv_arrays[context->tree][num_recv_segs_t-1] = context->frag_id;
    
    int new_position = num_recv_segs_t + RECV_NUM - 1;
    //receive new segment
    if (new_position < context->con->num_segs[context->tree]) {
        ompi_request_t *recv_req;
        int new_id;
        if (context->tree == 0) {
            new_id = new_position;
        }
        else{
            new_id = new_position + context->con->num_segs[0];
        }
        int recv_count = context->con->seg_count;
        if (new_id == (context->con->num_segs[0] + context->con->num_segs[1] - 1)) {
            recv_count = context->con->count - new_id * context->con->seg_count;
        }
        //get new context item from free list
        mca_coll_adapt_bcast_two_trees_context_t * recv_context = (mca_coll_adapt_bcast_two_trees_context_t *) opal_free_list_wait(context->con->context_lists[context->tree]);
        recv_context->buff = context->buff + (new_id - context->frag_id) * context->con->real_seg_size;
        recv_context->frag_id = new_id;
        recv_context->child_id = context->child_id;
        recv_context->peer = context->peer;
        recv_context->tree = context->tree;
        recv_context->con = context->con;
        OBJ_RETAIN(context->con);
        TEST("[%d]: Recv(start in recv cb): segment %d from %d at buff %p recv_count %d datatype %p tree %d\n", ompi_comm_rank(recv_context->con->comm), recv_context->frag_id, recv_context->peer, (void *)recv_context->buff, recv_count, (void *)recv_context->con->datatype, context->tree);
        MCA_PML_CALL(irecv(recv_context->buff, recv_count, recv_context->con->datatype, recv_context->peer, recv_context->frag_id, recv_context->con->comm, &recv_req));
        //invoke recvive call back
        if(!ompi_request_set_callback(recv_req, two_trees_recv_cb, recv_context)) {
            OPAL_THREAD_UNLOCK(context->con->mutex);
            int finished = two_trees_recv_cb(recv_req);
            if (finished) {
                return 1;
            }
            OPAL_THREAD_LOCK(context->con->mutex);
        }
    }
    
    //send segment to its children
    for (i = 0; i < context->con->trees[context->tree]->tree_nextsize; i++) {
        //if can send the segment now means the only segment need to be sent is the just arrived one
        if (num_recv_segs_t-1 == context->con->send_arrays[context->tree][i]) {
            ompi_request_t *send_req;
            int send_count = context->con->seg_count;
            if (context->frag_id == (context->con->num_segs[0] + context->con->num_segs[1] - 1)) {
                send_count = context->con->count - context->frag_id * context->con->seg_count;
            }
            mca_coll_adapt_bcast_two_trees_context_t * send_context = (mca_coll_adapt_bcast_two_trees_context_t *) opal_free_list_wait(context->con->context_lists[context->tree]);
            send_context->buff = context->buff;
            send_context->frag_id = context->frag_id;
            send_context->child_id = i;
            send_context->peer = context->con->trees[context->tree]->tree_next[i];
            send_context->tree = context->tree;
            send_context->con = context->con;
            OBJ_RETAIN(context->con);
            ++(send_context->con->send_arrays[context->tree][i]);
            TEST("[%d]: Send(start in recv cb): segment %d to %d at buff %p send_count %d datatype %p comm %p tree %d\n", ompi_comm_rank(send_context->con->comm), send_context->frag_id, send_context->peer, (void *)send_context->buff, send_count, (void *) send_context->con->datatype, (void *) send_context->con->comm, context->tree);
            err = MCA_PML_CALL(isend(send_context->buff, send_count, send_context->con->datatype, send_context->peer, send_context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, send_context->con->comm, &send_req));
            
            //invoke send call back
            if(!ompi_request_set_callback(send_req, two_trees_send_cb, send_context)) {
                OPAL_THREAD_UNLOCK(context->con->mutex);
                int finished = two_trees_send_cb(send_req);
                if (finished) {
                    return 1;
                }
                OPAL_THREAD_LOCK(context->con->mutex);
            }
        }
    }
    OPAL_THREAD_UNLOCK(context->con->mutex);
    
    
    
    //if this is leaf and has received all the segments
    if (context->con->trees[context->tree]->tree_nextsize == 0 && num_recv_segs_t == context->con->num_segs[context->tree]) {
        int complete = ++(context->con->complete);       //TODO:change to atomic add
        opal_free_list_t * temp = context->con->context_lists[context->tree];
        if (complete >= 2) {
            TEST("[%d]: Singal in recv, tree %d\n", ompi_comm_rank(context->con->comm), context->tree);
            ompi_request_t *temp_req = context->con->request;
            if (context->con->num_segs[0]!=0) {
                free(context->con->recv_arrays[0]);
            }
            if (context->con->trees[0]->tree_nextsize!=0) {
                free(context->con->send_arrays[0]);
            }
            if (context->con->num_segs[1]!=0) {
                free(context->con->recv_arrays[1]);
            }
            if (context->con->trees[1]->tree_nextsize!=0) {
                free(context->con->send_arrays[1]);
            }
            free(context->con->send_arrays);
            free(context->con->recv_arrays);
            free(context->con->num_sent_segs);
            free(context->con->num_recv_segs);
            free(context->con->num_segs);
            ompi_coll_base_topo_destroy_two_trees(context->con->trees);
            OBJ_RELEASE(context->con->mutex);
            OBJ_RELEASE(context->con->context_lists[1-context->tree]);
            free(context->con->context_lists);
            OBJ_RELEASE(context->con);
            OBJ_RELEASE(context->con);
            opal_free_list_return(temp, (opal_free_list_item_t*)context);
            OBJ_RELEASE(temp);
            ompi_request_complete(temp_req, 1);
        }
        else {
            OBJ_RELEASE(context->con);
            opal_free_list_return(temp, (opal_free_list_item_t*)context);
        }
        return 1;
    }
    else{
        opal_free_list_t * temp = context->con->context_lists[context->tree];
        OBJ_RELEASE(context->con);
        opal_free_list_return(temp, (opal_free_list_item_t*)context);
        
    }
    return 0;
}


int mca_coll_adapt_ibcast_two_trees_generic(void *buff, int count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request, mca_coll_base_module_t *module, ompi_coll_tree_t** trees){
    int i, j, t;       //temp variable for iteration
    int size;       //size of the communicator
    int rank;       //rank of this node
    int err;        //record return value
    int min[2];        //the min of num_segs and SEND_NUM or RECV_NUM, in case the num_segs is less than SEND_NUM or RECV_NUM
    
    size_t seg_size;            //the size of a segment
    int seg_count = count;      //number of datatype in a segment
    size_t type_size;           //the size of a datatype
    size_t real_seg_size;       //the real size of a segment
    ptrdiff_t extent, lb;
    int total_num_segs;               //the total number of segments
    int * num_segs;               //the number of segments for tree 0 and tree 1
    int * num_recv_segs;
    int * num_sent_segs;
    
    opal_free_list_t ** context_lists; //two free lists contain all the context of call backs
    ompi_request_t * temp_request = NULL;  //the request be passed outside
    opal_mutex_t * mutex;
    int **recv_arrays = NULL;   //store those segments which are received for two trees
    int **send_arrays = NULL;   //record how many isend has been issued for every child for two trees
    
    //set up free list
    context_lists = (opal_free_list_t **)malloc(sizeof(opal_free_list_t *)*2);
    context_lists[0] = OBJ_NEW(opal_free_list_t);
    opal_free_list_init(context_lists[0],
                        sizeof(mca_coll_adapt_bcast_two_trees_context_t),
                        opal_cache_line_size,
                        OBJ_CLASS(mca_coll_adapt_bcast_two_trees_context_t),
                        0,opal_cache_line_size,
                        FREE_LIST_NUM,
                        FREE_LIST_MAX,
                        FREE_LIST_INC,
                        NULL, 0, NULL, NULL, NULL);
    context_lists[1] = OBJ_NEW(opal_free_list_t);
    opal_free_list_init(context_lists[1],
                        sizeof(mca_coll_adapt_bcast_two_trees_context_t),
                        opal_cache_line_size,
                        OBJ_CLASS(mca_coll_adapt_bcast_two_trees_context_t),
                        0,opal_cache_line_size,
                        FREE_LIST_NUM,
                        FREE_LIST_MAX,
                        FREE_LIST_INC,
                        NULL, 0, NULL, NULL, NULL);
    
    //set up request
    temp_request = OBJ_NEW(ompi_request_t);
    OMPI_REQUEST_INIT(temp_request, false);
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
    
    seg_size = SEG_SIZE;
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    
    //Determine number of elements sent per operation
    ompi_datatype_type_size(datatype, &type_size);
    COLL_BASE_COMPUTED_SEGCOUNT(seg_size, type_size, seg_count);
    
    ompi_datatype_get_extent(datatype, &lb, &extent);
    total_num_segs = (count + seg_count - 1) / seg_count;
    real_seg_size = (ptrdiff_t)seg_count * extent;
    num_segs = (int *)malloc(sizeof(int) * 2);
    num_segs[0] = total_num_segs / 2;
    num_segs[1] = total_num_segs - num_segs[0];
    num_recv_segs = (int *)malloc(sizeof(int) * 2);
    num_recv_segs[0] = 0;
    num_recv_segs[1] = 0;
    num_sent_segs = (int *)malloc(sizeof(int) * 2);
    num_sent_segs[0] = 0;
    num_sent_segs[1] = 0;
    
    //set memory for recv_array and send_array, created on heap becasue they are needed to be accessed by other functions, callback function
    recv_arrays = (int **)malloc(sizeof(int *) * 2);
    send_arrays = (int **)malloc(sizeof(int *) * 2);
    if (num_segs[0]!=0) {
        recv_arrays[0] = (int *)malloc(sizeof(int) * num_segs[0]);
    }
    if (trees[0]->tree_nextsize!=0) {
        send_arrays[0] = (int *)malloc(sizeof(int) * trees[0]->tree_nextsize);
    }
    if (num_segs[1]!=0) {
        recv_arrays[1] = (int *)malloc(sizeof(int) * num_segs[1]);
    }
    if (trees[1]->tree_nextsize!=0) {
        send_arrays[1] = (int *)malloc(sizeof(int) * trees[1]->tree_nextsize);
    }
    
    //Set constant context for send and recv call back
    mca_coll_adapt_constant_bcast_two_trees_context_t *con = OBJ_NEW(mca_coll_adapt_constant_bcast_two_trees_context_t);
    con->count = count;
    con->seg_count = seg_count;
    con->datatype = datatype;
    con->comm = comm;
    con->real_seg_size = real_seg_size;
    con->num_segs = num_segs;
    con->context_lists = context_lists;
    con->recv_arrays = recv_arrays;
    con->num_recv_segs = num_recv_segs;
    con->send_arrays = send_arrays;
    con->num_sent_segs = num_sent_segs;
    con->mutex = mutex;
    con->request = temp_request;
    con->trees = trees;
    con->complete = 0;
    
    OPAL_THREAD_LOCK(mutex);
    TEST("[%d, %" PRIx64 "]: IBcast, root %d\n", rank, gettid(), root);
    TEST("[%d, %" PRIx64 "]: con->mutex = %p\n", rank, gettid(), (void *)con->mutex);
    //if root, send segment to the roots of two trees.
    if (rank == root){
        //handle the situation when num_segs < SEND_NUM
        if (num_segs[0] < SEND_NUM) {
            min[0] = num_segs[0];
        }
        else{
            min[0] = SEND_NUM;
        }
        if (num_segs[1] < SEND_NUM) {
            min[1] = num_segs[1];
        }
        else{
            min[1] = SEND_NUM;
        }
        
        //set recv_array and num_recv_segs, root has already had all the segments
        for (i = 0; i < total_num_segs; i++) {
            if (i < num_segs[0]) {
                recv_arrays[0][i] = i;
            }
            else{
                recv_arrays[1][i-num_segs[0]] = i;
            }
        }
        con->num_recv_segs[0] = num_segs[0];
        con->num_recv_segs[1] = num_segs[1];
        
        //set send_array
        for (i = 0; i < trees[0]->tree_nextsize; i++) {
            send_arrays[0][i] = min[0];
        }
        for (i = 0; i < trees[1]->tree_nextsize; i++) {
            send_arrays[1][i] = min[1];
        }
        
        for (t = 0; t < 2; t++) {
            ompi_request_t *send_req;
            int send_count = seg_count;             //number of datatype in each send
            for (i = 0; i < min[t]; i++) {
                if (t == 1 && i == (num_segs[1] - 1)) {
                    send_count = count - (num_segs[0]+i) * seg_count;
                }
                for (j=0; j<trees[t]->tree_nextsize; j++) {
                    mca_coll_adapt_bcast_two_trees_context_t * context = (mca_coll_adapt_bcast_two_trees_context_t *) opal_free_list_wait(context_lists[t]);
                    context->buff = (char *)buff + recv_arrays[t][i] * real_seg_size;
                    context->frag_id = recv_arrays[t][i];
                    context->child_id = j;              //the id of peer in in tree->tree_next
                    context->peer = trees[t]->tree_next[j];   //the actural rank of the peer
                    context->tree = t;
                    context->con = con;
                    OBJ_RETAIN(con);
                    TEST("[%d, %" PRIx64 "]: Send(start in main): segment %d to %d at buff %p send_count %d datatype %p tree %d\n", rank, gettid(), context->frag_id, context->peer, (void *)context->buff, send_count, (void *)datatype, t);
                    err = MCA_PML_CALL(isend(context->buff, send_count, datatype, context->peer, context->frag_id, MCA_PML_BASE_SEND_SYNCHRONOUS, comm, &send_req));
                    
                    if (MPI_SUCCESS != err) {
                        return err;
                    }
                    //invoke send call back
                    if(!ompi_request_set_callback(send_req, two_trees_send_cb, context)) {
                        OPAL_THREAD_UNLOCK(mutex);
                        two_trees_send_cb(send_req);
                        OPAL_THREAD_LOCK(mutex);
                    }
                    
                }
            }
        }
        
    }
    
    //if not root, receive data from parent in the tree.
    else {
        //handle the situation when num_segs < RECV_NUM
        if (num_segs[0] < RECV_NUM) {
            min[0] = num_segs[0];
        }
        else{
            min[0] = RECV_NUM;
        }
        if (num_segs[1] < RECV_NUM) {
            min[1] = num_segs[1];
        }
        else{
            min[1] = RECV_NUM;
        }
        
        //set recv_array, recv_array is empty and num_recv_segs is 0
        for (i = 0; i < num_segs[0]; i++) {
            recv_arrays[0][i] = 0;
        }
        for (i = 0; i < num_segs[1]; i++) {
            recv_arrays[1][i] = 0;
        }
        con->num_recv_segs[0] = 0;
        con->num_recv_segs[1] = 0;
        //set send_array to empty
        for (i = 0; i < trees[0]->tree_nextsize; i++) {
            send_arrays[0][i] = 0;
        }
        for (i = 0; i < trees[1]->tree_nextsize; i++) {
            send_arrays[1][i] = 0;
        }
        
        //create a recv request
        ompi_request_t *recv_req;
        
        for (t = 0; t < 2; t++){
            //recevice some segments from its parent
            int recv_count = seg_count;
            for (i = 0; i < min[t]; i++) {
                if (t == 1 && i == (num_segs[1] - 1)) {
                    recv_count = count - (num_segs[0]+i) * seg_count;
                }
                mca_coll_adapt_bcast_two_trees_context_t * context = (mca_coll_adapt_bcast_two_trees_context_t *) opal_free_list_wait(context_lists[t]);
                context->buff = (char *)buff + (t*num_segs[0]+i) * real_seg_size;
                context->frag_id = t*num_segs[0]+i;
                context->peer = trees[t]->tree_prev;
                context->con = con;
                context->tree = t;
                OBJ_RETAIN(con);
                TEST("[%d, %" PRIx64 "]: Recv(start in main): segment %d from %d at buff %p recv_count %d datatype %p comm %p tree %d\n", ompi_comm_rank(context->con->comm), gettid(), context->frag_id, context->peer, (void *)context->buff, recv_count, (void *)datatype, (void *)comm, t);
                err = MCA_PML_CALL(irecv(context->buff, recv_count, datatype, context->peer, context->frag_id, comm, &recv_req));
                if (MPI_SUCCESS != err) {
                    return err;
                }
                //invoke receive call back
                if(!ompi_request_set_callback(recv_req, two_trees_recv_cb, context)) {
                    OPAL_THREAD_UNLOCK(mutex);
                    two_trees_recv_cb(recv_req);
                    OPAL_THREAD_LOCK(mutex);
                    
                }
            }
        }
        
    }
    
    TEST("[%d, %" PRIx64 "]: End of ibcast\n", rank, gettid());
    
    
    OPAL_THREAD_UNLOCK(mutex);
    
    
    return MPI_SUCCESS;
}



