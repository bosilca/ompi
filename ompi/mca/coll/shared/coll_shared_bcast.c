#include "coll_shared.h"

int mca_coll_shared_bcast_intra(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    mca_coll_shared_module_t *shared_module = (mca_coll_shared_module_t*) module;
    if (!shared_module->enabled) {
        ompi_coll_shared_lazy_enable(module, comm);
    }
    
    int i;
    int w_rank = ompi_comm_rank(comm);
    int v_rank = w_rank;
    if (w_rank == root) {
        v_rank = 0;
    }
    else if (w_rank < root) {
        v_rank = w_rank + 1;
    }
    else {
        v_rank = w_rank;
    }
    //printf("In shared bcast w_rank %d v_rank %d\n", w_rank, v_rank);
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    int seg_size, l_seg_size;
    seg_size = count / (shared_module->sm_size-1);
    l_seg_size = seg_size;
    if (v_rank == shared_module->sm_size-1) {
        seg_size = count - (shared_module->sm_size-2) * l_seg_size;
    }
    //root copy data to shared memory
    if (v_rank == 0) {
        char *c;
        c = buff;
        for (i=1; i<shared_module->sm_size; i++) {
            if (i != shared_module->sm_size-1) {
                seg_size = l_seg_size;
            }
            else {
                seg_size = count - (shared_module->sm_size-2)*l_seg_size;
            }
            memcpy(shared_module->data_buf[i], c, seg_size*extent);
            c = c+seg_size*extent;
        }
        
    }

    shared_module->ctrl_buf[v_rank][0] = v_rank;
    shared_module->sm_ctrl_win->w_osc_module->osc_fence(0, shared_module->sm_ctrl_win);
    
    int cur = v_rank;
    if (v_rank > 0) {
        for (i=1; i<shared_module->sm_size; i++) {
            if (cur != shared_module->sm_size-1) {
                seg_size = l_seg_size;
            }
            else {
                seg_size = count - (shared_module->sm_size-2)*l_seg_size;
            }
            while (v_rank != shared_module->ctrl_buf[cur][0]) {;}
            memcpy((char *)buff+(cur-1)*l_seg_size*extent, shared_module->data_buf[cur], seg_size*extent);
            shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);
            //printf("[%d cur %d v_rank %d]: Copy %d (%d %d)\n", i, cur, v_rank, seg_size, shared_module->data_buf[cur][0], shared_module->data_buf[cur][1]);
            cur = (cur-2+shared_module->sm_size-1)%(shared_module->sm_size-1)+1;
            shared_module->ctrl_buf[cur][0] = (shared_module->ctrl_buf[cur][0])%(shared_module->sm_size-1)+1;
            shared_module->sm_ctrl_win->w_osc_module->osc_fence(0, shared_module->sm_ctrl_win);
        }
    }
    else {
        for (i=1; i<shared_module->sm_size; i++) {
            shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);
            shared_module->sm_ctrl_win->w_osc_module->osc_fence(0, shared_module->sm_ctrl_win);
        }

    }
    return OMPI_SUCCESS;
}


int mca_coll_shared_bcast_linear_intra(void *buff, int count, struct ompi_datatype_t *dtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module){
    mca_coll_shared_module_t *shared_module = (mca_coll_shared_module_t*) module;
    if (!shared_module->enabled) {
        ompi_coll_shared_lazy_enable(module, comm);
    }
    
    //printf("In shared linear bcast\n");
    ptrdiff_t extent, lower_bound;
    ompi_datatype_get_extent(dtype, &lower_bound, &extent);
    shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);
    int w_rank = ompi_comm_rank(comm);
    if (w_rank == root) {
        memcpy(shared_module->data_buf[root], (char*)buff, count*extent);
    }
    shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);

    if (w_rank != root) {
        memcpy((char*)buff, shared_module->data_buf[root], count*extent);
    }
    shared_module->sm_data_win->w_osc_module->osc_fence(0, shared_module->sm_data_win);
    return OMPI_SUCCESS;
}
