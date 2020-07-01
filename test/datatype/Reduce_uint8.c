#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif /* __ARM_FEATURE_SVE */

#include "mpi.h"

#define ARRAYSIZE  32*1024*1024

int8_t in_uint8[ARRAYSIZE];
int8_t inout_uint8[ARRAYSIZE];
int8_t inout_uint8_for_check[ARRAYSIZE];

int main(int argc, char **argv) {

    char *num_elem = argv[1];
    int count = atoi(num_elem);
    char *type = argv[2];
    char *elem_size = argv[3];
    int elem_size1  = atoi(elem_size);
    char *op = argv[4];

    int i;

    for (i=0; i<count; i++)
    {

        in_uint8[i] = 5;
        inout_uint8[i] = inout_uint8_for_check[i] = -3;

    }
    int rank, size, len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double tstart, tend;

    int correctness=1;

    tstart = MPI_Wtime();
    if(strcmp(op, "sum") == 0){
        MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_INT8_T, MPI_SUM);
    }
    else
        memcpy(in_uint8,inout_uint8, count);
    tend = MPI_Wtime();
    printf("PERF count  %d  time %.6f seconds\n",count, tend-tstart);
    MPI_Finalize();
#define L1size sysconf(_SC_LEVEL1_DCACHE_SIZE)
#define L2size sysconf(_SC_LEVEL2_CACHE_SIZE)
#define L3size sysconf(_SC_LEVEL2_CACHE_SIZE)
    char *cache = (char*)calloc(L1size+L2size+L3size, sizeof(char));
    free(cache);
    return 0;
}
