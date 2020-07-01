#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdbool.h>
#include <stdint.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif /* __ARM_FEATURE_SVE */

#include "mpi.h"

#define ARRAYSIZE  1024*1024

float in_float[ARRAYSIZE];
float inout_float[ARRAYSIZE];
float inout_float_for_check[ARRAYSIZE];

int8_t in_uint8[ARRAYSIZE];
int8_t inout_uint8[ARRAYSIZE];
int8_t inout_uint8_for_check[ARRAYSIZE];

uint16_t in_uint16[ARRAYSIZE];
uint16_t inout_uint16[ARRAYSIZE];
uint16_t inout_uint16_for_check[ARRAYSIZE];

uint32_t in_uint32[ARRAYSIZE];
uint32_t inout_uint32[ARRAYSIZE];
uint32_t inout_uint32_for_check[ARRAYSIZE];

uint64_t in_uint64[ARRAYSIZE];
uint64_t inout_uint64[ARRAYSIZE];
uint64_t inout_uint64_for_check[ARRAYSIZE];

double in_double[ARRAYSIZE];
double inout_double[ARRAYSIZE];
double inout_double_for_check[ARRAYSIZE];

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
        in_float[i] = 1000.0+1;
        inout_float[i] = inout_float_for_check[i] = 100.0+2;

        in_double[i] = 10.0+1;
        inout_double[i] = inout_double_for_check[i] = 1.0+2;

        in_uint8[i] = 5;
        inout_uint8[i] = inout_uint8_for_check[i] = 3;

        in_uint16[i] = 5;
        inout_uint16[i] = inout_uint16_for_check[i] = 3;

        in_uint32[i] = 5;
        inout_uint32[i] = inout_uint32_for_check[i] = 3;

        in_uint64[i] = 5;
        inout_uint64[i] = inout_uint64_for_check[i] = 3;
    }
    int rank, size, len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double tstart, tend;

    int correctness=1;

    if(*type=='i') {
        if(strcmp(op, "sum") == 0){
            printf("#Local Reduce SUM: %d \n", count);
            tstart = MPI_Wtime();
            if (elem_size1 == 8)
            {
                MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_INT8_T, MPI_SUM);
                for (i=0; i<count; i++)
                {
                    if(inout_uint8[i]!=in_uint8[i] + inout_uint8_for_check[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 8 SUM check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 8 SUM check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 16)
            {
                MPI_Reduce_local(in_uint16,inout_uint16,count, MPI_INT16_T, MPI_SUM);
                for (i=0; i<count; i++)
                {
                    if(inout_uint16[i]!=in_uint16[i] + inout_uint16_for_check[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 16 SUM check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 16 SUM check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 32)
            {
                MPI_Reduce_local(in_uint32,inout_uint32,count, MPI_UINT32_T, MPI_SUM);
                for (i=0; i<count; i++)
                {
                    if(inout_uint32[i]!=in_uint32[i] + inout_uint32_for_check[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 32 SUM check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 32 SUM check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 64)
            {
                MPI_Reduce_local(in_uint64,inout_uint64,count, MPI_UINT64_T, MPI_SUM);
                for (i=0; i<count; i++)
                {
                    if(inout_uint64[i]!=in_uint64[i] + inout_uint64_for_check[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 64 SUM check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 64 SUM check \033[1;31m fail\033[0m!");
            }
            tend = MPI_Wtime();
        }

        if(strcmp(op, "max") == 0){
            printf("#Local Reduce MAX: %d \n", count);
            tstart = MPI_Wtime();
            if (elem_size1 == 8)
            {
                MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_INT8_T, MPI_MAX);
                for (i=0; i<count; i++)
                {
                    if(inout_uint8[i]!=in_uint8[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 8 MAX check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 8 MAX check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 16)
            {
                MPI_Reduce_local(in_uint16,inout_uint16,count, MPI_INT16_T, MPI_MAX);
                for (i=0; i<count; i++)
                {
                    if(inout_uint16[i]!=in_uint16[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 16 MAX check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 16 MAX check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 32)
            {
                MPI_Reduce_local(in_uint32,inout_uint32,count, MPI_UINT32_T, MPI_MAX);
                for (i=0; i<count; i++)
                {
                    if(inout_uint32[i]!=in_uint32[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 32 MAX check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 32 MAX check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 64)
            {
                MPI_Reduce_local(in_uint64,inout_uint64,count, MPI_UINT64_T, MPI_MAX);
                for (i=0; i<count; i++)
                {
                    if(inout_uint64[i]!=in_uint64[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 64 MAX check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 64 MAX check \033[1;31m fail\033[0m!");
            }
            tend = MPI_Wtime();
        }
        //intentionly reversed in and out
        if(strcmp(op, "min") == 0) {
            printf("#Local Reduce MIN: %d \n", count);
            tstart = MPI_Wtime();
            if (elem_size1 == 8)
            {
                MPI_Reduce_local(inout_uint8,in_uint8,count, MPI_INT8_T, MPI_MIN);
                for (i=0; i<count; i++)
                {
                    if(inout_uint8[i]!=in_uint8[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 8 MIN check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 8 MIN check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 16)
            {
                MPI_Reduce_local(inout_uint16,in_uint16,count, MPI_INT16_T, MPI_MIN);
                for (i=0; i<count; i++)
                {
                    if(inout_uint16[i]!=in_uint16[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 16 MIN check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 16 MIN check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 32)
            {
                MPI_Reduce_local(inout_uint32,in_uint32,count, MPI_UINT32_T, MPI_MIN);
                for (i=0; i<count; i++)
                {
                    if(inout_uint32[i]!=in_uint32[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 32 MIN check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 32 MIN check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 64)
            {
                MPI_Reduce_local(inout_uint64,in_uint64,count, MPI_UINT64_T, MPI_MIN);
                for (i=0; i<count; i++)
                {
                    if(inout_uint64[i]!=in_uint64[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 64 MIN check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 64 MIN check \033[1;31m fail\033[0m!");
            }
            tend = MPI_Wtime();
        }

        if(strcmp(op, "bor") == 0) {
            printf("#Local Reduce Logical: %d \n", count);
            tstart = MPI_Wtime();
            if (elem_size1 == 8)
            {
                MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_INT8_T, MPI_BOR);
                for (i=0; i<count; i++)
                {
                    if(inout_uint8[i]!= (in_uint8[i] | inout_uint8_for_check[i]))
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 8 BOR check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 8 BOR check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 16)
            {
                MPI_Reduce_local(in_uint16,inout_uint16,count, MPI_INT16_T, MPI_BOR);
                for (i=0; i<count; i++)
                {
                    if(inout_uint16[i] != (in_uint16[i] | inout_uint16_for_check[i]))
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 16 BOR check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 16 BOR check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 32)
            {
                MPI_Reduce_local(in_uint32,inout_uint32,count, MPI_UINT32_T, MPI_BOR);
                for (i=0; i<count; i++)
                {
                    if(inout_uint32[i] != (in_uint32[i] | inout_uint32_for_check[i]))
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 32 BOR check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 32 BOR check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 64)
            {
                MPI_Reduce_local(in_uint64,inout_uint64,count, MPI_UINT64_T, MPI_BOR);
                for (i=0; i<count; i++)
                {
                    if(inout_uint64[i] != (in_uint64[i] | inout_uint64_for_check[i]))
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 64 BOR check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 64 BOR check \033[1;31m fail\033[0m!");
            }
            tend = MPI_Wtime();
        }

        if(strcmp(op, "bxor") == 0) {
            printf("#Local Reduce Logical: %d \n", count);
            tstart = MPI_Wtime();
            if (elem_size1 == 8)
            {
                MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_INT8_T, MPI_BXOR);
                for (i=0; i<count; i++)
                {
                    if(inout_uint8[i]!=(in_uint8[i] ^ inout_uint8_for_check[i]))
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 8 BXOR check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 8 BXOR check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 16)
            {
                MPI_Reduce_local(in_uint16,inout_uint16,count, MPI_INT16_T, MPI_BXOR);
                for (i=0; i<count; i++)
                {
                    if(inout_uint16[i]!=(in_uint16[i] ^ inout_uint16_for_check[i]))
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 16 BXOR check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 16 BXOR check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 32)
            {
                MPI_Reduce_local(in_uint32,inout_uint32,count, MPI_UINT32_T, MPI_BXOR);
                for (i=0; i<count; i++)
                {
                    if(inout_uint32[i]!=(in_uint32[i] ^ inout_uint32_for_check[i]))
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 32 BXOR check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 32 BXOR check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 64)
            {
                MPI_Reduce_local(in_uint64,inout_uint64,count, MPI_UINT64_T, MPI_BXOR);
                for (i=0; i<count; i++)
                {
                    if(inout_uint64[i]!=(in_uint64[i] ^ inout_uint64_for_check[i]))
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 64 BXOR check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 64 BXOR check \033[1;31m fail\033[0m!");
            }
            tend = MPI_Wtime();
        }


        if(strcmp(op, "mul") == 0) {
            printf("#Local Reduce Logical: %d \n", count);
            tstart = MPI_Wtime();
            if (elem_size1 == 8)
            {
                MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_INT8_T, MPI_PROD);
                for (i=0; i<count; i++)
                {
                    if(inout_uint8[i]!=in_uint8[i] * inout_uint8_for_check[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 8 PROD check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 8 PROD check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 16)
            {
                MPI_Reduce_local(in_uint16,inout_uint16,count, MPI_INT16_T, MPI_PROD);
                for (i=0; i<count; i++)
                {
                    if(inout_uint16[i]!=in_uint16[i] * inout_uint16_for_check[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 16 PROD check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 16 PROD check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 32)
            {
                MPI_Reduce_local(in_uint32,inout_uint32,count, MPI_UINT32_T, MPI_PROD);
                for (i=0; i<count; i++)
                {
                    if(inout_uint32[i]!=in_uint32[i] * inout_uint32_for_check[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 32 PROD check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 32 PROD check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 64)
            {
                MPI_Reduce_local(in_uint64,inout_uint64,count, MPI_UINT64_T, MPI_PROD);
                for (i=0; i<count; i++)
                {
                    if(inout_uint64[i]!=in_uint64[i] * inout_uint64_for_check[i])
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 64 PROD check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 64 PROD check \033[1;31m fail\033[0m!");
            }
            tend = MPI_Wtime();
        }

        if(strcmp(op, "band") == 0) {
            printf("#Local Reduce Logical: %d \n", count);
            tstart = MPI_Wtime();
            if (elem_size1 == 8)
            {
                MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_INT8_T, MPI_BAND);
                for (i=0; i<count; i++)
                {
                    if(inout_uint8[i]!=(in_uint8[i] & inout_uint8_for_check[i]) )
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 8 BAND check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 8 BAND check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 16)
            {
                MPI_Reduce_local(in_uint16,inout_uint16,count, MPI_INT16_T, MPI_BAND);
                for (i=0; i<count; i++)
                {
                    if(inout_uint16[i]!=(in_uint16[i] & inout_uint16_for_check[i]))
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 16 BAND check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 16 BAND check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 32)
            {
                MPI_Reduce_local(in_uint32,inout_uint32,count, MPI_UINT32_T, MPI_BAND);
                for (i=0; i<count; i++)
                {
                    if(inout_uint32[i]!=(in_uint32[i] & inout_uint32_for_check[i]))
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 32 BAND check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 32 BAND check \033[1;31m fail\033[0m!");
            }
            if (elem_size1 == 64)
            {
                MPI_Reduce_local(in_uint64,inout_uint64,count, MPI_UINT64_T, MPI_BAND);
                for (i=0; i<count; i++)
                {
                    if(inout_uint64[i]!= (in_uint64[i] & inout_uint64_for_check[i]) )
                        correctness = 0;
                }
                if(correctness)
                    printf("Integer Size 64 BAND check \033[1;32m success!\033[0m");
                else
                    printf("Integer Size 64 BAND check \033[1;31m fail\033[0m!");
            }
            tend = MPI_Wtime();
        }
    }

    if(*type=='f') {
        if(strcmp(op, "sum") == 0){
            tstart = MPI_Wtime();
            MPI_Reduce_local(in_float,inout_float,count, MPI_FLOAT, MPI_SUM);
            for (i=0; i<count; i++)
            {
                if(inout_float[i]!=inout_float_for_check[i]+in_float[i])
                    correctness = 0;
            }
            if(correctness)
                printf("Float Sum check \033[1;32m success!\033[0m");
            else
                printf("Float Sum check \033[1;31m fail\033[0m!");
            tend = MPI_Wtime();
        }

        if(strcmp(op, "max") == 0){
            tstart = MPI_Wtime();
            MPI_Reduce_local(in_float,inout_float,count, MPI_FLOAT, MPI_MAX);
            for (i=0; i<count; i++)
            {
                if(inout_float[i]!=in_float[i])
                    correctness = 0;
            }
            if(correctness)
                printf("Float Max check \033[1;32m success!\033[0m");
            else
                printf("Float Max check \033[1;31m fail\033[0m!");
            tend = MPI_Wtime();
        }

        if(strcmp(op, "min") == 0) {
            tstart = MPI_Wtime();
            MPI_Reduce_local(inout_float,in_float,count, MPI_FLOAT, MPI_MIN);
            for (i=0; i<count; i++)
            {
                if(inout_float[i]!=in_float[i])
                    correctness = 0;
            }
            if(correctness)
                printf("Float Min check \033[1;32m success!\033[0m");
            else
                printf("Float Min check \033[1;31m fail\033[0m!");
            tend = MPI_Wtime();
        }

        if(strcmp(op, "mul") == 0) {
            tstart = MPI_Wtime();
            MPI_Reduce_local(in_float,inout_float,count, MPI_FLOAT, MPI_PROD);
            for (i=0; i<count; i++)
            {
                if(inout_float[i]!=in_float[i] * inout_float_for_check[i])
                    correctness = 0;
            }
            if(correctness)
                printf("Float Prod check \033[1;32m success!\033[0m");
            else
                printf("Float Prod check \033[1;31m fail\033[0m!");
            tend = MPI_Wtime();
        }
    }

    if(*type=='d') {
        if(strcmp(op, "sum") == 0){
            tstart = MPI_Wtime();
            MPI_Reduce_local(in_double,inout_double,count, MPI_DOUBLE, MPI_SUM);
            for (i=0; i<count; i++)
            {
                if(inout_double[i]!=inout_double_for_check[i]+in_double[i])
                    correctness = 0;
            }
            if(correctness)
                printf("Double Sum check \033[1;32m success!\033[0m");
            else
                printf("Double Sum check \033[1;31m fail\033[0m!");
            tend = MPI_Wtime();
        }

        if(strcmp(op, "max") == 0){
            tstart = MPI_Wtime();
            MPI_Reduce_local(in_double,inout_double,count, MPI_DOUBLE, MPI_MAX);
            for (i=0; i<count; i++)
            {
                if(inout_double[i]!=in_double[i])
                    correctness = 0;
            }
            if(correctness)
                printf("Double Max check \033[1;32m success!\033[0m");
            else
                printf("Double Max check \033[1;31m fail\033[0m!");
            tend = MPI_Wtime();
        }

        if(strcmp(op, "min") == 0) {
            tstart = MPI_Wtime();
            MPI_Reduce_local(inout_double,in_double,count, MPI_DOUBLE, MPI_MIN);
            for (i=0; i<count; i++)
            {
                if(inout_double[i]!=in_double[i])
                    correctness = 0;
            }
            if(correctness)
                printf("Double Min check \033[1;32m success!\033[0m");
            else
                printf("Double Min check \033[1;31m fail\033[0m!");
            tend = MPI_Wtime();
        }
        if(strcmp(op, "mul") == 0) {
            tstart = MPI_Wtime();
            MPI_Reduce_local(in_double,inout_double,count, MPI_DOUBLE, MPI_PROD);
            for (i=0; i<count; i++)
            {
                if(inout_double[i]!=inout_double_for_check[i]*in_double[i])
                    correctness = 0;
            }
            if(correctness)
                printf("Double Prod check \033[1;32m success!\033[0m");
            else
                 printf("Double Prod check \033[1;31m fail\033[0m!");
             tend = MPI_Wtime();
         }
    }
    /*
       printf("\n ================\n");
       for (i=0; i<count; i++)
       {
       printf(" float: %f", inout_float[i]);
       if((i+1)%16==0)
       printf("\n");
       }

       printf("\n ================\n");
       for (i=0; i<count; i++)
       {
       printf(" out8: %d", inout_uint8[i]);
       if((i+1)%64==0)
       printf("\n");
       }

       printf("\n ================\n");
       for (i=0; i<count; i++)
       {
       printf(" out16: %d",inout_uint16[i]);
       if((i+1)%32==0)
       printf("\n");
       }
       printf("\n ================\n");
       for (i=0; i<count; i++)
       {
       printf(" out32: %d",inout_uint32[i]);
       if((i+1)%16==0)
       printf("\n");
       }

       printf("\n ================\n");
       for (i=0; i<count; i++)
       {
       printf(" out64: %d",inout_uint64[i]);
       if((i+1)%8==0)
       printf("\n");
       }
    printf("\n ================\n");
    for (i=0; i<count; i++)
    {
        printf(" float: %f", inout_double[i]);
        if((i+1)%8==0)
            printf("\n");
    }
*/
    printf("\n#Local Reduce time %.6f seconds\n", tend-tstart);
    MPI_Finalize();
    return 0;
}

