/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "mpi.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include "ompi/datatype/ompi_datatype.h"
#include "opal/datatype/opal_convertor.h"
#include "ompi/proc/proc.h"

#if 1 && OPEN_MPI
//extern void ompi_datatype_dump( MPI_Datatype ddt );
#define MPI_DDT_DUMP(ddt) ompi_datatype_dump( (ddt) )
#else
#define MPI_DDT_DUMP(ddt)
#endif  /* OPEN_MPI */

/* Create a non-contiguous resized datatype */
struct structure {
    double not_transfered;
    double transfered_1;
    double transfered_2;
};

static MPI_Datatype
create_struct_constant_gap_resized_ddt( int number,  /* IGNORED: number of repetitions */
                                        int contig_size,  /* IGNORED: number of elements in a contiguous chunk */
                                        int gap_size )    /* IGNORED: number of elements in a gap */
{
    struct structure data[1];
    MPI_Datatype struct_type, temp_type;
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    int blocklens[2] = {1, 1};
    MPI_Aint disps[3];

    MPI_Get_address(&data[0].transfered_1, &disps[0]);
    MPI_Get_address(&data[0].transfered_2, &disps[1]);
    MPI_Get_address(&data[0], &disps[2]);
    disps[1] -= disps[2]; /*  8 */
    disps[0] -= disps[2]; /* 16 */

    MPI_Type_create_struct(2, blocklens, disps, types, &temp_type);
    MPI_Type_create_resized(temp_type, 0, sizeof(data[0]), &struct_type);
    MPI_Type_commit(&struct_type);
    MPI_Type_free(&temp_type);
    MPI_DDT_DUMP( struct_type );

    return struct_type;
}

/* Create a datatype similar to the one use by HPL */
static MPI_Datatype
create_indexed_constant_gap_ddt( int number,  /* number of repetitions */
                                 int contig_size,  /* number of elements in a contiguous chunk */
                                 int gap_size )    /* number of elements in a gap */
{
    MPI_Datatype dt, *types;
    int i, *bLength;
    MPI_Aint* displ;

    types = (MPI_Datatype*)malloc( sizeof(MPI_Datatype) * number );
    bLength = (int*)malloc( sizeof(int) * number );
    displ = (MPI_Aint*)malloc( sizeof(MPI_Aint) * number );

    types[0] = MPI_DOUBLE;
    bLength[0] = contig_size;
    displ[0] = 0;
    for( i = 1; i < number; i++ ) {
        types[i] = MPI_DOUBLE;
        bLength[i] = contig_size;
        displ[i] = displ[i-1] + sizeof(double) * (contig_size + gap_size);
    }
    MPI_Type_create_struct( number, bLength, displ, types, &dt );
    MPI_DDT_DUMP( dt );
    free(types);
    free(bLength);
    free(displ);
    MPI_Type_commit( &dt );
    return dt;
}

static MPI_Datatype
create_optimized_indexed_constant_gap_ddt( int number,  /* number of repetitions */
                                           int contig_size,  /* number of elements in a contiguous chunk */
                                           int gap_size )    /* number of elements in a gap */
{
    MPI_Datatype dt;

    MPI_Type_vector( number, contig_size, (contig_size + gap_size), MPI_DOUBLE, &dt );
    MPI_Type_commit( &dt );
    MPI_DDT_DUMP( dt );
    return dt;
}

typedef struct {
   int i[2];
   float f;
} internal_struct;
typedef struct {
   int v1;
   int gap1;
   internal_struct is[3];
} ddt_gap;

static MPI_Datatype
create_indexed_gap_ddt( void )
{
    ddt_gap dt[2];
    MPI_Datatype dt1, dt2, dt3;
    int bLength[2] = { 2, 1 };
    MPI_Datatype types[2] = { MPI_INT, MPI_FLOAT };
    MPI_Aint displ[2];

    MPI_Get_address( &(dt[0].is[0].i[0]), &(displ[0]) );
    MPI_Get_address( &(dt[0].is[0].f), &(displ[1]) );
    displ[1] -= displ[0];
    displ[0] -= displ[0];
    MPI_Type_create_struct( 2, bLength, displ, types, &dt1 );
    /*MPI_DDT_DUMP( dt1 );*/
    MPI_Type_contiguous( 3, dt1, &dt2 );
    /*MPI_DDT_DUMP( dt2 );*/
    bLength[0] = 1;
    bLength[1] = 1;
    MPI_Get_address( &(dt[0].v1), &(displ[0]) );
    MPI_Get_address( &(dt[0].is[0]), &(displ[1]) );
    displ[1] -= displ[0];
    displ[0] -= displ[0];
    types[0] = MPI_INT;
    types[1] = dt2;
    MPI_Type_create_struct( 2, bLength, displ, types, &dt3 );
    /*MPI_DDT_DUMP( dt3 );*/
    MPI_Type_free( &dt1 );
    MPI_Type_free( &dt2 );
    MPI_Type_contiguous( 10, dt3, &dt1 );
    MPI_DDT_DUMP( dt1 );
    MPI_Type_free( &dt3 );
    MPI_Type_commit( &dt1 );
    return dt1;
}

static MPI_Datatype
create_indexed_gap_optimized_ddt( void )
{
    MPI_Datatype dt1, dt2, dt3;
    int bLength[3];
    MPI_Datatype types[3];
    MPI_Aint displ[3];

    MPI_Type_contiguous( 40, MPI_BYTE, &dt1 );
    MPI_Type_create_resized( dt1, 0, 44, &dt2 );

    bLength[0] = 4;
    bLength[1] = 9;
    bLength[2] = 36;

    types[0] = MPI_BYTE;
    types[1] = dt2;
    types[2] = MPI_BYTE;

    displ[0] = 0;
    displ[1] = 8;
    displ[2] = 44 * 9 + 8;

    MPI_Type_create_struct( 3, bLength, displ, types, &dt3 );

    MPI_Type_free( &dt1 );
    MPI_Type_free( &dt2 );
    MPI_DDT_DUMP( dt3 );
    MPI_Type_commit( &dt3 );
    return dt3;
}

#include <mpi.h>

/* automatically generated reconstruction of MPI_Datatype
   see source file >analyze_MPI_Datatype.h< for details */
static MPI_Datatype ICON_send_datatype()
{
    // packed size - - - - - - : 39040
    // lower bound - - - - - - : 16
    // extent  - - - - - - - - : 5120
    // min. allocation packed  : 39040
    // min. allocation unpacked: 5136
    int ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[49];
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[0] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[1] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[2] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[3] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[4] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[5] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[6] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[7] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[8] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[9] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[10] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[11] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[12] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[13] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[14] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[15] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[16] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[17] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[18] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[19] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[20] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[21] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[22] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[23] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[24] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[25] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[26] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[27] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[28] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[29] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[30] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[31] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[32] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[33] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[34] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[35] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[36] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[37] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[38] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[39] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[40] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[41] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[42] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[43] = 3;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[44] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[45] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[46] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[47] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl[48] = 1;
    int ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[49];
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[0] = 0; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[1] = 634; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[2] = 636; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[3] = 1915;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[4] = 1925; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[5] = 2556; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[6] = 2563; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[7] = 2565;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[8] = 6398; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[9] = 6403; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[10] = 16649; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[11] = 17274;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[12] = 17914; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[13] = 18564; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[14] = 21767;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[15] = 21769; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[16] = 22394; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[17] = 22396;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[18] = 22399; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[19] = 22402; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[20] = 50563;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[21] = 51206; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[22] = 51837; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[23] = 51839;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[24] = 52484; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[25] = 52486; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[26] = 53763;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[27] = 53766; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[28] = 53768; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[29] = 54395;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[30] = 56956; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[31] = 56967; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[32] = 56969;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[33] = 58234; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[34] = 58237; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[35] = 58239;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[36] = 58880; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[37] = 59515; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[38] = 59522;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[39] = 59529; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[40] = 60155; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[41] = 73604;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[42] = 73606; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[43] = 125445; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[44] = -4; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[45] = -2;
    ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[46] = 3; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[47] = 635; ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl[48] = 1289;
    MPI_Datatype ddt_send_from_35_to_36_0_0_0_0_0_0_0;
    MPI_Type_indexed(49, ddt_send_from_35_to_36_0_0_0_0_0_0_0_bkl, ddt_send_from_35_to_36_0_0_0_0_0_0_0_dpl, MPI_REAL8, &ddt_send_from_35_to_36_0_0_0_0_0_0_0);
    MPI_Datatype ddt_send_from_35_to_36_0_0_0_0_0_0_1;
    MPI_Type_vector(3, 1, 5, MPI_REAL8, &ddt_send_from_35_to_36_0_0_0_0_0_0_1);
    MPI_Datatype ddt_send_from_35_to_36_0_0_0_0_0_0_2;
    MPI_Type_dup(MPI_REAL8, &ddt_send_from_35_to_36_0_0_0_0_0_0_2);
    MPI_Datatype ddt_send_from_35_to_36_0_0_0_0_0_0_3;
    MPI_Type_vector(4, 1, 3, MPI_REAL8, &ddt_send_from_35_to_36_0_0_0_0_0_0_3);
    int ddt_send_from_35_to_36_0_0_0_0_0_0_4_bkl[9];
    ddt_send_from_35_to_36_0_0_0_0_0_0_4_bkl[0] = 2; ddt_send_from_35_to_36_0_0_0_0_0_0_4_bkl[1] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_4_bkl[2] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_4_bkl[3] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_4_bkl[4] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_4_bkl[5] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_4_bkl[6] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_4_bkl[7] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_4_bkl[8] = 1;
    int ddt_send_from_35_to_36_0_0_0_0_0_0_4_dpl[9];
    ddt_send_from_35_to_36_0_0_0_0_0_0_4_dpl[0] = 0; ddt_send_from_35_to_36_0_0_0_0_0_0_4_dpl[1] = 10245; ddt_send_from_35_to_36_0_0_0_0_0_0_4_dpl[2] = 10248; ddt_send_from_35_to_36_0_0_0_0_0_0_4_dpl[3] = 10883;
    ddt_send_from_35_to_36_0_0_0_0_0_0_4_dpl[4] = 10888; ddt_send_from_35_to_36_0_0_0_0_0_0_4_dpl[5] = 11515; ddt_send_from_35_to_36_0_0_0_0_0_0_4_dpl[6] = 12166; ddt_send_from_35_to_36_0_0_0_0_0_0_4_dpl[7] = 14728;
    ddt_send_from_35_to_36_0_0_0_0_0_0_4_dpl[8] = 15368;
    MPI_Datatype ddt_send_from_35_to_36_0_0_0_0_0_0_4;
    MPI_Type_indexed(9, ddt_send_from_35_to_36_0_0_0_0_0_0_4_bkl, ddt_send_from_35_to_36_0_0_0_0_0_0_4_dpl, MPI_REAL8, &ddt_send_from_35_to_36_0_0_0_0_0_0_4);
    MPI_Datatype ddt_send_from_35_to_36_0_0_0_0_0_0_5;
    MPI_Type_vector(3, 1, 3, MPI_REAL8, &ddt_send_from_35_to_36_0_0_0_0_0_0_5);
    int ddt_send_from_35_to_36_0_0_0_0_0_0_6_dpl[4];
    ddt_send_from_35_to_36_0_0_0_0_0_0_6_dpl[0] = 0; ddt_send_from_35_to_36_0_0_0_0_0_0_6_dpl[1] = 8; ddt_send_from_35_to_36_0_0_0_0_0_0_6_dpl[2] = 11; ddt_send_from_35_to_36_0_0_0_0_0_0_6_dpl[3] = 654;
    MPI_Datatype ddt_send_from_35_to_36_0_0_0_0_0_0_6;
    MPI_Type_create_indexed_block(4, 1, ddt_send_from_35_to_36_0_0_0_0_0_0_6_dpl, MPI_REAL8, &ddt_send_from_35_to_36_0_0_0_0_0_0_6);
    MPI_Datatype ddt_send_from_35_to_36_0_0_0_0_0_0_7;
    MPI_Type_vector(3, 1, 3, MPI_REAL8, &ddt_send_from_35_to_36_0_0_0_0_0_0_7);
    int ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[28];
    ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[0] = 0; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[1] = 628; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[2] = 636; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[3] = 1905;
    ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[4] = 1913; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[5] = 1916; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[6] = 1918; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[7] = 2545;
    ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[8] = 5109; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[9] = 5113; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[10] = 5119; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[11] = 5746;
    ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[12] = 5752; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[13] = 5760; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[14] = 6387; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[15] = 6389;
    ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[16] = 7033; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[17] = 7036; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[18] = 7665; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[19] = 7668;
    ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[20] = 7675; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[21] = 7678; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[22] = 8305; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[23] = 8308;
    ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[24] = 21756; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[25] = 73590; ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[26] = 73595;
    ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl[27] = 73599;
    MPI_Datatype ddt_send_from_35_to_36_0_0_0_0_0_0_8;
    MPI_Type_create_indexed_block(28, 1, ddt_send_from_35_to_36_0_0_0_0_0_0_8_dpl, MPI_REAL8, &ddt_send_from_35_to_36_0_0_0_0_0_0_8);
    int ddt_send_from_35_to_36_0_0_0_0_0_0_bkl[9];
    ddt_send_from_35_to_36_0_0_0_0_0_0_bkl[0] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_bkl[1] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_bkl[2] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_bkl[3] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_bkl[4] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_bkl[5] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_bkl[6] = 1; ddt_send_from_35_to_36_0_0_0_0_0_0_bkl[7] = 1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_bkl[8] = 1;
    MPI_Aint ddt_send_from_35_to_36_0_0_0_0_0_0_dpl[9];
    ddt_send_from_35_to_36_0_0_0_0_0_0_dpl[0] = 48;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dpl[1] = 15360;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dpl[2] = 15464;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dpl[3] = 20512;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dpl[4] = 51248;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dpl[5] = 179208;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dpl[6] = 404480;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dpl[7] = 414728;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dpl[8] = 414840;
    MPI_Datatype ddt_send_from_35_to_36_0_0_0_0_0_0_dts[9];
    ddt_send_from_35_to_36_0_0_0_0_0_0_dts[0] = ddt_send_from_35_to_36_0_0_0_0_0_0_0;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dts[1] = ddt_send_from_35_to_36_0_0_0_0_0_0_1;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dts[2] = ddt_send_from_35_to_36_0_0_0_0_0_0_2;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dts[3] = ddt_send_from_35_to_36_0_0_0_0_0_0_3;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dts[4] = ddt_send_from_35_to_36_0_0_0_0_0_0_4;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dts[5] = ddt_send_from_35_to_36_0_0_0_0_0_0_5;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dts[6] = ddt_send_from_35_to_36_0_0_0_0_0_0_6;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dts[7] = ddt_send_from_35_to_36_0_0_0_0_0_0_7;
    ddt_send_from_35_to_36_0_0_0_0_0_0_dts[8] = ddt_send_from_35_to_36_0_0_0_0_0_0_8;
    MPI_Datatype ddt_send_from_35_to_36_0_0_0_0_0_0;
    MPI_Type_create_struct(9, ddt_send_from_35_to_36_0_0_0_0_0_0_bkl, ddt_send_from_35_to_36_0_0_0_0_0_0_dpl, ddt_send_from_35_to_36_0_0_0_0_0_0_dts, &ddt_send_from_35_to_36_0_0_0_0_0_0);
    MPI_Datatype ddt_send_from_35_to_36_0_0_0_0_0;
    MPI_Type_dup(ddt_send_from_35_to_36_0_0_0_0_0_0, &ddt_send_from_35_to_36_0_0_0_0_0);
    MPI_Datatype ddt_send_from_35_to_36_0_0_0_0;
    MPI_Type_dup(ddt_send_from_35_to_36_0_0_0_0_0, &ddt_send_from_35_to_36_0_0_0_0);
    MPI_Datatype ddt_send_from_35_to_36_0_0_0;
    MPI_Type_create_resized(ddt_send_from_35_to_36_0_0_0_0, 16, 128, &ddt_send_from_35_to_36_0_0_0);
#if 1
    MPI_Datatype ddt_send_from_35_to_36_0_0;
    MPI_Type_contiguous(40, ddt_send_from_35_to_36_0_0_0, &ddt_send_from_35_to_36_0_0);
    MPI_Datatype ddt_send_from_35_to_36_0;
    MPI_Type_dup(ddt_send_from_35_to_36_0_0, &ddt_send_from_35_to_36_0);
#else
    MPI_Datatype ddt_send_from_35_to_36_0;
    MPI_Type_dup(ddt_send_from_35_to_36_0_0_0, &ddt_send_from_35_to_36_0);
#endif
    MPI_Type_free(&ddt_send_from_35_to_36_0_0_0_0_0_0_0);
    MPI_Type_free(&ddt_send_from_35_to_36_0_0_0_0_0_0_1);
    MPI_Type_free(&ddt_send_from_35_to_36_0_0_0_0_0_0_2);
    MPI_Type_free(&ddt_send_from_35_to_36_0_0_0_0_0_0_3);
    MPI_Type_free(&ddt_send_from_35_to_36_0_0_0_0_0_0_4);
    MPI_Type_free(&ddt_send_from_35_to_36_0_0_0_0_0_0_5);
    MPI_Type_free(&ddt_send_from_35_to_36_0_0_0_0_0_0_6);
    MPI_Type_free(&ddt_send_from_35_to_36_0_0_0_0_0_0_7);
    MPI_Type_free(&ddt_send_from_35_to_36_0_0_0_0_0_0_8);
    MPI_Type_free(&ddt_send_from_35_to_36_0_0_0_0_0_0);
    MPI_Type_free(&ddt_send_from_35_to_36_0_0_0_0_0);
    MPI_Type_free(&ddt_send_from_35_to_36_0_0_0_0);
    MPI_Type_free(&ddt_send_from_35_to_36_0_0_0);
#if 1
    MPI_Type_free(&ddt_send_from_35_to_36_0_0);
#endif
    MPI_Type_commit(&ddt_send_from_35_to_36_0);

    return ddt_send_from_35_to_36_0;
}

/* automatically generated reconstruction of MPI_Datatype
   see source file >analyze_MPI_Datatype.h< for details */
static MPI_Datatype ICON_recv_datatype()
{
    // packed size - - - - - - : 39360
    // lower bound - - - - - - : 0
    // extent  - - - - - - - - : 5120
    // min. allocation packed  : 39360
    // min. allocation unpacked: 5120
    int ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[85];
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[0] = 2; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[1] = 2; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[2] = 3; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[3] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[4] = 2; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[5] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[6] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[7] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[8] = 2; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[9] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[10] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[11] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[12] = 2; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[13] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[14] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[15] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[16] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[17] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[18] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[19] = 2;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[20] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[21] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[22] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[23] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[24] = 2; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[25] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[26] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[27] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[28] = 2; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[29] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[30] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[31] = 2;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[32] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[33] = 2; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[34] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[35] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[36] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[37] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[38] = 4; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[39] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[40] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[41] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[42] = 2; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[43] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[44] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[45] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[46] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[47] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[48] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[49] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[50] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[51] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[52] = 2; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[53] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[54] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[55] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[56] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[57] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[58] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[59] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[60] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[61] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[62] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[63] = 2;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[64] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[65] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[66] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[67] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[68] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[69] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[70] = 2; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[71] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[72] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[73] = 2; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[74] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[75] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[76] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[77] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[78] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[79] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[80] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[81] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[82] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[83] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl[84] = 1;
    int ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[85];
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[0] = 0; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[1] = 9; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[2] = 12; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[3] = 7679;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[4] = 7684; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[5] = 10890; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[6] = 10892; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[7] = 11519;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[8] = 11527; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[9] = 13452; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[10] = 14081; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[11] = 14083;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[12] = 14086; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[13] = 14089; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[14] = 14093;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[15] = 14724; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[16] = 18568; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[17] = 18573;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[18] = 19199; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[19] = 35202; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[20] = 35205;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[21] = 35209; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[22] = 35840; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[23] = 36483;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[24] = 36488; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[25] = 37774; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[26] = 38400;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[27] = 38403; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[28] = 38411; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[29] = 76811;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[30] = 76813; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[31] = 77442; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[32] = 96643;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[33] = 96645; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[34] = 96648; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[35] = 97293;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[36] = 97919; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[37] = 97923; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[38] = 97930;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[39] = 98563; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[40] = 98565; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[41] = 99208;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[42] = 99213; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[43] = 103050; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[44] = 103679;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[45] = -1; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[46] = 4; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[47] = 7; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[48] = 11;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[49] = 639; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[50] = 643; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[51] = 7053; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[52] = 7680;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[53] = 7683; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[54] = 10891; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[55] = 10893;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[56] = 11524; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[57] = 11526; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[58] = 13449;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[59] = 13451; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[60] = 14082; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[61] = 14084;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[62] = 14088; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[63] = 14091; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[64] = 14094;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[65] = 18565; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[66] = 18567; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[67] = 18574;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[68] = 19200; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[69] = 35204; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[70] = 35207;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[71] = 35210; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[72] = 36481; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[73] = 36484;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[74] = 36487; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[75] = 38399; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[76] = 38401;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[77] = 38408; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[78] = 38410; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[79] = 76169;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[80] = 76812; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[81] = 77441; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[82] = 77444;
    ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[83] = 77451; ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl[84] = 95373;
    MPI_Datatype ddt_send_from_34_to_35_0_0_0_0_0_0_0;
    MPI_Type_indexed(85, ddt_send_from_34_to_35_0_0_0_0_0_0_0_bkl, ddt_send_from_34_to_35_0_0_0_0_0_0_0_dpl, MPI_REAL8, &ddt_send_from_34_to_35_0_0_0_0_0_0_0);
    MPI_Datatype ddt_send_from_34_to_35_0_0_0_0_0_0_1;
    MPI_Type_vector(4, 1, 3, MPI_REAL8, &ddt_send_from_34_to_35_0_0_0_0_0_0_1);
    MPI_Datatype ddt_send_from_34_to_35_0_0_0_0_0_0_2;
    MPI_Type_dup(MPI_REAL8, &ddt_send_from_34_to_35_0_0_0_0_0_0_2);
    MPI_Datatype ddt_send_from_34_to_35_0_0_0_0_0_0_3;
    MPI_Type_contiguous(2, MPI_REAL8, &ddt_send_from_34_to_35_0_0_0_0_0_0_3);
    MPI_Datatype ddt_send_from_34_to_35_0_0_0_0_0_0_4;
    MPI_Type_vector(3, 1, 5, MPI_REAL8, &ddt_send_from_34_to_35_0_0_0_0_0_0_4);
    int ddt_send_from_34_to_35_0_0_0_0_0_0_5_dpl[3];
    ddt_send_from_34_to_35_0_0_0_0_0_0_5_dpl[0] = 0; ddt_send_from_34_to_35_0_0_0_0_0_0_5_dpl[1] = 4; ddt_send_from_34_to_35_0_0_0_0_0_0_5_dpl[2] = 646;
    MPI_Datatype ddt_send_from_34_to_35_0_0_0_0_0_0_5;
    MPI_Type_create_indexed_block(3, 1, ddt_send_from_34_to_35_0_0_0_0_0_0_5_dpl, MPI_REAL8, &ddt_send_from_34_to_35_0_0_0_0_0_0_5);
    int ddt_send_from_34_to_35_0_0_0_0_0_0_6_bkl[4];
    ddt_send_from_34_to_35_0_0_0_0_0_0_6_bkl[0] = 2; ddt_send_from_34_to_35_0_0_0_0_0_0_6_bkl[1] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_6_bkl[2] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_6_bkl[3] = 1;
    int ddt_send_from_34_to_35_0_0_0_0_0_0_6_dpl[4];
    ddt_send_from_34_to_35_0_0_0_0_0_0_6_dpl[0] = 0; ddt_send_from_34_to_35_0_0_0_0_0_0_6_dpl[1] = 3; ddt_send_from_34_to_35_0_0_0_0_0_0_6_dpl[2] = 3838; ddt_send_from_34_to_35_0_0_0_0_0_0_6_dpl[3] = 3840;
    MPI_Datatype ddt_send_from_34_to_35_0_0_0_0_0_0_6;
    MPI_Type_indexed(4, ddt_send_from_34_to_35_0_0_0_0_0_0_6_bkl, ddt_send_from_34_to_35_0_0_0_0_0_0_6_dpl, MPI_REAL8, &ddt_send_from_34_to_35_0_0_0_0_0_0_6);
    int ddt_send_from_34_to_35_0_0_0_0_0_0_bkl[7];
    ddt_send_from_34_to_35_0_0_0_0_0_0_bkl[0] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_bkl[1] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_bkl[2] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_bkl[3] = 1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_bkl[4] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_bkl[5] = 1; ddt_send_from_34_to_35_0_0_0_0_0_0_bkl[6] = 1;
    MPI_Aint ddt_send_from_34_to_35_0_0_0_0_0_0_dpl[7];
    ddt_send_from_34_to_35_0_0_0_0_0_0_dpl[0] = 8;
    ddt_send_from_34_to_35_0_0_0_0_0_0_dpl[1] = 773136;
    ddt_send_from_34_to_35_0_0_0_0_0_0_dpl[2] = 778360;
    ddt_send_from_34_to_35_0_0_0_0_0_0_dpl[3] = 783376;
    ddt_send_from_34_to_35_0_0_0_0_0_0_dpl[4] = 783400;
    ddt_send_from_34_to_35_0_0_0_0_0_0_dpl[5] = 788488;
    ddt_send_from_34_to_35_0_0_0_0_0_0_dpl[6] = 793680;
    MPI_Datatype ddt_send_from_34_to_35_0_0_0_0_0_0_dts[7];
    ddt_send_from_34_to_35_0_0_0_0_0_0_dts[0] = ddt_send_from_34_to_35_0_0_0_0_0_0_0;
    ddt_send_from_34_to_35_0_0_0_0_0_0_dts[1] = ddt_send_from_34_to_35_0_0_0_0_0_0_1;
    ddt_send_from_34_to_35_0_0_0_0_0_0_dts[2] = ddt_send_from_34_to_35_0_0_0_0_0_0_2;
    ddt_send_from_34_to_35_0_0_0_0_0_0_dts[3] = ddt_send_from_34_to_35_0_0_0_0_0_0_3;
    ddt_send_from_34_to_35_0_0_0_0_0_0_dts[4] = ddt_send_from_34_to_35_0_0_0_0_0_0_4;
    ddt_send_from_34_to_35_0_0_0_0_0_0_dts[5] = ddt_send_from_34_to_35_0_0_0_0_0_0_5;
    ddt_send_from_34_to_35_0_0_0_0_0_0_dts[6] = ddt_send_from_34_to_35_0_0_0_0_0_0_6;
    MPI_Datatype ddt_send_from_34_to_35_0_0_0_0_0_0;
    MPI_Type_create_struct(7, ddt_send_from_34_to_35_0_0_0_0_0_0_bkl, ddt_send_from_34_to_35_0_0_0_0_0_0_dpl, ddt_send_from_34_to_35_0_0_0_0_0_0_dts, &ddt_send_from_34_to_35_0_0_0_0_0_0);
    MPI_Datatype ddt_send_from_34_to_35_0_0_0_0_0;
    MPI_Type_dup(ddt_send_from_34_to_35_0_0_0_0_0_0, &ddt_send_from_34_to_35_0_0_0_0_0);
    MPI_Datatype ddt_send_from_34_to_35_0_0_0_0;
    MPI_Type_dup(ddt_send_from_34_to_35_0_0_0_0_0, &ddt_send_from_34_to_35_0_0_0_0);
    MPI_Datatype ddt_send_from_34_to_35_0_0_0;
    MPI_Type_create_resized(ddt_send_from_34_to_35_0_0_0_0, 0, 128, &ddt_send_from_34_to_35_0_0_0);
    MPI_Datatype ddt_send_from_34_to_35_0_0;
    MPI_Type_contiguous(40, ddt_send_from_34_to_35_0_0_0, &ddt_send_from_34_to_35_0_0);
    MPI_Datatype ddt_send_from_34_to_35_0;
    MPI_Type_dup(ddt_send_from_34_to_35_0_0, &ddt_send_from_34_to_35_0);
    MPI_Type_free(&ddt_send_from_34_to_35_0_0_0_0_0_0_0);
    MPI_Type_free(&ddt_send_from_34_to_35_0_0_0_0_0_0_1);
    MPI_Type_free(&ddt_send_from_34_to_35_0_0_0_0_0_0_2);
    MPI_Type_free(&ddt_send_from_34_to_35_0_0_0_0_0_0_3);
    MPI_Type_free(&ddt_send_from_34_to_35_0_0_0_0_0_0_4);
    MPI_Type_free(&ddt_send_from_34_to_35_0_0_0_0_0_0_5);
    MPI_Type_free(&ddt_send_from_34_to_35_0_0_0_0_0_0_6);
    MPI_Type_free(&ddt_send_from_34_to_35_0_0_0_0_0_0);
    MPI_Type_free(&ddt_send_from_34_to_35_0_0_0_0_0);
    MPI_Type_free(&ddt_send_from_34_to_35_0_0_0_0);
    MPI_Type_free(&ddt_send_from_34_to_35_0_0_0);
    MPI_Type_free(&ddt_send_from_34_to_35_0_0);
    MPI_Type_commit(&ddt_send_from_34_to_35_0);

    return ddt_send_from_34_to_35_0;
}

/********************************************************************
 *******************************************************************/

#define DO_CONTIG                       0x00000001
#define DO_CONSTANT_GAP                 0x00000002
#define DO_OPTIMIZED_CONSTANT_GAP       0x00000004
#define DO_INDEXED_GAP                  0x00000008
#define DO_OPTIMIZED_INDEXED_GAP        0x00000010
#define DO_STRUCT_CONSTANT_GAP_RESIZED  0x00000020
#define DO_ICON_SEND_DATATYPE           0x00000040
#define DO_ICON_RECV_DATATYPE           0x00000080


#define DO_PACK                         0x01000000
#define DO_UNPACK                       0x02000000
#define DO_ISEND_RECV                   0x04000000
#define DO_ISEND_IRECV                  0x08000000
#define DO_IRECV_SEND                   0x10000000
#define DO_IRECV_ISEND                  0x20000000
#define DO_IOV_PACK                     0x40000000
#define DO_IOV_UNPACK                   0x80000000
#define DO_FLEX_IOV_PACK                0x00100000
#define DO_FLEX_IOV_UNPACK              0x00200000

#define MIN_LENGTH   1024
#define MAX_LENGTH   (1024*1024)

static int cycles  = 100;
static int trials  = 20;
static int warmups = 2;

static void print_result( int length, int trials, double* timers )
{
    double bandwidth, clock_prec, temp;
    double min_time, max_time, average, std_dev = 0.0;
    double ordered[trials];
    int t, pos, quartile_start, quartile_end;

    for( t = 0; t < trials; ordered[t] = timers[t], t++ );
    for( t = 0; t < trials-1; t++ ) {
        temp = ordered[t];
        pos = t;
        for( int i = t+1; i < trials; i++ ) {
            if( temp > ordered[i] ) {
                temp = ordered[i];
                pos = i;
            }
        }
        if( pos != t ) {
            temp = ordered[t];
            ordered[t] = ordered[pos];
            ordered[pos] = temp;
        }
    }
    quartile_start = trials - (3 * trials) / 4;
    quartile_end   = trials - (1 * trials) / 4;
    clock_prec = MPI_Wtick();
    min_time = ordered[quartile_start];
    max_time = ordered[quartile_start];
    average = ordered[quartile_start];
    for( t = quartile_start + 1; t < quartile_end; t++ ) {
        if( min_time > ordered[t] ) min_time = ordered[t];
        if( max_time < ordered[t] ) max_time = ordered[t];
        average += ordered[t];
    }
    average /= (quartile_end - quartile_start);
    for( t = quartile_start; t < quartile_end; t++ ) {
        std_dev += (ordered[t] - average) * (ordered[t] - average);
    }
    std_dev = sqrt( std_dev/(quartile_end - quartile_start) );
    
    bandwidth = (length * clock_prec) / (1024.0 * 1024.0) / (average * clock_prec);
    printf( "%8d\t%15g\t%10.4f MB/s [min %10g max %10g std %2.2f%%]\n", length, average, bandwidth,
            min_time, max_time, (100.0 * std_dev) / average );
}

/*********************************************
 *
 *
 *
 *********************************************/

typedef struct opal_datatype_flexible_storage_s {
    int32_t iov_length;
    int32_t iov_pos;
    char* storage;
} opal_datatype_flexible_storage_t;

typedef struct opal_datatype_iovec_storage_marker_s {
    uint16_t left;
    uint16_t right;
    uint32_t length;
} opal_datatype_iovec_storage_marker_t;

typedef struct opal_datatype_iovec_storage_int8_s {
    uint8_t length;
    int8_t disp;
} opal_datatype_iovec_storage_int8_t;

typedef struct opal_datatype_iovec_storage_int16_s {
    uint16_t length;
    int16_t disp;
} opal_datatype_iovec_storage_int16_t;

typedef struct opal_datatype_iovec_storage_int32_s {
    uint32_t length;
    int32_t disp;
} opal_datatype_iovec_storage_int32_t;

typedef struct opal_datatype_iovec_storage_int64_s {
    uint64_t length;
    int64_t disp;
} opal_datatype_iovec_storage_int64_t;

static void dump_uint8_hex( char* ptr, size_t length, char* endmsg )
{
    for( size_t i = 0; i < length; i++ ) {
        printf("%02X ", ((uint8_t*)ptr)[i]);
    }
    if( NULL != endmsg )
        printf("%s", endmsg);
}

/**
 * A fast and quick compression. We identify what format is required for the lasrgest of the two
 * (displacement and length), and store everything using the respective number of bytes. The trick
 * used here is to shift to the left the length to use the last few bits to store the format used
 * to represent the data using the following encoding:
 *  - last bit = 0 for 1 bytes
 *  - last 2 bits = 01 (binary) for 2 bytes
 *  - last 3 bits = 011 (binary) for 4 bytes
 *  - last 3 bits = 111 (binary) for 8 bytes
 * This scheme quarantee that for retain the most bits for the smallest cases.
 *
 * Beware: The current implementation only works for little endian.
 */
#define DEBUG_IOV  0
static int
flexible_storage_allocator(void* disp, size_t len, void* cb_data)
{
    opal_datatype_flexible_storage_t* flexi = (opal_datatype_flexible_storage_t*)cb_data;
    uint8_t bytes = sizeof(opal_datatype_iovec_storage_int64_t);
    
    if( 0 == (0x7FFFFFFFFFFFFF80 & (intptr_t)disp) ) {
        bytes = sizeof(opal_datatype_iovec_storage_int8_t);
    } else if( 0 == (0x7FFFFFFFFFFF8000 & (intptr_t)disp) ) {
        bytes = sizeof(opal_datatype_iovec_storage_int16_t);
    } else if( 0 == (0x7FFFFFFF80000000 & (intptr_t)disp) ) {
        bytes = sizeof(opal_datatype_iovec_storage_int32_t);
    }
    if( bytes < sizeof(opal_datatype_iovec_storage_int32_t) ) {
        if( 0 == (0xFFFFFFFFFFFFFF80 & len) ) {  /* single bit = 0 */
            /* follow the number of bits in the displacement */
        } else if( 0 == (0xFFFFFFFFFFFFC000 & len) ) {  /* 2 bits = 10 */
            bytes = sizeof(opal_datatype_iovec_storage_int16_t);
        }
    } else if( 0 != (0xFFFFFFFFE0000000 & len) ) {  /* 3 bits = 110 */
        bytes = sizeof(opal_datatype_iovec_storage_int64_t);
    }  /* otherwise 3 bits = 111 */

    if( (flexi->iov_pos + bytes) > flexi->iov_length ) {
        size_t new_length = (0 == flexi->iov_length ? 128 : (flexi->iov_length * 2));
        void* ptr = realloc( flexi->storage, new_length);
        if( NULL == ptr ) {  /* oops */
            return 1;
        }
        flexi->storage = ptr;
        flexi->iov_length = new_length;
    }
    switch(bytes) {
    case sizeof(opal_datatype_iovec_storage_int8_t): {
        opal_datatype_iovec_storage_int8_t* s8 = (opal_datatype_iovec_storage_int8_t*)(flexi->storage + flexi->iov_pos);
        s8->length = (uint8_t)len << 1;
        s8->disp = (int8_t)(intptr_t)disp;
#if DEBUG_IOV
        printf(" s8: <disp %d[0x%x], len %d [%lu:0x%x]> ",
               (int32_t)s8->disp, (int32_t)s8->disp, (int32_t)s8->length, len, (int32_t)s8->length);
        dump_uint8_hex(flexi->storage + flexi->iov_pos, sizeof(opal_datatype_iovec_storage_int8_t), "\n");
#endif  /* DEBUG_IOV */
        flexi->iov_pos += sizeof(opal_datatype_iovec_storage_int8_t);
        break;
    }
    case sizeof(opal_datatype_iovec_storage_int16_t): {
        opal_datatype_iovec_storage_int16_t* s16 = (opal_datatype_iovec_storage_int16_t*)(flexi->storage + flexi->iov_pos);
        s16->length = (uint16_t)len << 2 | (uint16_t)0x01;
        s16->disp = (int16_t)(intptr_t)disp;
#if DEBUG_IOV
        printf("s16: <disp %d [0x%x], len %d [%lu:0x%x]> ", (int32_t)s16->disp, (int32_t)s16->disp,
               (int32_t)s16->length, len, (int32_t)s16->length);
        dump_uint8_hex(flexi->storage + flexi->iov_pos, sizeof(opal_datatype_iovec_storage_int16_t), "\n");
#endif  /* DEBUG_IOV */
        flexi->iov_pos += sizeof(opal_datatype_iovec_storage_int16_t);
        break;
    }
    case sizeof(opal_datatype_iovec_storage_int32_t): {
        opal_datatype_iovec_storage_int32_t* s32 = (opal_datatype_iovec_storage_int32_t*)(flexi->storage + flexi->iov_pos);
        s32->length = (uint32_t)len << 3 | (uint32_t)0x03;
        s32->disp = (int32_t)(intptr_t)disp;
#if DEBUG_IOV
        printf("s32: <disp %d [0x%x], len %d [%lu:0x%x]> ",
               (int32_t)s32->disp, (int32_t)s32->disp, (int32_t)s32->length, len, (int32_t)s32->length);
        dump_uint8_hex(flexi->storage + flexi->iov_pos, sizeof(opal_datatype_iovec_storage_int32_t), "\n");
#endif  /* DEBUG_IOV */
        flexi->iov_pos += sizeof(opal_datatype_iovec_storage_int32_t);
        break;
    }
    default: {
        opal_datatype_iovec_storage_int64_t* s64 = (opal_datatype_iovec_storage_int64_t*)(flexi->storage + flexi->iov_pos);
        s64->length = (uint64_t)len << 3 | 0x07ULL;
        s64->disp = (intptr_t)disp;
#if DEBUG_IOV
        printf("s64: <disp %lld [0x%llx], len %llu [%lu:0x%llx]> ",
               s64->disp, s64->disp, s64->length, len, s64->length);
        dump_uint8_hex(flexi->storage + flexi->iov_pos, sizeof(opal_datatype_iovec_storage_int64_t), "\n");
#endif  /* DEBUG_IOV */
        flexi->iov_pos += sizeof(opal_datatype_iovec_storage_int64_t);
        break;
    }
    }

    return 0;
}

static int
pack_from_flex_iovec(char* flex_iov, int iov_length,
                     MPI_Datatype ddt, int count,
                     char* memory, char* packed_buf)
{
    int i;
    ptrdiff_t disp;
    size_t length;
    MPI_Aint lb, extent;
    char* ptr;
    
    MPI_Type_get_extent(ddt, &lb, &extent);
    
    for( i = 0; i < count; i++ ) {
        for( ptr = flex_iov; ptr < (flex_iov + iov_length); ) {
            if( 0 == ((uint8_t)0x01 & ptr[0]) ) {  /* last bit = 0: 8 bits */
                opal_datatype_iovec_storage_int8_t* s8 = (opal_datatype_iovec_storage_int8_t*)ptr;
#if DEBUG_IOV
                dump_uint8_hex( ptr, sizeof(opal_datatype_iovec_storage_int8_t), "pack  8 bits\n" );
#endif  /* DEBUG_IOV */
                length = (size_t)s8->length >> 1;
                disp = (ptrdiff_t)s8->disp;
                ptr += sizeof(opal_datatype_iovec_storage_int8_t);
            } else if( 0 == (0x02 & ptr[0]) ) {  /* last 2 bits = 01: 16 bits */
                opal_datatype_iovec_storage_int16_t* s16 = (opal_datatype_iovec_storage_int16_t*)ptr;
#if DEBUG_IOV
                dump_uint8_hex( ptr, sizeof(opal_datatype_iovec_storage_int16_t), "pack 16 bits\n" );
#endif  /* DEBUG_IOV */
                length = (size_t)(s16->length >> 2);
                disp = (ptrdiff_t)s16->disp;
                ptr += sizeof(opal_datatype_iovec_storage_int16_t);
            } else if( 0 == (0x04 & ptr[0]) ) {  /* last 3 bits = 011: 32 bits */
                opal_datatype_iovec_storage_int32_t* s32 = (opal_datatype_iovec_storage_int32_t*)ptr;
#if DEBUG_IOV
                dump_uint8_hex( ptr, sizeof(opal_datatype_iovec_storage_int32_t), "pack 32 bits\n" );
#endif  /* DEBUG_IOV */
                length = (size_t)(s32->length >> 3);
                disp = s32->disp;
                ptr += sizeof(opal_datatype_iovec_storage_int32_t);
            } else {  /* last 3 bits = 111: 64 bits */
                opal_datatype_iovec_storage_int64_t* s64 = (opal_datatype_iovec_storage_int64_t*)ptr;
#if DEBUG_IOV
                dump_uint8_hex( ptr, sizeof(opal_datatype_iovec_storage_int64_t), "pack 64 bits\n" );
#endif  /* DEBUG_IOV */
                length = s64->length >> 3;
                disp = s64->disp;
                ptr += sizeof(opal_datatype_iovec_storage_int64_t);
            }

            memcpy( packed_buf, memory + (ptrdiff_t)disp, length );
#if DEBUG_IOV
            printf( "memcpy( %p, < %ld [%p], %" PRIsize_t ">)\n", packed_buf, disp, (void*)disp, length );
#endif  /* DEBUG_IOV */
            packed_buf += length;
        }
        memory += extent;
    }
    return 0;
}

static int flex_iov_pack(int cycles,
                         MPI_Datatype ddt, int count, void* buf,
                         void* packed_buf )
{
    opal_datatype_flexible_storage_t flex_storage = { 0, 0, NULL };
    opal_convertor_t* convertor;
    size_t length;
    int myself, t, c, outsize;
    double timers[trials];

    if ( 0 == (length = ddt->super.size) ) {
        return 0;
    }
    convertor = opal_convertor_create( 0xffffffff, 0 );
    if (OMPI_SUCCESS != opal_convertor_prepare_for_send (convertor,
                                                         &(ddt->super),
                                                         1,
                                                         NULL)) {
        return -1;
    }

    opal_convertor_raw_parser(convertor, flexible_storage_allocator, &flex_storage, &length);
    
    MPI_Comm_rank( MPI_COMM_WORLD, &myself );
    outsize = ddt->super.size * count;

    for( t = 0; t < warmups; t++ ) {
        for( c = 0; c < cycles; c++ ) {
            pack_from_flex_iovec(flex_storage.storage, flex_storage.iov_pos,
                                 ddt, count, buf, packed_buf);
        }
    }
    
    for( t = 0; t < trials; t++ ) {
        timers[t] = MPI_Wtime();
        for( c = 0; c < cycles; c++ ) {
            pack_from_flex_iovec(flex_storage.storage, flex_storage.iov_pos,
                                 ddt, count, buf, packed_buf);
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result( outsize, trials, timers );

    return 0;
}

static int
unpack_from_flex_iovec(char* flex_iov, int iov_length,
                       MPI_Datatype ddt, int count,
                       char* memory, char* packed_buf)
{
    int i;
    ptrdiff_t disp;
    size_t length;
    MPI_Aint lb, extent;
    char* ptr;
    
    MPI_Type_get_extent(ddt, &lb, &extent);
    
    for( i = 0; i < count; i++ ) {
        for( ptr = flex_iov; ptr < (flex_iov + iov_length); ) {
            if( 0 == ((uint8_t)0x01 & ptr[0]) ) {  /* last bit = 0: 8 bits */
                opal_datatype_iovec_storage_int8_t* s8 = (opal_datatype_iovec_storage_int8_t*)ptr;
#if DEBUG_IOV
                dump_uint8_hex( ptr, sizeof(opal_datatype_iovec_storage_int8_t), "pack  8 bits\n" );
#endif  /* DEBUG_IOV */
                length = (size_t)s8->length >> 1;
                disp = (ptrdiff_t)s8->disp;
                ptr += sizeof(opal_datatype_iovec_storage_int8_t);
            } else if( 0 == (0x02 & ptr[0]) ) {  /* last 2 bits = 01: 16 bits */
                opal_datatype_iovec_storage_int16_t* s16 = (opal_datatype_iovec_storage_int16_t*)ptr;
#if DEBUG_IOV
                dump_uint8_hex( ptr, sizeof(opal_datatype_iovec_storage_int16_t), "pack 16 bits\n" );
#endif  /* DEBUG_IOV */
                length = (size_t)(s16->length >> 2);
                disp = (ptrdiff_t)s16->disp;
                ptr += sizeof(opal_datatype_iovec_storage_int16_t);
            } else if( 0 == (0x04 & ptr[0]) ) {  /* last 3 bits = 011: 32 bits */
                opal_datatype_iovec_storage_int32_t* s32 = (opal_datatype_iovec_storage_int32_t*)ptr;
#if DEBUG_IOV
                dump_uint8_hex( ptr, sizeof(opal_datatype_iovec_storage_int32_t), "pack 32 bits\n" );
#endif  /* DEBUG_IOV */
                length = (size_t)(s32->length >> 3);
                disp = s32->disp;
                ptr += sizeof(opal_datatype_iovec_storage_int32_t);
            } else {  /* last 3 bits = 111: 64 bits */
                opal_datatype_iovec_storage_int64_t* s64 = (opal_datatype_iovec_storage_int64_t*)ptr;
#if DEBUG_IOV
                dump_uint8_hex( ptr, sizeof(opal_datatype_iovec_storage_int64_t), "pack 64 bits\n" );
#endif  /* DEBUG_IOV */
                length = s64->length >> 3;
                disp = s64->disp;
                ptr += sizeof(opal_datatype_iovec_storage_int64_t);
            }

            memcpy( memory + (ptrdiff_t)disp, packed_buf, length );
#if DEBUG_IOV
            printf( "memcpy( %p, < %ld [%p], %" PRIsize_t ">)\n", packed_buf, disp, (void*)disp, length );
#endif  /* DEBUG_IOV */
            packed_buf += length;
        }
        memory += extent;
    }
    return 0;
}

static int flex_iov_unpack(int cycles,
                           MPI_Datatype ddt, int count, void* buf,
                           void* packed_buf )
{
    opal_datatype_flexible_storage_t flex_storage = { 0, 0, NULL };
    opal_convertor_t* convertor;
    size_t length;
    int myself, t, c, outsize;
    double timers[trials];

    if ( 0 == (length = ddt->super.size) ) {
        return 0;
    }
    convertor = opal_convertor_create( 0xffffffff, 0 );
    if (OMPI_SUCCESS != opal_convertor_prepare_for_send (convertor,
                                                         &(ddt->super),
                                                         1,
                                                         NULL)) {
        return -1;
    }

    opal_convertor_raw_parser(convertor, flexible_storage_allocator, &flex_storage, &length);
    
    MPI_Comm_rank( MPI_COMM_WORLD, &myself );
    outsize = ddt->super.size * count;

    for( t = 0; t < warmups; t++ ) {
        for( c = 0; c < cycles; c++ ) {
            unpack_from_flex_iovec(flex_storage.storage, flex_storage.iov_pos,
                                   ddt, count, buf, packed_buf);
        }
    }
    
    for( t = 0; t < trials; t++ ) {
        timers[t] = MPI_Wtime();
        for( c = 0; c < cycles; c++ ) {
            unpack_from_flex_iovec(flex_storage.storage, flex_storage.iov_pos,
                                   ddt, count, buf, packed_buf);
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result( outsize, trials, timers );

    return 0;
}

/*********************************************
 *
 *
 *
 *********************************************/

typedef struct my_iovec_storage_s {
    int32_t iov_count;
    int32_t iov_pos;
    struct iovec* iov;
} my_iovec_storage_t;

static int
my_iov_management(void* base, size_t len, void* cb_data)
{
    my_iovec_storage_t* mys = (my_iovec_storage_t*)cb_data;
    if( mys->iov_pos == mys->iov_count ) {
        void* ptr = realloc( mys->iov, (1 + 2 * mys->iov_count) * sizeof(struct iovec));
        if( NULL == ptr ) {  /* oops */
            return 1;
        }
        mys->iov = ptr;
        mys->iov_count = (1 + 2 * mys->iov_count);
    }
    mys->iov[mys->iov_pos].iov_base = base;
    mys->iov[mys->iov_pos].iov_len = len;
    mys->iov_pos++;

    return 0;
}
#if 0
typedef struct working_stack_t {
    int pos;
    int size;
    char* where;
    struct iovec iov[0];
} working_stack_t;

static int my_memcpy(void *restrict dst, const void *restrict src, size_t n, working_stack_t* ws)
{
    if( n > PREFETCH_MEMCPY_LIMIT ) {
        /* force the current memcpy and then complete all pendings */
    } else {
    }
}
#endif
static int
pack_from_iovec(struct iovec* iov, int iov_count,
                MPI_Datatype ddt, int count,
                char* memory, char* packed_buf)
{
    int i, j;
    MPI_Aint lb, extent;

    MPI_Type_get_extent(ddt, &lb, &extent);
    
    for( i = 0; i < count; i++ ) {
        for( j = 0; j < iov_count; j++ ) {
            memcpy( packed_buf, memory + (ptrdiff_t)iov[j].iov_base, iov[j].iov_len );
            packed_buf += iov[j].iov_len;
        }
        memory += extent;
    }
    return 0;
}

static int iov_pack(int cycles,
                    MPI_Datatype ddt, int count, void* buf,
                    void* packed_buf )
{
    my_iovec_storage_t iov_storage = { 0, 0, NULL };
    opal_convertor_t* convertor;
    size_t length;
    int myself, t, c, outsize;
    double timers[trials];

    if ( 0 == (length = ddt->super.size) ) {
        return 0;
    }
    convertor = opal_convertor_create( 0xffffffff, 0 );
    if (OMPI_SUCCESS != opal_convertor_prepare_for_send (convertor,
                                                         &(ddt->super),
                                                         1,
                                                         NULL)) {
        return -1;
    }

    opal_convertor_raw_parser(convertor, my_iov_management, &iov_storage, &length);
#if 0
    for( int i = 0; i < iov_storage.iov_pos; i++ ) {
        printf( "<%p, %" PRIsize_t ">\n", iov_storage.iov[i].iov_base, iov_storage.iov[i].iov_len);
    }
#endif
    
    MPI_Comm_rank( MPI_COMM_WORLD, &myself );
    outsize = ddt->super.size * count;

    for( t = 0; t < warmups; t++ ) {
        for( c = 0; c < cycles; c++ ) {
            pack_from_iovec(iov_storage.iov, iov_storage.iov_pos,
                            ddt, count, buf, packed_buf);
        }
    }
    
    for( t = 0; t < trials; t++ ) {
        timers[t] = MPI_Wtime();
        for( c = 0; c < cycles; c++ ) {
            pack_from_iovec(iov_storage.iov, iov_storage.iov_pos,
                            ddt, count, buf, packed_buf);
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result( outsize, trials, timers );

    return 0;
}

static int
unpack_from_iovec(struct iovec* iov, int iov_count,
                  MPI_Datatype ddt, int count,
                  char* memory, char* packed_buf)
{
    int i, j;
    MPI_Aint lb, extent;

    MPI_Type_get_extent(ddt, &lb, &extent);
    
    for( i = 0; i < count; i++ ) {
        for( j = 0; j < iov_count; j++ ) {
            memcpy( memory + (ptrdiff_t)iov[j].iov_base, packed_buf, iov[j].iov_len );
            packed_buf += iov[j].iov_len;
        }
        memory += extent;
    }
    return 0;
}

static int iov_unpack(int cycles,
                      MPI_Datatype ddt, int count, void* buf,
                      void* packed_buf )
{
    my_iovec_storage_t iov_storage = { 0, 0, NULL };
    opal_convertor_t* convertor;
    size_t length;
    int myself, t, c, outsize;
    double timers[trials];

    if ( 0 == (length = ddt->super.size) ) {
        return 0;
    }
    convertor = opal_convertor_create( 0xffffffff, 0 );
    if (OMPI_SUCCESS != opal_convertor_prepare_for_send (convertor,
                                                         &(ddt->super),
                                                         1,
                                                         NULL)) {
        return -1;
    }

    opal_convertor_raw_parser(convertor, my_iov_management, &iov_storage, &length);
#if 0
    for( int i = 0; i < iov_storage.iov_pos; i++ ) {
        printf( "<%p, %" PRIsize_t ">\n", iov_storage.iov[i].iov_base, iov_storage.iov[i].iov_len);
    }
#endif
    
    MPI_Comm_rank( MPI_COMM_WORLD, &myself );
    outsize = ddt->super.size * count;

    for( t = 0; t < warmups; t++ ) {
        for( c = 0; c < cycles; c++ ) {
            unpack_from_iovec(iov_storage.iov, iov_storage.iov_pos,
                              ddt, count, buf, packed_buf);
        }
    }
    
    for( t = 0; t < trials; t++ ) {
        timers[t] = MPI_Wtime();
        for( c = 0; c < cycles; c++ ) {
            unpack_from_iovec(iov_storage.iov, iov_storage.iov_pos,
                              ddt, count, buf, packed_buf);
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result( outsize, trials, timers );

    return 0;
}

static int pack( int cycles,
                 MPI_Datatype sdt, int scount, void* sbuf,
                 void* packed_buf )
{
    int position, myself, c, t, outsize;
    double timers[trials];

    MPI_Type_size( sdt, &outsize );
    outsize *= scount;

    MPI_Comm_rank( MPI_COMM_WORLD, &myself );

    for( t = 0; t < warmups; t++ ) {
        for( c = 0; c < cycles; c++ ) {
            position = 0;
            MPI_Pack(sbuf, scount, sdt, packed_buf, outsize, &position, MPI_COMM_WORLD);
        }
    }
    
    for( t = 0; t < trials; t++ ) {
        timers[t] = MPI_Wtime();
        for( c = 0; c < cycles; c++ ) {
            position = 0;
            MPI_Pack(sbuf, scount, sdt, packed_buf, outsize, &position, MPI_COMM_WORLD);
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result( outsize, trials, timers );
    return 0;
}

static int unpack( int cycles,
                   void* packed_buf,
                   MPI_Datatype rdt, int rcount, void* rbuf )
{
    int position, myself, c, t, insize;
    double timers[trials];

    MPI_Type_size( rdt, &insize );
    insize *= rcount;

    MPI_Comm_rank( MPI_COMM_WORLD, &myself );

    for( t = 0; t < warmups; t++ ) {
        for( c = 0; c < cycles; c++ ) {
            position = 0;
            MPI_Unpack(packed_buf, insize, &position, rbuf, rcount, rdt, MPI_COMM_WORLD);
        }
    }

    for( t = 0; t < trials; t++ ) {
        timers[t] = MPI_Wtime();
        for( c = 0; c < cycles; c++ ) {
            position = 0;
            MPI_Unpack(packed_buf, insize, &position, rbuf, rcount, rdt, MPI_COMM_WORLD);
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result( insize, trials, timers );
    return 0;
}

static int isend_recv( int cycles,
                       MPI_Datatype sdt, int scount, void* sbuf,
                       MPI_Datatype rdt, int rcount, void* rbuf )
{
    int myself, tag = 0, c, t, slength, rlength;
    MPI_Status status;
    MPI_Request req;
    double timers[trials];

    MPI_Type_size( sdt, &slength );
    slength *= scount;
    MPI_Type_size( rdt, &rlength );
    rlength *= rcount;

    MPI_Comm_rank( MPI_COMM_WORLD, &myself );

    for( t = 0; t < trials; t++ ) {
        timers[t] = MPI_Wtime();
        for( c = 0; c < cycles; c++ ) {
            MPI_Isend( sbuf, scount, sdt, myself, tag, MPI_COMM_WORLD, &req );
            MPI_Recv( rbuf, rcount, rdt, myself, tag, MPI_COMM_WORLD, &status );
            MPI_Wait( &req, &status );
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result( rlength, trials, timers );
    return 0;
}

static int irecv_send( int cycles,
                       MPI_Datatype sdt, int scount, void* sbuf,
                       MPI_Datatype rdt, int rcount, void* rbuf )
{
    int myself, tag = 0, c, t, slength, rlength;
    MPI_Request req;
    MPI_Status status;
    double timers[trials];

    MPI_Type_size( sdt, &slength );
    slength *= scount;
    MPI_Type_size( rdt, &rlength );
    rlength *= rcount;

    MPI_Comm_rank( MPI_COMM_WORLD, &myself );

    for( t = 0; t < trials; t++ ) {
        timers[t] = MPI_Wtime();
        for( c = 0; c < cycles; c++ ) {
            MPI_Irecv( rbuf, rcount, rdt, myself, tag, MPI_COMM_WORLD, &req );
            MPI_Send( sbuf, scount, sdt, myself, tag, MPI_COMM_WORLD );
            MPI_Wait( &req, &status );
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result( rlength, trials, timers );
    return 0;
}

static int isend_irecv_wait( int cycles,
                             MPI_Datatype sdt, int scount, void* sbuf,
                             MPI_Datatype rdt, int rcount, void* rbuf )
{
    int myself, tag = 0, c, t, slength, rlength;
    MPI_Request requests[2];
    MPI_Status statuses[2];
    double timers[trials];

    MPI_Type_size( sdt, &slength );
    slength *= scount;
    MPI_Type_size( rdt, &rlength );
    rlength *= rcount;

    MPI_Comm_rank( MPI_COMM_WORLD, &myself );

    for( t = 0; t < trials; t++ ) {
        timers[t] = MPI_Wtime();
        for( c = 0; c < cycles; c++ ) {
            MPI_Isend( sbuf, scount, sdt, myself, tag, MPI_COMM_WORLD, &requests[0] );
            MPI_Irecv( rbuf, rcount, rdt, myself, tag, MPI_COMM_WORLD, &requests[1] );
            MPI_Waitall( 2, requests, statuses );
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result( rlength, trials, timers );
    return 0;
}

static int irecv_isend_wait( int cycles,
                             MPI_Datatype sdt, int scount, void* sbuf,
                             MPI_Datatype rdt, int rcount, void* rbuf )
{
    int myself, tag = 0, c, t, slength, rlength;
    MPI_Request requests[2];
    MPI_Status statuses[2];
    double timers[trials];

    MPI_Type_size( sdt, &slength );
    slength *= scount;
    MPI_Type_size( rdt, &rlength );
    rlength *= rcount;

    MPI_Comm_rank( MPI_COMM_WORLD, &myself );

    for( t = 0; t < trials; t++ ) {
        timers[t] = MPI_Wtime();
        for( c = 0; c < cycles; c++ ) {
            MPI_Irecv( rbuf, rcount, rdt, myself, tag, MPI_COMM_WORLD, &requests[0] );
            MPI_Isend( sbuf, scount, sdt, myself, tag, MPI_COMM_WORLD, &requests[1] );
            MPI_Waitall( 2, requests, statuses );
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result( rlength, trials, timers);
    return 0;
}

static int do_test_for_ddt( int doop, MPI_Datatype sddt, MPI_Datatype rddt, int length )
{
    MPI_Aint lb, extent, true_lb, true_extent;
    char *sbuf, *rbuf;
    int i, max_count, size;

    MPI_Type_size( sddt, &size );
    MPI_Type_get_extent( sddt, &lb, &extent );
    MPI_Type_get_true_extent( sddt, &true_lb, &true_extent );
    sbuf = (char*)malloc( length );
    rbuf = (char*)malloc( length );
    
    max_count = 1 + (length - true_extent) / extent;
    if( (max_count * size) > length ) {
        max_count = length / size;
    }

    if( doop & DO_PACK ) {
        printf("# Pack (max length %d)\n", length);
        for( i = 1; i <= max_count; i *= 2 ) {
            pack( cycles, sddt, i, sbuf, rbuf );
        }
    }

    if( doop & DO_UNPACK ) {
        printf("# Unpack (length %d)\n", length);
        for( i = 1; i <= max_count; i *= 2 ) {
            unpack( cycles, sbuf, rddt, i, rbuf );
        }
    }

    if( doop & DO_ISEND_RECV ) {
        printf( "# Isend recv (length %d)\n", length );
        for( i = 1; i <= max_count; i *= 2 ) {
            isend_recv( cycles, sddt, i, sbuf, rddt, i, rbuf );
        }
    }

    if( doop & DO_ISEND_IRECV ) {
        printf( "# Isend Irecv Wait (length %d)\n", length );
        for( i = 1; i <= max_count; i *= 2 ) {
            isend_irecv_wait( cycles, sddt, i, sbuf, rddt, i, rbuf );
        }
    }

    if( doop & DO_IRECV_SEND ) {
        printf( "# Irecv send (length %d)\n", length );
        for( i = 1; i <= max_count; i *= 2 ) {
            irecv_send( cycles, sddt, i, sbuf, rddt, i, rbuf );
        }
    }

    if( doop & DO_IRECV_SEND ) {
        printf( "# Irecv Isend Wait (length %d)\n", length );
        for( i = 1; i <= max_count; i *= 2 ) {
            irecv_isend_wait( cycles, sddt, i, sbuf, rddt, i, rbuf );
        }
    }

    if( doop & DO_IOV_PACK ) {
        printf( "# IOV pack (length %d)\n", length );
        for( i = 1; i <= max_count; i *= 2 ) {
            iov_pack( cycles, sddt, i, sbuf, rbuf );
        }
    }

    if( doop & DO_IOV_UNPACK ) {
        printf( "# IOV unpack (length %d)\n", length );
        for( i = 1; i <= max_count; i *= 2 ) {
            iov_unpack( cycles, rddt, i, rbuf, sbuf );
        }
    }

    if( doop & DO_FLEX_IOV_PACK) {
        printf( "# Flex IOV pack (length %d)\n", length );
        for( i = 1; i <= max_count; i *= 2 ) {
            flex_iov_pack( cycles, rddt, i, rbuf, sbuf );
        }
    }

    if( doop & DO_FLEX_IOV_UNPACK) {
        printf( "# Flex IOV unpack (length %d)\n", length );
        for( i = 1; i <= max_count; i *= 2 ) {
            flex_iov_unpack( cycles, rddt, i, rbuf, sbuf );
        }
    }

    free( sbuf );
    free( rbuf );
    return 0;
}

int main( int argc, char* argv[] )
{
    int run_tests = DO_ICON_SEND_DATATYPE | DO_ICON_RECV_DATATYPE;  /* do all datatype tests by default */
    int rank, size;
    MPI_Datatype ddt;

    run_tests |= DO_FLEX_IOV_PACK | DO_FLEX_IOV_UNPACK | DO_IOV_PACK | DO_IOV_UNPACK | DO_PACK | DO_UNPACK;
    
    MPI_Init (&argc, &argv);

    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    if( rank != 0 ) {
        MPI_Finalize();
        exit(0);
    }

    if( run_tests & DO_CONTIG ) {
        printf( "\ncontiguous datatype\n\n" );
        do_test_for_ddt( run_tests, MPI_INT, MPI_INT, MAX_LENGTH );
    }

    if( run_tests & DO_INDEXED_GAP ) {
        printf( "\nindexed gap\n\n" );
        ddt = create_indexed_gap_ddt();
        MPI_DDT_DUMP( ddt );
        do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH );
        MPI_Type_free( &ddt );
    }

    if( run_tests & DO_OPTIMIZED_INDEXED_GAP ) {
        printf( "\noptimized indexed gap\n\n" );
        ddt = create_indexed_gap_optimized_ddt();
        MPI_DDT_DUMP( ddt );
        do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH );
        MPI_Type_free( &ddt );
    }

    if( run_tests & DO_CONSTANT_GAP ) {
        printf( "\nconstant indexed gap\n\n" );
        ddt = create_indexed_constant_gap_ddt( 80, 100, 1 );
        MPI_DDT_DUMP( ddt );
        do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH );
        MPI_Type_free( &ddt );
    }

    if( run_tests & DO_OPTIMIZED_CONSTANT_GAP ) {
        printf( "\noptimized constant indexed gap\n\n" );
        ddt = create_optimized_indexed_constant_gap_ddt( 80, 100, 1 );
        MPI_DDT_DUMP( ddt );
        do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH );
        MPI_Type_free( &ddt );
    }

    if( run_tests & DO_STRUCT_CONSTANT_GAP_RESIZED ) {
        printf( "\nstruct constant gap resized\n\n" );
        ddt = create_struct_constant_gap_resized_ddt( 0 /* unused */, 0 /* unused */, 0 /* unused */ );
        MPI_DDT_DUMP( ddt );
        do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH );
        MPI_Type_free( &ddt );
    }

    if( run_tests & DO_ICON_SEND_DATATYPE ) {
        printf( "\nICON app send datatype\n\n" );
        ddt = ICON_send_datatype( 0 /* unused */, 0 /* unused */, 0 /* unused */ );
        MPI_DDT_DUMP( ddt );
        do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH );
        MPI_Type_free( &ddt );
    }

    if( run_tests & DO_ICON_RECV_DATATYPE ) {
        printf( "\nICON app recv datatype\n\n" );
        ddt = ICON_recv_datatype( 0 /* unused */, 0 /* unused */, 0 /* unused */ );
        MPI_DDT_DUMP( ddt );
        do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH );
        MPI_Type_free( &ddt );
    }

    if( argc > 1 ) {
        struct stat stat;

        for( int i = 1; i < argc; i++ ) {
            if( access(argv[i], R_OK) == -1 ) {
                printf("\n Cannot access datatype description file %s\n", argv[i]);
                continue;
            }
            int fd = open(argv[i], O_RDONLY);
            if( fd == -1 ) {
                printf("\n Cannot open datatype description from file %s\n", argv[i]);
                continue;
            }
            if( fstat(fd, &stat) == -1 ) {
                printf("\n Cannot stat the %s file\n", argv[i]);
                continue;
            }
            void* addr = mmap(NULL, stat.st_size, PROT_READ, MAP_FILE, fd, 0);
            if( MAP_FAILED == addr ) {
                printf("\nCannot map the datatype description file %s\n", argv[i]);
                close(fd);
                continue;
            }
            ompi_datatype_t* ddt = ompi_datatype_create_from_packed_description(addr, ompi_proc_local_proc);
            
            MPI_DDT_DUMP( ddt );
            do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH );
            MPI_Type_free( &ddt );

            munmap(addr, stat.st_size);
            close(fd);
        }
    }

    MPI_Finalize ();
    exit(0);
}

