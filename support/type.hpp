#pragma once

#include "../include/dim3.hpp"

#include <mpi.h>

/* use vector + hvector + hvector

vector is n blocks of size 1
 */
MPI_Datatype make_byte_vn_hv_hv(const Dim3 copyExt, // bytes
                                const Dim3 allocExt // bytes
);

/* use vector + hvector + hvector

   vector is 1 block of size n
 */
MPI_Datatype make_byte_v1_hv_hv(const Dim3 copyExt, // bytes
                                const Dim3 allocExt // bytes
);

/* use byte + vector + hvector
 */
MPI_Datatype make_byte_v_hv(const Dim3 copyExt, const Dim3 allocExt);

/* use float + vector + hvector
 */
MPI_Datatype make_float_v_hv(const Dim3 copyExt, const Dim3 allocExt);

/* use hindexed

  each block is a row
 */
MPI_Datatype make_hi(const Dim3 copyExt, const Dim3 allocExt);

/* use hindexed_block

  each block is a row
 */
MPI_Datatype make_hib(const Dim3 copyExt, const Dim3 allocExt);

// make a 3D cube with MPI_Type_create_subarray
MPI_Datatype make_subarray(const Dim3 copyExt, const Dim3 allocExt);


// make a 3D cube with a vector of subarray
MPI_Datatype make_subarray_v(const Dim3 copyExt, const Dim3 allocExt);

// make a 3D cube with MPI_Type_create_subarray and offset
MPI_Datatype make_off_subarray(const Dim3 copyExt, const Dim3 allocExt, const Dim3 &off);

// n contiguous bytes
MPI_Datatype make_contiguous_byte_v1(int n);
MPI_Datatype make_contiguous_byte_vn(int n);
MPI_Datatype make_contiguous_subarray(int n);
MPI_Datatype make_contiguous_contiguous(int n);
