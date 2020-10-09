#pragma once

#include "dim3.hpp"

#include <mpi.h>

/* use vector + hvector + hvector

vector is n blocks of size 1
 */
MPI_Datatype make_vn_hv_hv(const Dim3 copyExt, // bytes
                           const Dim3 allocExt // bytes
);

/* use vector + hvector + hvector

   vector is 1 block of size n
 */
MPI_Datatype make_v1_hv_hv(const Dim3 copyExt, // bytes
                           const Dim3 allocExt // bytes
);

/* use vector + hvector
 */
MPI_Datatype make_v_hv(const Dim3 copyExt, const Dim3 allocExt);

/* use hindexed

  each block is a row
 */
MPI_Datatype make_hi(const Dim3 copyExt, const Dim3 allocExt);

/* use hindexed_block

  each block is a row
 */
MPI_Datatype make_hib(const Dim3 copyExt, const Dim3 allocExt);
