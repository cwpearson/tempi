#include <mpi.h>

#include "../include/env.hpp"

#include "../support/type.hpp"

int main(int argc, char **argv) {
  environment::noTempi = false;
  MPI_Init(&argc, &argv);

  Dim3 copyExt = {.x = 100, .y = 13, .z = 47};
  Dim3 allocExt = {.x = 256, .y = 512, .z = 1024};

  {
    std::cerr << "TEST: MPI_Type_create_subarray\n";
    MPI_Datatype ty = make_subarray(copyExt, allocExt);
    MPI_Type_commit(&ty);
  }

  {
    std::cerr << "TEST: make_subarray_v\n";
    MPI_Datatype ty = make_subarray_v(copyExt, allocExt);
    MPI_Type_commit(&ty);
  }

  {
    std::cerr << "TEST: v hv\n";
    MPI_Datatype ty = make_byte_v_hv(copyExt, allocExt);
    MPI_Type_commit(&ty);
  }

  {
    std::cerr << "TEST: v hv\n";
    MPI_Datatype ty = make_float_v_hv(copyExt, allocExt);
    MPI_Type_commit(&ty);
  }

  {
    std::cerr << "TEST: v1 hv hv\n";
    MPI_Datatype ty = make_byte_v1_hv_hv(copyExt, allocExt);
    MPI_Type_commit(&ty);
  }

  {
    std::cerr << "TEST: vn hv hv\n";
    MPI_Datatype ty = make_byte_vn_hv_hv(copyExt, allocExt);
    MPI_Type_commit(&ty);
  }

  {
    Dim3 copyExt = {.x = 100, .y = 100, .z = 1};
    Dim3 allocExt = {.x = 100, .y = 100, .z = 100};
    std::cerr << "TEST: v1 hv hv\n";
    MPI_Datatype ty = make_byte_v1_hv_hv(copyExt, allocExt);
    MPI_Type_commit(&ty);
  }

  {
    std::cerr << "TEST: hindexed_block\n";
    MPI_Datatype ty = make_hib(copyExt, allocExt);
    MPI_Type_commit(&ty);
  }

  {
    std::cerr << "TEST: hi\n";
    MPI_Datatype ty = make_hi(copyExt, allocExt);
    MPI_Type_commit(&ty);
  }

  MPI_Finalize();
  return 0;
}