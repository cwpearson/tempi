#include <mpi.h>

#include "support/type.hpp"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  Dim3 copyExt = {.x = 100, .y = 13, .z = 47};
  Dim3 allocExt = {.x = 256, .y = 512, .z = 1024};

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

  {
    std::cerr << "TEST: v hv\n";
    MPI_Datatype ty = make_v_hv(copyExt, allocExt);
    MPI_Type_commit(&ty);
  }

  {
    std::cerr << "TEST: v1 hv hv\n";
    MPI_Datatype ty = make_v1_hv_hv(copyExt, allocExt);
    MPI_Type_commit(&ty);
  }

  {
    std::cerr << "TEST: vn hv hv\n";
    MPI_Datatype ty = make_vn_hv_hv(copyExt, allocExt);
    MPI_Type_commit(&ty);
  }

  MPI_Finalize();

  return 0;
}