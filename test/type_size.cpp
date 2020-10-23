#include <mpi.h>

#include "../support/type.hpp"

#include "test.hpp"

int main(int argc, char **argv) {

  setenv("TEMPI_DISABLE", "a", true);

  MPI_Init(&argc, &argv);

  Dim3 copyExt = {.x = 100, .y = 13, .z = 47};
  Dim3 allocExt = {.x = 256, .y = 512, .z = 1024};

  {
    std::cerr << "TEST: hindexed_block\n";
    MPI_Datatype ty = make_hib(copyExt, allocExt);
    MPI_Type_commit(&ty);
    int size;
    MPI_Type_size(ty, &size);
    REQUIRE(size == copyExt.flatten());
  }

  {
    std::cerr << "TEST: hi\n";
    MPI_Datatype ty = make_hi(copyExt, allocExt);
    MPI_Type_commit(&ty);
    int size;
    MPI_Type_size(ty, &size);
    REQUIRE(size == copyExt.flatten());
  }

  {
    std::cerr << "TEST: v hv\n";
    MPI_Datatype ty = make_byte_v_hv(copyExt, allocExt);
    MPI_Type_commit(&ty);
    int size;
    MPI_Type_size(ty, &size);
    REQUIRE(size == copyExt.flatten());
  }

  {
    std::cerr << "TEST: v1 hv hv\n";
    MPI_Datatype ty = make_v1_hv_hv(copyExt, allocExt);
    MPI_Type_commit(&ty);
    int size;
    MPI_Type_size(ty, &size);
    REQUIRE(size == copyExt.flatten());
  }

  {
    std::cerr << "TEST: vn hv hv\n";
    MPI_Datatype ty = make_vn_hv_hv(copyExt, allocExt);
    MPI_Type_commit(&ty);
    int size;
    MPI_Type_size(ty, &size);
    REQUIRE(size == copyExt.flatten());
  }

  {
    Dim3 copyExt = {.x = 100, .y = 100, .z = 1};
    Dim3 allocExt = {.x = 100, .y = 100, .z = 100};
    std::cerr << "TEST: v1 hv hv\n";
    MPI_Datatype ty = make_v1_hv_hv(copyExt, allocExt);
    MPI_Type_commit(&ty);
    int size;
    MPI_Type_size(ty, &size);
    REQUIRE(size == copyExt.flatten());
  }

  MPI_Finalize();

  unsetenv("TEMPI_DISABLE");

  return 0;
}