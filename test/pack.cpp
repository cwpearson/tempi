#include "support/test.hpp"
#include "support/type.hpp"

#include "cuda_runtime.hpp"

#include <mpi.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Dim3 pe{.x = 2, .y = 3, .z = 4};    // pack extent
  Dim3 ae{.x = 10, .y = 10, .z = 10}; // alloc extent

  void *src = nullptr, *dst = nullptr;
  CUDA_RUNTIME(cudaMallocManaged(&src, ae.x * ae.y * ae.z));
  CUDA_RUNTIME(cudaMallocManaged(&dst, pe.x * pe.y * pe.z));

  MPI_Datatype cube = make_v_hv(pe, ae);
  MPI_Type_commit(&cube);
  int position = 0;
  MPI_Pack(src, 1, cube, dst, pe.flatten(), &position, MPI_COMM_WORLD);

  REQUIRE(position == pe.flatten());

  CUDA_RUNTIME(cudaFree(src));
  CUDA_RUNTIME(cudaFree(dst));
  MPI_Finalize();
  return 0;
}