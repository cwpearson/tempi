//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvToolsExt.h>

#include <iostream>

#include "../include/env.hpp"
#include "../support/type.hpp"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    std::cerr << "ERROR: requires 2 ranks\n";
    return 1;
  }

  for (int n = 1; n < 8 * 1024 * 1024; n *= 2) {
    char *hostBuf = new char[n];
    char *deviceBuf{};
    cudaMalloc(&deviceBuf, n);

    // describe with a big contiguous region
    MPI_Datatype ty = make_contiguous_contiguous(n);
    MPI_Type_commit(&ty);

    // host send / recv
    std::cerr << "TEST: host send / recv\n";
    if (0 == rank) {
      MPI_Send(hostBuf, 1, ty, 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(hostBuf, 1, ty, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // device send/recv
    std::cerr << "TEST: device send / recv\n";
    if (0 == rank) {
      MPI_Send(deviceBuf, 1, ty, 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(deviceBuf, 1, ty, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    std::cerr << "done tests\n";

    delete[] hostBuf;
    cudaFree(deviceBuf);
  }

  MPI_Finalize();

  return 0;
}