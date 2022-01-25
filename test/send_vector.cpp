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

  if (size < 2) {
    std::cerr << "ERROR: requires >1 ranks\n";
    return 1;
  }

  if (rank < 2) {

    int numBlocks = 10;
    int blockLength = 8;
    int stride = 16;
    int count = 2;

    MPI_Datatype ty;
    ty = make_2d_byte_vector(numBlocks, blockLength, stride);
    MPI_Type_commit(&ty);

    MPI_Aint lb, ext;
    MPI_Type_get_extent(ty, &lb, &ext);

    char *hostBuf = new char[ext * count];
    char *deviceBuf{};
    cudaMalloc(&deviceBuf, ext * count);

    // host send / recv
    std::cerr << "TEST: host send / recv\n";
    if (0 == rank) {
      MPI_Send(hostBuf, count, ty, 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(hostBuf, count, ty, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // device send / recv
    std::cerr << "TEST: device send / recv\n";
    if (0 == rank) {
      MPI_Send(deviceBuf, count, ty, 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(deviceBuf, count, ty, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    delete[] hostBuf;
    cudaFree(deviceBuf);
    MPI_Type_free(&ty);
  }

  MPI_Finalize();

  return 0;
}