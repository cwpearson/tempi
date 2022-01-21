//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvToolsExt.h>

#include <iostream>

#include "../include/env.hpp"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    std::cerr << "requires >1 ranks\n";
    return 1;
  }

  if (rank < 2) {

    char *hostSend = new char[1024 * 1024];
    char *hostRecv = new char[1024 * 1024];
    char *deviceSend, *deviceRecv;
    cudaError_t err = cudaMalloc(&deviceSend, 1024 * 1024);
    if (cudaSuccess != err) {
      std::cerr << "failed to allocate device memory\n";
      return 1;
    }
    err = cudaMalloc(&deviceRecv, 1024 * 1024);
    if (cudaSuccess != err) {
      std::cerr << "failed to allocate device memory\n";
      return 1;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // device send/recv
    std::cerr << "TEST: device send / recv 2\n";
    if (0 == rank) {
      MPI_Send(deviceSend, 1024 * 1024, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(deviceRecv, 1024 * 1024, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // host send / recv
    std::cerr << rank << "TEST: host send / recv\n";
    if (0 == rank) {
      MPI_Send(hostSend, 100, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(hostRecv, 100, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // device send/recv
    std::cerr << rank << " TEST: device send / recv 1\n";
    if (0 == rank) {
      MPI_Send(deviceSend, 100, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
      std::cerr << rank << " returned from MPI_Send\n";
    } else {
      MPI_Recv(deviceRecv, 100, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
              MPI_STATUS_IGNORE);
      std::cerr << rank << " returned from MPI_Recv\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);


    std::cerr << "done tests\n";

    delete[] hostSend;
    delete[] hostRecv;
    cudaFree(deviceSend);
    cudaFree(deviceRecv);
  }

  MPI_Finalize();

  return 0;
}