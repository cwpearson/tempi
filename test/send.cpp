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
    cudaMalloc(&deviceSend, 1024 * 1024);
    cudaMalloc(&deviceRecv, 1024 * 1024);

    // host send / recv
    std::cerr << "TEST: host send / recv\n";
    if (0 == rank) {
      MPI_Send(hostSend, 100, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(hostRecv, 100, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // device send/recv
    std::cerr << "TEST: device send / recv\n";
    if (0 == rank) {
      MPI_Send(deviceSend, 100, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(deviceRecv, 100, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // device send/recv
    std::cerr << "TEST: device send / recv\n";
    if (0 == rank) {
      MPI_Send(deviceSend, 1024 * 1024, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(deviceRecv, 1024 * 1024, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
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