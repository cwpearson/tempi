#include <cuda_runtime.h>
#include <mpi.h>
#include <nvToolsExt.h>

#include "../include/env.hpp"

int main(int argc, char **argv) {
  environment::noTempi = false;
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank < 2) {

    char *hostSend = new char[1024 * 1024];
    char *hostRecv = new char[1024 * 1024];
    char *deviceSend, *deviceRecv;
    cudaMalloc(&deviceSend, 1024 * 1024);
    cudaMalloc(&deviceRecv, 1024 * 1024);

    nvtxRangePush("TEMPI");
    // host send / recv
    if (0 == rank) {
      MPI_Send(hostSend, 100, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(hostRecv, 100, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    // device send/recv
    if (0 == rank) {
      MPI_Send(deviceSend, 100, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(deviceRecv, 100, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    // device send/recv
    if (0 == rank) {
      MPI_Send(deviceSend, 1024 * 1024, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(deviceRecv, 1024 * 1024, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    nvtxRangePop(); // TEMPI

    delete[] hostSend;
    delete[] hostRecv;
    cudaFree(deviceSend);
    cudaFree(deviceRecv);
  }

  environment::noTempi = false;
  MPI_Finalize();

  return 0;
}