//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "../include/env.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <iostream>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (1 != size) {
    std::cerr << "ERROR: requires exactly 1 rank\n";
    exit(1);
  }

  std::cerr << "TEST: named types\n";

  float *hostSend = new float[100];
  float *hostRecv = new float[100];
  float *deviceSend, *deviceRecv;
  cudaMalloc(&deviceSend, sizeof(float) * 100);
  cudaMalloc(&deviceRecv, sizeof(float) * 100);
  MPI_Request reqSend, reqRecv;

  // host send / recv
  std::cerr << "HOST\n";
  MPI_Isend(hostSend, 100, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, &reqSend);
  MPI_Irecv(hostRecv, 100, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, &reqRecv);
  MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
  MPI_Wait(&reqRecv, MPI_STATUS_IGNORE);

  // device send/recv
  std::cerr << "DEVICE TEMPI\n";
  nvtxRangePush("TEMPI");
  MPI_Isend(deviceSend, 100, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, &reqSend);
  MPI_Irecv(deviceRecv, 100, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, &reqRecv);
  MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
  MPI_Wait(&reqRecv, MPI_STATUS_IGNORE);
  nvtxRangePop();

  environment::noTempi = true;

  nvtxRangePush("noTempi");
  // device send/recv
  MPI_Isend(deviceSend, 100, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, &reqSend);
  MPI_Irecv(deviceRecv, 100, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, &reqRecv);
  MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
  MPI_Wait(&reqRecv, MPI_STATUS_IGNORE);
  nvtxRangePop();

  environment::noTempi = false;
  MPI_Finalize();

  delete[] hostSend;
  delete[] hostRecv;
  cudaFree(deviceSend);
  cudaFree(deviceRecv);

  return 0;
}