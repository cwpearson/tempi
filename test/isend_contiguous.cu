//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "../include/env.hpp"
#include "../support/type.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <iostream>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  std::cerr << "TEST: contiguous\n";
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int n = 800;

  MPI_Datatype ty = make_contiguous_contiguous(n);
  MPI_Type_commit(&ty);

  char *hostSend = new char[n];
  char *hostRecv = new char[n];
  char *deviceSend, *deviceRecv;
  cudaMalloc(&deviceSend, sizeof(char) * n);
  cudaMalloc(&deviceRecv, sizeof(char) * n);
  MPI_Request reqSend, reqRecv;

  // host send / recv
  std::cerr << "HOST\n";
  MPI_Isend(hostSend, 1, ty, rank, 0, MPI_COMM_WORLD, &reqSend);
  MPI_Irecv(hostRecv, 1, ty, rank, 0, MPI_COMM_WORLD, &reqRecv);
  MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
  MPI_Wait(&reqRecv, MPI_STATUS_IGNORE);

  // device send/recv
  std::cerr << "DEVICE\n";
  nvtxRangePush("device");
  MPI_Isend(deviceSend, 1, ty, rank, 0, MPI_COMM_WORLD, &reqSend);
  MPI_Irecv(deviceRecv, 1, ty, rank, 0, MPI_COMM_WORLD, &reqRecv);
  MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
  MPI_Wait(&reqRecv, MPI_STATUS_IGNORE);
  nvtxRangePop();

#if 0
  environment::noTempi = true;

  nvtxRangePush("noTempi");
  // device send/recv
  MPI_Isend(deviceSend, 1, ty, rank, 0, MPI_COMM_WORLD, &reqSend);
  MPI_Irecv(deviceRecv, 1, ty, rank, 0, MPI_COMM_WORLD, &reqRecv);
  MPI_Wait(&reqSend, MPI_STATUS_IGNORE);
  MPI_Wait(&reqRecv, MPI_STATUS_IGNORE);
  nvtxRangePop();

  environment::noTempi = false;
#endif
  MPI_Finalize();

  delete[] hostSend;
  delete[] hostRecv;
  cudaFree(deviceSend);
  cudaFree(deviceRecv);

  return 0;
}