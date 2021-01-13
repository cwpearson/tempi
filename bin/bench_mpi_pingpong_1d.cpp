//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

/*! \file

    Sends messages of a fixed size between ranks 0-N/2 and N/2-N
*/

#include "../include/cuda_runtime.hpp"
#include "../include/env.hpp"
#include "../include/logging.hpp"
#include "../support/type.hpp"
#include "statistics.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <sstream>

struct BenchResult {
  int64_t bytes;       // number of bytes sent by this rank
  double pingPongTime; // elapsed time
};

/* Each rank i and i + size/2 are paired for a send/recv
   reported result is the min of the max time seen across any rank
*/
BenchResult bench(MPI_Datatype ty,  // message datatype
                  int count,        // number of datatypes
                  bool host,        // host allocations (or device)
                  const int nIters, // iterations to measure
                  const char *name = "<unnamed>") {

  if (ty != MPI_BYTE) {
    MPI_Type_commit(&ty);
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Aint lb, extent;
  MPI_Type_get_extent(ty, &lb, &extent);
  int typeSize;
  MPI_Type_size(ty, &typeSize);

  // create input device allocations
  char *src = {}, *dst = {};
  if (host) {
    src = new char[extent * count];
    dst = new char[extent * count];
  } else {
    CUDA_RUNTIME(cudaSetDevice(0));
    CUDA_RUNTIME(cudaMalloc(&src, extent * count));
    CUDA_RUNTIME(cudaMalloc(&dst, extent * count));
  }

  Statistics stats;
  nvtxRangePush(name);
  double itersStart = MPI_Wtime();
  for (int n = 0; n < nIters; ++n) {
    int position = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (rank < size / 2) {
      MPI_Send(src, count, ty, rank + size / 2, 0, MPI_COMM_WORLD);
      MPI_Recv(dst, count, ty, rank + size / 2, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

    } else {
      MPI_Recv(dst, count, ty, rank - size / 2, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Send(src, count, ty, rank - size / 2, 0, MPI_COMM_WORLD);
    }
    double stop = MPI_Wtime();

    // maximum time across all ranks
    double tmp = stop - start;
    MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    stats.insert(tmp);
  }
  nvtxRangePop();

  if (host) {
    delete[] src;
    delete[] dst;
  } else {
    CUDA_RUNTIME(cudaFree(src));
    CUDA_RUNTIME(cudaFree(dst));
  }

  if (ty != MPI_BYTE) {
    MPI_Type_free(&ty);
  }

  // type is send back and forth
  return BenchResult{.bytes = typeSize * count, .pingPongTime = stats.trimean()};
}

struct Factory1D {
  TypeFactory1D fn;
  const char *name;
};

int main(int argc, char **argv) {

  environment::noTempi = false;
  MPI_Init(&argc, &argv);

  // run on only ranks 0 and 1
  int size, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size % 2 != 0) {
    LOG_FATAL("needs even number of ranks");
  }

  int nIters;
  std::vector<int64_t> totals;
  std::vector<bool> hosts{true, false};

  /* 1D types
   */

  nIters = 200;


  if (0 == rank) {
    std::cout << "desc,host,B/rank,B,elapsed (s),bandwidth/rank "
                 "(MiB/s), bandwidth agg (MiB/s)\n";
  }

  totals = {1,     2,       4,       8,       16,      32,      64,    128,
            256,   512,     1024,    1 << 11, 4096,    1 << 13, 16384, 1 << 15,
            65536, 98304, 1 << 17, 1 << 18, 393216, 1 << 19, 1 << 20, 1 << 21, 1 << 24};

  for (bool host : hosts) {
    for (int total : totals) {

      std::string s = std::to_string(host) + "|" + 
          std::to_string(total);

      if (0 == rank) {
        std::cout << s;
        std::cout << "," << host;
        std::cout << "," << total;
        std::cout << "," << total * size;
        std::cout << std::flush;
      }

      nvtxRangePush(s.c_str());
      BenchResult result = bench(MPI_BYTE, total, host, nIters, s.c_str());
      nvtxRangePop();
      if (0 == rank) {
        std::cout << "," << result.pingPongTime / 2;
        std::cout << "," << 2 * result.bytes / 1024.0 / 1024.0 / result.pingPongTime;
        std::cout << "," << result.bytes * size / 1024.0 / 1024.0 / result.pingPongTime;
        std::cout << "\n";
        std::cout << std::flush;
      }
    }
  }

  MPI_Finalize();
  return 0;
}
