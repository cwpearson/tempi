/*! \file

    Sends messages of a fixed size between ranks 0-N/2 and N/2-N
*/

#include "../include/cuda_runtime.hpp"
#include "../include/env.hpp"
#include "../include/logging.hpp"
#include "statistics.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <chrono>
#include <sstream>

typedef std::chrono::system_clock Clock;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::time_point<Clock, Duration> Time;

struct BenchResult {
  double pingPongTime;
};

BenchResult bench(size_t numBytes, // size of each message
                  bool tempi,      // use tempi or not
                  bool host,       // host allocations (or device)
                  const int nIters) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // configure TEMPI
  environment::noTempi = !tempi;

  // create device allocations
  char *src = {}, *dst = {};
  if (host) {
    src = new char[numBytes];
    dst = new char[numBytes];
  } else {
    CUDA_RUNTIME(cudaSetDevice(0));
    CUDA_RUNTIME(cudaMalloc(&src, numBytes));
    CUDA_RUNTIME(cudaMalloc(&dst, numBytes));
  }

  Statistics stats;
  nvtxRangePush("loop");
  for (int n = 0; n < nIters; ++n) {
    int position = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = Clock::now();
    if (rank < size / 2) {
      MPI_Send(src, numBytes, MPI_BYTE, rank + size / 2, 0, MPI_COMM_WORLD);
      MPI_Recv(dst, numBytes, MPI_BYTE, rank + size / 2, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

    } else {
      MPI_Recv(dst, numBytes, MPI_BYTE, rank - size / 2, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Send(src, numBytes, MPI_BYTE, rank - size / 2, 0, MPI_COMM_WORLD);
    }
    auto stop = Clock::now();

    // maximum time across all ranks
    double tmp = Duration(stop - start).count();
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

  return BenchResult{.pingPongTime = stats.trimean()};
}

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

  int nIters = 10;

  BenchResult result;

  std::vector<bool> tempis{true, false};
  std::vector<bool> hosts{true, false};
  std::vector<int> ns{1,     2,       4,     8,       16,      32,     64,
                      128,   256,     512,   1024,    1 << 11, 4096,   1 << 13,
                      16384, 1 << 15, 65536, 1 << 17, 1 << 20, 1 << 24};

  if (0 == rank) {
    std::cout << "desc,tempi,host,B,elapsed (s),bandwidth (MiB/s)\n";
  }

  for (bool tempi : tempis) {
    for (bool host : hosts) {
      for (int n : ns) {

        std::string s;
        s = std::to_string(tempi) + "|" + std::to_string(host) + "|" +
            std::to_string(n);

        if (0 == rank) {
          std::cout << s;
          std::cout << "," << tempi;
          std::cout << "," << host;
          std::cout << "," << n;
          std::cout << std::flush;
        }

        nvtxRangePush(s.c_str());
        result = bench(n, tempi, host, nIters);
        nvtxRangePop();
        if (0 == rank) {
          std::cout << "," << result.pingPongTime << ","
                    << 2 * double(n) / 1024 / 1024 / result.pingPongTime;
          std::cout << std::flush;
        }

        if (0 == rank) {
          std::cout << "\n";
          std::cout << std::flush;
        }
      }
    }
  }
  LOG_DEBUG("at the end");

  environment::noTempi = false; // restore this to the same as Init
  MPI_Finalize();
  return 0;
}
