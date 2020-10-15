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
  double alltoallvTime;
};

BenchResult
bench(std::vector<std::vector<int>> bytes, // bytes from i -> j, square matrix
      bool tempi,                          // use tempi or not
      const int nIters) {

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int sendBufSize = 0;
  for (size_t dst = 0; dst < bytes.size(); ++dst) {
    sendBufSize += bytes[rank][dst];
  }
  // LOG_INFO("sendBufSize=" << sendBufSize);

  std::vector<int> sendcounts;
  for (size_t dst = 0; dst < bytes.size(); ++dst) {
    sendcounts.push_back(bytes[rank][dst]);
  }

  std::vector<int> sdispls;
  sdispls.push_back(0);
  for (size_t dst = 1; dst < bytes.size(); ++dst) {
    sdispls.push_back(sdispls[dst - 1] + sendcounts[dst - 1]);
  }

  int recvBufSize = 0;
  for (size_t src = 0; src < bytes.size(); ++src) {
    recvBufSize += bytes[src][rank];
  }
  // LOG_INFO("recvBufSize=" << recvBufSize);

  std::vector<int> recvcounts;
  for (size_t src = 0; src < bytes.size(); ++src) {
    recvcounts.push_back(bytes[src][rank]);
  }
  // LOG_INFO("recvcounts=" << recvcounts[0] << "," << recvcounts[1]);

  std::vector<int> rdispls;
  rdispls.push_back(0);
  for (size_t src = 1; src < bytes.size(); ++src) {
    rdispls.push_back(rdispls[src - 1] + recvcounts[src - 1]);
  }
  // LOG_INFO("rdispls=" << rdispls[0] << "," << rdispls[1]);

  // configure TEMPI
  environment::noTempi = !tempi;

  // create device allocations
  char *src = {}, *dst = {};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc(&src, sendBufSize));
  CUDA_RUNTIME(cudaMalloc(&dst, recvBufSize));

  std::vector<MPI_Request> sendReqs(bytes.size());
  std::vector<MPI_Request> recvReqs(bytes.size());

  Statistics stats;
  nvtxRangePush("loop");
  for (int n = 0; n < nIters; ++n) {
    int position = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = Clock::now();
    nvtxRangePush("alltoallv");
    MPI_Alltoallv(src, sendcounts.data(), sdispls.data(), MPI_BYTE, dst,
                  recvcounts.data(), rdispls.data(), MPI_BYTE, MPI_COMM_WORLD);

#if 0
    for (int j = 0; j < bytes.size(); ++j) {
      MPI_Isend(src + sdispls[j], sendcounts[j], MPI_BYTE, j, 0, MPI_COMM_WORLD,
                &sendReqs[j]);
    }
    for (int i = 0; i < bytes.size(); ++i) {
      MPI_Irecv(dst + rdispls[i], recvcounts[i], MPI_BYTE, i, 0, MPI_COMM_WORLD,
                &recvReqs[i]);
    }
    MPI_Waitall(sendReqs.size(), sendReqs.data(), MPI_STATUS_IGNORE);
    MPI_Waitall(recvReqs.size(), recvReqs.data(), MPI_STATUS_IGNORE);
#endif

    auto stop = Clock::now();
    nvtxRangePop();
    Duration dur = stop - start;
    stats.insert(dur.count());
  }
  nvtxRangePop();

  CUDA_RUNTIME(cudaFree(src));
  CUDA_RUNTIME(cudaFree(dst));

  return BenchResult{.alltoallvTime = stats.trimean()};
}

int main(int argc, char **argv) {

  environment::noTempi = false;
  MPI_Init(&argc, &argv);

  // run on only ranks 0 and 1
  int size, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<std::vector<int>> map;
  for (int i = 0; i < size; ++i) {
    map.push_back(std::vector<int>(size, 0));
  }

  srand(101);

  int total = 0;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      int val = (rand() % 10 + 1) * 1024 * 1024;
      map[i][j] = val;
      if (0 == rank)
        std::cout << val << " ";
      total += val;
    }
    if (0 == rank)
      std::cout << "\n";
  }

  int nIters = 10;

  BenchResult result;

  std::vector<bool> tempis = {true, false};

  if (0 == rank) {
    std::cout << "n,tempi, MiB/s\n";
  }

  for (bool tempi : tempis) {

    std::string s;
    s = std::to_string(total) + "/" + std::to_string(tempi);

    if (0 == rank) {
      std::cout << s;
      std::cout << "," << total;
      std::cout << "," << tempi;
      std::cout << std::flush;
    }

    nvtxRangePush(s.c_str());
    result = bench(map, tempi, nIters);
    nvtxRangePop();
    if (0 == rank) {
      std::cout << "," << double(total) / 1024 / 1024 / result.alltoallvTime;
      std::cout << std::flush;
    }

    if (0 == rank) {
      std::cout << "\n";
      std::cout << std::flush;
    }
  }

  environment::noTempi = false; // restore this to the same as Init
  MPI_Finalize();
  return 0;
}
