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
  uint64_t totalBytes;
};

BenchResult
bench(int64_t scale,
      bool tempi,                          // use tempi or not
      const int nIters) {

        BenchResult result{};

        // all ranks have the same seed to make the same map
  srand(101);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // bytes send i->j
  std::vector<std::vector<int>> bytes;
  for (int i = 0; i < size; ++i) {
    bytes.push_back(std::vector<int>(size, 0));
  }

  uint64_t total = 0, sendTotal, recvTotal;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      int val = (rand() % 10 + 1) * scale;
      bytes[i][j] = val;
      total += val;
    }
  }
  result.totalBytes = total;

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

  Statistics stats;
  nvtxRangePush("loop");
  for (int n = 0; n < nIters; ++n) {
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePush("alltoallv");
    auto start = Clock::now();
    MPI_Alltoallv(src, sendcounts.data(), sdispls.data(), MPI_BYTE, dst,
                  recvcounts.data(), rdispls.data(), MPI_BYTE, MPI_COMM_WORLD);
    auto stop = Clock::now();
    nvtxRangePop();
    Duration dur = stop - start;
    double tmp = dur.count();

    MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    stats.insert(tmp);
  }
  nvtxRangePop();

  CUDA_RUNTIME(cudaFree(src));
  CUDA_RUNTIME(cudaFree(dst));

result.alltoallvTime = stats.trimean();
  return result;
}

int main(int argc, char **argv) {

  environment::noTempi = false;
  MPI_Init(&argc, &argv);

  int size, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (0 == rank) {
    char version[MPI_MAX_LIBRARY_VERSION_STRING] = {};
    int len;
    MPI_Get_library_version(version, &len);
    std::cout << version << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  int nIters = 20;

  BenchResult result;

  std::vector<bool> tempis = {true, false};
  std::vector<int64_t> scales = {1, 1024, 1024ll * 1024ll};

  if (0 == rank) {
    std::cout << "n,tempi,s,MiB/s\n";
  }

  for (bool tempi : tempis) {

    for (int64_t scale : scales) {

      std::string s;
      s = std::to_string(scale) + "|" + std::to_string(tempi);

      if (0 == rank) {
        std::cout << s;
        std::cout << "," << scale;
        std::cout << "," << tempi;
        std::cout << std::flush;
      }

      nvtxRangePush(s.c_str());
      result = bench(scale, tempi, nIters);
      nvtxRangePop();
      if (0 == rank) {
        std::cout << "," << result.alltoallvTime << ","
                  << result.totalBytes / 1024 / 1024 / result.alltoallvTime;
        std::cout << std::flush;
      }

      if (0 == rank) {
        std::cout << "\n";
        std::cout << std::flush;
      }
    }
  }

  environment::noTempi = false; // restore this to the same as Init
  MPI_Finalize();
  return 0;
}
