#include "../include/cuda_runtime.hpp"
#include "../include/env.hpp"
#include "../include/logging.hpp"
#include "statistics.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <sstream>

typedef std::chrono::system_clock Clock;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::time_point<Clock, Duration> Time;

struct BenchResult {
  double alltoallvTime;
  uint64_t numBytes;
};

BenchResult bench(int64_t scale, float density,
                  bool tempi, // use tempi or not
                  const int nIters) {

  BenchResult result{};

  // all ranks have the same seed to make the same map
  srand(101);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // bytes send i->j
  std::vector<int> bytes(size * size, 0);

  // random number of bytes in each bin
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      int val = (rand() % 10 + 1) * scale;
      bytes[i * size + j] = val;
    }
  }

  // zero some bins to match target
  std::vector<size_t> zeroes(bytes.size());
  std::iota(zeroes.begin(), zeroes.end(), 0);
  std::shuffle(zeroes.begin(), zeroes.end(), std::default_random_engine(101));
  for (size_t i = 0; i < zeroes.size() * (1.0 - density); ++i) {
    bytes[zeroes[i]] = 0;
  }

  // communication traffic statistics
  uint64_t sendTotal = 0, recvTotal = 0;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      int val = bytes[i * size + j];
      if (i == rank) {
        sendTotal += val;
      }
      if (j == rank) {
        recvTotal += val;
      }
    }
  }
  if (sendTotal != recvTotal) {
    LOG_FATAL("send/recv total mismatch");
  }
  result.numBytes = sendTotal;

#if 0
  for (int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (r == rank) {
      std::cout << "rank:" << rank << "\n";
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          int val = bytes[i * size + j];
          std::cout << val << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  size_t sendBufSize = 0;
  for (size_t dst = 0; dst < size; ++dst) {
    sendBufSize += bytes[rank * size + dst];
  }
  // LOG_INFO("sendBufSize=" << sendBufSize);

  std::vector<int> sendcounts;
  for (size_t dst = 0; dst < size; ++dst) {
    sendcounts.push_back(bytes[rank * size + dst]);
  }

  std::vector<int> sdispls;
  sdispls.push_back(0);
  for (size_t dst = 1; dst < size; ++dst) {
    sdispls.push_back(sdispls[dst - 1] + sendcounts[dst - 1]);
  }

  size_t recvBufSize = 0;
  for (size_t src = 0; src < size; ++src) {
    recvBufSize += bytes[src * size + rank];
  }
  // LOG_INFO("recvBufSize=" << recvBufSize);

  std::vector<int> recvcounts;
  for (size_t src = 0; src < size; ++src) {
    recvcounts.push_back(bytes[src * size + rank]);
  }
  // LOG_INFO("recvcounts=" << recvcounts[0] << "," << recvcounts[1]);

  std::vector<int> rdispls;
  rdispls.push_back(0);
  for (size_t src = 1; src < size; ++src) {
    rdispls.push_back(rdispls[src - 1] + recvcounts[src - 1]);
  }
  // LOG_INFO("rdispls=" << rdispls[0] << "," << rdispls[1]);

#if 0
  for (int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (r == rank) {
      std::cout << "rank:" << rank << "\n";
      std::cout << "sendcounts=";
      for (int i = 0; i < size; ++i) {
        std::cout << sendcounts[i] << " ";
      }
      std::cout << "\n";
      std::cout << "sdispls=";
      for (int i = 0; i < size; ++i) {
        std::cout << sdispls[i] << " ";
      }
      std::cout << "\n";
      std::cout << "recvcounts=";
      for (int i = 0; i < size; ++i) {
        std::cout << recvcounts[i] << " ";
      }
      std::cout << "\n";
      std::cout << "rdispls=";
      for (int i = 0; i < size; ++i) {
        std::cout << rdispls[i] << " ";
      }
      std::cout << "\n";
      std::cout << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  std::cout << std::flush;
#endif

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

  int nIters = 30;

  BenchResult result;

  std::vector<bool> tempis{true, false};
  std::vector<int64_t> scales{1, 1024, 1024ll * 1024ll};
  std::vector<float> densities{0.05, 0.1, 0.5, 1.0};

  if (0 == rank) {
    std::cout << "description,tempi,B,elapsed (s),aggregate (MiB/s)\n";
  }

  for (bool tempi : tempis) {

    for (int64_t scale : scales) {

      for (float density : densities) {

        std::string s;
        s = std::to_string(scale) + "|" + std::to_string(density) + "|" +
            std::to_string(tempi);

        if (0 == rank) {
          std::cout << s;
          std::cout << "," << scale;
          std::cout << "," << tempi;
          std::cout << std::flush;
        }

        nvtxRangePush(s.c_str());
        result = bench(scale, density, tempi, nIters);
        nvtxRangePop();
        if (0 == rank) {
          std::cout << "," << result.numBytes << "," << result.alltoallvTime
                    << ","
                    << double(result.numBytes) / 1024 / 1024 /
                           result.alltoallvTime;
          std::cout << std::flush;
        }

        if (0 == rank) {
          std::cout << "\n";
          std::cout << std::flush;
        }
      }
    }
  }

  environment::noTempi = false; // restore this to the same as Init
  MPI_Finalize();
  return 0;
}
