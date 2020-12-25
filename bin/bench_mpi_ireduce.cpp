#include "../include/cuda_runtime.hpp"
#include "../include/env.hpp"
#include "../include/logging.hpp"
#include "statistics.hpp"
#include <array>

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

BenchResult bench(size_t numBytes,
                  bool tempi, // use tempi or not
                  const int nIters, const int mpi_reduction_api_idx,
                  const int mpi_op_idx) {

  // number of overlapping messages
  int tags = 10;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // configure TEMPI
  environment::noTempi = !tempi;

  // create device allocations
  std::vector<void *> srcs(tags, {});
  std::vector<void *> dsts(tags, {});
  CUDA_RUNTIME(cudaSetDevice(0));
  for (int i = 0; i < tags; ++i) {
    CUDA_RUNTIME(cudaMalloc(&srcs[i], numBytes));
    CUDA_RUNTIME(cudaMalloc(&dsts[i], numBytes));
  }
  std::vector<MPI_Request> reqs(tags);

  MPI_Op mpi_op_selected;
  switch (mpi_op_idx) {
  case 0: {
    mpi_op_selected = MPI_MAX;
    break;
  }
  case 1: {
    mpi_op_selected = MPI_MIN;
    break;
  }
  case 2: {
    mpi_op_selected = MPI_SUM;
    break;
  }
  case 3: {
    mpi_op_selected = MPI_PROD;
    break;
  }
  case 4: {
    mpi_op_selected = MPI_LAND;
    break;
  }
  case 5: {
    mpi_op_selected = MPI_BAND;
    break;
  }
  case 6: {
    mpi_op_selected = MPI_LOR;
    break;
  }
  case 7: {
    mpi_op_selected = MPI_BOR;
    break;
  }
  case 8: {
    mpi_op_selected = MPI_LXOR;
    break;
  }
  case 9: {
    mpi_op_selected = MPI_BXOR;
    break;
  }
  case 10: {
    mpi_op_selected = MPI_MAXLOC;
    break;
  }
  case 11: {
    mpi_op_selected = MPI_MINLOC;
    break;
  }
  default: {
    LOG_FATAL("unrecognized MPI_Op");
  }
  }
  Statistics stats;
  nvtxRangePush("loop");
  for (int n = 0; n < nIters; ++n) {
    int position = 0;
    auto start = Clock::now();
    for (int i = 0; i < tags; ++i) {
      if (mpi_reduction_api_idx == 0) {
        MPI_Ireduce(srcs[i], dsts[i], numBytes, MPI_BYTE, mpi_op_selected, 0,
                    MPI_COMM_WORLD, &reqs[i]);
      } else if (mpi_reduction_api_idx == 1) {
        MPI_Iallreduce(srcs[i], dsts[i], numBytes, MPI_BYTE, mpi_op_selected,
                       MPI_COMM_WORLD, &reqs[i]);
      } else if (mpi_reduction_api_idx == 2) {
        MPI_Iscan(srcs[i], dsts[i], numBytes, MPI_BYTE, mpi_op_selected,
                  MPI_COMM_WORLD, &reqs[i]);
      }
    }
    MPI_Waitall(tags, reqs.data(), MPI_STATUS_IGNORE);
    auto stop = Clock::now();
    Duration dur = stop - start;
    stats.insert(dur.count());
  }
  nvtxRangePop();

  for (int i = 0; i < tags; ++i) {
    CUDA_RUNTIME(cudaFree(srcs[i]));
    CUDA_RUNTIME(cudaFree(dsts[i]));
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
  if (size < 2) {
    if (0 == rank) {
      LOG_FATAL("needs at least 2 ranks");
    } else {
      exit(-1);
    }
  }
  if (!(0 == rank || 1 == rank)) {
    goto finalize;
  }

  { // prevent init bypass
    int nIters = 10;

    BenchResult result;

    std::vector<int> ns = {1,       2,       4,       8,       16,
                           32,      64,      128,     256,     512,
                           1024,    1 << 11, 4096,    1 << 13, 16384,
                           1 << 15, 65536,   1 << 17, 1 << 20};

    std::vector<bool> tempis = {true, false};

    constexpr std::array<int, 3> mpi_reduction_apis = {0, 1, 2};
    constexpr std::array<int, 12> mpi_reduction_ops = {0, 1, 2, 3, 4,  5,
                                                       6, 7, 8, 9, 10, 11};

    if (0 == rank) {
      std::cout << "n,tempi, MiB/s\n";
    }

    for (int mpi_reduction_api_idx : mpi_reduction_apis) {
      for (int mpi_reduction_op_idx : mpi_reduction_ops) {
        for (bool tempi : tempis) {
          for (int n : ns) {

            std::string s;
            s = std::to_string(mpi_reduction_api_idx) +
                std::to_string(mpi_reduction_op_idx) + std::to_string(n) + "/" +
                std::to_string(tempi);

            if (0 == rank) {
              std::cout << s;
              std::cout << "," << mpi_reduction_api_idx;
              std::cout << "," << mpi_reduction_op_idx;
              std::cout << "," << n;
              std::cout << "," << tempi;
              std::cout << std::flush;
            }

            nvtxRangePush(s.c_str());
            result = bench(n, tempi, nIters, mpi_reduction_api_idx,
                           mpi_reduction_op_idx);
            nvtxRangePop();
            if (0 == rank) {
              std::cout << ","
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
    }
  }
  LOG_DEBUG("at the end");
finalize:
  environment::noTempi = false; // restore this to the same as Init
  MPI_Finalize();
  return 0;
}
