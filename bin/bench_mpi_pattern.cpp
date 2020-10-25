/*! \file
    measure various MPI methods for achiving the same communication pattern
*/

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
  double minTime;
  uint64_t numBytes;
};

class SquareMat {
private:
  unsigned n_;

public:
  std::vector<int> data_;

public:
  SquareMat(int n, int val) : n_(n), data_(n * n, val) {}
  unsigned size() const noexcept { return n_; }

  int *operator[](size_t i) noexcept { return &data_[i * n_]; }
  const int *operator[](size_t i) const noexcept { return &data_[i * n_]; }
};

/* create a `ranks` x `ranks` matrix with `rowNz` in each row.
   each value will be in [`lb`, `ub`) * `scale`
 */
SquareMat make_random_sparse(int ranks, int rowNnz, int lb, int ub, int scale) {

  const int SEED = 101;
  srand(SEED);

  SquareMat mat(ranks, 0);

  std::vector<size_t> rowInd(ranks);
  std::iota(rowInd.begin(), rowInd.end(), 0);

  for (int r = 0; r < ranks; ++r) {
    // selet this row's nonzeros
    std::vector<size_t> nzs;
    std::shuffle(rowInd.begin(), rowInd.end(),
                 std::default_random_engine(SEED));
    for (size_t i = 0; i < rowNnz; ++i) {
      nzs.push_back(rowInd[i]);
    }

    for (auto c : nzs) {
      int val = (lb + rand() % (ub - lb)) * scale;
      mat[r][c] = val;
    }
  }
  return mat;
}

typedef BenchResult (*BenchFn)(const SquareMat &mat, const int nIters);

struct Benchmark {
  BenchFn fn;
  const char *name;
};

/* use MPI_Alltoallv
 */
BenchResult bench_alltoallv(const SquareMat &mat, const int nIters) {
  BenchResult result{};
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (mat.size() != size) {
    LOG_FATAL("size mismatch");
    exit(1);
  }

  // create GPU allocations
  size_t sendBufSize = 0, recvBufSize = 0;
  for (size_t i = 0; i < size; ++i) {
    sendBufSize += mat[rank][i];
    recvBufSize += mat[i][rank];
  }

  // create device allocations
  char *srcBuf = {}, *dstBuf = {};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc(&srcBuf, sendBufSize));
  CUDA_RUNTIME(cudaMalloc(&dstBuf, recvBufSize));

  // create Alltoallv arguments
  std::vector<int> sendcounts, recvcounts, sdispls, rdispls;
  for (size_t dst = 0; dst < size; ++dst) {
    sendcounts.push_back(mat[rank][dst]);
  }
  sdispls.push_back(0);
  for (size_t dst = 1; dst < size; ++dst) {
    sdispls.push_back(sdispls[dst - 1] + sendcounts[dst - 1]);
  }
  for (size_t src = 0; src < size; ++src) {
    recvcounts.push_back(mat[src][rank]);
  }
  rdispls.push_back(0);
  for (size_t src = 1; src < size; ++src) {
    rdispls.push_back(rdispls[src - 1] + recvcounts[src - 1]);
  }

  // benchmark loop
  Statistics stats;
  for (int i = 0; i < nIters; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePush("alltoallv");
    auto start = Clock::now();
    MPI_Alltoallv(srcBuf, sendcounts.data(), sdispls.data(), MPI_BYTE, dstBuf,
                  recvcounts.data(), rdispls.data(), MPI_BYTE, MPI_COMM_WORLD);
    auto stop = Clock::now();
    nvtxRangePop();
    Duration dur = stop - start;
    double tmp = dur.count();

    MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    stats.insert(tmp);
  }

  result.minTime = stats.min();
  result.numBytes = sendBufSize;
  MPI_Allreduce(MPI_IN_PLACE, &result.numBytes, 1, MPI_UINT64_T, MPI_SUM,
                MPI_COMM_WORLD);

  CUDA_RUNTIME(cudaFree(srcBuf));
  CUDA_RUNTIME(cudaFree(dstBuf));

  return result;
}

/* use isend/irecv from a single buffer
 */
BenchResult bench_isend_irecv(const SquareMat &mat, const int nIters) {
  BenchResult result{};
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (mat.size() != size) {
    LOG_FATAL("size mismatch");
    exit(1);
  }

  // create GPU allocations
  size_t sendBufSize = 0, recvBufSize = 0;
  for (size_t i = 0; i < size; ++i) {
    sendBufSize += mat[rank][i];
    recvBufSize += mat[i][rank];
  }

  // create device allocations
  char *srcBuf = {}, *dstBuf = {};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc(&srcBuf, sendBufSize));
  CUDA_RUNTIME(cudaMalloc(&dstBuf, recvBufSize));

  // create Alltoallv arguments
  std::vector<int> sendcounts, recvcounts, sdispls, rdispls;
  for (size_t dst = 0; dst < size; ++dst) {
    sendcounts.push_back(mat[rank][dst]);
  }
  sdispls.push_back(0);
  for (size_t dst = 1; dst < size; ++dst) {
    sdispls.push_back(sdispls[dst - 1] + sendcounts[dst - 1]);
  }
  for (size_t src = 0; src < size; ++src) {
    recvcounts.push_back(mat[src][rank]);
  }
  rdispls.push_back(0);
  for (size_t src = 1; src < size; ++src) {
    rdispls.push_back(rdispls[src - 1] + recvcounts[src - 1]);
  }

  std::vector<MPI_Request> sendReq(size, MPI_REQUEST_NULL);
  std::vector<MPI_Request> recvReq(size, MPI_REQUEST_NULL);

  // benchmark loop
  Statistics stats;
  for (int i = 0; i < nIters; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePush("alltoallv");
    auto start = Clock::now();

    for (size_t dst = 0; dst < size; ++dst) {
      MPI_Isend(srcBuf + sdispls[dst], sendcounts[dst], MPI_BYTE, dst, 0,
                MPI_COMM_WORLD, &sendReq[dst]);
    }
    for (size_t src = 0; src < size; ++src) {
      MPI_Irecv(dstBuf + rdispls[src], recvcounts[src], MPI_BYTE, src, 0,
                MPI_COMM_WORLD, &recvReq[src]);
    }
    MPI_Waitall(sendReq.size(), sendReq.data(), MPI_STATUS_IGNORE);
    MPI_Waitall(recvReq.size(), recvReq.data(), MPI_STATUS_IGNORE);
    auto stop = Clock::now();
    nvtxRangePop();
    Duration dur = stop - start;
    double tmp = dur.count();

    MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    stats.insert(tmp);
  }

  result.minTime = stats.min();
  result.numBytes = sendBufSize;
  MPI_Allreduce(MPI_IN_PLACE, &result.numBytes, 1, MPI_UINT64_T, MPI_SUM,
                MPI_COMM_WORLD);

  CUDA_RUNTIME(cudaFree(srcBuf));
  CUDA_RUNTIME(cudaFree(dstBuf));

  return result;
}

/* use isend/irecv from a single buffer, but don't do anything for zero-size
 */
BenchResult bench_sparse_isend_irecv(const SquareMat &mat, const int nIters) {
  BenchResult result{};
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (mat.size() != size) {
    LOG_FATAL("size mismatch");
    exit(1);
  }

  // create GPU allocations
  size_t sendBufSize = 0, recvBufSize = 0;
  for (size_t i = 0; i < size; ++i) {
    sendBufSize += mat[rank][i];
    recvBufSize += mat[i][rank];
  }

  // create device allocations
  char *srcBuf = {}, *dstBuf = {};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc(&srcBuf, sendBufSize));
  CUDA_RUNTIME(cudaMalloc(&dstBuf, recvBufSize));

  // create Alltoallv arguments
  std::vector<int> sendcounts, recvcounts, sdispls, rdispls;
  for (size_t dst = 0; dst < size; ++dst) {
    sendcounts.push_back(mat[rank][dst]);
  }
  sdispls.push_back(0);
  for (size_t dst = 1; dst < size; ++dst) {
    sdispls.push_back(sdispls[dst - 1] + sendcounts[dst - 1]);
  }
  for (size_t src = 0; src < size; ++src) {
    recvcounts.push_back(mat[src][rank]);
  }
  rdispls.push_back(0);
  for (size_t src = 1; src < size; ++src) {
    rdispls.push_back(rdispls[src - 1] + recvcounts[src - 1]);
  }

  std::vector<MPI_Request> sendReq(size, MPI_REQUEST_NULL);
  std::vector<MPI_Request> recvReq(size, MPI_REQUEST_NULL);

  // benchmark loop
  Statistics stats;
  for (int i = 0; i < nIters; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePush("alltoallv");
    auto start = Clock::now();

    for (size_t dst = 0; dst < size; ++dst) {
      if (0 != sendcounts[dst])
        MPI_Isend(srcBuf + sdispls[dst], sendcounts[dst], MPI_BYTE, dst, 0,
                  MPI_COMM_WORLD, &sendReq[dst]);
    }
    for (size_t src = 0; src < size; ++src) {
      if (0 != recvcounts[src])
        MPI_Irecv(dstBuf + rdispls[src], recvcounts[src], MPI_BYTE, src, 0,
                  MPI_COMM_WORLD, &recvReq[src]);
    }
    MPI_Waitall(sendReq.size(), sendReq.data(), MPI_STATUS_IGNORE);
    MPI_Waitall(recvReq.size(), recvReq.data(), MPI_STATUS_IGNORE);
    auto stop = Clock::now();
    nvtxRangePop();
    Duration dur = stop - start;
    double tmp = dur.count();

    MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    stats.insert(tmp);
  }

  result.minTime = stats.min();
  result.numBytes = sendBufSize;
  MPI_Allreduce(MPI_IN_PLACE, &result.numBytes, 1, MPI_UINT64_T, MPI_SUM,
                MPI_COMM_WORLD);

  CUDA_RUNTIME(cudaFree(srcBuf));
  CUDA_RUNTIME(cudaFree(dstBuf));

  return result;
}

/* use isend/irecv from a single buffer, but don't do anything for zero-size
 */
BenchResult bench_dist_graph_reorder(const SquareMat &mat, const int nIters) {
  BenchResult result{};
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (mat.size() != size) {
    LOG_FATAL("size mismatch");
    exit(1);
  }

  // if the matrix is empty, bail
  bool empty = true;
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      if (0 != mat[i][j]) {
        empty = false;
      }
    }
  }
  if (empty) {
    return result;
  }

#if 0
  if (0 == rank) {
    std::cerr << "\nmat\n";
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        std::cerr << mat[i][j] << " ";
      }
      std::cerr << "\n";
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // create the new communicator
  MPI_Comm graph;
  std::vector<int> sources;
  std::vector<int> sourceweights;
  std::vector<int> destinations;
  std::vector<int> destweights;
  int graphRank;

  {
    for (size_t i = 0; i < size; ++i) {
      if (0 != mat[rank][i]) {
        destinations.push_back(i);
        destweights.push_back(mat[rank][i]);
      }
      if (0 != mat[i][rank]) {
        sources.push_back(i);
        sourceweights.push_back(mat[i][rank]);
      }
    }

// print sources
#if 0
    for (int i = 0; i < size; ++i) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (i == rank) {
        std::cerr << "rank " << rank << " sources: ";
        for (auto &e : sources) {
          std::cerr << e << " ";
        }
        std::cerr << "\n";
      }
    }
#endif

// print destinations
#if 0
    for (int i = 0; i < size; ++i) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (i == rank) {
        std::cerr << "rank " << rank << " destinations: ";
        for (auto &e : destinations) {
          std::cerr << e << " ";
        }
        std::cerr << "\n";
      }
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Dist_graph_create_adjacent(
        MPI_COMM_WORLD, sources.size(), sources.data(), sourceweights.data(),
        destinations.size(), destinations.data(), destweights.data(),
        MPI_INFO_NULL, 1 /*reorder*/, &graph);
    MPI_Comm_rank(graph, &graphRank);
  }

  // map MPI_COMM_WORLD rank to graph rank
  std::vector<int> graphRanks(size); // get graph rank from world rank
  std::vector<int> worldRanks(size); // get world rank from graph rank
  MPI_Allgather(&graphRank, 1, MPI_INT, graphRanks.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
  MPI_Allgather(&rank, 1, MPI_INT, worldRanks.data(), 1, MPI_INT, graph);

// debug print
#if 0
  {
    if (0 == rank) {
      std::cerr << "graphRanks: ";
      for (auto &e : graphRanks) {
        std::cerr << e << " ";
      }
      std::cerr << "\n";

      std::cerr << "worldRanks: ";
      for (auto &e : worldRanks) {
        std::cerr << e << " ";
      }
      std::cerr << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  // update neighbor orders (it is not defined to be the same as graph factory
  MPI_Dist_graph_neighbors(graph, sources.size(), sources.data(),
                           sourceweights.data(), destinations.size(),
                           destinations.data(), destweights.data());

  // create GPU allocations
  size_t sendBufSize = 0;
  for (size_t i = 0; i < size; ++i) {
    sendBufSize += mat[rank][i];
  }
  size_t recvBufSize = 0;
  for (size_t i = 0; i < size; ++i) {
    recvBufSize += mat[i][rank];
  }

// debug print
#if 0
  {
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < size; ++i) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (i == rank) {
        std::cerr << "rank " << rank << " sendBufSize,recvBufSize=";
        std::cerr << sendBufSize << "," << recvBufSize << "\n";
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  // create device allocations
  char *sendbuf = {}, *recvbuf = {};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc(&sendbuf, sendBufSize));
  CUDA_RUNTIME(cudaMalloc(&recvbuf, recvBufSize));

  // create Alltoallv arguments
  std::vector<int> sendcounts, sdispls, recvcounts, rdispls;
  for (size_t i = 0; i < destinations.size(); ++i) {
    const int dest = destinations[i];
    sendcounts.push_back(mat[rank][worldRanks[dest]]);
  }
  sdispls.push_back(0);
  for (size_t i = 1; i < destinations.size(); ++i) {
    sdispls.push_back(sdispls[i - 1] + sendcounts[i - 1]);
  }
  for (size_t i = 0; i < sources.size(); ++i) {
    const int src = sources[i];
    recvcounts.push_back(mat[worldRanks[src]][rank]);
  }
  rdispls.push_back(0);
  for (size_t i = 1; i < sources.size(); ++i) {
    rdispls.push_back(rdispls[i - 1] + recvcounts[i - 1]);
  }

  // benchmark loop
  Statistics stats;
  for (int i = 0; i < nIters; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePush("alltoallv");
    auto start = Clock::now();
    MPI_Neighbor_alltoallv(sendbuf, sendcounts.data(), sdispls.data(), MPI_BYTE,
                           recvbuf, recvcounts.data(), rdispls.data(), MPI_BYTE,
                           graph);
    auto stop = Clock::now();
    nvtxRangePop();
    Duration dur = stop - start;
    double tmp = dur.count();

    MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    stats.insert(tmp);
  }

  result.minTime = stats.min();
  result.numBytes = sendBufSize;
  MPI_Allreduce(MPI_IN_PLACE, &result.numBytes, 1, MPI_UINT64_T, MPI_SUM,
                MPI_COMM_WORLD);

  CUDA_RUNTIME(cudaFree(sendbuf));
  CUDA_RUNTIME(cudaFree(recvbuf));

  return result;
}

int main(int argc, char **argv) {

  environment::noTempi = false; // enable at init
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

  std::vector<Benchmark> benchmarks{
      // clang-format off
      {bench_alltoallv, "alltoallv"},
      {bench_isend_irecv, "isend_irecv"},
      {bench_sparse_isend_irecv, "sp_isend_irecv"},
      {bench_dist_graph_reorder, "dist_graph_reorder"}
      // clang-format on
  };

  MPI_Barrier(MPI_COMM_WORLD);

  int nIters = 30;

  std::vector<bool> tempis{true, false};
  std::vector<int64_t> scales{1,         10,         100,        1 * 1000,
                              10 * 1000, 100 * 1000, 1000 * 1000};
  std::vector<float> densities{1.0, 0.5, 0.1, 0.05};

  // add some more densities to target particular nnz per row
  for (int targetNnzPerRow : {1, 2, 4, 8, 16}) {
    float density = double(targetNnzPerRow) / size;
    if (density <= 1) {
      densities.push_back(density);
    }
  }

  if (0 == rank) {
    std::cout << "description,name,tempi,scale,density,B,elapsed (s),aggregate "
                 "(MiB/s)\n";
  }

  for (bool tempi : tempis) {
    environment::noTempi = !tempi;

    for (Benchmark benchmark : benchmarks) {

      for (int64_t scale : scales) {

        for (float density : densities) {

          SquareMat mat =
              make_random_sparse(size, size * density + 0.5, 1, 10, scale);
          MPI_Bcast(mat.data_.data(), mat.data_.size(), MPI_INT, 0,
                    MPI_COMM_WORLD);

          std::string s;
          s = std::string(benchmark.name) + "|" + std::to_string(tempi) + "|" +
              std::to_string(scale) + "|" + std::to_string(density);

          if (0 == rank) {
            std::cout << s;
            std::cout << "," << benchmark.name;
            std::cout << "," << tempi;
            std::cout << "," << scale;
            std::cout << "," << density;
            std::cout << std::flush;
          }

          nvtxRangePush(s.c_str());
          BenchResult result = benchmark.fn(mat, nIters);
          nvtxRangePop();
          if (0 == rank) {
            std::cout << "," << result.numBytes << "," << result.minTime << ","
                      << double(result.numBytes) / 1024 / 1024 / result.minTime;
            std::cout << std::flush;
          }

          if (0 == rank) {
            std::cout << "\n";
            std::cout << std::flush;
          }
          MPI_Barrier(MPI_COMM_WORLD);
        }
      }
    }
  }

  environment::noTempi = false; // enable since it was enabled at init
  MPI_Finalize();
  return 0;
}
