#include "benchmark.hpp"

#include "../include/cuda_runtime.hpp"
#include "../include/env.hpp"
#include "../include/logging.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvToolsExt.h>

BM::Result BM::Method_alltoallv::operator()(const SquareMat &mat,
                                            const int nIters) {

  BM::Result result{};
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
    nvtxRangePush("alltoallv");
    MPI_Barrier(MPI_COMM_WORLD);
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
  result.iters = stats;

  CUDA_RUNTIME(cudaFree(srcBuf));
  CUDA_RUNTIME(cudaFree(dstBuf));

  return result;
}

BM::Result BM::Method_isend_irecv::operator()(const SquareMat &mat,
                                              const int nIters) {

  BM::Result result{};
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
  char *srcBuf{}, *dstBuf{}, *expBuf{};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc(&srcBuf, sendBufSize));
  CUDA_RUNTIME(cudaMalloc(&dstBuf, recvBufSize));
  CUDA_RUNTIME(cudaMalloc(&expBuf, recvBufSize)); // expected

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

  // generate expected result
  {
    environment::noTempi = true;
    MPI_Alltoallv(srcBuf, sendcounts.data(), sdispls.data(), MPI_BYTE, expBuf,
                  recvcounts.data(), rdispls.data(), MPI_BYTE, MPI_COMM_WORLD);
    environment::noTempi = false;
  }

  // benchmark loop
  Statistics stats;
  for (int i = 0; i < nIters; ++i) {
    nvtxRangePush(name());
    MPI_Barrier(MPI_COMM_WORLD);
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
  result.iters = stats;

  // compare results
  {
    std::vector<char> act(recvBufSize), exp(recvBufSize);
    CUDA_RUNTIME(
        cudaMemcpy(act.data(), dstBuf, recvBufSize, cudaMemcpyDeviceToHost));
    CUDA_RUNTIME(
        cudaMemcpy(exp.data(), expBuf, recvBufSize, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < act.size(); ++i) {
      if (act[i] != exp[i]) {
        LOG_FATAL("mismatch at byte" << i);
      }
    }
  }

  CUDA_RUNTIME(cudaFree(srcBuf));
  CUDA_RUNTIME(cudaFree(dstBuf));
  CUDA_RUNTIME(cudaFree(expBuf));

  return result;
}

BM::Result BM::Method_sparse_isend_irecv::operator()(const SquareMat &mat,
                                                     const int nIters) {

  BM::Result result{};
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
  char *srcBuf{}, *dstBuf{}, *expBuf{};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc(&srcBuf, sendBufSize));
  CUDA_RUNTIME(cudaMalloc(&dstBuf, recvBufSize));
  CUDA_RUNTIME(cudaMalloc(&expBuf, recvBufSize)); // expected

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

  // generate expected result
  {
    environment::noTempi = true;
    MPI_Alltoallv(srcBuf, sendcounts.data(), sdispls.data(), MPI_BYTE, expBuf,
                  recvcounts.data(), rdispls.data(), MPI_BYTE, MPI_COMM_WORLD);
    environment::noTempi = false;
  }

  // benchmark loop
  Statistics stats;
  for (int i = 0; i < nIters; ++i) {
    nvtxRangePush("alltoallv");
    MPI_Barrier(MPI_COMM_WORLD);
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

  result.iters = stats;

  // compare results
  {
    std::vector<char> act(recvBufSize), exp(recvBufSize);
    CUDA_RUNTIME(
        cudaMemcpy(act.data(), dstBuf, recvBufSize, cudaMemcpyDeviceToHost));
    CUDA_RUNTIME(
        cudaMemcpy(exp.data(), expBuf, recvBufSize, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < act.size(); ++i) {
      if (act[i] != exp[i]) {
        LOG_FATAL("mismatch at byte" << i);
      }
    }
  }

  CUDA_RUNTIME(cudaFree(srcBuf));
  CUDA_RUNTIME(cudaFree(dstBuf));
  CUDA_RUNTIME(cudaFree(expBuf));

  return result;
}

/* difficult to compare expected results here because we have to create a
 * different communicator too
 */
BM::Result BM::Method_neighbor_alltoallv::operator()(const SquareMat &mat,
                                                     const int nIters) {
  BM::Result result{};
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (mat.size() != size) {
    LOG_FATAL("size mismatch");
    exit(1);
  }

  // if the matrix is empty, nothing to measure
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

#if TEMPI_OUTPUT_LEVEL >= 4 && 1
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
    MPI_Barrier(MPI_COMM_WORLD);
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
#if TEMPI_OUTPUT_LEVEL >= 4 && 1
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < size; ++i) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (i == rank) {
        std::string s, t;
        for (auto &e : sources) {
          s += std::to_string(e) + " ";
        }
        for (auto &e : destinations) {
          t += std::to_string(e) + " ";
        }
        LOG_SPEW("sources=" << s << " destinations=" << t);
      }
      std::cerr << std::flush;
      MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    {
      auto start = Clock::now();
      MPI_Dist_graph_create_adjacent(
          MPI_COMM_WORLD, sources.size(), sources.data(), sourceweights.data(),
          destinations.size(), destinations.data(), destweights.data(),
          MPI_INFO_NULL, 1 /*reorder*/, &graph);
      auto stop = Clock::now();
      result.setup = stop - start;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // after reorder, my rank has changed, so determine the indegree / outdegree
  // again
  MPI_Comm_rank(graph, &rank);
  MPI_Comm_size(graph, &size);

  sources.clear();
  destinations.clear();
  sourceweights.clear();
  destweights.clear();
  for (size_t j = 0; j < size; ++j) {
    if (mat[rank][j] != 0) {
      destinations.resize(destinations.size() + 1);
    }
    if (mat[j][rank] != 0) {
      sources.resize(sources.size() + 1);
    }
  }
  sourceweights.resize(sources.size());
  destweights.resize(destinations.size());

  // get my neighbors (not guaranteed to be the same order as create)
  MPI_Dist_graph_neighbors(graph, sources.size(), sources.data(),
                           sourceweights.data(), destinations.size(),
                           destinations.data(), destweights.data());
#if TEMPI_OUTPUT_LEVEL >= 4 && 1
  {
    std::string s, t;
    for (int i = 0; i < sources.size(); ++i) {
      s += std::to_string(sources[i]) + " ";
      t += std::to_string(sourceweights[i]) + " ";
    }
    for (int r = 0; r < size; ++r) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (r == rank)
        LOG_SPEW("rank " << rank << ": sources=" << s
                         << " sourceweights=" << t);
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
#endif

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
      MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  // create device allocations
  char *sendbuf{}, *recvbuf{};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc(&sendbuf, sendBufSize));
  CUDA_RUNTIME(cudaMalloc(&recvbuf, recvBufSize));

  // create Alltoallv arguments
  std::vector<int> sendcounts, sdispls, recvcounts, rdispls;
  for (size_t i = 0; i < destinations.size(); ++i) {
    const int dest = destinations[i];
    sendcounts.push_back(mat[rank][dest]);
  }
  sdispls.push_back(0);
  for (size_t i = 1; i < destinations.size(); ++i) {
    sdispls.push_back(sdispls[i - 1] + sendcounts[i - 1]);
  }
  for (size_t i = 0; i < sources.size(); ++i) {
    const int src = sources[i];
    recvcounts.push_back(mat[src][rank]);
  }
  rdispls.push_back(0);
  for (size_t i = 1; i < sources.size(); ++i) {
    rdispls.push_back(rdispls[i - 1] + recvcounts[i - 1]);
  }

#if 1
  {
    std::string s, t;
    for (int i = 0; i < recvcounts.size(); ++i) {
      s += std::to_string(recvcounts[i]) + " ";
      t += std::to_string(rdispls[i]) + " ";
    }
    for (int r = 0; r < size; ++r) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (r == rank)
        LOG_SPEW("rank " << rank << ": recvcounts=" << s
                         << " recvdispls=" << t);
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
#endif

  // benchmark loop
  Statistics stats;
  for (int i = 0; i < nIters; ++i) {
    nvtxRangePush("alltoallv");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = Clock::now();

    // it's possible that we will not send or recv data,
    // but the MPI impl may not want a nullptr.
    MPI_Neighbor_alltoallv(
        sendbuf, sendcounts.data(), sdispls.data(), MPI_BYTE,
        recvbuf ? recvbuf : (void *)(0xDEADBEEF),
        recvcounts.data() ? recvcounts.data() : (int *)0xDEADBEEF,
        rdispls.data() ? rdispls.data() : (int *)0xDEADBEEF, MPI_BYTE, graph);
    auto stop = Clock::now();
    nvtxRangePop();
    Duration dur = stop - start;
    double tmp = dur.count();

    MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    stats.insert(tmp);
  }

  result.iters = stats;

  CUDA_RUNTIME(cudaFree(sendbuf));
  CUDA_RUNTIME(cudaFree(recvbuf));

  return result;
}
