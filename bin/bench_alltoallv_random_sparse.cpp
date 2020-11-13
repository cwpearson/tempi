/*! \file
    measure various MPI methods for achiving the same communication pattern
*/

#include "../include/allocators.hpp"
#include "../include/cuda_runtime.hpp"
#include "../include/env.hpp"
#include "../include/logging.hpp"
#include "../include/topology.hpp"

#include "benchmark.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <sstream>

struct Result {
  // elapsed time for setup, teardown
  double setup;
  double teardown;
  Statistics iters;

  int64_t maxPairwise; // max bytes sent between any pair of nodes
  int64_t maxOnNode;   // max bytes sent within a node by any node
  int64_t maxOffNode;  // max bytes sent offnode by any node

  int64_t totalOnNode;  // total bytes staying on node
  int64_t totalOffNode; // total bytes leaving nodes
};

SquareMat get_node_mat(MPI_Comm comm, const SquareMat &mat) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  SquareMat ret(topology::num_nodes(comm), 0);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      int64_t v = mat[i][j];
      ret[topology::node_of_rank(comm, i)][topology::node_of_rank(comm, j)] +=
          v;
    }
  }

  return ret;
}

void fill_comm_stats(MPI_Comm comm, Result &result, const SquareMat &mat) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // max pairwise
  result.maxPairwise = 0;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      int64_t v = mat[i][j];
      if (v > result.maxPairwise) {
        result.maxPairwise = v;
      }
    }
  }

  SquareMat nodeMat = get_node_mat(comm, mat);

  if (0 == rank) {
    LOG_DEBUG("\nrank matrix\n" << mat.str()<<"\n");
    LOG_DEBUG("\nnode matrix\n" << nodeMat.str()<<"\n");
  }

  result.maxOnNode = 0;
  result.totalOnNode = 0;
  result.maxOffNode = 0;
  result.totalOffNode = 0;
  for (int i = 0; i < nodeMat.size(); ++i) {
    int64_t offNode = 0; // num bytes sent offnode
    for (int j = 0; j < nodeMat.size(); ++j) {
      int64_t v = nodeMat[i][j];
      if (i == j) {
        result.totalOnNode += v;
        result.maxOnNode = std::max(result.maxOnNode, v);
      } else {
        offNode += v;
      }
    }
    result.totalOffNode += offNode;
    result.maxOffNode = std::max(result.maxOffNode, offNode);
  }
}

Result bench(const SquareMat &mat, const int nIters) {

  Result result{};
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  fill_comm_stats(MPI_COMM_WORLD, result, mat);

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
    double start = MPI_Wtime();
    MPI_Alltoallv(srcBuf, sendcounts.data(), sdispls.data(), MPI_BYTE, dstBuf,
                  recvcounts.data(), rdispls.data(), MPI_BYTE, MPI_COMM_WORLD);
    double tmp = MPI_Wtime() - start;
    nvtxRangePop();

    MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    stats.insert(tmp);
  }
  result.iters = stats;

  CUDA_RUNTIME(cudaFree(srcBuf));
  CUDA_RUNTIME(cudaFree(dstBuf));

  return result;
}

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int size, rank, numNodes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  numNodes = topology::num_nodes(MPI_COMM_WORLD);

  if (0 == rank) {
    char version[MPI_MAX_LIBRARY_VERSION_STRING] = {};
    int len;
    MPI_Get_library_version(version, &len);
    std::cout << version << std::endl;
  }

  int nIters = 30;
  std::vector<int64_t> scales{1,         10,         100,        1 * 1000,
                              10 * 1000, 100 * 1000, 1000 * 1000};
  std::set<float> densities{1.0, 0.5, 0.1, 0.05};

  // add some more densities to target particular nnz per row
  for (int targetNnzPerRow : {1, 2, 4, 8, 16}) {
    float density = double(targetNnzPerRow) / size;
    if (density <= 1) {
      densities.insert(density);
    }
  }

  if (0 == rank) {
    std::cout << "nodes,rankspernode,scale,density,max pairwise (B), maxOnNode "
                 "(B), maxOffNode (B), totalOnNode (B), totalOffNode(B),setup "
                 "(s),alltoallv (s),teardown (s)\n";
  }

  for (int64_t scale : scales) {
    for (float density : densities) {

      int rowNnz = density * size + 0.5;
      SquareMat mat = SquareMat::make_random_sparse(size, rowNnz, 1, 10, scale);
      Result result = bench(mat, nIters);

      if (0 == rank) {
        std::cout << numNodes << "," << size / numNodes;
        std::cout << "," << scale << "," << density;
        std::cout << "," << result.maxPairwise;
        std::cout << "," << result.maxOnNode;
        std::cout << "," << result.maxOffNode;
        std::cout << "," << result.totalOnNode;
        std::cout << "," << result.totalOffNode;
        std::cout << "," << result.setup;
        std::cout << "," << result.iters.min();
        std::cout << "," << result.teardown;

        std::cout << "\n" << std::flush;
      }
    }
  }

  MPI_Finalize();
  return 0;
}
