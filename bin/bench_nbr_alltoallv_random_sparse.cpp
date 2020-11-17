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
  int rank, size, numNodes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  numNodes = topology::num_nodes(MPI_COMM_WORLD);

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
      double start = MPI_Wtime();
      MPI_Dist_graph_create_adjacent(
          MPI_COMM_WORLD, sources.size(), sources.data(), sourceweights.data(),
          destinations.size(), destinations.data(), destweights.data(),
          MPI_INFO_NULL, 1 /*reorder*/, &graph);
      result.setup = MPI_Wtime() - start;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  fill_comm_stats(graph, result, mat);

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

#if TEMPI_OUTPUT_LEVEL >= 4 && 1
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
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePush("alltoallv");
    double start = MPI_Wtime();

    // it's possible that we will not send or recv data,
    // but the MPI impl may not want a nullptr.
    MPI_Neighbor_alltoallv(
        sendbuf, sendcounts.data(), sdispls.data(), MPI_BYTE,
        recvbuf ? recvbuf : (void *)(0xDEADBEEF),
        recvcounts.data() ? recvcounts.data() : (int *)0xDEADBEEF,
        rdispls.data() ? rdispls.data() : (int *)0xDEADBEEF, MPI_BYTE, graph);
    double tmp = MPI_Wtime() - start;
    nvtxRangePop();

    MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    stats.insert(tmp);
  }

  result.iters = stats;

  CUDA_RUNTIME(cudaFree(sendbuf));
  CUDA_RUNTIME(cudaFree(recvbuf));

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
