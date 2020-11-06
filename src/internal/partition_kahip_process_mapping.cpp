#include "partition.hpp"

#include "logging.hpp"

#include "kaHIP_interface.h"

#include <cassert>
#include <limits>
#include <vector>

partition::Result partition::kahip_process_mapping(
    const int nParts, const std::vector<int> &rowPtrs,
    const std::vector<int> &colInd, const std::vector<int> &colVal) {

  partition::Result result;
  result.part = std::vector<int>(rowPtrs.size() - 1, -1);

  /* normalize weights before partitioning
   * KaHIP seems to be much slower for matrices that are identical but with
   * larger values
   */
  std::vector<int> weight;
  int minNonZero = 0;
  for (int e : colVal) {
    if (e != 0) {
      if (0 == minNonZero) {
        minNonZero = e;
      }
      if (e < minNonZero) {
        minNonZero = e;
      }
    }
  }
  if (0 != minNonZero) {
    LOG_SPEW("normalize weights by " << minNonZero);
    for (int e : colVal) {
      weight.push_back(e / minNonZero);
    }
  } else {
    weight = colVal;
  }

  int n = rowPtrs.size() - 1;
  int *vwgt = nullptr; // unweighted vertices

  // kaffpa won't modify these
  int *xadj = const_cast<int *>(rowPtrs.data());
  int *adjcwgt = const_cast<int *>(weight.data());
  int *adjncy = const_cast<int *>(colInd.data());

  int mode_partitioning = FAST;
  int mode_mapping = MAPMODE_MULTISECTION;
  int edgecut;
  int qap;
  int *part = result.part.data();

  // 6 GPUs per node, nParts nodes
  int nGpus = (rowPtrs.size() - 1) / nParts;
  std::cerr << "nParts=" << nParts << " nGpus=" << nGpus << "\n";

  std::vector<int> hierarchy_parameter(nGpus, nParts);
  std::vector<int> distance_parameter(1, 5);
  int hierarchy_depth = distance_parameter.size();

  double imbalance = 1;
  bool suppress_output = false;

  assert(n >= 0);

  // try some variable initial conditions
  int bestQap = std::numeric_limits<int>::max();
  std::vector<int> best;
  for (int seed = 0; seed < 20; ++seed) {

    /*

    int* n, int* vwgt, int* xadj,
                       int* adjcwgt, int* adjncy,
                       int* hierarchy_parameter,  int* distance_parameter, int
    hierarchy_depth, int mode_partitioning, int mode_mapping, double* imbalance,
                       bool suppress_output, int seed,
                       int* edgecut, int* qap, int* part
    */

    // https://raw.githubusercontent.com/KaHIP/KaHIP/master/manual/kahip.pdf
    process_mapping(&n, vwgt, xadj, adjcwgt, adjncy, hierarchy_parameter.data(),
                    distance_parameter.data(), hierarchy_depth,
                    mode_partitioning, mode_mapping, &imbalance,
                    suppress_output, seed, &edgecut, &qap, part);
    LOG_SPEW("seed= " << seed << " qap=" << qap);
    if (qap < bestQap) {
      best = result.part;
      bestQap = qap;
    }
    if (bestQap == 0) {
      break;
    }
  }

  result.objective = bestQap;
  LOG_DEBUG("KaHIP process_mapping qap=" << result.objective);

  return result;
}
