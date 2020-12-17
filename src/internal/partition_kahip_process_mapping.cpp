#include "partition.hpp"

#include "logging.hpp"

#include "kaHIP_interface.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

partition::Result partition::kahip_process_mapping(
    const int nParts, const std::vector<int> &rowPtrs,
    const std::vector<int> &colInd, const std::vector<int> &colVal) {

  partition::Result result;
  result.part = std::vector<int>(rowPtrs.size() - 1, -1);

  /* process_mapping only works for unweighted graphs.
     To convert the graph to unweighted, delete any edge less than 1/10th the
     weight of the max
  */

  std::vector<int> unRowPtr = rowPtrs;
  std::vector<int> unColInd = colInd;
  std::vector<int> unColVal = colVal;

  // zero all weights < 1/10 of max
  int maxWeight = 0;
  for (int e : colVal) {
    maxWeight = std::max(e, maxWeight);
  }
  for (int &e : unColVal) {
    if (e < maxWeight / 10) {
      e = 0;
    }
  }

  // remove all zero values from CSR
  bool changed = true;
  while (changed) {
    changed = false;

    auto it = std::find(unColVal.begin(), unColVal.end(), 0);
    if (unColVal.end() != it) {
      changed = true;
      int off = it - unColVal.begin();
      unColVal.erase(it);
      unColInd.erase(unColInd.begin() + off);

      // reduce all rowPtr values pointing after the removed value
      auto lb = std::lower_bound(unRowPtr.begin(), unRowPtr.end(), off);
      if (lb != unRowPtr.end()) {
        assert(*lb <= off);
        ++lb;
        assert(*lb > off);
        for (; lb != unRowPtr.end(); ++lb) {
          --(*lb);
        }
      }
    }
  }

  std::cerr << "rowPtr: ";
  for (int e : unRowPtr) {
    std::cerr << e << " ";
  }
  std::cerr << "\n";

  std::cerr << "colInd: ";
  for (int e : unColInd) {
    std::cerr << e << " ";
  }
  std::cerr << "\n";

  int n = rowPtrs.size() - 1;
  int *vwgt = nullptr; // unweighted vertices

  // process_mapping won't modify these
  int *xadj = const_cast<int *>(unRowPtr.data());
  int *adjcwgt = nullptr; // unweighted
  int *adjncy = const_cast<int *>(unColInd.data());

  int mode_partitioning = FAST;
  int mode_mapping = MAPMODE_MULTISECTION;
  int edgecut;
  int qap;
  int *part = result.part.data();

  int nRanks = (unRowPtr.size() - 1) / nParts;
  LOG_SPEW("nParts=" << nParts << " nRanks=" << nRanks);

  std::vector<int> hierarchy_parameter{nRanks, nParts};
  std::vector<int> distance_parameter{1, 5};
  int hierarchy_depth = distance_parameter.size();

  double imbalance = 0.00;
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
    process_mapping_enforcebalance(&n, vwgt, xadj, adjcwgt, adjncy, hierarchy_parameter.data(),
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
