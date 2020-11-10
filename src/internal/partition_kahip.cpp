#include "partition.hpp"

#include "logging.hpp"

#include "kaHIP_interface.h"

#include <cassert>
#include <vector>
#include <limits>

partition::Result partition::partition_kahip(const int nParts,
                                             const std::vector<int> &rowPtrs,
                                             const std::vector<int> &colInd,
                                             const std::vector<int> &colVal) {

  partition::Result result;
  result.part = std::vector<int>(rowPtrs.size()-1, -1);

  /* normalize weights before partitioning
 * KaHIP seems to be much slower for matrices that are identical but with larger values
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
  int nparts = nParts;
  double imbalance = 0;
  bool suppress_output = false;
  bool perfectly_balance = true;
  int mode = FAST;
  int edgecut;
  int *part = result.part.data();

  assert(n >= 0);

  // try some variable initial conditions
  int bestCut = std::numeric_limits<int>::max();
  std::vector<int> best;
  for (int seed = 0; seed < 20; ++seed) {
    
    // https://raw.githubusercontent.com/KaHIP/KaHIP/master/manual/kahip.pdf
    kaffpa(&n, vwgt, xadj, adjcwgt, adjncy, &nparts, &imbalance, suppress_output,
         perfectly_balance, seed, mode, &edgecut, part);
    LOG_SPEW("seed= " << seed << " edgecut=" << edgecut);
    if (edgecut < bestCut) {
      best = result.part;
      bestCut = edgecut;
    }
    if (bestCut == 0) {
      break;
    }
  }


  result.objective = bestCut;
  LOG_DEBUG("KaHIP partition objective=" << result.objective);

  return result;
}
