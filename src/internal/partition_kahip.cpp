#include "kaHIP_interface.h"

#include "partition.hpp"

#include <cassert>
#include <vector>

partition::Result partition::partition_kahip(const int nParts,
                                             const std::vector<int> &rowPtrs,
                                             const std::vector<int> &colInd,
                                             const std::vector<int> &colVal) {

  partition::Result result;
  result.part = std::vector<int>(nParts, -1);

  int n = rowPtrs.size() - 1;
  int *vwgt = nullptr; // unweighted vertices

  // kaffpa won't modify these
  int *xadj = const_cast<int *>(rowPtrs.data());
  int *adjcwgt = const_cast<int *>(colVal.data());
  int *adjncy = const_cast<int *>(colInd.data());
  int nparts = nParts;
  double imbalance = 3;
  bool suppress_output = false;
  int seed = 0;
  int mode = STRONG;
  int edgecut;
  int *part = result.part.data();

  assert(n >= 0);
  // https://raw.githubusercontent.com/KaHIP/KaHIP/master/manual/kahip.pdf
  kaffpa(&n, vwgt, xadj, adjcwgt, adjncy, &nparts, &imbalance, suppress_output,
         seed, mode, &edgecut, part);

  result.objective = edgecut;
  return result;
}