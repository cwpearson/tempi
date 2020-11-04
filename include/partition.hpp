#pragma once

#include <vector>

namespace partition {

struct Result {
  std::vector<int> part;
  double objective;
};

std::vector<int> random(const int numRanks, const int numNodes);

/* use KaHIP to partition.
   the parameters are a CSR matrix.
   edges and back-edges should be represented.
*/
Result partition_kahip(const int nParts, const std::vector<int> &rowPtrs,
                       const std::vector<int> &colInd,
                       const std::vector<int> &colVal);

/* use METIS to partition.
   the parameters are a CSR matrix.
   edges and back-edges should be represented and be identical
*/
Result partition_metis(const int nParts, const std::vector<int> &rowPtrs,
                       const std::vector<int> &colInd,
                       const std::vector<int> &colVal);

/* true if all partitions are equal size
 */
bool is_balanced(const Result &result);

} // namespace partition