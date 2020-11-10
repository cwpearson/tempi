#pragma once

#include <vector>

namespace partition {

struct Result {
  std::vector<int> part;
  double objective;

  size_t num_parts() const noexcept;
};

std::vector<int> random(const int numRanks, const int numNodes);

#ifdef TEMPI_ENABLE_KAHIP
/* use KaHIP to partition.
   the parameters are a CSR matrix.
   edges and back-edges should be represented.
*/
Result partition_kahip(const int nParts, const std::vector<int> &rowPtrs,
                       const std::vector<int> &colInd,
                       const std::vector<int> &colVal);

/* use KaHIP process_mapping to partition.
   the parameters are a CSR matrix.
   edges and back-edges should be represented.
*/
Result kahip_process_mapping(const int nParts,
                                       const std::vector<int> &rowPtrs,
                                       const std::vector<int> &colInd,
                                       const std::vector<int> &colVal);
#endif

#ifdef TEMPI_ENABLE_METIS
/* use METIS to partition.
   the parameters are a CSR matrix.
   edges and back-edges should be represented and be identical
*/
Result partition_metis(const int nParts, const std::vector<int> &rowPtrs,
                       const std::vector<int> &colInd,
                       const std::vector<int> &colVal);
#endif

/* true if all partitions are equal size
 */
bool is_balanced(const Result &result);

} // namespace partition