#include "partition.hpp"

#include <algorithm>
#include <map>
#include <random>

// give all ranks the same seed
std::default_random_engine generator(0);

namespace partition {

/* return a node assignment 0..<numNodes for each rank 0..<numRanks
 */
std::vector<int> random(const int numRanks, const int numNodes) {
  std::vector<int> p(numRanks);
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] = i * numNodes / numRanks;
  }
  std::shuffle(p.begin(), p.end(), generator);
  return p;
}

/* true if each partition is the same size
 */
bool is_balanced(const Result &r) {
  std::map<int, int> sizes;
  for (int e : r.part) {
    sizes[e] += 1;
  }
  for (auto &kv : sizes) {
    if (sizes.begin()->second != kv.second) {
      return false;
    }
  }
  return true;
}

} // namespace partition