#include "partition.hpp"

#include <algorithm>
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

}