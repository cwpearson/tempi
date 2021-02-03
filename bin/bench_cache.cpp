#include "logging.hpp"

#include <mpi.h>

#include <cmath>
#include <map>
#include <unordered_map>

// TODO: directly use SystemPerformance::SendNonContigNd::Args / Method
/* key and value taken from SenderND
 */
// argument pack to sender
struct Key {
  bool colocated;
  int64_t bytes;
  struct Hasher { // unordered_map key
    size_t operator()(const Key &a) const noexcept {
      return std::hash<bool>()(a.colocated) ^ std::hash<int64_t>()(a.bytes);
    }
  };
  bool operator==(const Key &rhs) const { // unordered_map key
    return colocated == rhs.colocated && bytes == rhs.bytes;
  }
  bool operator<(const Key &rhs) const noexcept { // map key
    if (colocated < rhs.colocated) {
      return true;
    } else {
      return bytes < rhs.bytes;
    }
  }
};
enum class Value { DEVICE, ONESHOT };

int main(void) {

  int nfinds = 10000;

  std::cout << "#keys,std::unordered_map (ns),std::map (ns)\n";

  size_t hits = 0;
  for (int sz = 1; sz < 512 * 1024 * 1024; sz = std::ceil(sz * 1.3)) {

    std::unordered_map<Key, Value, Key::Hasher> cache1;
    std::map<Key, Value> cache2;
    cache1.reserve(sz);

    // create assortment of independent keys. this is is the slow part
    for (int i = 0; i < sz; ++i) {
      Key key{.colocated = bool(i % 2), .bytes = i / 2};
      cache1[key] = Value();
      cache2[key] = Value();
    }
    if (cache1.size() != sz) {
      LOG_FATAL("bad key generation");
    }

    std::cout << sz;

    {
      double start = MPI_Wtime();
      for (int i = 0; i < nfinds; ++i) {
        Key key{.colocated = i % 2, .bytes = i / 2};
        auto it = cache1.find(key);
        if (it != cache1.end()) {
          ++hits;
        }
      }
      double stop = MPI_Wtime();
      std::cout << "," << (stop - start) * 1e9 / double(nfinds) << std::flush;
    }

    {
      double start = MPI_Wtime();
      for (int i = 0; i < nfinds; ++i) {
        Key key{.colocated = bool(i % 2), .bytes = i / 2};
        auto it = cache2.find(key);
        if (it != cache2.end()) {
          ++hits;
        }
      }
      double stop = MPI_Wtime();
      std::cout << "," << (stop - start) / double(nfinds) << std::flush;
    }
    std::cout << "\n";
  }
  return hits;
}