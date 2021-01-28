#include "logging.hpp"

#include <map>
#include <mpi.h>
#include <unordered_map>

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

// which sender to use
enum class Value { DEVICE, ONESHOT };

int main(void) {

  int nfinds = 10000000;

  size_t hits = 0;
  for (int sz = 1; sz < 512 * 1024 * 1024; sz *= 2) {

    // std::unordered_map<Key, Value, Key::Hasher> cache;
    std::map<Key, Value> cache;
    // create assortment of independent keys
    for (int i = 0; i < sz; ++i) {
      Key key{.colocated = i % 2, .bytes = i / 2};
      cache[key] = Value();
    }
    if (cache.size() != sz) {
      LOG_FATAL("bad keys generation");
    }

    double start = MPI_Wtime();

    for (int i = 0; i < nfinds; ++i) {

      Key key{.colocated = i % 2, .bytes = i / 2};
      auto it = cache.find(key);
      if (it != cache.end()) {
        ++hits;
      }
    }

    double stop = MPI_Wtime();

    // std::cerr << sz << "," << (stop - start) / double(nfinds) << ","
    //           << cache.bucket_count() << "," << cache.load_factor() << "\n";

    std::cerr << sz << "," << (stop - start) / double(nfinds) << "\n";
  }
  return hits;
}