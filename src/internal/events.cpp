//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "events.hpp"

#include "cuda_runtime.hpp"
#include "logging.hpp"

#include <functional>
#include <memory>
#include <unordered_set>

class EventPool {
  std::vector<cudaEvent_t> available;
  std::unordered_set<cudaEvent_t> used;
  unsigned int flags_;

public:
  EventPool(unsigned int flags) : flags_(flags) {

    while (available.size() < 5) { // 5 events initially
      cudaEvent_t event;
      CUDA_RUNTIME(cudaEventCreateWithFlags(&event, flags_));
      available.push_back(event);
    }
  }
  ~EventPool() { clear(); }
  void clear() {
    for (cudaEvent_t event : available) {
      LOG_SPEW("destroy event " << uintptr_t(event));
      CUDA_RUNTIME(cudaEventDestroy(event));
    }
    available.clear();
    if (!used.empty()) {
      LOG_ERROR(used.size() << " events were never released.");
      for (cudaEvent_t event : used) {
        LOG_SPEW("destroy event " << uintptr_t(event));
        CUDA_RUNTIME(cudaEventDestroy(event));
      }
    }
    used.clear();
  }

  cudaEvent_t request() {
    if (available.empty()) {
      cudaEvent_t event;
      CUDA_RUNTIME(cudaEventCreateWithFlags(&event, flags_));
      available.push_back(event);
    }
    cudaEvent_t event = available.back();
    available.pop_back();
    used.insert(event);
    return event;
  }
  void release(cudaEvent_t event) {
    used.erase(event);
    available.push_back(event);
  }
};

EventPool pool(cudaEventDisableTiming | cudaEventBlockingSync |
               cudaEventDisableTiming);

namespace events {

void init() {}
void finalize() { pool.clear(); }

cudaEvent_t request() { return pool.request(); }
void release(cudaEvent_t event) { pool.release(event); }

} // namespace events