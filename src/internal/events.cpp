//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "events.hpp"

#include "cuda_runtime.hpp"
#include "logging.hpp"

#include <nvToolsExt.h>

#include <functional>
#include <memory>
#include <unordered_set>

class EventPool {
  std::vector<cudaEvent_t> available;
  std::unordered_set<cudaEvent_t> used;
  unsigned int flags_;

public:
  EventPool(unsigned int flags) : flags_(flags) {}
  ~EventPool() { clear(); }
  void clear() {
    for (cudaEvent_t event : available) {
      LOG_SPEW("destroy event " << uintptr_t(event));
      cudaError_t err = cudaEventDestroy(event);
      if (err == cudaErrorCudartUnloading) {
        LOG_WARN("cleanup after driver shutdown");
      } else {
        CUDA_RUNTIME(err);
      }
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

  void ensure_available(unsigned n) {
    while (available.size() < n) {
      LOG_SPEW("create new event");
      cudaEvent_t event;
      CUDA_RUNTIME(cudaEventCreateWithFlags(&event, flags_));
      available.push_back(event);
    }
  }

  cudaEvent_t request() {
    ensure_available(1);
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

EventPool pool(cudaEventDisableTiming | cudaEventBlockingSync);

namespace events {

void init() {
  nvtxRangePush("event::init()");
  pool.ensure_available(5);
  nvtxRangePop();
}
void finalize() {
  nvtxRangePush("event::finalize()");
  pool.clear();
  nvtxRangePop();
}

cudaEvent_t request() { return pool.request(); }
void release(cudaEvent_t event) { pool.release(event); }

} // namespace events