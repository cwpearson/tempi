//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "requests.hpp"

#include "logging.hpp"

#include <condition_variable>
#include <mutex>
#include <set>

// requests not yet given to the library
std::set<MPI_Request *> blocked;
std::mutex m;
std::condition_variable cv;

// future calls to MPI_Wait will block until release_request is called
void block_request(MPI_Request *request) {
  std::lock_guard<std::mutex> lg(m);
  blocked.insert(request);
}
void release_request(MPI_Request *request) {
  {
    std::lock_guard<std::mutex> lg(m);
    blocked.erase(request);
  }
  cv.notify_one();
}

// wait for a request to be unblocked
void wait(MPI_Request *request) {
  if (0 == blocked.count(request)) {
    return;
  }
  LOG_SPEW("request " << uintptr_t(request) << " is blocked...");
  std::unique_lock<std::mutex> lk(m);
  cv.wait(lk, [&] { return 0 == blocked.count(request); });
  lk.unlock();
  LOG_SPEW("request " << uintptr_t(request) << " was released");
}