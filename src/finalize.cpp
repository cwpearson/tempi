//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "allocators.hpp"
#include "async_operation.hpp"
#include "env.hpp"
#include "events.hpp"
#include "logging.hpp"
#include "streams.hpp"
#include "worker.hpp"

#include <mpi.h>

#include <dlfcn.h>

#define ARGS // empty args

extern "C" int MPI_Finalize() {
  typedef int (*Func_MPI_Finalize)();
  static Func_MPI_Finalize fn = nullptr;
  if (!fn) {
    fn = reinterpret_cast<Func_MPI_Finalize>(dlsym(RTLD_NEXT, "MPI_Finalize"));
  }
  // TODO: in tests, it's possible that TEMPI was enabled during MPI_Init and
  // disabled now
  TEMPI_DISABLE_GUARD;

#if TEMPI_OUTPUT_LEVEL >= 4
  {
    auto stats = hostAllocator.stats();
    LOG_SPEW("Host Allocator Requests:        " << stats.numRequests);
    LOG_SPEW("Host Allocator Releases:        " << stats.numReleases);
    LOG_SPEW("Host Allocator Max Usage (MiB): " << stats.maxUsage / 1024 /
                                                       1024);
    LOG_SPEW("Host Allocator allocs:          " << stats.numAllocs);
  }
  {
    auto stats = deviceAllocator.stats();
    LOG_SPEW("Device Allocator Requests:        " << stats.numRequests);
    LOG_SPEW("Device Allocator Releases:        " << stats.numReleases);
    LOG_SPEW("Device Allocator Max Usage (MiB): " << stats.maxUsage / 1024 /
                                                         1024);
    LOG_SPEW("Device Allocator allocs:          " << stats.numAllocs);
  }
#endif

  async::finalize();
  events::finalize();
  worker_finalize();
  streams_finalize();
  allocators::finalize();

  LOG_SPEW("library MPI_Finalize");
  int err = fn();
  return err;
}
