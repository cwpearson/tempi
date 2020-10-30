#include "allocators.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "streams.hpp"
#include "worker.hpp"

#include <mpi.h>

#include <dlfcn.h>

#define ARGS // empty args

extern "C" int MPI_Finalize() {
  LOG_DEBUG("enter MPI_Finalize");
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
    LOG_DEBUG("Host Allocator Requests:        " << stats.numRequests);
    LOG_DEBUG("Host Allocator Releases:        " << stats.numReleases);
    LOG_DEBUG("Host Allocator Max Usage (MiB): " << stats.maxUsage / 1024 /
                                                        1024);
    LOG_DEBUG("Host Allocator allocs:          " << stats.numAllocs);
  }
  {
    auto stats = deviceAllocator.stats();
    LOG_DEBUG("Device Allocator Requests:        " << stats.numRequests);
    LOG_DEBUG("Device Allocator Releases:        " << stats.numReleases);
    LOG_DEBUG("Device Allocator Max Usage (MiB): " << stats.maxUsage / 1024 /
                                                          1024);
    LOG_DEBUG("Device Allocator allocs:          " << stats.numAllocs);
  }
#endif

  worker_finalize();
  streams_finalize();
  allocators::finalize();

  LOG_DEBUG("library MPI_Finalize");
  int err = fn();
  return err;
}