#include "env.hpp"
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
  TEMPI_DISABLE_GUARD;
  LOG_DEBUG("MPI_Finalize");

  worker_finalize();
  streams_finalize();

  return fn();
}