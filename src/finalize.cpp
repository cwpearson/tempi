#include "logging.hpp"

#include "streams.hpp"

#include <mpi.h>

#include <dlfcn.h>

extern "C" int MPI_Finalize() {
  LOG_DEBUG("MPI_Finalize");
  typedef int (*Func_MPI_Finalize)();
  static Func_MPI_Finalize fn = nullptr;

  if (!fn) {
    fn = reinterpret_cast<Func_MPI_Finalize>(dlsym(RTLD_NEXT, "MPI_Finalize"));
  }

  streams_finalize();

  return fn();
}