#include "env.hpp"
#include "logging.hpp"
#include "streams.hpp"

#include <mpi.h>

#include <dlfcn.h>
#include <nvToolsExt.h>

#define PARAMS int *argc, char ***argv
#define ARGS argc, argv

extern "C" int MPI_Init(PARAMS) {

  // before anything else, read env vars to control tempi
  read_environment();

  typedef int (*Func_MPI_Init)(PARAMS);
  static Func_MPI_Init fn = nullptr;
  if (!fn) {
    fn = reinterpret_cast<Func_MPI_Init>(dlsym(RTLD_NEXT, "MPI_Init"));
  }
  TEMPI_DISABLE_GUARD;
  int err = fn(ARGS);
  LOG_DEBUG("finished MPI_Init");
  // can use logging now that MPI_Init has been called

  streams_init();

  return err;
}
