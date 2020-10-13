#include "streams.hpp"
#include "logging.hpp"

#include <mpi.h>

#include <dlfcn.h>
#include <nvToolsExt.h>



#define PARAMS int *argc, char ***argv
#define ARGS argc, argv

extern "C" int MPI_Init(PARAMS) {
  typedef int (*Func_MPI_Init)(PARAMS);
  static Func_MPI_Init fn = nullptr;
  if (!fn) {
    nvtxRangePush("dlsym(MPI_Init)");
    fn = reinterpret_cast<Func_MPI_Init>(dlsym(RTLD_NEXT, "MPI_Init"));
    nvtxRangePop();
  }
  int err = fn(ARGS);
  // can use logging now that MPI_Init has been called
  LOG_DEBUG("finished library MPI_Init");

  streams_init();

  return err;
}
