#include "send.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"


#include <cuda_runtime.h>
#include <mpi.h>

#include <dlfcn.h>

#include <vector>

extern "C" int MPI_Send(PARAMS_MPI_Send) {
  if (environment::noTempi) {
    return libmpi.MPI_Send(ARGS_MPI_Send);
  }
  return send::impl(ARGS_MPI_Send);
}
