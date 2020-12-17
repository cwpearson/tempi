#include "send.hpp"
#include "env.hpp"
#include "symbols.hpp"

#include <mpi.h>

extern "C" int MPI_Send(PARAMS_MPI_Send) {
  if (environment::noTempi) {
    return libmpi.MPI_Send(ARGS_MPI_Send);
  }
  return send::impl(ARGS_MPI_Send);
}
