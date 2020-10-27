#include "env.hpp"
#include "logging.hpp"

#include "symbols.hpp"

#include <mpi.h>

#include <nvToolsExt.h>

extern "C" int MPI_Comm_free(PARAMS_MPI_Comm_free) {
  int err = libmpi.MPI_Comm_free(ARGS_MPI_Comm_free);
  if (environment::noTempi) {
    return err;
  }
  topology::uncache(comm);
  return err;
}
