//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"
#include "topology.hpp"

#include <mpi.h>

extern "C" int MPI_Comm_rank(PARAMS_MPI_Comm_rank) {
  if (environment::noTempi) {
    return libmpi.MPI_Comm_rank(ARGS_MPI_Comm_rank);
  }

  int librank;
  int err = libmpi.MPI_Comm_rank(comm, &librank);

  // if there is placement, translate the library rank into the application's
  // rank
  *rank = topology::application_rank(comm, librank);
  return err;
}
