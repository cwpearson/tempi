//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"

#include <mpi.h>

extern "C" int MPI_Comm_free(PARAMS_MPI_Comm_free) {
  int err = libmpi.MPI_Comm_free(ARGS_MPI_Comm_free);
  if (environment::noTempi) {
    return err;
  }
  topology::uncache(comm);
  return err;
}
