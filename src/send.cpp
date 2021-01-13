//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

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
