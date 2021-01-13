//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "env.hpp"
#include "logging.hpp"
#include "packer.hpp"
#include "symbols.hpp"
#include "types.hpp"

#include <mpi.h>

extern "C" int MPI_Type_free(PARAMS_MPI_Type_free) {
  if (environment::noTempi) {
    return libmpi.MPI_Type_free(ARGS_MPI_Type_free);
  } else if (environment::noTypeCommit) {
    return libmpi.MPI_Type_free(ARGS_MPI_Type_free);
  }

  // this call can modify the pointer, so we need to release first
  release(*datatype);
  int result = libmpi.MPI_Type_free(ARGS_MPI_Type_free);

  return result;
}