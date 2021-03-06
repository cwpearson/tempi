//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "async_operation.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"

extern "C" int MPI_Wait(PARAMS_MPI_Wait) {
  if (environment::noTempi) {
    return libmpi.MPI_Wait(ARGS_MPI_Wait);
  }
  return async::wait(ARGS_MPI_Wait);
}
