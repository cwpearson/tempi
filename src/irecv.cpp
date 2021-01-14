//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "irecv.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"

extern "C" int MPI_Irecv(PARAMS_MPI_Irecv) {
  if (environment::noTempi) {
    return libmpi.MPI_Irecv(ARGS_MPI_Irecv);
  }
  return irecv::impl(ARGS_MPI_Irecv);
}
