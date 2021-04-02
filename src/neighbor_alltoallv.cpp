//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"

#include <string>

extern "C" int MPI_Neighbor_alltoallv(PARAMS_MPI_Neighbor_alltoallv) {
  if (environment::noTempi) {
    return libmpi.MPI_Neighbor_alltoallv(ARGS_MPI_Neighbor_alltoallv);
  }

      /* this call does not take ranks, so there is no need to handle
         reordering. The library ranks are different than the application ranks,
         but they have the right neighbors in a consistent order, just with
         different numbers
      */

  return libmpi.MPI_Neighbor_alltoallv(ARGS_MPI_Neighbor_alltoallv);
}
