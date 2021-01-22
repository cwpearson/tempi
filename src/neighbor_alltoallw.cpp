//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "neighbor_alltoallw.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"

extern "C" int MPI_Neighbor_alltoallw(PARAMS_MPI_Neighbor_alltoallw) {
  if (environment::noTempi) {
    return libmpi.MPI_Neighbor_alltoallw(ARGS_MPI_Neighbor_alltoallw);
  }

  return neighbor_alltoallw::isir(ARGS_MPI_Neighbor_alltoallw);
}
