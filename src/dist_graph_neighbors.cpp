//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "env.hpp"
#include "logging.hpp"
#include "partition.hpp"
#include "symbols.hpp"
#include "topology.hpp"

#include <cassert>

extern "C" int MPI_Dist_graph_neighbors(PARAMS_MPI_Dist_graph_neighbors) {
  if (environment::noTempi) {
    return libmpi.MPI_Dist_graph_neighbors(ARGS_MPI_Dist_graph_neighbors);
  }

  // the library will return it's ranks in the call.
  // if we reordered, those are not the same ranks that we want to present
  // to the application
  auto it = placements.find(comm);
  if (it != placements.end()) {

    std::vector<int> libsources(maxindegree);
    std::vector<int> libdestinations(maxoutdegree);

    int err = libmpi.MPI_Dist_graph_neighbors(
        comm, maxindegree, libsources.data(), sourceweights, maxoutdegree,
        libdestinations.data(), destweights);

    for (int i = 0; i < maxindegree; ++i) {
      int librank = libsources[i];
      int apprank = it->second.appRank[librank];
      sources[i] = apprank;
    }
    for (int i = 0; i < maxoutdegree; ++i) {
      int librank = libdestinations[i];
      destinations[i] = it->second.appRank[librank];
    }

    return err;
  }

  return libmpi.MPI_Dist_graph_neighbors(ARGS_MPI_Dist_graph_neighbors);
}
