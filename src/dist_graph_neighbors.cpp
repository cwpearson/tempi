#include "env.hpp"
#include "logging.hpp"
#include "partition.hpp"
#include "symbols.hpp"
#include "topology.hpp"

extern "C" int MPI_Dist_graph_neighbors(PARAMS_MPI_Dist_graph_neighbors) {
  if (environment::noTempi) {
    return libmpi.MPI_Dist_graph_neighbors(ARGS_MPI_Dist_graph_neighbors);
  }

  // the library provides its ranks in this call.
  // if we did placement, those are not the same ranks that we want to present
  // to the application
  auto it = placements.find(comm);
  if (it != placements.end()) {

    std::vector<int> libsources(maxindegree);
    std::vector<int> libdestinations(maxoutdegree);

    int err = libmpi.MPI_Dist_graph_neighbors(
        comm, maxindegree, libsources.data(), sourceweights, maxoutdegree,
        libdestinations.data(), destweights);
    if (MPI_SUCCESS != err) {
      return err;
    }

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
