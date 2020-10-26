#include "env.hpp"
#include "symbols.hpp"
#include "topology.hpp"

extern "C" int
MPI_Dist_graph_create_adjacent(PARAMS_MPI_Dist_graph_create_adjacent) {
  if (environment::noTempi) {
    return libmpi.MPI_Dist_graph_create_adjacent(
        ARGS_MPI_Dist_graph_create_adjacent);
  }

  // call the underlying impl to actually create a communicator
  int err = libmpi.MPI_Dist_graph_create_adjacent(
      ARGS_MPI_Dist_graph_create_adjacent);

  // get the topology for the new communicator
  topology::cache_communicator(*comm_dist_graph);

  // graph partitioning
  if (reorder) {
  }

  return err;
}
