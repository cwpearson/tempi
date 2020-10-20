#include "env.hpp"
#include "symbols.hpp"

extern "C" int MPI_Dist_graph_create(PARAMS_MPI_Dist_graph_create) {
  if (environment::noTempi) {
    return libmpi.MPI_Dist_graph_create(ARGS_MPI_Dist_graph_create);
  }
  return libmpi.MPI_Dist_graph_create(ARGS_MPI_Dist_graph_create);
}
