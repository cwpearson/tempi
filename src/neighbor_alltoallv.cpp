#include "env.hpp"
#include "symbols.hpp"

extern "C" int MPI_Neighbor_alltoallv(PARAMS_MPI_Neighbor_alltoallv) {
  if (environment::noTempi) {
    return libmpi.MPI_Neighbor_alltoallv(ARGS_MPI_Alltoallv);
  }
  return libmpi.MPI_Neighbor_alltoallv(ARGS_MPI_Alltoallv);
}
