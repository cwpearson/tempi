//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "symbols.hpp"

#include "logging.hpp"

#include <dlfcn.h>

/* extern */ MpiFunc libmpi;

#define DLSYM(A) \
libmpi.A = reinterpret_cast<Func_##A>(dlsym(RTLD_NEXT, #A));

void init_symbols() {
  DLSYM(MPI_Allgather);
  DLSYM(MPI_Alltoallv);
  DLSYM(MPI_Comm_free);
  DLSYM(MPI_Comm_rank);
  DLSYM(MPI_Comm_size);
  DLSYM(MPI_Comm_split_type);
  DLSYM(MPI_Dist_graph_create);
  DLSYM(MPI_Dist_graph_create_adjacent);
  DLSYM(MPI_Dist_graph_neighbors);
  DLSYM(MPI_Get_library_version);
  DLSYM(MPI_Init);
  DLSYM(MPI_Init_thread);
  DLSYM(MPI_Isend);
  DLSYM(MPI_Neighbor_alltoallv);
  DLSYM(MPI_Pack);
  DLSYM(MPI_Recv);
  DLSYM(MPI_Send);
  DLSYM(MPI_Type_commit);
  DLSYM(MPI_Type_free);
  DLSYM(MPI_Unpack);
}
