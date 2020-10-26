#pragma once

#include <mpi.h>

void topology_init();

// true if this rank is colocated with other for `comm`
bool is_colocated(MPI_Comm comm, int other);

namespace topology {
void cache_communicator(MPI_Comm comm);
} // namespace topology