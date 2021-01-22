#pragma once

#include <mpi.h>

namespace tags {

// tag for neighbor_alltoallw
int neighbor_alltoallw(MPI_Comm comm);

void init();
void finalize();
}; // namespace tags