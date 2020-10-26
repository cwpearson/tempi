#pragma once

#include <mpi.h>

#include <vector>

void topology_init();

// true if this rank is colocated with other for `comm`
bool is_colocated(MPI_Comm comm, int other);

namespace topology {
void cache_communicator(MPI_Comm comm);

size_t num_nodes(MPI_Comm comm);
size_t node_of_rank(MPI_Comm comm, int rank);
size_t num_nodes(MPI_Comm comm);
const std::vector<int> &ranks_of_node(MPI_Comm comm, size_t node);


// store the node that rank i should be on
void cache_node_assignment(MPI_Comm comm, const std::vector<int> &assignment);

} // namespace topology