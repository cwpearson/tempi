#pragma once

#include <mpi.h>

#include <map>
#include <vector>

// how application ranks are mapped to library ranks
struct Placement {
  std::vector<int> appRank; // application rank for each library rank
  std::vector<int> libRank; // library rank for each application rank
};

extern std::map<MPI_Comm, Placement> placements;

struct Degree {
  int indegree;
  int outdegree;
};

extern std::map<MPI_Comm, Degree> degrees;

void topology_init();

// true if this rank is colocated with other for `comm`
bool is_colocated(MPI_Comm comm, int other);

namespace topology {
void cache_communicator(MPI_Comm comm);
size_t num_nodes(MPI_Comm comm);
size_t node_of_rank(MPI_Comm comm, int rank);
size_t num_nodes(MPI_Comm comm);
const std::vector<int> &ranks_of_node(MPI_Comm comm, size_t node);

/* return the library rank that backs `rank`
   if no placement is changed, return `rank`
*/
int library_rank(MPI_Comm comm, int rank);

/* return the application rank executing on`rank`
   if no placement is changed, return `rank`
*/
int application_rank(MPI_Comm comm, int rank);

// store the node that rank i should be on
void cache_node_assignment(MPI_Comm comm, const std::vector<int> &assignment);

// forget about a communicator
void uncache(MPI_Comm comm);

} // namespace topology