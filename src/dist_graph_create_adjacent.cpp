#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"
#include "topology.hpp"

#include <algorithm>
#include <numeric>
#include <random>

std::default_random_engine generator(0);

/* return a node assignment 0..<numNodes for each rank 0..<numRanks
 */
static std::vector<int> random_partition(const int numRanks,
                                         const int numNodes) {
  std::vector<int> p(numRanks);
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] = i * numNodes / numRanks;
  }
  std::shuffle(p.begin(), p.end(), generator);
  return p;
}

extern "C" int
MPI_Dist_graph_create_adjacent(PARAMS_MPI_Dist_graph_create_adjacent) {
  if (environment::noTempi) {
    return libmpi.MPI_Dist_graph_create_adjacent(
        ARGS_MPI_Dist_graph_create_adjacent);
  }

  /* call the underlying impl to actually create a communicator

     This also places the ranks on the nodes (the "communicator rank")
     We may be able to determine a better mapping between ranks and nodes.
     If so, we will relabel all the ranks, and present a different rank number
     to the application (the "application rank");

     IF ranks 0 and 1 should be on the same node, we will map ranks 0 and 1 to
     ranks I and J, where I and J are actually on the same node. when rank I
     calls MPI_Comm_rank, they will get 0. When rank I calls MPI_Send to rank 1,
     we will call MPI_Send to rank J if these ranks call collectives, their send
     and recv buffers may be out of order, so we will neeed to permute them
     under the hood which will slow things down
  */
  int err = libmpi.MPI_Dist_graph_create_adjacent(
      ARGS_MPI_Dist_graph_create_adjacent);

  // get the topology for the new communicator
  topology::cache_communicator(*comm_dist_graph);

  // graph partitioning
  if (reorder) {

    const size_t numNodes = topology::num_nodes(*comm_dist_graph);
    int numRanks;
    libmpi.MPI_Comm_size(*comm_dist_graph, &numRanks);

    // based on the weights provided, we assign each rank to a partition
    std::vector<int> partAssignment = random_partition(numRanks, numNodes);
#if TEMPI_OUTPUT_LEVEL >= 4
{
  std::string s("node assignment app rank: ");
  for (int r : partAssignment) {
    s += std::to_string(r) + " ";
  }
  LOG_SPEW(s);
}

#endif

    // all the ranks assign to partition 0 will be backed by ranks on node 0
    topology::cache_node_assignment(*comm_dist_graph, partAssignment);

    // retrieve the comm rank for an application rank
    // topology::get_comm_rank(*comm_dist_graph, appRank);
    // retrieve the application rank for a communicator rank
    // topology::get_app_rank(*comm_dist_graph, commRank);
  }

  return err;
}
