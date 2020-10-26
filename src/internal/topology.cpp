#include "topology.hpp"

#include "logging.hpp"
#include "symbols.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <string>
#include <vector>

// topology info for a communicator
struct Topology {
  std::vector<size_t> nodeOfRank;
  std::vector<std::vector<int>> ranksOfNode;
};

// ranks colocated with this rank for each communicator
std::map<MPI_Comm, Topology> info;

/* retrieve the app rank for a commRank or vis-versa for a permuted communicator
 */
std::map<MPI_Comm, std::vector<int>> appRank, commRank;

namespace topology {

// determine and store topology information for `comm`
void cache_communicator(MPI_Comm comm) {
  nvtxRangePush("cache_communicator");

  Topology topo;

  int rank, size;
  libmpi.MPI_Comm_rank(comm, &rank);
  libmpi.MPI_Comm_size(comm, &size);

  {
    // get my node name
    char name[MPI_MAX_PROCESSOR_NAME]{};
    int namelen;
    MPI_Get_processor_name(name, &namelen);

    // distribute names to all nodes
    std::vector<char> names(MPI_MAX_PROCESSOR_NAME * size);
    MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, names.data(),
                  MPI_MAX_PROCESSOR_NAME, MPI_CHAR, comm);

    // assign a label to each node
    std::map<std::string, int> labels;
    for (int r = 0; r < size; ++r) {
      std::string s(&names[r * MPI_MAX_PROCESSOR_NAME]);
      if (0 == labels.count(s)) {
        labels[s] = labels.size();
      }
    }

    topo.ranksOfNode.resize(labels.size());
    topo.nodeOfRank.resize(size);
    for (int r = 0; r < size; ++r) {
      std::string s(&names[r * MPI_MAX_PROCESSOR_NAME]);
      size_t node = labels[s];
      topo.ranksOfNode[node].push_back(r);
      topo.nodeOfRank[r] = node;
    }

    if (0 == rank) {
      for (size_t node = 0; node < topo.ranksOfNode.size(); ++node) {
        std::string s("node ");
        s += std::to_string(node) + ": ";
        std::vector<int> &ranks = topo.ranksOfNode[node];
        for (int i : ranks) {
          s += std::to_string(i) + " ";
        }
        LOG_DEBUG(s);
      }
    }
  }

  nvtxRangePop();

  info[comm] = topo;
}

size_t num_nodes(MPI_Comm comm) {
  assert(info.count(comm));
  return info[comm].ranksOfNode.size();
}

void cache_node_assignment(MPI_Comm comm, const std::vector<int> &nodeOfRank) {

  // next free rank on each node
  std::vector<int> nextIdx(num_nodes(comm), 0);

  std::vector<int> &commAppRank = appRank[comm];
  std::vector<int> &commCommRank = commRank[comm];

  commAppRank.resize(nodeOfRank.size());
  commCommRank.resize(nodeOfRank.size());

  const std::vector<std::vector<int>> &ranksOfNode = info[comm].ranksOfNode;

  // comm rank
  for (int cr = 0; cr < int(nodeOfRank.size()); ++cr) {
    int node = nodeOfRank[cr];
    int ar = ranksOfNode[node][nextIdx[node]];
    nextIdx[node]++;

    LOG_SPEW("ar=" << ar << "->"
                   << "cr=" << cr);

    commAppRank[cr] = ar;
    commCommRank[ar] = cr;
  }
}

} // namespace topology

void topology_init() {
  // cache ranks in MPI_COMM_WORLD
  topology::cache_communicator(MPI_COMM_WORLD);
}

bool is_colocated(MPI_Comm comm, int other) {
  assert(info.count(comm));
  int rank;
  libmpi.MPI_Comm_rank(comm, &rank);
  return info[comm].nodeOfRank[rank] == info[comm].nodeOfRank[other];
}
