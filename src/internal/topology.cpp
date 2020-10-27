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

// topology topos for a communicator
struct Topology {
  std::vector<size_t> nodeOfRank;
  std::vector<std::vector<int>> ranksOfNode;
};

// how application ranks are mapped to library ranks
struct Placement {
  std::vector<int> appRank; // application rank for each library rank
  std::vector<int> libRank; // library rank for each application rank
};

// ranks colocated with this rank for each communicator
std::map<MPI_Comm, Topology> topos;
std::map<MPI_Comm, Placement> placements;

namespace topology {

// determine and store topology toposrmation for `comm`
void cache_communicator(MPI_Comm comm) {
  nvtxRangePush("cache_communicator");

  Topology topo;

  int rank, size;
  libmpi.MPI_Comm_rank(comm, &rank);
  libmpi.MPI_Comm_size(comm, &size);
  LOG_SPEW("got rank and size");
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
      LOG_SPEW(s << " is node " << labels.size());
      size_t node = labels.size();
      labels[s] = node;
    }
  }

  LOG_SPEW("nodes: " << labels.size());
  for (auto &p : labels) {
    LOG_SPEW(p.first << " " << p.second);
  }

  topo.ranksOfNode.resize(labels.size());
  topo.nodeOfRank.resize(size);
  for (int r = 0; r < size; ++r) {
    std::string s(&names[r * MPI_MAX_PROCESSOR_NAME]);
    assert(labels.count(s));
    size_t node = labels[s];
    LOG_SPEW("rank " << r << " name=" << s << " node=" << node);
    topo.nodeOfRank[r] = node;
    topo.ranksOfNode[node].push_back(r);
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

  nvtxRangePop();

  topos[comm] = topo;
}

size_t num_nodes(MPI_Comm comm) {
  assert(topos.count(comm));
  return topos[comm].ranksOfNode.size();
}

void cache_node_assignment(MPI_Comm comm, const std::vector<int> &nodeOfRank) {

  // next free rank on each node
  std::vector<int> nextIdx(num_nodes(comm), 0);

  std::vector<int> appRank = placements[comm].appRank;
  std::vector<int> libRank = placements[comm].libRank;

  appRank.resize(nodeOfRank.size());
  libRank.resize(nodeOfRank.size());

  const std::vector<std::vector<int>> &ranksOfNode = topos[comm].ranksOfNode;

  for (int ar = 0; ar < int(nodeOfRank.size()); ++ar) {
    int node = nodeOfRank[ar];
    int cr = ranksOfNode[node][nextIdx[node]];
    nextIdx[node]++;
    appRank[cr] = ar;
    libRank[ar] = cr;
  }
}

void uncache(MPI_Comm comm) {
  topos.erase(comm);
  placements.erase(comm);
}

int library_rank(int rank, MPI_Comm comm) {
  auto it = placements.find(comm);
  if (it != placements.end()) {
    return it->second.libRank[rank];
  }
}

} // namespace topology

void topology_init() {
  LOG_SPEW("topology_init()");
  // cache ranks in MPI_COMM_WORLD
  topology::cache_communicator(MPI_COMM_WORLD);
  LOG_SPEW("finish topology_init()");
}

bool is_colocated(MPI_Comm comm, int other) {
  assert(topos.count(comm));
  int rank;
  libmpi.MPI_Comm_rank(comm, &rank);
  return topos[comm].nodeOfRank[rank] == topos[comm].nodeOfRank[other];
}
