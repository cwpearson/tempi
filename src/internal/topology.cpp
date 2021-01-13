//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "topology.hpp"

#include "logging.hpp"
#include "symbols.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <algorithm>
#include <cassert>
#include <string>
#include <vector>
#include <map>

// topology topos for a communicator
struct Topology {
  std::vector<size_t> nodeOfRank;
  std::vector<std::vector<int>> ranksOfNode;
};

// ranks colocated with this rank for each communicator
std::unordered_map<MPI_Comm, Topology> topos;
/*extern*/ std::unordered_map<MPI_Comm, Placement> placements;
/*extern*/ std::unordered_map<MPI_Comm, Degree> degrees;

namespace topology {

// determine and store topology information for `comm`
void cache_communicator(MPI_Comm comm) {
  nvtxRangePush("cache_communicator");
  LOG_SPEW("cache_communicator(" << uintptr_t(comm));

  Topology topo;

  int rank, size;
  libmpi.MPI_Comm_rank(comm, &rank);
  libmpi.MPI_Comm_size(comm, &size);
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
      // LOG_SPEW(s << " is node " << labels.size());
      size_t node = labels.size();
      labels[s] = node;
    }
  }

  topo.ranksOfNode.resize(labels.size());
  topo.nodeOfRank.resize(size);
  for (int r = 0; r < size; ++r) {
    std::string s(&names[r * MPI_MAX_PROCESSOR_NAME]);
    assert(labels.count(s));
    size_t node = labels[s];
    // LOG_SPEW("rank " << r << " name=" << s << " node=" << node);
    topo.nodeOfRank[r] = node;
    topo.ranksOfNode[node].push_back(r);
  }

  if (0 == rank) {
    for (size_t node = 0; node < topo.ranksOfNode.size(); ++node) {
      std::string s("node ");
      s += std::to_string(node) + " ranks: ";
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

Placement make_placement(MPI_Comm comm, const std::vector<int> &nodeOfRank) {
  LOG_SPEW("make_placement(comm=" << uintptr_t(comm) << ", ...)");
  int size;
  libmpi.MPI_Comm_size(comm, &size);
  assert(size >= 0 && nodeOfRank.size() == size_t(size));

  // next free rank on each node
  std::vector<int> nextIdx(num_nodes(comm), 0);

  Placement placement;
  placement.appRank.resize(nodeOfRank.size());
  placement.libRank.resize(nodeOfRank.size());

  const std::vector<std::vector<int>> &ranksOfNode = topos[comm].ranksOfNode;

  for (int ar = 0; ar < int(nodeOfRank.size()); ++ar) {
    int node = nodeOfRank[ar];
    assert(node >= 0 && size_t(node) < nextIdx.size());
    int idx = nextIdx[node];
    assert(node < ranksOfNode.size());
    assert(idx < ranksOfNode[node].size());
    int cr = ranksOfNode[node][idx];
    assert(cr < placement.appRank.size());
    nextIdx[node]++;
    placement.appRank[cr] = ar;
    placement.libRank[ar] = cr;
  }

  {
    int rank;
    libmpi.MPI_Comm_rank(comm, &rank);
    if (0 == rank) {
      std::string s;
      for (int e : placement.appRank) {
        s += std::to_string(e) + " ";
      }
      LOG_DEBUG("comm= " << uintptr_t(comm) << " appRank: " << s);
    }
    if (0 == rank) {
      std::string s;
      for (int e : placement.libRank) {
        s += std::to_string(e) + " ";
      }
      LOG_DEBUG("comm= " << uintptr_t(comm) << " libRank: " << s);
    }
  }
  return placement;
}

void cache_placement(MPI_Comm comm, const Placement &placement) {
  placements[comm] = placement;
}

void uncache(MPI_Comm comm) {
  topos.erase(comm);
  placements.erase(comm);
}

int library_rank(MPI_Comm comm, int rank) {
  auto it = placements.find(comm);
  if (it != placements.end()) {
    return it->second.libRank[rank];
  } else {
    return rank;
  }
}

int application_rank(MPI_Comm comm, int rank) {
  auto it = placements.find(comm);
  if (it != placements.end()) {
    return it->second.appRank[rank];
  } else {
    return rank;
  }
}

size_t node_of_rank(MPI_Comm comm, int rank) {
  auto it = topos.find(comm);
  if (it != topos.end()) {
    return it->second.nodeOfRank[library_rank(comm, rank)];
  } else {
    LOG_FATAL("couldn't get node for rank " << rank);
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
