#include "topology.hpp"

#include "symbols.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

// MPI_COMM_WORLD ranks colocated with this rank
std::vector<int> colocatedRanks;

// rhanks colocated with this rank for each communicator
std::map<MPI_Comm, std::vector<int>> info;

namespace topology {

// determine and store topology information for `comm`
void cache_communicator(MPI_Comm comm) {
  nvtxRangePush("cache_communicator");
  // Give every rank a list of co-located ranks
  {
    MPI_Comm shm{};
    libmpi.MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                               &shm);
    int rank, size;
    libmpi.MPI_Comm_rank(comm, &rank);
    libmpi.MPI_Comm_size(shm, &size);
    std::vector<int> colocatedRanks(size);
    libmpi.MPI_Allgather(&rank, 1, MPI_INT, colocatedRanks.data(), 1, MPI_INT,
                         shm);

    // cache the colocated ranks
    info[comm] = colocatedRanks;

    // release temporary resources
    libmpi.MPI_Comm_free(&shm);
    shm = {};
  }
  nvtxRangePop();
}
} // namespace topology

void topology_init() {
  // cache ranks in MPI_COMM_WORLD
  topology::cache_communicator(MPI_COMM_WORLD);
}

bool is_colocated(MPI_Comm comm, int other) {
  assert(info.count(comm));
  const std::vector<int> &colo = info[comm];
  return colo.end() != std::find(colo.begin(), colo.end(), other);
}