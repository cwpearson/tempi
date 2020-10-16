#include "topology.hpp"

#include <nvToolsExt.h>

#include <algorithm>

/*extern*/ std::vector<int> colocatedRanks;

void topology_init() {

  nvtxRangePush("topology_init");

  // Give every rank a list of co-located ranks
  {
    MPI_Comm shm = {};
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shm);
    int rank, coloSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(shm, &coloSize);
    colocatedRanks.resize(coloSize);
    MPI_Allgather(&rank, 1, MPI_INT, colocatedRanks.data(), 1, MPI_INT, shm);
    MPI_Comm_free(&shm);
    shm = {};
  }

  nvtxRangePop();
}

bool is_colocated(int other) {
  return colocatedRanks.end() !=
         std::find(colocatedRanks.begin(), colocatedRanks.end(), other);
}