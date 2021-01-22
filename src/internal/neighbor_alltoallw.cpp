//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "neighbor_alltoallw.hpp"

#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "packer.hpp"
#include "tags.hpp"
#include "type_cache.hpp"

#include <vector>

/* use raw MPI_Isend/MPI_Irecv
 */
int neighbor_alltoallw::isir(PARAMS_MPI_Neighbor_alltoallw) {

#define OR_RETURN(err)                                                         \
  {                                                                            \
    if (MPI_SUCCESS != err) {                                                  \
      return err;                                                              \
    }                                                                          \
  }

  // int size = -1, rank = -1;
  // MPI_Comm_rank(comm, &rank);
  // MPI_Comm_size(comm, &size);

  // determine how many neighbors calling rank has
  int indegree = -1, outdegree = -1;
  {
    int _; // weighted
    int err = MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &_);
    OR_RETURN(err);
  }

  // recover neighbors to send to / recv from
  std::vector<int> sources(indegree), sourceweights(indegree);
  std::vector<int> destinations(outdegree), destweights(outdegree);
  libmpi.MPI_Dist_graph_neighbors(comm, indegree, sources.data(),
                                  sourceweights.data(), outdegree,
                                  destinations.data(), destweights.data());

  std::vector<MPI_Request> sreqs, rreqs;
  const int tag = tags::neighbor_alltoallw(comm);

  for (int i = 0; i < outdegree; ++i) {
    const char *buf = ((const char *)sendbuf) + sdispls[i];
    MPI_Request req{};
    int err = MPI_Isend(buf, sendcounts[i], sendtypes[i], destinations[i], tag,
                        comm, &req);
    OR_RETURN(err);
    sreqs.push_back(req);
  }

  for (int i = 0; i < indegree; ++i) {
    char *buf = ((char *)recvbuf) + rdispls[i];
    MPI_Request req{};
    int err = MPI_Irecv(buf, recvcounts[i], recvtypes[i], sources[i], tag, comm,
                        &req);
    OR_RETURN(err);
    rreqs.push_back(req);
  }

  for (MPI_Request &r : sreqs) {
    int err = MPI_Wait(&r, MPI_STATUS_IGNORE);
    OR_RETURN(err);
  }
  for (MPI_Request &r : rreqs) {
    int err = MPI_Wait(&r, MPI_STATUS_IGNORE);
    OR_RETURN(err);
  }

  return MPI_SUCCESS;

#undef OR_RETURN
}

#if 0
#include <functional>
#include <optional>

template <typename T> using ref = std::reference_wrapper<T>;

int neighbor_alltoallw::other(PARAMS_MPI_Neighbor_alltoallw) {


  int size = -1, rank = -1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // determine how many neighbors calling rank has
  int indegree = -1, outdegree = -1;
  {
    int _; // weighted
    MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &_);
  }

  // collect the packers for each datatype.
  // option will be null if we have no packer
  std::vector<std::optional<ref<Packer>>> sendPackers, recvPackers;

  bool noTempiPacker = true;
  for (int i = 0; i < outdegree; ++i) {
    auto it = typeCache.find(sendtypes[i]);
    if (typeCache.end() != it) {
      sendPackers.push_back(*(it->second.packer));
      noTempiPacker = false;
    } else {
      sendPackers.push_back(std::nullopt);
    }
  }
  for (int i = 0; i < indegree; ++i) {
    auto it = typeCache.find(recvtypes[i]);
    if (typeCache.end() != it) {
      recvPackers.push_back(*(it->second.packer));
      noTempiPacker = false;
    } else {
      recvPackers.push_back(std::nullopt);
    }
  }

  // if this is all on datatypes where we have no packer, use built-in alltoallw
  if (noTempiPacker) {
    LOG_SPEW("MPI_Neighbor_alltoallw: use library (packers for any datatype)")
    return libmpi.MPI_Neighbor_alltoallw(ARGS_MPI_Neighbor_alltoallw);
  }

  // compute the packed size

  PARAMS_MPI_Neighbor_alltoallv
}
#endif