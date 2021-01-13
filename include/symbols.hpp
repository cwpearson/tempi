//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <mpi.h>

#define PARAMS_MPI_Init int *argc, char ***argv
#define ARGS_MPI_Init argc, argv
typedef int (*Func_MPI_Init)(PARAMS_MPI_Init);

#define PARAMS_MPI_Init_thread                                                 \
  int *argc, char ***argv, int required, int *provided
#define ARGS_MPI_Init_thread argc, argv, required, provided
typedef int (*Func_MPI_Init_thread)(PARAMS_MPI_Init_thread);

#define PARAMS_MPI_Isend                                                       \
  const void *buf, int count, MPI_Datatype datatype, int dest, int tag,        \
      MPI_Comm comm, MPI_Request *request
#define ARGS_MPI_Isend buf, count, datatype, dest, tag, comm, request
typedef int (*Func_MPI_Isend)(PARAMS_MPI_Isend);

#define PARAMS_MPI_Get_library_version char *version, int *resultlen
#define ARGS_MPI_Get_library_version version, resultlen
typedef int (*Func_MPI_Get_library_version)(PARAMS_MPI_Get_library_version);

#define PARAMS_MPI_Alltoallv                                                   \
  const void *sendbuf, const int sendcounts[], const int sdispls[],            \
      MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],            \
      const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm
#define ARGS_MPI_Alltoallv                                                     \
  sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,        \
      recvtype, comm
typedef int (*Func_MPI_Alltoallv)(PARAMS_MPI_Alltoallv);

#define PARAMS_MPI_Neighbor_alltoallv                                          \
  const void *sendbuf, const int sendcounts[], const int sdispls[],            \
      MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],            \
      const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm
#define ARGS_MPI_Neighbor_alltoallv                                            \
  sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,        \
      recvtype, comm
typedef int (*Func_MPI_Neighbor_alltoallv)(PARAMS_MPI_Neighbor_alltoallv);

#define PARAMS_MPI_Dist_graph_create                                           \
  MPI_Comm comm_old, int n, const int sources[], const int degrees[],          \
      const int destinations[], const int weights[], MPI_Info info,            \
      int reorder, MPI_Comm *comm_dist_graph
#define ARGS_MPI_Dist_graph_create                                             \
  comm_old, n, sources, degrees, destinations, weights, info, reorder,         \
      comm_dist_graph
typedef int (*Func_MPI_Dist_graph_create)(PARAMS_MPI_Dist_graph_create);

#define PARAMS_MPI_Dist_graph_create_adjacent                                  \
  MPI_Comm comm_old, int indegree, const int sources[],                        \
      const int sourceweights[], int outdegree, const int destinations[],      \
      const int destweights[], MPI_Info info, int reorder,                     \
      MPI_Comm *comm_dist_graph
#define ARGS_MPI_Dist_graph_create_adjacent                                    \
  comm_old, indegree, sources, sourceweights, outdegree, destinations,         \
      destweights, info, reorder, comm_dist_graph
typedef int (*Func_MPI_Dist_graph_create_adjacent)(
    PARAMS_MPI_Dist_graph_create_adjacent);

#define PARAMS_MPI_Recv                                                        \
  void *buf, int count, MPI_Datatype datatype, int source, int tag,            \
      MPI_Comm comm, MPI_Status *status
#define ARGS_MPI_Recv buf, count, datatype, source, tag, comm, status
typedef int (*Func_MPI_Recv)(PARAMS_MPI_Recv);

#define PARAMS_MPI_Send                                                        \
  const void *buf, int count, MPI_Datatype datatype, int dest, int tag,        \
      MPI_Comm comm
#define ARGS_MPI_Send buf, count, datatype, dest, tag, comm
typedef int (*Func_MPI_Send)(PARAMS_MPI_Send);

//

#define PARAMS_MPI_Comm_split_type                                             \
  MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm
#define ARGS_MPI_Comm_split_type comm, split_type, key, info, newcomm
typedef int (*Func_MPI_Comm_split_type)(PARAMS_MPI_Comm_split_type);

#define PARAMS_MPI_Comm_rank MPI_Comm comm, int *rank
#define ARGS_MPI_Comm_rank comm, rank
typedef int (*Func_MPI_Comm_rank)(PARAMS_MPI_Comm_rank);

#define PARAMS_MPI_Comm_size MPI_Comm comm, int *size
#define ARGS_MPI_Comm_size comm, size
typedef int (*Func_MPI_Comm_size)(PARAMS_MPI_Comm_size);

#define PARAMS_MPI_Allgather                                                   \
  const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,    \
      int recvcount, MPI_Datatype recvtype, MPI_Comm comm
#define ARGS_MPI_Allgather                                                     \
  sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm
typedef int (*Func_MPI_Allgather)(PARAMS_MPI_Allgather);

#define PARAMS_MPI_Comm_free MPI_Comm *comm
#define ARGS_MPI_Comm_free comm
typedef int (*Func_MPI_Comm_free)(PARAMS_MPI_Comm_free);

#define PARAMS_MPI_Dist_graph_neighbors                                        \
  MPI_Comm comm, int maxindegree, int sources[], int sourceweights[],          \
      int maxoutdegree, int destinations[], int destweights[]
#define ARGS_MPI_Dist_graph_neighbors                                          \
  comm, maxindegree, sources, sourceweights, maxoutdegree, destinations,       \
      destweights
typedef int (*Func_MPI_Dist_graph_neighbors)(PARAMS_MPI_Dist_graph_neighbors);

#define PARAMS_MPI_Type_commit MPI_Datatype *datatype
#define ARGS_MPI_Type_commit datatype
typedef int (*Func_MPI_Type_commit)(PARAMS_MPI_Type_commit);

#define PARAMS_MPI_Type_free MPI_Datatype *datatype
#define ARGS_MPI_Type_free datatype
typedef int (*Func_MPI_Type_free)(PARAMS_MPI_Type_free);

#define PARAMS_MPI_Pack                                                        \
  const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf,         \
      int outsize, int *position, MPI_Comm comm
#define ARGS_MPI_Pack inbuf, incount, datatype, outbuf, outsize, position, comm
typedef int (*Func_MPI_Pack)(PARAMS_MPI_Pack);

#define PARAMS_MPI_Unpack                                                      \
  const void *inbuf, int insize, int *position, void *outbuf, int outcount,    \
      MPI_Datatype datatype, MPI_Comm comm
#define ARGS_MPI_Unpack                                                        \
  inbuf, insize, position, outbuf, outcount, datatype, comm
typedef int (*Func_MPI_Unpack)(PARAMS_MPI_Unpack);

struct MpiFunc {
  Func_MPI_Allgather MPI_Allgather;
  Func_MPI_Alltoallv MPI_Alltoallv;
  Func_MPI_Comm_free MPI_Comm_free;
  Func_MPI_Comm_rank MPI_Comm_rank;
  Func_MPI_Comm_size MPI_Comm_size;
  Func_MPI_Comm_split_type MPI_Comm_split_type;
  Func_MPI_Dist_graph_create MPI_Dist_graph_create;
  Func_MPI_Dist_graph_create_adjacent MPI_Dist_graph_create_adjacent;
  Func_MPI_Dist_graph_neighbors MPI_Dist_graph_neighbors;
  Func_MPI_Get_library_version MPI_Get_library_version;
  Func_MPI_Init MPI_Init;
  Func_MPI_Init_thread MPI_Init_thread;
  Func_MPI_Isend MPI_Isend;
  Func_MPI_Neighbor_alltoallv MPI_Neighbor_alltoallv;
  Func_MPI_Pack MPI_Pack;
  Func_MPI_Recv MPI_Recv;
  Func_MPI_Send MPI_Send;
  Func_MPI_Type_commit MPI_Type_commit;
  Func_MPI_Type_free MPI_Type_free;
  Func_MPI_Unpack MPI_Unpack;
};

extern MpiFunc libmpi;

// load all MPI symbols
void init_symbols();
