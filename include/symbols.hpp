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

struct MpiFunc {
  Func_MPI_Alltoallv MPI_Alltoallv;
  Func_MPI_Dist_graph_create MPI_Dist_graph_create;
  Func_MPI_Dist_graph_create_adjacent MPI_Dist_graph_create_adjacent;
  Func_MPI_Init MPI_Init;
  Func_MPI_Init_thread MPI_Init_thread;
  Func_MPI_Isend MPI_Isend;
  Func_MPI_Get_library_version MPI_Get_library_version;
  Func_MPI_Neighbor_alltoallv MPI_Neighbor_alltoallv;
  Func_MPI_Recv MPI_Recv;
  Func_MPI_Send MPI_Send;
};

extern MpiFunc libmpi;

// load all MPI symbols
void init_symbols();
