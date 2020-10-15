#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "types.hpp"
#include "worker.hpp"

#include "allocator_slab.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <dlfcn.h>

#include <vector>

#define PARAMS                                                                 \
  const void *sendbuf, const int sendcounts[], const int sdispls[],            \
      MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],            \
      const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm
#define ARGS                                                                   \
  sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,        \
      recvtype, comm

extern "C" int MPI_Alltoallv(PARAMS) {
  typedef int (*Func_MPI_Alltoallv)(PARAMS);
  static Func_MPI_Alltoallv fn = nullptr;
  if (!fn) {
    fn =
        reinterpret_cast<Func_MPI_Alltoallv>(dlsym(RTLD_NEXT, "MPI_Alltoallv"));
  }
  TEMPI_DISABLE_GUARD;
  if (environment::noAlltoallv) {
    return fn(ARGS);
  }
  LOG_DEBUG("MPI_Alltoallv");

  int commSize = 0;
  MPI_Comm_size(comm, &commSize);

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes sendAttr = {}, recvAttr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&sendAttr, sendbuf));
  CUDA_RUNTIME(cudaPointerGetAttributes(&recvAttr, recvbuf));
  if (nullptr == sendAttr.devicePointer || nullptr == recvAttr.devicePointer) {
    LOG_DEBUG("use library (host memory)");
    return fn(ARGS);
  }

  std::vector<MPI_Request> sendReqs(commSize);
  std::vector<MPI_Request> recvReqs(commSize);

  for (int j = 0; j < commSize; ++j) {
    MPI_Isend(((char *)sendbuf) + sdispls[j], sendcounts[j], sendtype, j, 0,
              comm, &sendReqs[j]);
  }
  for (int i = 0; i < commSize; ++i) {
    MPI_Irecv(((char *)recvbuf) + rdispls[i], recvcounts[i], recvtype, i, 0,
              comm, &recvReqs[i]);
  }
  MPI_Waitall(sendReqs.size(), sendReqs.data(), MPI_STATUS_IGNORE);
  MPI_Waitall(recvReqs.size(), recvReqs.data(), MPI_STATUS_IGNORE);

  return MPI_SUCCESS;
}
